#!/usr/bin/env python3
"""Serve a vla_foundry ``diffusion_policy`` (LBM) checkpoint over openpi's websocket protocol.

Why this exists instead of `vla_foundry/inference/robotics/inference_policy.py`: that server
speaks gRPC through `grpc_workspace.lbm_policy_server` and its observations are `robot_gym`
`MultiarmObservation` objects mapped through `field_mapping.yaml`. Nothing in the open-world
policy-DAgger stack can produce one of those. What that stack *does* speak, for the three pi0.5
fine-tunes on this exact data, is openpi's msgpack-over-websockets protocol:

    <- {"observation/image": u8[H,W,3], "observation/left_wrist_image": ...,
        "observation/right_wrist_image": ..., "observation/state": f32[16], "prompt": str}
    -> {"actions": f32[16, 20]}

Speaking that protocol here is what makes an LBM checkpoint substitutable for a pi0.5 one:
`openworld.policies.lbm_policy.LBMBikePolicy` subclasses the pi0.5 adapter and changes nothing
about the payload, so `serve` / `collect` / `convert` / `mix` in the DAgger loop are unmodified.

Out-of-process is not a workaround, it is the only option: pi0.5 needs jax 0.5.3 and this repo
needs torch 2.7 + numpy 2.x, which cannot share a venv. The socket is the seam.

Usage:
    .venv/bin/python -m vla_foundry.serving.lbm_policy_server \
        --checkpoint_directory experiments/lbm_clean_spill_<uuid> \
        --port 8010
"""

import argparse
import asyncio
import http
import logging
import os
import time
import traceback

import numpy as np
import torch
import websockets.asyncio.server as websockets_server
import websockets.frames
from websockets.exceptions import ConnectionClosed

from vla_foundry.data.processor.robotics_processor import RoboticsProcessor
from vla_foundry.file_utils import (
    get_latest_checkpoint,
    load_ema_checkpoint,
    load_model_checkpoint,
)
from vla_foundry.logger import setup_logging
from vla_foundry.models import create_model
from vla_foundry.params.train_experiment_params import load_experiment_params_from_yaml
from vla_foundry.precision import get_autocast
from vla_foundry.serving import msgpack_numpy

logger = logging.getLogger(__name__)

#: Camera short name -> openpi payload key, matching
#: `openpi/src/openpi/policies/bike_rotor_policy.py` and
#: `openworld/policies/openpi_bike_policy.py::VIEW_TO_PAYLOAD_KEY`. The short name is what is
#: left of a `camera_names` entry after dropping the `observation.images.` prefix, i.e. exactly
#: the video subdirectory names in the LeRobot dataset both policies were trained on.
VIEW_TO_PAYLOAD_KEY = {
    "base": "observation/image",
    "left_wrist": "observation/left_wrist_image",
    "right_wrist": "observation/right_wrist_image",
}

STATE_PAYLOAD_KEY = "observation/state"
PROMPT_PAYLOAD_KEY = "prompt"


def _short_camera_name(image_name: str) -> str:
    """`observation.images.left_wrist_t0` -> `left_wrist`.

    Mirrors the short-key derivation in `vla_foundry.data.pipelines.robotics`, which is what
    makes `camera_names` work with or without a dotted prefix.
    """
    stem = image_name.rsplit("_t", 1)[0]
    return stem.rsplit(".", 1)[-1]


class LBMOpenPIPolicy:
    """A trained `diffusion_policy` checkpoint behind the openpi `infer(obs) -> dict` interface.

    Deliberately stateless across calls. The pi0.5 server is too, and every bit of temporal
    machinery the rollout needs -- how much of the chunk to execute, chunk splicing, temporal
    ensembling, gripper majority voting -- already lives on the open-world side and is shared
    between the two policies. Adding a second, different open-loop counter in here would mean
    the two policies were no longer being evaluated under the same controller.
    """

    def __init__(
        self,
        checkpoint_directory: str,
        checkpoint_name: str | None = None,
        device: str = "cuda",
        num_flow_steps: int = 10,
        default_prompt: str | None = None,
    ):
        self.checkpoint_directory = checkpoint_directory.rstrip("/")
        self.num_flow_steps = int(num_flow_steps)
        self.default_prompt = default_prompt

        config_path = os.path.join(self.checkpoint_directory, "config.yaml")
        self.cfg = load_experiment_params_from_yaml(
            config_path, localize_params=not config_path.startswith("s3://")
        )
        self.data_params = self.cfg.data

        # The openpi payload carries exactly one frame per camera, so a checkpoint trained with
        # frame stacking cannot be served over this protocol -- there is nowhere to put t-1.
        if list(self.data_params.image_indices) != [0]:
            raise ValueError(
                f"this protocol supplies one frame per camera, but the checkpoint was trained "
                f"with image_indices={list(self.data_params.image_indices)}. Only [0] is servable."
            )
        # Likewise the payload carries one state vector, so the model must consume one.
        if self.data_params.lowdim_past_timesteps != 0:
            raise ValueError(
                f"this protocol supplies one proprioception step, but the checkpoint was trained "
                f"with lowdim_past_timesteps={self.data_params.lowdim_past_timesteps}."
            )

        self.image_names = list(self.data_params.image_names)
        self.payload_keys = []
        for image_name in self.image_names:
            short = _short_camera_name(image_name)
            if short not in VIEW_TO_PAYLOAD_KEY:
                raise ValueError(
                    f"camera '{short}' (from image_names entry {image_name!r}) has no openpi "
                    f"payload key; this server only serves the 3-view TRI bimanual setup "
                    f"{sorted(VIEW_TO_PAYLOAD_KEY)}."
                )
            self.payload_keys.append(VIEW_TO_PAYLOAD_KEY[short])

        # T = past + anchor + future. 0 + 1 + 15 = 16, matching Pi0Config(action_horizon=16).
        self.action_horizon = (
            self.data_params.lowdim_past_timesteps + 1 + self.data_params.lowdim_future_timesteps
        )
        self.action_dim = self.data_params.action_dim
        self.proprioception_dim = self.data_params.proprioception_dim

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = create_model(self.cfg.model, load_pretrained=False)
        self.processor = RoboticsProcessor.from_pretrained(self.checkpoint_directory)
        if self.processor.normalizer is None:
            raise ValueError(
                "checkpoint has no normalizer; actions would be returned in normalized units."
            )

        checkpoint_path = self._resolve_checkpoint_path(checkpoint_name)
        if self.cfg.ema.enabled:
            load_ema_checkpoint(self.model, checkpoint_path)
        else:
            load_model_checkpoint(self.model, checkpoint_path)
        self.checkpoint_path = checkpoint_path

        self.model.to(self.device)
        self.model.eval()
        self.autocast = get_autocast(self.cfg.hparams.precision)

        logger.info(
            "LBM policy ready: checkpoint=%s device=%s horizon=%d action_dim=%d state_dim=%d "
            "flow_steps=%d views=%s",
            self.checkpoint_path, self.device, self.action_horizon, self.action_dim,
            self.proprioception_dim, self.num_flow_steps, self.payload_keys,
        )

    def _resolve_checkpoint_path(self, checkpoint_name: str | None) -> str:
        if not checkpoint_name:
            latest = get_latest_checkpoint(self.checkpoint_directory)
            if not latest:
                raise FileNotFoundError(f"no checkpoints under {self.checkpoint_directory}")
            checkpoint_name = os.path.basename(latest)
        if not checkpoint_name.endswith(".pt"):
            checkpoint_name = f"{checkpoint_name}.pt"
        # EMA weights are what the LBM recipe trains for (ema.enabled: true), and they live in a
        # separate file. Same rename rule as InferenceDiffusionPolicy so a checkpoint name copied
        # from one server works in the other.
        if self.cfg.ema.enabled:
            checkpoint_name = checkpoint_name.replace("checkpoint_", "ema_")
            if not checkpoint_name.startswith("ema_"):
                checkpoint_name = f"ema_{checkpoint_name}"
        return os.path.join(self.checkpoint_directory, "checkpoints", checkpoint_name)

    # ------------------------------------------------------------------
    @property
    def metadata(self) -> dict:
        """Sent once on connect, like openpi's server. Purely informational."""
        return {
            "policy": "vla_foundry.diffusion_policy",
            "checkpoint": self.checkpoint_path,
            "action_horizon": self.action_horizon,
            "action_dim": self.action_dim,
            "state_dim": self.proprioception_dim,
            "num_flow_steps": self.num_flow_steps,
        }

    def _prepare_image(self, value, payload_key: str) -> torch.Tensor:
        """openpi's HWC uint8 payload -> the CHW uint8 tensor the training pipeline produced.

        `decode_and_augment_sample` hands the processor CHW uint8 torch tensors, and
        `CLIPImageProcessor` rescales by 1/255 for integer input. Passing HWC, or float in
        [0, 1], would silently change the input scale by 255x or transpose the image.
        """
        image = np.asarray(value)
        if image.ndim != 3:
            raise ValueError(f"{payload_key}: expected a 3-d image, got shape {image.shape}")
        if image.shape[-1] == 3:
            image = np.transpose(image, (2, 0, 1))
        elif image.shape[0] != 3:
            raise ValueError(f"{payload_key}: expected 3 channels, got shape {image.shape}")
        if np.issubdtype(image.dtype, np.floating):
            # The open-world adapter already sends uint8; accept float [0,1] defensively rather
            # than letting it through as a 255x-dark image.
            image = np.clip(image * 255.0, 0, 255)
        return torch.from_numpy(np.ascontiguousarray(image)).to(torch.uint8)

    def _build_batch(self, obs: dict) -> dict:
        missing = [k for k in (*self.payload_keys, STATE_PAYLOAD_KEY) if k not in obs]
        if missing:
            raise KeyError(f"observation is missing required key(s) {missing}")

        images = {
            name: self._prepare_image(obs[key], key)
            for name, key in zip(self.image_names, self.payload_keys, strict=True)
        }

        state = np.asarray(obs[STATE_PAYLOAD_KEY], dtype=np.float32).reshape(-1)
        if state.size != self.proprioception_dim:
            raise ValueError(
                f"{STATE_PAYLOAD_KEY} must be {self.proprioception_dim}-d, got {state.size}-d"
            )

        prompt = obs.get(PROMPT_PAYLOAD_KEY) or self.default_prompt
        if not prompt:
            raise KeyError(
                f"no '{PROMPT_PAYLOAD_KEY}' in the observation and no --default_prompt was set; "
                "this policy is language-conditioned."
            )
        if isinstance(prompt, bytes):
            prompt = prompt.decode("utf-8")

        # Per-timestep normalization indexes the statistics by position in the window, so the
        # lowdim fields have to arrive at full window length T even though only the anchor row
        # of the state is consumed (`add_action_and_proprioception_fields` slices
        # [:, :past+1]). Tiling the current state fills the discarded rows.
        lowdim = {
            field: np.tile(state, (self.action_horizon, 1))
            for field in self.data_params.proprioception_fields
        }
        # The action tensor only supplies shape and the "past" rows to preserve. With
        # lowdim_past_timesteps=0 the past mask is all-False (`create_past_and_future_masks`
        # treats the anchor as future), so `generate_actions` overwrites all 16 rows with noise
        # and these zeros never reach the model.
        for field in self.data_params.action_fields:
            lowdim[field] = np.zeros((self.action_horizon, self.action_dim), dtype=np.float32)

        return {
            "images": [images],
            "language_instruction": [prompt],
            "lowdim": [lowdim],
            "past_mask": torch.zeros(1, self.action_horizon, dtype=torch.bool),
            "metadata": [{"anchor_relative_idx": self.data_params.lowdim_past_timesteps}],
        }

    def infer(self, obs: dict) -> dict:
        batch = self._build_batch(obs)
        processed = self.processor.process_inputs(
            batch,
            image_names=self.image_names,
            max_text_seq_len=self.data_params.max_text_seq_len,
        )
        processed = self.processor.add_action_and_proprioception_fields(
            processed,
            action_fields=self.data_params.action_fields,
            proprioception_fields=self.data_params.proprioception_fields,
        )

        # Pass exactly the arguments generate_actions declares. Forwarding every tensor in the
        # batch (as the gRPC server does) would also hand it future_mask and friends via
        # **kwargs, which are then splatted into the vision-language backbone.
        model_input = {
            "input_ids": processed["input_ids"],
            "attention_mask": processed.get("attention_mask"),
            "pixel_values": processed["pixel_values"],
            "attention_mask_images": processed.get("attention_mask_images"),
            "actions": processed["actions"],
            "proprioception": processed["proprioception"],
            "past_mask": batch["past_mask"],
        }
        model_input = {
            k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
            for k, v in model_input.items()
        }

        with torch.no_grad(), self.autocast():
            actions = self.model.generate_actions(
                **model_input, num_inference_steps=self.num_flow_steps
            )

        actions = self.processor.normalizer.denormalize_tensor(
            actions.detach().float().cpu(),
            self.data_params.action_fields[0],
            anchor_timestep=self.data_params.lowdim_past_timesteps,
        )
        # [1, T, D] -> [T, D], the shape openpi's bike policy returns.
        return {"actions": actions[0].numpy().astype(np.float32)}


class WebsocketPolicyServer:
    """Byte-compatible with `openpi.serving.websocket_policy_server.WebsocketPolicyServer`.

    Reimplemented rather than imported for the same reason as the msgpack codec: openpi cannot
    be installed in this venv. Keep the framing (metadata on connect, one request/response pair
    per frame, traceback as a text frame on error, /healthz) identical -- `WebsocketClientPolicy`
    on the other end treats a text frame as a server error and everything else as msgpack.
    """

    def __init__(self, policy: LBMOpenPIPolicy, host: str = "0.0.0.0", port: int = 8000):
        self._policy = policy
        self._host = host
        self._port = port
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self._run())

    async def _run(self):
        async with websockets_server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            logger.info("LBM policy server listening on ws://%s:%d", self._host, self._port)
            await server.serve_forever()

    async def _handler(self, websocket):
        logger.info("Connection from %s opened", websocket.remote_address)
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._policy.metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                infer_time = time.monotonic()
                action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {"infer_ms": infer_time * 1000}
                if prev_total_time is not None:
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time
            except ConnectionClosed:
                logger.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection, request):
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint_directory", required=True, help="Training experiment dir (holds config.yaml)")
    parser.add_argument("--checkpoint_name", default=None, help="e.g. checkpoint_12; default: latest")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_flow_steps", type=int, default=10)
    parser.add_argument("--default_prompt", default=None, help="Used when the payload omits 'prompt'")
    args = parser.parse_args()

    # Positional (log_file, level) -- vla_foundry's setup_logging takes no keywords, and `level`
    # is a logging constant, not a name. Both are easy to get wrong and neither is caught until
    # the process starts, which is why this line is what the smoke test could not cover: it
    # builds the policy and server objects directly and never enters main().
    setup_logging(None, logging.INFO)
    policy = LBMOpenPIPolicy(
        checkpoint_directory=args.checkpoint_directory,
        checkpoint_name=args.checkpoint_name,
        device=args.device,
        num_flow_steps=args.num_flow_steps,
        default_prompt=args.default_prompt,
    )
    WebsocketPolicyServer(policy, host=args.host, port=args.port).serve_forever()


if __name__ == "__main__":
    main()
