#!/usr/bin/env python3
"""Run one `LBMOpenPIPolicy.infer` on a fixed synthetic observation and save the action chunk.

The point is to be importable from *either* tree: run it with
`PYTHONPATH=/path/to/lbm-serve` and again with `PYTHONPATH=/path/to/vla_foundry_internal`,
then diff the two `.npy` files. If the extraction dropped or reordered anything on the
serving path, the actions move.

Determinism: the observation is built from a seeded numpy Generator, and torch is re-seeded
immediately before `infer` so the flow-matching noise draw is reproducible. Run the same
tree twice to establish the GPU nondeterminism floor before comparing across trees.
"""

import argparse
import json
import os

import numpy as np
import torch


def build_obs(seed: int, height: int, width: int, state_dim: int, prompt: str) -> dict:
    rng = np.random.default_rng(seed)
    def img():
        return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    return {
        "observation/image": img(),
        "observation/left_wrist_image": img(),
        "observation/right_wrist_image": img(),
        "observation/state": rng.standard_normal(state_dim).astype(np.float32),
        "prompt": prompt,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_directory", required=True)
    p.add_argument("--checkpoint_name", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_flow_steps", type=int, default=10)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--prompt", default="clean up the spill")
    args = p.parse_args()

    from vla_foundry.serving.lbm_policy_server import LBMOpenPIPolicy

    policy = LBMOpenPIPolicy(
        checkpoint_directory=args.checkpoint_directory,
        checkpoint_name=args.checkpoint_name,
        device=args.device,
        num_flow_steps=args.num_flow_steps,
    )

    obs = build_obs(args.seed, args.height, args.width, policy.proprioception_dim, args.prompt)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    result = policy.infer(obs)

    actions = np.asarray(result["actions"], dtype=np.float64)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.save(args.out, actions)

    import vla_foundry
    meta = {
        "tree": os.path.dirname(os.path.dirname(os.path.abspath(vla_foundry.__file__))),
        "checkpoint_path": policy.checkpoint_path,
        "shape": list(actions.shape),
        "dtype": str(actions.dtype),
        "action_horizon": policy.action_horizon,
        "action_dim": policy.action_dim,
        "state_dim": policy.proprioception_dim,
        "views": policy.payload_keys,
        "sum": float(actions.sum()),
    }
    with open(args.out + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
