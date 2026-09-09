# lbm-serve

Serve a `vla_foundry` `diffusion_policy` (LBM) checkpoint over openpi's
msgpack-over-websocket protocol, without installing `vla_foundry_internal`.

Extracted from `vla_foundry_internal` @ `92dbc090` by taking the transitive import closure
of `vla_foundry/serving/lbm_policy_server.py`, then removing the model-registry branches the
open-world checkpoints never take. Package paths are unchanged (`vla_foundry/...`).

```
70 .py files · ~10.1k LOC · 17 dependencies
   (vla_foundry_internal: 373 .py files, 35 dependencies)
```

Verified bit-identical to the source repo on all three open-world checkpoints — see
[Parity](#parity-with-the-source-repo).

## Install

```bash
uv venv --python 3.12
uv pip install -e .
```

## Run

```bash
uv run lbm-serve --checkpoint_directory /path/to/lbm_clean_spill --port 8010
# or: python -m vla_foundry.serving.lbm_policy_server --checkpoint_directory ... --port 8010
```

`--checkpoint_directory` must be laid out the way a training run leaves it:

```
<dir>/config.yaml                 parsed by draccus; also selects EMA vs raw weights
<dir>/config_processor.yaml       \
<dir>/config_normalizer.yaml       > RoboticsProcessor.from_pretrained(<dir>)
<dir>/stats.json                  /
<dir>/checkpoints/ema_N.pt        the weights that actually get served
```

**The served file is `ema_N.pt`, not `checkpoint_N.pt`.** All three open-world runs have
`ema.enabled: true`, and `_resolve_checkpoint_path` rewrites `checkpoint_` → `ema_`
accordingly. The S3 bundles at
`s3://tri-ml-sandbox-16011-us-west-2-datasets/open-world-policy-ckpt/lbm/<task>/` store the
files flat, so move the checkpoint into a `checkpoints/` subdirectory after download.

To *fine-tune* from served weights rather than serve them,
`vla_foundry/tri/ema_to_model_checkpoint.py` repacks `ema_N.pt` into a `checkpoint_*.pt`-shaped
file — handing the EMA file straight to `--model.resume_from_checkpoint` dies with
`KeyError: 'state_dict'`, and handing it `checkpoint_N.pt` silently starts from the raw
weights instead.

### Protocol

```
<- {"observation/image": u8[H,W,3], "observation/left_wrist_image": ...,
    "observation/right_wrist_image": ..., "observation/state": f32[16], "prompt": str}
-> {"actions": f32[16, 20]}
```

One frame per camera only: a checkpoint trained with frame stacking (`image_indices != [0]`)
is rejected at construction, because the payload has nowhere to put `t-1`.

## Why this is a separate repo and not part of openpi

The two stacks cannot share a virtualenv — three exact-pin collisions:

| | vla_foundry | openpi |
|---|---|---|
| `torch` | `==2.7.0` | `==2.7.1` |
| `transformers` | `==4.57.3` | `==4.53.2` |
| `numpy` | 2.x | `>=1.22.4,<2.0.0` |
| `jax` | — | `==0.5.3` |

`lbm_policy_server.py` was designed around this: *"Out-of-process is not a workaround, it is
the only option… The socket is the seam."* Both policies speak the same wire protocol, so
`serve` / `collect` / `convert` / `mix` on the open-world side are identical either way.

This does **not** shrink the environment — ~5.1 GB of the venv is `torch` + `nvidia` +
`triton` + `cusparselt`, which GPU inference needs regardless. What you get is a small,
pinned, readable tree instead of a moving training repo, and no `ray` / `moto` / `rosbags` /
`cuda-fps` to fail at install time.

## Parity with the source repo

```bash
scripts/parity_check.sh                                   # defaults to clean_spill / ema_1.pt
CKPT_DIR=/path/to/lbm_bike_rotor CKPT_NAME=ema_5.pt scripts/parity_check.sh
```

Runs one fixed observation through `vla_foundry_internal` twice — establishing the GPU
nondeterminism floor — and through this repo once, then compares the `(16, 20)` action chunk.

| checkpoint | floor (A vs B) | full vs extracted (A vs C) | |
|---|---|---|---|
| `clean_spill` / `ema_1.pt` | `0.000e+00` | `0.000e+00` | bit-identical |
| `bike_rotor` / `ema_5.pt` | `0.000e+00` | `0.000e+00` | bit-identical |
| `breakfast_table` / `ema_3.pt` | `0.000e+00` | `0.000e+00` | bit-identical |

**Re-run this after any edit to the vendored files.** Registry pruning in particular can
silently swap a code path without raising.

## What was removed, and what that costs

Files are byte-identical copies of upstream except for five files, each carrying a
`TRIMMED FOR SERVING` note explaining what was dropped and why:

| file | change |
|---|---|
| `models/__init__.py` | dropped eager `maniflow`, `transformer_hf`, `vlm`, `vlm_hf` registration imports |
| `models/diffusion/__init__.py` | kept the noise schedulers; dropped `StableDiffusion` / `UNet` / `UNetDiffusers` |
| `models/diffusion_policy/__init__.py` | dropped `clip_openclip` and the two VLM backbone registrations |
| `models/vision_language_backbones/__init__.py` | kept `clip_backbone`; the other three now raise explicitly |
| `data/utils.py` | reduced to the two functions the serving path calls |

That made 17 files unreachable (maniflow ×7, `vit*`, `vlm*`, `stable_diffusion`,
`unet_diffusers`, `clip_openclip`), which were deleted, and removed `timm`,
`open-clip-torch`, `webdataset` and `tqdm` from the dependency set.

**The cost:** this repo serves `type: diffusion_policy` over a `clip_backbone` only. A
checkpoint using a `vit_backbone`, `vlm_backbone`, `vlm_foundry_backbone`, `clip_openclip`,
`maniflow` or `stable_diffusion` will now fail with an explicit error instead of working. All
three open-world checkpoints are `clip_backbone` on `openai/clip-vit-base-patch32` with a
24-layer/1024-hidden transformer and `use_flow_matching_scheduler: true`.

Two deps that look droppable but are not:

- **`diffusers`** — `FlowMatchingScheduler` is defined in `noise_scheduler_diffusers.py`,
  which imports `DDPMScheduler` at module scope. It is on the live path despite
  `use_diffusers_scheduler: false`.
- **`boto3` / `botocore` / `fsspec`** — `file_utils.py` uses `fsspec` in ~10 functions and
  imports `S3Path`. Removing them means rewriting `file_utils`, for three small pure-Python
  packages. Not worth the divergence.

## Upstream

To re-sync after a `vla_foundry_internal` change to the serving path, re-take the closure
from the new commit and re-apply the five trims, then run `scripts/parity_check.sh`. The
five trimmed files are the only merge conflicts you should ever see.
