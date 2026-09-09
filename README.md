# lbm-serve

Serve a `vla_foundry` `diffusion_policy` (LBM) checkpoint over openpi's
msgpack-over-websocket protocol, without installing `vla_foundry_internal`.

Extracted from `vla_foundry_internal` @ `92dbc090` on 2026-09-09 by taking the transitive
import closure of `vla_foundry/serving/lbm_policy_server.py`. **Files are byte-identical
copies — nothing was edited**, and package paths are unchanged (`vla_foundry/...`), so
imports resolve the same way and a checkpoint trained there loads here bit-for-bit.

```
87 .py files · 591 KB · ~13.6k LOC     (vs 373 .py files in vla_foundry_internal)
```

## Why this is a separate repo and not part of openpi

The two stacks cannot share a virtualenv — three exact-pin collisions:

| | vla_foundry | openpi |
|---|---|---|
| `torch` | `==2.7.0` | `==2.7.1` |
| `transformers` | `==4.57.3` | `==4.53.2` |
| `numpy` | 2.x | `>=1.22.4,<2.0.0` |
| `jax` | — | `==0.5.3` |

`lbm_policy_server.py` was designed around this: *"Out-of-process is not a workaround, it
is the only option… The socket is the seam."* Both policies speak the same wire protocol, so
`serve` / `collect` / `convert` / `mix` on the open-world side are identical either way.

Note this does **not** shrink the environment — ~5.1 GB of the 6.5 GB venv is
`torch` + `nvidia` + `triton` + `cusparselt`, which GPU inference needs regardless. What you
get is a small, pinned, readable tree instead of a moving training repo.

## Install

```bash
uv venv --python 3.12 && uv pip install -e .
```

## Run

```bash
python -m vla_foundry.serving.lbm_policy_server \
    --checkpoint_directory /path/to/lbm_clean_spill \
    --port 8010
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

To *fine-tune* from served weights rather than serve them, `vla_foundry/tri/ema_to_model_checkpoint.py`
repacks `ema_N.pt` into a `checkpoint_*.pt`-shaped file — handing the EMA file straight to
`--model.resume_from_checkpoint` dies with `KeyError: 'state_dict'`, and handing it
`checkpoint_N.pt` silently starts from the raw weights instead.

### Protocol

```
<- {"observation/image": u8[H,W,3], "observation/left_wrist_image": ...,
    "observation/right_wrist_image": ..., "observation/state": f32[16], "prompt": str}
-> {"actions": f32[16, 20]}
```

One frame per camera only: a checkpoint trained with frame stacking (`image_indices != [0]`)
is rejected at construction, because the payload has nowhere to put `t-1`.

## Parity with the source repo

```bash
scripts/parity_check.sh
```

Runs the same fixed observation through `vla_foundry_internal` twice (establishing the GPU
nondeterminism floor) and through this repo once, then compares action chunks. Re-run it
after any edit to the vendored files — registry pruning in particular can silently swap a
code path without raising.

## Trimming further

The tree imports 21 third-party roots, but most come from registry side-effect imports
rather than the serving path. These checkpoints are `type: diffusion_policy` with a
`clip_backbone` on `openai/clip-vit-base-patch32`, 24-layer/1024-hidden transformer,
`use_flow_matching_scheduler: true`, `use_diffusers_scheduler: false` — so one path is live
and the alternatives are dead weight:

| dep | reachable only from | safe to cut? |
|---|---|---|
| `timm` | `models/vit_hf.py`, `models/maniflow/*` | yes — alternative backbones |
| `open-clip-torch` | `models/diffusion_policy/clip_openclip.py`, `distributed.py` | yes — the openclip variant |
| `webdataset` | `data/utils.py` | yes — dataloading only |
| `boto3` / `botocore` / `fsspec` | `aws/s3_*`, `file_utils.py` | yes if you load from local disk |
| `diffusers` | `models/diffusion/*`, `diffusion_policy.py` | needs a lazy import; config disables it |
| `tdigest-rs` | `data/robotics/utils.py` | likely stats-computation only |
| `opencv-python` | `data/processor/robotics_processor.py` | on the live path (resize); torchvision could replace it |

Cutting the first four requires editing `vla_foundry/models/__init__.py` and
`models/vision_language_backbones/__init__.py` to drop eager registration imports — which is
the first divergence from upstream, so gate it on `scripts/parity_check.sh`. Estimated floor
is ~55 files / ~9k LOC / ~11 deps.
