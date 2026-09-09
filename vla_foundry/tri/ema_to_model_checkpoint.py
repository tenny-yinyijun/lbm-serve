"""Repack an `ema_N.pt` into a `checkpoint_*.pt`-shaped file, so a fine-tune can start from
the weights that were actually *served*.

Why this exists. A training run writes three files per checkpoint:

    checkpoints/checkpoint_N.pt   {"state_dict": ..., "checkpoint_num", "global_step", ...}
    checkpoints/ema_N.pt          {"ema_state_dict": ..., "ema_optimization_step", ...}
    checkpoints/optimizer_N.pt

`vla_foundry/serving/lbm_policy_server.py` serves `ema_N.pt` when `cfg.ema.enabled` -- the EMA
weights are the policy, and they are what a DAgger round is collected against. But
`file_utils.load_model_checkpoint`, which is what `--model.resume_from_checkpoint` goes
through, reads `checkpoint["state_dict"]`. Hand it the EMA file and it dies with
`KeyError: 'state_dict'`; hand it `checkpoint_N.pt` and it silently starts from the *raw*
weights instead -- a different policy from the one the corrections were recorded against,
which is precisely the thing DAgger's premise forbids. Neither failure mode is loud enough.

Both state dicts come from `get_unwrapped_model(model).state_dict()` /
`ema_model.model.state_dict()`, so they carry identical key sets with no `module.` prefix and
the repack is a pure key rename -- asserted below rather than assumed.

The output deliberately omits `datastrings` / `curr_shard_idx_per_dataset` /
`shard_shuffle_seed_per_dataset`: those describe where the *previous* run's dataloader was,
and a fine-tune reads a different dataset. It is therefore only valid with
`--model.resume_weights_only true`, which is checked for by refusing to write a name a
full resume would accept.

Usage:
    python -m vla_foundry.tri.ema_to_model_checkpoint \
        /path/to/run/checkpoints/ema_19.pt            # -> ema_19_as_weights.pt beside it
"""

import argparse
import os
import sys

import torch


def repack(ema_path: str, out_path: str | None = None, force: bool = False) -> str:
    base = os.path.basename(ema_path)
    if not base.startswith("ema_"):
        raise SystemExit(f"{base!r} is not an EMA checkpoint; expected a file named ema_<N>.pt")

    ckpt = torch.load(ema_path, map_location="cpu", weights_only=False)
    if "ema_state_dict" not in ckpt:
        raise SystemExit(
            f"{ema_path} has no 'ema_state_dict' (keys: {sorted(ckpt)}). It was probably saved "
            "by a run with ema.enabled=false, in which case checkpoint_<N>.pt already holds the "
            "only weights there are and no repack is needed."
        )

    if out_path is None:
        out_path = os.path.join(os.path.dirname(ema_path), base.removesuffix(".pt") + "_as_weights.pt")
    # Refuse to produce a `checkpoint_<N>.pt`. `get_latest_checkpoint` globs that pattern, so
    # writing one would make the serving path and `--resume_from_checkpoint` (without
    # weights_only) pick up a file that has no optimizer or dataloader state to go with it.
    if os.path.basename(out_path).startswith("checkpoint_") and not force:
        raise SystemExit(
            f"refusing to write {os.path.basename(out_path)!r}: that name is what "
            "`get_latest_checkpoint` looks for and what a full (non-weights-only) resume "
            "expects, and this file has no optimizer or dataloader state. Pass --force if you "
            "really mean it."
        )
    if os.path.exists(out_path) and not force:
        raise SystemExit(f"{out_path} already exists; pass --force to overwrite.")

    sanity_check(ema_path, ckpt["ema_state_dict"])

    torch.save(
        {
            "checkpoint_num": ckpt.get("checkpoint_num", 0),
            "state_dict": ckpt["ema_state_dict"],
            # Kept for the log line load_model_checkpoint prints; the value is not used when
            # resume_weights_only is set (main.py discards the returned step).
            "global_step": 0,
            "datastrings": None,
            "curr_shard_idx_per_dataset": None,
            "samples_seen": 0,
            "shard_shuffle_seed_per_dataset": None,
            # A breadcrumb, since the file is otherwise indistinguishable from a raw one.
            "repacked_from": os.path.abspath(ema_path),
        },
        out_path,
    )
    return out_path


def sanity_check(ema_path: str, ema_sd: dict) -> None:
    """Compare against the sibling `checkpoint_N.pt` when there is one.

    The keys must match exactly. If they do not, the two files came from different models and
    seeding a fine-tune with this one would either throw from `load_state_dict` or -- worse, if
    the mismatch is only in a few heads -- load partially under a `strict=False` somewhere.
    """
    sibling = os.path.join(
        os.path.dirname(ema_path), os.path.basename(ema_path).replace("ema_", "checkpoint_", 1)
    )
    if not os.path.exists(sibling):
        print(f"  (no sibling {os.path.basename(sibling)} to cross-check against)")
        return
    raw_sd = torch.load(sibling, map_location="cpu", weights_only=False)["state_dict"]
    missing, extra = set(raw_sd) - set(ema_sd), set(ema_sd) - set(raw_sd)
    if missing or extra:
        raise SystemExit(
            f"EMA and raw state dicts disagree on keys ({len(missing)} missing, {len(extra)} "
            f"extra); e.g. {sorted(missing)[:3]} / {sorted(extra)[:3]}"
        )
    # EMA weights track the raw ones, so they should be close but NOT identical. Identical
    # means the EMA never updated (alpha=1, or update_after_step past the run's end), which
    # would make "serve the EMA" and "serve the raw weights" the same thing.
    drift = max(
        float((ema_sd[k].float() - raw_sd[k].float()).abs().max())
        for k in raw_sd
        if raw_sd[k].is_floating_point() and raw_sd[k].numel()
    )
    print(f"  {len(ema_sd)} tensors, max |ema - raw| = {drift:.3e}")
    if drift == 0.0:
        print("  WARNING: EMA is bit-identical to the raw weights -- EMA never updated.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ema_checkpoint", help="Path to checkpoints/ema_<N>.pt")
    parser.add_argument("--out", default=None, help="Output path (default: <ema>_as_weights.pt beside it).")
    parser.add_argument("--force", action="store_true", help="Overwrite, and allow a checkpoint_* name.")
    args = parser.parse_args()

    out = repack(args.ema_checkpoint, args.out, args.force)
    print(f"wrote {out}")
    print("  use with: --model.resume_from_checkpoint {} --model.resume_weights_only true".format(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
