#!/usr/bin/env bash
# Prove the extracted tree serves the same policy as vla_foundry_internal.
#
#   run A, B : full vla_foundry_internal, twice  -> the GPU nondeterminism floor
#   run C    : this repo                         -> compared against A
#
# A vs B tells you how much drift the hardware alone produces. C vs A has to be no worse.
set -uo pipefail

LBM_SERVE="${LBM_SERVE:-/home/tenny.yin/workspace/lbm-serve}"
VLA_FULL="${VLA_FULL:-/home/tenny.yin/workspace/vla_foundry_internal}"
PY="${PY:-$VLA_FULL/.venv/bin/python}"
CKPT_DIR="${CKPT_DIR:-/home/tenny.yin/workspace/lbm_runs/lbm_clean_spill}"
CKPT_NAME="${CKPT_NAME:-ema_1.pt}"
OUT="${OUT:-/tmp/lbm_parity}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

mkdir -p "$OUT"
run () {  # $1=label  $2=tree
  echo "### $1  (PYTHONPATH=$2)"
  PYTHONPATH="$2" "$PY" "$LBM_SERVE/scripts/dump_actions.py" \
    --checkpoint_directory "$CKPT_DIR" --checkpoint_name "$CKPT_NAME" \
    --out "$OUT/$1.npy" >"$OUT/$1.log" 2>&1
  local rc=$?
  [[ $rc -eq 0 ]] || { echo "FAILED (rc=$rc), tail of log:"; tail -20 "$OUT/$1.log"; return $rc; }
  echo "ok -> $OUT/$1.npy"
}

run A "$VLA_FULL"   || exit 1
run B "$VLA_FULL"   || exit 1
run C "$LBM_SERVE"  || exit 1

"$PY" - "$OUT" <<'PY'
import sys, json, numpy as np
out = sys.argv[1]
A, B, C = (np.load(f"{out}/{k}.npy") for k in "ABC")
def cmp(x, y, label):
    if x.shape != y.shape:
        print(f"{label}: SHAPE MISMATCH {x.shape} vs {y.shape}"); return None
    d = np.abs(x - y)
    print(f"{label}: max|d|={d.max():.3e}  mean|d|={d.mean():.3e}  exact={np.array_equal(x,y)}")
    return d.max()
print(f"shape={A.shape}")
floor = cmp(A, B, "A vs B  (same tree, nondeterminism floor)")
delta = cmp(A, C, "A vs C  (full vs extracted)")
print()
if delta is None or floor is None:
    print("VERDICT: FAIL (shape mismatch)"); sys.exit(1)
if delta == 0.0:
    print("VERDICT: PASS (bit-identical)")
elif delta <= max(floor, 1e-6):
    print(f"VERDICT: PASS (within nondeterminism floor {floor:.3e})")
else:
    print(f"VERDICT: FAIL (extracted tree differs by {delta:.3e} > floor {floor:.3e})")
    sys.exit(1)
PY
