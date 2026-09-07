#!/usr/bin/env bash
set -euo pipefail

# 0906 off_range sweep -- INFERENCE ONLY, no retraining.
# ============================================================================
#
# The sampler's straight-through clamp lets offsets overshoot [-1,1] unpunished,
# so on the shipped 0901 4B ckpt 53.6% of sampling points sit pinned to the
# image border and 69.8% of the HD attention mass reads them. Bounding the
# offset to off_range*tanh(raw) is a pure forward-pass change: it needs no new
# parameters, so it can be switched on for an ALREADY TRAINED checkpoint.
#
# Measured on that ckpt with no retraining (19 images / 42 questions):
#   off_range=0.3 -> pinned 11.1%, attention-on-pinned 17.0%, w_hd 0.18 -> 0.26
#
# This sweep answers the question that decides everything downstream: does the
# accuracy follow? If it does, the fix is free and retraining only compounds it.
# If it drops, the offsets are entangled with the clamp and the retrain arm
# (run_chain_offbound.sh) is the only route.
#
# off_range is read from config.json's dat_extra_args, so each arm is just the
# same weights with a patched config -- symlinked, not copied (8.6 GB each).
#
# Usage:
#   bash scripts/qwen3_5_0906/eval_offrange_sweep.sh
#   RANGES="0 0.3" PIXELS_LIST=1310720 bash ...   # quick version
#   DRY_RUN=1 bash ...                            # print the plan only

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Self-contained: eval_pixel_sweep.sh needs LMMS_EVAL_DIR / VLDAT_VENV / the
# offline flags, and defaults to an autodl path when they are unset.
if [[ -z "${LMMS_EVAL_DIR:-}" || ! -d "${LMMS_EVAL_DIR:-/nonexistent}" ]]; then
    source scripts/eval_pod_env.sh
fi

XZF_ROOT="${XZF_ROOT:-/home/ea-cvfa-aigc-x2v-2/xzf}"
CKPT_ROOT="${CKPT_ROOT:-$XZF_ROOT/vldat_experiments}"
BASE_CKPT="${BASE_CKPT:-$CKPT_ROOT/0901_sft_qwen35_4b_dat_ivcap-merged}"
WORK="${WORK:-$CKPT_ROOT/_offrange_arms}"
LOG_DIR="${LOG_DIR:-$XZF_ROOT/vldat_experiments/logs_0906_offrange}"
TAG="${TAG:-0906or}"
TASK="${TASK:-hrbench4k}"

# 0 = legacy clamp (the control). 0.2/0.3/0.4 are 2x/3x/4x the grid pitch
# (0.0997 at grid=20).
RANGES="${RANGES:-0 0.2 0.3 0.4}"

# Two points from the corrected resolution grid: LR 1120^2 (1225 tokens) and
# LR 1344^2 (1764 tokens), both at hr_scale=3 against the 4K native cap.
PIXELS_LIST="${PIXELS_LIST:-1310720 1806336}"

mkdir -p "$LOG_DIR" "$WORK"

echo "[0906-or] base ckpt : $BASE_CKPT"
echo "[0906-or] ranges    : $RANGES"
echo "[0906-or] pixels    : $PIXELS_LIST"
echo "[0906-or] logs      : $LOG_DIR"

if [[ ! -f "$BASE_CKPT/config.json" ]]; then
    echo "[ERROR] base ckpt not found: $BASE_CKPT" >&2; exit 1
fi

STAMP="$(date +%m%d_%H%M%S)"
RUN_IDX=0

for r in $RANGES; do
    slug="or$(echo "$r" | tr -d '.')"
    arm="$WORK/$slug"

    # Symlink the weights, rewrite only config.json.
    rm -rf "$arm"; mkdir -p "$arm"
    for f in "$BASE_CKPT"/*; do ln -s "$f" "$arm/"; done
    rm -f "$arm/config.json"
    python3 - "$BASE_CKPT" "$arm" "$r" <<'PY'
import json, sys
src, dst, r = sys.argv[1], sys.argv[2], float(sys.argv[3])
c = json.load(open(f"{src}/config.json"))
c.setdefault("dat_extra_args", {})["off_range"] = r
json.dump(c, open(f"{dst}/config.json", "w"), indent=1)
PY

    export PORT=$((34000 + RUN_IDX * 100))
    RUN_IDX=$((RUN_IDX + 1))
    LOG="$LOG_DIR/${STAMP}_${TASK}_${slug}.log"

    echo
    echo "########## [$(date '+%F %T')] off_range=$r  task=$TASK  port=$PORT ##########"
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "  DRY_RUN: PIXELS=\"$PIXELS_LIST\" bash scripts/eval_pixel_sweep.sh dat35 $arm $TASK ${TAG}_${slug}"
        continue
    fi
    if ! { PIXELS="$PIXELS_LIST" bash scripts/eval_pixel_sweep.sh \
            dat35 "$arm" "$TASK" "${TAG}_${slug}" 2>&1 | tee "$LOG"; }; then
        echo "[0906-or] FAILED off_range=$r — continuing" >&2
    fi
done

echo
echo "[0906-or] DONE. Compare against the off_range=0 arm in the same sweep;"
echo "          the 0906_res run is a different geometry and is NOT the control."
