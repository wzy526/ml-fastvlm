#!/usr/bin/env bash
# Source from a training script. Expects $CKPT_ROOT, $EXP_NAME, $MODEL_PATH set.
#
# After torchrun returns (it is blocking, so all workers have exited and
# rank-0 has flushed adapter_model.* + non_lora_trainables.bin), this helper
# runs scripts/merge_lora_dat_weights.py once in a single Python process.
#
# Output: $CKPT_ROOT/$EXP_NAME-merged/  (override via $MERGED_DIR env)
# Skipped when adapter_config.json is missing (e.g. non-LoRA runs).
#
# Set DISABLE_AUTO_MERGE=1 to skip the merge step entirely.

set -uo pipefail

if [[ "${DISABLE_AUTO_MERGE:-0}" == "1" ]]; then
    echo "[merge] DISABLE_AUTO_MERGE=1, skipping LoRA merge"
    return 0 2>/dev/null || exit 0
fi

LORA_DIR="$CKPT_ROOT/$EXP_NAME"
MERGED_DIR="${MERGED_DIR:-${LORA_DIR}-merged}"

echo
echo "==================== auto-merge ===================="
echo "[merge] EXP_NAME      = $EXP_NAME"
echo "[merge] LORA_DIR      = $LORA_DIR"
echo "[merge] MERGED_DIR    = $MERGED_DIR"
echo "[merge] MODEL_BASE    = $MODEL_PATH"
echo "[merge] TORCH_DTYPE   = ${MERGE_DTYPE:-bfloat16}"
echo "===================================================="

if [[ ! -f "$LORA_DIR/adapter_config.json" ]]; then
    echo "[merge] No adapter_config.json found in $LORA_DIR; skipping merge."
    return 0 2>/dev/null || exit 0
fi

mkdir -p "$MERGED_DIR"

# Run merge in a sub-shell so a non-zero exit doesn't take down the parent
# script after a successful training run.
(
    cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && \
    python scripts/merge_lora_dat_weights.py \
        --model_base  "$MODEL_PATH" \
        --lora_path   "$LORA_DIR" \
        --output_dir  "$MERGED_DIR" \
        --torch_dtype "${MERGE_DTYPE:-bfloat16}"
)
MERGE_RC=$?

if [[ $MERGE_RC -eq 0 ]]; then
    echo "[merge] Done. Merged checkpoint at $MERGED_DIR"
else
    echo "[merge] FAILED with exit code $MERGE_RC. Lora ckpt is preserved at $LORA_DIR"
fi

return 0 2>/dev/null || true
