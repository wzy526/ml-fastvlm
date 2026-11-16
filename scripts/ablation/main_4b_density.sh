#!/usr/bin/env bash
# Main table, not an ablation: a density head for 4B with the 2B recipe, so both
# Q-LAS rows use a trained head. Scored on V* (old prompt) / HR-Bench 4K / 8K at
# LR 640 and 1280. ~4 h training + ~1.5 h eval.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
W4=$M4B/where_lm8_ce20.pt
need "$M4B" "$W4" "$QA"

train_ldat ldat4b_hinge_noratio "$(train_base "$M4B" "$W4")" "$FULL_LOSS" || exit 1
ev d4b_qlas MODEL_PATH="$M4B" WHERE_PT="$W4" LDAT_PT="$A/heads/ldat4b_hinge_noratio.pt" \
    TOKS="640 1280" TASKS="vstar_bench hrbench4k hrbench8k"
say "done"
