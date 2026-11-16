#!/usr/bin/env bash
# Ablation 3, LR tokens at matched total tokens.
#   Q-LAS: LR 320 / 1280 / 2560 + the 400-token window (LR 640 is comp_qlas).
#   The same SFT checkpoint without a window at 640 / 1040 / 1680 / 2960 tokens,
#   i.e. the Q-LAS totals. Eval only.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ev tok_lr TOKS="320 1280 2560"

px=""
for t in 640 1040 1680 2960; do
    px+="$((t * 1024)) "
done
say "eval native_sft (no window) 640 1040 1680 2960"
PIXELS="${px% }" OUT_ROOT="$A/native_sft" PORT="$PORT" \
    bash scripts/eval_pixel_sweep.sh base35 "$M2" "${TASKS_ABL// /,}" native_sft \
    >> "$L/native_sft.log" 2>&1
say "done"
