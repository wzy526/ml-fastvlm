#!/usr/bin/env bash
# 8 HR-Bench 4K questions through the two new eval paths (whole-image window,
# density-ratio cap 4) before the ablations. Writes to /tmp only; ~5 min.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
mkzero "$Z2" "$M2"

smoke() {   # smoke NAME [VAR=value ...]
    local out=/tmp/abl_smoke_$1
    shift
    say "smoke $*"
    env MODEL_PATH="$M2" WHERE_PT="$W2" LDAT_PT="$H2" CROP_TOK=400 N_WIN=1 GRID=0 TOKS=640 \
        FAST_PREFIX=1 TASKS=hrbench4k LIMIT=8 PORT="$PORT" "$@" OUT="$out" \
        bash scripts/eval_ldat_pareto.sh > "$out.log" 2>&1
    PORT=$((PORT + 10))
    grep -aE "\[ldat\] windows|\| *hrbench4k|Traceback|Error" "$out.log" | tail -n 4
}
smoke win_full WIN_FULL=1 LDAT_PT="$Z2"
smoke ratio4 DAT_RATIO=4 LDAT_PT="$X/hinge_ldat.pt"
say "done"
