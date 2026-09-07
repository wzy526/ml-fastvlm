#!/usr/bin/env bash
set -uo pipefail

# Fill the holes in the 4B results grid so every arm has every dataset.
#
# What exists (see _test_outputs): vanilla / k0 / EE12 / EE18 have the 0903
# general tasks (chartqa, docvqa_val, textvqa_val, gen7); EE6 has only the 0906
# HR-Bench / V* points. Nothing has the SFT-only control.
#
# PART=hdoff   DAT k=0 checkpoint with the HD branch OFF (LR tokens only) on
#              hrbench4k / hrbench8k / vstar_bench at the 0906 DAT grid. This is
#              the control the general-task table needs: vanilla never saw the
#              369k SFT mix, so DAT-minus-vanilla mixes the HD pathway with the
#              SFT data. DAT(HD on) minus DAT(HD off) is the HD pathway alone.
# PART=ee6gen  EE6 on the 0903 general tasks at the four budgets that are not
#              saturated (256/640/1280/2560 tok). 6400/11520 were identical to
#              2560 for every arm because the images are smaller than the budget.
# PART=hdoffgen  HD-off control on the general tasks at 1280 tok.
# PART=all     hdoff, ee6gen, hdoffgen in that order (~8 h on 8 GPUs).
#
# Geometry: fixed wrapper, hr_scale 3, HR cap 16.26 MP. The 0903 general-task
# numbers for k0 / EE12 / EE18 were produced by the pre-fix wrapper; on these
# datasets the images are smaller than 9x LR at all but the lowest budget, so
# HR = native either way and the comparison holds to within that caveat.
#
# Usage (tmux):  PART=all bash scripts/qwen3_5_0906/eval_fill_4b_gaps.sh

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source scripts/eval_pod_env.sh

XZF_ROOT="${XZF_ROOT:-/home/ea-cvfa-aigc-x2v-2/xzf}"
CKPT_ROOT="${CKPT_ROOT:-$XZF_ROOT/vldat_experiments}"
LOG_DIR="${LOG_DIR:-$XZF_ROOT/vldat_experiments/logs_0907_fill}"
mkdir -p "$LOG_DIR"
export HR_CAP="${HR_CAP:-16257024}"
export HR_SCALE="${HR_SCALE:-3}"

K0="$CKPT_ROOT/0901_sft_qwen35_4b_dat_ivcap-merged"
EE6="$CKPT_ROOT/0901_sft_qwen35_4b_dat_ivcap_ee6-merged"
DAT_PIXELS="262144 655360 1310720 1806336"
GEN_PIXELS="262144 655360 1310720 2621440"
GEN7="gqa,scienceqa_img,vizwiz_vqa_val,pope,mmbench_en_dev,seedbench,mme"
PART="${PART:-all}"
STAMP="$(date +%m%d_%H%M%S)"
RUN_IDX=0

run() {  # kind ckpt task pixels tag [DISABLE_HD]
    local kind=$1 ckpt=$2 task=$3 pixels=$4 tag=$5 hdoff=${6:-0}
    export PORT=$((37000 + RUN_IDX * 100)); RUN_IDX=$((RUN_IDX + 1))
    echo; echo "########## [$(date '+%F %T')] $tag / $task px=[$pixels] hdoff=$hdoff port=$PORT ##########"
    if ! { DISABLE_HD="$hdoff" PIXELS="$pixels" bash scripts/eval_pixel_sweep.sh "$kind" "$ckpt" "$task" "$tag" \
            2>&1 | tee "$LOG_DIR/${STAMP}_${task//,/+}_${tag}.log"; }; then
        echo "[fill] FAILED $tag / $task — continuing" >&2
    fi
}

for p in $(pgrep -f "auto_burn.py" 2>/dev/null || true); do pkill -P "$p" 2>/dev/null || true; done

if [[ "$PART" == "all" || "$PART" == "hdoff" ]]; then
    for t in hrbench4k hrbench8k vstar_bench; do
        run dat35 "$K0" "$t" "$DAT_PIXELS" 0907_4b_dat_k0_hdoff 1
    done
fi
if [[ "$PART" == "all" || "$PART" == "ee6gen" ]]; then
    for t in chartqa docvqa_val textvqa_val; do
        run dat35 "$EE6" "$t" "$GEN_PIXELS" 0907_4b_dat_ee6
    done
    run dat35 "$EE6" "$GEN7" "$GEN_PIXELS" 0907_4b_dat_ee6
fi
if [[ "$PART" == "all" || "$PART" == "hdoffgen" ]]; then
    for t in chartqa docvqa_val textvqa_val; do
        run dat35 "$K0" "$t" 1310720 0907_4b_dat_k0_hdoff 1
    done
    run dat35 "$K0" "$GEN7" 1310720 0907_4b_dat_k0_hdoff 1
fi
echo; echo "[fill] DONE $(date '+%F %T')"
