#!/usr/bin/env bash
set -euo pipefail

# 0906 bounded-offset chain (FOREGROUND, platform entry command):
#   1. pretrain Qwen3.5-4B DAT (nogate), off_range=0.3
#   2. SFT      Qwen3.5-4B DAT + 369k ivcap mix, same off_range
#
# WHY
# ---
# The legacy sampler uses a straight-through clamp,
#     sample_locs = x + (x.clamp(-1,1) - x).detach()
# which caps the forward value but lets the gradient keep pushing outward, so
# nothing ever penalizes an offset that overshoots. Measured on the 0901 4B
# ckpt over 19 HR-Bench 4K images (42 questions):
#
#   53.6% of sampling points pinned to the [-1,1] border   (L5 99.9%, L7 97.2%)
#   69.8% of HD attention mass lands on those pinned points
#   73.2% of the top-12 attended keys are pinned
#
# i.e. about half the sampling budget was reading the image edge.
#
# Bounding the offset with off_range*tanh(raw) -- what deformable-attention
# work normally does -- fixes it. Measured on the SAME ckpt with NO retraining,
# purely by switching the sampler at inference:
#
#   pinned points        53.6% -> 11.1%
#   attention on pinned  69.8% -> 17.0%
#   w_hd (HD share)       0.18 -> 0.26     (+44%, the model itself upweights HD)
#
# So this chain retrains with the bound in place, where the offsets can adapt
# to it instead of being squashed after the fact.
#
# SCOPE: single variable. intention_inject stays 'gate' and the spatial guide
# stays off, so any delta is attributable to the offset bound alone. The
# post-norm FiLM route (intention_inject=film) is implemented and warm-start
# safe, but belongs in a separate arm.
#
# off_range=0.3 is ~3x the reference grid pitch (0.0997 at grid=20), so each
# point can still travel three cells from its reference while 8 sampling groups
# keep the union coverage dense.
#
# Run as the platform task command (no nohup -- the platform kills the pod when
# the entry process exits):
#   bash /home/ea-cvfa-aigc-x2v-2/xzf/ml-fastvlm/scripts/qwen3_5_0906/run_chain_offrange.sh
#
# Quick validation run (a few dozen steps per stage):
#   MAX_STEPS=50 SAVE_STEPS=50 WARMUP_STEPS=10 bash <this script>

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Two ways to bound the sampling positions; set exactly one.
#
#   OFF_RANGE=r    offsets become r*tanh(raw). Simple, but any bounded smooth
#                  map has a vanishing derivative far out, so it damps exactly
#                  the points that travel furthest (60x at the measured max raw
#                  offset of 2.754) and it also compresses points that never
#                  left the region.
#
#   OFF_PENALTY=w  honest clamp plus loss += w*mean(relu(|ref+off|-1)^2).
#                  Gradient is exactly 1 inside the region and the penalty
#                  supplies the pull-back a bare clamp lacks, so nothing is
#                  damped and the in-region reach stays unbounded. Verified
#                  against the loss-routed form in scripts/_test_off_penalty.py.
export OFF_RANGE="${OFF_RANGE:-0}"
export OFF_PENALTY="${OFF_PENALTY:-1.0}"
export EARLY_K="${EARLY_K:-0}"          # full-depth HD, so k is not a confound

if [[ "$OFF_RANGE" != "0" && "$OFF_PENALTY" != "0" ]]; then
    echo "[ERROR] set only one of OFF_RANGE / OFF_PENALTY (got $OFF_RANGE / $OFF_PENALTY)" >&2
    exit 1
fi
if [[ "$OFF_RANGE" == "0" && "$OFF_PENALTY" == "0" ]]; then
    echo "[ERROR] both bounds disabled — that is the legacy arm, already trained" >&2
    exit 1
fi

if [[ "$OFF_RANGE" != "0" ]]; then
    DEF_SUFFIX="_or$(echo "$OFF_RANGE" | tr -d '.')"
else
    DEF_SUFFIX="_op$(echo "$OFF_PENALTY" | tr -d '.')"
fi
export EXP_SUFFIX="${EXP_SUFFIX:-$DEF_SUFFIX}"

# SIZE picks the exp scripts. 4B names always carry _ee<k>; the 2B family keeps
# its 0902 name for k=0 and appends _ee<k> otherwise (see the 2B exp scripts).
export SIZE="${SIZE:-4B}"
case "$SIZE" in
    4B) PRE_SCRIPT=scripts/qwen3_5_0902/exp_pretrain_qwen35_4b_dat_nogate_ee12.sh
        SFT_SCRIPT=scripts/qwen3_5_0902/exp_sft_qwen35_4b_dat_ivcap_ee12.sh
        STAGE1_NAME="0901_pretrain_qwen35_4b_dat_nogate_ee${EARLY_K}${EXP_SUFFIX}"
        STAGE2_NAME="0901_sft_qwen35_4b_dat_ivcap_ee${EARLY_K}${EXP_SUFFIX}" ;;
    2B) PRE_SCRIPT=scripts/qwen3_5_0902/exp_pretrain_qwen35_2b_dat_nogate.sh
        SFT_SCRIPT=scripts/qwen3_5_0902/exp_sft_qwen35_2b_dat_ivcap.sh
        EE_TAG=""; if [[ "$EARLY_K" != "0" ]]; then EE_TAG="_ee${EARLY_K}"; fi
        STAGE1_NAME="0902_pretrain_qwen35_2b_dat_nogate${EE_TAG}${EXP_SUFFIX}"
        STAGE2_NAME="0902_sft_qwen35_2b_dat_ivcap${EE_TAG}${EXP_SUFFIX}" ;;
    *)  echo "[ERROR] SIZE must be 4B or 2B (got $SIZE)" >&2; exit 1 ;;
esac

# The exp scripts write straight into $CKPT_ROOT/<EXP_NAME> with no overwrite
# check, so a reused suffix would clobber a finished arm's checkpoints.
_STAGE1_DIR="${XZF_ROOT:-/home/ea-cvfa-aigc-x2v-2/xzf}/vldat_experiments/$STAGE1_NAME"
if [[ -e "$_STAGE1_DIR" ]]; then
    echo "[ERROR] $_STAGE1_DIR already exists — pick a new EXP_SUFFIX (e.g. EXP_SUFFIX=${EXP_SUFFIX}_nz)" >&2
    exit 1
fi

# Ports offset from the ee6 arm (40995 / 40999).
export MASTER_PORT_PRETRAIN="${MASTER_PORT_PRETRAIN:-41005}"
export MASTER_PORT_SFT="${MASTER_PORT_SFT:-41009}"

XZF_ROOT="${XZF_ROOT:-/home/ea-cvfa-aigc-x2v-2/xzf}"
LOG_DIR="${LOG_DIR:-$XZF_ROOT/vldat_experiments/logs_0906}"
STAMP="$(date +%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

HOST_TAG="$(hostname -s 2>/dev/null || echo pod)"
GPU_STATS="$LOG_DIR/${HOST_TAG}_${SIZE}${EXP_SUFFIX}_${STAMP}_gpu.csv"
CPU_STATS="$LOG_DIR/${HOST_TAG}_${SIZE}${EXP_SUFFIX}_${STAMP}_cpu.log"
echo "ts,idx,util_pct,mem_used_mib,mem_total_mib,power_w" >> "$GPU_STATS"
(
    while true; do
        TS="$(date '+%F %T')"
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw \
            --format=csv,noheader,nounits 2>/dev/null \
            | awk -v t="$TS" -F', *' '{OFS=","; print t,$1,$2,$3,$4,$5}' >> "$GPU_STATS"
        echo "$TS loadavg=$(cut -d' ' -f1-3 /proc/loadavg) nproc=$(nproc)" >> "$CPU_STATS"
        sleep 60
    done
) &
GPU_STATS_PID=$!
trap 'kill $GPU_STATS_PID 2>/dev/null' EXIT

banner() { echo; echo "########## [$(date '+%F %T')] $1 ##########"; echo; }

banner "Stage 1/2: pretrain $SIZE DAT off_range=$OFF_RANGE off_penalty=$OFF_PENALTY k=$EARLY_K -> $STAGE1_NAME"
MASTER_PORT="$MASTER_PORT_PRETRAIN" \
bash "$PRE_SCRIPT" 2>&1 \
    | tee "$LOG_DIR/ob${EXP_SUFFIX}_${STAMP}_1_pretrain_${SIZE}.log"

banner "Stage 2/2: SFT $SIZE DAT off_range=$OFF_RANGE off_penalty=$OFF_PENALTY k=$EARLY_K -> $STAGE2_NAME"
MASTER_PORT="$MASTER_PORT_SFT" \
bash "$SFT_SCRIPT" 2>&1 \
    | tee "$LOG_DIR/ob${EXP_SUFFIX}_${STAMP}_2_sft_${SIZE}.log"

banner "bounded-offset chain DONE"
echo "stage-1 ckpt: $STAGE1_NAME"
echo "stage-2 ckpt: $STAGE2_NAME"
echo "next: merge LoRA, then eval (off_range is read from config, no eval flag needed)"
