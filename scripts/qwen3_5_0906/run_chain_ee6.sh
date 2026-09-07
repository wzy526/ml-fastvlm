#!/usr/bin/env bash
set -euo pipefail

# 0906 EE6 chain (FOREGROUND, platform entry command):
#   1. pretrain Qwen3.5-4B DAT (nogate), hd_early_exit_k=6
#   2. SFT      Qwen3.5-4B DAT + 369k ivcap mix, same k
#
# Arm 4 of the early-exit sweep. Existing arms: k=0 (full 24/24), k=18 (75%),
# k=12 (50%) — all three are accuracy-neutral on HR-Bench, so this pushes to
# 25% depth to find where accuracy finally breaks.
#
# Measured speed at k=6 (B300, E2E prefill vs native Qwen3.5-4B, 0906):
#   R=2016 1.50x   R=2688 2.48x   R=3360 3.18x
# versus k=12's 1.22x / 1.72x / 1.99x. HD-ViT time is linear in k with no
# per-call floor, so if accuracy holds this is a free 1.6x over the k=12 arm.
#
# Both stages reuse the ee12 scripts unchanged — EARLY_K is already a knob,
# and EXP_NAME / STAGE1_NAME derive from it, so stage 2 finds stage 1 by name:
#   stage 1 -> 0901_pretrain_qwen35_4b_dat_nogate_ee6
#   stage 2 -> 0901_sft_qwen35_4b_dat_ivcap_ee6
#
# Run as the platform task command (no nohup — the platform kills the pod
# when the entry process exits):
#   bash /home/ea-cvfa-aigc-x2v-2/xzf/ml-fastvlm/scripts/qwen3_5_0906/run_chain_ee6.sh
#
# Quick validation run (a few dozen steps per stage, e2e chain smoke test):
#   MAX_STEPS=50 SAVE_STEPS=50 WARMUP_STEPS=10 bash <this script>

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export EARLY_K="${EARLY_K:-6}"

# Ports offset from the ee12 arm (40985 / 40989) so a leftover run cannot clash.
export MASTER_PORT_PRETRAIN="${MASTER_PORT_PRETRAIN:-40995}"
export MASTER_PORT_SFT="${MASTER_PORT_SFT:-40999}"

XZF_ROOT="${XZF_ROOT:-/home/ea-cvfa-aigc-x2v-2/xzf}"
LOG_DIR="${LOG_DIR:-$XZF_ROOT/vldat_experiments/logs_0906}"
STAMP="$(date +%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# GPU/CPU telemetry sidecar: 60s snapshots to CFS, so the REAL training pod's
# utilization can be inspected from the dev machine (pods are not directly
# reachable over ssh; logs on CFS are the only window).
HOST_TAG="$(hostname -s 2>/dev/null || echo pod)"
GPU_STATS="$LOG_DIR/${HOST_TAG}_${STAMP}_gpu.csv"
CPU_STATS="$LOG_DIR/${HOST_TAG}_${STAMP}_cpu.log"
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

banner "Stage 1/2: pretrain 4B DAT k=$EARLY_K (grid 20)"
MASTER_PORT="$MASTER_PORT_PRETRAIN" \
bash scripts/qwen3_5_0902/exp_pretrain_qwen35_4b_dat_nogate_ee12.sh 2>&1 \
    | tee "$LOG_DIR/ee6_${STAMP}_1_pretrain_4b_k${EARLY_K}.log"

banner "Stage 2/2: SFT 4B DAT k=$EARLY_K (grid 20)"
MASTER_PORT="$MASTER_PORT_SFT" \
bash scripts/qwen3_5_0902/exp_sft_qwen35_4b_dat_ivcap_ee12.sh 2>&1 \
    | tee "$LOG_DIR/ee6_${STAMP}_2_sft_4b_k${EARLY_K}.log"

banner "EE6 chain DONE"
echo "stage-1 ckpt: 0901_pretrain_qwen35_4b_dat_nogate_ee${EARLY_K}"
echo "stage-2 ckpt: 0901_sft_qwen35_4b_dat_ivcap_ee${EARLY_K}"
echo "next: merge LoRA, then eval with dat_extra_args hd_early_exit_k=${EARLY_K}"
