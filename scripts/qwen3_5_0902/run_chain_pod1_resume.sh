#!/usr/bin/env bash
set -euo pipefail

# 0902 Pod-1 RESUME chain (FOREGROUND, platform entry command):
#   P0 and P18 pretrains finished on 09-02 morning (global batch 128,
#   bsz16x1); the original chains died post-pretrain from an in-flight
#   script edit (bash re-read a replaced file on CFS). Remaining work here:
#     1. S0  = SFT k=0 baseline   (stage-1: 0901_pretrain_qwen35_4b_dat_nogate)
#     2. S18 = SFT ee18           (stage-1: 0901_pretrain_qwen35_4b_dat_nogate_ee18)
#   Both stage-1 ckpts are verified complete on CFS — no waiting needed.
#
#   bash /home/ea-cvfa-aigc-x2v-2/xzf/ml-fastvlm/scripts/qwen3_5_0902/run_chain_pod1_resume.sh

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

XZF_ROOT="${XZF_ROOT:-/home/ea-cvfa-aigc-x2v-2/xzf}"
LOG_DIR="${LOG_DIR:-$XZF_ROOT/vldat_experiments/logs_0901}"
STAMP="$(date +%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# GPU/CPU telemetry sidecar (60s snapshots to CFS).
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

banner "Resume stage 1/2: SFT k=0 (baseline)"
bash scripts/qwen3_5_0902/exp_sft_qwen35_4b_dat_ivcap.sh 2>&1 \
    | tee "$LOG_DIR/pod1_${STAMP}_2_sft_k0.log"

banner "Resume stage 2/2: SFT ee18"
bash scripts/qwen3_5_0902/exp_sft_qwen35_4b_dat_ivcap_ee18.sh 2>&1 \
    | tee "$LOG_DIR/pod1_${STAMP}_3_sft_ee18.log"

banner "Pod-1 resume chain DONE"
