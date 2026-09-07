#!/usr/bin/env bash
set -euo pipefail

# 0902 Pod-2 RESUME chain (FOREGROUND, platform entry command):
#   P18 finished on 09-02 morning; the chain died post-pretrain from an
#   in-flight script edit (bash re-read a replaced file on CFS). Remaining:
#     1. P12 = pretrain ee12 (50% HD ViT depth; same recipe as P0/P18:
#              global batch 128 = bsz16 x 8ranks x accum1 on B300)
#     2. S12 = SFT ee12 (consumes P12 from this pod)
#
#   bash /home/ea-cvfa-aigc-x2v-2/xzf/ml-fastvlm/scripts/qwen3_5_0902/run_chain_pod2_resume.sh

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

banner "Resume stage 1/2: pretrain ee12"
bash scripts/qwen3_5_0902/exp_pretrain_qwen35_4b_dat_nogate_ee12.sh 2>&1 \
    | tee "$LOG_DIR/pod2_${STAMP}_2_pretrain_ee12.log"

banner "Resume stage 2/2: SFT ee12"
bash scripts/qwen3_5_0902/exp_sft_qwen35_4b_dat_ivcap_ee12.sh 2>&1 \
    | tee "$LOG_DIR/pod2_${STAMP}_3_sft_ee12.log"

banner "Pod-2 resume chain DONE"
