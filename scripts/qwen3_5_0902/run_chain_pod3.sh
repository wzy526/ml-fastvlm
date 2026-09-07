#!/usr/bin/env bash
set -euo pipefail

# 0902 Pod-3 chain (FOREGROUND, platform entry command):
#   1. P2B = pretrain Qwen3.5-2B, grid 20x20, hd_early_exit_k=0 (no ViT EE)
#   2. S2B = SFT      Qwen3.5-2B, same DAT args, stage-1 from step 1
#
# Small-base control for the 4B early-exit sweep; both stages stay on this pod
# so there is nothing to wait for on CFS.
#
# Run as the platform task command (no nohup — the platform kills the pod
# when the entry process exits):
#   bash /home/ea-cvfa-aigc-x2v-2/xzf/ml-fastvlm/scripts/qwen3_5_0902/run_chain_pod3.sh
#
# Quick validation run (a few dozen steps per stage, e2e chain smoke test):
#   MAX_STEPS=50 SAVE_STEPS=50 WARMUP_STEPS=10 bash <this script>
#
# Output goes to stdout (platform log collector) AND per-stage files on CFS
# (readable from the dev machine).

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

XZF_ROOT="${XZF_ROOT:-/home/ea-cvfa-aigc-x2v-2/xzf}"
LOG_DIR="${LOG_DIR:-$XZF_ROOT/vldat_experiments/logs_0901}"
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

banner "Stage 1/2: pretrain 2B k=0 (grid 20)"
bash scripts/qwen3_5_0902/exp_pretrain_qwen35_2b_dat_nogate.sh 2>&1 \
    | tee "$LOG_DIR/pod3_${STAMP}_1_pretrain_2b_k0.log"

banner "Stage 2/2: SFT 2B k=0 (grid 20)"
bash scripts/qwen3_5_0902/exp_sft_qwen35_2b_dat_ivcap.sh 2>&1 \
    | tee "$LOG_DIR/pod3_${STAMP}_2_sft_2b_k0.log"

banner "Pod-3 chain DONE"
