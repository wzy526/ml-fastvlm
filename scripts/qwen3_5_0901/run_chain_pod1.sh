#!/usr/bin/env bash
set -euo pipefail

# 0901 Pod-1 chain (FOREGROUND, platform entry command):
#   1. P0  = pretrain k=0 baseline
#   2. S0  = SFT k=0 baseline
#   3. S18 = SFT ee18  (stage-1 ckpt comes from Pod-2's P18 via shared CFS;
#            we poll for it before launching, in case Pod-2 is slower)
#
# Run as the platform task command (no nohup — the platform kills the pod
# when the entry process exits):
#   bash /home/ea-cvfa-aigc-x2v-2/xzf/ml-fastvlm/scripts/qwen3_5_0901/run_chain_pod1.sh
#
# Output goes to stdout (platform log collector) AND per-stage files on CFS
# (readable from the dev machine).

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

XZF_ROOT="${XZF_ROOT:-/home/ea-cvfa-aigc-x2v-2/xzf}"
CKPT_ROOT="${CKPT_ROOT:-$XZF_ROOT/vldat_experiments}"
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

# Poll CFS until a stage-1 ckpt's final save appears (config.json is written
# at the very start of the final save), then wait a grace period for the
# weight shards to finish writing before loading.
wait_for_ckpt() {
    local dir="$1"
    local max_h="${WAIT_MAX_HOURS:-24}"
    local waited=0
    while [[ ! -f "$dir/config.json" ]]; do
        if (( waited >= max_h * 3600 )); then
            echo "[ERROR] Waited ${max_h}h for $dir — giving up." >&2
            return 1
        fi
        echo "[wait] $(date '+%F %T') $dir not ready, sleeping 300s..."
        sleep 300
        waited=$((waited + 300))
    done
    if (( waited > 0 )); then
        echo "[wait] config.json found; 300s grace for weight shards..."
        sleep 300
    fi
}

banner "Stage 1/3: pretrain k=0 (baseline)"
bash scripts/qwen3_5_0901/exp_pretrain_qwen35_4b_dat_nogate.sh 2>&1 \
    | tee "$LOG_DIR/pod1_${STAMP}_1_pretrain_k0.log"

banner "Stage 2/3: SFT k=0 (baseline)"
bash scripts/qwen3_5_0901/exp_sft_qwen35_4b_dat_ivcap.sh 2>&1 \
    | tee "$LOG_DIR/pod1_${STAMP}_2_sft_k0.log"

banner "Stage 3/3: SFT ee18 (stage-1 from Pod-2)"
wait_for_ckpt "$CKPT_ROOT/0901_pretrain_qwen35_4b_dat_nogate_ee18"
bash scripts/qwen3_5_0901/exp_sft_qwen35_4b_dat_ivcap_ee18.sh 2>&1 \
    | tee "$LOG_DIR/pod1_${STAMP}_3_sft_ee18.log"

banner "Pod-1 chain DONE"
