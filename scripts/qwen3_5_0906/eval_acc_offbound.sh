#!/usr/bin/env bash
set -uo pipefail

# Acceptance for the two bounded-offset retrain arms:
#   learned vs near-rigid (off_range=0.01) on HR-Bench 4K.
#
# If rigid ≈ learned, deformable sampling is still unused and should be cut.
# TAG is unique so resume-safe eval_pixel_sweep does not skip into old results.
#
# Usage (on the eval pod):
#   bash scripts/qwen3_5_0906/eval_acc_offbound.sh

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source scripts/eval_pod_env.sh

XZF_ROOT="${XZF_ROOT:-/home/ea-cvfa-aigc-x2v-2/xzf}"
CKPT_ROOT="${CKPT_ROOT:-$XZF_ROOT/vldat_experiments}"
LOG_DIR="${LOG_DIR:-$CKPT_ROOT/logs_0906_acc}"
PIXELS_LIST="${PIXELS_LIST:-1310720 1806336}"
TASK="${TASK:-hrbench4k}"
STAMP="$(date +%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# Quiet burn *workers* only; keep the auto_burn parent so the platform does
# not reclaim the pod. Same pattern as scripts/run_sysbench_qwen35.sh.
for p in $(pgrep -f "auto_burn.py" 2>/dev/null || true); do
    pkill -P "$p" 2>/dev/null || true
done
sleep 8
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader \
    | sed 's/^/[burn] /' || true

banner() { echo; echo "########## [$(date '+%F %T')] $1 ##########"; echo; }

banner "ACC 1/2: op10  learned(off_range=0, penalty in config) vs rigid(0.01)"
BASE_CKPT="$CKPT_ROOT/0901_sft_qwen35_4b_dat_ivcap_ee0_op10-merged" \
WORK="$CKPT_ROOT/_acc_op10" \
TAG=0906acc_op10 RANGES="0 0.01" \
PIXELS_LIST="$PIXELS_LIST" TASK="$TASK" LOG_DIR="$LOG_DIR" \
bash scripts/qwen3_5_0906/eval_offrange_sweep.sh \
  2>&1 | tee "$LOG_DIR/${STAMP}_op10_acc.log"

banner "ACC 2/2: or10  learned(off_range=1.0) vs rigid(0.01)"
# Must use 1.0 not 0 — writing off_range=0 would wipe the trained bound.
BASE_CKPT="$CKPT_ROOT/0901_sft_qwen35_4b_dat_ivcap_ee0_or10-merged" \
WORK="$CKPT_ROOT/_acc_or10" \
TAG=0906acc_or10 RANGES="1.0 0.01" \
PIXELS_LIST="$PIXELS_LIST" TASK="$TASK" LOG_DIR="$LOG_DIR" \
bash scripts/qwen3_5_0906/eval_offrange_sweep.sh \
  2>&1 | tee "$LOG_DIR/${STAMP}_or10_acc.log"

banner "ACC DONE — compare learned vs rigid within each arm"
