#!/usr/bin/env bash
set -uo pipefail

# Latency for every configuration the 0906 resolution sweep scored, so accuracy
# and latency can be paired point-for-point on one Pareto plot.
#
# Geometry must match eval_resolution_sweep.sh, NOT the earlier pareto runs
# (bench_q35_4B_k*_0906_0121, HR cap 2240², old token grid):
#   DAT   tokens 256/640/1280/1764  -> LR 512²/800²/1120²/1344², HR = 3x LR
#         (1536²/2400²/3360²/4032²), cap 16 257 024 px = native 4K, never hit.
#   base  tokens 2304/5625/11025/15876 -> the same four HR resolutions fed to
#         vanilla Qwen directly (equal-visual-information protocol), plus
#         2560/6400/11520 so the 0903 vanilla accuracy points (256..11520, still
#         valid: the wrapper bug only touched DAT's HR branch) get latency too.
#         Passing them as extra pareto tokens yields t_qwen_ms at those points;
#         the DAT columns for those rows are discarded. Native at 256..1764 comes
#         out of the DAT rows for free (equal-token protocol).
#
# One GPU, and the node must be otherwise idle: an 8-GPU lmms-eval on the same
# host skews single-stream latency, so this waits for it to drain first.
#
# Usage (tmux on the eval pod):
#   bash scripts/qwen3_5_0906/bench_pareto_0906res.sh
#   EARLY_KS="6" bash ...          # subset of exit depths

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source scripts/eval_pod_env.sh
source "$VLDAT_VENV/bin/activate"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

BASE_MODEL="${BASE_MODEL:-/workspace/model_cache/Qwen3.5-4B}"
DAT_CKPT="${DAT_CKPT:-/workspace/model_cache/0901_sft_qwen35_4b_dat_ivcap-merged}"
[[ -f "$BASE_MODEL/config.json" ]] || BASE_MODEL="/home/ea-cvfa-aigc-x2v-2/xzf/models/Qwen3.5-4B"
[[ -f "$DAT_CKPT/config.json" ]]   || DAT_CKPT="/home/ea-cvfa-aigc-x2v-2/xzf/vldat_experiments/0901_sft_qwen35_4b_dat_ivcap-merged"

HR_CAP="${HR_CAP:-16257024}"
DAT_TOKENS="${DAT_TOKENS:-256 640 1280 1764}"
BASE_TOKENS="${BASE_TOKENS:-2304 2560 5625 6400 11025 11520 15876}"
EARLY_KS="${EARLY_KS:-0 6 12 18}"
ITERS="${ITERS:-5}"
WARMUP="${WARMUP:-2}"
STAMP="$(date +%m%d_%H%M)"

echo "[pareto-0906res] waiting for lmms_eval to drain ($(date '+%T'))"
while pgrep -f "[l]mms_eval" > /dev/null 2>&1; do sleep 60; done
echo "[pareto-0906res] clear ($(date '+%T'))"

# Quiet the platform burn workers; keep the parent so the pod is not reclaimed.
for p in $(pgrep -f "auto_burn.py" 2>/dev/null || true); do pkill -P "$p" 2>/dev/null || true; done
sleep 20
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | sed 's/^/[gpu] /'

for K in $EARLY_KS; do
    TOKENS="$DAT_TOKENS"
    # The base protocol-C points only need measuring once; ride along with k=0.
    [[ "$K" == "0" ]] && TOKENS="$DAT_TOKENS $BASE_TOKENS"
    TAG="pareto_0906res_4B_k${K}_${STAMP}"
    echo
    echo "########## [$(date '+%F %T')] k=$K tokens=[$TOKENS] hr_cap=$HR_CAP tag=$TAG ##########"
    python test_inference_bench.py \
        --model-family qwen3_5 \
        --base-model "$BASE_MODEL" \
        --dat-ckpt "$DAT_CKPT" \
        --synthetic \
        --tasks pareto \
        --pareto-tokens $TOKENS \
        --hr-cap "$HR_CAP" \
        --hd-early-k "$K" \
        --warmup "$WARMUP" --iters "$ITERS" \
        --tag "$TAG" \
        || echo "[pareto-0906res] k=$K FAILED — continuing" >&2
done

echo
echo "[pareto-0906res] DONE $(date '+%F %T') -> bench_outputs/bench_pareto_0906res_4B_k*_${STAMP}.json"
