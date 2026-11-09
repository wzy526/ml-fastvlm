#!/usr/bin/env bash
# Pareto of the finalized learnable sampler (hinge_noratio, no window head).
#
# LR tokens: 640 1280 2560 6400. The window stays at 400 tokens.
# N_WIN=1 (default) is one aspect-preserving window of CROP_TOK.
# N_WIN>1 keeps each window at GRID x GRID (default 20) and writes a separate OUT.
# One lmms-eval launch per task per budget, 8 GPUs. A finished point
# (a *_results.json under the output dir) is skipped, so the script resumes.
#
#   bash scripts/eval_ldat_pareto.sh
#   LIMIT=8 TASKS=vstar_bench TOKS=640 bash scripts/eval_ldat_pareto.sh
#
# Scores land in $OUT/<task>_lr<N>/.

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
FASTVLM_DIR="$(pwd)"
# shellcheck disable=SC1091
source "${CONDA_SH:-$HOME/miniconda3/bin/activate}" "${CONDA_ENV:-$HOME/.conda/envs/fastvlm}"

E="${E:-$HOME/vldat_experiments/0924_miniE1_2k}"
X="${X:-/data/oss_bucket_0/wangziyi/vldat_experiments/sa1b_ldat}"
LMMS_EVAL_DIR="${LMMS_EVAL_DIR:-$HOME/lmms-eval}"
CROP_TOK="${CROP_TOK:-400}"
N_WIN="${N_WIN:-1}"
GRID="${GRID:-0}"
if [[ "$N_WIN" != 1 && "$GRID" == 0 ]]; then
    GRID=20
fi
if [[ -z "${OUT+x}" ]]; then
    if [[ "$N_WIN" == 1 && "$GRID" == 0 ]]; then
        OUT="$X/pareto_ldat_lr"
    else
        OUT="$X/pareto_ldat_lr_w${N_WIN}g${GRID}"
    fi
fi
MODEL_PATH="${MODEL_PATH:-$E/checkpoint-1000}"
WHERE_PT="${WHERE_PT:-$E/loc/where_lm8_ce20.pt}"
LDAT_PT="${LDAT_PT:-$X/hinge_noratio_ldat.pt}"
NPROC="${NPROC:-8}"
PORT="${PORT:-29720}"
LIMIT="${LIMIT:-}"

TASKS="${TASKS:-chartqa docvqa_val textvqa_val vstar_bench hrbench4k hrbench8k refcoco_bbox_rec_val gqa scienceqa_img vizwiz_vqa_val pope mmbench_en_dev seedbench mme}"
TOKS="${TOKS:-640 1280 2560 6400}"

export LMMS_EVAL_PLUGINS=ldat_eval
export LMMS_FASTVLM_PATH="$FASTVLM_DIR"
export PYTHONPATH="${LMMS_EVAL_DIR}:${FASTVLM_DIR}${PYTHONPATH:+:$PYTHONPATH}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DOWNLOAD_TIMEOUT=1200
export NUMEXPR_MAX_THREADS=64
export NCCL_DEBUG="${NCCL_DEBUG_LEVEL:-WARN}"
export MMBENCH_SKIP_GPT_EVAL=1
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/${USER:-$(id -un)}_triton_cache}"
mkdir -p "$TRITON_CACHE_DIR" "$OUT"

if [[ ! -d "$LMMS_EVAL_DIR" ]]; then
    echo "[ERROR] LMMS_EVAL_DIR not found: $LMMS_EVAL_DIR" >&2
    exit 1
fi
cd "$LMMS_EVAL_DIR"

echo "[pareto] tasks: $TASKS"
echo "[pareto] lr tokens: $TOKS   window=$CROP_TOK   n_win=$N_WIN grid=$GRID   out=$OUT"
echo "[pareto] head=$LDAT_PT"

for tok in $TOKS; do
    for task in $TASKS; do
        out="$OUT/${task}_lr${tok}"
        if [[ -n "$(find "$out" -name '*results.json' -print -quit 2>/dev/null)" ]]; then
            echo "[skip] $task lr=$tok"
            continue
        fi
        mkdir -p "$out"
        extra=()
        if [[ -n "$LIMIT" ]]; then
            extra=(--limit "$LIMIT")
        fi
        echo "------------------------------------------------------------------"
        echo "[$(date '+%F %T')] $task  lr_tok=$tok  window=$CROP_TOK  n_win=$N_WIN grid=$GRID"
        echo "------------------------------------------------------------------"
        if accelerate launch --num_processes "$NPROC" --main_process_port "$PORT" -m lmms_eval \
            --model qwen3_5_ldat \
            --model_args "pretrained=${MODEL_PATH},where_head_pt=${WHERE_PT},ldat_pt=${LDAT_PT},crop_tok=${CROP_TOK},lr_tok=${tok},n_win=${N_WIN},grid=${GRID},dat_alpha=0.8,dat_ratio=0,src_scale=4,ldat_d=256,attn_implementation=sdpa" \
            --tasks "$task" \
            --batch_size 1 \
            --log_samples \
            --output_path "$out" \
            ${extra[@]+"${extra[@]}"}; then
            echo "[done] $task lr=$tok"
        else
            echo "[FAIL] $task lr=$tok" >&2
        fi
        PORT=$((PORT + 1))
    done
done

echo "[pareto] finished $(date '+%F %T')"
