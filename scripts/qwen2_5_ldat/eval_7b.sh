#!/usr/bin/env bash
# Use the same lmms-eval task prompts / scorer as the existing Qwen3.5 runner.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD${LMMS_EVAL_DIR:+:$LMMS_EVAL_DIR}${PYTHONPATH:+:$PYTHONPATH}"
export LMMS_EVAL_PLUGINS=ldat_eval

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
RUN_ROOT="${RUN_ROOT:-$HOME/vldat_experiments/qwen25_7b_ldat}"
WHERE_LAYER="${WHERE_LAYER:-8}"
WHERE_PT="${WHERE_PT:-$RUN_ROOT/where_lm${WHERE_LAYER}.pt}"
LDAT_PT="${LDAT_PT:-$RUN_ROOT/ldat.pt}"
OUT="${OUT:-$RUN_ROOT/eval}"
TASKS="${TASKS:-vstar_bench hrbench4k hrbench8k}"
NPROC="${NPROC:-8}"
DRY_RUN="${DRY_RUN:-0}"
MODE="${MODE:-ldat}"
GRID="${GRID:-0}"
N_WIN="${N_WIN:-1}"
# Unlike the older Pareto script, multi-window mode must choose its budget
# explicitly, so train/eval cannot silently disagree about tokens per window.
if [[ "$N_WIN" != 1 && "$GRID" == 0 ]]; then
    echo 'Set GRID explicitly for multi-window evaluation (e.g. GRID=20).' >&2; exit 1
fi
for task in $TASKS; do
    case "$MODE" in
    ldat)
        model=qwen2_5_ldat
        model_args="pretrained=$MODEL_PATH,where_head_pt=$WHERE_PT,ldat_pt=$LDAT_PT,where_layer=$WHERE_LAYER,lr_tok=${LR_TOK:-640},crop_tok=${CROP_TOK:-400},n_win=$N_WIN,grid=$GRID,dat_alpha=0.8,dat_ratio=0,dat_k=0.6,src_scale=4,supersample=2,ldat_d=256,hr_cap=${HR_CAP:-4096},prefix_cache=${PREFIX_CACHE:-False},fast_prefix=${FAST_PREFIX:-False},attn_implementation=sdpa"
        ;;
    base)
        model=qwen2_5_vl
        model_args="pretrained=$MODEL_PATH,min_pixels=784,max_pixels=$((${BASE_TOK:-640} * 784)),attn_implementation=sdpa"
        ;;
    *) echo "MODE must be ldat or base" >&2; exit 1 ;;
    esac
    cmd=(accelerate launch --num_processes "$NPROC" --main_process_port "${PORT:-29691}" -m lmms_eval
        --model "$model" --model_args "$model_args"
        --tasks "$task" --batch_size 1 --log_samples --output_path "$OUT/${MODE}_$task")
    if (( NPROC > 1 )); then cmd=("${cmd[@]:0:2}" --multi_gpu "${cmd[@]:2}"); fi
    if [[ -n "${LIMIT:-}" ]]; then cmd+=(--limit "$LIMIT"); fi
    printf '%q ' "${cmd[@]}"; printf '\n'
    if [[ "$DRY_RUN" != 1 ]]; then "${cmd[@]}"; fi
done
