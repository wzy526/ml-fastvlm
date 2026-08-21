#!/usr/bin/env bash
# One-shot Qwen3-VL baseline: the full 14-benchmark suite at a SINGLE token
# budget (no pixel sweep). Companion to eval_pixel_sweep.sh /
# overnight_general_sweep.sh, which own the multi-point DAT curves.
#
# Token-budget alignment (IMPORTANT):
#   Qwen2.5-VL : patch 14 x merge 2 -> 28x28 =  784 px per LLM visual token
#   Qwen3-VL   : patch 16 x merge 2 -> 32x32 = 1024 px per LLM visual token
# So the pixel grid from eval_pixel_sweep.sh is NOT reusable as-is: the same
# `max_pixels` buys Qwen3-VL ~24% fewer tokens. This script takes the token
# count directly and derives pixels as TOK*1024, so the comparison against a
# DAT `tok<N>` column is apples-to-apples.
#
# Usage:
#   bash scripts/eval_qwen3vl_single_point.sh [CKPT] [TAG] [TOK]
#
# Example (defaults):
#   nohup bash scripts/eval_qwen3vl_single_point.sh > qwen3vl2b.log 2>&1 &
#
# Env knobs:
#   TASKS_OVERRIDE  space-separated task list  (default: the 14-task suite)
#   MODEL           lmms-eval model name       (default: qwen3_vl; `qwen3_5`
#                   switches to Qwen's sampled+thinking recipe for Qwen3.5)
#   MARGS_EXTRA     appended to --model_args    (e.g. "enable_thinking=False")
#   GPUS / NPROC    devices / ranks            (default: 0..7 / 8)
#   PORT            starting main_process_port (default: 30300, +1 per task)
#   OUT_ROOT_DIR    results root               (default: <repo>/_test_outputs)
#   CONDA_ENV       conda env                  (default: fastvlm)
#   LMMS_EVAL_DIR   lmms-eval checkout         (default: ~/lmms-eval)
#
# Resumable: any task with a `done` marker is skipped.

# No `-e`: one failing benchmark must not abort the remaining 13.
set -uo pipefail

CKPT="${1:-/data/oss_bucket_0/wangziyi/official_ckpt/Qwen3-VL-2B-Instruct}"
TAG="${2:-qwen3vl2b}"
TOK="${3:-2560}"

PX=$((TOK * 1024))
MIN_PIXELS="${MIN_PIXELS:-32768}"   # 32 tokens' worth, mirrors 28224 for Qwen2.5-VL

# Same suite as overnight_general_sweep.sh: 7 high-res + 7 general, no MMMU
# (multi-image) and no VQAv2 (val too large).
if [[ -n "${TASKS_OVERRIDE:-}" ]]; then
    read -r -a TASKS <<< "$TASKS_OVERRIDE"
else
    TASKS=(
        docvqa_val chartqa gqa textvqa_val vstar_bench hrbench4k hrbench8k
        scienceqa_img vizwiz_vqa_val pope mme mmbench_en_dev mmbench_cn_dev seedbench
    )
fi

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
NPROC="${NPROC:-8}"
PORT="${PORT:-30300}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT_DIR="${OUT_ROOT_DIR:-$REPO_DIR/_test_outputs}"
LMMS_EVAL_DIR="${LMMS_EVAL_DIR:-$HOME/lmms-eval}"

if [[ ! -d "$CKPT" ]]; then
    echo "[ERROR] CKPT dir not found: $CKPT" >&2; exit 1
fi
if [[ ! -f "$CKPT/config.json" ]]; then
    echo "[ERROR] $CKPT lacks config.json (not an HF ckpt)" >&2; exit 1
fi

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-fastvlm}"
if [[ ! -d "$LMMS_EVAL_DIR" ]]; then
    echo "[ERROR] LMMS_EVAL_DIR not found: $LMMS_EVAL_DIR" >&2; exit 1
fi
cd "$LMMS_EVAL_DIR"

MODEL="${MODEL:-qwen3_vl}"
if ! python -c "import lmms_eval.models as m, sys; sys.exit(0 if '$MODEL' in m.AVAILABLE_SIMPLE_MODELS else 1)"; then
    echo "[ERROR] $MODEL not registered in $LMMS_EVAL_DIR — git pull the fork first" >&2; exit 1
fi

# HF_HOME left over from a data-download session points at a cache without the
# auth token, which makes token-gated task loaders (docvqa) die with
# LocalTokenNotFoundError. Force the default cache.
unset HF_HOME
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DOWNLOAD_TIMEOUT=1200
export NUMEXPR_MAX_THREADS=64
export CUDA_VISIBLE_DEVICES="$GPUS"
unset TRANSFORMERS_OFFLINE HF_HUB_OFFLINE HF_DATASETS_OFFLINE 2>/dev/null || true

# MMBench's answer extraction calls a GPT endpoint that this cluster can't
# reach; the fork falls back to local matching when this is set.
export MMBENCH_SKIP_GPT_EVAL=1

export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/${USER}_triton_cache}"
mkdir -p "$TRITON_CACHE_DIR"

MARGS="pretrained=${CKPT},attn_implementation=sdpa,max_pixels=${PX},min_pixels=${MIN_PIXELS}"
if [[ -n "${MARGS_EXTRA:-}" ]]; then
    MARGS="${MARGS},${MARGS_EXTRA}"
fi

echo "############################################################"
echo "# $MODEL single-point   ckpt=$CKPT  tag=$TAG"
echo "# tokens=$TOK  ->  max_pixels=$PX  (1024 px/token)"
echo "# tasks: ${TASKS[*]}"
echo "# start: $(date)"
echo "############################################################"

for t in "${TASKS[@]}"; do
    out="$OUT_ROOT_DIR/_sweep_${t}_${TAG}/base_tok${TOK}"
    if [[ -f "$out/done" ]]; then echo "[skip] $t already done"; continue; fi

    echo; echo "########## $t  tok${TOK}  $(date +%H:%M:%S) ##########"
    if accelerate launch --num_processes "$NPROC" --main_process_port $((PORT++)) -m lmms_eval \
        --model "$MODEL" \
        --model_args "$MARGS" \
        --tasks "$t" \
        --batch_size 1 \
        --log_samples \
        --output_path "$out"; then
        touch "$out/done"
    else
        echo "[FAIL] $t exited non-zero — continuing to next task" >&2
    fi
done

# ---- summary: one row per benchmark ----------------------------------------
python3 - << EOF
import json, glob, os
OUT_ROOT_DIR, TAG, TOK = "$OUT_ROOT_DIR", "$TAG", "$TOK"
tasks = "${TASKS[*]}".split()
print(f"\n==== {TAG} @ ~{TOK} LLM visual tokens ====")
print(f'{"task":>18} | {"metric":>28} | {"value":>8}')
print("-" * 62)
for t in tasks:
    d = f"{OUT_ROOT_DIR}/_sweep_{t}_{TAG}/base_tok{TOK}"
    files = glob.glob(f"{d}/**/*_results.json", recursive=True)
    if not files:
        print(f"{t:>18} | {'(no results)':>28} | {'—':>8}")
        continue
    try:
        res = json.load(open(files[0]))["results"].get(t, {})
    except Exception as e:
        print(f"{t:>18} | {'(parse error)':>28} | {'—':>8}")
        continue
    metrics = {k: v for k, v in res.items()
               if k.endswith(",none") and isinstance(v, (int, float))}
    if not metrics:
        print(f"{t:>18} | {'(no numeric metric)':>28} | {'—':>8}")
    for k in sorted(metrics):
        v = metrics[k]
        shown = v * 100 if v <= 1.0 else v
        print(f"{t:>18} | {k.replace(',none',''):>28} | {shown:>8.2f}")
EOF

echo; echo "############################################################"
echo "# ALL DONE: $(date)"
echo "# results under: $OUT_ROOT_DIR/_sweep_<task>_${TAG}/base_tok${TOK}/"
echo "############################################################"
