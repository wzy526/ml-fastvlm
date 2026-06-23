#!/usr/bin/env bash
# Generic LLM-visual-token sweep for lmms-eval (DAT or base Qwen2.5-VL).
#
# Loops one benchmark across several pixel budgets to trace the
# accuracy-vs-LLM-tokens curve. Swap model / ckpt / task / pixel list from the
# command line — nothing is hardcoded except sane defaults.
#
#   Token axis:
#     base : LLM visual tokens are set by `max_pixels`.
#     dat  : LLM visual tokens are set by `lr_max_pixels` (LR-first default);
#            `max_pixels` only caps the HR cross-attn resolution (0 LLM tokens).
#     LLM visual tokens per image ≈ pixels / 784.
#
# Usage:
#   bash eval_pixel_sweep.sh <dat|base> <CKPT> <TASK> [TAG]
#
# Examples:
#   bash eval_pixel_sweep.sh dat  /root/autodl-tmp/vldat_experiments/0606_sft_dirA_nogate_full_12dat  docvqa_val  0606_12dat
#   bash eval_pixel_sweep.sh base /root/autodl-tmp/models_data/Qwen2.5-VL-3B-Instruct                docvqa_val  base3b
#   PIXELS="200704 501760 1003520" bash eval_pixel_sweep.sh dat <CKPT> ocrbench dat_ocr
#
# Env knobs (all optional):
#   PIXELS      space-separated pixel budgets   (default: 200704 501760 1003520 2007040 5017600 9031680
#               — same grid as _eval_pareto.sh; ~256/640/1280/2560/6400/11520 LLM tokens)
#   HR_CAP      DAT HR pixel cap                 (default: 5017600, matches training hd_max)
#   HR_SCALE    DAT hr_scale                     (default: 3)
#   MIN_PIXELS  lower pixel band                 (default: 28224)
#   GPUS        CUDA_VISIBLE_DEVICES             (default: 0,1,2,3,4,5,6,7)
#   NPROC       accelerate num_processes         (default: 8)
#   PORT        starting main_process_port       (default: 30200, +1 per point)
#   OUT_ROOT    output dir                       (default: /root/autodl-tmp/ml-fastvlm/_test_outputs/_sweep_<TASK>_<TAG>)
#   BASE_REF    base model to source preprocessor_config.json from when a DAT ckpt lacks it
#               (default: /root/autodl-tmp/models_data/Qwen2.5-VL-3B-Instruct)

set -euo pipefail

MODEL_TYPE="${1:?usage: $0 <dat|base> CKPT TASK [TAG]}"
CKPT="${2:?usage: $0 <dat|base> CKPT TASK [TAG]}"
TASK="${3:?usage: $0 <dat|base> CKPT TASK [TAG]}"
TAG="${4:-$(basename "$CKPT")}"

case "$MODEL_TYPE" in
    dat)  MODEL=qwen2_5_dat_vl ;;
    base) MODEL=qwen2_5_vl ;;
    *) echo "[ERROR] MODEL_TYPE must be 'dat' or 'base', got '$MODEL_TYPE'" >&2; exit 1 ;;
esac

# Default grid matches scripts/qwen2_5vl_adl_0528/_eval_pareto.sh
# (200704≈256 tok … 9031680≈11520 tok, the Qwen2.5-VL default ceiling).
PIXELS="${PIXELS:-200704 501760 1003520 2007040 5017600 9031680}"
HR_CAP="${HR_CAP:-5017600}"
HR_SCALE="${HR_SCALE:-3}"
MIN_PIXELS="${MIN_PIXELS:-28224}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
NPROC="${NPROC:-8}"
PORT="${PORT:-30200}"
OUT_ROOT="${OUT_ROOT:-/root/autodl-tmp/ml-fastvlm/_test_outputs/_sweep_${TASK}_${TAG}}"
BASE_REF="${BASE_REF:-/root/autodl-tmp/models_data/Qwen2.5-VL-3B-Instruct}"

eval "$(conda shell.bash hook)"
conda activate vldat
cd /root/autodl-tmp/lmms-eval

# DAT ckpts often ship `processor_config.json` but not `preprocessor_config.json`
# (the image-processor config). Without it the wrapper falls back to a HF repo id
# and dies offline. Source it from the local base model once.
if [[ "$MODEL_TYPE" == "dat" && ! -f "$CKPT/preprocessor_config.json" ]]; then
    if [[ -f "$BASE_REF/preprocessor_config.json" ]]; then
        echo "[fix] copying preprocessor_config.json from $BASE_REF into $CKPT"
        cp "$BASE_REF/preprocessor_config.json" "$CKPT/"
    else
        echo "[WARN] $CKPT lacks preprocessor_config.json and BASE_REF has none either" >&2
    fi
fi

# This box can't reach huggingface.co; use the mirror for dataset downloads.
# Keep TRANSFORMERS_OFFLINE so the (local) model never touches the network.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DOWNLOAD_TIMEOUT=1200
export NUMEXPR_MAX_THREADS=64
export CUDA_VISIBLE_DEVICES="$GPUS"

echo "=================================================================="
echo " model=$MODEL  ckpt=$CKPT"
echo " task=$TASK  pixels=[$PIXELS]"
echo " out=$OUT_ROOT"
echo "=================================================================="

for px in $PIXELS; do
    tok=$((px / 784))
    tag="${MODEL_TYPE}_tok${tok}"
    out="$OUT_ROOT/$tag"
    if [[ -f "$out/done" ]]; then echo "[skip] $tag already done"; continue; fi

    if [[ "$MODEL_TYPE" == "dat" ]]; then
        # Sweep LR (LLM tokens); HR auto = LR*hr_scale^2 capped at HR_CAP.
        margs="pretrained=${CKPT},attn_implementation=sdpa,hr_scale=${HR_SCALE},max_pixels=${HR_CAP},min_pixels=${MIN_PIXELS},lr_max_pixels=${px},lr_min_pixels=${MIN_PIXELS}"
    else
        margs="pretrained=${CKPT},attn_implementation=sdpa,max_pixels=${px},min_pixels=${MIN_PIXELS}"
    fi

    echo "------------------------------------------------------------------"
    echo "[$tag] $TASK  px=$px  (~$tok LLM tokens)"
    echo "------------------------------------------------------------------"
    accelerate launch --num_processes "$NPROC" --main_process_port $((PORT++)) -m lmms_eval \
        --model "$MODEL" \
        --model_args "$margs" \
        --tasks "$TASK" \
        --batch_size 1 \
        --log_samples \
        --output_path "$out"
    touch "$out/done"
done

# ---- summary: dump every numeric metric for $TASK, sorted by token count ----
python3 - << EOF
import json, glob, os, re
OUT_ROOT, TASK = "$OUT_ROOT", "$TASK"
rows = []
for d in sorted(glob.glob(f"{OUT_ROOT}/*/")):
    tag = os.path.basename(d.rstrip("/"))
    m = re.search(r"tok(\d+)", tag)
    tok = int(m.group(1)) if m else -1
    for f in glob.glob(f"{d}**/*_results.json", recursive=True):
        try:
            res = json.load(open(f))["results"].get(TASK, {})
            metrics = {k: v for k, v in res.items()
                       if k.endswith(",none") and isinstance(v, (int, float))}
            rows.append((tok, tag, metrics)); break
        except Exception:
            pass
if not rows:
    print("no results yet"); raise SystemExit
rows.sort()
keys = sorted({k for _, _, m in rows for k in m})
print(f"\n==== {TASK} : metric vs ~LLM tokens ====")
hdr = f'{"~tokens":>8} | ' + " ".join(f"{k.replace(',none',''):>16}" for k in keys)
print(hdr); print("-" * len(hdr))
for tok, tag, m in rows:
    line = f"{tok:>8} | " + " ".join(
        (f"{m[k]*100:>16.2f}" if (k in m and m[k] <= 1.0) else
         (f"{m[k]:>16.2f}" if k in m else f'{"—":>16}')) for k in keys)
    print(line)
EOF
