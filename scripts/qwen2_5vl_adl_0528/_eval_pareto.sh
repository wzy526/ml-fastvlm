#!/usr/bin/env bash
# Pareto-frontier evaluation: sweep base across multiple LR resolutions,
# then put DAT as a single point on the curve.
#
# Usage:
#   bash _eval_pareto.sh <DAT_CKPT> <TAG>
# Example:
#   bash _eval_pareto.sh \
#     /root/autodl-tmp/vldat_experiments/0528_expJ_sft_from_fixinit_unfreeze_mlp-merged \
#     0528_expJ
#
# Output: per-config results.json under
#   /root/autodl-tmp/ml-fastvlm/_test_outputs/_pareto_<TAG>/<config>/

set -euo pipefail

DAT_CKPT="${1:?usage: $0 DAT_CKPT TAG}"
TAG="${2:?usage: $0 DAT_CKPT TAG}"

BASE_CKPT=/root/autodl-tmp/models_data/Qwen2.5-VL-3B-Instruct
OUT_ROOT=/root/autodl-tmp/ml-fastvlm/_test_outputs/_pareto_${TAG}
TASKS="${TASKS:-ocrbench,hrbench4k,hrbench8k,vstar_bench}"

# Resolutions to sweep (LR max_pixels, in units of pixels).
# 200704 = 256*28*28 = ~256 LLM tokens (smallest meaningful Qwen2.5-VL setting)
# 501760 = ~640 tokens (matches our DAT's lr_max_pixels)
# 1003520 = ~1280 tokens (~2x DAT LR)
# 2007040 = ~2560 tokens
# 5017600 = ~6400 tokens (~ DAT's hr_max_pixels; high-res base reference)
# 9031680 = ~11520 tokens (Qwen2.5-VL default; max realistic base)
PIXEL_SWEEP=(200704 501760 1003520 2007040 5017600 9031680)

cd /root/autodl-tmp/lmms-eval

PORT=30100

# ---- base sweep ----------------------------------------------------------
for px in "${PIXEL_SWEEP[@]}"; do
    tag="base_px${px}"
    out="$OUT_ROOT/$tag"
    if [[ -f "$out/done" ]]; then
        echo "[skip] $tag already done"; continue
    fi
    echo "=================================================================="
    echo "[base] max_pixels=$px  (~$((px/784)) LLM tokens)"
    echo "=================================================================="
    HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    accelerate launch --num_processes 8 --main_process_port $((PORT++)) -m lmms_eval \
        --model qwen2_5_vl \
        --model_args "pretrained=${BASE_CKPT},attn_implementation=sdpa,min_pixels=200704,max_pixels=${px}" \
        --tasks "$TASKS" \
        --batch_size 1 \
        --log_samples \
        --output_path "$out"
    touch "$out/done"
done

# ---- DAT single point ----------------------------------------------------
# DAT's LLM token count is determined by lr_max_pixels (default 501760 in
# the qwen2_5_dat_vl wrapper). hr_max_pixels controls the HR ViT cost but
# does NOT enter the LLM seq.
tag="dat_default"
out="$OUT_ROOT/$tag"
if [[ ! -f "$out/done" ]]; then
    echo "=================================================================="
    echo "[DAT] ${DAT_CKPT}  (lr_max=501760 → ~640 LLM tokens, hr cross-attn)"
    echo "=================================================================="
    HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    accelerate launch --num_processes 8 --main_process_port $((PORT++)) -m lmms_eval \
        --model qwen2_5_dat_vl \
        --model_args "pretrained=${DAT_CKPT},attn_implementation=sdpa" \
        --tasks "$TASKS" \
        --batch_size 1 \
        --log_samples \
        --output_path "$out"
    touch "$out/done"
fi

# ---- summary table -------------------------------------------------------
python3 - << EOF
import json, glob, os
OUT_ROOT = "$OUT_ROOT"
TASKS = "$TASKS".split(",")

rows = []
for d in sorted(glob.glob(f"{OUT_ROOT}/*/")):
    tag = os.path.basename(d.rstrip("/"))
    for f in glob.glob(f"{d}**/*_results.json", recursive=True):
        try:
            r = json.load(open(f))['results']
            row = {"config": tag}
            for t in TASKS:
                v = r.get(t, {})
                if t.startswith("ocrbench"):
                    row[t] = v.get("ocrbench_accuracy,none")
                elif t.startswith("hrbench"):
                    row[t+"_avg"]    = v.get("average,none")
                    row[t+"_single"] = v.get("single,none")
                    row[t+"_cross"]  = v.get("cross,none")
                elif t == "vstar_bench":
                    row[t] = v.get("vstar_overall_acc,none")
            rows.append(row)
            break
        except: pass

if not rows: print("no results yet"); raise SystemExit
keys = sorted({k for r in rows for k in r if k != "config"})
hdr = f'{"config":<22} | ' + " ".join(f"{k[:14]:>14}" for k in keys)
print(hdr); print("-"*len(hdr))
for r in rows:
    line = f'{r["config"]:<22} | '
    for k in keys:
        v = r.get(k)
        line += f"{v:>14.4f} " if isinstance(v,(int,float)) else f"{'—':>14} "
    print(line)
EOF
