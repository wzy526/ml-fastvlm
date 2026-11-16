#!/usr/bin/env bash
# Q-LAS ablations on the 2B pipeline of the main table, plus the 4B density head.
#
# Every row is V* / HR-Bench 4K / 8K through lmms-eval with the old V* prompt
# (LMMS_EVAL_DIR=~/lmms-eval-oldvstar): LR 640 + one 400-token window, where-head
# layer 8, trained density head, unless the row says otherwise. Finished rows and
# heads are skipped, so a group can be relaunched after a failure.
#
#   bash scripts/run_ablations.sh component native tokens density4b loss layer
#
# Groups:
#   component  crop (fixed / adaptive window), + analytic warp, + learned warp (Q-LAS),
#              whole-image uniform / analytic / learned warp            eval only
#   native     SFT checkpoint without window at 640..4096 LR tokens     eval only
#   tokens     Q-LAS at LR 320 / 1280 / 2560 and window 100 / 196 / 900 eval only
#   density4b  density head for 4B (2B recipe), eval at LR 640 and 1280 ~4 h train
#   loss       density head without hinge / box KL / trust region / answer CE  4 x ~2.5 h
#   layer      where-head at layers 4 / 8 / 12 / 16 / last, analytic warp,
#              plus first-pass latency per layer                         5 x ~4 h
#
# Rows: $A/<row>/<task>_lr<N>/   heads: $A/heads/   logs: $L/   chain: $L/chain.log
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CONDA_SH=${CONDA_SH:-$HOME/miniconda3/bin/activate}
export CONDA_ENV=${CONDA_ENV:-$HOME/miniconda3/envs/fastvlm}
# shellcheck disable=SC1090
source "$CONDA_SH" "$CONDA_ENV"

E=${E:-$HOME/vldat_experiments/0924_miniE1_2k}
M4B=${M4B:-$HOME/vldat_experiments/1110_4b_viscot_lora-merged}
X=${X:-/data/oss_bucket_0/wangziyi/vldat_experiments/sa1b_ldat}
OSS_DATA=${OSS_DATA:-/data/oss_bucket_0/wangziyi/models_data}
A=${A:-$X/ablation}
L=${L:-/tmp/ablation}
PORT=${PORT:-30410}
export LMMS_EVAL_DIR=${LMMS_EVAL_DIR:-$HOME/lmms-eval-oldvstar}
export HRBENCH_SKIP_GPT_EVAL=1 PYTHONUNBUFFERED=1
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

M2=$E/checkpoint-1000
W2=$E/loc/where_lm8_ce20.pt
H2=$X/hinge_noratio_ldat.pt
Z2=$A/heads/ldat_zero_2b.pt
W4=$M4B/where_lm8_ce20.pt
TASKS3="vstar_bench hrbench4k hrbench8k"
mkdir -p "$A/heads" "$L"

say() { echo "$(date '+%F %T') $*" | tee -a "$L/chain.log"; }

for p in "$M2" "$W2" "$H2" "$M4B" "$W4" "$LMMS_EVAL_DIR" "$X/data/sa1b_small_qa.json"; do
    [ -e "$p" ] || { say "missing $p"; exit 1; }
done

# ev ROW [VAR=value ...]: one eval_ldat_pareto.sh call, 2B main settings unless overridden.
ev() {
    local row=$1
    shift
    say "eval $row $*"
    env MODEL_PATH="$M2" WHERE_PT="$W2" LDAT_PT="$H2" CROP_TOK=400 N_WIN=1 GRID=0 TOKS=640 \
        DAT_ALPHA=0.8 DAT_K=0.6 WIN_FULL=0 WHERE_LAYER=8 FAST_PREFIX=1 PREFIX_CACHE=0 SAMPLE_FEAT=0 \
        TASKS="$TASKS3" PORT="$PORT" "$@" OUT="$A/$row" \
        bash scripts/eval_ldat_pareto.sh >> "$L/$row.log" 2>&1
    PORT=$((PORT + 10))
}

# mkzero OUT MODEL: density head whose residual is zero (density = where-head).
mkzero() {
    [ -f "$1" ] && return 0
    python - "$1" "$2" <<'PY'
import sys
sys.path.insert(0, "scripts")
import torch
from transformers import AutoConfig
from offset_feat_train import LDatHead
cfg = AutoConfig.from_pretrained(sys.argv[2])
hidden = getattr(cfg, "text_config", cfg).hidden_size
torch.save(LDatHead(hidden, d=256).state_dict(), sys.argv[1])
print("[zero]", sys.argv[1], "hidden", hidden)
PY
}

# The recipe of hinge_noratio_ldat.pt; the loss weights come separately.
train_base() {   # train_base MODEL WHERE
    echo "--model_path $1 --where_head_pt $2 --n_win 1 --crop_tok 400 --pe global --ldat_in native" \
         "--ldat_d 256 --ldat_win dat --src_scale 4 --dat_alpha 0.8 --dat_ratio 0" \
         "--image_root $HOME/sft_data/train_split --held_from_train 2000 --held_eval_n 400" \
         "--max_train 60000 --accum 4 --eval_every 250 --save_best --steps 2000" \
         "--train_json $X/data/sa1b_small_qa.json --cfgs crop,dat,ldat --train_cfg ldat --lr 1e-4"
}
FULL_LOSS="--ans_w 1 --box_w 0.1 --tr_w 0.1 --hinge_w 1"

train_ldat() {   # train_ldat NAME "base flags" "loss flags"
    local out=$A/heads/$1.pt
    [ -f "$out" ] && return 0
    say "train $1: $3"
    # shellcheck disable=SC2086
    torchrun --nproc_per_node 8 --master_port "$PORT" scripts/offset_feat_train.py $2 $3 \
        --save "$out" > "$L/train_$1.log" 2>&1
    PORT=$((PORT + 10))
    [ -f "$out" ] || { say "train $1 FAILED, see $L/train_$1.log"; return 1; }
}

component() {
    mkzero "$Z2" "$M2"
    ev comp_crop_fixed DAT_ALPHA=0 DAT_K=0
    ev comp_crop_adapt DAT_ALPHA=0
    ev comp_dat_zero LDAT_PT="$Z2"
    ev comp_qlas
    ev comp_full_uniform WIN_FULL=1 DAT_ALPHA=0
    ev comp_full_dat WIN_FULL=1 LDAT_PT="$Z2"
    ev comp_full_qlas WIN_FULL=1
}

native() {
    local px="" t
    for t in ${NATIVE_TOKS:-640 720 1040 1680 2960 4096}; do
        px+="$((t * 1024)) "
    done
    say "native SFT sweep: ${NATIVE_TOKS:-640 720 1040 1680 2960 4096}"
    PIXELS="${px% }" OUT_ROOT="$A/native_sft" PORT="$PORT" \
        bash scripts/eval_pixel_sweep.sh base35 "$M2" "vstar_bench,hrbench4k,hrbench8k" native_sft \
        >> "$L/native_sft.log" 2>&1
    PORT=$((PORT + 10))
}

tokens() {
    ev tok_lr TOKS="320 1280 2560"
    local c
    for c in 100 196 900; do
        ev "tok_win$c" CROP_TOK=$c
    done
}

density4b() {
    train_ldat ldat4b_hinge_noratio "$(train_base "$M4B" "$W4")" "$FULL_LOSS" || return 1
    ev d4b_qlas MODEL_PATH="$M4B" WHERE_PT="$W4" LDAT_PT="$A/heads/ldat4b_hinge_noratio.pt" TOKS="640 1280"
}

loss() {
    local v n f
    for v in "no_hinge:--ans_w 1 --box_w 0.1 --tr_w 0.1 --hinge_w 0" \
             "no_box:--ans_w 1 --box_w 0 --tr_w 0.1 --hinge_w 1" \
             "no_tr:--ans_w 1 --box_w 0.1 --tr_w 0 --hinge_w 1" \
             "no_ans:--ans_w 0 --box_w 0.1 --tr_w 0.1 --hinge_w 1"; do
        n=${v%%:*}
        f=${v#*:}
        train_ldat "loss_$n" "$(train_base "$M2" "$W2")" "$f" && ev "loss_$n" LDAT_PT="$A/heads/loss_$n.pt"
    done
}

layer() {
    mkzero "$Z2" "$M2"
    local last l w
    last=$(python -c "import json; c = json.load(open('$M2/config.json')); print(c.get('text_config', c)['num_hidden_layers'] - 1)")
    for l in ${LAYERS:-4 8 12 16 $last}; do
        w=$A/heads/where_lm$l.pt
        if [ ! -f "$w" ]; then
            say "where-head layer $l"
            torchrun --nproc_per_node 8 --master_port "$PORT" scripts/where_head.py --model_path "$M2" \
                --train_json "$OSS_DATA/extra_0916/viscot_bbox.train.json" \
                             "$OSS_DATA/extra_1018/refcoco_bbox.json" \
                             "$OSS_DATA/extra_1018/coco_small_bbox.json" \
                --max_per_json 40000 40000 60000 --accum 4 --steps 1500 --eval_every 500 \
                --head xattn --d 384 --lr 1e-4 --ce_weight 20 --dilate 0 --eval_query question \
                --feat "lm:$l" --save "$w" --out_boxes "$A/heads/where_lm${l}_vstar.json" \
                > "$L/where_lm$l.log" 2>&1
            PORT=$((PORT + 10))
            [ -f "$w" ] || { say "where-head layer $l FAILED, see $L/where_lm$l.log"; continue; }
        fi
        ev "layer_$l" WHERE_PT="$w" WHERE_LAYER="$l" LDAT_PT="$Z2"
        if ! grep -qs "\"name\": \"layer $l\"" "$A/layer_cost.jsonl"; then
            say "latency layer $l"
            CUDA_VISIBLE_DEVICES=0 python -u scripts/bench_cost.py --name "layer $l" --model ldat \
                --model_args "pretrained=$M2,where_head_pt=$w,ldat_pt=$Z2,where_layer=$l,fast_prefix=True" \
                --datasets vstar --max_samples 50 --out "$A/layer_cost.jsonl" > "$L/cost_layer$l.log" 2>&1
        fi
    done
}

[ $# -gt 0 ] || { echo "usage: $0 component|native|tokens|density4b|loss|layer ..."; exit 1; }
for g in "$@"; do
    case "$g" in
        component|native|tokens|density4b|loss|layer) say "== $g"; "$g" ;;
        *) say "unknown group $g"; exit 1 ;;
    esac
done
say "all done: $*"
