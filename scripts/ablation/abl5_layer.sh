#!/usr/bin/env bash
# Ablation 5, where-head layer: 4 / 8 / 12 / 16 / last.
#   Layer 8 is the main head; the others follow its two stages (three box sets,
#   1500 steps at 3e-4, then ce20 for 600 steps at 1e-4). ~5.5 h per layer.
#   Every layer is scored with the zero density residual, since the trained
#   density head was fit to layer 8; layer 8's row is comp_dat_zero of ablation 1.
#   Per layer: V* window hit (training log), HR-Bench 4K, first-pass latency
#   on 50 HR-Bench 4K images (the first pass stops at the where-head layer).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
DATA=("$OSS_DATA/extra_0916/viscot_bbox.train.json" "$OSS_DATA/extra_1018/refcoco_bbox.json"
      "$OSS_DATA/extra_1018/coco_small_bbox.json")
need "${DATA[@]}" "$HOME/sft_data/train_split" "$HOME/data/vstar_bench"
mkzero "$Z2" "$M2"

where_train() {   # where_train LAYER OUT
    local l=$1 out=$2 mix=$A/heads/where_lm$1_mix.pt
    if [ ! -f "$mix" ]; then
        say "train where-head layer $l, stage 1"
        torchrun --nproc_per_node 8 --master_port "$PORT" scripts/where_head.py --model_path "$M2" \
            --train_json "${DATA[@]}" --max_per_json 20000 40000 60000 --accum 4 --steps 1500 \
            --eval_every 500 --head xattn --d 384 --lr 3e-4 --eval_query question --feat "lm:$l" \
            --save "$mix" --out_boxes "$A/heads/where_lm${l}_mix.json" > "$L/where_lm${l}_mix.log" 2>&1
        PORT=$((PORT + 10))
        [ -f "$mix" ] || { say "layer $l stage 1 FAILED, see $L/where_lm${l}_mix.log"; return 1; }
    fi
    say "train where-head layer $l, stage 2"
    torchrun --nproc_per_node 8 --master_port "$PORT" scripts/where_head.py --model_path "$M2" \
        --load "$mix" --train_json "${DATA[@]}" --max_per_json 20000 40000 60000 --accum 4 --steps 600 \
        --eval_every 200 --head xattn --d 384 --lr 1e-4 --ce_weight 20 --dilate 0 --eval_query question \
        --feat "lm:$l" --save "$out" --out_boxes "$A/heads/where_lm${l}_ce20.json" \
        > "$L/where_lm${l}_ce20.log" 2>&1
    PORT=$((PORT + 10))
    [ -f "$out" ] || { say "layer $l stage 2 FAILED, see $L/where_lm${l}_ce20.log"; return 1; }
}

last=$(python -c "import json; c = json.load(open('$M2/config.json')); print(c.get('text_config', c)['num_hidden_layers'] - 1)")
for l in 4 8 12 16 "$last"; do
    if [ "$l" = 8 ]; then
        w=$W2
    else
        w=$A/heads/where_lm${l}_ce20.pt
        [ -f "$w" ] || where_train "$l" "$w" || continue
        ev "layer_$l" WHERE_PT="$w" WHERE_LAYER="$l" LDAT_PT="$Z2"
    fi
    if ! grep -qs "\"name\": \"layer $l\"" "$A/layer_cost.jsonl"; then
        say "latency layer $l"
        CUDA_VISIBLE_DEVICES=0 python -u scripts/bench_cost.py --name "layer $l" --model ldat \
            --model_args "pretrained=$M2,where_head_pt=$w,ldat_pt=$Z2,where_layer=$l,fast_prefix=True" \
            --datasets hrbench4k --max_samples 50 --out "$A/layer_cost.jsonl" > "$L/cost_layer$l.log" 2>&1
    fi
done
say "done"
