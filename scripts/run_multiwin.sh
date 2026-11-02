#!/bin/bash
# 1) accuracy: crop vs analytic DAT with K windows of 10 x 10 tokens each (K = 1 2 4 8 16), V* / HR-4K / HR-8K;
# 2) ldat density head, box-only on grey composites: input crop-resized d256 / native d256 / native d1024.
# Waits for any running offset_feat_train.py first. Run from the repo root with the conda env active.
set -u
E=${E:-$HOME/vldat_experiments/0924_miniE1_2k}
OSS_DATA=${OSS_DATA:-/data/oss_bucket_0/wangziyi/models_data}
X=/data/oss_bucket_0/wangziyi/vldat_experiments
O=$X/multiwin
L=${L:-/tmp/multiwin}
mkdir -p "$O" "$L"
say() { echo "$(date '+%F %T') $*" >> "$L/chain.log"; }

while pgrep -f offset_feat_train.py > /dev/null; do sleep 60; done

MODEL="--model_path $E/checkpoint-1000 --where_head_pt $E/loc/where_lm8_ce20.pt"

for K in ${KS:-1 2 4 8 16}; do
  say "eval w$K"
  for i in 0 1 2 3 4 5 6 7; do
    ( export HF_DATASETS_CACHE=/tmp/hfds_${USER}_$i
      for d in hrbench4k hrbench8k vstar; do
        CUDA_VISIBLE_DEVICES=$i python scripts/offset_feat_train.py $MODEL --n_win $K --grid 10 --pe global \
          --cfgs crop,dat --train_cfg crop --eval_dataset $d --max_samples 1000 --shard $i --shards 8 \
          --out $O/w${K}_${d}_$i.json > $L/w${K}_${d}_$i.log 2>&1
      done ) &
  done
  wait
done

BOX="$MODEL --train_json $OSS_DATA/extra_0916/viscot_bbox.train.json --held_from_train 3000 --held_eval_n 400 \
  --image_root $HOME/sft_data/train_split --n_win 1 --crop_tok 400 --pe global \
  --cfgs crop,dat,ldat --train_cfg ldat --ldat_win dat --ans_w 0 --box_w 1 --accum 4 --lr 1e-3 \
  --dat_alpha 0.8 --dat_ratio 4 --src_scale 4 --composite 3840 --comp_bg grey --steps 1500 --eval_every 500"
for v in "crop 256" "native 256" "native 1024"; do
  set -- $v
  say "box ldat_in=$1 d=$2"
  torchrun --nproc_per_node 8 --master_port 29651 scripts/offset_feat_train.py $BOX --ldat_in $1 --ldat_d $2 \
    --save $O/ldatbox_$1_d$2.pt > $L/ldatbox_$1_d$2.log 2>&1
done
say "all done"
