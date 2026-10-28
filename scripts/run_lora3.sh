#!/bin/bash
# LoRA + {learned offsets, zero offsets, pixel crop}: train on 8 GPUs, then eval V* / HR-4K / HR-8K, one after another.
# Run from the repo root with the conda env active. Override with CFGS="crop" STEPS=400 etc.
set -u
E=${E:-$HOME/vldat_experiments/0924_miniE1_2k}
OSS_DATA=${OSS_DATA:-/data/oss_bucket_0/wangziyi/models_data}
O=${O:-/data/oss_bucket_0/wangziyi/vldat_experiments/lora3}
L=${L:-/tmp/lora3}
mkdir -p "$O" "$L"

MODEL="--model_path $E/checkpoint-1000 --where_head_pt $E/loc/where_lm8_ce20.pt"
WIN="--n_win 1 --crop_tok 400 --hr_source full --pe global --k 7"
HEAD="--in_ln --d 256 --pool before --norm in --max_off 0 --no_conv_bias --pw_std 0.01"
LORA="--lora_r ${LORA_R:-16} --lora_alpha ${LORA_ALPHA:-32}"

for cfg in ${CFGS:-net fzero crop}; do
  ck=$O/lora_$cfg.pt
  echo "$(date '+%F %T') train $cfg" >> "$L/chain.log"
  torchrun --nproc_per_node 8 --master_port 29570 scripts/offset_feat_train.py $MODEL \
    --train_json "$OSS_DATA/extra_0916/viscot_bbox.train.json" --held_from_train 3000 --held_eval_n 1000 \
    --image_root ~/sft_data/train_split $WIN --min_win_ratio 2 $HEAD $LORA --train_cfg $cfg \
    --accum 4 --lr 1e-4 --lora_lr ${LORA_LR:-1e-4} --steps ${STEPS:-800} --eval_every 200 \
    --save "$ck" > "$L/train_$cfg.log" 2>&1
  cp "$L/train_$cfg.log" "$O/"
  if [ ! -f "$ck" ]; then
    echo "$(date '+%F %T') train $cfg FAILED, skipping its eval" >> "$L/chain.log"
    continue
  fi
  echo "$(date '+%F %T') eval $cfg" >> "$L/chain.log"
  for i in 0 1 2 3 4 5 6 7; do
    (
      export HF_DATASETS_CACHE=/tmp/hfds_${USER}_$i
      for d in vstar hrbench4k hrbench8k; do
        CUDA_VISIBLE_DEVICES=$i python scripts/offset_feat_train.py $MODEL $WIN $HEAD $LORA \
          --eval_dataset $d --load "$ck" --max_samples 1000 --shard $i --shards 8 \
          --out "$O/${cfg}_${d}_$i.json" > "$L/eval_${cfg}_${d}_$i.log" 2>&1
      done
    ) &
  done
  wait
  tar -czf "$O/eval_logs_$cfg.tgz" -C "$L" $(cd "$L" && ls eval_${cfg}_*.log)
  echo "$(date '+%F %T') done $cfg" >> "$L/chain.log"
done
echo "$(date '+%F %T') all done" >> "$L/chain.log"
