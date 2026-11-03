#!/bin/bash
# SA-1B small-object QA -> stage 1 (frozen LM, learned DAT density) -> stage 2 (LoRA + density, joint)
# and its control (LoRA + analytic DAT, same data / steps) -> V* / HR-4K / HR-8K evals. 8 GPUs, sequential;
# every step is skipped when its output exists, so the chain can be relaunched after a failure.
# Run from the repo root with the conda env active; progress in $L/chain.log.
set -u
E=${E:-$HOME/vldat_experiments/0924_miniE1_2k}
OSS_DATA=${OSS_DATA:-/data/oss_bucket_0/wangziyi/models_data}
X=${X:-/data/oss_bucket_0/wangziyi/vldat_experiments/sa1b_ldat}
L=${L:-/tmp/sa1b_ldat}
CAP=${CAP:-$OSS_DATA/sft_data/llava_sa1b_caption_pretrain.json}
N_IMAGES=${N_IMAGES:-40000}
S1_STEPS=${S1_STEPS:-2000}
S2_STEPS=${S2_STEPS:-2000}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
mkdir -p "$X/data" "$L"
say() { echo "$(date '+%F %T') $*" | tee -a "$L/chain.log"; }
QA=$X/data/sa1b_small_qa.json

[ -f "$CAP" ] || { say "caption json not found: $CAP (set CAP=...)"; exit 1; }

# 0. data: smoke on 100 images, then 8 shards, then merge
if [ ! -f "$QA" ]; then
  say "data smoke"
  CUDA_VISIBLE_DEVICES=0 python scripts/build_sa1b_small_qa.py --caption_json "$CAP" --n_images 100 \
    --out_dir "$L/smoke" > "$L/data_smoke.log" 2>&1
  if grep -q Traceback "$L/data_smoke.log"; then say "data smoke FAILED, see $L/data_smoke.log"; exit 1; fi
  say "data smoke: $(tail -n 1 "$L/data_smoke.log")"
  say "data build ($N_IMAGES images)"
  for i in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$i python scripts/build_sa1b_small_qa.py --caption_json "$CAP" --n_images $N_IMAGES \
      --shard $i --shards 8 --out_dir "$X/data/shards" > "$L/data_$i.log" 2>&1 &
  done
  wait
  python scripts/build_sa1b_small_qa.py --merge --out_dir "$X/data/shards" --out "$QA" > "$L/data_merge.log" 2>&1
  say "data: $(cat "$L/data_merge.log")"
fi

MODEL="--model_path $E/checkpoint-1000 --where_head_pt $E/loc/where_lm8_ce20.pt"
WIN="--n_win 1 --crop_tok 400 --pe global --ldat_in native --ldat_d 256 --ldat_win dat --src_scale 4 \
  --dat_alpha 0.8 --dat_ratio 4"
TRAIN="--image_root $HOME/sft_data/train_split --held_from_train 2000 --held_eval_n 400 --max_train 60000 \
  --ans_w 1 --box_w 0.1 --accum 4 --eval_every 250 --save_best"
LORA="--lora_r 16 --lora_alpha 32 --lora_lr 1e-4"
MIX="$QA,$OSS_DATA/extra_0916/viscot_bbox.train.json"

evals() {   # name ckpt "extra flags"
  say "eval $1"
  for i in 0 1 2 3 4 5 6 7; do
    ( export HF_DATASETS_CACHE=/tmp/hfds_${USER}_$i
      for d in vstar hrbench4k hrbench8k; do
        [ -f "$X/eval/$1_${d}_$i.json" ] && continue
        CUDA_VISIBLE_DEVICES=$i python scripts/offset_feat_train.py $MODEL $WIN $3 --load "$2" \
          --eval_dataset $d --max_samples 1000 --shard $i --shards 8 \
          --out "$X/eval/$1_${d}_$i.json" > "$L/eval_$1_${d}_$i.log" 2>&1
      done ) &
  done
  wait
}
mkdir -p "$X/eval"

# 1. stage 1: LM frozen, density residual on top of the analytic DAT, SA-1B QA only
S1=$X/stage1_ldat.pt
if [ ! -f "$S1" ]; then
  say "stage 1"
  torchrun --nproc_per_node 8 --master_port 29671 scripts/offset_feat_train.py $MODEL $WIN $TRAIN \
    --train_json "$QA" --cfgs crop,dat,ldat --train_cfg ldat --lr 1e-4 --tr_w 0.1 --steps $S1_STEPS \
    --save "$S1" > "$L/stage1.log" 2>&1
  cp "$L/stage1.log" "$X/"
fi
[ -f "$S1" ] || { say "stage 1 FAILED, see $L/stage1.log"; exit 1; }
evals stage1 "$S1" "--cfgs crop,dat,ldat --train_cfg ldat"

# 2. stage 2: LoRA + density from stage 1 (density lr 10x lower), SA-1B QA + VisCoT
S2=$X/stage2_ldat_lora.pt
if [ ! -f "$S2" ]; then
  say "stage 2 (ldat + LoRA)"
  torchrun --nproc_per_node 8 --master_port 29672 scripts/offset_feat_train.py $MODEL $WIN $TRAIN $LORA \
    --train_json "$MIX" --cfgs dat,ldat --train_cfg ldat --load "$S1" --lr 1e-5 --tr_w 0.1 --steps $S2_STEPS \
    --save "$S2" > "$L/stage2.log" 2>&1
  cp "$L/stage2.log" "$X/"
fi

# 3. control: LoRA + analytic DAT, same data / steps / LoRA
C2=$X/stage2_dat_lora.pt
if [ ! -f "$C2" ]; then
  say "control (analytic DAT + LoRA)"
  torchrun --nproc_per_node 8 --master_port 29673 scripts/offset_feat_train.py $MODEL $WIN $TRAIN $LORA \
    --train_json "$MIX" --cfgs dat --train_cfg dat --steps $S2_STEPS \
    --save "$C2" > "$L/control.log" 2>&1
  cp "$L/control.log" "$X/"
fi

[ -f "$S2" ] && evals stage2 "$S2" "$LORA --cfgs dat,ldat --train_cfg ldat"
[ -f "$C2" ] && evals control "$C2" "$LORA --cfgs dat --train_cfg dat"
say "all done"
