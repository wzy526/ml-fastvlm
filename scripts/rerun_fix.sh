#!/bin/bash
# Re-run every multi-GPU offset_feat_train.py training that predates the per-rank-init fix (52d1c3d), one after
# another on 8 GPUs: ldat grey + its HR-Bench eval, ldat plain, LoRA3 (net fzero crop), LoRA+DAT, and the six
# early feature-offset heads. Run from the repo root with the conda env active; progress in $L/chain.log.
set -u
E=${E:-$HOME/vldat_experiments/0924_miniE1_2k}
OSS_DATA=${OSS_DATA:-/data/oss_bucket_0/wangziyi/models_data}
export E OSS_DATA
X=/data/oss_bucket_0/wangziyi/vldat_experiments
L=${L:-/tmp/rerun_fix}
mkdir -p "$L" "$X/ldat_eval_fix" "$X/offset_fix"
say() { echo "$(date '+%F %T') $*" >> "$L/chain.log"; }

while pgrep -f offset_feat_train.py > /dev/null; do sleep 60; done     # e.g. a grey run started by hand

MODEL="--model_path $E/checkpoint-1000 --where_head_pt $E/loc/where_lm8_ce20.pt"
DATA="--train_json $OSS_DATA/extra_0916/viscot_bbox.train.json --held_from_train 3000 --image_root $HOME/sft_data/train_split"
LDAT="$MODEL $DATA --held_eval_n 400 --n_win 1 --crop_tok 400 --pe global --cfgs crop,dat,ldat --train_cfg ldat \
  --ldat_in crop --ans_w 0 --box_w 1 --accum 4 --lr 1e-3 --dat_alpha 0.8 --dat_ratio 4 --steps 1500 --eval_every 500"

# 1. ldat on grey composites (skipped if the hand-started run finished)
if ! grep -q "EVAL step 1500" /tmp/ldatfix_grey.log 2>/dev/null; then
  say "train ldat grey"
  torchrun --nproc_per_node 8 --master_port 29631 scripts/offset_feat_train.py $LDAT \
    --composite 3840 --comp_bg grey --save $X/ldat_crop_grey_fix.pt > /tmp/ldatfix_grey.log 2>&1
fi

# 2. HR-Bench / V*: crop, dat, ldat with relaxed (r4) and finalized (def) density limits
say "eval ldat grey"
WIN="--n_win 1 --crop_tok 400 --hr_source full --pe global --train_cfg ldat --ldat_in crop"
for i in 0 1 2 3 4 5 6 7; do
  ( export HF_DATASETS_CACHE=/tmp/hfds_${USER}_$i
    for d in hrbench4k hrbench8k vstar; do
      CUDA_VISIBLE_DEVICES=$i python scripts/offset_feat_train.py $MODEL $WIN --cfgs crop,dat,ldat \
        --dat_alpha 0.8 --dat_ratio 4 --load $X/ldat_crop_grey_fix.pt --eval_dataset $d --max_samples 1000 \
        --shard $i --shards 8 --out $X/ldat_eval_fix/r4_${d}_$i.json > $L/ldat_r4_${d}_$i.log 2>&1
      CUDA_VISIBLE_DEVICES=$i python scripts/offset_feat_train.py $MODEL $WIN --cfgs dat,ldat \
        --load $X/ldat_crop_grey_fix.pt --eval_dataset $d --max_samples 1000 \
        --shard $i --shards 8 --out $X/ldat_eval_fix/def_${d}_$i.json > $L/ldat_def_${d}_$i.log 2>&1
    done ) &
done
wait

# 3. ldat on plain VisCoT images
say "train ldat plain"
torchrun --nproc_per_node 8 --master_port 29632 scripts/offset_feat_train.py $LDAT \
  --save $X/ldat_crop_plain_fix.pt > $L/ldatfix_plain.log 2>&1

# 4. LoRA3 (net fzero crop) and LoRA + DAT, train + eval each
say "lora3"
O=$X/lora3_fix L=/tmp/lora3_fix bash scripts/run_lora3.sh
say "lora_dat"
CFGS=dat EVAL_CFGS=crop,dat O=$X/lora_dat_fix L=/tmp/lora_dat_fix bash scripts/run_lora3.sh

# 5. early feature-offset heads (no LoRA), same flags as the originals, held-out eval only
OFF="$MODEL $DATA --k 7 --min_win_ratio 2 --accum 4 --lr 1e-4 --steps 400 --eval_every 100"
MD="--n_win 4 --crop_tok 400 --hr_source full --pe global"
HD="--in_ln --pool before --norm in --max_off 0"
run_off() {
  say "offset $1"
  torchrun --nproc_per_node 8 --master_port 29633 scripts/offset_feat_train.py $OFF "${@:2}" \
    --save $X/offset_fix/offset_feat_$1.pt > $X/offset_fix/offset_feat_$1.log 2>&1
}
run_off base --crop_tok 100 --n_win 1
run_off w4 --crop_tok 400 --n_win 4
run_off md $MD
run_off md2 $MD $HD --d 256
run_off md3 $MD $HD --d 256 --no_conv_bias --pw_std 0.01
run_off md4 $MD $HD --d 512 --no_conv_bias --pw_std 0.01
say "all done"
