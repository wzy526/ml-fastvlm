#!/usr/bin/env bash
cd /root/ml-fastvlm

if ! command -v pdsh >/dev/null 2>&1; then
  cd /home/zhuofan.xia/pdsh-2.29
  ./configure --with-ssh --with-rsh --with-mrsh --with-mqshell --with-dshgroups --with-machines=/etc/pdsh/machines
  make && sudo make install
  cd -
fi

if ! command -v ifconfig >/dev/null 2>&1; then
  sudo apt update && sudo apt install net-tools
fi
MASTER_ADDR=`ifconfig -a|grep inet|grep -v 127.0.0.1|grep -v inet6|awk '{print $2}'|tr -d "addr:"`
MASTER_PORT=25001

export TRANSFORMERS_OFFLINE=1
export DS_SKIP_CUDA_CHECK=1
source /root/miniconda3/bin/activate fastvlm
TOTAL_STEPS=2595
EXP_NAME="tdat-7b-l0d32-s12g8z3_ep2"

mkdir -p /data/checkpoints/$EXP_NAME

WANDB_PROJECT="DECOAT" \
ds --master_port=$MASTER_PORT --master_addr=$MASTER_ADDR --hostfile "/etc/volcano/all.host" llava/train/train_dat.py \
    --deepspeed ./scripts/zero_configs/zero2.json \
    --model_name_or_path /data/gsva_pretrains/llava-v1_5-7b-hf \
    --version v1 \
    --extra_yaml_path ./configs/llava1_5_v1.yaml \
    --data_path /data/llava_v1_5_mix665k.json \
    --image_folder /data \
    --vision_tower /data/gsva_pretrains/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --tf32 True \
    --max_grad_norm 1.0 \
    --ddp_find_unused_parameters True \
    --output_dir /data/checkpoints/$EXP_NAME \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 5e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.10 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --seed 3463 \
    --report_to none \
    --run_name $EXP_NAME

# rm -rf /data/checkpoints/$EXP_NAME/checkpoint-*
# cp -vr /data/checkpoints/$EXP_NAME /data/checkpoints/