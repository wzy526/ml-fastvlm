#!/usr/bin/env bash

source /home/coder/miniforge3/bin/activate fastvlm


CKPT_ROOT=/mnt/ephemeral/exeperiments/$EXP_NAME
EXP_NAME="tdat-7b-l0d32-s12g8z3_ep2-flash-trial"


mkdir -p /mnt/ephemeral/exeperiments/$EXP_NAME

ds llava/train/train_dat.py \
    --deepspeed ./scripts/zero_configs/zero2.json \
    --model_name_or_path /home/coder/work/llava-v1.5-7b \
    --version v1 \
    --extra_yaml_path ./configs/llava1_5_v1.yaml \
    --data_path  /home/coder/work/llava-665k/llava_v1_5_mix665k.json \
    --image_folder /home/coder/work/llava-665k/train_split \
    --vision_tower /home/coder/work/clip-vit-large-patch14-336 \
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
    --output_dir $CKPT_ROOT/$EXP_NAME \
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