#!/usr/bin/env bash

CKPT_ROOT=Qwen2-VL-2B
EXP_NAME="qwen2vl-2b-baseline"
mkdir -p $CKPT_ROOT/$EXP_NAME

ds llava/train/train_qwen_dat.py \
    --deepspeed ./scripts/zero_configs/zero3.json \
    --model_name_or_path /mnt/ephemeral/base_models/Qwen2-VL-2B-Instruct \
    --data_path /mnt/ephemeral/sft_data/llava_v1_5_mix665k_shuffled_full.json \
    --image_folder /mnt/ephemeral/sft_data/train_split \
    --freeze_vision False \
    --bf16 True \
    --tf32 True \
    --max_grad_norm 10 \
    --output_dir $CKPT_ROOT/$EXP_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_steps 100 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --group_by_modality_length True \
    --dataloader_num_workers 4 \
    --dataloader_pin_memory True \
    --dataloader_prefetch_factor 3 \
    --dataloader_persistent_workers True \
    --dataloader_drop_last True \
    --seed 42 \
    --report_to "none" \
    --max_steps 300
