#!/usr/bin/env bash

export WANDB_PROJECT="vldat_experiments"

CKPT_ROOT="/mnt/ephemeral/vldat_experiments"
EXP_NAME="qwen2_5vl-3b-dat-z3_8l1d_s12_g6_i128_inten_gate_ld_1M"
mkdir -p $CKPT_ROOT/$EXP_NAME

# 36-layer Qwen2.5-VL-3B: 8L1D pattern, DAT on layers 8, 17, 26, 35 (0-indexed)
DAT_LAYERS="LLLLLLLLDLLLLLLLLDLLLLLLLLDLLLLLLLLD"

ds llava/train/train_qwen_dat.py \
    --deepspeed ./scripts/zero_configs/zero2.json \
    --model_name_or_path /mnt/ephemeral/base_models/Qwen2.5-VL-3B-Instruct \
    --model_family qwen2_5_vl \
    --data_path /mnt/ephemeral/sft_data/llava_hd_merged_1m.json \
    --image_folder /mnt/ephemeral/sft_data/train_split \
    --coupled_lr_hd True \
    --use_dat True \
    --dat_layers "$DAT_LAYERS" \
    --dat_grid_size 12 \
    --dat_off_grps 8 \
    --dat_inter_size 128 \
    --dat_hr_scale 3 \
    --dat_hd_proj False \
    --dat_use_intention_branch True \
    --dat_intention_as_gate True \
    --dat_freeze_base False \
    --dat_lr 1e-5 \
    --tune_mm_vision False \
    --tune_mm_mlp False \
    --tune_mm_llm True \
    --bf16 True \
    --tf32 True \
    --max_grad_norm 1.0 \
    --output_dir $CKPT_ROOT/$EXP_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --learning_rate 1e-6 \
    --weight_decay 0. \
    --warmup_steps 100 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --logging_nan_inf_filter False \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --group_by_modality_length True \
    --dataloader_num_workers 4 \
    --dataloader_pin_memory True \
    --dataloader_prefetch_factor 3 \
    --dataloader_persistent_workers True \
    --dataloader_drop_last True \
    --seed 42 \
    --report_to "wandb" \
    --run_name $EXP_NAME
