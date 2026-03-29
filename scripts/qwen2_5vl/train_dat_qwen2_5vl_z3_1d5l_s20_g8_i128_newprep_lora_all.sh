#!/usr/bin/env bash

export WANDB_PROJECT="vldat_experiments"
export CUDA_VISIBLE_DEVICES=4,5,6,7

CKPT_ROOT="/mnt/ephemeral/vldat_experiments"
EXP_NAME="qwen2_5vl-3b-dat-z3_1d5l_s20_g8_i128_newprep_lora_all"
mkdir -p $CKPT_ROOT/$EXP_NAME

# 36-layer Qwen2.5-VL-3B: 1D5L pattern, DAT on layers 0, 6, 12, 18, 24, 30 (0-indexed)
DAT_LAYERS="DLLLLLDLLLLLDLLLLLDLLLLLDLLLLLDLLLLL"

torchrun --nproc_per_node=4 --master_port 39501 llava/train/train_qwen_dat.py \
    --model_name_or_path /home/coder/downloaded_data/base_models/Qwen2.5-VL-3B-Instruct \
    --model_family qwen2_5_vl \
    --data_path /home/coder/downloaded_data/sft_data/llava_hd_merged_1m.json \
    --image_folder /home/coder/downloaded_data/sft_data/train_split \
    --use_dat True \
    --dat_layers "$DAT_LAYERS" \
    --dat_grid_size 20 \
    --dat_off_grps 8 \
    --dat_inter_size 128 \
    --dat_hr_scale 3 \
    --dat_hd_proj True \
    --dat_use_intention_branch True \
    --dat_intention_as_gate True \
    --dat_freeze_base False \
    --dat_lr 1e-4 \
    --lora_enable True \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_target_layers all \
    --lora_lr 2e-5 \
    --tune_mm_vision False \
    --tune_mm_mlp False \
    --tune_mm_llm False \
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
    --save_steps 1000 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_steps 100 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --group_by_modality_length True \
    --dataloader_num_workers 8 \
    --dataloader_pin_memory True \
    --dataloader_prefetch_factor 2 \
    --dataloader_persistent_workers True \
    --dataloader_drop_last True \
    --seed 42 \
    --report_to "wandb" \
    --run_name $EXP_NAME
