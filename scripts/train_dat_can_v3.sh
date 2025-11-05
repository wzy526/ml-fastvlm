#!/usr/bin/env bash

source /home/coder/miniforge3/bin/activate fastvlm

CKPT_ROOT=/mnt/ephemeral/exps/
EXP_NAME="tdat-llava1_5-7b-7l1d-s12g8z3-cat-hdproj-ep1"
export WANDB_PROJECT="MMDAT-2025"
mkdir -p $CKPT_ROOT/$EXP_NAME

ds llava/train/train_dat.py \
    --deepspeed ./scripts/zero_configs/zero2.json \
    --model_name_or_path /home/coder/work/llava-v1.5-7b \
    --version v1 \
    --extra_yaml_path ./configs/llava1_5_v3.yaml \
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
    --max_grad_norm 10 \
    --ddp_find_unused_parameters False \
    --output_dir $CKPT_ROOT/$EXP_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --dataloader_pin_memory True \
    --dataloader_prefetch_factor 3 \
    --dataloader_persistent_workers True \
    --dataloader_drop_last True \
    --lazy_preprocess True \
    --seed 42 \
    --report_to "wandb" \
    --run_name $EXP_NAME \
    --resume_from_checkpoint True