#!/usr/bin/env bash

source /home/coder/miniforge3/bin/activate fastvlm

# 使用 Hugging Face 镜像站（国内网络）
export HF_ENDPOINT=https://hf-mirror.com

CKPT_ROOT=/Qwen2-VL-2B/
EXP_NAME="tdat-qwen2vl-2b-V1"
export WANDB_PROJECT="MMDAT-2025"
mkdir -p $CKPT_ROOT/$EXP_NAME

# 使用 Qwen2-VL 自己的 vision encoder + 1D RoPE (默认)
# 使用本地模型路径避免网络下载
ds llava/train/train_dat.py \
    --deepspeed ./scripts/zero_configs/zero2.json \
    --model_name_or_path ./Qwen2-VL-2B \
    --version v1 \
    --extra_yaml_path ./configs/llava1_5_v2.yaml \
    --data_path /home/coder/work/llava-665k/llava_v1_5_mix665k.json \
    --image_folder /home/coder/work/llava-665k/train_split \
    --vision_tower ./Qwen2-VL-2B \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
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
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 8192 \
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

