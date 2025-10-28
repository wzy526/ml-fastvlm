#!/usr/bin/env bash
cd /root/ml-fastvlm

export TRANSFORMERS_OFFLINE=1
export DS_SKIP_CUDA_CHECK=1
source /root/miniconda3/bin/activate fastvlm
TOTAL_STEPS=2595
EXP_NAME="tdat-7b-l0d32-s12g8z3_ep2"

mkdir -p /data/checkpoints/$EXP_NAME

ds llava/train/train_dat.py \
    --deepspeed ./scripts/zero_configs/zero2.json \
    --model_name_or_path /data/gsva_pretrains/llava-v1_5-7b-hf \
    --version v1 \
    --extra_yaml_path ./configs/llava1_5_v2.yaml \
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
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --seed 3463 \
    --report_to none \
    --run_name $EXP_NAME