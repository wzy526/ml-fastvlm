#!/usr/bin/env bash

export WANDB_PROJECT="vldat_experiments"
# node06 RTX PRO 6000 — only 4 GPUs available.
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# --- Keep compile/autotune caches OFF the NFS-mounted $HOME ---
# /cluster/home1/xzf lives on NFS, so the defaults (~/.triton, ~/.cache/...)
# cause DeepSpeed to warn about autotune hangs on process exit. Redirect
# every compilation cache to node06-local /tmp instead.
LOCAL_CACHE_ROOT="/tmp/${USER:-xzf}/vldat_cache"
export TRITON_CACHE_DIR="$LOCAL_CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$LOCAL_CACHE_ROOT/torchinductor"
export CUDA_CACHE_PATH="$LOCAL_CACHE_ROOT/cuda"
export XDG_CACHE_HOME="$LOCAL_CACHE_ROOT/xdg"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME"

# Checkpoints go on nvme6a (separate disk from training data to avoid IO contention).
CKPT_ROOT="/cluster/nvme6a/xzf/vldat_experiments"
EXP_NAME="dat_qwen2_5vl_z3_1d5l_s20_g8_i128_hd251k_lora_dat"
mkdir -p $CKPT_ROOT/$EXP_NAME

# Training data lives on nvme6.
DATA_ROOT="/cluster/nvme6/xzf/sft_data"
MODEL_PATH="/cluster/nvme6/xzf/base_models/Qwen2.5-VL-3B-Instruct"

# 36-layer Qwen2.5-VL-3B: 1D5L pattern, DAT on layers 0, 6, 12, 18, 24, 30 (0-indexed)
DAT_LAYERS="DLLLLLDLLLLLDLLLLLDLLLLLDLLLLLDLLLLL"

# 4 GPUs × per_device_batch 8 × grad_accum 2  →  effective batch 64 (same as the original 8-GPU recipe).
# ddp_find_unused_parameters=True is REQUIRED under LoRA+DAT: DAT layers conditionally
# skip k_proj_hd/v_proj_hd/off-conv/proj_intention/hd_gate when a sample has no image
# (see modeling_qwen2_5vl_dat.py forward), so the set of params contributing gradients
# differs across ranks / micro-steps. Default DDP static bucket layout deadlocks at
# the first NCCL AllReduce → 100% GPU util hang.
torchrun --nproc_per_node=4 --master_port 40010 llava/train/train_qwen_dat.py \
    --model_name_or_path $MODEL_PATH \
    --model_family qwen2_5_vl \
    --data_path $DATA_ROOT/llava_hd251k.json \
    --image_folder $DATA_ROOT/train_split \
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
    --lora_target_layers "dat" \
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
    --gradient_accumulation_steps 2 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_steps 50 \
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
