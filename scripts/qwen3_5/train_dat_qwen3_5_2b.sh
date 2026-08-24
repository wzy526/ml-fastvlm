#!/usr/bin/env bash
set -euo pipefail

# Qwen3.5-2B DAT 训练模板 (实验 pod, 无外网)。
#
# Qwen3.5 是混合架构 (3/4 GatedDeltaNet 线性注意力 + 1/4 full attention):
# - DAT 层只能落在 full_attention 位置 (2B: 层 3,7,11,15,19,23 共 6 个)。
#   --dat_layers auto  = 全部 full-attn 层挂 DAT (推荐起点)
#   --dat_layers auto2 = 每隔一个 full-attn 层挂 DAT
# - tokenizer 是 250k 新词表, token id 由 train 脚本运行时自动解析。
# - chat template 默认 thinking; SFT 数据经 apply_chat_template 会自动带
#   空 <think></think> 块 (官方非思考 SFT 推荐格式), 无需特殊处理。
# - 线性注意力 kernel 依赖 fla (pod 已装 0.5.0); causal_conv1d 缺失时走
#   torch fallback, 只影响速度不影响正确性。

export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NUMEXPR_MAX_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

XZF_ROOT="${XZF_ROOT:-/home/ea-cvfa-aigc-x2v-2/xzf}"
MODEL_PATH="${MODEL_PATH:-/workspace/model_cache/Qwen3.5-2B}"
DATA_JSON="${DATA_JSON:?set DATA_JSON to the sft json}"
IMAGE_ROOT="${IMAGE_ROOT:?set IMAGE_ROOT to the image folder}"
CKPT_ROOT="${CKPT_ROOT:-$XZF_ROOT/vldat_experiments}"
EXP_NAME="${EXP_NAME:-qwen3_5_2b_dat_auto_freeze_base}"

mkdir -p "$CKPT_ROOT/$EXP_NAME"

torchrun --nproc_per_node="${NPROC:-8}" --master_port "${MASTER_PORT:-40977}" \
    llava/train/train_qwen_dat.py \
    --deepspeed ./scripts/zero_configs/zero2.json \
    --model_name_or_path "$MODEL_PATH" \
    --model_family qwen3_5 \
    --data_path "$DATA_JSON" \
    --image_folder "$IMAGE_ROOT" \
    --coupled_lr_hd True \
    --use_dat True \
    --dat_layers auto \
    --dat_grid_size 20 \
    --dat_off_grps 8 \
    --dat_inter_size 128 \
    --dat_hr_scale 3 \
    --dat_hd_proj True \
    --dat_use_intention_branch True \
    --dat_intention_as_gate True \
    --dat_freeze_base False \
    --dat_lr 1e-4 \
    --tune_mm_vision False \
    --tune_mm_mlp False \
    --tune_mm_llm False \
    --bf16 True \
    --tf32 True \
    --max_grad_norm 1.0 \
    --output_dir "$CKPT_ROOT/$EXP_NAME" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 500 \
    --learning_rate 1e-6 \
    --weight_decay 0. \
    --warmup_steps 100 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --group_by_modality_length True \
    --dataloader_num_workers 8 \
    --dataloader_pin_memory True \
    --dataloader_drop_last True \
    --seed 42 \
    --report_to none \
    --run_name "$EXP_NAME"
