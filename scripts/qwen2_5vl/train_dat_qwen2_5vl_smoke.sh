#!/usr/bin/env bash
# ============================================================
# Smoke-test for the two-pass + LSE merge DAT implementation.
# Runs 10 training steps on 1 GPU with gradient checkpointing.
# Validates: forward pass, backward pass, GC compatibility.
# ============================================================
# Usage:
#   bash scripts/qwen2_5vl/train_dat_qwen2_5vl_smoke.sh
# ============================================================

set -e   # exit immediately on any error

export WANDB_MODE=disabled
export HF_ENDPOINT=https://hf-mirror.com   # remove if not in China

# ── debug flags ──────────────────────────────────────────────────────────────
export DAT_DEBUG=1          # enable per-block prints in modeling_qwen2_5vl_dat.py
export DAT_DEBUG_STEPS=2    # only print the first 2 calls per key (avoid log spam)
# ─────────────────────────────────────────────────────────────────────────────

OUT_DIR="/tmp/dat_smoke_$(date +%s)"
mkdir -p "$OUT_DIR"
echo "Output dir : $OUT_DIR"

# Same DAT layer pattern as local training script
DAT_LAYERS="DLLLLLDLLLLLDLLLLLDLLLLLDLLLLLDLLLLL"

deepspeed --num_gpus=1 \
    llava/train/train_qwen_dat.py \
    --deepspeed ./scripts/zero_configs/zero2.json \
    --model_name_or_path /data/base_models/Qwen2.5-VL-3B-Instruct \
    --model_family qwen2_5_vl \
    --data_path /data/sft_data/llava_hd_merged_1m.json \
    --image_folder /data/sft_data/train_split \
    --coupled_lr_hd True \
    --use_dat True \
    --dat_layers "$DAT_LAYERS" \
    --dat_grid_size 12 \
    --dat_off_grps 8 \
    --dat_inter_size 128 \
    --dat_hr_scale 3 \
    --dat_hd_proj True \
    --dat_use_intention_branch True \
    --dat_intention_as_gate True \
    --dat_freeze_base False \
    --dat_lr 5e-6 \
    --visualization_every_n_steps 5 \
    --tune_mm_vision False \
    --tune_mm_mlp False \
    --tune_mm_llm True \
    --bf16 True \
    --tf32 True \
    --max_grad_norm 1.0 \
    --output_dir "$OUT_DIR" \
    --max_steps 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "no" \
    --save_strategy "no" \
    --learning_rate 1e-6 \
    --weight_decay 0. \
    --warmup_steps 0 \
    --lr_scheduler_type "constant" \
    --logging_steps 1 \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --group_by_modality_length False \
    --dataloader_num_workers 2 \
    --dataloader_pin_memory False \
    --dataloader_drop_last False \
    --seed 42 \
    --report_to "none" \
    2>&1 | tee "$OUT_DIR/smoke.log"

echo "============================================"
echo "Smoke test finished.  Log: $OUT_DIR/smoke.log"
echo "============================================"
