#!/usr/bin/env bash
set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate vldat

# 0606 Stage-1 pretrain: SA-1B caption, Direction A, NO hd_gate.
# LAYER-COUNT ABLATION: 2 DAT layers per 6-layer module (DDLLLL x6 = 12 DAT).
# ============================================================================
#
# Design
# ------
# - Direction A (--dat_image_hd_for_question True): question tokens also receive
#   image-conditioned HD. Offsets for that path depend ONLY on the LR image, so
#   the HD K/V is causally available to every question position.
#
# - NO hd_gate (omit --dat_hd_gate_init → self.hd_gate = None). Cold-start
#   safety comes from zero-init v_proj_hd (HD value = 0 at step 0, so HD is
#   silent at init and cannot dilute the LR branch). Without a gate, v_proj_hd
#   receives full-strength gradient and the HD level is decided organically by
#   the LSE attention competition — no scalar to collapse closed.
#
# - LLM + LR ViT FROZEN. Only DAT module + visual.merger (projector) train.
#   This is the clean stage to align projector + DAT to both HD consumption
#   paths (answer = intention-HD, question = image-HD).
#
# Architecture: LSE merge / STE on sample_locs / no D1 / intention_branch+gate
# ON / grid_size=20, off_grps=8, inter_size=128, hr_scale=3, hd_max_pixels=5M
# / DAT_LAYERS = DDLLLLDDLLLLDDLLLLDDLLLLDDLLLLDDLLLL (12 DAT, 2 per module).
#   ^ ONLY difference vs 0604 baseline (which was DLLLLL x6 = 6 DAT).
#
# Data: llava_sa1b_caption_pretrain.json (503k SA-1B captions).

export WANDB_PROJECT="${WANDB_PROJECT:-vldat_experiments}"

ADL_TMP="/root/autodl-tmp"

export NUMEXPR_MAX_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# -------- Path config --------
DATA_ROOT="${DATA_ROOT:-$ADL_TMP/models_data/sft_data}"
MODEL_PATH="${MODEL_PATH:-$ADL_TMP/models_data/Qwen2.5-VL-3B-Instruct}"
CKPT_ROOT="${CKPT_ROOT:-$ADL_TMP/vldat_experiments}"
CACHE_ROOT="${CACHE_ROOT:-$ADL_TMP/cache/vldat}"
EXP_NAME="${EXP_NAME:-0606_pretrain_sa1b_caption_dirA_nogate_12dat}"

DATA_JSON="${DATA_JSON:-$DATA_ROOT/llava_sa1b_caption_pretrain.json}"

if [[ ! -f "$DATA_JSON" ]]; then
    echo "[ERROR] Missing data file: $DATA_JSON" >&2
    echo "        Build it first via:" >&2
    echo "          python scripts/qwen2_5vl_adl_0430/build_sa1b_caption_pretrain.py" >&2
    exit 1
fi
if [[ ! -d "$DATA_ROOT/train_split" ]]; then echo "[ERROR] Missing $DATA_ROOT/train_split" >&2; exit 1; fi
if [[ ! -e "$DATA_ROOT/train_split/sa1b" ]]; then echo "[ERROR] Missing sa1b symlink" >&2; exit 1; fi
if [[ ! -d "$MODEL_PATH" ]]; then echo "[ERROR] Missing $MODEL_PATH" >&2; exit 1; fi

mkdir -p "$CKPT_ROOT/$EXP_NAME"

export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_ROOT/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$CACHE_ROOT/torchinductor}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$CACHE_ROOT/cuda}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT/xdg}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME"

# -------- Single-node 8 GPU --------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0

# ABLATION: 2 DAT layers per 6-layer module (12 DAT total). Baseline = DLLLLL x6.
DAT_LAYERS="DDLLLLDDLLLLDDLLLLDDLLLLDDLLLLDDLLLL"

torchrun --nproc_per_node=8 --master_port "${MASTER_PORT:-40841}" llava/train/train_qwen_dat.py \
    --model_name_or_path "$MODEL_PATH" \
    --model_family qwen2_5_vl \
    --data_path "$DATA_JSON" \
    --image_folder "$DATA_ROOT/train_split" \
    --use_hr_first_resize False \
    --hd_max_pixels 5017600 \
    --use_dat True \
    --dat_layers "$DAT_LAYERS" \
    --dat_grid_size 20 \
    --dat_off_grps 8 \
    --dat_inter_size 128 \
    --dat_hr_scale 3 \
    --dat_hd_proj True \
    --dat_use_intention_branch True \
    --dat_intention_as_gate True \
    --dat_use_spatial_attn_guide False \
    --dat_shared_vit False \
    --dat_freeze_base False \
    --dat_warmup_steps 0 \
    --dat_inject_lr_image False \
    --dat_image_hd_for_question True \
    --dat_lr 1e-4 \
    --lora_enable False \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm False \
    --mm_projector_lr 1e-4 \
    --kd_on False \
    --bf16 True \
    --tf32 True \
    --max_grad_norm 1.0 \
    --output_dir "$CKPT_ROOT/$EXP_NAME" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS:-1}" \
    --per_device_train_batch_size "${PER_DEVICE_BATCH:-4}" \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps "${GRAD_ACCUM:-2}" \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps "${SAVE_STEPS:-1000}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT:-3}" \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_steps "${WARMUP_STEPS:-100}" \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --group_by_modality_length False \
    --dataloader_num_workers 8 \
    --dataloader_pin_memory True \
    --dataloader_prefetch_factor 2 \
    --dataloader_persistent_workers True \
    --dataloader_drop_last True \
    --seed 42 \
    --report_to "wandb" \
    --run_name "$EXP_NAME"

# No LoRA merge needed (LLM frozen). Output is a self-contained HF ckpt that
# the SFT stage (exp_sft_dirA_nogate_full_12dat.sh) consumes via --model_name_or_path.
