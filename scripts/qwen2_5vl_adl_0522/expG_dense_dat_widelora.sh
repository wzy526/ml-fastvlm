#!/usr/bin/env bash
set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate vldat

# Exp G (0522): DENSE DAT (all 36 layers D) + wide LoRA on all 36 QKVO, LSE merge.
# ============================================================================
#
# Architecture choice: LSE merge (NOT D3 residual merge). The residual-merge
# / hd_out_proj code path has been deleted from modeling_qwen2_5vl_dat.py
# (commit "remove D3 residual gating"), so this run is structurally a
# 0514-style hd_gate-controlled LSE merge applied at every single layer.
#
# Changes vs. exp9_full_1d5l_sa1b_ivcap_hdgate-2.sh (0514 LSE baseline):
#   dat_layers      DLLLLL... (6 D)   ->   DDDD...DDDD (36 D)
#   lora_target_layers   "dat"        ->   "all"   (= 144 adapters with all-D)
#   dat_lr          1e-4              ->   5e-5    (safety: never trained all-D before)
#
# Hypothesis being tested
# -----------------------
# Even with wider LoRA (Run F), LSE-DAT might still be neutral because HD
# is injected at 6 sparse layers only -- 5 vanilla Qwen layers between
# each DAT injection wash out the HD perturbation via standard attention,
# leaving the LM head to never "see" HD content. Run G makes HD injection
# DENSE (every layer = D), so the HD signal is refreshed at every depth
# and has the maximum chance of reaching the LM head.
#
# Decision table (read alongside Run F)
# -------------------------------------
# F → 0.65, G > F          LSE-DAT helps when both LR scope and HD density
#                          are fixed. Recovery path: more density, longer
#                          training, possibly larger hr_scale.
# F → 0.65, G ≈ F          LSE-DAT is structurally neutral at any density;
#                          HD path projects orthogonal to LM head. Need a
#                          new HD-to-LM-head bridge (e.g., HD as prefix
#                          vision tokens à la LLaVA-NeXT tiling, or fix
#                          the offset-drift bug that pins HD sampling to
#                          image edges).
# F → 0.65, G < F          Dense DAT introduces a tax that outweighs HD
#                          contribution -- the original 6 D sparsity was
#                          right. Need cheaper-per-layer DAT (smaller
#                          inter_size / dropping intention branch / etc.).
# Both F, G ≈ 0.58         Wider LoRA can't recover anything; LSE-DAT
#                          actively poisons LR regardless of density or
#                          scope. Time to throw out current DAT and redesign.
#
# Risk notes
# ----------
# • All-D pattern has NEVER been trained before. Watch the first 50 steps:
#   if train/loss diverges or grad_norm explodes (>5), kill and retry with
#   dat_lr 2e-5 + warmup_steps 200.
# • Memory per GPU estimate (3B base, bf16, gradient checkpoint, batch 4):
#     base weight       ~6.0 GB
#     DAT params (all-D)~1.4 GB  (180M * 8 bytes inc. fp32 optim state)
#     LoRA + optim      ~0.2 GB
#     activations       ~5-8 GB peak
#     ====================
#     Total             ~13-16 GB per GPU at peak. Fits 24GB; if 16GB GPUs,
#     drop PER_DEVICE_BATCH to 2 and GRAD_ACCUM to 4 to keep effective
#     batch=64.
# • Expected per-step time is 1.5-2x Run E (6x cross-attn passes per step).
#   1 epoch should still finish in ~6-7 hours on 8 GPUs.

export WANDB_PROJECT="${WANDB_PROJECT:-vldat_experiments}"

ADL_TMP="/root/autodl-tmp"

export NUMEXPR_MAX_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

DATA_ROOT="${DATA_ROOT:-$ADL_TMP/models_data/sft_data}"
MODEL_PATH="${MODEL_PATH:-$ADL_TMP/models_data/Qwen2.5-VL-3B-Instruct}"
CKPT_ROOT="${CKPT_ROOT:-$ADL_TMP/vldat_experiments}"
CACHE_ROOT="${CACHE_ROOT:-$ADL_TMP/cache/vldat}"
EXP_NAME="${EXP_NAME:-0522_expG_dense_widelora_lse}"

DATA_JSON="${DATA_JSON:-$DATA_ROOT/llava_hr_essential_sa1b_ivcap.json}"

if [[ ! -f "$DATA_JSON" ]]; then echo "[ERROR] Missing data file: $DATA_JSON" >&2; exit 1; fi
if [[ ! -d "$DATA_ROOT/train_split" ]]; then echo "[ERROR] Missing image folder" >&2; exit 1; fi
if [[ ! -e "$DATA_ROOT/train_split/sa1b" ]]; then echo "[ERROR] Missing sa1b symlink" >&2; exit 1; fi
if [[ ! -d "$MODEL_PATH" ]]; then echo "[ERROR] Missing model path: $MODEL_PATH" >&2; exit 1; fi

mkdir -p "$CKPT_ROOT/$EXP_NAME"

export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_ROOT/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$CACHE_ROOT/torchinductor}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$CACHE_ROOT/cuda}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT/xdg}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0

# Dense DAT: every layer = D (36 D layers total).
DAT_LAYERS="DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD"

# Sanity check the pattern length matches Qwen2.5-VL-3B layer count.
if [[ "${#DAT_LAYERS}" -ne 36 ]]; then
    echo "[ERROR] DAT_LAYERS length=${#DAT_LAYERS}, expected 36" >&2
    exit 1
fi

torchrun --nproc_per_node=8 --master_port "${MASTER_PORT:-40751}" llava/train/train_qwen_dat.py \
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
    --dat_hd_gate_init -2.0 \
    --dat_warmup_steps 0 \
    --dat_inject_lr_image False \
    --dat_lr 5e-5 \
    --lora_enable True \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_target_layers "all" \
    --lora_lr 2e-5 \
    --tune_mm_vision False \
    --tune_mm_mlp False \
    --tune_mm_llm False \
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
    --save_steps "${SAVE_STEPS:-500}" \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_steps "${WARMUP_STEPS:-50}" \
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
    --run_name "$EXP_NAME"

source "$(dirname "${BASH_SOURCE[0]}")/_merge_after_train.sh"
