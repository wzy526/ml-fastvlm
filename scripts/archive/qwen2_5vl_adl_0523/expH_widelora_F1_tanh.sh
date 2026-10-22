#!/usr/bin/env bash
set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate vldat

# Exp H (0523): F config + F1 spatial_attn_guide ON + tanh sample_locs binding.
# ============================================================================
#
# What changed vs. Run F (0522_expF_widelora_lse):
#   --dat_use_spatial_attn_guide  False  -->  True       (F1 re-enabled)
#   modeling: sample_locs = ...clamp(-1,1)  -->  ...tanh()  (offset drift fix)
# Nothing else moves.
#
# Why this run
# ------------
# Run F (sparse 6D, wide LoRA, LSE merge, no F1, hard clamp on sample_locs)
# scored hrbench4k=0.5975 / hrbench8k=0.5262 — still 2-3 pt under baseline
# (0.62). Run G (dense 36D, same other knobs) scored 0.5938 / 0.5450 — F≈G
# on 4k, confirming HD path is essentially dead: adding 30 more DAT layers
# contributes <0.5 pt because each layer's HD samples are useless.
#
# Wandb offset visualization on 0520+ runs shows sample_locs drifting to
# image borders / padding regions. Two compounding bugs cause this:
#
#   (i) clamp(-1, 1) on (refs+offsets) has dL/d(offset)=0 outside the
#       boundary. Once a cell's offset random-walks out of [-1, 1], it
#       receives zero gradient forever — a one-way ticket to the edge.
#
#  (ii) conv_off_proj is zero-init and gets only the weak gradient from
#       the HD-loss back-prop path (which travels through cross-attn,
#       attention merge, and 30+ LLM layers). The signal-to-noise of
#       this gradient is so low that random walk dominates → offsets
#       drift before they learn anything meaningful → drift hits the
#       clamp boundary → dies.
#
# Two structural fixes, both in this run:
#
#   (a) tanh sample_locs (modeling-side, applies to ALL future DAT runs):
#       sample_locs = (refs + offsets).tanh() instead of .clamp(-1, 1).
#       Gradient is strictly positive everywhere, so a cell that drifted
#       out can always come back. Init trade-off: sample_locs at
#       offsets=0 are tanh(refs) ∈ [-0.738, 0.738] instead of refs ∈
#       [-0.947, 0.947] — ~26% inward compression of the initial grid,
#       recoverable as offsets learn to push outward.
#
#   (b) F1 spatial_attn_guide ON: multiplies embed_lr_rep by a softmax-
#       normalized Q_int·Q_lr attention map before feeding off_guide
#       into conv_off_proj. Effect: low-LR-attention positions get
#       embed_lr_rep ≈ 0 → offsets ≈ 0 → sample_locs stay near refs;
#       high-LR-attention positions drive the conv with strong magnitude
#       → offsets get meaningful gradient there. Implicit "where to
#       look" supervision that 0519 runC actually achieved (uniform,
#       non-drifting offset distribution observed in that run's wandb).
#       0520+ runs disabled F1 over a Q·Q-not-principled concern, then
#       lost the only mechanism that was keeping offsets stable.
#
# Tanh and F1 are complementary, not redundant:
#   • tanh = "soft landing": no cell can die, but doesn't actively
#     pull offsets back to useful positions.
#   • F1 = "purposeful gradient": gives conv_off_proj a strong, data-
#     driven signal where to look; reduces random-walk amplitude.
#
# Decision table for hrbench4k avg
# --------------------------------
# H ≈ 0.65-0.67    -> tanh + F1 together unlocked HD. DAT is finally
#                     ahead of baseline using only 1/3 the LR tokens.
#                     Promote this config; next: dense (G-style) + F1
#                     + tanh to see if HD gain scales with density.
# H ≈ 0.60-0.62    -> Offset bug fixed, but HD signal is weak. Likely
#                     N4 bottleneck (sa1b caption data doesn't reward
#                     HD use); need to mine HD-discriminative data.
# H ≈ 0.60 (≈F)    -> tanh + F1 didn't move the needle. HD path is
#                     structurally orthogonal to LM head regardless of
#                     sampling quality. Time for architecture rework
#                     (HD-as-prefix-tokens / LLaVA-NeXT tiling).

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
EXP_NAME="${EXP_NAME:-0523_expH_widelora_F1_tanh}"

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

# Same DAT layer pattern as Run F: 1D every 6 layers (sparse, 6 D layers).
DAT_LAYERS="DLLLLLDLLLLLDLLLLLDLLLLLDLLLLLDLLLLL"

torchrun --nproc_per_node=8 --master_port "${MASTER_PORT:-40761}" llava/train/train_qwen_dat.py \
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
    --dat_use_spatial_attn_guide True \
    --dat_shared_vit False \
    --dat_freeze_base False \
    --dat_hd_gate_init -2.0 \
    --dat_warmup_steps 0 \
    --dat_inject_lr_image False \
    --dat_lr 1e-4 \
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
