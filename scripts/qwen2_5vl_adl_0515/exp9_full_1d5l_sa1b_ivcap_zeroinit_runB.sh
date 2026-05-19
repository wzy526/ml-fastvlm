#!/usr/bin/env bash
set -euo pipefail

# Activate vldat conda env
eval "$(conda shell.bash hook)"
conda activate vldat

# Exp 9 (0515): Run B = Tier-1 + F4 (RMSNorm).
#
# Builds on exp9_full_1d5l_sa1b_ivcap_zeroinit.sh (zero-init HD KV, trainable
# hd_gate at -1.0), with three additive changes:
#
#   F1  --dat_use_spatial_attn_guide True
#       Switch the offset predictor's branch from "intention-only" to
#       "spatial-attention-guided". The conv_off_proj input becomes
#       [local_embed_lr, embed_intention] concatenated (ln_2 covers 2·D),
#       giving the sampler explicit spatial-attention cues rather than
#       deriving them from the embedded intention alone. This restores
#       the original DAT design (matches the layer-set assumption that the
#       LR attention is informative for HD sampling).
#
#   F3  --dat_warmup_steps 500
#       Two-phase schedule under LoRA: for the first 500 steps freeze the
#       LoRA adapters and only train DAT modules (offset, k_proj_hd,
#       v_proj_hd, hd_input_layernorm, hd_gate). After step 500 the LoRA
#       adapters unfreeze and joint training resumes. Goal: let the HD
#       pathway calibrate its sampling + projection alone before LLM-side
#       adaptation starts to react to a noisy HD signal.
#
#   F4  (architectural, in modeling_qwen2_5vl_dat.py) RMSNorm before k/v_proj_hd
#       sampled_hr (channel dim = hidden_size = 2048) is RMSNormed via a
#       new learnable Qwen2_5_VLRMSNorm before k_proj_hd / v_proj_hd. This
#       matches the LR-path attention input (q/k/v_proj always see
#       RMSNormed hidden_states via the decoder's input_layernorm). Without
#       this norm, value_hd has a different magnitude than value, and the
#       LSE merge injects a distribution-shifted signal at every DAT layer.
#
# Data mix: UNCHANGED from prior runs (llava_hr_essential_sa1b_ivcap.json).
#
# Expected wandb signatures vs the zeroinit baseline:
#   • Steps 0..499:
#       - dat/kvhd_weight_norm grows (DAT-only learning, no LoRA dilution)
#       - dat/hd_gate_raw stays near -1.0 or slowly drifts up (gate now
#         has a useful signal to amplify, no longer the only escape valve)
#       - train/loss high (LoRA adapters frozen, base LLM frozen too)
#   • Step 500:
#       - "[DATWarmup] Step 500: Phase 2 [LoRA]" log line
#       - trainable params count jumps (LoRA adapters thawed)
#       - loss drops sharply as LoRA starts compensating for the now-warm
#         HD path
#   • Post-warmup:
#       - dat/hd_gate_raw should ascend (sigmoid > 0.5) if HD content is
#         useful, descend if it is harmful — diagnostic of F1+F4 success
#       - dat/kvhd_weight_norm continues climbing
#
# Data prep (unchanged):
#   python scripts/qwen2_5vl_adl_0430/build_sa1b_mix.py --no_as_core

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
EXP_NAME="${EXP_NAME:-0515_full_1d5l_sa1b_ivcap_runB_spatial_warmup_rmsnorm}"

DATA_JSON="${DATA_JSON:-$DATA_ROOT/llava_hr_essential_sa1b_ivcap.json}"

if [[ ! -f "$DATA_JSON" ]]; then
    echo "[ERROR] Missing data file: $DATA_JSON" >&2
    echo "        Run: python scripts/qwen2_5vl_adl_0430/build_sa1b_mix.py --no_as_core" >&2
    exit 1
fi
if [[ ! -d "$DATA_ROOT/train_split" ]]; then
    echo "[ERROR] Missing image folder: $DATA_ROOT/train_split" >&2; exit 1
fi
if [[ ! -e "$DATA_ROOT/train_split/sa1b" ]]; then
    echo "[ERROR] Missing sa1b symlink: $DATA_ROOT/train_split/sa1b" >&2
    echo "        Run: python scripts/qwen2_5vl_adl_0430/build_sa1b_mix.py --no_as_core" >&2
    exit 1
fi
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "[ERROR] Missing model path: $MODEL_PATH" >&2; exit 1
fi

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

# Full 1D5L pattern (same as exp 8 / baseline_hr_essential)
DAT_LAYERS="DLLLLLDLLLLLDLLLLLDLLLLLDLLLLLDLLLLL"

torchrun --nproc_per_node=8 --master_port "${MASTER_PORT:-40617}" llava/train/train_qwen_dat.py \
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
    --dat_hd_gate_init -1.0 \
    --dat_warmup_steps 500 \
    --dat_lr 1e-4 \
    --lora_enable True \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_target_layers "dat" \
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

# Auto-merge LoRA + DAT weights
source "$(dirname "${BASH_SOURCE[0]}")/_merge_after_train.sh"
