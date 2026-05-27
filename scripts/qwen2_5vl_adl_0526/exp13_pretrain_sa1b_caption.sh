#!/usr/bin/env bash
set -euo pipefail

# Activate vldat conda env
eval "$(conda shell.bash hook)"
conda activate vldat

# Exp 13 (0526): Pretrain stage on SA-1B captions.
# ============================================================================
#
# Goal
# ----
# Bootstrap the projector (visual.merger) and the newly added HD path (all
# DAT_KEYS_MATCH params) on a large pool of generic image captions, with the
# LLM and ViT held completely frozen. This is the LLaVA-Stage-1-style
# pretrain that should precede any SFT (e.g. Run E1 on llava_hr_essential
# mixes) when the model has previously-untrained modules.
#
# What is trainable here vs. Run E1 (SFT)
# ---------------------------------------
#     Module                          Run E1   Pretrain (this)
#     visual.blocks (ViT)             frozen   frozen
#     visual.merger (projector)       frozen   TRAINABLE  <-- tune_mm_mlp True
#     language_model.layers (LLM)     frozen   frozen
#     lm_head                         frozen   frozen
#     LoRA adapters on DAT layers     trained  DISABLED   <-- lora_enable False
#     DAT params (DAT_KEYS_MATCH)     trained  TRAINABLE
#
# Net trainable surface: visual.merger (~50M) + DAT (~60M) = ~110M / ~3.5B.
# LLM has strictly zero gradient flow (no LoRA, tune_mm_llm False).
#
# Important: do NOT pass --dat_freeze_base True
# ---------------------------------------------
# That branch in train_qwen_dat.py:2782-2784 calls freeze_base_unfreeze_dat()
# (defined in modeling_qwen2_5vl_dat.py:2135-2145), which freezes EVERY
# non-DAT param including visual.merger. We want merger trainable, so we
# take the else-branch at train_qwen_dat.py:2785-2790 instead, which
# 1. runs set_model(model, model_args) honoring tune_mm_{vision,mlp,llm}, and
# 2. then loops over all params, re-enabling every DAT_KEYS_MATCH-tagged one.
# That gives us the exact (visual.merger + DAT) trainable surface above.
#
# Architecture / regularization (post D3 + STE)
# ----------------------------------------------
# The codebase has been simplified since the original Run E1 D1+D3 era:
#   * D3 removed: ``dat_use_residual_merge`` / ``hd_out_proj`` no longer
#     exist anywhere (verified: 0 hits in modeling_qwen2_5vl_dat.py /
#     train_qwen_dat.py, also dropped from DAT_KEYS_MATCH). Merge is back
#     to the standard LSE form (modeling_qwen2_5vl_dat.py:999-1060):
#         lse = logaddexp(lse1_ans, lse2)
#         w1 = exp(lse1_ans - lse),  w2 = exp(lse2 - lse)
#         out[ans] = w1·out1_ans + w2·out2
#     gated optionally by ``hd_gate`` (lse2 += log σ(hd_gate)) — disabled
#     here, see below.
#   * sample_locs uses a Straight-Through Estimator on clamp
#     (modeling_qwen2_5vl_dat.py:955-956):
#         x = references + offsets
#         sample_locs = (x + (x.clamp(-1, 1) - x).detach())
#     Forward is physically valid for F.grid_sample (always in [-1, 1]);
#     backward is identity, so offsets that drift past ±1 keep getting
#     gradient pointing back to the clamp surface — no dead zone (plain
#     clamp) and no center-collapse from gradient decay (tanh).
#   * D1 OFF here (--dat_inject_lr_image False). The original answer-span-
#     only injection is in effect. Pretrain stage; we keep the surface
#     minimal and let D1 be re-enabled by the SFT recipe if desired.
#   * F1 OFF (--dat_use_spatial_attn_guide False), intention_branch ON.
#   * hd_gate: parameter NOT created (omit --dat_hd_gate_init entirely
#     => hd_gate_init=None => modeling line 622-624 sets
#     self.hd_gate=None => merge skips the log σ(g) bias). With LSE merge
#     and no gate, w2 is purely driven by the cross-attn lse — projector
#     and DAT/HD KV projections learn freely from step 0, no cold-start.
#
# Pretrain-specific changes vs. Run E1
# ------------------------------------
# - Data:        llava_sa1b_caption_pretrain.json (pure IV-en captions, no
#                HR-essential, no AS-Core). Build with
#                scripts/qwen2_5vl_adl_0430/build_sa1b_caption_pretrain.py.
# - LoRA:        --lora_enable False (LLM truly untouched).
# - Projector:   --tune_mm_mlp True + --mm_projector_lr 1e-4.
# - LR sched:    Cosine, warmup 100 (vs. 50 in Run E1) since the projector
#                starts from a checkpointed but distribution-shifted state.
# - Grouping:    --group_by_modality_length False; SA-1B is uniformly
#                single-image, length bucketing buys nothing.
# - No merge:    Skips _merge_after_train.sh; nothing to merge (no LoRA).
#
# Following stages
# ----------------
# After this pretrain finishes, the projector and HD path will be warmed up.
# To continue with the SFT recipe (Run E1 / E2), point
# --model_name_or_path at this pretrain's output_dir and re-enable
# --lora_enable True. The DAT weights are reloaded automatically by
# Qwen2_5_VLDATForConditionalGeneration.from_pretrained (conversion mapping
# is registered in modeling_qwen2_5vl_dat.py:2193+).
#
# Expected wandb signatures
# -------------------------
# - train/loss: starts higher than Run E1 (captions are much more diverse
#   than HR-essential SFT), drops monotonically. Should plateau within
#   1-2 epochs; longer doesn't help without unfreezing LLM.
# - visual.merger grad: should be sizable; if it stays at machine epsilon
#   something's wrong with tune_mm_mlp wiring.
# - DAT grads: conv_off_proj / k_proj_hd / v_proj_hd / hd_input_layernorm
#   should all be non-zero. No hd_gate to watch (omitted on purpose).

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
EXP_NAME="${EXP_NAME:-0526_pretrain_sa1b_caption_lse_ste}"

DATA_JSON="${DATA_JSON:-$DATA_ROOT/llava_sa1b_caption_pretrain.json}"

if [[ ! -f "$DATA_JSON" ]]; then
    echo "[ERROR] Missing data file: $DATA_JSON" >&2
    echo "        Build it first via:" >&2
    echo "          python scripts/qwen2_5vl_adl_0430/build_sa1b_caption_pretrain.py" >&2
    exit 1
fi
if [[ ! -d "$DATA_ROOT/train_split" ]]; then
    echo "[ERROR] Missing image folder: $DATA_ROOT/train_split" >&2; exit 1
fi
if [[ ! -e "$DATA_ROOT/train_split/sa1b" ]]; then
    echo "[ERROR] Missing sa1b symlink: $DATA_ROOT/train_split/sa1b" >&2; exit 1
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

# Full 1D5L pattern (same as Run B/C/D/E)
DAT_LAYERS="DLLLLLDLLLLLDLLLLLDLLLLLDLLLLLDLLLLL"

torchrun --nproc_per_node=8 --master_port "${MASTER_PORT:-40631}" llava/train/train_qwen_dat.py \
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

# NOTE: no _merge_after_train.sh here — there are no LoRA adapters to merge.
# The output_dir already contains a fully-loadable HF checkpoint:
#   - visual.merger weights (trained)
#   - DAT params (trained)
#   - ViT / LLM / lm_head weights (unchanged, copied through by Trainer)
# Point any downstream SFT script at $CKPT_ROOT/$EXP_NAME via --model_name_or_path.
