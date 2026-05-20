#!/usr/bin/env bash
set -euo pipefail

# Activate vldat conda env
eval "$(conda shell.bash hook)"
conda activate vldat

# Exp 12 (0520): Run E1 = D1 + D3 + spatial guide FULLY OFF.
# ============================================================================
#
# Companion: exp12_full_runE_imgonly.sh (Run E2) drops the intention branch
# too, leaving offset prediction with image-only conv stack.
#
# Motivation (from per-layer attention diagnostics)
# -------------------------------------------------
# scripts/qwen2_5vl_adl_0519/_diagnose_attn_per_dat_layer.py on Run C +
# scripts/qwen2_5vl_adl_0519/_diagnose_baseline_attn.py on base Qwen2.5-VL
# revealed two architectural defects in the original DAT design:
#
# (1) **HD injected at the wrong positions.**  The original DAT cross-attn
#     only feeds HD K/V into Q at the ANSWER span. But the baseline Qwen
#     LR attention shows answer-Q gives LR-image only ~2-12% of its mass —
#     the image -> answer information path goes via qa_prefix tokens, not
#     direct answer->image attention. Injecting HD into a position that
#     barely consults image features means HD info has no downstream
#     consumer. (D1 fix.)
#
# (2) **LSE-merge weight is anti-correlated with HD attention quality.**
#     On Run C: L6/L12 have w_HD ≈ 0.63/0.75 but HD_top10 mass ≈ 0.31/0.34
#     (diffuse HD attn, but loudly merged in), while L24 has w_HD ≈ 0.08
#     but HD_top10 mass ≈ 0.41 (peakier HD attn, suppressed by the merge).
#     LSE merge picks layers by ``lse2 - lse1`` magnitude, not by HD attn
#     informativeness. (D3 fix.)
#
# What D1 + D3 change (see llava/model/language_model/modeling_qwen2_5vl_dat.py)
# -----------------------------------------------------------------------------
# D1 (--dat_inject_lr_image True): in addition to the existing answer-Q
# segment, pass-2 ALSO runs HD cross-attention with Q taken from the
# lr_image positions. Same HD K/V (sampled once per b_idx via answer 0's
# intention prediction) is shared between answer-Q and lr_image-Q. After
# this, the LR-image positions in the residual stream carry both their
# original causal-attention output AND a slice of HD content, so any
# downstream qa_prefix -> lr_image attention transfers HD info to the
# answer.
#
# D3 (--dat_use_residual_merge True): replace LSE merge with additive
# residual injection
#     out[q_start:q_end] += sigmoid(hd_gate) * hd_out_proj(out2)
# where ``hd_out_proj`` is a per-DAT-layer Linear(hidden_size, hidden_size)
# initialized to zero (LoRA-style adapter). Contribution at step 0 is
# identically 0 → strict baseline preservation. As ``hd_out_proj.weight``
# escapes 0 via its gradient, HD contribution can grow in any direction
# the gate + projection jointly authorize, independent of lse2 magnitude.
#
# What's NEW vs. Run C (besides D1+D3)
# ------------------------------------
# • F1 OFF (--dat_use_spatial_attn_guide False).
#   The current F1 implementation in _generate_offsets_and_sample is
#     softmax(Q_int_text · Q_lr_image / sqrt(C))  →  spatial saliency
#   This uses ``query_states`` on BOTH sides — Q·Q, not Q·K. ``W_Q`` was
#   trained against ``W_K``, not against itself, so the dot product has
#   no principled "relevance" semantics. Empirically the prompt-conditional
#   offset diagnostic (_diagnose_offsets_prompt.py) shows the offset
#   network is largely prompt-blind even with F1 on. We disable F1 here
#   to (a) remove a noisy/broken branch from the offset graph and (b)
#   leave only one prompt-conditioning channel (intention_branch gate)
#   active, so the ablation is clean.
# • intention_branch STAYS ON (--dat_use_intention_branch True). This is
#   the cleaner prompt-conditioning path: gate = sigmoid(W_proj(Q_int)),
#   used to modulate embed_lr_rep multiplicatively. No Q·Q contract issue.
#
# Everything else inherits from Run C v2 (exp10_full_runC_bf16fix.sh):
#   • Data mix: llava_hr_essential_sa1b_ivcap.json
#   • F4 (hd_input_layernorm RMSNorm + fp32 weights)
#   • bf16 round-off protection on offset-path params
#   • dat_warmup_steps 0 (LoRA trains from step 0 alongside DAT)
#   • dat_hd_gate_init -1.0 (sigmoid ≈ 0.27, but multiplied by 0-init
#     hd_out_proj → effective contribution = 0 at step 0)
#
# Expected wandb signatures (vs. Run C)
# -------------------------------------
# • ``train/loss`` should stay close to Run C's at step 0-50 (zero-init
#   adapter preserves baseline), then bend below as hd_out_proj learns.
# • ``dat/grad_norm`` should be NON-zero on hd_out_proj.* params from
#   step 1 onward (LoRA-style: dL/dW = dL/dout * sigmoid(hd_gate) * out2).
# • ``dat/hd_gate_raw`` evolution becomes interpretable: it now controls
#   the *intensity* of an additive residual update (not a partition
#   weight), so monotone climbs/declines no longer have the LSE-merge
#   confound. We hypothesize a gentle climb if D1's lr_image injection
#   surfaces useful HD signal.
# • HR-Bench downstream: per-layer attn diag predicts L24's peaked HD
#   attention should now actually move the prediction (was suppressed by
#   LSE merge in Run C).

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
EXP_NAME="${EXP_NAME:-0520_full_runE1_D1D3_noSpatial_F4_bf16fix}"

DATA_JSON="${DATA_JSON:-$DATA_ROOT/llava_hr_essential_sa1b_ivcap.json}"

if [[ ! -f "$DATA_JSON" ]]; then
    echo "[ERROR] Missing data file: $DATA_JSON" >&2; exit 1
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

# Full 1D5L pattern (same as Run B/C/D)
DAT_LAYERS="DLLLLLDLLLLLDLLLLLDLLLLLDLLLLLDLLLLL"

torchrun --nproc_per_node=8 --master_port "${MASTER_PORT:-40627}" llava/train/train_qwen_dat.py \
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
    --dat_hd_gate_init -1.0 \
    --dat_warmup_steps 0 \
    --dat_use_residual_merge True \
    --dat_inject_lr_image True \
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
