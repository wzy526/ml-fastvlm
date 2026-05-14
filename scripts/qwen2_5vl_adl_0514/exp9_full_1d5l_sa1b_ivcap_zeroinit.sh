#!/usr/bin/env bash
set -euo pipefail

# Activate vldat conda env
eval "$(conda shell.bash hook)"
conda activate vldat

# Exp 9 (0514 re-run): Full 1D5L + SA-1B IV-cap, zero-init HD KV.
#
# Identical to scripts/qwen2_5vl_adl_0514/exp9_full_1d5l_sa1b_ivcap_hdgate-2.sh,
# except:
#   1. k_proj_hd / v_proj_hd are now zero-init adapter style (K=Kaiming, V=0),
#      see Qwen2_5_VLAttentionDAT._init_hd_proj_weights /
#      Qwen2_5_VLDATForConditionalGeneration.init_hd_proj_from_kv in
#      llava/model/language_model/modeling_qwen2_5vl_dat.py.
#   2. --dat_hd_gate_init -1.0   (was -2.0; with V=0 HD output is no longer
#      noise, so the gate doesn't need to be aggressively suppressed).
#   3. EXP_NAME / MASTER_PORT bumped so this run does not collide.
#
# Background — why the previous runs degraded:
#
# In hdgate-{-2,-4} we copied pretrained k_proj / v_proj weights into
# k_proj_hd / v_proj_hd ("init_hd_proj_from_kv"). For DAT layer L, k_proj
# was trained to map *layer-L hidden states* into K-space, but k_proj_hd
# is applied to image_hd_features = the visual merger's pooler_output,
# which lives in layer-0 input embedding space. For deep DAT layers
# (L=30) this is a ~30-layer domain mismatch — Q · K_hd^T has no
# geometric meaning, out2 is pure noise, and the LSE merge pushes
# hd_gate down every time it opens enough for CE to "feel" the noise:
#
#   hd_gate_init=-2  →  monotone descent over the first 100 steps
#                       (HD ≈ 12% of the merge weight, decisively bad)
#   hd_gate_init=-8  →  stuck (HD ≈ 0.03%, gate gradient lost in noise)
#
# Fix — zero-init adapter pattern (LoRA-style):
#
#   K_hd ← Kaiming normal (nonlinearity='linear')
#         → provides symmetry-breaking statistics, no OOD bias
#   V_hd ← 0
#         → out2 = softmax(Q·K_hd^T) · 0 ≡ 0 at step 0 (HD = strict no-op)
#   hd_gate ← -1.0  (σ ≈ 0.27, log σ ≈ -1.31)
#         → mild perturbation on out1 from the (1 − w₂) scaling
#           (w₂ ~ a few % at this gate setting, with lse2 from random K)
#         → enough gate aperture for v_proj_hd's gradient signal to be
#           clearly above bf16 round-off noise
#
# v_proj_hd still receives non-zero gradients via the LSE merge
# (dL/dV_hd ∝ softmax(...) · w₂ · dL/dout_merged), so the path can only
# grow toward directions that reduce loss. K_hd's Kaiming init breaks
# the rank-1 transient that would occur with strict K=V=0 init.
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
EXP_NAME="${EXP_NAME:-0514_full_1d5l_sa1b_ivcap_zeroinit}"

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

torchrun --nproc_per_node=8 --master_port "${MASTER_PORT:-40615}" llava/train/train_qwen_dat.py \
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
