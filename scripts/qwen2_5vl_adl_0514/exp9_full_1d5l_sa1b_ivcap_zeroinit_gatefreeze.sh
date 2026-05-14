#!/usr/bin/env bash
set -euo pipefail

# Activate vldat conda env
eval "$(conda shell.bash hook)"
conda activate vldat

# Exp 9 (0514 re-run): Full 1D5L + SA-1B IV-cap,
#                     zero-init HD KV + frozen hd_gate (diagnostic).
#
# Identical to scripts/qwen2_5vl_adl_0514/exp9_full_1d5l_sa1b_ivcap_zeroinit.sh,
# except:
#   1. --dat_hd_gate_freeze True
#        hd_gate is created with requires_grad=False and stays at -1.0
#        for the entire run. The WandbDATMonitorCallback still logs
#        hd_gate_raw / hd_gate_sigmoid; they should be perfectly flat.
#   2. EXP_NAME / MASTER_PORT bumped so this run does not collide.
#
# What this experiment is for — controlled diagnostic of the zero-init-V
# cold start. In the zeroinit run (gate trainable, init=-1.0) we observe:
#
#   • dat/hd_gate_raw drifts very slowly (≈ −2e-5 / step) toward more
#     negative values, i.e. the gate is closing — but on a timescale
#     3 000× slower than the previous "copy-from-k_proj" runs.
#
# Interpretation: with V=0, the HD path contributes 0 strictly but the
# LSE merge still scales out1 by (1 − w₂) ≈ 0.97. The LLM was pretrained
# without this scaling, so CE has a small structural pressure pushing
# w₂ → 0  ⇔  hd_gate → −∞. The pressure is tiny because w₂(1−w₂) ≈ 0.026
# is small at this gate setting, but it's persistent. Meanwhile v_proj_hd
# is growing out of zero (kvhd_weight_norm climbing) but its
# contribution to the merge is still negligible early on.
#
# Freezing hd_gate removes the "close the door" escape route entirely:
# v_proj_hd cannot rely on the gate to dampen its own bad updates and
# must learn HD content that *actually reduces loss*. This isolates the
# question "is the bottleneck (a) hd_gate trying to escape, or (b)
# v_proj_hd not yet learning useful HD content?".
#
# Expected wandb signatures (vs the trainable-gate zeroinit run):
#   • dat/hd_gate_raw       = -1.0     ± 0   (exactly flat, frozen)
#   • dat/hd_gate_sigmoid   = 0.2689   ± 0
#   • dat/kvhd_weight_norm  = growing (same or faster than zeroinit run)
#   • dat/kvhd_grad_norm    = comparable magnitude
#   • train/loss            = comparable or slightly higher early on
#                              (no escape valve), but the *eval* HR-Bench
#                              should be the deciding metric
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
EXP_NAME="${EXP_NAME:-0514_full_1d5l_sa1b_ivcap_zeroinit_gatefreeze}"

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

torchrun --nproc_per_node=8 --master_port "${MASTER_PORT:-40616}" llava/train/train_qwen_dat.py \
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
    --dat_hd_gate_freeze True \
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
