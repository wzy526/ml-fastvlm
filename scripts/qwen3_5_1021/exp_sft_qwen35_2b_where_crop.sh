#!/usr/bin/env bash
set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-fastvlm}"

# 1021 Phase 1: where-head + crop readout, trained jointly (Qwen3.5-2B DAT).
# ============================================================================
#
# Background (V* 191 MCQ, LR 640 tokens): no HD 64.4; frozen scripts/where_head.py
# head -> crop as a 2nd image 77.5 (where_crop_generate); same pipeline with
# the GT box centre 83.8. The 1600-token head covers 85% but still answers
# 77.0-77.5: the remaining gap is 37 far misses whose targets are < half an LR
# cell. Native DAT reads the same LR features (equally blind) and its K/V
# readout is lossier than a re-encoded crop. Accepted framing: the sampler
# decides where the HD tokens go; the readout is a crop in the sequence.
#
# What this run trains (train_qwen_dat.py, dat_where_*):
#   - the head is a model submodule (dat_extra_args['where_head'], saved with
#     the checkpoint), initialised from where_lm8_mix.pt, trained on the LM's
#     layer-8 LR image-token states with the cell-mask loss (trunk detached);
#   - bbox samples carry [LR, question, hint, crop, question] where the crop is
#     exactly the --oracle_box window (GT centre -> 4 LR cells -> x2 pad,
#     >= 20 HD cells -> original pixels -> <= 400 tokens); the LM (LoRA) learns
#     to answer from it; where_only samples (coco_small / refcoco: boxes
#     without answers) supervise the head only (labels masked);
#   - plain SFT samples are unchanged (no HD, no crop); pixel_values_hd is not
#     emitted at all, the K/V bypass stays frozen (dat_trainable_keys=where_head).
#
# Data (compose on the cluster, ~80k; 1000 steps x 64 = 64k seen):
#   python scripts/compose_sft_mix.py \
#       --base $OSS_DATA/llava_hr_gen_vs_0817.json --base_max_n 15000 \
#       --add $OSS_DATA/extra_0916/viscot_bbox.train.json --max_n 40000 \
#       --add $OSS_DATA/extra_1018/coco_small_bbox.json --max_n 15000 --where_only 1 \
#       --add $OSS_DATA/extra_1018/refcoco_bbox.json     --max_n 10000 --where_only 1 \
#       --out $OSS_DATA/extra_1021/where_crop_mix.json
#
# Run (8 GPUs, ~1000 steps):
#   EXP_NAME=1021_where_crop_p1 bash scripts/qwen3_5_1021/exp_sft_qwen35_2b_where_crop.sh
# wandb: where/loss (head loss; frozen head starts ~1.5-3 on viscot), where/hit
# (top-1 cell in the GT cells; 640 head ~0.98 held-out viscot, ~0.76 V*),
# where/n (masked samples per forward, ~0.8 x batch), train/loss (LM CE on
# viscot + base).
# Pass (scripts/where_crop_eval.py --head_prefix train on the merged ckpt):
#   V* > 77.5 (frozen head) and --oracle_box >= 83.8 (readout not degraded);
#   coverage from the box json vs 76% / 85%.

export WANDB_PROJECT="${WANDB_PROJECT:-vldat_experiments}"
export NUMEXPR_MAX_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

OSS_DATA="${OSS_DATA:-/data/oss_bucket_0/wangziyi/models_data}"
LOCAL_ROOT="${LOCAL_ROOT:-/home/pingping.wzy}"
CKPT_ROOT="${CKPT_ROOT:-$LOCAL_ROOT/vldat_experiments}"
MODEL_PATH="${MODEL_PATH:-$CKPT_ROOT/0924_miniE1_2k/checkpoint-1000}"
WHERE_INIT="${WHERE_INIT:-$CKPT_ROOT/0924_miniE1_2k/loc/where_lm8_mix.pt}"
CACHE_ROOT="${CACHE_ROOT:-$LOCAL_ROOT/cache/vldat}"
EXP_NAME="${EXP_NAME:-1021_where_crop_p1}"
IMAGE_ROOT="${IMAGE_ROOT:-$LOCAL_ROOT/sft_data}"
DATA_JSON="${DATA_JSON:-$OSS_DATA/extra_1021/where_crop_mix.json}"

export DAT_EXACT_MERGE_GRAD=1
export DAT_ATTN_BACKEND="${DAT_ATTN_BACKEND:-auto}"

for f in "$DATA_JSON" "$MODEL_PATH/config.json" "$WHERE_INIT"; do
    if [[ ! -f "$f" ]]; then echo "[ERROR] Missing $f" >&2; exit 1; fi
done
if [[ ! -d "$IMAGE_ROOT/train_split" ]]; then echo "[ERROR] Missing $IMAGE_ROOT/train_split" >&2; exit 1; fi

python - <<'PY'
import transformers
print(f"[preflight] transformers {transformers.__version__}")
from transformers import Qwen3_5ForConditionalGeneration  # noqa: F401
from llava.model.language_model import modeling_qwen3_5_dat as M
assert M.WHERE_HINT, "where-head code missing (git pull?)"
print("[preflight] where-head modeling present")
PY

mkdir -p "$CKPT_ROOT/$EXP_NAME"

export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache_$USER/train-$(hostname -s)}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$CACHE_ROOT/torchinductor-$(hostname -s)}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$CACHE_ROOT/cuda}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT/xdg}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export DEEPSPEED_TIMEOUT="${DEEPSPEED_TIMEOUT:-7200}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-7200}"

echo "[1021-where] model=$MODEL_PATH head_init=$WHERE_INIT data=$(basename "$DATA_JSON") crop_prob=${WHERE_CROP_PROB:-1.0}"

# DAT architecture args come from the checkpoint config (dat_inherit_ckpt_args);
# the CLI dat_* below are placeholders that inheritance overrides.
torchrun --nproc_per_node="${NPROC:-8}" --master_port "${MASTER_PORT:-40995}" llava/train/train_qwen_dat.py \
    --deepspeed ./scripts/zero_configs/zero2.json \
    --model_name_or_path "$MODEL_PATH" \
    --model_family qwen3_5 \
    --data_path "$DATA_JSON" \
    --image_folder "$IMAGE_ROOT/train_split" \
    --use_hr_first_resize False \
    --lr_min_pixels 1024 \
    --lr_max_pixels "${LR_MAX_PIXELS:-655360}" \
    --use_dat True \
    --dat_inherit_ckpt_args True \
    --dat_layers auto \
    --dat_grid_size 20 \
    --dat_off_grps 8 \
    --dat_inter_size 128 \
    --dat_hr_scale 3 \
    --dat_hd_proj True \
    --dat_freeze_base False \
    --dat_warmup_steps 0 \
    --dat_trainable_keys "${DAT_TRAINABLE:-where_head}" \
    --dat_where_head True \
    --dat_where_layer "${WHERE_LAYER:-8}" \
    --dat_where_loss_weight "${WHERE_LOSS_W:-1.0}" \
    --dat_where_trunk_grad "${WHERE_TRUNK_GRAD:-0}" \
    --dat_where_init "$WHERE_INIT" \
    --dat_where_crop_prob "${WHERE_CROP_PROB:-1.0}" \
    --dat_where_out_cells 4 \
    --dat_where_pad 2.0 \
    --dat_where_min_cells 20 \
    --dat_where_crop_tok 400 \
    --dat_where_max_box_frac "${WHERE_MAX_BOX_FRAC:-0.25}" \
    --dat_lr "${DAT_LR:-1e-4}" \
    --lora_enable "${LORA_ENABLE:-True}" \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_target_layers all \
    --lora_lr "${LORA_LR:-2e-5}" \
    --tune_mm_vision False \
    --tune_mm_mlp "${TUNE_MM_MLP:-False}" \
    --tune_mm_llm False \
    --mm_projector_lr 5e-6 \
    --kd_on False \
    --bf16 True \
    --tf32 True \
    --max_grad_norm 1.0 \
    --ddp_timeout 7200 \
    --output_dir "$CKPT_ROOT/$EXP_NAME" \
    --num_train_epochs 1 \
    --max_steps "${MAX_STEPS:-1000}" \
    --per_device_train_batch_size "${PER_DEVICE_BATCH:-4}" \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps "${GRAD_ACCUM:-2}" \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps "${SAVE_STEPS:-250}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT:-2}" \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_steps "${WARMUP_STEPS:-30}" \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --group_by_modality_length True \
    --dataloader_num_workers "${DATALOADER_WORKERS:-8}" \
    --dataloader_pin_memory True \
    --dataloader_prefetch_factor 2 \
    --dataloader_persistent_workers True \
    --dataloader_drop_last True \
    --seed "${SEED:-42}" \
    --report_to "${REPORT_TO:-wandb}" \
    --run_name "$EXP_NAME"

source "$(dirname "${BASH_SOURCE[0]}")/../qwen2_5vl_adl_0701/_merge_after_train.sh"
