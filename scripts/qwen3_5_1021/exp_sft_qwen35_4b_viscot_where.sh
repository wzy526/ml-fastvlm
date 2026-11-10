#!/usr/bin/env bash
# 4B, one job, two phases in series. Phase B starts only after the merge.
# Phase A: LoRA on official Qwen3.5-4B. Data is the boxed Visual-CoT set only
#          (extra_0916/viscot_bbox.train.json). One image + question + short
#          answer. q/k/v/o, r=8, alpha=16, lr=2e-5. Projector lr=5e-6. No DAT layers.
# Phase B: freeze that model, train the where-head on layer 8. Data is Visual-CoT
#          plus RefCOCO plus COCO-small. Same mix that took the 2B head from
#          66.5% to 79.1% window hit.
#
#   bash scripts/qwen3_5_1021/exp_sft_qwen35_4b_viscot_where.sh
#
# Writes, under $CKPT_ROOT/$EXP_NAME-merged:
#   the merged language model
#   where_lm8_ce20.pt
#   ldat_zero.pt          zero LDatHead, so a later eval starts from the where-head density
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# shellcheck disable=SC1091
source "${CONDA_SH:-$HOME/miniconda3/bin/activate}" "${CONDA_ENV:-$HOME/.conda/envs/fastvlm}"

export WANDB_PROJECT="${WANDB_PROJECT:-vldat_experiments}"
export NUMEXPR_MAX_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

OSS_DATA="${OSS_DATA:-/data/oss_bucket_0/wangziyi/models_data}"
CKPT_ROOT="${CKPT_ROOT:-$HOME/vldat_experiments}"
MODEL_PATH="${MODEL_PATH:-/data/oss_bucket_0/wangziyi/official_ckpt/Qwen3.5-4B}"
EXP_NAME="${EXP_NAME:-1110_4b_viscot_lora}"
IMAGE_ROOT="${IMAGE_ROOT:-$HOME/sft_data}"
MERGED_DIR="${MERGED_DIR:-$CKPT_ROOT/${EXP_NAME}-merged}"
VSTAR_ROOT="${VSTAR_ROOT:-$HOME/data/vstar_bench}"

DATA_JSON="${DATA_JSON:-}"
if [[ -z "$DATA_JSON" ]]; then
    for c in \
        "$OSS_DATA/extra_0916/viscot_bbox.train.json" \
        "$OSS_DATA/extra_0916/viscot_bbox.json"
    do
        if [[ -f "$c" ]]; then DATA_JSON="$c"; break; fi
    done
fi

export DAT_EXACT_MERGE_GRAD=1
export DAT_ATTN_BACKEND="${DAT_ATTN_BACKEND:-auto}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache_${USER}/train-$(hostname -s)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export DEEPSPEED_TIMEOUT="${DEEPSPEED_TIMEOUT:-7200}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-7200}"
mkdir -p "$TRITON_CACHE_DIR" "$CKPT_ROOT/$EXP_NAME"

REF_JSON="${REF_JSON:-$OSS_DATA/extra_1018/refcoco_bbox.json}"
COCO_JSON="${COCO_JSON:-$OSS_DATA/extra_1018/coco_small_bbox.json}"
if [[ -z "$DATA_JSON" || ! -f "$DATA_JSON" ]]; then
    echo "[ERROR] boxed Visual-CoT json not found." >&2
    echo "        looked for extra_0916/viscot_bbox.train.json and viscot_bbox.json under $OSS_DATA" >&2
    exit 1
fi
if [[ ! -f "$REF_JSON" || ! -f "$COCO_JSON" ]]; then
    echo "[ERROR] where-head json missing: $REF_JSON or $COCO_JSON" >&2
    exit 1
fi
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
    echo "[ERROR] Missing $MODEL_PATH/config.json" >&2
    exit 1
fi
if [[ ! -d "$IMAGE_ROOT/train_split" ]]; then
    echo "[ERROR] Missing $IMAGE_ROOT/train_split" >&2
    exit 1
fi

N_ROWS="$(python - "$DATA_JSON" << 'PY'
import json, sys
data = json.load(open(sys.argv[1]))
nbox = sum(isinstance(s.get("bbox"), list) and len(s["bbox"]) == 4 for s in data)
print(len(data), nbox)
PY
)"
N_ALL="${N_ROWS%% *}"
N_BOX="${N_ROWS##* }"
if (( N_ALL > 80000 )); then
    echo "[ERROR] $DATA_JSON has $N_ALL rows. This run wants the ~40k boxed Visual-CoT file." >&2
    exit 1
fi
if (( N_BOX < 1000 )); then
    echo "[ERROR] $DATA_JSON has only $N_BOX boxed rows." >&2
    exit 1
fi

# One pass. Effective batch is per-device x accum x 8 GPUs (default 2 x 4 x 8 = 64).
EFF=$(( ${PER_DEVICE_BATCH:-2} * ${GRAD_ACCUM:-4} * ${NPROC:-8} ))
ONE_PASS=$(( (N_ALL + EFF - 1) / EFF ))
MAX_STEPS="${MAX_STEPS:-$ONE_PASS}"

echo "[1110-4b] data=$DATA_JSON rows=$N_ALL boxed=$N_BOX"
echo "[1110-4b] lora steps=$MAX_STEPS effective_batch=$EFF -> $CKPT_ROOT/$EXP_NAME"
echo "[1110-4b] merged -> $MERGED_DIR"

if [[ -f "$MERGED_DIR/config.json" && "${FORCE:-0}" != 1 ]]; then
    echo "[1110-4b] merged checkpoint already present, skipping LoRA"
else

torchrun --nproc_per_node="${NPROC:-8}" --master_port "${MASTER_PORT:-40997}" llava/train/train_qwen_dat.py \
    --deepspeed ./scripts/zero_configs/zero2.json \
    --model_name_or_path "$MODEL_PATH" \
    --model_family qwen3_5 \
    --data_path "$DATA_JSON" \
    --image_folder "$IMAGE_ROOT/train_split" \
    --use_dat False \
    --lr_min_pixels 1024 \
    --lr_max_pixels "${LR_MAX_PIXELS:-655360}" \
    --lora_enable True \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_target_layers qkvo \
    --lora_lr "${LORA_LR:-2e-5}" \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm False \
    --mm_projector_lr 5e-6 \
    --bf16 True \
    --tf32 True \
    --max_grad_norm 1.0 \
    --ddp_timeout 7200 \
    --output_dir "$CKPT_ROOT/$EXP_NAME" \
    --num_train_epochs 1 \
    --max_steps "$MAX_STEPS" \
    --per_device_train_batch_size "${PER_DEVICE_BATCH:-2}" \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps "${GRAD_ACCUM:-4}" \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps "$MAX_STEPS" \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_steps "${WARMUP_STEPS:-20}" \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 4096 \
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

if [[ ! -f "$CKPT_ROOT/$EXP_NAME/adapter_config.json" ]]; then
    echo "[ERROR] LoRA adapter missing under $CKPT_ROOT/$EXP_NAME" >&2
    exit 1
fi

MODEL_PATH="$MODEL_PATH" LORA_PATH="$CKPT_ROOT/$EXP_NAME" MERGED_DIR="$MERGED_DIR" python - << 'PY'
import json, os, torch
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from peft import PeftModel

base_path = os.environ["MODEL_PATH"]
lora_path = os.environ["LORA_PATH"]
out_path = os.environ["MERGED_DIR"]
os.makedirs(out_path, exist_ok=True)
base = Qwen3_5ForConditionalGeneration.from_pretrained(
    base_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
nlt_path = os.path.join(lora_path, "non_lora_trainables.bin")
raw = torch.load(nlt_path, map_location="cpu")
cleaned = {(k[11:] if k.startswith("base_model.") else k): v for k, v in raw.items()}
if any(k.startswith("model.model.") for k in cleaned):
    cleaned = {(k[6:] if k.startswith("model.") else k): v for k, v in cleaned.items()}
missing, unexpected = base.load_state_dict(cleaned, strict=False)
print(f"[merge] projector tensors={len(cleaned)} unexpected={len(unexpected)}")
if unexpected:
    raise SystemExit("[merge] unexpected keys: " + ", ".join(unexpected[:8]))
model = PeftModel.from_pretrained(base, lora_path)
model = model.merge_and_unload()
model.save_pretrained(out_path, safe_serialization=True)
processor = AutoProcessor.from_pretrained(base_path, trust_remote_code=True, use_fast=True)
processor.save_pretrained(out_path)
for attr_name in ("image_processor", "video_processor", "feature_extractor"):
    sub = getattr(processor, attr_name, None)
    if sub is not None and hasattr(sub, "save_pretrained"):
        sub.save_pretrained(out_path)
chat = getattr(processor, "chat_template", None)
if isinstance(chat, str):
    with open(os.path.join(out_path, "chat_template.json"), "w", encoding="utf-8") as f:
        json.dump({"chat_template": chat}, f, ensure_ascii=False, indent=2)
print("[merge]", out_path)
PY

fi

echo "[1110-4b] where-head on $MERGED_DIR"
echo "[1110-4b] where data: $DATA_JSON + $REF_JSON + $COCO_JSON"
torchrun --nproc_per_node="${NPROC:-8}" --master_port "${WHERE_PORT:-40998}" scripts/where_head.py \
    --model_path "$MERGED_DIR" \
    --train_json "$DATA_JSON" "$REF_JSON" "$COCO_JSON" \
    --image_root "$IMAGE_ROOT/train_split" \
    --max_per_json 40000 40000 60000 \
    --heldout 0 \
    --steps "${WHERE_STEPS:-1500}" \
    --accum 4 \
    --lr 1e-4 \
    --d 384 \
    --dilate 0 \
    --ce_weight 20 \
    --head xattn \
    --xlayers 2 \
    --xheads 6 \
    --feat lm:8 \
    --lr_tok 640 \
    --eval_every 200 \
    --vstar_root "$VSTAR_ROOT" \
    --vstar_n 191 \
    --save "$MERGED_DIR/where_lm8_ce20.pt"

MERGED_DIR="$MERGED_DIR" python - << 'PY'
import os, sys, torch
from transformers import AutoConfig
sys.path.insert(0, "scripts")
from offset_feat_train import LDatHead
cfg = AutoConfig.from_pretrained(os.environ["MERGED_DIR"])
hidden = cfg.text_config.hidden_size
head = LDatHead(hidden, d=256)
out = os.path.join(os.environ["MERGED_DIR"], "ldat_zero.pt")
torch.save(head.state_dict(), out)
print("[ldat-zero]", out, "hidden", hidden)
PY

echo "[1110-4b] done $MERGED_DIR"
