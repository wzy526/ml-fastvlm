#!/usr/bin/env bash

# 8卡并行：复现 LLaVA-1.5 7B 在 TextVQA with OCR（目标≈58.2）
# 用法：bash scripts/eval/run_textvqa_llava15_7b_8gpu.sh

set -euo pipefail

# 模型与数据路径（可通过环境变量覆盖）
MODEL_PATH=${MODEL_PATH:-/home/zhuofan.xia/gsva_pretrains/llava-v1_5-7b}
QUESTION_FILE=${QUESTION_FILE:-/perception-hl/zhuofan.xia/data/textvqa/val_questions.json}
ANNOTATION_FILE=${ANNOTATION_FILE:-/perception-hl/zhuofan.xia/data/textvqa/val_annotations.json}
IMAGE_FOLDER=${IMAGE_FOLDER:-/perception-hl/zhuofan.xia/data/textvqa/train_images}
OCR_FILE=${OCR_FILE:-/perception-hl/zhuofan.xia/data/textvqa/val_ocr_tokens.json}
OUT_DIR=${OUT_DIR:-/perception-hl/zhuofan.xia/vlm_exps/textvqa_llava15_7b_ocr}

# LLaVA-1.5 官方模板
CONV_MODE=${CONV_MODE:-llava_v1}

mkdir -p "$OUT_DIR"

NUM_SHARDS=${NUM_SHARDS:-8}

echo "[TextVQA-LLaVA1.5-7B-8GPU] 生成分片预测 NUM_SHARDS=$NUM_SHARDS"

# 基础检查
if [ ! -f "$QUESTION_FILE" ]; then
  echo "错误: 问题文件不存在: $QUESTION_FILE"; exit 1
fi
if [ ! -d "$IMAGE_FOLDER" ]; then
  echo "错误: 图像目录不存在: $IMAGE_FOLDER"; exit 1
fi

echo "开始8卡并行预测..."
for ((i=0; i<NUM_SHARDS; i++)); do
  echo "启动GPU $i..."
  CUDA_VISIBLE_DEVICES=$i \
  stdbuf -oL -eL python -u eval_textvqa_official.py \
    --model-path "$MODEL_PATH" \
    --conv-mode "$CONV_MODE" \
    --question-file "$QUESTION_FILE" \
    --annotation-file "$ANNOTATION_FILE" \
    --image-folder "$IMAGE_FOLDER" \
    --ocr-file "$OCR_FILE" \
    --output-file "$OUT_DIR/textvqa_val_pred.s${i}.jsonl" \
    --chunks $NUM_SHARDS \
    --chunk-idx $i \
    2>&1 | sed -u "s/^/[GPU ${i}] /" | tee "$OUT_DIR/textvqa_s${i}.log" &
done

echo "等待所有GPU完成..."
wait

echo "检查生成的文件..."
ls -la "$OUT_DIR"/textvqa_val_pred.s*.jsonl || echo "警告: 没有找到预测文件"

echo "合并预测..."
cat "$OUT_DIR"/textvqa_val_pred.s*.jsonl > "$OUT_DIR"/textvqa_val_pred.jsonl

echo "基于合并预测计算最终分数（TextVQA soft-accuracy）..."
ANNOTATION_FILE="$ANNOTATION_FILE" OUT_DIR="$OUT_DIR" python -u - << 'PY'
import os
from eval_textvqa_official import eval_single_official

annotation_file = os.environ["ANNOTATION_FILE"]
out_dir = os.environ["OUT_DIR"]
result_file = os.path.join(out_dir, "textvqa_val_pred.jsonl")
acc = eval_single_official(annotation_file, result_file)
print(f"Final Accuracy: {acc*100:.2f}%")
PY

echo "[TextVQA-LLaVA1.5-7B-8GPU] 完成"


