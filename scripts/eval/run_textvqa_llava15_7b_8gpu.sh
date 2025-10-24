#!/usr/bin/env bash

# 8卡并行：使用LLaVA官方流程复现 LLaVA-1.5 7B 在 TextVQA（目标≈58.2）
# 用法：bash scripts/eval/run_textvqa_llava15_7b_8gpu.sh

set -euo pipefail

# 模型与数据路径（可通过环境变量覆盖）
MODEL_PATH=${MODEL_PATH:-/data/gsva_pretrains/llava-v1_5-7b-hf}
QUESTION_FILE=${QUESTION_FILE:-/data/textvqa/llava_textvqa_val_v051_ocr.jsonl}
ANNOTATION_FILE=${ANNOTATION_FILE:-/data/textvqa/TextVQA_0.5.1_val.json}
IMAGE_FOLDER=${IMAGE_FOLDER:-/data/textvqa/train_images}
OUT_DIR=${OUT_DIR:-/root/ml-fastvlm/textvqa_llava15_7b_results}

# LLaVA-1.5 官方模板
CONV_MODE=${CONV_MODE:-vicuna_v1}

mkdir -p "$OUT_DIR"

NUM_SHARDS=${NUM_SHARDS:-8}

echo "[TextVQA-LLaVA1.5-7B-8GPU] 使用LLaVA官方流程生成分片预测 NUM_SHARDS=$NUM_SHARDS"

# 基础检查
if [ ! -f "$QUESTION_FILE" ]; then
  echo "错误: 问题文件不存在: $QUESTION_FILE"; exit 1
fi
if [ ! -d "$IMAGE_FOLDER" ]; then
  echo "错误: 图像目录不存在: $IMAGE_FOLDER"; exit 1
fi
if [ ! -f "$ANNOTATION_FILE" ]; then
  echo "错误: 标注文件不存在: $ANNOTATION_FILE"; exit 1
fi

echo "开始8卡并行预测（使用LLaVA model_vqa_loader）..."
for ((i=0; i<NUM_SHARDS; i++)); do
  echo "启动GPU $i..."
  CUDA_VISIBLE_DEVICES=$i \
  stdbuf -oL -eL python -m llava.eval.model_vqa_loader \
    --model-path "$MODEL_PATH" \
    --question-file "$QUESTION_FILE" \
    --image-folder "$IMAGE_FOLDER" \
    --answers-file "$OUT_DIR/llava-v1.5-7b.s${i}.jsonl" \
    --temperature 0 \
    --conv-mode "$CONV_MODE" \
    --num-chunks $NUM_SHARDS \
    --chunk-idx $i \
    2>&1 | sed -u "s/^/[GPU ${i}] /" | tee "$OUT_DIR/textvqa_s${i}.log" &
done

echo "等待所有GPU完成..."
wait

echo "检查生成的文件..."
ls -la "$OUT_DIR"/llava-v1.5-7b.s*.jsonl || echo "警告: 没有找到预测文件"

echo "合并预测..."
cat "$OUT_DIR"/llava-v1.5-7b.s*.jsonl > "$OUT_DIR"/llava-v1.5-7b.jsonl

echo "使用LLaVA官方评估器计算TextVQA分数..."
python -m llava.eval.eval_textvqa \
  --annotation-file "$ANNOTATION_FILE" \
  --result-file "$OUT_DIR/llava-v1.5-7b.jsonl"

echo "[TextVQA-LLaVA1.5-7B-8GPU] 完成"


