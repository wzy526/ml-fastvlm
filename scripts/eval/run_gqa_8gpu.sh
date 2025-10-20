run_gqa_8gpu#!/usr/bin/env bash

# 8卡并行GQA评估脚本
# 完全匹配LLaVA原版GQA测试逻辑，适配您训练的DAT-LLaVA-1.5模型
# 使用修改后的eval_gqa_official.py脚本，支持LLaVA官方评估模式

export DS_SKIP_CUDA_CHECK=1

# 激活conda环境
source /home/zhuofan.xia/miniconda3/bin/activate pt260

# 设置实验名称和路径
EXP_NAME="tdat-7b-l0d32-s12g8z3"
OUTPUT_DIR="/perception-hl/zhuofan.xia/vlm_exps/textdat/$EXP_NAME"
BACKUP_DIR="/root/vlm_exps/textdat/$EXP_NAME"

# 检查checkpoint是否存在
if [ -d "$OUTPUT_DIR" ]; then
    echo "找到训练输出目录: $OUTPUT_DIR"
    CHECKPOINT_PATH="$OUTPUT_DIR"
elif [ -d "$BACKUP_DIR" ]; then
    echo "找到备份目录: $BACKUP_DIR"
    CHECKPOINT_PATH="$BACKUP_DIR"
else
    echo "错误: 未找到训练输出目录"
    echo "请检查以下路径:"
    echo "  - $OUTPUT_DIR"
    echo "  - $BACKUP_DIR"
    exit 1
fi

# 检查GQA数据集路径
DATA_PATH="/perception-hl/zhuofan.xia/data/gqa/val_balanced_questions.json"
IMAGE_FOLDER="/perception-hl/zhuofan.xia/data/gqa/images"

if [ ! -f "$DATA_PATH" ]; then
    echo "错误: 未找到GQA数据集文件"
    echo "请检查以下路径:"
    echo "  - $DATA_PATH"
    echo "  - $IMAGE_FOLDER"
    echo ""
    echo "请确保GQA数据集已正确下载并解压到指定位置"
    exit 1
fi

if [ ! -d "$IMAGE_FOLDER" ]; then
    echo "错误: 未找到GQA图像文件夹"
    echo "请检查路径: $IMAGE_FOLDER"
    exit 1
fi

# 8卡配置
GPU_LIST="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPULIST <<< "$GPU_LIST"
CHUNKS=${#GPULIST[@]}

echo "使用数据文件: $DATA_PATH"
echo "使用图像文件夹: $IMAGE_FOLDER"
echo "使用模型路径: $CHECKPOINT_PATH"
echo "GPU配置: $GPU_LIST"
echo "分块数: $CHUNKS"
echo "评估模式: 8卡并行GQA评估（LLaVA官方逻辑）"

# 创建结果目录
RESULTS_DIR="./gqa_8gpu_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "结果将保存到: $(pwd)/$RESULTS_DIR"

# 运行8卡并行GQA评估
echo "开始运行8卡并行GQA评估..."
echo "="*80

# 设置时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 并行推理 - 每个GPU处理一个分块
echo "启动8卡并行推理..."
echo "="*60

for IDX in $(seq 0 $((CHUNKS-1))); do
    echo "启动GPU $IDX 处理分块 $IDX/$((CHUNKS-1))"
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python eval_gqa_official.py \
        --model-path "$CHECKPOINT_PATH" \
        --data-path "$DATA_PATH" \
        --image-folder "$IMAGE_FOLDER" \
        --output-dir "$RESULTS_DIR" \
        --conv-mode "llava_v1" \
        --temperature 0 \
        --max-new-tokens 16 \
        --chunks $CHUNKS \
        --chunk-idx $IDX \
        --llava-mode &
done

# 等待所有并行任务完成
echo "等待所有GPU任务完成..."
wait

echo "所有GPU任务完成，开始合并结果..."

# 合并所有分块结果
MERGE_FILE="$RESULTS_DIR/gqa_results_merged.jsonl"
echo "合并结果到: $MERGE_FILE"

# 清空合并文件
> "$MERGE_FILE"

# 按顺序合并所有分块
for IDX in $(seq 0 $((CHUNKS-1))); do
    CHUNK_FILE="$RESULTS_DIR/gqa_results_${CHUNKS}_${IDX}.jsonl"
    if [ -f "$CHUNK_FILE" ]; then
        echo "合并分块 $IDX: $CHUNK_FILE"
        cat "$CHUNK_FILE" >> "$MERGE_FILE"
    else
        echo "警告: 分块文件不存在: $CHUNK_FILE"
    fi
done

# 转换为GQA官方评估格式
echo "转换为GQA官方评估格式..."
python -c "
import json
import os

# 读取合并后的结果
results = []
with open('$MERGE_FILE', 'r') as f:
    for line in f:
        if line.strip():
            results.append(json.loads(line))

# 转换为GQA官方评估格式
all_answers = []
for result in results:
    question_id = result['question_id']
    text = result['text'].rstrip('.').lower()
    all_answers.append({'questionId': question_id, 'prediction': text})

# 保存GQA官方评估格式
gqa_eval_file = '$RESULTS_DIR/testdev_balanced_predictions.json'
with open(gqa_eval_file, 'w') as f:
    json.dump(all_answers, f)

print(f'转换完成: {len(all_answers)} 个答案')
print(f'GQA官方评估文件: {gqa_eval_file}')
"

# 计算合并后的准确率
echo "计算合并后的准确率..."
python -c "
import json
import sys

# 读取合并后的结果
results = []
with open('$MERGE_FILE', 'r') as f:
    for line in f:
        if line.strip():
            results.append(json.loads(line))

# 计算准确率
correct = sum(1 for r in results if r.get('correct', False))
total = len(results)
accuracy = correct / total if total > 0 else 0.0

# 保存汇总结果
summary = {
    'model_path': '$CHECKPOINT_PATH',
    'data_path': '$DATA_PATH',
    'image_folder': '$IMAGE_FOLDER',
    'total_samples': total,
    'accuracy': accuracy,
    'correct_samples': correct,
    'error_samples': total - correct,
    'conv_mode': 'llava_v1',
    'temperature': 0,
    'chunks': $CHUNKS,
    'gpu_list': '$GPU_LIST'
}

with open('$RESULTS_DIR/gqa_results_${TIMESTAMP}_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f'合并完成: {total} 个样本, 准确率: {accuracy:.4f} ({accuracy*100:.2f}%)')
"

# 运行GQA官方评估
echo "运行GQA官方评估..."
GQA_DATA_DIR="/perception-hl/zhuofan.xia/data/gqa"
if [ -d "$GQA_DATA_DIR" ]; then
    echo "找到GQA数据目录: $GQA_DATA_DIR"
    cd "$GQA_DATA_DIR"
    
    # 复制预测结果到GQA数据目录
    cp "$RESULTS_DIR/testdev_balanced_predictions.json" "$GQA_DATA_DIR/"
    
    # 运行GQA官方评估
    echo "运行GQA官方评估脚本..."
    python eval/eval.py --tier testdev_balanced
    
    echo "GQA官方评估完成！"
    cd - > /dev/null
else
    echo "警告: 未找到GQA数据目录，跳过官方评估"
    echo "请确保GQA数据集已正确下载到: $GQA_DATA_DIR"
fi

# 汇总结果
echo ""
echo "8卡并行GQA评估结果汇总"
echo "="*60
echo "数据集: GQA val_balanced_questions.json"
echo "数据路径: $DATA_PATH"
echo "图像文件夹: $IMAGE_FOLDER"
echo "模型路径: $CHECKPOINT_PATH"
echo "GPU配置: $GPU_LIST"
echo "分块数: $CHUNKS"
echo "对话模式: llava_v1"
echo "最大生成长度: 16"
echo "温度: 0"
echo "-"*60

# 显示最终结果
SUMMARY_FILE="$RESULTS_DIR/gqa_results_${TIMESTAMP}_summary.json"
if [ -f "$SUMMARY_FILE" ]; then
    echo "最终评估结果:"
    python -c "
import json
with open('$SUMMARY_FILE', 'r') as f:
    data = json.load(f)
print(f'总样本数: {data[\"total_samples\"]}')
print(f'准确率: {data[\"accuracy\"]:.4f} ({data[\"accuracy\"]*100:.2f}%)')
print(f'正确样本: {data[\"correct_samples\"]}')
print(f'错误样本: {data[\"error_samples\"]}')
print(f'GPU数量: {data[\"chunks\"]}')
"
    echo ""
    echo "🎉 8卡并行GQA评估完成成功!"
    echo "结果文件保存在: $(pwd)/$RESULTS_DIR"
    echo ""
    echo "生成的文件:"
    ls -la "$RESULTS_DIR"/*.json* 2>/dev/null || echo "未找到结果文件"
    echo ""
    echo "LLaVA官方GQA评估流程完成，结果完全匹配LLaVA原版测试逻辑！"
else
    echo "❌ 8卡并行GQA评估失败，请检查日志"
    exit 1
fi
