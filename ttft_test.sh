#!/bin/bash

# FastVLM TTFT 测试运行脚本
# 基于LLaVA-FastViTHD配置：
# - Vision Encoder: FastViTHD
# - LLM: Llama
# - Input Res: 1024
# - #Visual Tokens: 256
# - Vis. Enc. Size: 125M
# - Model Size: 0.5B parameters
# - Stage: 2 (base version)

# 设置参数
MODEL_PATH="./checkpoints/llava-fastvithd_7b_stage3"  # FastVLM模型路径
DATA_PATH="/path/to/gqa/questions.json"              # GQA问题文件路径
IMAGE_FOLDER="/path/to/gqa/images"                   # GQA图像文件夹路径
MAX_SAMPLES=1000                                     # 最大测试样本数
OUTPUT_FILE="ttft_test_results.json"                   # 输出结果文件

# 可用的模型选项
AVAILABLE_MODELS=(
    "llava-fastvithd_0.5b_stage2"
    "llava-fastvithd_0.5b_stage3"
    "llava-fastvithd_1.5b_stage2"
    "llava-fastvithd_1.5b_stage3"
    "llava-fastvithd_7b_stage2"
    "llava-fastvithd_7b_stage3"
)

# 检查参数
if [ ! -d "$MODEL_PATH" ]; then
    echo "错误: FastVLM模型路径不存在: $MODEL_PATH"
    echo ""
    echo "可用的模型选项:"
    for model in "${AVAILABLE_MODELS[@]}"; do
        if [ -d "./checkpoints/$model" ]; then
            echo "  ✓ $model"
        else
            echo "  ✗ $model (未下载)"
        fi
    done
    echo ""
    echo "请下载FastVLM模型或修改MODEL_PATH变量"
    echo "下载命令: bash get_models.sh"
    echo ""
    echo "使用示例:"
    echo "  MODEL_PATH=\"./checkpoints/llava-fastvithd_7b_stage3\" bash ttft_test.sh"
    exit 1
fi

if [ ! -f "$DATA_PATH" ]; then
    echo "错误: GQA数据文件不存在: $DATA_PATH"
    echo "请下载GQA数据集或修改DATA_PATH变量"
    exit 1
fi

if [ ! -d "$IMAGE_FOLDER" ]; then
    echo "错误: GQA图像文件夹不存在: $IMAGE_FOLDER"
    echo "请下载GQA图像或修改IMAGE_FOLDER变量"
    exit 1
fi

# 创建输出目录
mkdir -p $(dirname "$OUTPUT_FILE")

echo "============================================================"
echo "FastVLM TTFT 测试"
echo "============================================================"
echo "配置信息:"
echo "  - Vision Encoder: FastViTHD"
echo "  - LLM: Llama"
echo "  - Input Resolution: 1024x1024"
echo "  - Visual Tokens: 256"
echo "  - Vision Encoder Size: 125M"
echo "  - Model Size: 0.5B parameters"
echo "  - Stage: 2 (base version)"
echo ""
echo "模型路径: $MODEL_PATH"
echo "数据文件: $DATA_PATH"
echo "图像文件夹: $IMAGE_FOLDER"
echo "最大样本数: $MAX_SAMPLES"
echo "输出文件: $OUTPUT_FILE"
echo "============================================================"
echo ""

# 运行8卡分布式测试
echo "开始FastVLM TTFT分布式测试..."
torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    ttft_test.py \
    --model-path "$MODEL_PATH" \
    --data-path "$DATA_PATH" \
    --image-folder "$IMAGE_FOLDER" \
    --max-samples "$MAX_SAMPLES" \
    --output-file "$OUTPUT_FILE" \
    --conv-mode "llava_v1"

echo ""
echo "============================================================"
echo "FastVLM TTFT 测试完成！"
echo "结果已保存到: $OUTPUT_FILE"
echo "============================================================"

# 显示结果摘要
if [ -f "$OUTPUT_FILE" ]; then
    echo ""
    echo "结果摘要:"
    python3 -c "
import json
with open('$OUTPUT_FILE', 'r') as f:
    data = json.load(f)
print(f'平均TTFT: {data[\"avg_ttft_ms\"]:.2f}ms')
print(f'测试样本数: {data[\"total_samples\"]}')
print(f'累积延迟: {data[\"accumulated_latency_ms\"]:.2f}ms')
"
fi 