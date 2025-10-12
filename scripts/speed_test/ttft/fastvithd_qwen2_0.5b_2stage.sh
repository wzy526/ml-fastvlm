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
MODEL_PATH="./checkpoints/llava-fastvithd_0.5b_stage2"  # FastVLM ckpt
DATA_PATH="/cluster/nvme2/wzy/gqa/questions/val_balanced_questions.json"              # GQA问题文件路径
IMAGE_FOLDER="/cluster/nvme2/wzy/gqa/images"              
MAX_SAMPLES=""                                     # 最大测试样本数（空值表示测试全部样本）
RESOLUTION=1024                                    # 输入分辨率 (1024, 1536, 2048)
OUTPUT_FILE=""                                 

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
    echo "当前节点: $(hostname)"
    echo "当前用户: $(whoami)"
    echo "当前目录: $(pwd)"
    echo ""
    echo "调试信息:"
    echo "检查父目录..."
    if [ -d "/cluster/home/data/gqa/questions" ]; then
        echo "✓ questions目录存在"
        echo "  目录内容:"
        ls -la /cluster/home/data/gqa/questions/ | head -5
    else
        echo "✗ questions目录不存在"
    fi
    echo ""
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
echo "  - Input Resolution: ${RESOLUTION}x${RESOLUTION}"
echo "  - Visual Tokens: 256"
echo "  - Vision Encoder Size: 125M"
echo "  - Model Size: 0.5B parameters"
echo "  - Stage: 2 (base version)"
echo ""
echo "模型路径: $MODEL_PATH"
echo "数据文件: $DATA_PATH"
echo "图像文件夹: $IMAGE_FOLDER"
if [ -n "$MAX_SAMPLES" ]; then
    echo "最大样本数: $MAX_SAMPLES"
else
    echo "测试样本数: 全部样本"
fi
echo "分辨率: ${RESOLUTION}x${RESOLUTION}"
if [ -n "$OUTPUT_FILE" ]; then
    echo "输出文件: $OUTPUT_FILE"
else
    echo "输出文件: 自动生成 (ttft_test_results_${RESOLUTION}x${RESOLUTION}.json)"
fi
echo "============================================================"
echo ""

# 运行8卡分布式测试
echo "开始FastVLM TTFT分布式测试..."

# 构建命令参数
CMD_ARGS=(
    --nproc_per_node=8
    --master_port=29500
    ttft_test.py
    --model-path "$MODEL_PATH"
    --data-path "$DATA_PATH"
    --image-folder "$IMAGE_FOLDER"
    --conv-mode "llava_v1"
    --resolution "$RESOLUTION"
)

# 只有当MAX_SAMPLES不为空时才添加该参数
if [ -n "$MAX_SAMPLES" ]; then
    CMD_ARGS+=(--max-samples "$MAX_SAMPLES")
fi

# 只有当OUTPUT_FILE不为空时才添加该参数
if [ -n "$OUTPUT_FILE" ]; then
    CMD_ARGS+=(--output-file "$OUTPUT_FILE")
fi

torchrun "${CMD_ARGS[@]}"

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