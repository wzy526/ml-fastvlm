#!/bin/bash

# FastVLM-7B-Stage2 + FastViTHD 单卡TTFT 测试脚本
# Vision Encoder: FastViTHD
# LLM: FastVLM-7B-Stage2 (Qwen2-based)
# 使用单卡测试

# 指定GPU设备（可选）
# 如果您想指定特定的GPU设备，请取消注释并修改下面的行
# 例如：指定使用GPU 0
# export CUDA_VISIBLE_DEVICES=0
# 或者指定使用GPU 1
# export CUDA_VISIBLE_DEVICES=1
# 或者指定使用GPU 2
# export CUDA_VISIBLE_DEVICES=2

# 如果您想动态指定GPU，可以在运行脚本时设置环境变量：
# CUDA_VISIBLE_DEVICES= bash fastvithd_qwen2_7b_2stage_single.sh

echo "=========================================="
echo "FastVLM-7B-Stage2 + FastViTHD 单卡TTFT 测试"
echo "Vision Encoder: FastViTHD"
echo "LLM: FastVLM-7B-Stage2 (Qwen2-based)"
echo "单卡测试"
echo "=========================================="

# 显示GPU设备信息
echo "GPU设备信息："
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    echo "指定使用的GPU: $CUDA_VISIBLE_DEVICES"
else
    echo "使用默认GPU设备"
fi
echo "当前可用的GPU数量: $(nvidia-smi --list-gpus | wc -l)"
echo "当前节点: $(hostname)"
echo ""

# 模型路径 - FastVLM-7B-Stage2模型
MODEL_PATH="./checkpoints/llava-fastvithd_7b_stage2"

# 数据路径
DATA_PATH="/root/gqa_opendatalab/testdev_balanced_questions.json"
IMAGE_FOLDER="/root/gqa_opendatalab/images"

# 测试参数 - 支持通过环境变量覆盖
RESOLUTION=${RESOLUTION:-1024}  # 默认1024，可通过环境变量覆盖
MAX_SAMPLES=${MAX_SAMPLES:-""}  
VISUAL_TOKENS=${VISUAL_TOKENS:-256}  # 默认256，可通过环境变量覆盖
OUTPUT_FILE="ttft_results/ttft_test_results_fastvlm_7b_stage2_${RESOLUTION}x${RESOLUTION}_vt${VISUAL_TOKENS}_single.json"

# 可用的模型选项
AVAILABLE_MODELS=(
    "llava-fastvithd_0.5b_stage2"
    "llava-fastvithd_0.5b_stage3"
    "llava-fastvithd_1.5b_stage2"
    "llava-fastvithd_1.5b_stage3"
    "llava-fastvithd_7b_stage2"
    "llava-fastvithd_7b_stage3"
)

# 检查模型路径是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "错误: 模型路径不存在: $MODEL_PATH"
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
    echo "请确保FastVLM-7B-Stage2模型已下载并放置在正确位置"
    echo "模型应该位于: $MODEL_PATH"
    echo "下载命令: bash get_models.sh"
    exit 1
fi

# 检查数据路径是否存在
if [ ! -f "$DATA_PATH" ]; then
    echo "错误: 数据文件不存在: $DATA_PATH"
    echo "请确保GQA数据文件已下载"
    exit 1
fi

if [ ! -d "$IMAGE_FOLDER" ]; then
    echo "错误: 图像文件夹不存在: $IMAGE_FOLDER"
    echo "请确保GQA图像文件夹已下载"
    exit 1
fi

echo "开始FastVLM-7B-Stage2 + FastViTHD 单卡TTFT测试..."
echo "模型路径: $MODEL_PATH"
echo "数据路径: $DATA_PATH"
echo "图像路径: $IMAGE_FOLDER"
echo "分辨率: ${RESOLUTION}x${RESOLUTION}"
echo "最大样本数: ${MAX_SAMPLES:-全部样本}"
echo "Visual Tokens: $VISUAL_TOKENS"
echo "输出文件: $OUTPUT_FILE"
echo "测试配置: 单卡"
echo ""
echo "模型配置："
echo "- Vision Encoder: FastViTHD"
echo "- LLM: FastVLM-7B-Stage2 (Qwen2-based)"
echo "- 分辨率: ${RESOLUTION}x${RESOLUTION}"
echo "- Visual Tokens: $VISUAL_TOKENS"
echo "- 模型大小: 7B参数"
echo "- 对话模式: FastVLM"
echo "- 测试模式: 单卡"
echo ""

# 设置环境变量优化内存使用
echo "设置环境变量以优化内存使用..."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1

# 设置Hugging Face镜像以加速下载
echo "设置Hugging Face镜像..."
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_URL=https://hf-mirror.com
echo "使用镜像: $HF_ENDPOINT"

# 构建单卡测试命令
echo "启动单卡测试..."
CMD_ARGS=(
    python
    ttft_test.py
    --model-path "$MODEL_PATH"
    --data-path "$DATA_PATH"
    --image-folder "$IMAGE_FOLDER"
    --vision-encoder "fastvithd"
    --resolution "$RESOLUTION"
    --visual-tokens "$VISUAL_TOKENS"
    --output-file "$OUTPUT_FILE"
    --conv-mode "qwen_2"
)

# 只有当MAX_SAMPLES不为空时才添加该参数
if [ -n "$MAX_SAMPLES" ]; then
    CMD_ARGS+=(--max-samples "$MAX_SAMPLES")
fi

# 运行单卡测试
"${CMD_ARGS[@]}"

# 检查测试是否成功
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "FastVLM-7B-Stage2 + FastViTHD 单卡TTFT测试完成！"
    echo "结果保存在: $OUTPUT_FILE"
    echo "=========================================="
    
    # 显示结果摘要
    if [ -f "$OUTPUT_FILE" ]; then
        echo "结果摘要："
        echo "Vision Encoder: FastViTHD"
        echo "LLM: FastVLM-7B-Stage2"
        echo "分辨率: ${RESOLUTION}x${RESOLUTION}"
        echo "测试模式: 单卡"
        echo "模型: FastVLM-7B-Stage2"
        cat "$OUTPUT_FILE" | grep -E '"avg_ttft_ms"|"total_samples"|"vision_encoder"|"llm"|"world_size"'
    fi
else
    echo ""
    echo "=========================================="
    echo "测试失败！请检查错误信息"
    echo "可能的原因："
    echo "1. FastVLM-7B-Stage2模型路径不正确"
    echo "2. 缺少FastViTHD视觉编码器相关依赖"
    echo "3. 模型配置不正确"
    echo "4. GPU内存不足"
    echo "5. Python环境问题"
    echo "6. 模型文件未解压"
    echo "=========================================="
    exit 1
fi
