#!/usr/bin/env bash

# DAT-LLaVA-1.5 综合测试脚本
# 基于训练脚本 train_dat_llava1_5_v2.sh 的配置

export DS_SKIP_CUDA_CHECK=1

# 激活conda环境
source /home/zhuofan.xia/miniconda3/bin/activate pt260

# 设置实验名称和路径
EXP_NAME="tdat-7b-l0d32-s12g8z3"
# 主要训练输出目录 (在另一个remote上)
OUTPUT_DIR="/perception-hl/zhuofan.xia/vlm_exps/textdat/$EXP_NAME"
# 本地备份目录 (如果存在)
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

# 检查数据路径
DATA_PATH="/perception-hl/zhuofan.xia/data/llava_v1_5_mix665k.json"
IMAGE_FOLDER="/perception-hl/zhuofan.xia/data"

if [ ! -f "$DATA_PATH" ]; then
    echo "警告: 数据文件不存在: $DATA_PATH"
    echo "将使用GQA数据集进行测试"
    DATA_PATH="/root/gqa_opendatalab/testdev_balanced_questions.json"
    IMAGE_FOLDER="/root/gqa_opendatalab/images"
fi

if [ ! -f "$DATA_PATH" ]; then
    echo "错误: 未找到测试数据文件"
    echo "请检查以下路径:"
    echo "  - /perception-hl/zhuofan.xia/data/llava_v1_5_mix665k.json"
    echo "  - /root/gqa_opendatalab/testdev_balanced_questions.json"
    exit 1
fi

echo "使用数据文件: $DATA_PATH"
echo "使用图像文件夹: $IMAGE_FOLDER"

# 创建结果目录
RESULTS_DIR="./test_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "结果将保存到: $(pwd)/$RESULTS_DIR"

# 运行综合测试
echo "开始运行DAT-LLaVA-1.5综合测试..."
echo "="*80

# 设置时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 1. 运行TTFT测试
echo "1. 运行TTFT测试"
echo "="*60
python /home/zhuofan.xia/ml-fastvlm/ttft_test.py \
    --model-path "$CHECKPOINT_PATH" \
    --data-path "$DATA_PATH" \
    --image-folder "$IMAGE_FOLDER" \
    --resolution 336 \
    --vision-encoder clip \
    --max-samples 1000 \
    --output-file "$RESULTS_DIR/ttft_results_dat_llava1_5_${TIMESTAMP}.json"

TTFT_SUCCESS=$?

# 2. 运行FLOPs测试
echo ""
echo "2. 运行FLOPs测试"
echo "="*60
python /home/zhuofan.xia/ml-fastvlm/flops_test.py \
    --model-path "$CHECKPOINT_PATH" \
    --resolution 336 \
    --vision-encoder clip \
    --output-file "$RESULTS_DIR/flops_results_dat_llava1_5_${TIMESTAMP}.json"

FLOPS_SUCCESS=$?

# 3. 汇总结果
echo ""
echo "测试结果汇总"
echo "="*60
echo "模型路径: $CHECKPOINT_PATH"
echo "分辨率: 336x336"
echo "视觉编码器: clip"
echo "LLM类型: llama"
echo "-"*60
if [ $TTFT_SUCCESS -eq 0 ]; then
    echo "TTFT测试: ✅ 成功"
else
    echo "TTFT测试: ❌ 失败"
fi

if [ $FLOPS_SUCCESS -eq 0 ]; then
    echo "FLOPs测试: ✅ 成功"
else
    echo "FLOPs测试: ❌ 失败"
fi

# 创建综合结果文件
cat > "$RESULTS_DIR/comprehensive_test_results_dat_llava1_5_${TIMESTAMP}.json" << EOF
{
  "model_path": "$CHECKPOINT_PATH",
  "resolution": "336x336",
  "vision_encoder": "clip",
  "llm_type": "llama",
  "test_results": {
    "ttft_success": $([ $TTFT_SUCCESS -eq 0 ] && echo "true" || echo "false"),
    "flops_success": $([ $FLOPS_SUCCESS -eq 0 ] && echo "true" || echo "false"),
    "ttft_output_file": "ttft_results_dat_llava1_5_${TIMESTAMP}.json",
    "flops_output_file": "flops_results_dat_llava1_5_${TIMESTAMP}.json"
  },
  "timestamp": "$(date '+%Y-%m-%d %H:%M:%S')"
}
EOF

echo "综合结果文件: $RESULTS_DIR/comprehensive_test_results_dat_llava1_5_${TIMESTAMP}.json"

# 检查测试结果
if [ $TTFT_SUCCESS -eq 0 ] && [ $FLOPS_SUCCESS -eq 0 ]; then
    echo ""
    echo "🎉 所有测试完成成功!"
    echo "结果文件保存在: $(pwd)/$RESULTS_DIR"
    echo ""
    echo "生成的文件:"
    ls -la "$RESULTS_DIR"/*.json
else
    echo ""
    echo "❌ 部分测试失败，请检查日志"
    if [ $TTFT_SUCCESS -ne 0 ]; then
        echo "TTFT测试失败"
    fi
    if [ $FLOPS_SUCCESS -ne 0 ]; then
        echo "FLOPs测试失败"
    fi
    exit 1
fi
