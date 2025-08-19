#!/bin/bash

# FastVLM FLOPs 测试运行脚本
# 支持多种视觉编码器和模型配置

echo "=========================================="
echo "FastVLM FLOPs 计算测试"
echo "=========================================="

# 检查是否安装了FLOPs计算库
echo "检查FLOPs计算库..."
python3 -c "
try:
    from fvcore.nn import FlopCountMode
    print('✓ fvcore 已安装')
except ImportError:
    try:
        from thop import profile
        print('✓ thop 已安装')
    except ImportError:
        print('✗ 未安装FLOPs计算库')
        print('建议安装: pip install fvcore thop')
"

echo ""

# 模型路径配置
MODEL_PATH="./checkpoints/llava-fastvithd_7b_stage2"

# 检查模型路径是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "错误: 模型路径不存在: $MODEL_PATH"
    echo "请确保FastVLM模型已下载并放置在正确位置"
    echo "下载命令: bash get_models.sh"
    exit 1
fi

echo "开始FLOPs计算测试..."
echo "模型路径: $MODEL_PATH"
echo ""

# 测试不同的配置
echo "=========================================="
echo "测试1: FastViTHD + 1024x1024分辨率"
echo "=========================================="
python3 flops_test_advanced.py \
    --model-path "$MODEL_PATH" \
    --vision-encoder "fastvithd" \
    --resolution 1024 \
    --output-file "flops_results_fastvithd_1024x1024.json"

echo ""
echo "=========================================="
echo "测试2: CLIP + 336x336分辨率"
echo "=========================================="
python3 flops_test_advanced.py \
    --model-path "$MODEL_PATH" \
    --vision-encoder "clip" \
    --resolution 336 \
    --output-file "flops_results_clip_336x336.json"

echo ""
echo "=========================================="
echo "测试3: CLIP-S2 + 672x672分辨率"
echo "=========================================="
python3 flops_test_advanced.py \
    --model-path "$MODEL_PATH" \
    --vision-encoder "clip_s2" \
    --resolution 672 \
    --output-file "flops_results_clip_s2_672x672.json"

echo ""
echo "=========================================="
echo "测试4: CLIP-LLaVA16 + 1008x1008分辨率"
echo "=========================================="
python3 flops_test_advanced.py \
    --model-path "$MODEL_PATH" \
    --vision-encoder "clip_llava16" \
    --resolution 1008 \
    --output-file "flops_results_clip_llava16_1008x1008.json"

echo ""
echo "=========================================="
echo "FLOPs测试完成！"
echo "=========================================="

# 显示结果摘要
echo "结果文件:"
ls -la flops_results_*.json

echo ""
echo "结果摘要:"
for file in flops_results_*.json; do
    if [ -f "$file" ]; then
        echo "=== $file ==="
        python3 -c "
import json
with open('$file', 'r') as f:
    data = json.load(f)
print(f'模型: {data[\"model_config\"]}')
print(f'视觉编码器: {data[\"vision_encoder\"]}')
print(f'分辨率: {data[\"input_resolution\"]}')
print(f'Visual Tokens: {data[\"visual_tokens\"]}')
print(f'总FLOPs: {data[\"total_flops_g\"]:.2f}G')
print(f'编码器占比: {data[\"encoder_ratio\"]:.1f}%')
print(f'LLM占比: {data[\"llm_ratio\"]:.1f}%')
print()
"
    fi
done

echo "=========================================="
echo "所有测试完成！"
echo "=========================================="
