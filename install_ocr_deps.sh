#!/bin/bash
# 安装LLaVA风格的OCR依赖

echo "安装LLaVA风格的OCR依赖..."

# 安装EasyOCR（LLaVA官方推荐）
echo "安装EasyOCR..."
pip install easyocr

# 安装OpenCV
echo "安装OpenCV..."
pip install opencv-python

# 安装PIL
echo "安装PIL..."
pip install Pillow

# 安装tqdm
echo "安装tqdm..."
pip install tqdm

echo "OCR依赖安装完成！"
echo ""
echo "使用方法："
echo "1. 生成测试OCR tokens:"
echo "   python generate_ocr_test.py --max-samples 50"
echo ""
echo "2. 生成真实OCR tokens:"
echo "   python generate_ocr_llava_style.py --max-samples 100"
echo ""
echo "3. 使用OCR tokens评估:"
echo "   python eval_textvqa_official.py --max-samples 5 --ocr-file /path/to/ocr_tokens.json"
