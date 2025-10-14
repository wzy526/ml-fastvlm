#!/usr/bin/env python3
"""
生成TextVQA的OCR tokens文件
使用PaddleOCR提取图像中的文本
"""

import os
import json
import argparse
from PIL import Image
from tqdm import tqdm
import cv2
import numpy as np

try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    print("警告: PaddleOCR未安装，请安装: pip install paddlepaddle paddleocr")

def extract_ocr_with_paddle(image_path, ocr_model):
    """使用PaddleOCR提取文本"""
    try:
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            return ""
        
        # 执行OCR
        result = ocr_model.ocr(image, cls=True)
        
        # 提取文本
        texts = []
        if result and result[0]:
            for line in result[0]:
                if line and len(line) >= 2:
                    text = line[1][0]  # 提取文本内容
                    confidence = line[1][1]  # 置信度
                    if confidence > 0.5:  # 只保留高置信度的文本
                        texts.append(text)
        
        return " ".join(texts)
    except Exception as e:
        print(f"OCR处理错误 {image_path}: {e}")
        return ""

def generate_ocr_tokens(question_file, image_folder, output_file, max_samples=None):
    """生成OCR tokens文件"""
    print("初始化PaddleOCR...")
    if not PADDLE_AVAILABLE:
        print("错误: PaddleOCR未安装")
        return
    
    # 初始化OCR模型
    ocr_model = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
    
    # 加载问题数据
    print(f"加载问题文件: {question_file}")
    with open(question_file, 'r') as f:
        question_data = json.load(f)
    
    questions = question_data['questions']
    if max_samples:
        questions = questions[:max_samples]
    
    print(f"处理 {len(questions)} 个问题...")
    
    ocr_results = []
    for i, q in enumerate(tqdm(questions)):
        question_id = q['question_id']
        image_id = q['image_id']
        
        # 尝试不同的图像路径
        possible_paths = [
            os.path.join(image_folder, f"{image_id}.jpg"),
            os.path.join(image_folder, f"{image_id}.png"),
            os.path.join(image_folder, f"n{image_id}.jpg"),
            os.path.join(image_folder, f"n{image_id}.png")
        ]
        
        image_path = None
        for path in possible_paths:
            if os.path.exists(path):
                image_path = path
                break
        
        if not image_path:
            print(f"警告: 找不到图像 {image_id}")
            ocr_text = ""
        else:
            # 提取OCR文本
            ocr_text = extract_ocr_with_paddle(image_path, ocr_model)
        
        ocr_results.append({
            "question_id": question_id,
            "ocr_tokens": ocr_text
        })
        
        # 每100个样本保存一次
        if (i + 1) % 100 == 0:
            print(f"已处理 {i + 1} 个样本")
    
    # 保存结果
    print(f"保存OCR tokens到: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(ocr_results, f, indent=2)
    
    print(f"完成! 生成了 {len(ocr_results)} 个OCR tokens")

def main():
    parser = argparse.ArgumentParser(description="生成TextVQA OCR tokens")
    parser.add_argument("--question-file", default="/perception-hl/zhuofan.xia/data/textvqa/val_questions.json")
    parser.add_argument("--image-folder", default="/perception-hl/zhuofan.xia/data/textvqa/train_images")
    parser.add_argument("--output-file", default="/perception-hl/zhuofan.xia/data/textvqa/val_ocr_tokens.json")
    parser.add_argument("--max-samples", type=int, default=None)
    
    args = parser.parse_args()
    
    generate_ocr_tokens(args.question_file, args.image_folder, args.output_file, args.max_samples)

if __name__ == "__main__":
    main()
