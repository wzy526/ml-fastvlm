#!/usr/bin/env python3
"""
LLaVA风格的OCR tokens生成 - 无OpenCV版本
使用PIL和EasyOCR，避免OpenCV依赖问题
"""

import os
import json
import argparse
from tqdm import tqdm
from PIL import Image
import numpy as np

def extract_ocr_with_easyocr(image_path):
    """使用EasyOCR提取文本 - 无OpenCV版本"""
    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False)  # 不使用GPU避免依赖问题
        
        # 使用PIL读取图像，避免OpenCV
        image = Image.open(image_path)
        image_array = np.array(image)
        
        # 执行OCR
        results = reader.readtext(image_array)
        
        # 提取文本
        texts = []
        for (bbox, text, confidence) in results:
            if confidence > 0.5:  # 只保留高置信度的文本
                texts.append(text.strip())
        
        return " ".join(texts)
    except ImportError:
        print("EasyOCR未安装，使用模拟OCR tokens")
        return "SAMPLE TEXT"
    except Exception as e:
        print(f"OCR处理错误 {image_path}: {e}")
        return ""

def generate_ocr_tokens(question_file, image_folder, output_file, max_samples=50, use_real_ocr=False):
    """生成OCR tokens - 支持真实OCR和模拟OCR"""
    print("生成OCR tokens...")
    
    # 加载问题数据
    print(f"加载问题文件: {question_file}")
    with open(question_file, 'r') as f:
        question_data = json.load(f)
    
    questions = question_data['questions'][:max_samples]
    print(f"处理 {len(questions)} 个问题...")
    
    # 图像OCR缓存，避免重复处理
    image_ocr_cache = {}
    ocr_results = []
    
    for i, q in enumerate(tqdm(questions)):
        question_id = q['question_id']
        image_id = q['image_id']
        
        # 检查是否已经处理过这张图像
        if image_id in image_ocr_cache:
            ocr_text = image_ocr_cache[image_id]
        else:
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
            elif use_real_ocr:
                # 使用真实OCR
                ocr_text = extract_ocr_with_easyocr(image_path)
            else:
                # 使用模拟OCR tokens
                question = q['question'].lower()
                if 'sign' in question or 'text' in question:
                    ocr_text = "STOP SIGN"
                elif 'number' in question or 'digit' in question:
                    ocr_text = "123"
                elif 'word' in question or 'letter' in question:
                    ocr_text = "HELLO WORLD"
                else:
                    ocr_text = "SAMPLE TEXT"
            
            # 缓存结果
            image_ocr_cache[image_id] = ocr_text
        
        ocr_results.append({
            "question_id": question_id,
            "ocr_tokens": ocr_text
        })
    
    # 保存结果
    print(f"保存OCR tokens到: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(ocr_results, f, indent=2)
    
    print(f"完成! 生成了 {len(ocr_results)} 个OCR tokens")
    print(f"处理了 {len(image_ocr_cache)} 张唯一图像")

def main():
    parser = argparse.ArgumentParser(description="生成OCR tokens - 支持真实OCR和模拟OCR")
    parser.add_argument("--question-file", default="/perception-hl/zhuofan.xia/data/textvqa/val_questions.json")
    parser.add_argument("--image-folder", default="/perception-hl/zhuofan.xia/data/textvqa/train_images")
    parser.add_argument("--output-file", default="/perception-hl/zhuofan.xia/data/textvqa/ocr_tokens.json")
    parser.add_argument("--max-samples", type=int, default=50, help="处理样本数量")
    parser.add_argument("--use-real-ocr", action="store_true", help="使用真实OCR（需要安装EasyOCR）")
    
    args = parser.parse_args()
    
    generate_ocr_tokens(args.question_file, args.image_folder, args.output_file, args.max_samples, args.use_real_ocr)

if __name__ == "__main__":
    main()
