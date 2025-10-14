#!/usr/bin/env python3
"""
使用Tesseract生成TextVQA的OCR tokens - LLaVA官方方法
"""

import os
import json
import argparse
from tqdm import tqdm
from PIL import Image

def extract_ocr_with_tesseract(image_path):
    """使用Tesseract提取文本 - LLaVA官方方法"""
    try:
        import pytesseract
        image = Image.open(image_path)
        # 使用英文语言包，提高OCR准确率
        text = pytesseract.image_to_string(image, lang='eng')
        return text.strip()
    except ImportError:
        print("错误: pytesseract未安装，请安装: pip install pytesseract")
        return ""
    except Exception as e:
        print(f"Tesseract处理错误 {image_path}: {e}")
        return ""

def generate_ocr_tokens_tesseract(question_file, image_folder, output_file, max_samples=None):
    """使用Tesseract生成OCR tokens - LLaVA官方方法"""
    print("使用Tesseract生成OCR tokens...")
    
    # 加载问题数据
    print(f"加载问题文件: {question_file}")
    with open(question_file, 'r') as f:
        question_data = json.load(f)
    
    questions = question_data['questions']
    if max_samples:
        questions = questions[:max_samples]
    
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
            else:
                # 使用Tesseract提取OCR文本
                ocr_text = extract_ocr_with_tesseract(image_path)
            
            # 缓存结果
            image_ocr_cache[image_id] = ocr_text
        
        ocr_results.append({
            "question_id": question_id,
            "ocr_tokens": ocr_text
        })
        
        # 每100个样本保存一次中间结果
        if (i + 1) % 100 == 0:
            print(f"已处理 {i + 1} 个样本，缓存了 {len(image_ocr_cache)} 张图像")
            # 保存中间结果
            temp_file = output_file + ".temp"
            with open(temp_file, 'w') as f:
                json.dump(ocr_results, f, indent=2)
    
    # 保存最终结果
    print(f"保存OCR tokens到: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(ocr_results, f, indent=2)
    
    # 删除临时文件
    temp_file = output_file + ".temp"
    if os.path.exists(temp_file):
        os.remove(temp_file)
    
    print(f"完成! 生成了 {len(ocr_results)} 个OCR tokens")
    print(f"处理了 {len(image_ocr_cache)} 张唯一图像")

def main():
    parser = argparse.ArgumentParser(description="使用Tesseract生成TextVQA OCR tokens - LLaVA官方方法")
    parser.add_argument("--question-file", default="/perception-hl/zhuofan.xia/data/textvqa/val_questions.json")
    parser.add_argument("--image-folder", default="/perception-hl/zhuofan.xia/data/textvqa/train_images")
    parser.add_argument("--output-file", default="/perception-hl/zhuofan.xia/data/textvqa/val_ocr_tokens.json")
    parser.add_argument("--max-samples", type=int, default=None, help="处理样本数量（None表示处理全部）")
    
    args = parser.parse_args()
    
    generate_ocr_tokens_tesseract(args.question_file, args.image_folder, args.output_file, args.max_samples)

if __name__ == "__main__":
    main()
