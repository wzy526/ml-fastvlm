#!/usr/bin/env python3
"""
8卡并行解压GQA图像数据集
"""

import os
import zipfile
import multiprocessing as mp
from tqdm import tqdm
import argparse

def extract_chunk(args):
    """解压一个文件块"""
    chunk_files, zip_path, output_dir, gpu_id = args
    
    print(f"GPU {gpu_id}: 开始解压 {len(chunk_files)} 个文件")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_file:
        for i, filename in enumerate(tqdm(chunk_files, desc=f"GPU {gpu_id}")):
            try:
                # 提取文件
                zip_file.extract(filename, output_dir)
            except Exception as e:
                print(f"GPU {gpu_id}: 解压 {filename} 失败: {e}")
    
    print(f"GPU {gpu_id}: 完成解压")

def parallel_extract_gqa(zip_path, output_dir, num_gpus=8):
    """8卡并行解压GQA数据集"""
    
    print(f"开始8卡并行解压: {zip_path}")
    print(f"输出目录: {output_dir}")
    
    # 获取zip文件中的所有文件
    with zipfile.ZipFile(zip_path, 'r') as zip_file:
        all_files = zip_file.namelist()
        # 只处理images目录下的文件
        image_files = [f for f in all_files if f.startswith('images/') and f.endswith('.jpg')]
    
    print(f"总文件数: {len(image_files)}")
    
    # 将文件分成8个块
    chunk_size = len(image_files) // num_gpus
    chunks = []
    
    for i in range(num_gpus):
        start_idx = i * chunk_size
        if i == num_gpus - 1:  # 最后一个块包含剩余文件
            end_idx = len(image_files)
        else:
            end_idx = (i + 1) * chunk_size
        
        chunk = image_files[start_idx:end_idx]
        chunks.append((chunk, zip_path, output_dir, i))
        print(f"GPU {i}: 分配 {len(chunk)} 个文件")
    
    # 并行解压
    print("开始并行解压...")
    with mp.Pool(processes=num_gpus) as pool:
        pool.map(extract_chunk, chunks)
    
    print("所有GPU解压完成！")
    
    # 验证结果
    extracted_files = os.listdir(os.path.join(output_dir, 'images'))
    print(f"解压完成，共 {len(extracted_files)} 个文件")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="8卡并行解压GQA数据集")
    parser.add_argument("--zip_path", default="/perception-hl/zhuofan.xia/data/gqa_imgs.zip", help="zip文件路径")
    parser.add_argument("--output_dir", default="/perception-hl/zhuofan.xia/data/gqa", help="输出目录")
    parser.add_argument("--num_gpus", type=int, default=8, help="GPU数量")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 开始并行解压
    parallel_extract_gqa(args.zip_path, args.output_dir, args.num_gpus)
