#!/usr/bin/env python3
"""
高效解压GQA图像数据集
使用多进程和优化参数
"""

import os
import zipfile
import multiprocessing as mp
from tqdm import tqdm
import argparse

def extract_files_batch(args):
    """批量解压文件"""
    file_batch, zip_path, output_dir, worker_id = args
    
    print(f"Worker {worker_id}: 开始解压 {len(file_batch)} 个文件")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_file:
        for filename in tqdm(file_batch, desc=f"Worker {worker_id}"):
            try:
                # 提取文件到指定目录
                zip_file.extract(filename, output_dir)
            except Exception as e:
                print(f"Worker {worker_id}: 解压 {filename} 失败: {e}")
    
    print(f"Worker {worker_id}: 完成解压")

def fast_extract_gqa(zip_path, output_dir, num_workers=8):
    """高效解压GQA数据集"""
    
    print(f"开始高效解压: {zip_path}")
    print(f"输出目录: {output_dir}")
    print(f"使用 {num_workers} 个进程")
    
    # 获取zip文件中的所有图像文件
    print("读取zip文件列表...")
    with zipfile.ZipFile(zip_path, 'r') as zip_file:
        all_files = zip_file.namelist()
        # 只处理images目录下的jpg文件
        image_files = [f for f in all_files if f.startswith('images/') and f.endswith('.jpg')]
    
    print(f"总文件数: {len(image_files)}")
    
    # 将文件分成多个批次
    batch_size = len(image_files) // num_workers
    batches = []
    
    for i in range(num_workers):
        start_idx = i * batch_size
        if i == num_workers - 1:  # 最后一个批次包含剩余文件
            end_idx = len(image_files)
        else:
            end_idx = (i + 1) * batch_size
        
        batch = image_files[start_idx:end_idx]
        batches.append((batch, zip_path, output_dir, i))
        print(f"Worker {i}: 分配 {len(batch)} 个文件")
    
    # 并行解压
    print("开始并行解压...")
    with mp.Pool(processes=num_workers) as pool:
        pool.map(extract_files_batch, batches)
    
    print("所有进程解压完成！")
    
    # 验证结果
    images_dir = os.path.join(output_dir, 'images')
    if os.path.exists(images_dir):
        extracted_files = os.listdir(images_dir)
        print(f"解压完成，共 {len(extracted_files)} 个文件")
    else:
        print("解压失败，images目录不存在")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="高效解压GQA数据集")
    parser.add_argument("--zip_path", default="/perception-hl/zhuofan.xia/data/gqa_imgs.zip", help="zip文件路径")
    parser.add_argument("--output_dir", default="/perception-hl/zhuofan.xia/data/gqa", help="输出目录")
    parser.add_argument("--num_workers", type=int, default=8, help="进程数量")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 开始高效解压
    fast_extract_gqa(args.zip_path, args.output_dir, args.num_workers)
