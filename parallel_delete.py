#!/usr/bin/env python3
"""
8卡并行删除大量文件
"""

import os
import multiprocessing as mp
from tqdm import tqdm
import argparse

def delete_files_batch(args):
    """删除一批文件"""
    file_batch, worker_id = args
    
    print(f"Worker {worker_id}: 开始删除 {len(file_batch)} 个文件")
    
    deleted_count = 0
    for file_path in tqdm(file_batch, desc=f"Worker {worker_id}"):
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                deleted_count += 1
        except Exception as e:
            print(f"Worker {worker_id}: 删除 {file_path} 失败: {e}")
    
    print(f"Worker {worker_id}: 完成删除 {deleted_count} 个文件")
    return deleted_count

def parallel_delete_files(directory, num_workers=8):
    """8卡并行删除文件"""
    
    print(f"开始8卡并行删除: {directory}")
    print(f"使用 {num_workers} 个进程")
    
    # 获取所有文件
    print("扫描文件...")
    all_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            all_files.append(os.path.join(root, file))
    
    print(f"总文件数: {len(all_files)}")
    
    if len(all_files) == 0:
        print("没有文件需要删除")
        return
    
    # 将文件分成多个批次
    batch_size = len(all_files) // num_workers
    batches = []
    
    for i in range(num_workers):
        start_idx = i * batch_size
        if i == num_workers - 1:  # 最后一个批次包含剩余文件
            end_idx = len(all_files)
        else:
            end_idx = (i + 1) * batch_size
        
        batch = all_files[start_idx:end_idx]
        batches.append((batch, i))
        print(f"Worker {i}: 分配 {len(batch)} 个文件")
    
    # 并行删除
    print("开始并行删除...")
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(delete_files_batch, batches)
    
    total_deleted = sum(results)
    print(f"所有进程删除完成！共删除 {total_deleted} 个文件")
    
    # 删除空目录
    print("删除空目录...")
    for root, dirs, files in os.walk(directory, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            try:
                os.rmdir(dir_path)
            except OSError:
                pass  # 目录不为空，跳过
    
    # 最后删除主目录
    try:
        os.rmdir(directory)
        print(f"目录 {directory} 已完全删除")
    except OSError:
        print(f"目录 {directory} 删除失败，可能还有文件")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="8卡并行删除文件")
    parser.add_argument("--directory", default="/perception-hl/zhuofan.xia/data/gqa_rm", help="要删除的目录")
    parser.add_argument("--num_workers", type=int, default=8, help="进程数量")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.directory):
        print(f"目录不存在: {args.directory}")
        exit(1)
    
    # 开始并行删除
    parallel_delete_files(args.directory, args.num_workers)
