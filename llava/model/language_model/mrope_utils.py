"""
mRoPE 位置编码工具 for DAT
支持动态采样的 3D 位置分配
"""

import torch
from typing import List, Tuple, Optional


def generate_3d_positions_for_dat(
    batch_size: int,
    seq_len: int,
    image_range_list: List[List[Tuple[int, int]]],
    sampling_locs: Optional[List[torch.Tensor]] = None,
    lr_size: int = 24,
    hr_size: int = 72,
    device: torch.device = None,
    dtype: torch.dtype = torch.long
) -> torch.Tensor:
    """
    为 DAT 序列生成 3D 位置编码 (temporal, height, width)
    
    Args:
        batch_size: 批次大小
        seq_len: KV 序列长度（包含插入的 HD features）
        image_range_list: 每个样本的图像范围列表
            格式: [[(lr_start, lr_end), (qa1_start, qa1_end), ...], ...]
        sampling_locs: DAT 采样位置 [batch, Lp, groups, grid_h, grid_w, 2]
            如果为 None，HD tokens 使用简单的网格位置
        lr_size: 低分辨率大小 (24)
        hr_size: 高分辨率大小 (72)
        device: 设备
        dtype: 数据类型
    
    Returns:
        position_ids_3d: [batch, seq_len, 3] 的位置张量 (完全对齐 Qwen2-VL)
            dim 0: temporal = 1 (静态图像/文本)
            dim 1: height 
                - 文本: 1 (表示1D序列)
                - 图像: patch在图像中的行坐标
            dim 2: width
                - 文本: 序列位置
                - 图像: patch在图像中的列坐标
    """
    if device is None:
        device = torch.device('cpu')
    
    # 初始化 3D 位置张量
    position_ids_3d = torch.zeros(batch_size, seq_len, 3, dtype=dtype, device=device)
    
    zoom_ratio = hr_size // lr_size  # 3
    
    for b_idx in range(batch_size):
        if len(image_range_list[b_idx]) == 0:
            # 没有图像的样本：所有位置用序列索引（纯文本）
            position_ids_3d[b_idx, :, 0] = 1  # temporal = 1 (静态内容)
            position_ids_3d[b_idx, :, 1] = 1  # height = 1 (1D序列)
            position_ids_3d[b_idx, :, 2] = torch.arange(seq_len, device=device, dtype=dtype)  # width = seq_pos
            continue
        
        # === 1. 处理 LR 图像 tokens ===
        lr_start, lr_end = image_range_list[b_idx][0]
        lr_tokens = lr_end - lr_start  # 应该是 576 (24*24)
        
        for idx in range(lr_tokens):
            abs_pos = lr_start + idx
            
            # 在 24×24 网格中的位置
            h_lr = idx // lr_size  # 0-23
            w_lr = idx % lr_size   # 0-23
            
            # 映射到 72×72 空间：每个 LR token 代表 3×3 区域的中心
            h_hr = h_lr * zoom_ratio + zoom_ratio // 2  # 1, 4, 7, ..., 70
            w_hr = w_lr * zoom_ratio + zoom_ratio // 2  # 1, 4, 7, ..., 70
            
            position_ids_3d[b_idx, abs_pos, 0] = 1     # temporal = 1 (静态图像)
            position_ids_3d[b_idx, abs_pos, 1] = h_hr  # height in 72×72
            position_ids_3d[b_idx, abs_pos, 2] = w_hr  # width in 72×72
        
        # === 2. 处理文本 tokens (LR 图像之前) ===
        # 参考 Qwen2-VL: 文本用 [1, 1, seq_pos] 表示 1D 序列
        for pos in range(lr_start):
            position_ids_3d[b_idx, pos, 0] = 1  # temporal = 1 (静态内容)
            position_ids_3d[b_idx, pos, 1] = 1  # height = 1 (1D序列，无2D结构)
            position_ids_3d[b_idx, pos, 2] = pos  # width = 序列位置
        
        # === 3. 处理 HD 采样 tokens 和中间的文本 ===
        if len(image_range_list[b_idx]) > 1:
            # 有问答对
            num_qa_pairs = len(image_range_list[b_idx]) - 1
            
            for qa_idx in range(num_qa_pairs):
                qa_start, qa_end = image_range_list[b_idx][qa_idx + 1]
                
                # 这个范围之前的文本（如果有）
                prev_end = image_range_list[b_idx][qa_idx][1] if qa_idx == 0 else image_range_list[b_idx][qa_idx][1]
                for pos in range(prev_end, qa_start):
                    if pos < seq_len:
                        position_ids_3d[b_idx, pos, 0] = 1  # temporal = 1
                        position_ids_3d[b_idx, pos, 1] = 1  # height = 1
                        position_ids_3d[b_idx, pos, 2] = pos
                
                # HD tokens 本身
                if sampling_locs is not None and b_idx < len(sampling_locs):
                    # 使用真实的采样位置
                    sample_locs_qa = sampling_locs[b_idx][qa_idx]  # [groups, grid_h, grid_w, 2]
                    
                    # 平均跨 groups（或使用第一个 group）
                    if sample_locs_qa.dim() == 4:
                        sample_locs_qa = sample_locs_qa.mean(dim=0)  # [grid_h, grid_w, 2]
                    
                    sample_locs_flat = sample_locs_qa.reshape(-1, 2)  # [grid_h*grid_w, 2]
                    
                    for token_idx in range(min(len(sample_locs_flat), qa_end - qa_start)):
                        abs_pos = qa_start + token_idx
                        
                        # 归一化坐标 [-1, 1] → HR 坐标，对齐 Qwen2-VL
                        # 映射到 [1, hr_size]，与 LR 的坐标范围一致
                        norm_h, norm_w = sample_locs_flat[token_idx]
                        h_hr = int((norm_h.item() + 1) / 2 * (hr_size - 1)) + 1
                        w_hr = int((norm_w.item() + 1) / 2 * (hr_size - 1)) + 1
                        
                        # Clamp 到有效范围
                        h_hr = max(1, min(hr_size, h_hr))
                        w_hr = max(1, min(hr_size, w_hr))
                        
                        position_ids_3d[b_idx, abs_pos, 0] = 1     # temporal = 1 (静态图像)
                        position_ids_3d[b_idx, abs_pos, 1] = h_hr  # 采样的高度
                        position_ids_3d[b_idx, abs_pos, 2] = w_hr  # 采样的宽度
                else:
                    # Fallback: 使用简单网格
                    grid_size = int((qa_end - qa_start) ** 0.5)
                    for token_idx in range(qa_end - qa_start):
                        abs_pos = qa_start + token_idx
                        h_grid = token_idx // grid_size
                        w_grid = token_idx % grid_size
                        
                        # 映射到 72×72 空间，坐标从 1 开始
                        h_hr = int(h_grid / (grid_size - 1) * (hr_size - 1)) + 1 if grid_size > 1 else hr_size // 2
                        w_hr = int(w_grid / (grid_size - 1) * (hr_size - 1)) + 1 if grid_size > 1 else hr_size // 2
                        
                        position_ids_3d[b_idx, abs_pos, 0] = 1  # temporal = 1
                        position_ids_3d[b_idx, abs_pos, 1] = h_hr
                        position_ids_3d[b_idx, abs_pos, 2] = w_hr
                
                # QA 范围之后的文本
                for pos in range(qa_end, seq_len):
                    if qa_idx == num_qa_pairs - 1:  # 最后一个 QA 之后
                        position_ids_3d[b_idx, pos, 0] = 1  # temporal = 1
                        position_ids_3d[b_idx, pos, 1] = 1  # height = 1
                        position_ids_3d[b_idx, pos, 2] = pos
        else:
            # 只有 LR，没有 QA
            for pos in range(lr_end, seq_len):
                position_ids_3d[b_idx, pos, 0] = 1  # temporal = 1
                position_ids_3d[b_idx, pos, 1] = 1  # height = 1
                position_ids_3d[b_idx, pos, 2] = pos
    
    return position_ids_3d


def test_position_generation():
    """测试位置生成"""
    batch_size = 2
    
    # 模拟序列：[text, LR_image, text, HD_sample, text]
    image_range_list = [
        [(10, 586), (600, 744)],  # batch 0: LR at 10-586, QA1 at 600-744
        [(5, 581), (590, 734), (740, 884)]  # batch 1: LR + 2 QAs
    ]
    
    seq_len = 900
    
    # 模拟采样位置
    sampling_locs = [
        torch.randn(1, 1, 12, 12, 2) * 0.5,  # batch 0, 1 QA
        torch.randn(2, 1, 12, 12, 2) * 0.5   # batch 1, 2 QAs
    ]
    
    position_ids_3d = generate_3d_positions_for_dat(
        batch_size=batch_size,
        seq_len=seq_len,
        image_range_list=image_range_list,
        sampling_locs=sampling_locs,
        lr_size=24,
        hr_size=72
    )
    
    print("Position IDs shape:", position_ids_3d.shape)
    print("\nBatch 0 samples:")
    print("  Text[0]:", position_ids_3d[0, 0])
    print("  LR[10]:", position_ids_3d[0, 10])
    print("  LR[585]:", position_ids_3d[0, 585])
    print("  HD[600]:", position_ids_3d[0, 600])
    print("  HD[700]:", position_ids_3d[0, 700])
    
    # 验证 LR 坐标
    lr_pos = position_ids_3d[0, 10:586]
    print(f"\nLR height range: [{lr_pos[:, 1].min()}, {lr_pos[:, 1].max()}]")
    print(f"LR width range: [{lr_pos[:, 2].min()}, {lr_pos[:, 2].max()}]")
    
    # 验证 HD 坐标
    hd_pos = position_ids_3d[0, 600:744]
    print(f"\nHD height range: [{hd_pos[:, 1].min()}, {hd_pos[:, 1].max()}]")
    print(f"HD width range: [{hd_pos[:, 2].min()}, {hd_pos[:, 2].max()}]")


if __name__ == "__main__":
    test_position_generation()
