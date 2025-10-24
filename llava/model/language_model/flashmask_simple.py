"""
简单的FlashMask实现，基于PyTorch原生优化
直接调用PyTorch的高效注意力实现
"""

import torch
import torch.nn.functional as F
from typing import Optional, List
import math

def flashmask_attention(
    query: torch.Tensor,
    key: torch.Tensor, 
    value: torch.Tensor,
    image_range_list: Optional[List[List[List[int]]]] = None,
    grid_size: int = 12,
    dropout_p: float = 0.0,
    scale: Optional[float] = None
) -> torch.Tensor:
    """
    基于PyTorch原生优化的FlashMask注意力
    
    Args:
        query: [batch, seq_len, num_heads, head_dim]
        key: [batch, seq_len, num_heads, head_dim] 
        value: [batch, seq_len, num_heads, head_dim]
        image_range_list: 图像范围列表
        grid_size: 网格大小
        dropout_p: dropout概率
        scale: 缩放因子
        
    Returns:
        注意力输出 [batch, seq_len, num_heads, head_dim]
    """
    if scale is None:
        scale = 1.0 / math.sqrt(query.shape[-1])
    
    # 如果没有复杂掩码，直接使用PyTorch优化注意力
    if image_range_list is None or all(len(ranges) == 0 for ranges in image_range_list):
        return F.scaled_dot_product_attention(
            query, key, value,
            dropout_p=dropout_p,
            is_causal=False,
            scale=scale
        )
    
    # 处理动态掩码
    batch_size, seq_len, num_heads, head_dim = query.shape
    device = query.device
    dtype = query.dtype
    
    # 创建注意力掩码
    attention_mask = create_dynamic_mask(
        batch_size, num_heads, seq_len, image_range_list, grid_size, device, dtype
    )
    
    # 使用PyTorch的优化注意力
    return F.scaled_dot_product_attention(
        query, key, value,
        attn_mask=attention_mask,
        dropout_p=dropout_p,
        is_causal=False,
        scale=scale
    )

def create_dynamic_mask(
    batch_size: int,
    num_heads: int, 
    seq_len: int,
    image_range_list: List[List[List[int]]],
    grid_size: int,
    device: torch.device,
    dtype: torch.dtype
) -> torch.Tensor:
    """
    创建动态注意力掩码
    
    Args:
        batch_size: 批次大小
        num_heads: 注意力头数
        seq_len: 序列长度
        image_range_list: 图像范围列表
        grid_size: 网格大小
        device: 设备
        dtype: 数据类型
        
    Returns:
        注意力掩码 [batch, num_heads, seq_len, seq_len]
    """
    # 初始化掩码（全为0，表示允许注意力）
    mask = torch.zeros(batch_size, num_heads, seq_len, seq_len, device=device, dtype=dtype)
    
    for b_idx in range(batch_size):
        if len(image_range_list[b_idx]) == 0:
            continue
            
        for l_idx, (answer_start, answer_end) in enumerate(image_range_list[b_idx][1:]):
            if answer_end > 0:
                # 计算高分辨率特征范围
                hd_start = answer_start
                hd_end = answer_start + grid_size * grid_size
                
                # 应用动态掩码
                for row in range(seq_len):
                    if answer_start <= row < answer_end:
                        # 问题中：允许关注高分辨率特征
                        mask[b_idx, :, row, hd_start:hd_end] = 0.0
                    else:
                        # 问题前/后：阻止关注高分辨率特征
                        mask[b_idx, :, row, hd_start:hd_end] = float('-inf')
    
    return mask

# 为了兼容性，创建一个类包装器
class FlashMaskPyTorch:
    """简单的FlashMask包装器"""
    
    def __init__(self, use_flash_attn: bool = True):
        self.use_flash_attn = use_flash_attn
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        image_range_list: Optional[List[List[List[int]]]] = None,
        grid_size: int = 12,
        dropout_p: float = 0.0,
        scale: Optional[float] = None
    ) -> torch.Tensor:
        """前向传播"""
        return flashmask_attention(
            query, key, value, image_range_list, grid_size, dropout_p, scale
        )
