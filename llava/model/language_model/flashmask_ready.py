"""
现成的FlashMask实现，基于社区库
直接调用支持自定义掩码的现成实现
"""

import torch
import torch.nn.functional as F
from typing import Optional, List
import math

# 尝试导入各种现成的实现
try:
    from flash_dmattn import flash_dmattn_func_auto
    FLASH_DMATTN_AVAILABLE = True
except ImportError:
    FLASH_DMATTN_AVAILABLE = False

try:
    import xformers.ops as xops
    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False

try:
    from torch.nn.attention import flex_attention
    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False

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
    使用现成的FlashMask实现
    
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
    
    # 如果没有复杂掩码，使用标准注意力
    if image_range_list is None or all(len(ranges) == 0 for ranges in image_range_list):
        return F.scaled_dot_product_attention(
            query, key, value,
            dropout_p=dropout_p,
            is_causal=False,
            scale=scale
        )
    
    # 创建动态掩码
    mask = create_dynamic_mask(
        query.shape[0], query.shape[2], query.shape[1], 
        image_range_list, grid_size, query.device, query.dtype
    )
    
    # 尝试使用现成的实现
    if FLASH_DMATTN_AVAILABLE:
        try:
            flash_dmattn_func = flash_dmattn_func_auto(backend="cuda")
            return flash_dmattn_func(
                q=query, k=key, v=value,
                attn_mask=mask,
                is_causal=False,
                scale=scale
            )
        except Exception as e:
            print(f"Flash Dynamic Mask Attention failed: {e}")
    
    if XFORMERS_AVAILABLE:
        try:
            return xops.memory_efficient_attention(
                query, key, value,
                attn_bias=mask,
                dropout_p=dropout_p
            )
        except Exception as e:
            print(f"xFormers failed: {e}")
    
    if FLEX_ATTENTION_AVAILABLE:
        try:
            return flex_attention(
                query, key, value,
                attn_mask=mask,
                dropout_p=dropout_p,
                is_causal=False
            )
        except Exception as e:
            print(f"FlexAttention failed: {e}")
    
    # 回退到标准实现
    print("Warning: No optimized implementation available, using standard attention")
    return F.scaled_dot_product_attention(
        query, key, value,
        attn_mask=mask,
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
    """创建动态注意力掩码"""
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
    """现成的FlashMask包装器"""
    
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

# 检查可用的实现
def check_available_implementations():
    """检查可用的实现"""
    print("检查可用的FlashMask实现:")
    print(f"Flash Dynamic Mask Attention: {'✅' if FLASH_DMATTN_AVAILABLE else '❌'}")
    print(f"xFormers: {'✅' if XFORMERS_AVAILABLE else '❌'}")
    print(f"FlexAttention: {'✅' if FLEX_ATTENTION_AVAILABLE else '❌'}")
    
    if not any([FLASH_DMATTN_AVAILABLE, XFORMERS_AVAILABLE, FLEX_ATTENTION_AVAILABLE]):
        print("\n建议安装以下库之一:")
        print("pip install flash-dmattn  # Flash Dynamic Mask Attention")
        print("pip install xformers     # xFormers")
        print("# FlexAttention通常包含在PyTorch 2.1+中")

if __name__ == "__main__":
    check_available_implementations()
