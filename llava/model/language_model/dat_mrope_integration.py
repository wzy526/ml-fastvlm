"""
DAT + mRoPE 集成补丁

这个文件包含了将 mRoPE 集成到 LlamaAttentionDAT 的完整修改方案
"""

import torch
import torch.nn as nn
from typing import List, Optional
from .mrope_utils import generate_3d_positions_for_dat


def apply_mrope_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    position_ids_3d: torch.Tensor,
    rotary_emb: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    应用 mRoPE (3D rotary position embedding)
    
    Args:
        query: [batch, heads, seq_len, head_dim]
        key: [batch, heads, seq_len, head_dim]
        position_ids_3d: [batch, seq_len, 3] - (temporal, height, width)
        rotary_emb: RoPE embedding 模块
    
    Returns:
        rotated query and key
    """
    batch, heads, seq_len, head_dim = query.shape
    device, dtype = query.device, query.dtype
    
    # 确保 head_dim 可以被 3 整除（对于 3D mRoPE）
    # 如果不能，我们需要特殊处理
    if head_dim % 3 != 0:
        # Fallback: 只用 width 维度（等价于 1D RoPE）
        print(f"Warning: head_dim {head_dim} not divisible by 3, using 1D RoPE fallback")
        pos_1d = position_ids_3d[:, :, 2]  # 只用 width
        cos, sin = rotary_emb(key, pos_1d)
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
        return apply_rotary_pos_emb(query, key, cos, sin)
    
    # 将 head_dim 分成 3 部分，分别用于 temporal, height, width
    dim_per_axis = head_dim // 3
    
    # 为每个维度生成 cos/sin
    cos_t, sin_t = rotary_emb(query[:, :, :, :dim_per_axis], position_ids_3d[:, :, 0])
    cos_h, sin_h = rotary_emb(query[:, :, :, dim_per_axis:2*dim_per_axis], position_ids_3d[:, :, 1])
    cos_w, sin_w = rotary_emb(query[:, :, :, 2*dim_per_axis:], position_ids_3d[:, :, 2])
    
    # 分别旋转每个维度
    from transformers.models.llama.modeling_llama import rotate_half
    
    # Query
    q_t = query[:, :, :, :dim_per_axis]
    q_h = query[:, :, :, dim_per_axis:2*dim_per_axis]
    q_w = query[:, :, :, 2*dim_per_axis:]
    
    q_t_rot = q_t * cos_t[:, None, :, :] + rotate_half(q_t) * sin_t[:, None, :, :]
    q_h_rot = q_h * cos_h[:, None, :, :] + rotate_half(q_h) * sin_h[:, None, :, :]
    q_w_rot = q_w * cos_w[:, None, :, :] + rotate_half(q_w) * sin_w[:, None, :, :]
    
    query_rotated = torch.cat([q_t_rot, q_h_rot, q_w_rot], dim=-1)
    
    # Key
    k_t = key[:, :, :, :dim_per_axis]
    k_h = key[:, :, :, dim_per_axis:2*dim_per_axis]
    k_w = key[:, :, :, 2*dim_per_axis:]
    
    k_t_rot = k_t * cos_t[:, None, :, :] + rotate_half(k_t) * sin_t[:, None, :, :]
    k_h_rot = k_h * cos_h[:, None, :, :] + rotate_half(k_h) * sin_h[:, None, :, :]
    k_w_rot = k_w * cos_w[:, None, :, :] + rotate_half(k_w) * sin_w[:, None, :, :]
    
    key_rotated = torch.cat([k_t_rot, k_h_rot, k_w_rot], dim=-1)
    
    return query_rotated, key_rotated


def integrate_mrope_into_dat_forward(
    self,
    # RoPE 相关
    query_bhnc: torch.Tensor,
    key_bhnc: torch.Tensor,
    value_bhnc: torch.Tensor,
    # 位置信息
    Nq: int,
    kv_len: int,
    image_range_list: List,
    sampling_locs: List[torch.Tensor],
    pos_q_concat: List[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    # 其他
    device: torch.device,
    B: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    在 DAT forward 中集成 mRoPE 的位置编码逻辑
    
    这个函数替代原来的 RoPE 应用部分（line 559-585）
    
    Returns:
        (query_rotated, key_rotated)
    """
    
    if self.use_mrope and Nq < kv_len:
        # ===== mRoPE 路径 =====
        # 1. 构建真实的 image_range_list（包含 KV 序列中的位置）
        # 注意：image_range_list 原本是 Query 序列的范围，需要转换为 KV 序列范围
        kv_image_range_list = []
        for b_idx in range(B):
            kv_ranges = []
            if len(image_range_list[b_idx]) == 0:
                kv_image_range_list.append([])
                continue
            
            # LR 图像范围保持不变
            lr_start, lr_end = image_range_list[b_idx][0]
            kv_ranges.append((lr_start, lr_end))
            
            # QA 范围需要根据 HD 插入调整
            # 这需要根据你的具体插入逻辑来计算
            # 简化版本：假设我们可以从 keys_concat 的长度推断
            if len(image_range_list[b_idx]) > 1:
                # TODO: 这里需要更精确的范围计算
                # 暂时使用原始范围作为近似
                for qa_range in image_range_list[b_idx][1:]:
                    kv_ranges.append(qa_range)
            
            kv_image_range_list.append(kv_ranges)
        
        # 2. 生成 3D 位置
        position_ids_3d = generate_3d_positions_for_dat(
            batch_size=B,
            seq_len=kv_len,
            image_range_list=kv_image_range_list,
            sampling_locs=sampling_locs if len(sampling_locs) > 0 else None,
            lr_size=self.lr_size,
            hr_size=self.hr_size,
            device=device
        )
        
        # 3. 为 Query 生成对应的 3D 位置
        # ⚠️ 关键：Query 需要从 KV 的 3D 位置中索引
        # 因为 Query 和 KV 的序列长度不同（KV 插入了 HD features）
        position_ids_q = nn.utils.rnn.pad_sequence(pos_q_concat, batch_first=True, padding_value=0)
        
        # 使用 gather 索引 KV 的 3D 位置
        position_ids_q_index = position_ids_q.unsqueeze(-1).expand(-1, -1, 3)  # [B, Nq, 3]
        position_ids_q_3d = torch.gather(position_ids_3d, 1, position_ids_q_index)  # [B, Nq, 3]
        
        # 4. 应用 mRoPE
        query_rotated, key_rotated = apply_mrope_rotary_pos_emb(
            query_bhnc,
            key_bhnc,
            position_ids_q_3d,  # Query 用自己的 3D 位置
            self.rotary_emb
        )
        
        # 对 Key，我们需要用完整的 KV 3D 位置
        # 但由于 apply_mrope_rotary_pos_emb 需要相同 seq_len，我们需要分别处理
        key_full_3d = position_ids_3d  # [B, kv_len, 3]
        
        # Reshape key for full sequence
        key_temp = torch.zeros_like(key_bhnc)
        cos_k, sin_k = [], []
        
        # 为每个维度生成 embedding
        dim_per_axis = self.head_dim // 3
        for dim_idx, axis_idx in enumerate([0, 1, 2]):  # t, h, w
            start_dim = dim_idx * dim_per_axis
            end_dim = (dim_idx + 1) * dim_per_axis if dim_idx < 2 else self.head_dim
            
            key_slice = key_bhnc[:, :, :, start_dim:end_dim]
            cos, sin = self.rotary_emb(key_slice, key_full_3d[:, :, axis_idx])
            
            from transformers.models.llama.modeling_llama import rotate_half
            key_rot = key_slice * cos[:, None, :, :] + rotate_half(key_slice) * sin[:, None, :, :]
            key_temp[:, :, :, start_dim:end_dim] = key_rot
        
        key_rotated = key_temp
        
        return query_rotated, key_rotated
        
    elif Nq < kv_len:
        # ===== 原始 1D RoPE 路径 =====
        kv_position_ids = torch.arange(kv_len, device=device, dtype=torch.int64)[None, :]
        cos, sin = self.rotary_emb(value_bhnc, kv_position_ids)
        position_ids_q = nn.utils.rnn.pad_sequence(pos_q_concat, batch_first=True, padding_value=0)
        position_ids_q_index = position_ids_q[..., None].expand(-1, -1, self.head_dim)
        cos_q = cos.expand(B, -1, -1).gather(1, position_ids_q_index)
        sin_q = sin.expand(B, -1, -1).gather(1, position_ids_q_index)
        
        from transformers.models.llama.modeling_llama import rotate_half
        query_rotated = query_bhnc * cos_q[:, None, :, :] + rotate_half(query_bhnc) * sin_q[:, None, :, :]
        key_rotated = key_bhnc * cos[:, None, :, :] + rotate_half(key_bhnc) * sin[:, None, :, :]
        
        return query_rotated, key_rotated
    else:
        # ===== Nq == kv_len 情况 =====
        assert Nq == kv_len, "Nq should be the same as kv_len"
        cos, sin = self.rotary_emb(value_bhnc, position_ids)
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
        query_rotated, key_rotated = apply_rotary_pos_emb(query_bhnc, key_bhnc, cos, sin)
        return query_rotated, key_rotated


# 使用说明
"""
在 modeling_llava_dat.py 的 LlamaAttentionDAT.forward() 中：

替换原来的代码（line 559-585）:

```python
# 原代码：
if Nq < kv_len:
    kv_position_ids = torch.arange(kv_len, device=device, dtype=torch.int64)[None, :]
    cos, sin = self.rotary_emb(value_bhnc, kv_position_ids)
    # ... 等等
else:
    # ...

# 新代码：
query_bhnc, key_bhnc = integrate_mrope_into_dat_forward(
    self,
    query_bhnc, key_bhnc, value_bhnc,
    Nq, kv_len, image_range_list, sampling_locs, pos_q_concat, position_ids,
    device, B
)
```
"""
