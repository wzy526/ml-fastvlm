"""
mRoPE (Multi-Resolution Rotary Position Embedding) for Qwen2-VL DAT
基于 Qwen2-VL 的实现，适配 DAT 框架
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import math


class Qwen2VLRotaryEmbedding(nn.Module):
    """
    Multi-Resolution Rotary Position Embedding (mRoPE)
    
    将位置编码分成三个部分：
    - temporal: 时间/帧维度
    - height: 空间高度维度
    - width: 空间宽度维度
    """
    
    def __init__(self, config, device=None):
        super().__init__()
        
        self.config = config
        
        # mRoPE 配置
        rope_scaling = getattr(config, 'rope_scaling', None)
        if rope_scaling and rope_scaling.get('type') == 'mrope':
            self.mrope_section = rope_scaling.get('mrope_section', [16, 24, 24])
        else:
            # 默认配置
            self.mrope_section = [16, 24, 24]
        
        self.rope_theta = getattr(config, 'rope_theta', 1000000.0)
        
        # 计算每个部分的维度
        self.temporal_dim = self.mrope_section[0]
        self.height_dim = self.mrope_section[1]
        self.width_dim = self.mrope_section[2]
        self.total_dim = sum(self.mrope_section)
        
        # 验证维度
        head_dim = config.hidden_size // config.num_attention_heads
        if self.total_dim > head_dim:
            raise ValueError(
                f"mRoPE total dim ({self.total_dim}) exceeds head_dim ({head_dim})"
            )
        
        # 预计算频率
        self.register_buffer(
            "freq_temporal",
            self._compute_inv_freq(self.temporal_dim, self.rope_theta),
            persistent=False
        )
        self.register_buffer(
            "freq_height",
            self._compute_inv_freq(self.height_dim, self.rope_theta),
            persistent=False
        )
        self.register_buffer(
            "freq_width",
            self._compute_inv_freq(self.width_dim, self.rope_theta),
            persistent=False
        )
    
    def _compute_inv_freq(self, dim, theta):
        """计算逆频率"""
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        return inv_freq
    
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        grid_thw: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 输入 tensor [batch, num_heads, seq_len, head_dim]
            position_ids: 位置 ID
                - 如果是标准 1D: [batch, seq_len]
                - 如果是 3D: [batch, seq_len, 3] 包含 (t, h, w)
            grid_thw: 图像网格大小 [batch, 3] (temporal, height, width)
        
        Returns:
            cos, sin: 位置编码 [batch, seq_len, dim]
        """
        batch_size, seq_len = x.shape[0], x.shape[2]
        device = x.device
        
        # 如果没有提供 3D position_ids，使用标准 RoPE（降级模式）
        if position_ids.dim() == 2:
            return self._standard_rope(x, position_ids)
        
        # 3D position IDs: [batch, seq_len, 3]
        pos_t = position_ids[..., 0]  # temporal
        pos_h = position_ids[..., 1]  # height
        pos_w = position_ids[..., 2]  # width
        
        # 计算每个维度的位置编码
        # Temporal
        freqs_t = torch.outer(pos_t.flatten(), self.freq_temporal.to(device))
        cos_t = freqs_t.cos().view(batch_size, seq_len, -1)
        sin_t = freqs_t.sin().view(batch_size, seq_len, -1)
        
        # Height
        freqs_h = torch.outer(pos_h.flatten(), self.freq_height.to(device))
        cos_h = freqs_h.cos().view(batch_size, seq_len, -1)
        sin_h = freqs_h.sin().view(batch_size, seq_len, -1)
        
        # Width
        freqs_w = torch.outer(pos_w.flatten(), self.freq_width.to(device))
        cos_w = freqs_w.cos().view(batch_size, seq_len, -1)
        sin_w = freqs_w.sin().view(batch_size, seq_len, -1)
        
        # 拼接三个部分
        cos = torch.cat([cos_t, cos_h, cos_w], dim=-1)  # [batch, seq_len, total_dim]
        sin = torch.cat([sin_t, sin_h, sin_w], dim=-1)
        
        # 如果 head_dim > total_dim，用零填充剩余部分
        head_dim = x.shape[-1]
        if head_dim > self.total_dim:
            pad_dim = head_dim - self.total_dim
            cos = torch.cat([
                cos,
                torch.zeros(batch_size, seq_len, pad_dim, device=device, dtype=cos.dtype)
            ], dim=-1)
            sin = torch.cat([
                sin,
                torch.zeros(batch_size, seq_len, pad_dim, device=device, dtype=sin.dtype)
            ], dim=-1)
        
        return cos, sin
    
    def _standard_rope(self, x, position_ids):
        """降级到标准 RoPE（当没有 3D position_ids 时）"""
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
        
        if not hasattr(self, '_fallback_rope'):
            self._fallback_rope = Qwen2RotaryEmbedding(self.config)
        
        return self._fallback_rope(x, position_ids)


def generate_3d_position_ids(
    seq_len: int,
    grid_t: int,
    grid_h: int,
    grid_w: int,
    image_ranges: list,
    device: torch.device,
) -> torch.Tensor:
    """
    为包含图像的序列生成 3D position IDs
    
    Args:
        seq_len: 序列长度
        grid_t: temporal grid size
        grid_h: height grid size  
        grid_w: width grid size
        image_ranges: 图像token范围列表 [(start, end), ...]
        device: 设备
    
    Returns:
        position_ids: [1, seq_len, 3] 的 (t, h, w) 坐标
    """
    position_ids = torch.zeros(1, seq_len, 3, dtype=torch.long, device=device)
    
    # 初始化：所有文本 token 的位置为 (0, 0, position)
    for i in range(seq_len):
        position_ids[0, i, 2] = i  # width 维度用作序列位置
    
    # 为图像 token 设置 3D 坐标
    for start, end in image_ranges:
        num_patches = end - start
        patches_per_row = grid_w
        
        for idx in range(num_patches):
            abs_pos = start + idx
            
            # 计算 (t, h, w) 坐标
            t = idx // (grid_h * grid_w)
            hw = idx % (grid_h * grid_w)
            h = hw // grid_w
            w = hw % grid_w
            
            position_ids[0, abs_pos, 0] = t
            position_ids[0, abs_pos, 1] = h
            position_ids[0, abs_pos, 2] = w
    
    return position_ids


# 使用示例
"""
# 1. 在模型初始化时使用 mRoPE
if config.rope_scaling and config.rope_scaling.get('type') == 'mrope':
    self.rotary_emb = Qwen2VLRotaryEmbedding(config)
else:
    self.rotary_emb = Qwen2RotaryEmbedding(config)

# 2. 在前向传播时生成 3D position IDs
if hasattr(self, 'rotary_emb') and isinstance(self.rotary_emb, Qwen2VLRotaryEmbedding):
    # 生成 3D position IDs
    position_ids_3d = generate_3d_position_ids(
        seq_len=seq_len,
        grid_t=1,  # 静态图像
        grid_h=24,  # 图像高度网格
        grid_w=24,  # 图像宽度网格
        image_ranges=[(10, 586)],  # 图像 token 范围
        device=device
    )
    cos, sin = self.rotary_emb(hidden_states, position_ids_3d)
else:
    # 使用标准 RoPE
    cos, sin = self.rotary_emb(hidden_states, position_ids)
"""
