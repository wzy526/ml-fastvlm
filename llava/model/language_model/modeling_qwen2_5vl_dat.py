"""
Qwen2.5VL-DAT: Dynamic Attention Token extension for Qwen2.5VL.

Extends official Qwen2_5_VLForConditionalGeneration with DAT mechanism:
- Offset-based sampling from high-resolution vision features
- Per-layer HD feature injection via modified attention
- Proper 3D mRoPE position encoding for HD tokens

Architecture (transformers >= 5.2.0):
    Qwen2_5_VLDATForConditionalGeneration(Qwen2_5_VLForConditionalGeneration)
    ├── model: Qwen2_5_VLModel (unmodified — kwargs flow natively)
    │   ├── visual: Qwen2_5_VisionTransformerPretrainedModel  (shared for LR & HR)
    │   └── language_model: Qwen2_5_VLTextModel (unmodified)
    │       └── layers: mixed 'L' (standard) + 'D' (DAT) decoder layers
    └── lm_head: nn.Linear

HD kwargs flow: ForConditionalGeneration → Model(**kwargs) → TextModel(**kwargs)
    → DecoderLayer(**kwargs) → AttentionDAT(**kwargs)

Attention mechanism: Two-pass + LSE merge (GC-safe, shape-static):
    Pass 1: standard causal attention (full sequence, shape = [B, H, Nq, D])
    Pass 2: HD cross-attention per answer segment (Q_ans × K_hd, non-causal)
    Merge:  o* = exp(ℓ₁−ℓ)·o₁ + exp(ℓ₂−ℓ)·o₂,  ℓ = logaddexp(ℓ₁, ℓ₂)
    → No mask matrix, no pad_sequence, no data-dependent shapes.
    → Fully compatible with gradient checkpointing and torch.compile.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLModel,
    Qwen2_5_VLDecoderLayer,
    Qwen2_5_VLAttention,
    Qwen2_5_VLRotaryEmbedding,
    Qwen2_5_VLCausalLMOutputWithPast,
    apply_multimodal_rotary_pos_emb,
    rotate_half,
    repeat_kv,
)
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
from transformers.cache_utils import Cache

# Two-pass attention + LSE merging approach for DAT:
#
# Instead of building an extended KV sequence with a data-dependent mask
# (which breaks torch.compile and gradient checkpointing), we compute:
#   Pass 1: standard causal attention over the full Nq-length sequence
#   Pass 2: HD cross-attention for answer tokens × HD KV (non-causal, per-segment)
# Then merge via the LSE (log-sum-exp) trick:
#   ℓ = logaddexp(ℓ₁, ℓ₂)
#   o* = exp(ℓ₁ − ℓ) · o₁ + exp(ℓ₂ − ℓ) · o₂
#
# This is mathematically equivalent to joint attention over both KV sets,
# has static shapes (no padding, no mask matrix), and is fully GC-compatible.
#
# flash_attn (if available) returns LSE directly; otherwise we fall back to
# a manual O(N²) SDPA implementation that also computes LSE.
try:
    from flash_attn import flash_attn_func as _flash_attn_func
    _FLASH_ATTN_AVAILABLE = True
except (ImportError, Exception):
    _flash_attn_func = None
    _FLASH_ATTN_AVAILABLE = False


def _dat_attn_with_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute attention output and log-sum-exp (LSE).

    Args:
        q, k, v: [B, H, N, D]  (internal multi-head layout)
        causal:  apply causal mask

    Returns:
        out: [B, N, H, D]  (transposed — ready for position-slicing)
        lse: [B, H, N]
    """
    if _FLASH_ATTN_AVAILABLE:
        # flash_attn expects / returns [B, N, H, D]; lse is [B, H, N]
        out_fa, lse, _ = _flash_attn_func(
            q.transpose(1, 2).contiguous(),
            k.transpose(1, 2).contiguous(),
            v.transpose(1, 2).contiguous(),
            causal=causal,
            return_attn_probs=True,
        )
        return out_fa, lse  # out_fa: [B, N, H, D], lse: [B, H, N]

    # Manual fallback: O(N²) memory, but GC-safe and always correct
    B, H, Nq, D = q.shape
    scale = D ** -0.5
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale  # [B, H, Nq, Nk]
    if causal:
        causal_mask = torch.triu(
            torch.ones(Nq, k.size(2), device=q.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, float('-inf'))
    lse = torch.logsumexp(scores.float(), dim=-1)          # [B, H, Nq]
    attn_w = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    out = torch.matmul(attn_w, v).transpose(1, 2)          # [B, Nq, H, D]
    return out, lse


logger = logging.getLogger(__name__)

# Qwen2.5-VL special token IDs (same as Qwen2-VL)
IM_START_TOKEN_ID = 151644   # <|im_start|>


def _find_im_start_backward(ids, ans_start, im_start_token_id=IM_START_TOKEN_ID):
    """Scan backward from ans_start to find the nearest <|im_start|> token."""
    for pos in range(ans_start - 1, -1, -1):
        if ids[pos].item() == im_start_token_id:
            return pos
    raise ValueError(
        f"No <|im_start|> (token {im_start_token_id}) found before position {ans_start}. "
        f"Check that input follows ChatML format."
    )


# ============================================================================
# Config
# ============================================================================

class Qwen2_5_VLDATConfig(Qwen2_5_VLConfig):
    """Qwen2.5VL config extended with DAT parameters."""
    model_type = "qwen2_5_vl_dat"

    def __init__(self, dat_extra_args=None, **kwargs):
        super().__init__(**kwargs)
        self.dat_extra_args = dat_extra_args or {
            'grid_size': 6,            # DAT sampling grid size
            'off_ksize': 3,            # Offset conv kernel size
            'off_grps': 1,             # Offset groups (must divide hidden_size)
            'inter_size': 64,          # Intermediate size for offset generation
            'hr_scale': 3,             # Upscale factor for HD features
            'hd_proj': True,           # Use separate k/v projection for HD
            'layers': '',              # Layer type string, e.g. "LLLDLLLD..."
            'use_intention_branch': True,
            'intention_as_gate': True,
            'hd_attn_bias': 0.0,       # <0 to enable learnable HD attention bias (init value)
        }


# ============================================================================
# Helpers
# ============================================================================

class LayerNorm2d(nn.Module):
    """Channel-first LayerNorm for 2D feature maps [B, C, H, W]."""
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[None, :, None, None] * x + self.bias[None, :, None, None]
        return x


def apply_multimodal_rotary_pos_emb_single(x, cos, sin, mrope_section, unsqueeze_dim=1):
    """Apply mRoPE to a single tensor (Q or K separately).

    Needed when Q and KV have different sequence lengths in DAT layers.
    Same cycling logic as apply_multimodal_rotary_pos_emb but for one tensor.

    Args:
        x: [batch, heads, seq_len, head_dim]
        cos: [3, batch, seq_len, head_dim]
        sin: [3, batch, seq_len, head_dim]
        mrope_section: list of ints, e.g. [16, 24, 24]
    """
    mrope_section_doubled = [s * 2 for s in mrope_section]
    cos = torch.cat(
        [m[i % 3] for i, m in enumerate(cos.split(mrope_section_doubled, dim=-1))],
        dim=-1
    ).unsqueeze(unsqueeze_dim)
    sin = torch.cat(
        [m[i % 3] for i, m in enumerate(sin.split(mrope_section_doubled, dim=-1))],
        dim=-1
    ).unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


def compute_image_range_list(input_ids, labels, image_token_id,
                              im_start_token_id=IM_START_TOKEN_ID,
                              image_grid_thw=None, spatial_merge_size=2):
    """Compute image_range_list from Qwen2.5VL-format inputs.

    Scans input_ids for image token regions and labels for answer regions.
    For each answer range, dynamically locates the preceding <|im_start|>
    token as the intention_idx.

    Args:
        input_ids: [B, seq_len]
        labels: [B, seq_len] or None (inference mode)
        image_token_id: int
        im_start_token_id: int, token ID for <|im_start|>
        image_grid_thw: [num_images, 3] grid dimensions (t, h, w) per image
        spatial_merge_size: spatial merge factor (default 2)

    Returns:
        List per batch of:
            [(lr_start, lr_end, lr_h, lr_w), [ans1_start, ans1_end, intention_idx], ...]
        Empty list for batch items without images.
    """
    batch_size = input_ids.shape[0]
    result = []
    img_idx = 0

    for b in range(batch_size):
        ids = input_ids[b]
        ranges = []

        image_mask = (ids == image_token_id)
        if not image_mask.any():
            result.append(ranges)
            continue

        image_indices = torch.where(image_mask)[0]
        lr_start = image_indices[0].item()
        lr_end = image_indices[-1].item() + 1

        if image_grid_thw is not None and img_idx < len(image_grid_thw):
            thw = image_grid_thw[img_idx]
            t, h, w = thw[0].item(), thw[1].item(), thw[2].item()
            lr_h = h // spatial_merge_size
            lr_w = w // spatial_merge_size
            img_idx += 1
        else:
            lr_len = lr_end - lr_start
            lr_h = lr_w = int(lr_len ** 0.5)

        ranges.append((lr_start, lr_end, lr_h, lr_w))

        if labels is not None:
            lab = labels[b]
            ans_mask = (lab != -100)
            if ans_mask.any():
                ans_indices = torch.where(ans_mask)[0]
                seg_start = ans_indices[0].item()
                for i in range(1, len(ans_indices)):
                    if ans_indices[i] - ans_indices[i - 1] > 1:
                        seg_end = ans_indices[i - 1].item()
                        intention_idx = _find_im_start_backward(ids, seg_start, im_start_token_id)
                        ranges.append([seg_start, seg_end, intention_idx])
                        seg_start = ans_indices[i].item()
                intention_idx = _find_im_start_backward(ids, seg_start, im_start_token_id)
                ranges.append([seg_start, ans_indices[-1].item(), intention_idx])
        else:
            seq_len = ids.shape[0]
            intention_idx = _find_im_start_backward(ids, seq_len, im_start_token_id)
            ranges.append([seq_len, -1, intention_idx])

        result.append(ranges)

    return result


# ============================================================================
# DAT Attention
# ============================================================================

class Qwen2_5_VLAttentionDAT(Qwen2_5_VLAttention):
    """
    Core DAT mechanism for Qwen2.5-VL (two-pass + LSE merge):
    1. Extract LR features from query → generate sampling offsets
    2. Grid sample from HD features → project to KV (key_hd, value_hd)
    3. Pass 1: standard causal attention (full sequence, static shapes)
    4. Pass 2: HD cross-attention per answer segment (Q_ans × K_hd, non-causal)
    5. Merge outputs via LSE trick — mathematically equivalent to joint attention
    """

    def __init__(self, config, layer_idx: int, dat_extra_args: dict):
        super().__init__(config, layer_idx)

        dat = dat_extra_args
        self.grid_size = dat['grid_size']
        self.off_ksize = dat['off_ksize']
        self.off_grps = dat['off_grps']
        self.inter_size = dat['inter_size']
        self.hd_proj = dat['hd_proj']
        self.intention_as_gate = dat['intention_as_gate']
        self.use_intention_branch = dat['use_intention_branch']

        _hd_attn_bias_init = float(dat.get('hd_attn_bias', 0.0))
        if _hd_attn_bias_init < 0:
            self.hd_attn_bias = nn.Parameter(torch.zeros(1).fill_(_hd_attn_bias_init))
        else:
            self.hd_attn_bias = None

        self.off_dim = self.hidden_size // self.off_grps

        # --- Offset generation pipeline ---
        self.conv_lr_dw = nn.Conv2d(
            self.off_dim, self.off_dim,
            kernel_size=self.off_ksize, stride=1, padding=self.off_ksize // 2,
            groups=self.off_dim, bias=False,
        )
        self.ln_1 = LayerNorm2d(self.off_dim)
        self.conv_lr_proj = nn.Conv2d(
            self.off_dim, self.inter_size,
            kernel_size=1, stride=1, padding=0,
        )

        # Intention branch
        if self.use_intention_branch:
            self.proj_intention = nn.Linear(self.off_dim, self.inter_size)
        else:
            self.proj_intention = nn.Identity()

        # Offset prediction
        if self.intention_as_gate:
            self.ln_2 = LayerNorm2d(self.inter_size)
            self.conv_off_proj = nn.Conv2d(
                self.inter_size, 2, kernel_size=1, stride=1, padding=0, bias=False,
            )
        else:
            self.ln_2 = LayerNorm2d(self.inter_size * 2)
            self.conv_off_proj = nn.Conv2d(
                self.inter_size * 2, 2, kernel_size=1, stride=1, padding=0, bias=False,
            )

        # HD feature KV projection
        if self.hd_proj:
            kv_dim = self.num_key_value_heads * self.head_dim
            self.k_proj_hd = nn.Linear(self.hidden_size, kv_dim)
            self.v_proj_hd = nn.Linear(self.hidden_size, kv_dim)
        else:
            self.k_proj_hd = None
            self.v_proj_hd = None

        # Rotary embedding for extended KV (Qwen2.5-VL Attention doesn't have one)
        self._dat_rotary_emb = Qwen2_5_VLRotaryEmbedding(config=config)

        self._init_dat_weights()

    @torch.no_grad()
    def _init_dat_weights(self):
        # Conv layers: Kaiming normal
        nn.init.kaiming_normal_(self.conv_lr_dw.weight)
        nn.init.kaiming_normal_(self.conv_lr_proj.weight)
        nn.init.kaiming_normal_(self.conv_off_proj.weight)
        if self.conv_lr_proj.bias is not None:
            nn.init.zeros_(self.conv_lr_proj.bias)
        # Linear layers: Xavier uniform
        if isinstance(self.proj_intention, nn.Linear):
            nn.init.xavier_uniform_(self.proj_intention.weight)
            if self.proj_intention.bias is not None:
                nn.init.zeros_(self.proj_intention.bias)
        if self.k_proj_hd is not None:
            nn.init.xavier_uniform_(self.k_proj_hd.weight)
            nn.init.xavier_uniform_(self.v_proj_hd.weight)
            if self.k_proj_hd.bias is not None:
                nn.init.zeros_(self.k_proj_hd.bias)
            if self.v_proj_hd.bias is not None:
                nn.init.zeros_(self.v_proj_hd.bias)

    def _grid_generate(self, h, w, n_repeats, device):
        """Generate reference sampling grid in [-1, 1]."""
        grid_y = torch.linspace(-1, 1, h, device=device, dtype=torch.float32)
        grid_x = torch.linspace(-1, 1, w, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=0)  # [2, H, W]
        return grid.unsqueeze(0).repeat(n_repeats * self.off_grps, 1, 1, 1)

    def _construct_hd_position_ids(self, pos_3d_b, lr_start, lr_end, lr_h, lr_w, device):
        """Construct proper 3D mRoPE position IDs for HD tokens.

        Maps HD sampling grid positions into the original coordinate system
        used by get_rope_index. This ensures HD tokens have positions that are
        consistent with the LR image tokens' coordinate system.

        Args:
            pos_3d_b: [3, Nq] — original 3D mRoPE positions for this batch element
            lr_start: start index of LR image tokens in the sequence
            lr_end: end index of LR image tokens in the sequence
            lr_h: LR image height in merged patches
            lr_w: LR image width in merged patches
            device: torch device

        Returns:
            hd_pos: [3, Ns] — 3D position IDs for HD tokens (grid_size * grid_size)
        """
        Ns = self.grid_size * self.grid_size

        # Extract LR image position IDs
        lr_pos = pos_3d_b[:, lr_start:lr_end]  # [3, lr_h * lr_w]

        # Temporal: constant (same as LR image tokens)
        t_val = lr_pos[0, 0]

        # Height: get range from LR positions
        h_min = lr_pos[1].min()
        h_max = lr_pos[1].max()

        # Width: get range from LR positions
        w_min = lr_pos[2].min()
        w_max = lr_pos[2].max()

        # Map grid coordinates to position ID space
        # Use linspace [0, 1] then scale to [min, max]
        grid_y = torch.linspace(0, 1, self.grid_size, device=device, dtype=torch.float32)
        grid_x = torch.linspace(0, 1, self.grid_size, device=device, dtype=torch.float32)

        h_hd = (grid_y * (h_max - h_min) + h_min).long()
        w_hd = (grid_x * (w_max - w_min) + w_min).long()

        # Expand to full grid [grid_size, grid_size] -> flatten to [Ns]
        h_grid = h_hd.unsqueeze(1).expand(-1, self.grid_size).flatten()
        w_grid = w_hd.unsqueeze(0).expand(self.grid_size, -1).flatten()
        t_grid = t_val.expand(Ns).long()

        return torch.stack([t_grid, h_grid, w_grid])  # [3, Ns]

    def _generate_offsets_and_sample(self, query_states, image_hd_features, image_range_list, b_idx, hd_feat_idx):
        """Generate sampling offsets from LR queries and sample from HD features.

        Args:
            query_states: [B, Nq, hidden_size]
            image_hd_features: list of [H_hr, W_hr, C] per image
            image_range_list: per-batch range info
            b_idx: batch index
            hd_feat_idx: index into image_hd_features

        Returns:
            key_hd: [Lp, Ns, kv_dim]
            value_hd: [Lp, Ns, kv_dim]
            sampling_locs: [Lp, off_grps, grid_h, grid_w, 2]
        """
        device = query_states.device
        Ns = self.grid_size * self.grid_size

        # 1. Extract LR image region from Q
        lr_start, lr_end, lr_h, lr_w = image_range_list[b_idx][0]
        lr_len = lr_end - lr_start
        assert lr_h * lr_w == lr_len, (
            f"LR dimensions mismatch: {lr_h}*{lr_w}={lr_h * lr_w} != {lr_len} tokens"
        )

        image_range_index = torch.arange(lr_start, lr_end, device=device)
        img_lr = einops.rearrange(
            query_states[b_idx, image_range_index],
            '(h w) (g c) -> g c h w',
            g=self.off_grps, c=self.off_dim, h=lr_h, w=lr_w,
        )
        # 2. Offset generation: DW Conv -> LN -> SiLU -> Conv1x1 -> Pool
        local_embed_lr = self.conv_lr_dw(img_lr)
        local_embed_lr = F.silu(self.ln_1(local_embed_lr))
        embed_lr = self.conv_lr_proj(local_embed_lr)
        embed_lr = F.adaptive_avg_pool2d(embed_lr, (self.grid_size, self.grid_size))

        # 3. Per-answer offset prediction
        answer_ranges = image_range_list[b_idx][1:]
        Lp = len(answer_ranges)

        if self.use_intention_branch:
            intention_indices = [ar[2] for ar in answer_ranges]
            intention_tokens = query_states[b_idx, intention_indices]
            intention_per_group = einops.rearrange(
                intention_tokens, 'l (g c) -> l g c',
                g=self.off_grps, c=self.off_dim,
            )
            embed_intention = self.proj_intention(intention_per_group)
            embed_intention = einops.rearrange(
                einops.repeat(
                    embed_intention, 'l g c -> l g c h w',
                    h=self.grid_size, w=self.grid_size,
                ),
                'l g c h w -> (l g) c h w',
            )

        embed_lr_rep = einops.rearrange(
            einops.repeat(embed_lr, 'g c h w -> l g c h w', l=Lp),
            'l g c h w -> (l g) c h w',
        )

        if self.use_intention_branch:
            if self.intention_as_gate:
                gate = embed_intention.sigmoid()
                off_guide = embed_lr_rep * (gate * 2.0)
                if self.training:
                    self._dat_gate_stats = (
                        gate.detach().mean().item(),
                        gate.detach().std().item(),
                    )
            else:
                off_guide = torch.cat([embed_lr_rep, embed_intention], dim=1)
        else:
            off_guide = embed_lr_rep

        # 4. Predict offsets
        offsets = self.conv_off_proj(F.silu(self.ln_2(off_guide))).float() # fp32 offsets
        if self.training:
            self._dat_offset_stats = (
                offsets.detach().mean().item(),
                offsets.detach().std().item(),
            )
        references = self._grid_generate(offsets.size(2), offsets.size(3), Lp, device) # fp32 grid
        sample_locs = (references + offsets).clamp(-1., 1.).permute(0, 2, 3, 1) # fp32 sample_locs
        
        # 5. Grid sample from HD features
        hd_feat = image_hd_features[hd_feat_idx]  # [H_hr, W_hr, C]
        img_hr = einops.rearrange(
            hd_feat, 'h w (g c) -> g c h w',
            g=self.off_grps, c=self.off_dim,
        )
        img_hr = einops.repeat(img_hr, 'g c h w -> (l g) c h w', l=Lp)

        orig_dtype = img_hr.dtype
        sampled_hr = F.grid_sample(
            img_hr.float(), sample_locs[..., (1, 0)],
            mode='bilinear', align_corners=True,
        ).to(orig_dtype) # bf16 sampled_hr

        sampled_hr = einops.rearrange(
            sampled_hr,
            '(l g) c h w -> l (h w) (g c)',
            l=Lp, g=self.off_grps, c=self.off_dim,
        )

        # 6. Project to KV
        if self.hd_proj:
            key_hd = self.k_proj_hd(sampled_hr)
            value_hd = self.v_proj_hd(sampled_hr)
        else:
            key_hd = self.k_proj(sampled_hr)
            value_hd = self.v_proj(sampled_hr)

        sampling_locs_out = sample_locs.reshape(
            Lp, self.off_grps, self.grid_size, self.grid_size, 2
        ).clone().detach()

        return key_hd, value_hd, sampling_locs_out

    def _merge_two_pass_lse(
        self,
        out1: torch.Tensor,
        lse1: torch.Tensor,
        out2: torch.Tensor,
        lse2: torch.Tensor,
        ans_start: int,
        ans_end: int,
    ) -> torch.Tensor:
        """Merge causal attention and HD cross-attention outputs via the LSE trick.

        For answer tokens [ans_start, ans_end), the joint attention over
        S₁ (causal KV) ∪ S₂ (HD KV) equals:
            o* = exp(ℓ₁ − ℓ) · o₁ + exp(ℓ₂ − ℓ) · o₂
        where ℓ = logaddexp(ℓ₁, ℓ₂).  Weights sum to 1 exactly. ✓

        Critical: out2 / lse2 must use the SAME q as Pass 1 (already mRoPE'd),
        not re-projected from hidden_states, so s₁ and s₂ share the same query.

        Args:
            out1: [B, Nq, H, D]  — causal attention output (full sequence)
            lse1: [B, H, Nq]     — causal log-sum-exp
            out2: [1, Nans, H, D] — HD cross-attention output for answer slice
            lse2: [1, H, Nans]   — HD cross-attention log-sum-exp
            ans_start, ans_end: slice of the answer range in the sequence

        Returns:
            out: [B, Nq, H, D] with merged values written at [ans_start:ans_end]
        """
        out = out1.clone()

        lse1_ans = lse1[:, :, ans_start:ans_end]          # [B, H, Nans]
        out1_ans = out1[:, ans_start:ans_end, :, :]        # [B, Nans, H, D]

        lse = torch.logaddexp(lse1_ans, lse2)              # [B, H, Nans]

        # Weights: [B, H, Nans] → [B, Nans, H, 1] for broadcasting
        w1 = (lse1_ans - lse).exp().permute(0, 2, 1).unsqueeze(-1)
        w2 = (lse2     - lse).exp().permute(0, 2, 1).unsqueeze(-1)

        out[:, ans_start:ans_end, :, :] = w1 * out1_ans + w2 * out2

        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        # DAT-specific kwargs (flow through **kwargs chain)
        image_hd_features: Optional[List[torch.Tensor]] = None,
        image_range_list: Optional[List[List]] = None,
        mrope_position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        # --- Standard path: no image data ---
        if image_range_list is None or image_hd_features is None:
            return super().forward(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        has_images = any(len(r) > 0 for r in image_range_list)
        if not has_images:
            return super().forward(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        # Inference decode phase: HD already cached during prefill
        if use_cache and past_key_values is not None and past_key_values.get_seq_length(self.layer_idx) > 0:
            return super().forward(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        # --- DAT path (two-pass + LSE merge) ---
        # Pass 1: standard causal attention on the full Nq-length sequence (static shapes).
        # Pass 2: per-answer-segment HD cross-attention (Q_ans × K_hd, non-causal).
        # Merge via LSE trick — mathematically equivalent to joint attention, GC-safe.
        B, Nq, C = hidden_states.size()
        device = hidden_states.device
        Ns = self.grid_size * self.grid_size
        mrope_section = self.config.rope_parameters["mrope_section"]

        # Project Q, K, V
        query_states = self.q_proj(hidden_states)   # [B, Nq, C]
        key_states   = self.k_proj(hidden_states)   # [B, Nq, kv_dim]
        value_states = self.v_proj(hidden_states)   # [B, Nq, kv_dim]

        # Handle inference: add dummy answer range if compute_image_range_list wasn't called
        if use_cache:
            for b in range(B):
                if len(image_range_list[b]) == 1:
                    image_range_list[b].append([Nq, -1, max(0, Nq - 3)])

        # Build mapping from batch index to image_hd_features index
        b_idx_to_hd_idx: Dict[int, int] = {}
        hd_idx = 0
        for b in range(B):
            if len(image_range_list[b]) > 0:
                b_idx_to_hd_idx[b] = hd_idx
                hd_idx += 1

        # === Pass 1: standard causal attention (full sequence, shape static) ===
        # mRoPE — same as the non-DAT path (position_embeddings are Nq-length)
        query_bhnc = query_states.view(B, Nq, self.num_heads,           self.head_dim).transpose(1, 2)
        key_bhnc   = key_states  .view(B, Nq, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_bhnc = value_states.view(B, Nq, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_bhnc, key_bhnc = apply_multimodal_rotary_pos_emb(
            query_bhnc, key_bhnc, cos, sin, mrope_section,
        )

        # KV cache: always standard Nq-length (HD KV is never cached)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_bhnc, value_bhnc = past_key_values.update(
                key_bhnc, value_bhnc, self.layer_idx, cache_kwargs,
            )

        key_bhnc   = repeat_kv(key_bhnc,   self.num_key_value_groups)
        value_bhnc = repeat_kv(value_bhnc, self.num_key_value_groups)

        if query_bhnc.device.type == "cuda":
            query_bhnc = query_bhnc.contiguous()
            key_bhnc   = key_bhnc.contiguous()
            value_bhnc = value_bhnc.contiguous()

        # Causal attention + LSE  →  out1: [B, Nq, H, D],  lse1: [B, H, Nq]
        out1, lse1 = _dat_attn_with_lse(query_bhnc, key_bhnc, value_bhnc, causal=True)

        _want_vis = self.training and getattr(self, '_dat_request_vis', False)
        _dat_vis_entry = None  # (b_idx, slocs) for visualization

        # === Pass 2: HD cross-attention per batch element, per answer segment ===
        # We accumulate per-batch outputs in a list to avoid in-place modification of
        # out1 (flash_attn saves out in ctx.saved_tensors; writing to out1 in-place
        # would corrupt the saved tensor and cause incorrect backward gradients).
        out_parts: List[torch.Tensor] = []

        for b_idx in range(B):
            # Start with Pass 1 output for this batch element (view, no copy yet)
            out_b = out1[b_idx:b_idx + 1]   # [1, Nq, H, D]

            if len(image_range_list[b_idx]) <= 1:
                out_parts.append(out_b)
                continue  # No HD for this sample; Pass 1 result is final

            hd_feat_idx = b_idx_to_hd_idx[b_idx]
            lr_start, lr_end, lr_h, lr_w = image_range_list[b_idx][0]

            # Get 3D mRoPE positions for this batch element
            if mrope_position_ids is not None:
                orig_pos_b = mrope_position_ids[:, b_idx, :]  # [3, Nq]
            else:
                orig_pos_b = torch.arange(Nq, device=device, dtype=torch.long).unsqueeze(0).expand(3, -1)

            # HD position IDs (shared across all answer segments for this image)
            hd_pos_ids = self._construct_hd_position_ids(
                orig_pos_b, lr_start, lr_end, lr_h, lr_w, device,
            )  # [3, Ns]
            hd_pos_ids_batched = hd_pos_ids.unsqueeze(1)  # [3, 1, Ns]

            # Generate all HD KV for this batch element (all Lp segments at once)
            key_hd_all, value_hd_all, _slocs = self._generate_offsets_and_sample(
                query_states, image_hd_features, image_range_list, b_idx, hd_feat_idx,
            )
            # key_hd_all, value_hd_all: [Lp, Ns, kv_dim]

            if _want_vis and _dat_vis_entry is None:
                _dat_vis_entry = (b_idx, _slocs)

            for l_idx, answer_range in enumerate(image_range_list[b_idx][1:]):
                ans_start     = answer_range[0]
                ans_end       = answer_range[1]
                intention_idx = answer_range[2]

                if ans_end > 0:
                    # Training: answer tokens are [ans_start, ans_end)
                    q_ans_start = ans_start
                    Nans = ans_end - ans_start
                else:
                    # Inference prefill: tokens after intention marker attend to HD.
                    # ans_start = seq_len here; use intention_idx+1 as the actual start.
                    q_ans_start = intention_idx + 1
                    Nans = Nq - q_ans_start

                if Nans <= 0:
                    continue

                # q_ans: taken from Pass 1's mRoPE'd query — MUST be the same q as Pass 1.
                # (Re-projecting from hidden_states would break LSE merge correctness.)
                q_ans = query_bhnc[b_idx:b_idx + 1, :, q_ans_start:q_ans_start + Nans, :]
                # q_ans: [1, H, Nans, D]

                # HD KV for this answer segment: [1, H, Ns, D]
                k_hd_l = (key_hd_all[l_idx:l_idx + 1]
                           .view(1, Ns, self.num_key_value_heads, self.head_dim)
                           .transpose(1, 2))
                v_hd_l = (value_hd_all[l_idx:l_idx + 1]
                           .view(1, Ns, self.num_key_value_heads, self.head_dim)
                           .transpose(1, 2))

                # Apply mRoPE to HD KV using image coordinate position IDs
                cos_hd, sin_hd = self._dat_rotary_emb(k_hd_l, hd_pos_ids_batched)
                k_hd_l = apply_multimodal_rotary_pos_emb_single(
                    k_hd_l, cos_hd, sin_hd, mrope_section,
                )

                k_hd_l = repeat_kv(k_hd_l, self.num_key_value_groups)  # [1, H, Ns, D]
                v_hd_l = repeat_kv(v_hd_l, self.num_key_value_groups)

                if q_ans.device.type == "cuda":
                    q_ans  = q_ans.contiguous()
                    k_hd_l = k_hd_l.contiguous()
                    v_hd_l = v_hd_l.contiguous()

                # Cross-attention: Q_ans × K_hd (non-causal — all Ns HD tokens visible)
                # out2: [1, Nans, H, D],  lse2: [1, H, Nans]
                out2, lse2 = _dat_attn_with_lse(q_ans, k_hd_l, v_hd_l, causal=False)

                # _merge_two_pass_lse clones out_b internally → returns a NEW tensor
                # (no in-place modification of the flash_attn output).
                # For Lp > 1: out_b chains: view(out1) → clone1 → clone2 → ...
                out_b = self._merge_two_pass_lse(
                    out_b, lse1[b_idx:b_idx + 1],
                    out2, lse2,
                    q_ans_start, q_ans_start + Nans,
                )

            out_parts.append(out_b)

        # Re-assemble batch.  torch.cat handles mixed views (unmerged) and new tensors
        # (merged) correctly, distributing gradients to the appropriate source.
        out_final = torch.cat(out_parts, dim=0)  # [B, Nq, H, D]

        # Visualization: record sampling locations (no attention-map for two-pass path)
        if _want_vis and _dat_vis_entry is not None:
            self._dat_request_vis = False
            b_idx_v, slocs_v = _dat_vis_entry
            self._dat_vis_data = (slocs_v, None)
            self._dat_vis_b_idx = b_idx_v

        # [B, Nq, H, D]  →  [B, Nq, C]
        attn_output = out_final.reshape(B, Nq, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, None


# ============================================================================
# DAT Decoder Layer
# ============================================================================

class Qwen2_5_VLDecoderLayerDAT(Qwen2_5_VLDecoderLayer):
    """Decoder layer with DAT attention.

    The base class already passes **kwargs to self_attn, so DAT-specific kwargs
    (image_hd_features, image_range_list, mrope_position_ids) flow through
    automatically.
    """

    def __init__(self, config, layer_idx: int, dat_extra_args: dict):
        super().__init__(config, layer_idx)
        # Replace standard attention with DAT attention
        self.self_attn = Qwen2_5_VLAttentionDAT(config, layer_idx, dat_extra_args)
        # DAT layers always use full attention (not sliding window)
        self.attention_type = "full_attention"


# ============================================================================
# DAT ForConditionalGeneration (top-level model)
# ============================================================================

class Qwen2_5_VLDATForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    """Qwen2.5VL with DAT: uses native vision encoder for both LR and HD features.

    The model/language_model/decoder layers are NOT subclassed. DAT kwargs flow
    through the native **kwargs chain to reach DAT attention layers.
    """
    config_class = Qwen2_5_VLDATConfig

    def __init__(self, config: Qwen2_5_VLDATConfig):
        super().__init__(config)

        # Replace specified decoder layers with DAT layers
        dat_args = config.dat_extra_args
        layers_str = dat_args.get('layers', '')
        text_config = config.text_config

        if layers_str:
            assert len(layers_str) == text_config.num_hidden_layers, (
                f"Layer string length {len(layers_str)} != num_hidden_layers {text_config.num_hidden_layers}"
            )
            for i, layer_type in enumerate(layers_str):
                if layer_type == 'D':
                    self.model.language_model.layers[i] = Qwen2_5_VLDecoderLayerDAT(
                        text_config, i, dat_args
                    )
                elif layer_type == 'L':
                    pass
                else:
                    raise ValueError(f"Unknown layer type '{layer_type}' at index {i}")

            dat_count = sum(1 for c in layers_str if c == 'D')
            logger.info(f"Qwen2.5VL-DAT: {dat_count} DAT layers, "
                        f"{text_config.num_hidden_layers - dat_count} standard layers")

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        is_first_iteration=False,
        pixel_values_hd=None,
        image_grid_thw_hd=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

        if is_first_iteration:
            model_inputs["pixel_values_hd"] = pixel_values_hd
            model_inputs["image_grid_thw_hd"] = image_grid_thw_hd
        else:
            model_inputs["pixel_values_hd"] = None
            model_inputs["image_grid_thw_hd"] = None

        return model_inputs

    def _generate_hd_features(self, pixel_values_hd, image_grid_thw_hd):
        """Generate HD feature maps from high-resolution pixel values."""
        with torch.no_grad():
            pixel_values_hd = pixel_values_hd.type(self.model.visual.dtype)
            hd_output = self.model.visual(pixel_values_hd, grid_thw=image_grid_thw_hd, return_dict=True)
            hd_embeds = hd_output.pooler_output

        spatial_merge = self.config.vision_config.spatial_merge_size
        image_hd_features = []
        offset = 0

        for thw in image_grid_thw_hd:
            t = thw[0].item()
            h_merged = thw[1].item() // spatial_merge
            w_merged = thw[2].item() // spatial_merge
            n_patches = t * h_merged * w_merged

            feat = hd_embeds[offset:offset + n_patches]
            if t > 1:
                logger.warning(
                    f"HD features: video with t={t} detected, using first frame only. "
                    f"DAT is designed for single-frame images."
                )
            feat = feat[:h_merged * w_merged].view(h_merged, w_merged, -1)
            image_hd_features.append(feat)
            offset += n_patches

        return image_hd_features

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        # DAT-specific
        pixel_values_hd: Optional[torch.Tensor] = None,
        image_grid_thw_hd: Optional[torch.LongTensor] = None,
        image_hd_features: Optional[List[torch.Tensor]] = None,
        image_range_list: Optional[List[List]] = None,
        **kwargs,
    ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
        """Forward pass with DAT HD feature injection.

        In addition to standard Qwen2.5-VL arguments, accepts:
            pixel_values_hd: High-resolution pixel values for HD features
            image_grid_thw_hd: Grid dimensions for HD images
            image_hd_features: Pre-computed HD features (alternative to pixel_values_hd)
            image_range_list: Pre-computed image/answer ranges (optional, auto-computed)
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # === Step 1: HD feature generation ===
        if image_hd_features is None and pixel_values_hd is not None and image_grid_thw_hd is not None:
            image_hd_features = self._generate_hd_features(pixel_values_hd, image_grid_thw_hd)

        # === Step 2: Compute image_range_list if not provided ===
        if image_range_list is None and image_hd_features is not None and input_ids is not None:
            image_range_list = compute_image_range_list(
                input_ids, labels,
                image_token_id=self.config.image_token_id,
                im_start_token_id=IM_START_TOKEN_ID,
                image_grid_thw=image_grid_thw,
                spatial_merge_size=self.config.vision_config.spatial_merge_size,
            )

        # === Step 3: Pre-compute 3D mRoPE position IDs for DAT layers ===
        # DAT layers need [3, B, seq_len] positions (temporal, height, width).
        # In transformers 5.x, get_rope_index returns [4, B, seq_len] where
        # dim-0 is text position; strip it so DAT always works with 3D.
        mrope_position_ids = None
        if image_hd_features is not None and position_ids is None and input_ids is not None:
            position_ids, rope_deltas = self.model.get_rope_index(
                input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
            )
            mrope_position_ids = position_ids[-3:] if position_ids.size(0) == 4 else position_ids
        elif position_ids is not None:
            mrope_position_ids = position_ids[-3:] if position_ids.size(0) == 4 else position_ids

        # === Step 4: Call base model with DAT kwargs ===
        # These flow through: Model -> TextModel -> DecoderLayer -> Attention
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            second_per_grid_ts=second_per_grid_ts,
            # DAT kwargs — flow through to DAT attention layers
            image_hd_features=image_hd_features,
            image_range_list=image_range_list,
            mrope_position_ids=mrope_position_ids,
            **kwargs,
        )

        # === Vis: capture input_ids + image_path for the selected sample ===
        if self.training and input_ids is not None:
            for _m in self.modules():
                if hasattr(_m, '_dat_vis_b_idx'):
                    vis_b = _m._dat_vis_b_idx
                    self._dat_vis_input_ids = input_ids[vis_b].detach().cpu()
                    if hasattr(self, '_batch_image_paths'):
                        self._dat_vis_image_path = self._batch_image_paths[vis_b]
                    del _m._dat_vis_b_idx
                    break

        # === Step 5: LM head + loss ===
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            logits = logits.float()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = shift_logits.view(-1, self.config.text_config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = F.cross_entropy(shift_logits, shift_labels)

        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )


# ============================================================================
# Model Conversion Utility
# ============================================================================

# DAT-specific parameter name patterns (for selective unfreezing)
DAT_KEYS_MATCH = [
    'conv_lr_dw', 'ln_1', 'conv_lr_proj', 'proj_intention',
    'ln_2', 'conv_off_proj', 'k_proj_hd', 'v_proj_hd',
    'hd_attn_bias',
]


def convert_qwen2_5vl_to_dat(base_model_or_path, dat_extra_args, torch_dtype=None):
    """Convert a pretrained Qwen2.5VL to Qwen2.5VL-DAT.

    Uses from_pretrained() directly with the DAT model class so that
    DeepSpeed ZeRO-3 parameter partitioning works correctly.

    Args:
        base_model_or_path: path to pretrained Qwen2.5VL checkpoint
        dat_extra_args: dict with DAT parameters
        torch_dtype: optional torch dtype for loading

    Returns:
        Qwen2_5_VLDATForConditionalGeneration with base weights + fresh DAT weights
    """
    if isinstance(base_model_or_path, str):
        base_config = Qwen2_5_VLConfig.from_pretrained(base_model_or_path)
    else:
        base_config = base_model_or_path.config

    # Create DAT config from base config
    dat_config = Qwen2_5_VLDATConfig(**base_config.to_dict())
    dat_config.dat_extra_args = dat_extra_args

    if isinstance(base_model_or_path, str):
        dat_model = Qwen2_5_VLDATForConditionalGeneration.from_pretrained(
            base_model_or_path,
            config=dat_config,
            torch_dtype=torch_dtype,
            ignore_mismatched_sizes=False,
        )
    else:
        # Already-instantiated base model: swap class and reinit DAT layers
        base_model_or_path.config = dat_config
        base_model_or_path.__class__ = Qwen2_5_VLDATForConditionalGeneration
        layers_str = dat_extra_args.get('layers', '')
        text_config = dat_config.text_config
        if layers_str:
            assert len(layers_str) == text_config.num_hidden_layers
            for i, lt in enumerate(layers_str):
                if lt == 'D':
                    base_model_or_path.model.language_model.layers[i] = \
                        Qwen2_5_VLDecoderLayerDAT(text_config, i, dat_extra_args)
        dat_model = base_model_or_path

    return dat_model


def freeze_base_unfreeze_dat(model):
    """Freeze all parameters except DAT-specific ones."""
    total, trainable = 0, 0
    for name, param in model.named_parameters():
        total += 1
        if any(k in name for k in DAT_KEYS_MATCH):
            param.requires_grad = True
            trainable += 1
        else:
            param.requires_grad = False
    logger.info(f"Frozen: {total - trainable}/{total} params. Trainable (DAT): {trainable}/{total}")
