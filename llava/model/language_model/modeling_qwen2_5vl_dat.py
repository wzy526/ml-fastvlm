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
    → Backend selectable via _FA_BACKEND constant: "fa2" | "fa3" | "fa4".
"""

import logging
import math
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
    Qwen2_5_VLRMSNorm,
    Qwen2_5_VLRotaryEmbedding,
    Qwen2_5_VLCausalLMOutputWithPast,
    apply_multimodal_rotary_pos_emb,
    rotate_half,
    repeat_kv,
)


class _FP32WeightRMSNorm(Qwen2_5_VLRMSNorm):
    """RMSNorm that keeps ``weight`` in fp32 to avoid bf16 round-off near unity.

    The stock ``Qwen2_5_VLRMSNorm`` ends its forward with
    ``self.weight * hidden_states.to(input_dtype)``. If ``input_dtype`` is
    bf16 and ``self.weight`` is also bf16, the gradient updates to
    ``weight`` (massyo ≈ 1.0) are rounded out by the bf16 grid (resolution
    2⁻⁷ ≈ 7.8e-3 vs. AdamW per-step update ≈ 1e-4 at dat_lr=1e-4) and the
    scale never learns.

    Fix: keep ``weight`` in fp32 storage, multiply in fp32 (PyTorch
    promotes ``fp32_weight * bf16_h`` to fp32), then cast the result back
    to ``input_dtype`` so downstream ``k_proj_hd`` / ``v_proj_hd`` still
    receive bf16. ``Qwen2_5_VLAttentionDAT._apply`` enforces the fp32
    storage on every ``.to(...)``/``.bfloat16()`` call.

    Empirical evidence (0515 Run B ckpt-500): with the stock bf16 weight,
    ``hd_input_layernorm.weight`` stayed at exactly 1.000000 (std=0) after
    500 steps. With this subclass + ``_apply`` override, the weight is
    free to drift.
    """

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        h = hidden_states.to(torch.float32)
        variance = h.pow(2).mean(-1, keepdim=True)
        h = h * torch.rsqrt(variance + self.variance_epsilon)
        # fp32_weight * fp32_h → fp32; cast back to bf16 (or whatever input
        # dtype is) so downstream Linear ops stay dtype-consistent.
        return (self.weight * h).to(input_dtype)
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

# ── Backend selection ────────────────────────────────────────────────
# Change this constant to switch flash_attn backend:
#   "fa2"  — flash_attn 2.x   (flash_attn.flash_attn_func)
#   "fa3"  — flash_attn 3 / Hopper  (flash_attn_interface)
#   "fa4"  — flash_attn 4 / Cute    (flash_attn.cute)
_FA_BACKEND = "fa2"
# ─────────────────────────────────────────────────────────────────────

_flash_attn_func = None
_flash_attn_varlen_func = None
_FA_HAS_SOFTMAX_LSE = False

if _FA_BACKEND == "fa4":
    from flash_attn.cute import flash_attn_func as _flash_attn_func          # type: ignore[assignment]
    from flash_attn.cute import flash_attn_varlen_func as _flash_attn_varlen_func  # type: ignore[assignment]
    import flash_attn as _fa_mod
    _fa_ver = getattr(_fa_mod, "__version__", "unknown")
    print(f"[DAT-LSE] flash_attn 4 (cute) v{_fa_ver} — return_lse=True + varlen")

elif _FA_BACKEND == "fa3":
    try:
        from flash_attn_interface import flash_attn_func as _flash_attn_func           # type: ignore[assignment]
        from flash_attn_interface import flash_attn_varlen_func as _flash_attn_varlen_func  # type: ignore[assignment]
        print("[DAT-LSE] flash_attn 3 (hopper) — return_attn_probs=True + varlen")
    except ImportError:
        import inspect as _inspect
        from flash_attn import flash_attn_func as _flash_attn_func                # type: ignore[assignment]
        from flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func  # type: ignore[assignment]
        import flash_attn as _fa_mod
        _FA_HAS_SOFTMAX_LSE = "return_softmax_lse" in _inspect.signature(_flash_attn_func).parameters
        _fa_ver = getattr(_fa_mod, "__version__", "unknown")
        _FA_BACKEND = "fa2"  # type: ignore[assignment]
        _lse_api = "return_softmax_lse" if _FA_HAS_SOFTMAX_LSE else "return_attn_probs"
        print(f"[DAT-LSE] fa3 not available, fell back to flash_attn 2 v{_fa_ver} — {_lse_api} + varlen")

elif _FA_BACKEND == "fa2":
    import inspect as _inspect
    from flash_attn import flash_attn_func as _flash_attn_func                # type: ignore[assignment]
    from flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func  # type: ignore[assignment]
    import flash_attn as _fa_mod
    _FA_HAS_SOFTMAX_LSE = "return_softmax_lse" in _inspect.signature(_flash_attn_func).parameters
    _fa_ver = getattr(_fa_mod, "__version__", "unknown")
    _lse_api = "return_softmax_lse" if _FA_HAS_SOFTMAX_LSE else "return_attn_probs"
    print(f"[DAT-LSE] flash_attn 2 v{_fa_ver} — {_lse_api} + varlen")

else:
    raise ValueError(f"Unknown _FA_BACKEND={_FA_BACKEND!r}. Choose 'fa2', 'fa3', or 'fa4'.")


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
        lse: [B, H, N]     (float32)
    """
    q_fa = q.transpose(1, 2).contiguous()
    k_fa = k.transpose(1, 2).contiguous()
    v_fa = v.transpose(1, 2).contiguous()

    if _FA_BACKEND == "fa4":
        out_fa, lse = _flash_attn_func(q_fa, k_fa, v_fa, causal=causal,
                                       return_lse=True)
    elif _FA_BACKEND == "fa3":
        out_fa, lse = _flash_attn_func(q_fa, k_fa, v_fa, causal=causal,
                                       return_attn_probs=True)
    else:  # fa2
        if _FA_HAS_SOFTMAX_LSE:
            out_fa, lse = _flash_attn_func(q_fa, k_fa, v_fa, causal=causal,
                                           return_softmax_lse=True)
        else:
            out_fa, lse, _ = _flash_attn_func(q_fa, k_fa, v_fa, causal=causal,
                                               return_attn_probs=True)
    return out_fa, lse


def _dat_cross_attn_varlen(
    q_list: List[torch.Tensor],
    k_list: List[torch.Tensor],
    v_list: List[torch.Tensor],
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Batched non-causal cross-attention via flash_attn_varlen_func (single kernel).

    Args:
        q_list: list of [Nans_i, H, D] tensors (variable query lengths)
        k_list: list of [Ns_i, H, D] tensors
        v_list: list of [Ns_i, H, D] tensors

    Returns:
        out_list: list of [1, Nans_i, H, D] tensors
        lse_list: list of [1, H, Nans_i] tensors
    """
    n_segs = len(q_list)
    device = q_list[0].device
    H = q_list[0].shape[1]

    nq_lens = [q.shape[0] for q in q_list]
    nk_lens = [k.shape[0] for k in k_list]

    q_packed = torch.cat(q_list, dim=0)          # [total_q, H, D]
    k_packed = torch.cat(k_list, dim=0)          # [total_k, H, D]
    v_packed = torch.cat(v_list, dim=0)          # [total_k, H, D]

    cu_q = torch.zeros(n_segs + 1, dtype=torch.int32, device=device)
    cu_k = torch.zeros(n_segs + 1, dtype=torch.int32, device=device)
    for i in range(n_segs):
        cu_q[i + 1] = cu_q[i] + nq_lens[i]
        cu_k[i + 1] = cu_k[i] + nk_lens[i]

    if _FA_BACKEND == "fa4":
        out_packed, lse_packed = _flash_attn_varlen_func(
            q_packed, k_packed, v_packed,
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=max(nq_lens), max_seqlen_k=max(nk_lens),
            causal=False, return_lse=True,
        )
        # out_packed: [total_q, H, D],  lse_packed: [H, total_q]
    elif _FA_BACKEND == "fa3":
        out_packed, lse_packed = _flash_attn_varlen_func(
            q_packed, k_packed, v_packed,
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=max(nq_lens), max_seqlen_k=max(nk_lens),
            causal=False, return_attn_probs=True,
        )
    else:  # fa2
        if _FA_HAS_SOFTMAX_LSE:
            out_packed, lse_packed = _flash_attn_varlen_func(
                q_packed, k_packed, v_packed,
                cu_q, cu_k, max(nq_lens), max(nk_lens),
                causal=False, return_softmax_lse=True,
            )
        else:
            out_packed, lse_packed, _ = _flash_attn_varlen_func(
                q_packed, k_packed, v_packed,
                cu_q, cu_k, max(nq_lens), max(nk_lens),
                causal=False, return_attn_probs=True,
            )
        # FA2 varlen: lse_packed is already [H, total_q] (packed, not batched).
        # Unlike non-varlen flash_attn_func which returns [B, H, N],
        # flash_attn_varlen_func packs all sequences into a single axis.

    # Unpack: split by segment query lengths
    out_list = []
    lse_list = []
    q_offset = 0
    for i in range(n_segs):
        nq = nq_lens[i]
        out_list.append(out_packed[q_offset:q_offset + nq].unsqueeze(0))     # [1, Nans, H, D]
        lse_list.append(lse_packed[:, q_offset:q_offset + nq].unsqueeze(0))  # [1, H, Nans]
        q_offset += nq

    return out_list, lse_list


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
        # Critical: strip ``model_type`` from kwargs so ``base_config.to_dict()``
        # (which carries ``model_type='qwen2_5_vl'``) cannot pollute the instance
        # attribute. Without this the instance ends up with
        # ``self.model_type='qwen2_5_vl'`` even though the CLASS attribute
        # (used by ``to_dict()`` / ``config.json``) is ``'qwen2_5_vl_dat'`` —
        # producing an asymmetric save/load where ``save_pretrained`` applies
        # the qwen2_vl key-remap (``model.language_model → model``, ``visual
        # → model.visual`` reverse), but ``from_pretrained`` later queries the
        # mapping by ``'qwen2_5_vl_dat'`` (no remap registered) and fails to
        # translate the flat on-disk keys back to the hierarchical param names.
        kwargs.pop("model_type", None)
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
            'hd_gate_init': None,      # Learnable HD gate init value (e.g. -10.0); None = disabled
            'hd_gate_freeze': False,   # If True, hd_gate is created with requires_grad=False (diagnostic mode)
            'use_fused_vit': False,    # Fuse LR+HD into one ViT call (saves kernel launch; costs ~2× activation memory)
            'use_shared_vit': False,   # Single HD ViT call; LR tokens are adaptive-pooled from HD features
                                       # (no LR ViT at all). Overrides use_fused_vit when True.
                                       # Semantics differ from the baseline LR ViT path → requires retraining.
            'use_spatial_attn_guide': True,  # Q_intention × Q_lr spatial attention for offset guidance
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


class _FP32WeightLayerNorm2d(LayerNorm2d):
    """LayerNorm2d that keeps ``weight`` (≈1.0) and ``bias`` (≈0.0) in fp32.

    Same bf16 round-off problem as ``_FP32WeightRMSNorm``: weight initialized
    at exactly 1.0, bf16 grid spacing at unity is 2⁻⁷ ≈ 7.8e-3, and AdamW
    updates at lr≈1e-4 silently round to zero. Empirically observed on the
    DAT offset path's ``ln_1`` / ``ln_2`` after Run B (no fp32 protection).

    Storage is fp32; forward runs entirely in fp32 then casts back to the
    input dtype. The compute is small (off_dim or inter_size ≤ 128, grid ≤
    20×20) so the fp32 overhead is negligible.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        h = x.to(torch.float32)
        u = h.mean(1, keepdim=True)
        s = (h - u).pow(2).mean(1, keepdim=True)
        h = (h - u) / torch.sqrt(s + self.eps)
        h = self.weight[None, :, None, None] * h + self.bias[None, :, None, None]
        return h.to(input_dtype)


class _FP32WeightConv2d(nn.Conv2d):
    """Conv2d whose ``weight``/``bias`` are stored in fp32 to avoid bf16
    round-off on AdamW updates.

    Forward casts the fp32 weight back to the input dtype, so the conv math
    itself runs at the surrounding (typically bf16) precision and remains
    cheap. Storage in fp32 is what matters: it lets small AdamW updates
    (``lr * m / sqrt(v)`` ≈ 1e-4) accumulate without being silently rounded
    out by the bf16 grid (which at weight magnitude ≈ 0.05 has spacing
    ≈ 4e-4, comparable to the per-step update).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight.dtype != x.dtype:
            w = self.weight.to(x.dtype)
            b = self.bias.to(x.dtype) if self.bias is not None else None
            return F.conv2d(
                x, w, b, self.stride, self.padding, self.dilation, self.groups,
            )
        return super().forward(x)


class _FP32WeightLinear(nn.Linear):
    """Linear with fp32 master weight/bias; forward downcasts to input dtype.

    Same motivation as ``_FP32WeightConv2d``. Used for ``proj_intention``
    (≤ 128 × 128 = 16K params) where the storage cost of fp32 is trivial
    and bf16 round-off on small kaiming-initialized weights is a real risk.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight.dtype != x.dtype:
            w = self.weight.to(x.dtype)
            b = self.bias.to(x.dtype) if self.bias is not None else None
            return F.linear(x, w, b)
        return super().forward(x)


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
        self.use_spatial_attn_guide = dat.get('use_spatial_attn_guide', True)

        self.off_dim = self.hidden_size // self.off_grps

        # --- Offset generation pipeline ---
        # All offset-path params use fp32-storage subclasses (see
        # _FP32Weight{LayerNorm2d, Conv2d, Linear}). The bf16 round-off that
        # froze ``hd_input_layernorm.weight`` and ``hd_gate`` at init also
        # applies to these: ``ln_*.weight`` init at 1.0 (bf16 spacing 2⁻⁷ ≈
        # 7.8e-3), ``conv_*.weight`` kaiming-init magnitude ≈ 0.05–0.13 (bf16
        # spacing ≈ 4e-4 — comparable to per-step AdamW update lr*grad ≈
        # 1e-4). Empirically Run B's trained offset behavior is virtually
        # identical to a kaiming-random-init DAT (see
        # scripts/qwen2_5vl_adl_0515/_diagnose_offsets*.py).
        self.conv_lr_dw = _FP32WeightConv2d(
            self.off_dim, self.off_dim,
            kernel_size=self.off_ksize, stride=1, padding=self.off_ksize // 2,
            groups=self.off_dim, bias=False,
        )
        self.ln_1 = _FP32WeightLayerNorm2d(self.off_dim)
        self.conv_lr_proj = _FP32WeightConv2d(
            self.off_dim, self.inter_size,
            kernel_size=1, stride=1, padding=0,
        )

        # Intention branch
        if self.use_intention_branch:
            self.proj_intention = _FP32WeightLinear(self.off_dim, self.inter_size)
        else:
            self.proj_intention = nn.Identity()

        # Offset prediction
        if self.intention_as_gate:
            self.ln_2 = _FP32WeightLayerNorm2d(self.inter_size)
            self.conv_off_proj = _FP32WeightConv2d(
                self.inter_size, 2, kernel_size=1, stride=1, padding=0, bias=False,
            )
        else:
            self.ln_2 = _FP32WeightLayerNorm2d(self.inter_size * 2)
            self.conv_off_proj = _FP32WeightConv2d(
                self.inter_size * 2, 2, kernel_size=1, stride=1, padding=0, bias=False,
            )

        # HD feature KV projection
        if self.hd_proj:
            kv_dim = self.num_key_value_heads * self.head_dim
            self.k_proj_hd = nn.Linear(self.hidden_size, kv_dim)
            self.v_proj_hd = nn.Linear(self.hidden_size, kv_dim)
            # RMSNorm applied to ``sampled_hr`` before k_proj_hd / v_proj_hd
            # (F4 from the Run B fix).
            #
            # Motivation: q_proj / k_proj / v_proj on the LR path see
            # ``self.input_layernorm(hidden_states)`` (RMSNormed layer-L
            # hidden states), so their output value vectors are in the
            # "RMSNormed layer-L value-space". But k_proj_hd / v_proj_hd
            # see raw ``sampled_hr`` = pooler_output from the visual
            # merger, which is *not* RMSNormed. Without this norm, the
            # statistical distribution of value_hd cannot match value, and
            # the LSE merge ``out = (1 − w₂)·out₁ + w₂·out₂`` injects a
            # distribution-shifted signal at each DAT layer — accumulating
            # to a ~3% per-layer corruption of the residual stream at
            # inference time.
            #
            # Adding a learnable Qwen2_5_VLRMSNorm here is the cheapest fix:
            # weight initializes to ones (acts as a pure normalization at
            # step 0), the model can then learn a per-channel rescaling if
            # the LR/HD distributions need different shaping.
            # Use a fp32-weight subclass to avoid bf16 round-off near 1.0
            # (see _FP32WeightRMSNorm docstring + Qwen2_5_VLAttentionDAT._apply).
            self.hd_input_layernorm = _FP32WeightRMSNorm(
                self.hidden_size, eps=config.rms_norm_eps
            )
        else:
            self.k_proj_hd = None
            self.v_proj_hd = None
            self.hd_input_layernorm = None

        # Rotary embedding for extended KV (Qwen2.5-VL Attention doesn't have one)
        self._dat_rotary_emb = Qwen2_5_VLRotaryEmbedding(config=config)

        # Learnable HD gate: sigmoid(hd_gate) scales the HD partition function.
        # Init to a large negative value (e.g. -10) so sigmoid ≈ 0 at the start,
        # preventing HD features from dominating before the offset/sampling modules
        # have learned meaningful representations.
        #
        # When ``hd_gate_freeze=True`` the parameter is created but
        # ``requires_grad`` is left False, so the gate stays at its init value
        # for the entire run. This is a diagnostic mode for the zero-init-V
        # cold start: with the gate locked, v_proj_hd cannot escape via
        # "close the door" and must learn useful HD content.
        hd_gate_init = dat.get('hd_gate_init', None)
        hd_gate_freeze = bool(dat.get('hd_gate_freeze', False))
        if hd_gate_init is not None:
            self.hd_gate = nn.Parameter(
                torch.tensor(float(hd_gate_init)),
                requires_grad=not hd_gate_freeze,
            )
            self._hd_gate_freeze = hd_gate_freeze
        else:
            self.hd_gate = None
            self._hd_gate_freeze = False

        # D1 (inject_lr_image): in addition to the existing answer-span Q,
        # also feed lr_image-position Q into the HD cross-attention. This
        # lets HD info enter the residual stream via lr_image tokens (the
        # path actually used by Qwen2.5-VL: image → qa_prefix tokens →
        # answer query, verified via scripts/qwen2_5vl_adl_0519/_diagnose_baseline_attn.py).
        self.dat_inject_lr_image = bool(dat.get('inject_lr_image', False))

        self._init_dat_weights()

    def _apply(self, fn, recurse=True):
        """Run base ``_apply`` then force ``hd_gate`` back to fp32.

        ``hd_gate`` is a single scalar nn.Parameter used as a log-σ bias
        in the LSE merge (``lse2 += log σ(g)``). When the wrapping training
        script does ``model.to(torch.bfloat16)`` (e.g. exp9.sh's
        ``--lora_enable True --bf16 True`` path), this scalar gets cast to
        bf16. The bf16 representational gap at |g|≈8 is

            2^(exponent − mantissa_bits) = 2^(3 − 7) = 0.0625 ,

        whereas AdamW's per-step update on ``hd_gate`` with ``dat_lr=1e-4``
        is on the order of 1e-4 — three orders of magnitude below the bf16
        grid. The update is silently rounded back to -8.0 and ``hd_gate``
        appears frozen even though gradients flow correctly (other DAT
        params and ``dat/grad_norm`` keep moving). Verified in
        ``scripts/qwen2_5vl_adl_0514/_verify_hd_gate_bf16_roundoff.py``:
        1000 bf16 Adam steps with grad ≈ 1.4 produce zero motion; the same
        loop in fp32 walks cleanly from -8.0 to -8.09.

        Overriding ``_apply`` here is the canonical PyTorch hook for "this
        param has a fixed dtype" — any future ``.to(...)``, ``.cuda()``,
        ``.bfloat16()``, ``.half()`` call routes through ``_apply``, so the
        scalar is restored to fp32 unconditionally.
        """
        result = super()._apply(fn, recurse=recurse)
        if self.hd_gate is not None and self.hd_gate.dtype != torch.float32:
            with torch.no_grad():
                self.hd_gate.data = self.hd_gate.data.to(torch.float32)
        # Same bf16 round-off problem for ``hd_input_layernorm.weight``: the
        # RMSNorm scale lives near 1.0, where bf16 resolution is 2^-7 ≈ 7.8e-3,
        # while AdamW's per-step update at dat_lr=1e-4 is ≈ 1e-4. After 500
        # steps the cumulative update (≈ 5e-3) is *still* below the bf16
        # grid → the scale stays at exactly 1.000000 forever (empirically
        # verified on ckpt-500 of the 0515 Run B). RMSNorm Forward still
        # normalises sampled_hr to RMS=1, but the per-channel adaptation
        # never learns. Force fp32 storage like hd_gate; ``Qwen2_5_VLRMSNorm.forward``
        # type-promotes ``fp32_weight * bf16_hidden_state`` to fp32 then
        # casts back to ``input_dtype`` on the final return, so downstream
        # ``k_proj_hd`` still receives bf16 (no compute-dtype regression).
        if (
            self.hd_input_layernorm is not None
            and self.hd_input_layernorm.weight.dtype != torch.float32
        ):
            with torch.no_grad():
                self.hd_input_layernorm.weight.data = (
                    self.hd_input_layernorm.weight.data.to(torch.float32)
                )
        # Offset-path bf16 round-off protection. See
        # _FP32Weight{LayerNorm2d, Conv2d, Linear} docstrings: ln_*.weight
        # init at 1.0 has bf16 grid 2⁻⁷ ≈ 7.8e-3; conv_*.weight kaiming-init
        # magnitude ≈ 0.05–0.13 has bf16 grid ≈ 4e-4, both comparable to or
        # larger than AdamW per-step update ≈ 1e-4 → updates round off to
        # zero and the offset network stays at init (cf. Run B vs random-init
        # offset diagnostic: virtually identical statistics).
        for sub in (self.conv_lr_dw, self.ln_1, self.conv_lr_proj,
                    self.proj_intention, self.ln_2, self.conv_off_proj):
            if not isinstance(sub, nn.Module):
                continue
            for p in sub.parameters(recurse=False):
                if p.dtype != torch.float32:
                    with torch.no_grad():
                        p.data = p.data.to(torch.float32)
        return result

    @torch.no_grad()
    def _init_dat_weights(self):
        # Conv layers: Kaiming normal for the LR feature extractors.
        nn.init.kaiming_normal_(self.conv_lr_dw.weight)
        nn.init.kaiming_normal_(self.conv_lr_proj.weight)
        if self.conv_lr_proj.bias is not None:
            nn.init.zeros_(self.conv_lr_proj.bias)
        # uninformative-but-safe reference grid.
        nn.init.zeros_(self.conv_off_proj.weight)
        # Linear layers: Xavier uniform
        if isinstance(self.proj_intention, nn.Linear):
            nn.init.xavier_uniform_(self.proj_intention.weight)
            if self.proj_intention.bias is not None:
                nn.init.zeros_(self.proj_intention.bias)
        # HD KV projections: zero-init adapter pattern (K=Kaiming, V=0).
        #
        # Previously we copied k_proj / v_proj into k_proj_hd / v_proj_hd to
        # "reuse pretrained weights", but this is an OOD initialization:
        # k_proj was trained to project layer-L *hidden states* into layer-L
        # K-space, while k_proj_hd is applied to image_hd_features = the
        # visual merger's pooler_output, which lives in layer-0 input
        # embedding space. For a deep DAT layer (e.g. L=30) this is a 30-
        # layer domain mismatch — Q · K_hd^T has no geometric meaning, out2
        # is pure noise, and CE pushes hd_gate to close every time the gate
        # opens enough to register the noise (verified empirically: starting
        # hd_gate at -2 produces a monotone descent over the first 100 steps).
        #
        # Fix: zero-init V so out2 = softmax(Q·K_hd^T) · 0 ≡ 0 at step 0
        # (HD path is a strict no-op, no noise to fight off). Kaiming-normal
        # K with nonlinearity='linear' provides symmetry-breaking for the
        # first backward pass (V's gradient is shaped by a non-uniform
        # attention pattern from step 1 onward, avoiding the rank-1
        # transient that would occur with K=V=0). This is the same pattern
        # as LoRA's (A=Kaiming, B=0).
        self._init_hd_proj_weights()

    @torch.no_grad()
    def _init_hd_proj_weights(self):
        """Init k_proj_hd / v_proj_hd with the zero-init adapter pattern.

        K_hd ← Kaiming normal (nonlinearity='linear'): provides symmetry-
        breaking statistics so attention is non-uniform from step 1 onward.

        V_hd ← 0: guarantees ``out2 = softmax(Q·K_hd^T) · 0 ≡ 0`` at step 0,
        so the HD path is a strict no-op until V_hd has learned useful
        directions. ``v_proj_hd`` still receives non-zero gradients through
        the LSE merge (``dL/dV_hd ∝ softmax(...) · w₂ · dL/dout_merged``),
        so the path can only grow toward directions that reduce loss.

        Biases (if present) are zeroed in both projections.

        See the matching prose in ``_init_dat_weights`` for the architectural
        motivation (OOD copy-from-k_proj problem).
        """
        if self.k_proj_hd is None:
            return
        nn.init.kaiming_normal_(self.k_proj_hd.weight, nonlinearity='linear')
        nn.init.zeros_(self.v_proj_hd.weight)
        if self.k_proj_hd.bias is not None:
            nn.init.zeros_(self.k_proj_hd.bias)
        if self.v_proj_hd.bias is not None:
            nn.init.zeros_(self.v_proj_hd.bias)

    def _grid_generate(self, h, w, n_repeats, device):
        """Generate reference sampling grid with a half-cell margin from
        the [-1, 1] clamp boundary.

        Why the margin (not just ``linspace(-1, 1, h)``):
            The original DAT layout put reference points at exactly ±1 at
            the four edges of the grid. Combined with the downstream
            ``sample_locs = (reference + offset).clamp(-1, 1)``, the
            edge cells were single-side dead-locked: a negative offset
            at ref_x=-1 is clipped to -1, the clamp gradient is zero
            outside [-1, 1], so the network never receives a learning
            signal "your edge cell should move further left". Edge cells
            (4*(G-1) ≈ 19% of the grid for G=20) could only learn to
            move INWARD, never outward.

            We solve this by inset-ing the reference grid by half a cell
            on every side (``align_corners=False``-style cell-center
            convention):
                margin_y = 1/(h-1),  margin_x = 1/(w-1)
                grid_y ∈ [-1+m_y, 1-m_y]
                grid_x ∈ [-1+m_x, 1-m_x]
            Now every cell — interior and edge — has at least a
            half-cell of bidirectional offset freedom before hitting the
            clamp boundary. The 5% strip of HD feature near the literal
            image borders becomes reachable only via positive offsets,
            which is the correct learning signal ("reach further out for
            corner content").
        """
        m_y = 1.0 / max(h - 1, 1)
        m_x = 1.0 / max(w - 1, 1)
        grid_y = torch.linspace(-1.0 + m_y, 1.0 - m_y, h, device=device, dtype=torch.float32)
        grid_x = torch.linspace(-1.0 + m_x, 1.0 - m_x, w, device=device, dtype=torch.float32)
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
            # Keep singleton spatial dims; broadcasting handles the
            # grid_size × grid_size expansion at use sites (gate-multiply
            # path is pure broadcast; cat path expands at the cat call).
            # Avoids a Lp·off_grps·inter_size·grid_size² fp32 memcpy and
            # the ~400× sigmoid blow-up.
            embed_intention = einops.rearrange(
                embed_intention, 'l g c -> (l g) c 1 1',
            )

        embed_lr_rep = einops.repeat(
            embed_lr, 'g c h w -> (l g) c h w', l=Lp,
        )

        # Spatial attention guide: Q_intention × Q_lr → spatial saliency map,
        # applied as a multiplicative bias on LR features so offset prediction
        if self.use_intention_branch and self.use_spatial_attn_guide:
            q_lr_flat = query_states[b_idx, image_range_index]    # [lr_h*lr_w, C]
            q_int_flat = query_states[b_idx, intention_indices]   # [Lp, C]
            spatial_attn = torch.matmul(
                q_int_flat.float(), q_lr_flat.float().transpose(0, 1),
            ) / math.sqrt(q_lr_flat.shape[-1])                    # [Lp, lr_h*lr_w]
            spatial_attn = spatial_attn.softmax(dim=-1).view(Lp, 1, lr_h, lr_w)
            spatial_attn_guide = F.adaptive_avg_pool2d(
                spatial_attn, (self.grid_size, self.grid_size),
            ) * (lr_h * lr_w)  # normalize: uniform attention ≈ 1.0
            spatial_guide_rep = einops.repeat(
                spatial_attn_guide, 'l 1 h w -> (l g) 1 h w', g=self.off_grps,
            ).to(embed_lr_rep.dtype)
            embed_lr_rep = embed_lr_rep * spatial_guide_rep

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
                # cat does not broadcast — expand the singleton spatial dims
                # added above to match embed_lr_rep before concatenation.
                off_guide = torch.cat([
                    embed_lr_rep,
                    embed_intention.expand(-1, -1, self.grid_size, self.grid_size),
                ], dim=1)
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
            # Normalize sampled_hr to match the RMSNormed distribution that
            # k_proj / v_proj see on the LR path (Qwen attention always
            # consumes ``self.input_layernorm(hidden_states)``). Without
            # this norm, value_hd has a different magnitude than value, and
            # the LSE merge injects a distribution-shifted signal at every
            # DAT layer.
            sampled_hr = self.hd_input_layernorm(sampled_hr)
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

        HD gating via LSE bias (linear warmup):
            During warmup, gate g ∈ (0, 1] linearly increases from ~0 to 1.
            We apply ℓ₂_gated = ℓ₂ + log(g), which scales the HD partition
            function Z₂ by g:  Z₂_gated = g · Z₂.
            The merge weights remain normalized (w₁+w₂ = 1) but the effective
            HD contribution is smoothly suppressed:
                w₂ = g·Z₂ / (Z₁ + g·Z₂)
            At g→0: o* ≈ o₁ (pure causal).  At g=1: standard merge.

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

        # Ensure fp32 for numerically sensitive LSE merge (logaddexp / exp).
        # flash_attn returns fp32 LSE today, but we make it explicit.
        lse1_ans = lse1[:, :, ans_start:ans_end].float()   # [B, H, Nans]
        lse2 = lse2.float()                                # [1, H, Nans]
        out1_ans = out1[:, ans_start:ans_end, :, :]        # [B, Nans, H, D]

        # HD gating: suppress HD contribution at the start of training.
        # hd_gate is an nn.Parameter, applied as lse2_gated = lse2 + log(sigmoid(hd_gate)).
        # With hd_gate init ≈ -10, sigmoid ≈ 4.5e-5, so HD is ~silent initially.
        # The model learns to open the gate during training.
        if self.hd_gate is not None:
            lse2 = lse2 + F.logsigmoid(self.hd_gate)
            if self.training:
                self._dat_hd_gate_value = self.hd_gate.detach().sigmoid().item()

        lse = torch.logaddexp(lse1_ans, lse2)              # [B, H, Nans] fp32

        # Weights: [B, H, Nans] → [B, Nans, H, 1] for broadcasting (fp32)
        w1 = (lse1_ans - lse).exp().permute(0, 2, 1).unsqueeze(-1)
        w2 = (lse2     - lse).exp().permute(0, 2, 1).unsqueeze(-1)

        out[:, ans_start:ans_end, :, :] = (w1 * out1_ans + w2 * out2).to(out.dtype)

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

        # === Pass 2: HD cross-attention (batched via varlen for minimal kernel launches) ===
        # Phase 2a: Prepare all cross-attention segment pairs
        seg_q_list: List[torch.Tensor] = []   # [Nans_i, H, D] each
        seg_k_list: List[torch.Tensor] = []   # [Ns, H, D] each
        seg_v_list: List[torch.Tensor] = []   # [Ns, H, D] each
        seg_meta: List[Tuple[int, int, int]] = []  # (b_idx, q_ans_start, Nans)

        for b_idx in range(B):
            if len(image_range_list[b_idx]) <= 1:
                continue

            hd_feat_idx = b_idx_to_hd_idx[b_idx]
            lr_start, lr_end, lr_h, lr_w = image_range_list[b_idx][0]

            if mrope_position_ids is not None:
                orig_pos_b = mrope_position_ids[:, b_idx, :]
            else:
                orig_pos_b = torch.arange(Nq, device=device, dtype=torch.long).unsqueeze(0).expand(3, -1)

            hd_pos_ids = self._construct_hd_position_ids(
                orig_pos_b, lr_start, lr_end, lr_h, lr_w, device,
            )
            hd_pos_ids_batched = hd_pos_ids.unsqueeze(1)

            key_hd_all, value_hd_all, _slocs = self._generate_offsets_and_sample(
                query_states, image_hd_features, image_range_list, b_idx, hd_feat_idx,
            )

            if _want_vis and _dat_vis_entry is None:
                _dat_vis_entry = (b_idx, _slocs)

            # ----- Track the FIRST answer's prepped K/V for D1 reuse ------
            # D1 lr_image segment shares HD K/V with answer 0 (single offset
            # prediction per b_idx, conditioned on the FIRST answer's
            # intention token).
            k_hd_l_first: Optional[torch.Tensor] = None
            v_hd_l_first: Optional[torch.Tensor] = None

            for l_idx, answer_range in enumerate(image_range_list[b_idx][1:]):
                ans_start     = answer_range[0]
                ans_end       = answer_range[1]
                intention_idx = answer_range[2]

                if ans_end > 0:
                    q_ans_start = ans_start
                    Nans = ans_end - ans_start
                else:
                    q_ans_start = intention_idx + 1
                    Nans = Nq - q_ans_start

                if Nans <= 0:
                    continue

                # q: [H, Nans, D] → [Nans, H, D] for varlen packing
                q_seg = query_bhnc[b_idx, :, q_ans_start:q_ans_start + Nans, :] \
                    .transpose(0, 1).contiguous()   # [Nans, H, D]

                k_hd_l = (key_hd_all[l_idx]
                          .view(Ns, self.num_key_value_heads, self.head_dim)
                          .unsqueeze(0).unsqueeze(0))  # [1, 1, Ns, D_kv]
                v_hd_l = (value_hd_all[l_idx]
                          .view(Ns, self.num_key_value_heads, self.head_dim)
                          .unsqueeze(0).unsqueeze(0))  # [1, 1, Ns, D_kv]

                # Reshape for RoPE: [1, H_kv, Ns, D]
                k_hd_l = k_hd_l.view(1, Ns, self.num_key_value_heads, self.head_dim).transpose(1, 2)
                v_hd_l = v_hd_l.view(1, Ns, self.num_key_value_heads, self.head_dim).transpose(1, 2)

                cos_hd, sin_hd = self._dat_rotary_emb(k_hd_l, hd_pos_ids_batched)
                k_hd_l = apply_multimodal_rotary_pos_emb_single(
                    k_hd_l, cos_hd, sin_hd, mrope_section,
                )

                k_hd_l = repeat_kv(k_hd_l, self.num_key_value_groups)  # [1, H, Ns, D]
                v_hd_l = repeat_kv(v_hd_l, self.num_key_value_groups)

                # [1, H, Ns, D] → [Ns, H, D]
                k_seg = k_hd_l.squeeze(0).transpose(0, 1).contiguous()
                v_seg = v_hd_l.squeeze(0).transpose(0, 1).contiguous()
                seg_k_list.append(k_seg)
                seg_v_list.append(v_seg)
                seg_q_list.append(q_seg)
                seg_meta.append((b_idx, q_ans_start, Nans))

                if l_idx == 0:
                    k_hd_l_first = k_seg
                    v_hd_l_first = v_seg

            # ----- D1: also inject HD info at lr_image positions ----------
            # The base Qwen2.5-VL pathway carries image info via:
            #   image tokens → qa_prefix tokens → answer query.
            # The original DAT only injected HD into the answer-span Q,
            # which (per the baseline diag) gets only ~5% of total attention
            # in the LR pass. Routing HD through lr_image positions lets
            # HD enter the residual stream where it can be picked up by
            # later layers' qa_prefix-conditioned attention.
            if (self.dat_inject_lr_image
                    and k_hd_l_first is not None
                    and v_hd_l_first is not None):
                Nlr = lr_end - lr_start
                if Nlr > 0:
                    q_lr = query_bhnc[b_idx, :, lr_start:lr_end, :] \
                        .transpose(0, 1).contiguous()    # [Nlr, H, D]
                    seg_q_list.append(q_lr)
                    seg_k_list.append(k_hd_l_first)
                    seg_v_list.append(v_hd_l_first)
                    seg_meta.append((b_idx, lr_start, Nlr))

        # Phase 2b: Batched cross-attention — ONE kernel for all segments
        if seg_q_list:
            out2_list, lse2_list = _dat_cross_attn_varlen(
                seg_q_list, seg_k_list, seg_v_list,
            )
        else:
            out2_list, lse2_list = [], []

        # Phase 2c: LSE merge.
        out_parts: List[torch.Tensor] = []
        seg_iter = 0
        for b_idx in range(B):
            out_b = out1[b_idx:b_idx + 1]  # [1, Nq, H, D]

            while seg_iter < len(seg_meta) and seg_meta[seg_iter][0] == b_idx:
                _, q_start, Nseg = seg_meta[seg_iter]
                out_b = self._merge_two_pass_lse(
                    out_b, lse1[b_idx:b_idx + 1],
                    out2_list[seg_iter], lse2_list[seg_iter],
                    q_start, q_start + Nseg,
                )
                seg_iter += 1

            out_parts.append(out_b)

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

        self._patch_text_model_init_weights()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """Override to post-fix raw nn.Parameter loading for ``hd_gate``.

        HF's checkpoint key remapping (``model.layers.X`` ↔
        ``model.language_model.layers.X`` for the Qwen2.5-VL Module/TextModel
        split) handles submodule parameters (``k_proj_hd.weight``,
        ``conv_off_proj.weight`` …) correctly, but silently drops raw
        ``nn.Parameter`` attributes such as ``hd_gate``. Result: every
        ``hd_gate`` was being reset to ``hd_gate_init`` at inference time even
        when the merged ckpt stored a properly trained value.

        After the standard ``from_pretrained`` returns, we scan the ckpt
        directory for any ``*hd_gate*`` keys in safetensors / pytorch_model.bin
        and write them into the corresponding ``model.language_model.layers[i]
        .self_attn.hd_gate`` parameter. No-op when loading from the base Qwen
        ckpt (it contains no ``hd_gate`` keys) or when the path is a Hub id.
        """
        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        cls._manual_load_dat_raw_params(model, pretrained_model_name_or_path)
        return model

    @staticmethod
    def _manual_load_dat_raw_params(model, path):
        """Post-load fix for raw nn.Parameter attributes not covered by HF
        key remapping (currently only ``hd_gate``)."""
        import os
        if not isinstance(path, str) or not os.path.isdir(path):
            return

        weights: dict[str, torch.Tensor] = {}
        try:
            from safetensors import safe_open  # type: ignore
        except ImportError:
            safe_open = None  # type: ignore

        for fn in sorted(os.listdir(path)):
            full = os.path.join(path, fn)
            if fn.endswith('.safetensors') and safe_open is not None:
                with safe_open(full, framework='pt') as st:
                    for k in st.keys():
                        if 'hd_gate' in k:
                            weights[k] = st.get_tensor(k)
            elif fn in ('pytorch_model.bin', 'model.bin'):
                sd = torch.load(full, map_location='cpu')
                for k, v in sd.items():
                    if 'hd_gate' in k:
                        weights[k] = v

        if not weights:
            return

        if hasattr(model.model, 'language_model'):
            text_model = model.model.language_model
        else:
            text_model = model.model

        n_loaded = 0
        for k, v in weights.items():
            if not k.endswith('.self_attn.hd_gate'):
                continue
            parts = k.split('.')
            try:
                layer_idx = int(parts[parts.index('layers') + 1])
            except (ValueError, IndexError):
                continue
            if layer_idx >= len(text_model.layers):
                continue
            attn = text_model.layers[layer_idx].self_attn
            if hasattr(attn, 'hd_gate') and attn.hd_gate is not None:
                with torch.no_grad():
                    attn.hd_gate.data.copy_(
                        v.to(attn.hd_gate.dtype).to(attn.hd_gate.device)
                    )
                n_loaded += 1

        if n_loaded > 0:
            try:
                logger.info(
                    f"[DAT post-load] manually loaded hd_gate for {n_loaded} DAT layers"
                )
            except NameError:
                print(
                    f"[DAT post-load] manually loaded hd_gate for {n_loaded} DAT layers"
                )

    def _patch_text_model_init_weights(self):
        """Make smart_apply use DAT-aware ``_init_weights`` for the inner text model.

        HF's ``smart_apply`` (called by ``_finalize_model_loading`` →
        ``_initialize_missing_keys``) dispatches each ``PreTrainedModel`` child
        to use its own ``_init_weights``, not the top-level wrapper's. Since the
        DAT attention modules live inside ``self.model.language_model``
        (``Qwen2_5_VLTextModel``), the active ``_init_weights`` for them is
        ``Qwen2_5_VLPreTrainedModel._init_weights`` — which only handles
        Linear / Conv / Embedding / LayerNorm-named modules. The standalone
        ``hd_gate`` ``nn.Parameter`` and the Kaiming / Xavier inits from
        ``_init_dat_weights`` are therefore never restored after
        ``from_pretrained`` blew them away under ``torch.device("meta")`` /
        ``deepspeed.zero.Init()``.

        We monkey-patch the text-model instance's ``_init_weights`` to wrap the
        original implementation and add the DAT-specific handling. ``smart_apply``
        does ``module._initialize_weights`` which calls ``self._init_weights``,
        so an instance-level attribute is enough — no class-wide patching.
        """
        import types
        text_model = self.model.language_model
        text_model_cls = type(text_model)
        cls_init_weights = text_model_cls._init_weights
        hd_gate_init = self.config.dat_extra_args.get('hd_gate_init', None)
        hd_gate_freeze = bool(self.config.dat_extra_args.get('hd_gate_freeze', False))

        def _dat_init_weights(text_self, module):
            cls_init_weights(text_self, module)
            if isinstance(module, Qwen2_5_VLAttentionDAT):
                if module.hd_gate is not None and hd_gate_init is not None:
                    module.hd_gate.data.fill_(float(hd_gate_init))
                    # smart_apply on a MISSING key re-instantiates the
                    # parameter with default requires_grad=True, so we have
                    # to re-apply the freeze flag here too.
                    if hd_gate_freeze:
                        module.hd_gate.requires_grad = False
                nn.init.kaiming_normal_(module.conv_lr_dw.weight)
                nn.init.kaiming_normal_(module.conv_lr_proj.weight)
                nn.init.kaiming_normal_(module.conv_off_proj.weight)
                if module.conv_lr_proj.bias is not None:
                    nn.init.zeros_(module.conv_lr_proj.bias)
                if isinstance(module.proj_intention, nn.Linear):
                    nn.init.xavier_uniform_(module.proj_intention.weight)
                    if module.proj_intention.bias is not None:
                        nn.init.zeros_(module.proj_intention.bias)
                # HD KV projections must also be re-set here: HF's default
                # _init_weights treats k_proj_hd / v_proj_hd as ordinary
                # nn.Linear and overwrites them with normal_(std=0.02), which
                # would break V_hd=0. Apply the zero-init adapter pattern
                # (K=Kaiming, V=0) so out2 ≡ 0 at step 0.
                module._init_hd_proj_weights()

        text_model._init_weights = types.MethodType(_dat_init_weights, text_model)

    @torch.no_grad()
    def init_hd_proj_from_kv(self):
        """Init k_proj_hd / v_proj_hd with the zero-init adapter pattern.

        Historically this method copied pretrained k_proj / v_proj weights
        into k_proj_hd / v_proj_hd, but that was an OOD initialization:
        k_proj at DAT layer L was trained to map *layer-L hidden states*
        into K-space, while k_proj_hd is applied to image_hd_features =
        visual.pooler_output ≈ layer-0 input embedding space. For deep DAT
        layers (L=30) this is a 30-layer domain mismatch — Q · K_hd^T has
        no geometric meaning, out2 is pure noise, and CE consistently
        pushes hd_gate down (empirically verified: hd_gate_init=-2 → 100
        steps of monotone descent toward -inf).

        New scheme:
            K_hd ← Kaiming normal (nonlinearity='linear')
            V_hd ← 0
            biases zeroed

        Math: ``out2 = softmax(Q·K_hd^T) · 0 ≡ 0`` at step 0, so HD path is
        a strict no-op. V_hd still receives non-zero gradients via the LSE
        merge, so it grows only toward loss-reducing directions. Equivalent
        to LoRA's (A=Kaiming, B=0) adapter pattern.

        The method name is kept (``init_hd_proj_from_kv``) for compatibility
        with existing training scripts that call it after ``from_pretrained``;
        the "_from_kv" suffix is now historical.
        """
        n = 0
        for m in self.modules():
            if isinstance(m, Qwen2_5_VLAttentionDAT) and m.k_proj_hd is not None:
                m._init_hd_proj_weights()
                n += 1
        logger.info(
            f"Zero-init adapter pattern (K=Kaiming, V=0) applied to "
            f"k_proj_hd / v_proj_hd for {n} DAT layers"
        )

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
        """Generate HD feature maps from high-resolution pixel values (separate ViT call)."""
        with torch.no_grad():
            pixel_values_hd = pixel_values_hd.type(self.model.visual.dtype)
            hd_output = self.model.visual(pixel_values_hd, grid_thw=image_grid_thw_hd, return_dict=True)
            hd_embeds = hd_output.pooler_output

        return self._parse_hd_embeds(hd_embeds, image_grid_thw_hd)

    def _parse_hd_embeds(self, hd_embeds, image_grid_thw_hd):
        """Parse flat HD embeddings into per-image [H_hd, W_hd, C] feature maps."""
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

    def _fused_vit_forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.LongTensor,
        pixel_values_hd: torch.Tensor,
        image_grid_thw_hd: torch.LongTensor,
        input_ids: torch.LongTensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Fused ViT call for LR + HD in one kernel, returning inputs_embeds and HD features.

        The Qwen2.5-VL ViT internally builds cu_seqlens from grid_thw and uses
        flash_attn_varlen, so LR and HD images never cross-attend — semantically
        identical to two separate calls.

        Gradient handling:
          - ViT frozen (standard DAT) or inference: one combined ViT call.
          - ViT trainable: HD under torch.no_grad(), LR with grad (two ViT calls).
            Avoids ~10× activation memory from storing HD patch activations for backward.

        By building inputs_embeds here and passing pixel_values=None to self.model(),
        the base model's internal ViT call is bypassed in both branches.

        Returns:
            inputs_embeds: [B, seq_len, C] with LR patches already scattered in.
            image_hd_features: List[[H_hd, W_hd, C]], one feature map per image.
        """
        spatial_merge = self.config.vision_config.spatial_merge_size
        vit_dtype = self.model.visual.dtype
        # Only split calls when ViT weights are actually receiving gradients this step.
        # requires_grad=True alone is not sufficient: eval mode or torch.no_grad() context
        # means no backward will run, so the activation-memory concern doesn't apply.
        vit_trainable = any(p.requires_grad for p in self.model.visual.parameters())
        need_split = self.training and torch.is_grad_enabled() and vit_trainable

        if need_split:
            # ViT trainable + grad enabled: HD strictly no_grad to avoid storing HD activations for backward.
            with torch.no_grad():
                hd_embeds = self.model.visual(
                    pixel_values_hd.to(vit_dtype),
                    grid_thw=image_grid_thw_hd,
                    return_dict=True,
                ).pooler_output
            lr_embeds = self.model.visual(
                pixel_values.to(vit_dtype),
                grid_thw=image_grid_thw,
                return_dict=True,
            ).pooler_output
        else:
            # Inference, or training with ViT frozen / no_grad: one combined ViT call.
            # grid_thw drives cu_seqlens inside the ViT → block-diagonal attention,
            # LR and HD patches attend only within their own image boundaries.
            pv_combined = torch.cat([
                pixel_values.to(vit_dtype),
                pixel_values_hd.to(vit_dtype),
            ], dim=0)
            thw_combined = torch.cat([image_grid_thw, image_grid_thw_hd], dim=0)

            combined_embeds = self.model.visual(
                pv_combined, grid_thw=thw_combined, return_dict=True,
            ).pooler_output  # [N_lr + N_hd, C]

            split_sizes = (thw_combined.prod(-1) // spatial_merge ** 2).tolist()
            splits = torch.split(combined_embeds, [int(s) for s in split_sizes])
            n_lr = len(image_grid_thw)
            lr_embeds = torch.cat(list(splits[:n_lr]), dim=0)   # [N_lr, C]
            hd_embeds = torch.cat(list(splits[n_lr:]), dim=0)   # [N_hd, C]

        image_hd_features = self._parse_hd_embeds(hd_embeds, image_grid_thw_hd)

        # Scatter LR embeddings into token sequence (replaces base model's internal scatter).
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        lr_embeds = lr_embeds.to(inputs_embeds.dtype)
        image_mask, _ = self.model.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=lr_embeds,
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, lr_embeds)

        return inputs_embeds, image_hd_features

    def _shared_vit_forward(
        self,
        pixel_values_hd: torch.Tensor,
        image_grid_thw_hd: torch.LongTensor,
        image_grid_thw: torch.LongTensor,
        input_ids: torch.LongTensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Shared-ViT path: one HD ViT call, LR tokens are pooled from HD features.

        Trades one ViT forward for ``adaptive_avg_pool2d`` — the theoretical lower
        bound for LR+HD encoding cost (≈ HD-only ViT baseline).

        Semantics:
          - LR tokens come from ``F.adaptive_avg_pool2d(hd_feat, (lr_h, lr_w))``.
            Different from running the LR image through the ViT separately
            (the baseline path), so models trained with ``use_shared_vit=False``
            are not compatible — retraining is required.

        Gradient handling:
          Entire ViT call is wrapped in ``torch.no_grad()`` — identical to the
          ``_generate_hd_features`` convention. ViT is treated as frozen
          feature extractor; neither LR pool nor DAT grid_sample can back-prop
          into ViT weights. Pool / LLM still train normally on top of the
          frozen HD features.

        Args:
            pixel_values_hd:   HD pixel values.
            image_grid_thw_hd: HD grid dims (drives cu_seqlens inside ViT).
            image_grid_thw:    LR grid dims. Only used to derive the pool target
                               (lr_h_merged, lr_w_merged) and to validate token
                               counts in ``get_placeholder_mask``. pixel_values (LR)
                               is never touched.
            input_ids:         token IDs, used to scatter LR tokens into
                               ``inputs_embeds``.

        Returns:
            inputs_embeds:     [B, seq_len, C] with pooled LR tokens scattered in.
            image_hd_features: List[[H_hd, W_hd, C]] — HD features for DAT layers.
        """
        spatial_merge = self.config.vision_config.spatial_merge_size
        vit_dtype = self.model.visual.dtype

        with torch.no_grad():
            hd_embeds = self.model.visual(
                pixel_values_hd.to(vit_dtype),
                grid_thw=image_grid_thw_hd,
                return_dict=True,
            ).pooler_output
            image_hd_features = self._parse_hd_embeds(hd_embeds, image_grid_thw_hd)

            lr_feats: List[torch.Tensor] = []
            for i, hd_feat in enumerate(image_hd_features):
                thw_lr = image_grid_thw[i]
                lr_h = int(thw_lr[1].item()) // spatial_merge
                lr_w = int(thw_lr[2].item()) // spatial_merge

                hd_chw = hd_feat.permute(2, 0, 1).unsqueeze(0)  # [1, C, H_hd, W_hd]
                # adaptive_avg_pool2d is more robust than fixed kernel: LR / HD dims
                # are independently derived from the processor (not strictly hr_scale×).
                # Upcast to fp32 when in bf16/fp16 to avoid accumulation drift.
                pool_dtype = hd_chw.dtype
                pooled = F.adaptive_avg_pool2d(
                    hd_chw.float() if pool_dtype in (torch.bfloat16, torch.float16) else hd_chw,
                    (lr_h, lr_w),
                ).to(pool_dtype)
                lr_feats.append(pooled.squeeze(0).permute(1, 2, 0).reshape(lr_h * lr_w, -1))

            lr_embeds = torch.cat(lr_feats, dim=0)  # [sum(lr_h*lr_w), C]

        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        lr_embeds = lr_embeds.to(inputs_embeds.dtype)
        image_mask, _ = self.model.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=lr_embeds,
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, lr_embeds)

        return inputs_embeds, image_hd_features

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
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # === Step 1: ViT feature extraction ===
        # Three-way dispatch (priority: shared > fused > separate):
        #   · shared : single HD ViT call; LR tokens = adaptive_avg_pool2d(HD features).
        #              ViT cost ≈ HD-only baseline. Requires retraining (new LR semantics).
        #   · fused  : one ViT call on cat([LR, HD]). Mathematically identical to the
        #              separate path (ViT is block-diagonal via cu_seqlens). Saves kernel
        #              launches; costs ~2× activation memory when ViT is trainable.
        #   · separate: legacy path, LR ViT + HD ViT in two calls.
        _pixel_values_for_model = pixel_values       # passed to self.model(); set to None when ViT bypassed
        _image_grid_thw_for_model = image_grid_thw   # same

        _use_shared_vit = self.config.dat_extra_args.get('use_shared_vit', False)
        _use_fused_vit = self.config.dat_extra_args.get('use_fused_vit', False)

        _have_full_vit_inputs = (
            image_hd_features is None
            and pixel_values_hd is not None and image_grid_thw_hd is not None
            and image_grid_thw is not None
            and input_ids is not None and inputs_embeds is None
        )

        if _use_shared_vit and _have_full_vit_inputs:
            # Shared path: one HD ViT call; LR tokens are pooled from HD features.
            # pixel_values (LR) is intentionally ignored — LR semantics come from HD.
            inputs_embeds, image_hd_features = self._shared_vit_forward(
                pixel_values_hd, image_grid_thw_hd,
                image_grid_thw,
                input_ids,
            )
            _pixel_values_for_model = None
            _image_grid_thw_for_model = None
        elif (_use_fused_vit
                and _have_full_vit_inputs
                and pixel_values is not None):
            # Fused path: one ViT call covers both LR and HD (saves kernel launches;
            # costs ~2× activation memory vs separate calls — enable only when VRAM allows).
            inputs_embeds, image_hd_features = self._fused_vit_forward(
                pixel_values, image_grid_thw,
                pixel_values_hd, image_grid_thw_hd,
                input_ids,
            )
            _pixel_values_for_model = None
            _image_grid_thw_for_model = None
        elif image_hd_features is None and pixel_values_hd is not None and image_grid_thw_hd is not None:
            # Fallback: HD-only separate call (inputs_embeds already provided or no pixel_values).
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
        # transformers 5.5+ requires mm_token_type_ids (0=text,1=image,2=video) and
        # get_rope_index now returns [3, B, seq_len]. Older 5.x used [4, B, seq_len]
        # (dim-0 = text); we tolerate both by stripping the leading text dim.
        mrope_position_ids = None
        if image_hd_features is not None and position_ids is None and input_ids is not None:
            # Build mm_token_type_ids from input_ids (processor-side helper isn't
            # available here since the data pipeline doesn't emit it).
            mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)
            image_token_id = getattr(self.config, 'image_token_id', None)
            if image_token_id is not None:
                mm_token_type_ids[input_ids == image_token_id] = 1
            video_token_id = getattr(self.config, 'video_token_id', None)
            if video_token_id is not None:
                mm_token_type_ids[input_ids == video_token_id] = 2
            position_ids, rope_deltas = self.model.get_rope_index(
                input_ids,
                mm_token_type_ids,
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
            pixel_values=_pixel_values_for_model,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=_image_grid_thw_for_model,
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
    'hd_gate',
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

    # Now that pretrained weights are loaded, copy k_proj/v_proj → k_proj_hd/v_proj_hd
    dat_model.init_hd_proj_from_kv()

    # Force fp32 storage for DAT scalar / near-unity-weight params.
    # ``from_pretrained(torch_dtype=bf16)`` casts the *weights* of missing
    # submodules (such as ``hd_input_layernorm``) to bf16 directly via
    # ``state_dict`` rewriting, bypassing the module's ``_apply`` chain.
    # ``Qwen2_5_VLAttentionDAT._apply`` will re-protect these on every later
    # ``.to(...)``/``.cuda()`` call, but we still need an explicit pass
    # immediately after load to fix the initial bf16 cast.
    n_fixed = 0
    for m in dat_model.modules():
        if isinstance(m, Qwen2_5_VLAttentionDAT):
            if m.hd_gate is not None and m.hd_gate.dtype != torch.float32:
                with torch.no_grad():
                    m.hd_gate.data = m.hd_gate.data.to(torch.float32)
                n_fixed += 1
            if (
                m.hd_input_layernorm is not None
                and m.hd_input_layernorm.weight.dtype != torch.float32
            ):
                with torch.no_grad():
                    m.hd_input_layernorm.weight.data = (
                        m.hd_input_layernorm.weight.data.to(torch.float32)
                    )
                n_fixed += 1
            # Offset-path fp32 protection (see Qwen2_5_VLAttentionDAT._apply).
            for sub in (m.conv_lr_dw, m.ln_1, m.conv_lr_proj,
                        m.proj_intention, m.ln_2, m.conv_off_proj):
                if not isinstance(sub, nn.Module):
                    continue
                for p in sub.parameters(recurse=False):
                    if p.dtype != torch.float32:
                        with torch.no_grad():
                            p.data = p.data.to(torch.float32)
                        n_fixed += 1
    if n_fixed > 0:
        logger.info(
            f"[DAT] Forced {n_fixed} DAT scalar/near-unity params back to fp32 "
            f"after from_pretrained (anti-bf16-roundoff)."
        )

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


def get_lora_target_modules(dat_layers_str, target_layers="all"):
    """Build regex pattern for PEFT LoRA targeting QKVO projections.

    Args:
        dat_layers_str: Layer type string e.g. 'DLLLLLDLLLLL...'
            (L=standard, D=DAT). Length must equal num_hidden_layers.
        target_layers: 'dat' applies LoRA only to DAT layer QKVO,
            'all' applies LoRA to every decoder layer's QKVO.

    Returns:
        Regex pattern string suitable for ``LoraConfig(target_modules=...)``.
        Matched against full module keys via ``re.fullmatch``.
    """
    qkvo = r"(q_proj|k_proj|v_proj|o_proj)"
    if target_layers == "dat" and dat_layers_str:
        dat_indices = [str(i) for i, c in enumerate(dat_layers_str) if c == 'D']
        layer_pattern = "|".join(dat_indices)
        return rf"model\.language_model\.layers\.({layer_pattern})\.self_attn\.{qkvo}"
    else:
        return rf"model\.language_model\.layers\.\d+\.self_attn\.{qkvo}"


# ============================================================================
# Checkpoint conversion mapping
# ============================================================================
# transformers ≥ 5.x uses a *model_type-keyed* conversion table to translate
# between the flat on-disk layout (``model.layers.*``, ``visual.*``) used by
# the official Qwen2.5-VL checkpoints and the hierarchical in-memory layout
# (``model.language_model.layers.*``, ``model.visual.*``) used by the
# ``Qwen2_5_VLForConditionalGeneration`` class.
#
# ``qwen2_5_vl`` is registered as an alias of ``qwen2_vl`` in
# ``transformers/conversion_mapping.py``, but our DAT subclass advertises
# ``model_type='qwen2_5_vl_dat'`` which has NO entry, producing an asymmetric
# save/load:
#
#   save_pretrained()  → flat keys on disk (pollution path, see the note in
#                        ``Qwen2_5_VLDATConfig.__init__``)
#   from_pretrained()  → no remap → flat keys never translated to hierarchical
#                        names → every parameter is "unexpected" / "missing".
#
# Registering the DAT ``model_type`` to reuse the qwen2_5_vl rules makes the
# save/load path fully symmetric and also lets freshly-saved merged DAT
# checkpoints load cleanly via ``Qwen2_5_VLDATForConditionalGeneration
# .from_pretrained``.
try:
    from transformers.conversion_mapping import (
        get_checkpoint_conversion_mapping as _get_ckpt_mapping,
        register_checkpoint_conversion_mapping as _register_ckpt_mapping,
    )

    _qwen25vl_mapping = _get_ckpt_mapping("qwen2_5_vl")
    if _qwen25vl_mapping is not None and _get_ckpt_mapping("qwen2_5_vl_dat") is None:
        try:
            _register_ckpt_mapping("qwen2_5_vl_dat", _qwen25vl_mapping, overwrite=False)
        except ValueError:
            # Already registered by another import path — fine.
            pass
except ImportError:
    # Older transformers versions don't expose the conversion_mapping module;
    # nothing to do (those versions also don't apply the remap in the first
    # place, so the flat/hierarchical mismatch never arises).
    pass
