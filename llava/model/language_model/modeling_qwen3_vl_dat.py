"""
Qwen3VL-DAT: Dynamic Attention Token extension for Qwen3-VL.

Extends official Qwen3VLForConditionalGeneration with DAT mechanism:
- Offset-based sampling from high-resolution vision features
- Per-layer HD feature injection via modified attention
- Proper 3D interleaved mRoPE position encoding for HD tokens
- QKNorm (RMSNorm on Q and K heads) applied consistently to HD keys

Architecture (transformers >= 5.x):
    Qwen3VLDATForConditionalGeneration(Qwen3VLForConditionalGeneration)
    ├── model: Qwen3VLModel (unmodified — kwargs flow natively)
    │   ├── visual: Qwen3VLVisionModel  (shared for LR & HR)
    │   │   └── deepstack: intermediate features injected between layers (independent of DAT)
    │   └── language_model: Qwen3VLTextModel (unmodified)
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

Key differences from Qwen2.5-VL DAT:
    - QKNorm: k_norm applied to HD keys (matching standard path consistency)
    - Interleaved mRoPE: uses Qwen3VLTextRotaryEmbedding (THWTHW layout)
    - DeepStack: coexists independently (ViT intermediate features → early LLM layers)
    - 4D position IDs: [text, T, H, W] with text split off for causal mask
"""

import logging
import math
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLModel,
    Qwen3VLTextDecoderLayer,
    Qwen3VLTextAttention,
    Qwen3VLTextRMSNorm,
    Qwen3VLTextRotaryEmbedding,
    Qwen3VLCausalLMOutputWithPast,
    apply_rotary_pos_emb,
    rotate_half,
)
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig, Qwen3VLTextConfig
from transformers.cache_utils import Cache


# ── Utility: repeat_kv for GQA ───────────────────────────────────────────────
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads for grouped-query attention. [B, H_kv, N, D] → [B, H_q, N, D]."""
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


# ── Backend selection ────────────────────────────────────────────────────────
_FA_BACKEND = "fa2"

_flash_attn_func = None
_flash_attn_varlen_func = None
_FA_HAS_SOFTMAX_LSE = False

if _FA_BACKEND == "fa4":
    from flash_attn.cute import flash_attn_func as _flash_attn_func
    from flash_attn.cute import flash_attn_varlen_func as _flash_attn_varlen_func
    import flash_attn as _fa_mod
    _fa_ver = getattr(_fa_mod, "__version__", "unknown")
    print(f"[DAT-LSE] flash_attn 4 (cute) v{_fa_ver} — return_lse=True + varlen")

elif _FA_BACKEND == "fa3":
    try:
        from flash_attn_interface import flash_attn_func as _flash_attn_func
        from flash_attn_interface import flash_attn_varlen_func as _flash_attn_varlen_func
        print("[DAT-LSE] flash_attn 3 (hopper) — return_attn_probs=True + varlen")
    except ImportError:
        import inspect as _inspect
        from flash_attn import flash_attn_func as _flash_attn_func
        from flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func
        import flash_attn as _fa_mod
        _FA_HAS_SOFTMAX_LSE = "return_softmax_lse" in _inspect.signature(_flash_attn_func).parameters
        _fa_ver = getattr(_fa_mod, "__version__", "unknown")
        _FA_BACKEND = "fa2"
        _lse_api = "return_softmax_lse" if _FA_HAS_SOFTMAX_LSE else "return_attn_probs"
        print(f"[DAT-LSE] fa3 not available, fell back to flash_attn 2 v{_fa_ver} — {_lse_api} + varlen")

elif _FA_BACKEND == "fa2":
    import inspect as _inspect
    from flash_attn import flash_attn_func as _flash_attn_func
    from flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func
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

    q_packed = torch.cat(q_list, dim=0)
    k_packed = torch.cat(k_list, dim=0)
    v_packed = torch.cat(v_list, dim=0)

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

    out_list = []
    lse_list = []
    q_offset = 0
    for i in range(n_segs):
        nq = nq_lens[i]
        out_list.append(out_packed[q_offset:q_offset + nq].unsqueeze(0))
        lse_list.append(lse_packed[:, q_offset:q_offset + nq].unsqueeze(0))
        q_offset += nq

    return out_list, lse_list


logger = logging.getLogger(__name__)

# Qwen3-VL special token IDs
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
# FP32 Weight Helpers (anti-bf16-roundoff)
# ============================================================================

class _FP32WeightRMSNorm(Qwen3VLTextRMSNorm):
    """RMSNorm that keeps weight in fp32 to avoid bf16 round-off near unity."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        h = hidden_states.to(torch.float32)
        variance = h.pow(2).mean(-1, keepdim=True)
        h = h * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * h).to(input_dtype)


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
    """LayerNorm2d with fp32 weight/bias storage."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        h = x.to(torch.float32)
        u = h.mean(1, keepdim=True)
        s = (h - u).pow(2).mean(1, keepdim=True)
        h = (h - u) / torch.sqrt(s + self.eps)
        h = self.weight[None, :, None, None] * h + self.bias[None, :, None, None]
        return h.to(input_dtype)


class _FP32WeightConv2d(nn.Conv2d):
    """Conv2d with fp32 master weight/bias; forward downcasts to input dtype."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight.dtype != x.dtype:
            w = self.weight.to(x.dtype)
            b = self.bias.to(x.dtype) if self.bias is not None else None
            return F.conv2d(x, w, b, self.stride, self.padding, self.dilation, self.groups)
        return super().forward(x)


class _FP32WeightLinear(nn.Linear):
    """Linear with fp32 master weight/bias; forward downcasts to input dtype."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight.dtype != x.dtype:
            w = self.weight.to(x.dtype)
            b = self.bias.to(x.dtype) if self.bias is not None else None
            return F.linear(x, w, b)
        return super().forward(x)


# ============================================================================
# RoPE Helper for Single Tensor
# ============================================================================

def apply_rotary_pos_emb_single(x, cos, sin, unsqueeze_dim=1):
    """Apply RoPE to a single tensor (Q or K separately).

    Needed when Q and KV have different sequence lengths in DAT layers.
    Uses the same interleaved mRoPE layout as Qwen3-VL.

    Args:
        x: [batch, heads, seq_len, head_dim]
        cos: [batch, seq_len, head_dim] or broadcastable
        sin: [batch, seq_len, head_dim] or broadcastable
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


# ============================================================================
# Config
# ============================================================================

class Qwen3VLDATConfig(Qwen3VLConfig):
    """Qwen3-VL config extended with DAT parameters."""
    model_type = "qwen3_vl_dat"

    def __init__(self, dat_extra_args=None, **kwargs):
        kwargs.pop("model_type", None)
        super().__init__(**kwargs)
        self.dat_extra_args = dat_extra_args or {
            'grid_size': 6,
            'off_ksize': 3,
            'off_grps': 1,
            'inter_size': 64,
            'hr_scale': 3,
            'hd_proj': True,
            'layers': '',
            'use_intention_branch': True,
            'intention_as_gate': True,
            'hd_gate_init': None,
            'hd_gate_freeze': False,
            'use_fused_vit': False,
            'use_shared_vit': False,
            'use_spatial_attn_guide': True,
            'image_hd_for_question': False,
        }


# ============================================================================
# Helpers
# ============================================================================

def compute_image_range_list(input_ids, labels, image_token_id,
                              im_start_token_id=IM_START_TOKEN_ID,
                              image_grid_thw=None, spatial_merge_size=2):
    """Compute image_range_list from Qwen3-VL-format inputs.

    Scans input_ids for image token regions and labels for answer regions.
    For each answer range, dynamically locates the preceding <|im_start|>
    token as the intention_idx.

    Returns:
        List per batch of:
            [[(lr_start, lr_end, lr_h, lr_w), ...per image...],
             [ans1_start, ans1_end, intention_idx], ...]
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

        runs = []
        run_start = image_indices[0].item()
        prev = run_start
        for idx in image_indices[1:].tolist():
            if idx != prev + 1:
                runs.append((run_start, prev + 1))
                run_start = idx
            prev = idx
        runs.append((run_start, prev + 1))

        lr_tuples = []
        for (r_start, r_end) in runs:
            if image_grid_thw is not None and img_idx < len(image_grid_thw):
                thw = image_grid_thw[img_idx]
                h, w = thw[1].item(), thw[2].item()
                lr_h = h // spatial_merge_size
                lr_w = w // spatial_merge_size
                img_idx += 1
            else:
                lr_len = r_end - r_start
                lr_h = lr_w = int(lr_len ** 0.5)
            lr_tuples.append((r_start, r_end, lr_h, lr_w))

        ranges.append(lr_tuples)

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
# DAT Attention (Qwen3-VL)
# ============================================================================

class Qwen3VLTextAttentionDAT(Qwen3VLTextAttention):
    """
    Core DAT mechanism for Qwen3-VL (two-pass + LSE merge):
    1. Extract LR features from query → generate sampling offsets
    2. Grid sample from HD features → project to KV (key_hd, value_hd)
    3. Pass 1: standard causal attention (full sequence, static shapes)
    4. Pass 2: HD cross-attention per answer segment (Q_ans × K_hd, non-causal)
    5. Merge outputs via LSE trick — mathematically equivalent to joint attention

    New vs Qwen2.5-VL DAT:
    - QKNorm: self.k_norm applied to HD keys for Pass 1/Pass 2 consistency
    - Interleaved mRoPE: uses Qwen3VLTextRotaryEmbedding for HD positions
    """

    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int, dat_extra_args: dict):
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

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.off_dim = self.hidden_size // self.off_grps

        # --- Offset generation pipeline (fp32-storage subclasses) ---
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
            self.k_proj_hd = nn.Linear(self.hidden_size, kv_dim, bias=config.attention_bias)
            self.v_proj_hd = nn.Linear(self.hidden_size, kv_dim, bias=config.attention_bias)
            self.hd_input_layernorm = _FP32WeightRMSNorm(
                self.hidden_size, eps=config.rms_norm_eps
            )
        else:
            self.k_proj_hd = None
            self.v_proj_hd = None
            self.hd_input_layernorm = None

        # Rotary embedding for HD positions (Qwen3-VL interleaved mRoPE)
        self._dat_rotary_emb = Qwen3VLTextRotaryEmbedding(config=config)

        # Learnable HD gate
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

        self.dat_inject_lr_image = bool(dat.get('inject_lr_image', False))
        self.dat_image_hd_for_question = bool(dat.get('image_hd_for_question', False))
        self.dat_multi_image = (
            bool(dat.get('multi_image', False))
            or os.environ.get('DAT_MULTI_IMAGE', '0') == '1'
        )

        self._init_dat_weights()

    def _apply(self, fn, recurse=True):
        """Force fp32 storage for DAT scalar/near-unity params after any .to() call."""
        result = super()._apply(fn, recurse=recurse)
        if self.hd_gate is not None and self.hd_gate.dtype != torch.float32:
            with torch.no_grad():
                self.hd_gate.data = self.hd_gate.data.to(torch.float32)
        if (
            self.hd_input_layernorm is not None
            and self.hd_input_layernorm.weight.dtype != torch.float32
        ):
            with torch.no_grad():
                self.hd_input_layernorm.weight.data = (
                    self.hd_input_layernorm.weight.data.to(torch.float32)
                )
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
        nn.init.kaiming_normal_(self.conv_lr_dw.weight)
        nn.init.kaiming_normal_(self.conv_lr_proj.weight)
        if self.conv_lr_proj.bias is not None:
            nn.init.zeros_(self.conv_lr_proj.bias)
        nn.init.zeros_(self.conv_off_proj.weight)
        if isinstance(self.proj_intention, nn.Linear):
            nn.init.xavier_uniform_(self.proj_intention.weight)
            if self.proj_intention.bias is not None:
                nn.init.zeros_(self.proj_intention.bias)
        self._init_hd_proj_weights()

    @torch.no_grad()
    def _init_hd_proj_weights(self):
        """Zero-init adapter pattern: K=Kaiming, V=0."""
        if self.k_proj_hd is None:
            return
        nn.init.kaiming_normal_(self.k_proj_hd.weight, nonlinearity='linear')
        nn.init.zeros_(self.v_proj_hd.weight)
        if self.k_proj_hd.bias is not None:
            nn.init.zeros_(self.k_proj_hd.bias)
        if self.v_proj_hd.bias is not None:
            nn.init.zeros_(self.v_proj_hd.bias)

    def _grid_generate(self, h, w, n_repeats, device):
        """Generate reference sampling grid with half-cell margin from [-1,1] boundary."""
        m_y = 1.0 / max(h - 1, 1)
        m_x = 1.0 / max(w - 1, 1)
        grid_y = torch.linspace(-1.0 + m_y, 1.0 - m_y, h, device=device, dtype=torch.float32)
        grid_x = torch.linspace(-1.0 + m_x, 1.0 - m_x, w, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=0)
        return grid.unsqueeze(0).repeat(n_repeats * self.off_grps, 1, 1, 1)

    def _construct_hd_position_ids(self, pos_3d_b, lr_start, lr_end, lr_h, lr_w, device):
        """Construct 3D position IDs for HD tokens (interleaved mRoPE compatible).

        Args:
            pos_3d_b: [3, Nq] — T/H/W mRoPE positions for this batch element
            lr_start: start index of LR image tokens
            lr_end: end index of LR image tokens
            lr_h, lr_w: LR image spatial dims
            device: torch device

        Returns:
            hd_pos: [3, Ns] — T/H/W position IDs for HD tokens
        """
        Ns = self.grid_size * self.grid_size

        lr_pos = pos_3d_b[:, lr_start:lr_end]  # [3, lr_h * lr_w]

        # Temporal: constant (same as LR image tokens)
        t_val = lr_pos[0, 0]

        # Height/Width: interpolate from LR position range
        h_min = lr_pos[1].min()
        h_max = lr_pos[1].max()
        w_min = lr_pos[2].min()
        w_max = lr_pos[2].max()

        grid_y = torch.linspace(0, 1, self.grid_size, device=device, dtype=torch.float32)
        grid_x = torch.linspace(0, 1, self.grid_size, device=device, dtype=torch.float32)

        h_hd = (grid_y * (h_max - h_min) + h_min).long()
        w_hd = (grid_x * (w_max - w_min) + w_min).long()

        h_grid = h_hd.unsqueeze(1).expand(-1, self.grid_size).flatten()
        w_grid = w_hd.unsqueeze(0).expand(self.grid_size, -1).flatten()
        t_grid = t_val.expand(Ns).long()

        return torch.stack([t_grid, h_grid, w_grid])  # [3, Ns]

    def _sample_hd_from_off_guide(self, off_guide, image_hd_features, hd_feat_idx, Lp, device):
        """Core deformable sampling: off_guide -> offsets -> grid_sample -> KV.

        Returns:
            key_hd:        [Lp, Ns, kv_dim]
            value_hd:      [Lp, Ns, kv_dim]
            sampling_locs: [Lp, off_grps, grid_size, grid_size, 2]
        """
        offsets = self.conv_off_proj(F.silu(self.ln_2(off_guide))).float()
        if self.training:
            self._dat_offset_stats = (
                offsets.detach().mean().item(),
                offsets.detach().std().item(),
            )
        references = self._grid_generate(offsets.size(2), offsets.size(3), Lp, device)

        x = references + offsets
        sample_locs = (x + (x.clamp(-1, 1) - x).detach()).permute(0, 2, 3, 1)

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
        ).to(orig_dtype)

        sampled_hr = einops.rearrange(
            sampled_hr,
            '(l g) c h w -> l (h w) (g c)',
            l=Lp, g=self.off_grps, c=self.off_dim,
        )

        if self.hd_proj:
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

    def _generate_offsets_and_sample(self, query_states, image_hd_features, image_range_list, b_idx, hd_feat_idxs, want_image=False):
        """Generate intention-conditioned sampling offsets and sample HD K/V.

        Multi-image: each image is sampled independently and K/V are concatenated.

        Returns:
            key_hd:       [Lp, M*Ns, kv_dim]
            value_hd:     [Lp, M*Ns, kv_dim]
            sampling_locs:[Lp, off_grps, grid_h, grid_w, 2]
            key_img:      [1, M*Ns, kv_dim] or None
            value_img:    [1, M*Ns, kv_dim] or None
        """
        device = query_states.device

        lr_list = image_range_list[b_idx][0]
        answer_ranges = image_range_list[b_idx][1:]
        Lp = len(answer_ranges)

        intention_indices = None
        embed_intention = None
        if self.use_intention_branch:
            intention_indices = [ar[2] for ar in answer_ranges]
            intention_tokens = query_states[b_idx, intention_indices]
            intention_per_group = einops.rearrange(
                intention_tokens, 'l (g c) -> l g c',
                g=self.off_grps, c=self.off_dim,
            )
            embed_intention = self.proj_intention(intention_per_group)
            embed_intention = einops.rearrange(
                embed_intention, 'l g c -> (l g) c 1 1',
            )

        if want_image:
            assert not (self.use_intention_branch and not self.intention_as_gate), (
                "image-conditioned HD requires intention_as_gate=True "
                "(or use_intention_branch=False)"
            )

        key_parts, value_parts = [], []
        kimg_parts, vimg_parts = [], []
        slocs_first = None

        for m, (lr_start, lr_end, lr_h, lr_w) in enumerate(lr_list):
            hd_feat_idx = hd_feat_idxs[m]
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
            local_embed_lr = self.conv_lr_dw(img_lr)
            local_embed_lr = F.silu(self.ln_1(local_embed_lr))
            embed_lr = self.conv_lr_proj(local_embed_lr)
            embed_lr = F.adaptive_avg_pool2d(embed_lr, (self.grid_size, self.grid_size))

            embed_lr_rep = einops.repeat(
                embed_lr, 'g c h w -> (l g) c h w', l=Lp,
            )

            if self.use_intention_branch and self.use_spatial_attn_guide:
                q_lr_flat = query_states[b_idx, image_range_index]
                q_int_flat = query_states[b_idx, intention_indices]
                spatial_attn = torch.matmul(
                    q_int_flat.float(), q_lr_flat.float().transpose(0, 1),
                ) / math.sqrt(q_lr_flat.shape[-1])
                spatial_attn = spatial_attn.softmax(dim=-1).view(Lp, 1, lr_h, lr_w)
                spatial_attn_guide = F.adaptive_avg_pool2d(
                    spatial_attn, (self.grid_size, self.grid_size),
                ) * (lr_h * lr_w)
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
                    off_guide = torch.cat([
                        embed_lr_rep,
                        embed_intention.expand(-1, -1, self.grid_size, self.grid_size),
                    ], dim=1)
            else:
                off_guide = embed_lr_rep

            if want_image:
                off_guide_img = einops.repeat(embed_lr, 'g c h w -> (l g) c h w', l=1)
                off_guide_all = torch.cat([off_guide_img, off_guide], dim=0)
                key_all, value_all, slocs_all = self._sample_hd_from_off_guide(
                    off_guide_all, image_hd_features, hd_feat_idx, Lp + 1, device,
                )
                kimg_parts.append(key_all[0:1])
                vimg_parts.append(value_all[0:1])
                key_parts.append(key_all[1:])
                value_parts.append(value_all[1:])
                if slocs_first is None:
                    slocs_first = slocs_all[1:]
            else:
                key_hd, value_hd, slocs = self._sample_hd_from_off_guide(
                    off_guide, image_hd_features, hd_feat_idx, Lp, device,
                )
                key_parts.append(key_hd)
                value_parts.append(value_hd)
                if slocs_first is None:
                    slocs_first = slocs

        key_hd = torch.cat(key_parts, dim=1)
        value_hd = torch.cat(value_parts, dim=1)

        if want_image:
            key_img = torch.cat(kimg_parts, dim=1)
            value_img = torch.cat(vimg_parts, dim=1)
            return key_hd, value_hd, slocs_first, key_img, value_img

        return key_hd, value_hd, slocs_first, None, None

    def _merge_two_pass_lse(
        self,
        out1: torch.Tensor,
        lse1: torch.Tensor,
        out2: torch.Tensor,
        lse2: torch.Tensor,
        ans_start: int,
        ans_end: int,
    ) -> torch.Tensor:
        """Merge causal attention and HD cross-attention outputs via the LSE trick."""
        out = out1.clone()

        lse1_ans = lse1[:, :, ans_start:ans_end].float()
        lse2 = lse2.float()
        out1_ans = out1[:, ans_start:ans_end, :, :]

        if self.hd_gate is not None:
            lse2 = lse2 + F.logsigmoid(self.hd_gate)
            if self.training:
                self._dat_hd_gate_value = self.hd_gate.detach().sigmoid().item()

        lse = torch.logaddexp(lse1_ans, lse2)

        w1 = (lse1_ans - lse).exp().permute(0, 2, 1).unsqueeze(-1)
        w2 = (lse2     - lse).exp().permute(0, 2, 1).unsqueeze(-1)

        out[:, ans_start:ans_end, :, :] = (w1 * out1_ans + w2 * out2).to(out.dtype)

        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        # DAT-specific kwargs (flow through **kwargs chain)
        image_hd_features: Optional[List[torch.Tensor]] = None,
        image_range_list: Optional[List[List]] = None,
        mrope_position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # --- Standard path: no image data ---
        if image_range_list is None or image_hd_features is None:
            return super().forward(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )

        has_images = any(len(r) > 0 for r in image_range_list)
        if not has_images:
            return super().forward(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )

        # Inference decode phase: HD already cached during prefill
        if past_key_values is not None and past_key_values.get_seq_length(self.layer_idx) > 0:
            return super().forward(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )

        # --- DAT path (two-pass + LSE merge) ---
        input_shape = hidden_states.shape[:-1]
        B, Nq = input_shape
        C = hidden_states.shape[-1]
        device = hidden_states.device
        Ns = self.grid_size * self.grid_size
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Project Q, K, V with QKNorm
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # Apply interleaved mRoPE
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # KV cache
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx,
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if query_states.device.type == "cuda":
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # === Pass 1: standard causal attention + LSE ===
        out1, lse1 = _dat_attn_with_lse(query_states, key_states, value_states, causal=True)

        # Build mapping from batch index to image_hd_features indices
        b_idx_to_hd_idxs: Dict[int, List[int]] = {}
        hd_idx = 0
        for b in range(B):
            if len(image_range_list[b]) > 0:
                n_img = len(image_range_list[b][0])
                b_idx_to_hd_idxs[b] = list(range(hd_idx, hd_idx + n_img))
                hd_idx += n_img

        # Use the raw hidden_states for offset generation (pre-projection features)
        query_for_offsets = hidden_states

        # === Pass 2: HD cross-attention (batched via varlen) ===
        seg_q_list: List[torch.Tensor] = []
        seg_k_list: List[torch.Tensor] = []
        seg_v_list: List[torch.Tensor] = []
        seg_meta: List[Tuple[int, int, int]] = []

        for b_idx in range(B):
            if len(image_range_list[b_idx]) <= 1:
                continue

            lr_list = image_range_list[b_idx][0]
            M = len(lr_list)
            if M > 1 and not self.dat_multi_image:
                continue

            hd_feat_idxs = b_idx_to_hd_idxs[b_idx]
            Ns_total = Ns * M

            if mrope_position_ids is not None:
                orig_pos_b = mrope_position_ids[:, b_idx, :]  # [3, Nq]
            else:
                orig_pos_b = torch.arange(Nq, device=device, dtype=torch.long).unsqueeze(0).expand(3, -1)

            # Per-image HD positions concatenated → [3, M*Ns]
            hd_pos_ids = torch.cat([
                self._construct_hd_position_ids(orig_pos_b, _s, _e, _h, _w, device)
                for (_s, _e, _h, _w) in lr_list
            ], dim=1)
            hd_pos_ids_batched = hd_pos_ids.unsqueeze(1)  # [3, 1, M*Ns]

            # Direction A: image-conditioned HD for question tokens
            question_segs: List[Tuple[int, int]] = []
            if self.dat_image_hd_for_question and M == 1:
                _q_prev_end = lr_list[0][1]
                for _ar in image_range_list[b_idx][1:]:
                    _a_s, _a_e, _a_int = _ar
                    if _a_e > 0:
                        _q_ans_start = _a_s
                        _next_prev = _a_e
                    else:
                        _q_ans_start = _a_int + 1
                        _next_prev = Nq
                    if _q_ans_start > _q_prev_end:
                        question_segs.append((_q_prev_end, _q_ans_start))
                    _q_prev_end = _next_prev

            # Fused sampling: answer K/V + optional image-conditioned K/V
            key_hd_all, value_hd_all, _slocs, k_img_all, v_img_all = \
                self._generate_offsets_and_sample(
                    query_for_offsets, image_hd_features, image_range_list,
                    b_idx, hd_feat_idxs, want_image=bool(question_segs),
                )

            if question_segs:
                # RoPE the shared image-HD K/V
                k_img = (k_img_all[0]
                         .view(Ns_total, self.num_key_value_heads, self.head_dim)
                         .unsqueeze(0).transpose(1, 2))
                v_img = (v_img_all[0]
                         .view(Ns_total, self.num_key_value_heads, self.head_dim)
                         .unsqueeze(0).transpose(1, 2))

                # Apply QKNorm to HD keys
                k_img = self.k_norm(k_img.transpose(1, 2).reshape(1, Ns_total, -1).view(1, Ns_total, self.num_key_value_heads, self.head_dim)).transpose(1, 2)

                # Compute HD position embeddings via interleaved mRoPE
                cos_img, sin_img = self._dat_rotary_emb(k_img, hd_pos_ids_batched)
                k_img = apply_rotary_pos_emb_single(k_img, cos_img, sin_img)

                k_img = repeat_kv(k_img, self.num_key_value_groups)
                v_img = repeat_kv(v_img, self.num_key_value_groups)
                k_img_seg = k_img.squeeze(0).transpose(0, 1).contiguous()
                v_img_seg = v_img.squeeze(0).transpose(0, 1).contiguous()

                for (_qs, _qe) in question_segs:
                    _nq_seg = _qe - _qs
                    if _nq_seg <= 0:
                        continue
                    q_seg = query_states[b_idx, :, _qs:_qe, :] \
                        .transpose(0, 1).contiguous()
                    seg_q_list.append(q_seg)
                    seg_k_list.append(k_img_seg)
                    seg_v_list.append(v_img_seg)
                    seg_meta.append((b_idx, _qs, _nq_seg))

            k_hd_l_first: Optional[torch.Tensor] = None
            v_hd_l_first: Optional[torch.Tensor] = None

            for l_idx, answer_range in enumerate(image_range_list[b_idx][1:]):
                ans_start = answer_range[0]
                ans_end = answer_range[1]
                intention_idx = answer_range[2]

                if ans_end > 0:
                    q_ans_start = ans_start
                    Nans = ans_end - ans_start
                else:
                    q_ans_start = intention_idx + 1
                    Nans = Nq - q_ans_start

                if Nans <= 0:
                    continue

                q_seg = query_states[b_idx, :, q_ans_start:q_ans_start + Nans, :] \
                    .transpose(0, 1).contiguous()

                k_hd_l = (key_hd_all[l_idx]
                          .view(Ns_total, self.num_key_value_heads, self.head_dim)
                          .unsqueeze(0).transpose(1, 2))
                v_hd_l = (value_hd_all[l_idx]
                          .view(Ns_total, self.num_key_value_heads, self.head_dim)
                          .unsqueeze(0).transpose(1, 2))

                # Apply QKNorm to HD keys (consistency with Pass 1)
                k_hd_l = self.k_norm(
                    k_hd_l.transpose(1, 2).reshape(1, Ns_total, self.num_key_value_heads, self.head_dim)
                ).transpose(1, 2)

                # Interleaved mRoPE for HD positions
                cos_hd, sin_hd = self._dat_rotary_emb(k_hd_l, hd_pos_ids_batched)
                k_hd_l = apply_rotary_pos_emb_single(k_hd_l, cos_hd, sin_hd)

                k_hd_l = repeat_kv(k_hd_l, self.num_key_value_groups)
                v_hd_l = repeat_kv(v_hd_l, self.num_key_value_groups)

                k_seg = k_hd_l.squeeze(0).transpose(0, 1).contiguous()
                v_seg = v_hd_l.squeeze(0).transpose(0, 1).contiguous()
                seg_k_list.append(k_seg)
                seg_v_list.append(v_seg)
                seg_q_list.append(q_seg)
                seg_meta.append((b_idx, q_ans_start, Nans))

                if l_idx == 0:
                    k_hd_l_first = k_seg
                    v_hd_l_first = v_seg

            # D1: inject HD at lr_image positions
            if (self.dat_inject_lr_image
                    and k_hd_l_first is not None
                    and v_hd_l_first is not None):
                for _m, (_s, _e, _h, _w) in enumerate(lr_list):
                    Nlr = _e - _s
                    if Nlr <= 0:
                        continue
                    q_lr = query_states[b_idx, :, _s:_e, :] \
                        .transpose(0, 1).contiguous()
                    seg_q_list.append(q_lr)
                    seg_k_list.append(k_hd_l_first[_m * Ns:(_m + 1) * Ns])
                    seg_v_list.append(v_hd_l_first[_m * Ns:(_m + 1) * Ns])
                    seg_meta.append((b_idx, _s, Nlr))

        # Phase 2b: Batched cross-attention
        if seg_q_list:
            out2_list, lse2_list = _dat_cross_attn_varlen(
                seg_q_list, seg_k_list, seg_v_list,
            )
        else:
            out2_list, lse2_list = [], []

        # Phase 2c: LSE merge
        out_parts: List[torch.Tensor] = []
        seg_iter = 0
        for b_idx in range(B):
            out_b = out1[b_idx:b_idx + 1]

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

        # [B, Nq, H, D] → [B, Nq, C]
        attn_output = out_final.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, None


# ============================================================================
# DAT Decoder Layer
# ============================================================================

class Qwen3VLTextDecoderLayerDAT(Qwen3VLTextDecoderLayer):
    """Decoder layer with DAT attention.

    Inherits from Qwen3VLTextDecoderLayer (GradientCheckpointingLayer).
    The base class forward passes **kwargs to self_attn, so DAT-specific kwargs
    flow through automatically.
    """

    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int, dat_extra_args: dict):
        super().__init__(config, layer_idx)
        self.self_attn = Qwen3VLTextAttentionDAT(config, layer_idx, dat_extra_args)


# ============================================================================
# DAT ForConditionalGeneration (top-level model)
# ============================================================================

class Qwen3VLDATForConditionalGeneration(Qwen3VLForConditionalGeneration):
    """Qwen3-VL with DAT: uses native vision encoder for both LR and HD features.

    DAT and DeepStack coexist independently:
    - DeepStack injects intermediate ViT features between early LLM layers
    - DAT uses a separate HD ViT pass and injects via two-pass attention
    """
    config_class = Qwen3VLDATConfig

    def __init__(self, config: Qwen3VLDATConfig):
        super().__init__(config)

        dat_args = config.dat_extra_args
        layers_str = dat_args.get('layers', '')
        text_config = config.text_config

        if layers_str:
            assert len(layers_str) == text_config.num_hidden_layers, (
                f"Layer string length {len(layers_str)} != num_hidden_layers {text_config.num_hidden_layers}"
            )
            for i, layer_type in enumerate(layers_str):
                if layer_type == 'D':
                    self.model.language_model.layers[i] = Qwen3VLTextDecoderLayerDAT(
                        text_config, i, dat_args
                    )
                elif layer_type == 'L':
                    pass
                else:
                    raise ValueError(f"Unknown layer type '{layer_type}' at index {i}")

            dat_count = sum(1 for c in layers_str if c == 'D')
            logger.info(f"Qwen3VL-DAT: {dat_count} DAT layers, "
                        f"{text_config.num_hidden_layers - dat_count} standard layers")

        self._patch_text_model_init_weights()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """Override to post-fix raw nn.Parameter loading for hd_gate."""
        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        cls._manual_load_dat_raw_params(model, pretrained_model_name_or_path)
        return model

    @staticmethod
    def _manual_load_dat_raw_params(model, path):
        """Post-load fix for raw nn.Parameter attributes not covered by HF key remapping."""
        if not isinstance(path, str) or not os.path.isdir(path):
            return

        weights: dict = {}
        try:
            from safetensors import safe_open
        except ImportError:
            safe_open = None

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
            logger.info(f"[DAT post-load] manually loaded hd_gate for {n_loaded} DAT layers")

    def _patch_text_model_init_weights(self):
        """Monkey-patch text model's _init_weights for DAT-specific initialization."""
        import types
        text_model = self.model.language_model
        text_model_cls = type(text_model)
        cls_init_weights = text_model_cls._init_weights
        hd_gate_init = self.config.dat_extra_args.get('hd_gate_init', None)
        hd_gate_freeze = bool(self.config.dat_extra_args.get('hd_gate_freeze', False))

        def _dat_init_weights(text_self, module):
            cls_init_weights(text_self, module)
            if isinstance(module, Qwen3VLTextAttentionDAT):
                if module.hd_gate is not None and hd_gate_init is not None:
                    module.hd_gate.data.fill_(float(hd_gate_init))
                    if hd_gate_freeze:
                        module.hd_gate.requires_grad = False
                nn.init.kaiming_normal_(module.conv_lr_dw.weight)
                nn.init.kaiming_normal_(module.conv_lr_proj.weight)
                nn.init.zeros_(module.conv_off_proj.weight)
                if module.conv_lr_proj.bias is not None:
                    nn.init.zeros_(module.conv_lr_proj.bias)
                if isinstance(module.proj_intention, nn.Linear):
                    nn.init.xavier_uniform_(module.proj_intention.weight)
                    if module.proj_intention.bias is not None:
                        nn.init.zeros_(module.proj_intention.bias)
                module._init_hd_proj_weights()

        text_model._init_weights = types.MethodType(_dat_init_weights, text_model)

    @torch.no_grad()
    def init_hd_proj_from_kv(self):
        """Apply zero-init adapter pattern (K=Kaiming, V=0) to all DAT layers."""
        n = 0
        for m in self.modules():
            if isinstance(m, Qwen3VLTextAttentionDAT) and m.k_proj_hd is not None:
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
        mm_token_type_ids=None,
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
            mm_token_type_ids=mm_token_type_ids,
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

        # hd_embeds may be a list (from vision model) — concatenate if needed
        if isinstance(hd_embeds, list):
            hd_embeds = torch.cat(hd_embeds, dim=0)

        for thw in image_grid_thw_hd:
            t = thw[0].item()
            h_merged = thw[1].item() // spatial_merge
            w_merged = thw[2].item() // spatial_merge
            n_patches = t * h_merged * w_merged

            feat = hd_embeds[offset:offset + n_patches]
            if t > 1:
                logger.warning(
                    f"HD features: video with t={t} detected, using first frame only."
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
        """Fused ViT call for LR + HD in one kernel."""
        spatial_merge = self.config.vision_config.spatial_merge_size
        vit_dtype = self.model.visual.dtype
        vit_trainable = any(p.requires_grad for p in self.model.visual.parameters())
        need_split = self.training and torch.is_grad_enabled() and vit_trainable

        if need_split:
            with torch.no_grad():
                hd_output = self.model.visual(
                    pixel_values_hd.to(vit_dtype),
                    grid_thw=image_grid_thw_hd,
                    return_dict=True,
                )
                hd_embeds = hd_output.pooler_output
            lr_output = self.model.visual(
                pixel_values.to(vit_dtype),
                grid_thw=image_grid_thw,
                return_dict=True,
            )
            lr_embeds = lr_output.pooler_output
        else:
            pv_combined = torch.cat([
                pixel_values.to(vit_dtype),
                pixel_values_hd.to(vit_dtype),
            ], dim=0)
            thw_combined = torch.cat([image_grid_thw, image_grid_thw_hd], dim=0)

            combined_output = self.model.visual(
                pv_combined, grid_thw=thw_combined, return_dict=True,
            )
            combined_embeds = combined_output.pooler_output

            if isinstance(combined_embeds, list):
                combined_embeds = torch.cat(combined_embeds, dim=0)

            split_sizes = (thw_combined.prod(-1) // spatial_merge ** 2).tolist()
            splits = torch.split(combined_embeds, [int(s) for s in split_sizes])
            n_lr = len(image_grid_thw)
            lr_embeds = torch.cat(list(splits[:n_lr]), dim=0)
            hd_embeds = torch.cat(list(splits[n_lr:]), dim=0)

        if isinstance(hd_embeds, list):
            hd_embeds = torch.cat(hd_embeds, dim=0)
        if isinstance(lr_embeds, list):
            lr_embeds = torch.cat(lr_embeds, dim=0)

        image_hd_features = self._parse_hd_embeds(hd_embeds, image_grid_thw_hd)

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
        """Shared-ViT path: one HD ViT call, LR tokens are pooled from HD features."""
        spatial_merge = self.config.vision_config.spatial_merge_size
        vit_dtype = self.model.visual.dtype

        with torch.no_grad():
            hd_output = self.model.visual(
                pixel_values_hd.to(vit_dtype),
                grid_thw=image_grid_thw_hd,
                return_dict=True,
            )
            hd_embeds = hd_output.pooler_output
            if isinstance(hd_embeds, list):
                hd_embeds = torch.cat(hd_embeds, dim=0)
            image_hd_features = self._parse_hd_embeds(hd_embeds, image_grid_thw_hd)

            lr_feats: List[torch.Tensor] = []
            for i, hd_feat in enumerate(image_hd_features):
                thw_lr = image_grid_thw[i]
                lr_h = int(thw_lr[1].item()) // spatial_merge
                lr_w = int(thw_lr[2].item()) // spatial_merge

                hd_chw = hd_feat.permute(2, 0, 1).unsqueeze(0)
                pool_dtype = hd_chw.dtype
                pooled = F.adaptive_avg_pool2d(
                    hd_chw.float() if pool_dtype in (torch.bfloat16, torch.float16) else hd_chw,
                    (lr_h, lr_w),
                ).to(pool_dtype)
                lr_feats.append(pooled.squeeze(0).permute(1, 2, 0).reshape(lr_h * lr_w, -1))

            lr_embeds = torch.cat(lr_feats, dim=0)

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
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        mm_token_type_ids: Optional[torch.IntTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        # DAT-specific
        pixel_values_hd: Optional[torch.Tensor] = None,
        image_grid_thw_hd: Optional[torch.LongTensor] = None,
        image_hd_features: Optional[List[torch.Tensor]] = None,
        image_range_list: Optional[List[List]] = None,
        **kwargs,
    ) -> Union[Tuple, Qwen3VLCausalLMOutputWithPast]:
        """Forward pass with DAT HD feature injection.

        In addition to standard Qwen3-VL arguments, accepts:
            pixel_values_hd: High-resolution pixel values for HD features
            image_grid_thw_hd: Grid dimensions for HD images
            image_hd_features: Pre-computed HD features (alternative to pixel_values_hd)
            image_range_list: Pre-computed image/answer ranges (optional, auto-computed)
        """
        # === Step 1: ViT feature extraction ===
        _pixel_values_for_model = pixel_values
        _image_grid_thw_for_model = image_grid_thw

        _use_shared_vit = self.config.dat_extra_args.get('use_shared_vit', False)
        _use_fused_vit = self.config.dat_extra_args.get('use_fused_vit', False)

        _have_full_vit_inputs = (
            image_hd_features is None
            and pixel_values_hd is not None and image_grid_thw_hd is not None
            and image_grid_thw is not None
            and input_ids is not None and inputs_embeds is None
        )

        if _use_shared_vit and _have_full_vit_inputs:
            inputs_embeds, image_hd_features = self._shared_vit_forward(
                pixel_values_hd, image_grid_thw_hd,
                image_grid_thw, input_ids,
            )
            _pixel_values_for_model = None
            _image_grid_thw_for_model = None
        elif (_use_fused_vit
                and _have_full_vit_inputs
                and pixel_values is not None):
            inputs_embeds, image_hd_features = self._fused_vit_forward(
                pixel_values, image_grid_thw,
                pixel_values_hd, image_grid_thw_hd,
                input_ids,
            )
            _pixel_values_for_model = None
            _image_grid_thw_for_model = None
        elif image_hd_features is None and pixel_values_hd is not None and image_grid_thw_hd is not None:
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
        # Qwen3-VL uses [4, B, seq] position IDs: [text, T, H, W].
        # DAT needs the [3, B, seq] T/H/W portion for HD position construction.
        mrope_position_ids = None
        if image_hd_features is not None and position_ids is None and input_ids is not None:
            # Build mm_token_type_ids if not provided
            if mm_token_type_ids is None:
                mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)
                image_token_id = getattr(self.config, 'image_token_id', None)
                if image_token_id is not None:
                    mm_token_type_ids[input_ids == image_token_id] = 1
                video_token_id = getattr(self.config, 'video_token_id', None)
                if video_token_id is not None:
                    mm_token_type_ids[input_ids == video_token_id] = 2

            # Use the model's get_rope_index to compute 4D position IDs
            position_ids_4d, rope_deltas = self.model.get_rope_index(
                input_ids,
                mm_token_type_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
            )
            # position_ids_4d: [4, B, seq] — [text, T, H, W] OR [3, B, seq] — [T, H, W]
            if position_ids_4d.shape[0] == 4:
                mrope_position_ids = position_ids_4d[1:]  # [3, B, seq]
                position_ids = position_ids_4d
            else:
                mrope_position_ids = position_ids_4d  # already [3, B, seq]
                position_ids = position_ids_4d
        elif position_ids is not None:
            if position_ids.shape[0] == 4:
                mrope_position_ids = position_ids[1:]
            elif position_ids.shape[0] == 3:
                mrope_position_ids = position_ids

        # === Step 4: Call base model with DAT kwargs ===
        # DAT kwargs flow through: Model → TextModel → DecoderLayer → Attention
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
            mm_token_type_ids=mm_token_type_ids,
            # DAT kwargs
            image_hd_features=image_hd_features,
            image_range_list=image_range_list,
            mrope_position_ids=mrope_position_ids,
            **kwargs,
        )

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

        return Qwen3VLCausalLMOutputWithPast(
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

DAT_KEYS_MATCH = [
    'conv_lr_dw', 'ln_1', 'conv_lr_proj', 'proj_intention',
    'ln_2', 'conv_off_proj', 'k_proj_hd', 'v_proj_hd',
    'hd_gate', 'hd_input_layernorm',
]


def convert_qwen3_vl_to_dat(base_model_or_path, dat_extra_args, torch_dtype=None):
    """Convert a pretrained Qwen3-VL to Qwen3VL-DAT.

    Args:
        base_model_or_path: path to pretrained Qwen3-VL checkpoint
        dat_extra_args: dict with DAT parameters
        torch_dtype: optional torch dtype for loading

    Returns:
        Qwen3VLDATForConditionalGeneration with base weights + fresh DAT weights
    """
    if isinstance(base_model_or_path, str):
        base_config = Qwen3VLConfig.from_pretrained(base_model_or_path)
    else:
        base_config = base_model_or_path.config

    dat_config = Qwen3VLDATConfig(**base_config.to_dict())
    dat_config.dat_extra_args = dat_extra_args

    if isinstance(base_model_or_path, str):
        dat_model = Qwen3VLDATForConditionalGeneration.from_pretrained(
            base_model_or_path,
            config=dat_config,
            torch_dtype=torch_dtype,
            ignore_mismatched_sizes=False,
        )
    else:
        base_model_or_path.config = dat_config
        base_model_or_path.__class__ = Qwen3VLDATForConditionalGeneration
        layers_str = dat_extra_args.get('layers', '')
        text_config = dat_config.text_config
        if layers_str:
            assert len(layers_str) == text_config.num_hidden_layers
            for i, lt in enumerate(layers_str):
                if lt == 'D':
                    base_model_or_path.model.language_model.layers[i] = \
                        Qwen3VLTextDecoderLayerDAT(text_config, i, dat_extra_args)
        dat_model = base_model_or_path

    dat_model.init_hd_proj_from_kv()

    # Force fp32 storage for DAT scalar/near-unity params
    n_fixed = 0
    for m in dat_model.modules():
        if isinstance(m, Qwen3VLTextAttentionDAT):
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
        target_layers: 'dat' applies LoRA only to DAT layer QKVO,
            'all' applies LoRA to every decoder layer's QKVO.

    Returns:
        Regex pattern string suitable for LoraConfig(target_modules=...).
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
try:
    from transformers.conversion_mapping import (
        get_checkpoint_conversion_mapping as _get_ckpt_mapping,
        register_checkpoint_conversion_mapping as _register_ckpt_mapping,
    )

    _qwen3vl_mapping = _get_ckpt_mapping("qwen3_vl")
    if _qwen3vl_mapping is not None and _get_ckpt_mapping("qwen3_vl_dat") is None:
        try:
            _register_ckpt_mapping("qwen3_vl_dat", _qwen3vl_mapping, overwrite=False)
        except ValueError:
            pass
except ImportError:
    pass
