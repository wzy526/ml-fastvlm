"""
Qwen3.5-DAT: Dynamic Attention Token extension for Qwen3.5 (natively multimodal).

Extends official Qwen3_5ForConditionalGeneration with the DAT mechanism:
- Offset-based sampling from high-resolution vision features
- Per-layer HD feature injection via modified attention
- Interleaved mRoPE with partial rotary (rotary_dim = head_dim * 0.25)
- QKNorm (RMSNorm on Q and K heads) applied consistently to HD keys
- Gated attention: LSE merge happens BEFORE the sigmoid output gate

Architecture (transformers >= 5.10, hybrid GatedDeltaNet + full attention):
    Qwen3_5DATForConditionalGeneration(Qwen3_5ForConditionalGeneration)
    ├── model: Qwen3_5Model (unmodified — kwargs flow natively)
    │   ├── visual: Qwen3_5VisionModel  (shared for LR & HD, no deepstack)
    │   └── language_model: Qwen3_5TextModel (unmodified)
    │       └── layers: 3/4 'linear_attention' (GatedDeltaNet, untouched)
    │                   1/4 'full_attention'  → replaceable by DAT layers
    └── lm_head: nn.Linear (tied to embed_tokens for small models)

Constraints vs Qwen3-VL DAT:
    - DAT layers MUST sit on 'full_attention' positions (config.text_config.layer_types).
      GatedDeltaNet layers have no KV concept; the two-pass LSE trick only applies
      to softmax attention. HD info reaches linear layers via the residual stream.
    - Gated attention: q_proj emits [Q | gate]; official forward does
      `attn_output * sigmoid(gate)` after attention. DAT merges Pass1/Pass2 via
      LSE first, then applies the gate (gate is Q-side only, so this is exact).
    - Partial rotary: only the first `rotary_dim = head_dim * partial_rotary_factor`
      dims are rotated. `apply_rotary_pos_emb_single` slices by cos width.
    - Token ids differ from Qwen2.5/3-VL (250k vocab): im_start = 248045.

Attention mechanism: Two-pass + LSE merge (GC-safe, shape-static):
    Pass 1: standard causal attention (full sequence, shape = [B, H, Nq, D])
    Pass 2: HD cross-attention per answer segment (Q_ans × K_hd, non-causal)
    Merge:  o* = exp(ℓ₁−ℓ)·o₁ + exp(ℓ₂−ℓ)·o₂,  ℓ = logaddexp(ℓ₁, ℓ₂)
    Gate:   out = merge(o₁, o₂) * sigmoid(gate);  out = o_proj(out)
"""

import contextlib
import logging
import json
import math
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5Model,
    Qwen3_5DecoderLayer,
    Qwen3_5Attention,
    Qwen3_5TextRotaryEmbedding,
    Qwen3_5CausalLMOutputWithPast,
    apply_rotary_pos_emb,
    rotate_half,
)
from transformers.models.qwen3_5.configuration_qwen3_5 import (
    Qwen3_5Config,
    Qwen3_5TextConfig,
)
from transformers.cache_utils import Cache


# ── Utility: repeat_kv for GQA ───────────────────────────────────────────────
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads for grouped-query attention. [B, H_kv, N, D] → [B, H_q, N, D]."""
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


# ── Backend selection: FA4 (CuTe-DSL, sm90+) preferred, FA2 fallback ────────
# DAT_ATTN_BACKEND env: 'auto' (default) picks FA4 when flash_attn.cute is
# importable, exposes return_lse, and the visible GPU is sm90+; 'fa4' forces
# FA4 (raises if unavailable); 'fa2' forces the FA2 path.
# FA4 parity vs FA2 (verified on B200, bf16, H=16 D=256, causal + varlen):
# identical out/LSE layouts — dense out [B,N,H,D] + lse [B,H,N] fp32, varlen
# out [total,H,D] + lse [H,total] — out/grad rel err ~0.2-0.5% (bf16 kernel
# noise), lse ~1e-5. CAUTION: FA4's 4th positional arg is `qv`, so varlen
# args after q/k/v must be passed by keyword. First call JIT-compiles CuTe
# kernels (~2 min, cached on disk under XDG_CACHE_HOME).
import inspect as _inspect
from flash_attn import flash_attn_func as _flash_attn_func
from flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func
import flash_attn as _fa_mod

from ._cutlass_warn_filter import install as _install_cutlass_warn_filter

_install_cutlass_warn_filter()

_FA_HAS_SOFTMAX_LSE = "return_softmax_lse" in _inspect.signature(_flash_attn_func).parameters
_fa_ver = getattr(_fa_mod, "__version__", "unknown")
_lse_api = "return_softmax_lse" if _FA_HAS_SOFTMAX_LSE else "return_attn_probs"

_fa4_func = None
_fa4_varlen_func = None
_backend_req = os.environ.get("DAT_ATTN_BACKEND", "auto").lower()
if _backend_req in ("auto", "fa4"):
    try:
        from flash_attn.cute.interface import flash_attn_func as _fa4_f
        from flash_attn.cute.interface import flash_attn_varlen_func as _fa4_v
        if "return_lse" not in _inspect.signature(_fa4_f).parameters:
            raise ImportError("flash_attn.cute present but lacks return_lse")
        if _backend_req == "auto" and not (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability(0)[0] >= 9
        ):
            raise ImportError("FA4 needs an sm90+ GPU (set DAT_ATTN_BACKEND=fa4 to force)")
        _fa4_func, _fa4_varlen_func = _fa4_f, _fa4_v
    except Exception as _fa4_err:
        if _backend_req == "fa4":
            raise
        print(f"[DAT-LSE/qwen3_5] FA4 unavailable ({_fa4_err}); using FA2")

_USE_FA4 = _fa4_func is not None
if _USE_FA4:
    print(f"[DAT-LSE/qwen3_5] backend=FA4 (flash_attn.cute, return_lse) — FA2 v{_fa_ver} fallback available")
else:
    print(f"[DAT-LSE/qwen3_5] backend=FA2 v{_fa_ver} — {_lse_api} + varlen")


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

    if _USE_FA4:
        out_fa, lse = _fa4_func(q_fa, k_fa, v_fa, causal=causal,
                                return_lse=True)
    elif _FA_HAS_SOFTMAX_LSE:
        out_fa, lse = _flash_attn_func(q_fa, k_fa, v_fa, causal=causal,
                                       return_softmax_lse=True)
    else:
        out_fa, lse, _ = _flash_attn_func(q_fa, k_fa, v_fa, causal=causal,
                                          return_attn_probs=True)
    # Detach LSE at the backend boundary. FA2's autograd silently DROPS dlse
    # in backward (extra output grads are ignored), so all historical DAT
    # training effectively treated the LSE-merge weights as constants w.r.t.
    # q/k/v. FA4 asserts instead of dropping ("SM100 backward with
    # head_dim=256 does not support dlse"), so make the historical semantics
    # explicit. Trainable terms added downstream (e.g. hd_gate via
    # logsigmoid) still get their gradients through the merge.
    return out_fa, lse.detach()


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

    if _USE_FA4:
        # keyword args mandatory: FA4's 4th positional parameter is `qv`
        out_packed, lse_packed = _fa4_varlen_func(
            q_packed, k_packed, v_packed,
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=max(nq_lens), max_seqlen_k=max(nk_lens),
            causal=False, return_lse=True,
        )
    elif _FA_HAS_SOFTMAX_LSE:
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

    # Same rationale as _dat_attn_with_lse: FA2 drops dlse, FA4 asserts on it.
    lse_packed = lse_packed.detach()

    out_list = []
    lse_list = []
    q_offset = 0
    for i in range(n_segs):
        nq = nq_lens[i]
        out_list.append(out_packed[q_offset:q_offset + nq].unsqueeze(0))
        lse_list.append(lse_packed[:, q_offset:q_offset + nq].unsqueeze(0))
        q_offset += nq

    return out_list, lse_list


# ── Exact merged-gradient two-pass attention (raw flash backward) ───────────
# DAT_EXACT_MERGE_GRAD=1 (default) enables ring-attention-style EXACT
# gradients through the LSE merge; =0 falls back to the legacy detached-LSE
# semantics (identical to FA2 history) everywhere.
#
# Backend-agnostic: the merged-stats trick only needs a "raw" flash backward
# that accepts (q, k, v, out, dout, lse) — both FA4 (flash_attn.cute
# _flash_attn_bwd) and FA2 (flash_attn_interface._flash_attn_backward /
# _flash_attn_varlen_backward) expose one. FA2 is the common case on the
# training pods (no nvidia-cutlass-dsl), so it is a first-class path here,
# not a fallback that silently degrades to detached-LSE gradients.
_EXACT_MERGE_GRAD = os.environ.get("DAT_EXACT_MERGE_GRAD", "1") == "1"
_fa4_raw_bwd = None
_fa2_raw_bwd = None
_fa2_raw_varlen_bwd = None
if _USE_FA4:
    try:
        from flash_attn.cute.interface import _flash_attn_bwd as _fa4_raw_bwd
    except Exception as _raw_err:  # pragma: no cover
        print(f"[DAT-LSE/qwen3_5] FA4 raw bwd unavailable ({_raw_err})")
if _fa4_raw_bwd is None:
    try:
        from flash_attn.flash_attn_interface import (
            _flash_attn_backward as _fa2_raw_bwd,
            _flash_attn_varlen_backward as _fa2_raw_varlen_bwd,
        )
    except Exception as _raw_err:  # pragma: no cover
        print(f"[DAT-LSE/qwen3_5] FA2 raw bwd unavailable ({_raw_err})")

_EXACT_BWD_BACKEND: Optional[str] = (
    "fa4" if _fa4_raw_bwd is not None
    else "fa2" if _fa2_raw_bwd is not None
    else None
)
_EXACT_MERGE_AVAILABLE = _EXACT_MERGE_GRAD and _EXACT_BWD_BACKEND is not None
if _EXACT_MERGE_AVAILABLE:
    print(f"[DAT-LSE/qwen3_5] exact merge gradients ENABLED via {_EXACT_BWD_BACKEND.upper()} "
          f"raw backward (merged-stats, requires hd_gate=None)")
elif _EXACT_MERGE_GRAD:
    print("[DAT-LSE/qwen3_5] WARNING: DAT_EXACT_MERGE_GRAD=1 but no raw flash backward "
          "is importable — training will REFUSE to run the two-pass merge (set "
          "DAT_EXACT_MERGE_GRAD=0 to knowingly accept legacy detached-LSE gradients)")
else:
    print("[DAT-LSE/qwen3_5] exact merge gradients DISABLED by DAT_EXACT_MERGE_GRAD=0 "
          "(legacy detached-LSE semantics)")


def _raw_attn_fwd(q_t, k_t, v_t, causal: bool):
    """Dense attention fwd in flash layout. q/k/v: [B, N, H, D] -> (out [B,N,H,D], lse [B,H,N] fp32)."""
    if _USE_FA4:
        return _fa4_func(q_t, k_t, v_t, causal=causal, return_lse=True)
    if _FA_HAS_SOFTMAX_LSE:
        return _flash_attn_func(q_t, k_t, v_t, causal=causal, return_softmax_lse=True)
    out, lse, _ = _flash_attn_func(q_t, k_t, v_t, causal=causal, return_attn_probs=True)
    return out, lse


def _raw_attn_varlen_fwd(q_p, k_p, v_p, cu_q, cu_k, max_q, max_k, causal: bool):
    """Varlen attention fwd. q/k/v packed [total, H, D] -> (out [total,H,D], lse [H,total] fp32)."""
    if _USE_FA4:
        return _fa4_varlen_func(
            q_p, k_p, v_p,
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=max_q, max_seqlen_k=max_k,
            causal=causal, return_lse=True,
        )
    if _FA_HAS_SOFTMAX_LSE:
        return _flash_attn_varlen_func(
            q_p, k_p, v_p, cu_q, cu_k, max_q, max_k,
            causal=causal, return_softmax_lse=True,
        )
    out, lse, _ = _flash_attn_varlen_func(
        q_p, k_p, v_p, cu_q, cu_k, max_q, max_k,
        causal=causal, return_attn_probs=True,
    )
    return out, lse


def _raw_attn_bwd(q_t, k_t, v_t, out, dout, lse, causal: bool):
    """Dense raw flash backward with caller-supplied (out, lse) row stats.

    Feeding MERGED (out*, lse*) here makes the kernel reconstruct the
    union-softmax probabilities p = exp(s - lse*) and use D = g·out*, which is
    exactly the merged-attention gradient (ring-attention backward).
    Returns (dq, dk, dv) in flash layout, same dtype as q/k/v.
    """
    if _EXACT_BWD_BACKEND == "fa4":
        return _fa4_raw_bwd(q_t, k_t, v_t, out, dout, lse, causal=causal)
    # FA2: raw kernel needs preallocated dq/dk/dv and an explicit softmax scale
    # (the public API defaults it to head_dim**-0.5; the raw op does not).
    dq = torch.empty_like(q_t)
    dk = torch.empty_like(k_t)
    dv = torch.empty_like(v_t)
    _fa2_raw_bwd(
        dout.contiguous(), q_t, k_t, v_t, out.contiguous(), lse.contiguous(),
        dq, dk, dv,
        0.0,                          # dropout_p
        q_t.shape[-1] ** -0.5,        # softmax_scale
        causal,
        -1, -1,                       # window_size_left / right
        0.0,                          # softcap
        None,                         # alibi_slopes
        False,                        # deterministic
        None,                         # rng_state
    )
    return dq, dk, dv


def _raw_attn_varlen_bwd(q_p, k_p, v_p, out, dout, lse, cu_q, cu_k, max_q, max_k, causal: bool):
    """Varlen raw flash backward with caller-supplied (out, lse). lse: [H, total] fp32."""
    if _EXACT_BWD_BACKEND == "fa4":
        return _fa4_raw_bwd(
            q_p, k_p, v_p, out, dout, lse,
            causal=causal,
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=max_q, max_seqlen_k=max_k,
        )
    dq = torch.empty_like(q_p)
    dk = torch.empty_like(k_p)
    dv = torch.empty_like(v_p)
    _fa2_raw_varlen_bwd(
        dout.contiguous(), q_p, k_p, v_p, out.contiguous(), lse.contiguous(),
        dq, dk, dv,
        cu_q, cu_k, int(max_q), int(max_k),
        0.0,                          # dropout_p
        q_p.shape[-1] ** -0.5,        # softmax_scale
        causal,
        -1, -1,                       # window_size_left / right
        0.0,                          # softcap
        None,                         # alibi_slopes
        False,                        # deterministic
        None,                         # rng_state
        False,                        # zero_tensors
    )
    return dq, dk, dv


class _TwoPassMergedAttnFn(torch.autograd.Function):
    """Two-pass attention + LSE merge with EXACT gradients.

    Forward math is identical to the legacy path (_dat_attn_with_lse +
    _dat_cross_attn_varlen + _merge_two_pass_lse with hd_gate=None). The
    difference is the backward: the legacy path detaches LSE, so autograd
    treats the merge weights w_i = exp(lse_i - lse*) as constants (FA2's
    historical semantics — FA2 silently drops dlse). Here each pass's raw
    flash backward is instead fed the MERGED per-row stats (out*, lse*): the
    kernel reconstructs p = exp(s - lse*) — the union-softmax probabilities —
    and yields ds = p * (g·v - g·out*), exactly the gradient of a single
    attention over the concatenated KV set. This is the standard ring-
    attention / context-parallel merged backward. No dlse is involved, so
    FA4's "SM100 backward with head_dim=256 does not support dlse" assert is
    never hit, and FA2's silent dlse drop is irrelevant. Works on either
    backend (see _raw_attn_bwd / _raw_attn_varlen_bwd).

    Constraints:
      - segments must be row-disjoint (guaranteed by construction: answer /
        question / lr-inject spans never overlap);
      - hd_gate must be None (a trainable gate on lse2 would need an explicit
        dlse2 term); callers fall back to the legacy path otherwise.
    """

    @staticmethod
    def forward(ctx, q, k, v, q2p, k2p, v2p, cu_q, cu_k, max_q, max_k, seg_meta,
                stats_sink=None, lse2_bias=0.0):
        # q/k/v: [B, H, N, D] (GQA already repeated); q2p/k2p/v2p: packed
        # [total, H, D]; seg_meta: tuple of (b_idx, row_start, n_rows);
        # stats_sink: attention module to stash w2 (HD attention-mass share)
        # diagnostics on during training (harvested by DATMonitor).
        # lse2_bias: constant added to every pass-2 logit (hd_lse_bias). It
        # leaves the within-pass-2 softmax (out2) unchanged and only shifts
        # the merge weights; the backward feeds pass 2 `lse* - bias` so the
        # kernel reconstructs exp(s + bias - lse*) — still exact.
        q_t = q.transpose(1, 2).contiguous()
        k_t = k.transpose(1, 2).contiguous()
        v_t = v.transpose(1, 2).contiguous()

        q2p = q2p.contiguous()
        k2p = k2p.contiguous()
        v2p = v2p.contiguous()

        out1, lse1 = _raw_attn_fwd(q_t, k_t, v_t, causal=True)
        out2p, lse2p = _raw_attn_varlen_fwd(
            q2p, k2p, v2p, cu_q, cu_k, max_q, max_k, causal=False,
        )
        lse1 = lse1.float()
        lse2p = lse2p.float()

        # Row-disjoint merge; also build the per-row merged stats that the
        # backward feeds to both raw kernels.
        out = out1.clone()          # [B, N, H, D]
        lse_c = lse1.clone()        # [B, H, N] fp32
        w2_sum = None
        w2_max = None
        w2_cnt = 0
        qoff = 0
        for (b, s, n) in seg_meta:
            l1 = lse1[b, :, s:s + n]                    # [H, n] fp32
            l2 = lse2p[:, qoff:qoff + n] + lse2_bias    # [H, n] fp32
            lm = torch.logaddexp(l1, l2)
            w1 = (l1 - lm).exp().transpose(0, 1).unsqueeze(-1)   # [n, H, 1]
            w2 = (l2 - lm).exp().transpose(0, 1).unsqueeze(-1)
            merged = w1 * out1[b, s:s + n] + w2 * out2p[qoff:qoff + n]
            out[b, s:s + n] = merged.to(out.dtype)
            lse_c[b, :, s:s + n] = lm
            if stats_sink is not None:
                w2_sum = w2.sum() if w2_sum is None else w2_sum + w2.sum()
                w2_max = w2.max() if w2_max is None else torch.maximum(w2_max, w2.max())
                w2_cnt += w2.numel()
            qoff += n

        if stats_sink is not None and w2_cnt > 0:
            stats_sink._dat_hd_w2_stats = (
                (w2_sum / w2_cnt).item(),
                w2_max.item(),
            )

        ctx.save_for_backward(q_t, k_t, v_t, q2p, k2p, v2p, out, lse_c, cu_q, cu_k)
        ctx.seg_meta = seg_meta
        ctx.max_q = max_q
        ctx.max_k = max_k
        ctx.lse2_bias = float(lse2_bias)
        return out

    @staticmethod
    def backward(ctx, g):
        q_t, k_t, v_t, q2p, k2p, v2p, out, lse_c, cu_q, cu_k = ctx.saved_tensors
        seg_meta = ctx.seg_meta
        g = g.contiguous()

        # Pass 1 backward with merged stats. At merged rows out/lse_c hold
        # (out*, lse*): the kernel's reconstructed p becomes the union-softmax
        # probability restricted to sequence keys, and its D-term uses g·out*.
        # Non-merged rows carry their own (out1, lse1) — plain exact backward.
        dq_t, dk_t, dv_t = _raw_attn_bwd(q_t, k_t, v_t, out, g, lse_c, causal=True)

        # Pass 2 backward, same merged stats gathered per segment row.
        out2s = torch.cat([out[b, s:s + n] for (b, s, n) in seg_meta], dim=0).contiguous()
        g2s = torch.cat([g[b, s:s + n] for (b, s, n) in seg_meta], dim=0).contiguous()
        lse2s = torch.cat([lse_c[b, :, s:s + n] for (b, s, n) in seg_meta], dim=1)
        if ctx.lse2_bias != 0.0:
            lse2s = lse2s - ctx.lse2_bias
        lse2s = lse2s.contiguous()
        dq2p, dk2p, dv2p = _raw_attn_varlen_bwd(
            q2p, k2p, v2p, out2s, g2s, lse2s,
            cu_q, cu_k, ctx.max_q, ctx.max_k, causal=False,
        )

        return (dq_t.transpose(1, 2), dk_t.transpose(1, 2), dv_t.transpose(1, 2),
                dq2p, dk2p, dv2p, None, None, None, None, None, None, None)


logger = logging.getLogger(__name__)

# Qwen3.5 special token IDs (250k vocab — differs from Qwen2/2.5/3-VL!)
IM_START_TOKEN_ID = 248045   # <|im_start|>


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

class _GradScaleFn(torch.autograd.Function):
    """Identity forward; backward multiplies the gradient by `scale`."""

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g * ctx.scale, None


def _grad_scale(x, scale):
    """x with its upstream gradient scaled by `scale` (1 = no-op, 0 = detach
    that keeps the tensor in the graph). Used at the trunk inputs of the DAT
    offset head so the offset-supervision pull trains the head without
    rewriting the LLM's image features (0916 offsup: HD-off V* 55.5 -> 50.3,
    DocVQA 69 -> 36 when the pull reached the trunk at full strength)."""
    if scale == 1.0 or not torch.is_grad_enabled() or not x.requires_grad:
        return x
    if scale == 0.0:
        return x.detach()
    return _GradScaleFn.apply(x, float(scale))


class _FP32WeightRMSNorm(nn.Module):
    """Standard `w * rmsnorm(x)` with fp32 weight storage.

    Deliberately NOT a subclass of Qwen3_5RMSNorm: the official class uses the
    `(1 + weight)` parameterization with zero-init, and
    Qwen3_5PreTrainedModel._init_weights zero-fills any Qwen3_5RMSNorm instance.
    Subclassing it would silently zero this norm's output for missing keys.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.variance_epsilon = eps

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
# RoPE Helper for Single Tensor (partial-rotary aware)
# ============================================================================

def apply_rotary_pos_emb_single(x, cos, sin, unsqueeze_dim=1):
    """Apply RoPE to a single tensor (Q or K separately), partial-rotary aware.

    Qwen3.5 uses partial_rotary_factor=0.25: cos/sin cover only
    rotary_dim = head_dim // 4 dims; the rest pass through unrotated.
    Slicing by cos width makes this also correct for full-rotary models.

    Args:
        x: [batch, heads, seq_len, head_dim]
        cos: [batch, seq_len, rotary_dim] or broadcastable
        sin: [batch, seq_len, rotary_dim] or broadcastable
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    x_embed = (x_rot * cos) + (rotate_half(x_rot) * sin)
    if x_pass.shape[-1] == 0:
        return x_embed
    return torch.cat([x_embed, x_pass], dim=-1)


# ============================================================================
# Config
# ============================================================================

class Qwen3_5DATConfig(Qwen3_5Config):
    """Qwen3.5 config extended with DAT parameters."""
    model_type = "qwen3_5_dat"

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
            'intention_inject': 'gate',   # 'gate' (legacy) | 'film' (post-norm)
            'question_inject': 'none',    # 'none' (legacy) | 'xattn' (question->cell cross-attn residual)
            'qr_heads': 4,                # xattn: number of attention heads
            'qr_layerscale_init': 1e-2,   # xattn: initial per-channel LayerScale on the residual
            'off_range': 0.0,             # 0 = legacy clamp; >0 = off_range*tanh
            'off_penalty': 0.0,           # >0 = honest clamp + out-of-range pull-back
            # Offset supervision (training-only, bbox samples): >0 turns the
            # dat_force_window of a sample into a regression TARGET for the
            # learned sampling grid instead of replacing it. Equivalent to
            #   loss += off_sup_weight * mean(huber_delta(ref + off - target))
            # summed over DAT layers; gives conv_off_proj / intention / readout
            # the first direct, non-local gradient toward "where to look".
            'off_sup_weight': 0.0,
            'off_sup_delta': 0.1,         # huber transition, in [-1, 1] grid units
            # Gradient scale on the offset head's trunk inputs (LR image hidden
            # states, intention token, question span). 1 = legacy: whatever
            # trains the head also trains the LLM; 0 = the head decodes from
            # the trunk's features as they are. Applies to LM and supervision
            # gradients alike (the LM gradient through the head never taught
            # the offsets anything, so nothing is lost).
            'off_head_trunk_grad': 1.0,
            # Global localisation term in the offset head. The per-point offset
            # head is local (3x3 dw conv + 1x1): a point far from the target has
            # no information about which way to move, so supervised offsets only
            # ever travel part of the way (0916/0917: off_sup_dist plateaus at
            # ~0.5). With this on, a 1x1 conv on the intention-gated cell
            # features gives a relevance logit per LR cell; its softmax is a
            # spatial distribution whose mean translates the whole grid and
            # whose per-axis std (relative to the uniform grid's) shrinks it:
            #   x = centroid + scale * reference + local_offset
            # Zero-init => uniform relevance => centroid 0, scale 1 => exactly
            # the legacy grid at load time (warm-start safe).
            'use_global_offset': False,
            'glob_min_scale': 0.1,        # floor on the per-axis grid scale
            # LR dropout (training-only regulariser): with prob lr_drop_prob per
            # sample, replace lr_drop_ratio of that sample's LR image-token
            # embeddings by the sample's mean LR embedding (content-free, scale
            # matched). Positions are kept, so layout/offsets survive while fine
            # content can only be recovered from the HD path.
            'lr_drop_prob': 0.0,
            'lr_drop_ratio': 0.75,
            'hd_gate_init': None,
            'hd_gate_freeze': False,
            'use_fused_vit': False,
            'use_shared_vit': False,
            'use_spatial_attn_guide': True,
            'image_hd_for_question': False,
            'hd_early_exit_k': 0,      # HD ViT early exit: run only the first k vision blocks
                                       # (0 = off, full depth). Only affects the separate HD path
                                       # (_generate_hd_features); ignored by use_fused_vit /
                                       # use_shared_vit. Unlike Qwen2.5-VL (32 blocks, window+full
                                       # mix), the Qwen3.5 ViT has `depth` (4B: 24) uniform
                                       # full-attention blocks, so any 0 < k < depth is valid and
                                       # HD ViT runtime scales ~k/depth. Valid because DAT consumes
                                       # HD features only through the from-scratch k_proj_hd /
                                       # v_proj_hd adapters — but the adapters bind to whatever
                                       # depth they were trained with, so train/infer k must match.
        }


# ============================================================================
# Helpers
# ============================================================================

def build_dat_layers_string(text_config, mode: str = "auto") -> str:
    """Build a DAT layer string anchored to the hybrid layer_types.

    Qwen3.5 constraint: DAT ('D') may only replace 'full_attention' layers.

    Args:
        text_config: Qwen3_5TextConfig with .layer_types
        mode: "auto"  — every full_attention layer becomes 'D'
              "auto2" — every 2nd full_attention layer becomes 'D'
              "autoN" — every Nth full_attention layer becomes 'D'

    Returns:
        Layer string, e.g. 'LLLDLLLDLLLD...' for 2B (interval 4).
    """
    layer_types = list(getattr(text_config, 'layer_types', []))
    if not layer_types:
        raise ValueError("text_config has no layer_types; not a hybrid Qwen3.5 config?")
    stride = 1
    if mode.startswith("auto") and len(mode) > 4:
        stride = int(mode[4:])
    chars = []
    full_seen = 0
    for lt in layer_types:
        if lt == "full_attention":
            chars.append('D' if full_seen % stride == 0 else 'L')
            full_seen += 1
        else:
            chars.append('L')
    return ''.join(chars)


def compute_image_range_list(input_ids, labels, image_token_id,
                              im_start_token_id=IM_START_TOKEN_ID,
                              image_grid_thw=None, spatial_merge_size=2):
    """Compute image_range_list from Qwen3.5-format inputs.

    Scans input_ids for image token regions and labels for answer regions.
    For each answer range, dynamically locates the preceding <|im_start|>
    token as the intention_idx.

    Returns:
        List per batch of:
            [[(lr_start, lr_end, lr_h, lr_w), ...per image...],
             [ans1_start, ans1_end, intention_idx, q_start, q_end], ...]

    Each answer range carries its question span (q_start, q_end): the tokens
    from the end of the previous segment (image / prior answer) up to where the
    turn's answer begins (training) or the assistant <|im_start|> (inference).
    Consumed by the 'xattn' question-conditioned offset readout.
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
                # Question span bookkeeping: starts after the (last) image, and
                # after each answer for subsequent multi-turn questions.
                img_end = lr_tuples[-1][1] if lr_tuples else 0
                q_prev_end = img_end
                seg_start = ans_indices[0].item()
                for i in range(1, len(ans_indices)):
                    if ans_indices[i] - ans_indices[i - 1] > 1:
                        seg_end = ans_indices[i - 1].item()
                        intention_idx = _find_im_start_backward(ids, seg_start, im_start_token_id)
                        q_start = q_prev_end
                        q_end = seg_start if seg_start >= q_prev_end else q_prev_end
                        ranges.append([seg_start, seg_end, intention_idx, q_start, q_end])
                        q_prev_end = seg_end + 1
                        seg_start = ans_indices[i].item()
                intention_idx = _find_im_start_backward(ids, seg_start, im_start_token_id)
                q_start = q_prev_end
                q_end = seg_start if seg_start >= q_prev_end else q_prev_end
                ranges.append([seg_start, ans_indices[-1].item(), intention_idx, q_start, q_end])
        else:
            seq_len = ids.shape[0]
            intention_idx = _find_im_start_backward(ids, seq_len, im_start_token_id)
            img_end = lr_tuples[-1][1] if lr_tuples else 0
            q_start = img_end
            q_end = (intention_idx + 1) if (intention_idx is not None and intention_idx + 1 > img_end) else seq_len
            ranges.append([seq_len, -1, intention_idx, q_start, q_end])

        result.append(ranges)

    return result


# ============================================================================
# DAT Attention (Qwen3.5 gated attention + partial interleaved mRoPE)
# ============================================================================

# conv_off_proj init. Not zero: a zero readout sends zero gradient to conv_lr_dw /
# conv_lr_proj / proj_intention, and on the 0901 / op10 / or10 4B ckpts those never
# left their random init. 0.005 puts the initial offsets at ~0.3 grid pitch
# (ln_2 output ~N(0,1) over 128 channels -> std(offset) ~= 6.8 * std).
OFF_PROJ_INIT_STD = 0.005


class QuestionReadout(nn.Module):
    """A second LR readout that cross-attends grid cells to the question span.

    The offset head reads a pooled LR grid (``embed_lr``: [off_grps, C, gs, gs]).
    The only spatially-resolved, content-addressed path from the question to the
    per-cell sampling offsets is this module: each grid cell is a *query* that
    reads content from the variable-length question hidden states (*key/value*),
    and the readout is added back to the cell feature as a small, per-channel
    LayerScale-gated residual (CaiT-style)::

        off_guide = embed_lr + layerscale ⊙ o_proj( softmax(Q Kᵀ) V )

    Because the fusion is a residual into the feature the offset head already
    consumes (not an extra concatenated column that a fixed 1x1 conv must read
    out), the question can steer *where* a cell samples: cell (i,j) chooses
    which question tokens to attend to, so different questions move different
    cells to different offsets. ``layerscale`` starts small and ``pos_emb``
    starts at zero, so a checkpoint warm-started from a non-xattn run begins
    ≈ LR-only and escapes the zero-readout deadlock without a discontinuous
    jump, while each channel's residual can grow independently as needed.

    Attention math runs in fp32; parameters stay in the module dtype (the small
    residual then passes through the fp32 ``conv_off_proj`` downstream).
    """

    def __init__(self, cell_dim, hidden_size, grid_size, num_heads=4, layerscale_init=1e-2):
        super().__init__()
        d = cell_dim
        if num_heads <= 0 or d % num_heads != 0:
            num_heads = 1
        self.num_heads = num_heads
        self.head_dim = d // num_heads
        self.cell_dim = d
        self.grid_size = grid_size
        self.q_ln = nn.LayerNorm(hidden_size)
        self.c_ln = nn.LayerNorm(d)
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(hidden_size, d, bias=False)
        self.v_proj = nn.Linear(hidden_size, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.pos_emb = nn.Parameter(torch.zeros(d, grid_size, grid_size))
        # CaiT LayerScale: per-channel diagonal gate on the residual branch.
        self.layerscale = nn.Parameter(torch.full((d,), float(layerscale_init)))
        self._layerscale_init = float(layerscale_init)

    @torch.no_grad()
    def reset_parameters(self):
        """Full, self-contained (re)initialization of EVERY parameter.

        Called from both DAT init paths (`_init_dat_weights` at construction and
        the monkey-patched `_dat_init_weights` that `from_pretrained` runs for
        missing keys). Crucial for the meta-device from_pretrained flow: there
        the constructor's `torch.full`/`torch.zeros` values are NEVER
        materialized, and the base `_init_weights` covers only nn.Linear /
        RMSNorm — so the bare `pos_emb`/`layerscale` Parameters and the two
        nn.LayerNorm modules would otherwise stay as uninitialized garbage
        (observed: layerscale ~1e37 -> off_guide NaN on the first forward)."""
        for lin in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            nn.init.xavier_uniform_(lin.weight)
            if lin.bias is not None:
                nn.init.zeros_(lin.bias)
        for ln in (self.q_ln, self.c_ln):
            nn.init.ones_(ln.weight)
            nn.init.zeros_(ln.bias)
        nn.init.zeros_(self.pos_emb)
        # Small non-zero per-channel gate: question residual starts small
        # (warm-start ≈ LR-only) but each channel is independently trainable.
        self.layerscale.data.fill_(self._layerscale_init)

    def forward(self, cells, q_hidden_list, off_grps):
        """cells: [(Lp*G), C, gs, gs]; q_hidden_list: len-Lp list of [Lq, hidden]
        (or empty tensors for turns with no question span). Returns the residual
        layerscale ⊙ readout, shape [(Lp*G), C, gs, gs]."""
        LpG, C, gs, _ = cells.shape
        G = off_grps
        Lp = LpG // G
        H, hd, N = self.num_heads, self.head_dim, gs * gs

        # Env-gated per-stage NaN probe (DAT_QR_PROBE=1). Zero cost when unset.
        # First healthy call prints every stage (baseline); later calls print only
        # a stage that is already non-finite -> pinpoints the exact op on real data.
        _qrp = os.environ.get("DAT_QR_PROBE")
        _qrp_first = bool(_qrp) and getattr(self, "_qrp_n", 0) == 0

        def _p(tag, t):
            if not _qrp:
                return
            tf = t.detach().float()
            nan = bool(torch.isnan(tf).any())
            inf = bool(torch.isinf(tf).any())
            if nan or inf or _qrp_first:
                amax = float(tf.abs().max()) if tf.numel() else 0.0
                print(f"[QRP {'BAD' if (nan or inf) else 'ok '}] "
                      f"{tag:12s} shape={tuple(t.shape)} "
                      f"dt={str(t.dtype).replace('torch.', '')} "
                      f"nan={nan} inf={inf} absmax={amax:.4g}", flush=True)

        x = cells + self.pos_emb.to(cells.dtype).unsqueeze(0)     # [(Lp*G), C, gs, gs]
        xt = self.c_ln(x.flatten(2).transpose(1, 2))              # [(Lp*G), N, C]
        q = self.q_proj(xt).float()                               # [(Lp*G), N, C]
        _p("cells", cells); _p("xt(c_ln)", xt); _p("q", q)

        delta = cells.new_zeros(LpG, N, C)
        for l in range(Lp):
            Hq = q_hidden_list[l] if l < len(q_hidden_list) else None
            if Hq is None or Hq.shape[0] == 0:
                continue
            Hn = self.q_ln(Hq)
            k = self.k_proj(Hn).float()                           # [Lq, C]
            v = self.v_proj(Hn).float()                           # [Lq, C]
            Lq = k.shape[0]
            sl = slice(l * G, (l + 1) * G)
            qh = q[sl].reshape(G, N, H, hd).permute(0, 2, 1, 3)   # [G, H, N, hd]
            kh = k.reshape(Lq, H, hd).permute(1, 0, 2) * (hd ** -0.5)   # [H, Lq, hd]
            vh = v.reshape(Lq, H, hd).permute(1, 0, 2)            # [H, Lq, hd]
            scores = torch.einsum('ghnd,hld->ghnl', qh, kh)
            attn = F.softmax(scores, dim=3)
            oh = torch.einsum('ghnl,hld->ghnd', attn, vh)         # [G, H, N, hd]
            o = oh.permute(0, 2, 1, 3).reshape(G, N, C)           # [G, N, C]
            delta[sl] = self.o_proj(o.to(xt.dtype))
            _p(f"Hq[{l}]", Hq); _p(f"Hn[{l}]", Hn); _p(f"k[{l}]", k)
            _p(f"v[{l}]", v); _p(f"scores[{l}]", scores)
            _p(f"attn[{l}]", attn); _p(f"delta[{l}]", delta[sl])
        delta = delta.transpose(1, 2).reshape(LpG, C, gs, gs)
        if _qrp:
            self._qrp_n = getattr(self, "_qrp_n", 0) + 1
        return self.layerscale.to(cells.dtype).view(1, C, 1, 1) * delta


class Qwen3_5AttentionDAT(Qwen3_5Attention):
    """
    Core DAT mechanism for Qwen3.5 (two-pass + LSE merge):
    1. Extract LR features from query → generate sampling offsets
    2. Grid sample from HD features → project to KV (key_hd, value_hd)
    3. Pass 1: standard causal attention (full sequence, static shapes)
    4. Pass 2: HD cross-attention per answer segment (Q_ans × K_hd, non-causal)
    5. Merge outputs via LSE trick — mathematically equivalent to joint attention
    6. Apply the sigmoid output gate to the MERGED output (gate is Q-side only)

    New vs Qwen3-VL DAT:
    - Gated attention: q_proj emits [Q | gate]; gate applied post-merge
    - Partial rotary (0.25): apply_rotary_pos_emb_single slices by cos width
    - QKNorm: self.k_norm applied to HD keys (same as standard path)
    """

    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int, dat_extra_args: dict):
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

        # Question conditioning of the offsets: where it is applied decides
        # whether it survives. ln_2 is a channel-wise LayerNorm evaluated per
        # spatial position, so a factor multiplied in BEFORE it is either
        # partially removed (a per-channel gate keeps only its relative
        # profile) or removed outright (a per-position scalar s > 0 cancels
        # exactly, since mu -> s*mu and sigma -> s*sigma). Measured on the
        # 0901 4B ckpt: sampling points move only ~4% of the grid pitch across
        # different questions on the same image.
        #
        # 'film' adds a second, post-norm route that nothing downstream can
        # divide out: per-channel FiLM plus an additive spatial map. All new
        # parameters are zero-init, so a checkpoint trained under 'gate' keeps
        # bit-identical outputs and can be warm-started.
        self.intention_inject = dat.get('intention_inject', 'gate')

        # Question conditioning by a second LR readout that cross-attends the
        # grid cells to the full question span (residual-injected into embed_lr).
        # 'none' = disabled (bit-identical to legacy gate/film ckpts); 'xattn'
        # = enabled. See QuestionReadout. Uses the single-width offset head
        # (conv_off_proj: inter_size -> 2), so it does NOT concat / gate.
        self.question_inject = dat.get('question_inject', 'none')

        # Offset magnitude. With the legacy straight-through clamp nothing
        # penalizes an offset that overshoots [-1,1]: the forward value is
        # capped while the gradient keeps pushing outward. Measured on the 0901
        # 4B ckpt, 53.6% of sampling points end up pinned to the border and
        # 69.8% of the HD attention mass lands on those pinned points, i.e.
        # roughly half the sampling budget reads the image edge. A positive
        # off_range bounds each point to a neighbourhood of its reference
        # instead, which is what deformable-attention work normally does.
        self.off_range = float(dat.get('off_range', 0.0))

        # Alternative to off_range that costs nothing inside the valid region.
        # Any bounded smooth squashing function must have a vanishing derivative
        # far out (otherwise it would be unbounded), so tanh necessarily damps
        # exactly the points that travel furthest -- at the measured max raw
        # offset of 2.754 the gradient is already attenuated ~60x, and it also
        # compresses points that never left the region. Moving the constraint
        # from the forward map into the loss avoids both: an honest clamp keeps
        # the in-region gradient at exactly 1, and a quadratic out-of-range
        # penalty supplies the pull-back that the clamp alone lacks.
        self.off_penalty = float(dat.get('off_penalty', 0.0))
        # See _merge_two_pass_lse; 0 = off (all trained checkpoints).
        self.hd_lse_bias = float(dat.get('hd_lse_bias', 0.0))
        # Probe hook (scripts/probe_whd_qwen35.py --hd_source oracle*): when set
        # to a [Ns, 2] tensor of (x, y) in [-1, 1], the learned offsets are
        # ignored and HD is sampled at exactly these locations; the HD position
        # ids follow them too. None = normal operation.
        self._dat_force_locs = None
        # Training-time teacher forcing (dat_force_window kwarg of the top-level
        # forward): per-sample window [x0, y0, x1, y1] in [0, 1] image fractions
        # (or None). Converted to a grid of forced locations per b_idx inside
        # the attention forward; samples with None keep their learned offsets.
        self._dat_force_batch = None
        # Offset supervision (off_sup_weight > 0): the same per-sample windows,
        # but used as a regression target for the learned grid (see
        # _sample_hd_from_off_guide). _dat_off_target is the current sample's
        # [Ns, 2] target or None; _dat_off_target_batch the per-batch list.
        self.off_sup_weight = float(dat.get('off_sup_weight', 0.0))
        self.off_sup_delta = float(dat.get('off_sup_delta', 0.1))
        self.off_head_trunk_grad = float(dat.get('off_head_trunk_grad', 1.0))
        self._dat_off_target = None
        self._dat_off_target_batch = None

        # Offset prediction
        if self.intention_as_gate or self.question_inject == 'xattn':
            self.ln_2 = _FP32WeightLayerNorm2d(self.inter_size)
            self.conv_off_proj = _FP32WeightConv2d(
                self.inter_size, 2, kernel_size=1, stride=1, padding=0, bias=False,
            )
        else:
            self.ln_2 = _FP32WeightLayerNorm2d(self.inter_size * 2)
            self.conv_off_proj = _FP32WeightConv2d(
                self.inter_size * 2, 2, kernel_size=1, stride=1, padding=0, bias=False,
            )

        if self.intention_inject == 'film' and self.use_intention_branch:
            ln2_ch = self.ln_2.weight.numel()
            self.proj_film = _FP32WeightLinear(self.off_dim, 2 * ln2_ch)
            self.spatial_gain = nn.Parameter(torch.zeros(ln2_ch))
        else:
            self.proj_film = None
            self.spatial_gain = None

        # Global localisation head (see 'use_global_offset' in the defaults):
        # one relevance logit per cell from the same post-ln_2 features that
        # feed conv_off_proj. Zero-init = uniform = identity on the grid.
        self.use_global_offset = bool(dat.get('use_global_offset', False))
        self.glob_min_scale = float(dat.get('glob_min_scale', 0.1))
        if self.use_global_offset:
            self.conv_glob = _FP32WeightConv2d(
                self.ln_2.weight.numel(), 1, kernel_size=1, stride=1, padding=0, bias=True,
            )
        else:
            self.conv_glob = None

        # Question-conditioned LR readout (residual cross-attention).
        if self.question_inject == 'xattn':
            self.q_readout = QuestionReadout(
                self.inter_size, self.hidden_size, self.grid_size,
                num_heads=int(dat.get('qr_heads', 4)),
                layerscale_init=float(dat.get('qr_layerscale_init', 1e-2)),
            )
        else:
            self.q_readout = None

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

        # Rotary embedding for HD positions (partial + interleaved mRoPE)
        self._dat_rotary_emb = Qwen3_5TextRotaryEmbedding(config=config)

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
        if self.spatial_gain is not None and self.spatial_gain.dtype != torch.float32:
            with torch.no_grad():
                self.spatial_gain.data = self.spatial_gain.data.to(torch.float32)
        for sub in (self.conv_lr_dw, self.ln_1, self.conv_lr_proj,
                    self.proj_intention, self.ln_2, self.conv_off_proj,
                    self.proj_film, self.conv_glob):
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
        nn.init.normal_(self.conv_off_proj.weight, std=OFF_PROJ_INIT_STD)
        if isinstance(self.proj_intention, nn.Linear):
            nn.init.xavier_uniform_(self.proj_intention.weight)
            if self.proj_intention.bias is not None:
                nn.init.zeros_(self.proj_intention.bias)
        if self.proj_film is not None:
            # zero => gamma = beta = 0 => post-norm modulation is the identity,
            # so a 'gate'-trained ckpt is reproduced exactly on load
            nn.init.zeros_(self.proj_film.weight)
            nn.init.zeros_(self.proj_film.bias)
            nn.init.zeros_(self.spatial_gain)
        if self.q_readout is not None:
            self.q_readout.reset_parameters()
        if self.conv_glob is not None:
            # zero => uniform relevance => centroid 0, scale 1 => legacy grid
            nn.init.zeros_(self.conv_glob.weight)
            nn.init.zeros_(self.conv_glob.bias)
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
        if self.hd_input_layernorm is not None:
            nn.init.ones_(self.hd_input_layernorm.weight)

    def _window_to_locs(self, win, device):
        """[x0, y0, x1, y1] in [0, 1] image fractions -> [Ns, 2] (x, y) in [-1, 1]:
        a grid_size x grid_size uniform grid over the window (row-major, the
        same token order as the reference grid). None -> None."""
        if win is None:
            return None
        # No torch.linspace(w[0], w[2]) here: tensor endpoints go through
        # .item() -> a GPU sync per call, which stalls the launch queue.
        w = win.to(device=device).float()
        u = torch.linspace(0.0, 1.0, self.grid_size, device=device)
        gx = w[0] + (w[2] - w[0]) * u
        gy = w[1] + (w[3] - w[1]) * u
        gy, gx = torch.meshgrid(gy, gx, indexing='ij')
        return (torch.stack([gx, gy], dim=-1).reshape(-1, 2) * 2.0 - 1.0).clamp(-1.0, 1.0)

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

        if self._dat_force_locs is not None:
            # Positions follow the forced sampling locations (row-major, same
            # order as the sampled tokens), mapped from [-1, 1] onto the LR range.
            fl = self._dat_force_locs.to(device=device).float()
            fx = (fl[:, 0] + 1.0) * 0.5
            fy = (fl[:, 1] + 1.0) * 0.5
            h_grid = (fy * (h_max - h_min).float() + h_min.float()).round().long()
            w_grid = (fx * (w_max - w_min).float() + w_min.float()).round().long()
            t_grid = t_val.expand(Ns).long()
            return torch.stack([t_grid, h_grid, w_grid])  # [3, Ns]

        grid_y = torch.linspace(0, 1, self.grid_size, device=device, dtype=torch.float32)
        grid_x = torch.linspace(0, 1, self.grid_size, device=device, dtype=torch.float32)

        h_hd = (grid_y * (h_max - h_min) + h_min).long()
        w_hd = (grid_x * (w_max - w_min) + w_min).long()

        h_grid = h_hd.unsqueeze(1).expand(-1, self.grid_size).flatten()
        w_grid = w_hd.unsqueeze(0).expand(self.grid_size, -1).flatten()
        t_grid = t_val.expand(Ns).long()

        return torch.stack([t_grid, h_grid, w_grid])  # [3, Ns]

    def _sample_hd_from_off_guide(self, off_guide, image_hd_features, hd_feat_idx, Lp, device,
                                  film=None, spatial=None, n_unsup_lead=0):
        """Core deformable sampling: off_guide -> offsets -> grid_sample -> KV.

        Args:
            film:    optional (gamma, beta), each [Lp*off_grps, C, 1, 1]. Applied
                     after ln_2 so the channel modulation cannot be normalized
                     away (see the note in __init__).
            spatial: optional [Lp*off_grps, 1, gh, gw] map added after ln_2;
                     additive, because any multiplicative per-position scalar is
                     cancelled exactly by a channel-wise LayerNorm.
            n_unsup_lead: leading slots (each off_grps rows) EXCLUDED from offset
                     supervision -- the question-agnostic image-conditioned slot
                     of the fused path cannot know where the answer is, so
                     pulling it toward the bbox would only teach a mean location
                     on the shared conv_off_proj.

        Returns:
            key_hd:        [Lp, Ns, kv_dim]
            value_hd:      [Lp, Ns, kv_dim]
            sampling_locs: [Lp, off_grps, grid_size, grid_size, 2]
        """
        h = self.ln_2(off_guide)
        if film is not None:
            gamma, beta = film
            h = h * (1.0 + gamma.to(h.dtype)) + beta.to(h.dtype)
        if spatial is not None:
            h = h + self.spatial_gain.view(1, -1, 1, 1).to(h.dtype) * spatial.to(h.dtype)
        offsets = self.conv_off_proj(F.silu(h)).float()
        if self.off_range > 0:
            offsets = self.off_range * torch.tanh(offsets)
        self._fn_chk("sample.offsets", offsets)
        if self.training:
            self._dat_offset_stats = (
                offsets.detach().mean().item(),
                offsets.detach().std().item(),
            )
        references = self._grid_generate(offsets.size(2), offsets.size(3), Lp, device)

        if self.conv_glob is not None:
            # Global localisation: soft-argmax over a per-cell relevance map
            # translates the grid to the relevant region and its spread shrinks
            # the grid onto it; the local offsets then refine per point.
            #   x = c + s * reference + offset,   c = E_p[g],  s = std_p[g] / std_u[g]
            # p = softmax(relevance) over the gh*gw cells, g = cell coordinates,
            # std_u = std of the uniform grid (so a flat map gives s = 1 exactly).
            # Both c and s are differentiable through p, so the offset
            # supervision (and the LM loss via grid_sample) train conv_glob to
            # put mass on the cells that matter -- a signal every point of the
            # grid shares, unlike the 3x3-local per-point offsets.
            logits = self.conv_glob(F.silu(h)).float().flatten(1)          # [R, N]
            p = torch.softmax(logits, dim=-1).unsqueeze(1)                 # [R, 1, N]
            g = references[:1].flatten(2)                                  # [1, 2, N]
            c = (p * g).sum(-1)                                            # [R, 2]
            var = (p * (g - c.unsqueeze(-1)) ** 2).sum(-1)                 # [R, 2]
            var_u = g.var(dim=-1, unbiased=False)                          # [1, 2]
            s = (var / var_u).clamp_min(1e-12).sqrt().clamp(self.glob_min_scale, 1.0)
            x = (c.unsqueeze(-1).unsqueeze(-1)
                 + s.unsqueeze(-1).unsqueeze(-1) * references + offsets)
            self._fn_chk("sample.glob", x)
            # (mean |shift|, mean scale): no .item(). Training: appended to a
            # buffer the DATMonitor drains; always: kept as the last value so
            # probe_whd_qwen35.py can read it per layer at inference.
            gstat = torch.stack([c.detach().norm(dim=1).mean(), s.detach().mean()])
            self._dat_glob_last = gstat
            if self.training:
                gb = getattr(self, '_dat_glob_buf', None)
                if gb is None:
                    gb = self._dat_glob_buf = []
                gb.append(gstat)
                if len(gb) > 512:
                    del gb[:-512]
        else:
            x = references + offsets
        if self._dat_force_locs is not None:
            fl = self._dat_force_locs.to(device=x.device, dtype=x.dtype)   # [Ns, 2] (x, y)
            x = fl.t().reshape(1, 2, x.size(2), x.size(3)).expand_as(x)
        elif self._dat_off_target is not None and self.off_sup_weight > 0 and self.training:
            # Offset supervision: pull the learned grid toward the target grid
            # (a grid_size x grid_size grid over the sample's bbox window).
            #   loss_sup = off_sup_weight * mean_i huber_delta(x_i - t_i)
            # Injected as a gradient like off_penalty (a stashed loss would
            # carry no grad_fn under gradient checkpointing):
            #   d huber_delta(r) / dr = clamp(r, -delta, delta)
            # so the pull has constant magnitude until a point is within delta
            # of its target, then fades linearly -- L1-like far away (points
            # keep moving), L2-like near (no oscillation). The gradient of the
            # LM loss through grid_sample is added on top untouched, so once the
            # grid is on target the readout decides the fine placement.
            r0 = n_unsup_lead * self.off_grps          # first supervised row
            xs = x[r0:]                                 # [(Lp-n)*G, 2, gh, gw]
            tgt = self._dat_off_target.to(device=x.device, dtype=x.dtype)  # [Ns, 2] (x, y)
            tgt = tgt.t().reshape(1, 2, x.size(2), x.size(3)).expand_as(xs)
            resid = (xs - tgt).detach()
            d = self.off_sup_delta
            # monitor: (weighted huber, mean point-to-target distance in grid
            # units) kept as 0-d GPU tensors -- no .item() here, one sync per
            # (layer, sample) would stall the launch queue like the linspace
            # sync did in the 0915 tfbox run; the DATMonitor drains and reads them.
            hub = torch.where(resid.abs() <= d, 0.5 * resid ** 2, d * (resid.abs() - 0.5 * d))
            buf = getattr(self, '_dat_off_sup_buf', None)
            if buf is None:
                buf = self._dat_off_sup_buf = []
            buf.append(torch.stack([self.off_sup_weight * hub.mean(),
                                    resid.norm(dim=1).mean()]))
            if len(buf) > 512:                 # bounded if no monitor drains it
                del buf[:-512]
            if xs.numel() > 0 and x.requires_grad:
                # loss = w * mean over supervised POINTS of huber summed over (x, y)
                coef = self.off_sup_weight / (resid.numel() // 2)
                pull = torch.zeros_like(x)
                pull[r0:] = resid.clamp(-d, d) * coef
                gbuf = getattr(self, '_dat_off_sup_grad_buf', None)
                if gbuf is None:
                    gbuf = self._dat_off_sup_grad_buf = []

                def _hook(g, p=pull, r0=r0, gbuf=gbuf):
                    # ||LM-loss grad on the supervised rows|| vs ||pull||: the
                    # ratio tells whether off_sup_weight is large enough to be
                    # seen through the LM gradient noise (wandb dat/off_sup_grad_ratio).
                    gbuf.append(torch.stack([g[r0:].detach().norm(), p[r0:].norm()]))
                    if len(gbuf) > 512:
                        del gbuf[:-512]
                    return g + p
                x.register_hook(_hook)
        if self.training:
            self._dat_offset_oob = (x.abs() > 1.0).float().mean().item()

        if self.off_range > 0:
            # A plain clamp, not the straight-through one: offsets are already
            # bounded, so the only points that can still reach the border are
            # those whose reference sits next to it, and a zero gradient there
            # is the correct signal rather than a runaway.
            sample_locs = x.clamp(-1, 1).permute(0, 2, 3, 1)
        elif self.off_penalty > 0:
            # Honest clamp plus a pull-back. The clamp gives an exact gradient
            # of 1 everywhere inside the region — unlike tanh, which also
            # compresses points that never went out of bounds — and zero
            # outside, which is at least truthful about a further push having
            # no effect. The missing ingredient is a force that returns an
            # escaped point, and that is what the penalty adds.
            #
            # Equivalent to  loss += off_penalty * mean(relu(|x| - 1) ** 2),
            # injected as a gradient rather than routed through the loss: under
            # gradient checkpointing the graph-building forward is the
            # recomputed one, so a penalty stashed during the first pass would
            # carry no grad_fn and silently contribute nothing.
            if self.training and x.requires_grad:
                over = ((x.abs() - 1.0).clamp_min(0) * torch.sign(x)).detach()
                coef = self.off_penalty * 2.0 / x.numel()
                x.register_hook(lambda g, o=over, c=coef: g + c * o)
            sample_locs = x.clamp(-1, 1).permute(0, 2, 3, 1)
        else:
            # Plain clamp: the straight-through gradient this branch used to
            # pass to out-of-bounds points kept pushing them further out (they
            # accumulate on the border with zero pull-back and whole layers
            # collapse onto a corner). Zero gradient outside is the truthful
            # signal; the reference grid half-cell margin keeps edge points
            # useful.
            sample_locs = x.clamp(-1, 1).permute(0, 2, 3, 1)

        hd_feat = image_hd_features[hd_feat_idx]  # [H_hr, W_hr, C]
        img_hr = einops.rearrange(
            hd_feat, 'h w (g c) -> g c h w',
            g=self.off_grps, c=self.off_dim,
        )
        img_hr = einops.repeat(img_hr, 'g c h w -> (l g) c h w', l=Lp)

        orig_dtype = img_hr.dtype
        # _grid_generate stores coordinates as (x, y), which is exactly the
        # convention grid_sample expects in its last dimension. The previous
        # (1, 0) indexing transposed the sampling geometry on rectangular maps.
        sampled_hr = F.grid_sample(
            img_hr.float(), sample_locs,
            mode='bilinear', align_corners=True,
        ).to(orig_dtype)

        sampled_hr = einops.rearrange(
            sampled_hr,
            '(l g) c h w -> l (h w) (g c)',
            l=Lp, g=self.off_grps, c=self.off_dim,
        )

        if self.hd_proj:
            self._fn_chk("sample.sampled_hr(pre_ln)", sampled_hr)
            sampled_hr = self.hd_input_layernorm(sampled_hr)
            self._fn_chk("sample.hd_input_layernorm", sampled_hr)
            key_hd = self.k_proj_hd(sampled_hr)
            value_hd = self.v_proj_hd(sampled_hr)
            self._fn_chk("sample.key_hd", key_hd)
            self._fn_chk("sample.value_hd", value_hd)
        else:
            # Fallback: reuse base projections (k_proj input dim must match)
            key_hd = self.k_proj(sampled_hr)
            value_hd = self.v_proj(sampled_hr)

        sampling_locs_out = sample_locs.reshape(
            Lp, self.off_grps, self.grid_size, self.grid_size, 2
        ).clone().detach()

        return key_hd, value_hd, sampling_locs_out

    def _fn_chk(self, tag, t, b_idx=None):
        """DAT_FWD_NAN: print the FIRST non-finite intermediate (ordered by
        forward execution) so the first-forward init-NaN can be localized to an
        exact stage/layer. Prints ONLY when the tensor is non-finite."""
        if not os.environ.get("DAT_FWD_NAN") or not torch.is_tensor(t):
            return
        tf = t.detach().float()
        nan = bool(torch.isnan(tf).any()); inf = bool(torch.isinf(tf).any())
        if not (nan or inf):
            return
        fin = tf[torch.isfinite(tf)]
        fmax = float(fin.abs().max()) if fin.numel() else float("nan")
        print(f"[FN-BAD] L{getattr(self, 'layer_idx', '?')} b={b_idx} {tag} "
              f"shape={tuple(t.shape)} nan={nan} inf={inf} "
              f"finabsmax={fmax:.4g} nan_frac={float(torch.isnan(tf).float().mean()):.3g}",
              flush=True)

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

        # Every read of the trunk (query_states) by the offset head goes through
        # this; see off_head_trunk_grad. The sampled HD K/V path is untouched.
        _tg = self.off_head_trunk_grad

        intention_indices = None
        embed_intention = None
        film = None            # post-norm channel modulation (intention_inject='film')
        if self.use_intention_branch:
            intention_indices = [ar[2] for ar in answer_ranges]
            intention_tokens = _grad_scale(query_states[b_idx, intention_indices], _tg)
            intention_per_group = einops.rearrange(
                intention_tokens, 'l (g c) -> l g c',
                g=self.off_grps, c=self.off_dim,
            )
            embed_intention = self.proj_intention(intention_per_group)
            embed_intention = einops.rearrange(
                embed_intention, 'l g c -> (l g) c 1 1',
            )
            if self.proj_film is not None:
                _film = einops.rearrange(
                    self.proj_film(intention_per_group), 'l g c -> (l g) c 1 1',
                )
                film = tuple(_film.chunk(2, dim=1))

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
                _grad_scale(query_states[b_idx, image_range_index], _tg),
                '(h w) (g c) -> g c h w',
                g=self.off_grps, c=self.off_dim, h=lr_h, w=lr_w,
            )
            local_embed_lr = self.conv_lr_dw(img_lr)
            self._fn_chk("conv_lr_dw", local_embed_lr, b_idx)
            local_embed_lr = F.silu(self.ln_1(local_embed_lr))
            self._fn_chk("silu(ln_1)", local_embed_lr, b_idx)
            embed_lr = self.conv_lr_proj(local_embed_lr)
            embed_lr = F.adaptive_avg_pool2d(embed_lr, (self.grid_size, self.grid_size))
            self._fn_chk("embed_lr", embed_lr, b_idx)

            embed_lr_rep = einops.repeat(
                embed_lr, 'g c h w -> (l g) c h w', l=Lp,
            )

            spatial_add = None
            if self.use_intention_branch and self.use_spatial_attn_guide:
                q_lr_flat = _grad_scale(query_states[b_idx, image_range_index], _tg)
                q_int_flat = _grad_scale(query_states[b_idx, intention_indices], _tg)
                spatial_attn = torch.matmul(
                    q_int_flat.float(), q_lr_flat.float().transpose(0, 1),
                ) / math.sqrt(q_lr_flat.shape[-1])
                spatial_attn = spatial_attn.softmax(dim=-1).view(Lp, 1, lr_h, lr_w)
                spatial_attn_guide = F.adaptive_avg_pool2d(
                    spatial_attn, (self.grid_size, self.grid_size),
                ) * (lr_h * lr_w)
                spatial_guide_rep = einops.repeat(
                    spatial_attn_guide, 'l 1 h w -> (l g) 1 h w', g=self.off_grps,
                )
                if self.intention_inject == 'film':
                    # Log-scale so uniform attention maps to exactly 0, then add
                    # it after ln_2. Multiplying it in here would be a no-op: a
                    # positive per-position scalar cancels in a channel-wise
                    # LayerNorm (verified numerically at 2e-5 relative change).
                    spatial_add = torch.log(spatial_guide_rep.clamp_min(1e-6))
                else:
                    embed_lr_rep = embed_lr_rep * spatial_guide_rep.to(embed_lr_rep.dtype)

            if self.question_inject == 'xattn':
                # Grid cells cross-attend to the full question span; the readout
                # is residual-injected into embed_lr (see QuestionReadout). This
                # is the only spatially-resolved question->offset path.
                q_hidden_list = [
                    _grad_scale(query_states[b_idx, ar[3]:ar[4]], _tg)
                    if len(ar) > 4 and ar[4] > ar[3]
                    else query_states.new_zeros((0, query_states.shape[-1]))
                    for ar in answer_ranges
                ]
                if os.environ.get("DAT_QR_PROBE"):
                    _qs = query_states[b_idx].detach().float()
                    _spans = [(int(ar[3]), int(ar[4])) for ar in answer_ranges if len(ar) > 4]
                    print(f"[QRP-SRC] L{getattr(self, 'layer_idx', '?')} b={b_idx} "
                          f"query_states nan={bool(torch.isnan(_qs).any())} "
                          f"inf={bool(torch.isinf(_qs).any())} "
                          f"absmax={float(_qs.abs().max()):.4g} spans={_spans}", flush=True)
                off_guide = embed_lr_rep + self.q_readout(
                    embed_lr_rep, q_hidden_list, self.off_grps,
                )
                self._fn_chk("xattn.embed_lr_rep", embed_lr_rep, b_idx)
                self._fn_chk("xattn.off_guide", off_guide, b_idx)
            elif self.use_intention_branch:
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
                # The image-conditioned slot is question-agnostic by definition,
                # so it is padded with the identity (gamma=beta=0, no spatial add).
                film_all, spatial_all = film, spatial_add
                if film is not None:
                    pad = film[0].new_zeros((self.off_grps,) + film[0].shape[1:])
                    film_all = (torch.cat([pad, film[0]], dim=0),
                                torch.cat([pad, film[1]], dim=0))
                if spatial_add is not None:
                    pad = spatial_add.new_zeros((self.off_grps,) + spatial_add.shape[1:])
                    spatial_all = torch.cat([pad, spatial_add], dim=0)
                key_all, value_all, slocs_all = self._sample_hd_from_off_guide(
                    off_guide_all, image_hd_features, hd_feat_idx, Lp + 1, device,
                    film=film_all, spatial=spatial_all, n_unsup_lead=1,
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
                    film=film, spatial=spatial_add,
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

        # Inference-only diagnostic: a constant added to the HD log-partition
        # before the merge. With N sampled keys instead of the 400 trained on,
        # lse2 grows by ~log(N/400) and w_hd = sigmoid(lse2 - lse1) shifts with
        # it; -log(N/400) undoes that so a grid-density sweep isolates the
        # effect of denser sampling from the effect of re-weighting the merge.
        # Only reaches this legacy merge (the exact-gradient path is training).
        if self.hd_lse_bias != 0.0:
            lse2 = lse2 + self.hd_lse_bias

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
        device = hidden_states.device
        Ns = self.grid_size * self.grid_size
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Project Q (+output gate), K, V with QKNorm — mirrors official forward
        query_states, out_gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
        )
        out_gate = out_gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.reshape(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # Apply partial + interleaved mRoPE
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # KV cache (HD KV is never cached)
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx,
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        self._fn_chk("base.query_states", query_states)
        self._fn_chk("base.key_states", key_states)
        self._fn_chk("base.value_states", value_states)

        if query_states.device.type == "cuda":
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # (Pass 1 runs below: the exact-merge path executes it inside the
        # autograd Function so its backward can be fed the merged stats.)

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

        # Sampling-distribution visualization (WandbSamplingVisCallback arms
        # _dat_request_vis one log before each vis step; we record the first
        # image-bearing sample's sampling locations).
        _want_vis = self.training and getattr(self, '_dat_request_vis', False)
        _dat_vis_entry = None  # (b_idx, slocs)

        # === Pass 2: HD cross-attention (batched via varlen) ===
        seg_q_list: List[torch.Tensor] = []
        seg_k_list: List[torch.Tensor] = []
        seg_v_list: List[torch.Tensor] = []
        seg_meta: List[Tuple[int, int, int]] = []

        for b_idx in range(B):
            if self._dat_force_batch is not None:
                # teacher forcing: this sample's forced grid [Ns, 2] (None -> learned offsets);
                # converted from the window once per forward in the top-level model.
                self._dat_force_locs = self._dat_force_batch[b_idx] \
                    if b_idx < len(self._dat_force_batch) else None
            # offset supervision: this sample's target grid [Ns, 2], or None.
            # Unconditional so a target can never leak across batches/samples.
            _tb = self._dat_off_target_batch
            self._dat_off_target = _tb[b_idx] if _tb is not None and b_idx < len(_tb) else None
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
                    _a_s, _a_e, _a_int = _ar[0], _ar[1], _ar[2]
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

            if _want_vis and _dat_vis_entry is None and _slocs is not None:
                _dat_vis_entry = (b_idx, _slocs)

            if question_segs:
                k_img = (k_img_all[0]
                         .view(Ns_total, self.num_key_value_heads, self.head_dim)
                         .unsqueeze(0).transpose(1, 2))
                v_img = (v_img_all[0]
                         .view(Ns_total, self.num_key_value_heads, self.head_dim)
                         .unsqueeze(0).transpose(1, 2))

                # Apply QKNorm to HD keys
                k_img = self.k_norm(
                    k_img.transpose(1, 2).reshape(1, Ns_total, self.num_key_value_heads, self.head_dim)
                ).transpose(1, 2)

                # HD position embeddings (partial + interleaved mRoPE)
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

                # Partial + interleaved mRoPE for HD positions
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

        if self._dat_force_batch is not None:
            self._dat_force_locs = None     # per-sample forcing ends with the loop
        self._dat_off_target = None

        # Phase 2b/2c: cross-attention + LSE merge.
        # Exact path (training default): one autograd Function owns Pass 1 +
        # Pass 2 + merge and implements the merged-stats backward — gradients
        # are exactly those of a single attention over the concatenated KV
        # set. Legacy path (inference / hd_gate / kill switch): detached-LSE
        # merge, FA2-historical stop-gradient semantics.
        use_exact_merge = (
            _EXACT_MERGE_AVAILABLE
            and self.hd_gate is None
            and torch.is_grad_enabled()
            and bool(seg_q_list)
        )
        if (
            not use_exact_merge
            and _EXACT_MERGE_GRAD
            and _EXACT_BWD_BACKEND is None
            and self.training
            and torch.is_grad_enabled()
            and bool(seg_q_list)
        ):
            # Refuse to train with silently-degraded gradients: this is the
            # exact failure mode that starved k_proj_hd/v_proj_hd historically.
            raise RuntimeError(
                "[DAT-LSE/qwen3_5] DAT_EXACT_MERGE_GRAD=1 but no raw flash backward "
                "(FA2 flash_attn_interface._flash_attn_backward or FA4 "
                "flash_attn.cute._flash_attn_bwd) is importable. Fix the flash_attn "
                "install, or set DAT_EXACT_MERGE_GRAD=0 to knowingly train with legacy "
                "detached-LSE gradients."
            )

        if use_exact_merge:
            q2_packed = torch.cat(seg_q_list, dim=0)
            k2_packed = torch.cat(seg_k_list, dim=0)
            v2_packed = torch.cat(seg_v_list, dim=0)
            nq_lens = [m[2] for m in seg_meta]
            nk_lens = [k_.shape[0] for k_ in seg_k_list]
            cu_q = torch.zeros(len(seg_meta) + 1, dtype=torch.int32, device=device)
            cu_k = torch.zeros(len(seg_meta) + 1, dtype=torch.int32, device=device)
            for i in range(len(seg_meta)):
                cu_q[i + 1] = cu_q[i] + nq_lens[i]
                cu_k[i + 1] = cu_k[i] + nk_lens[i]
            out_final = _TwoPassMergedAttnFn.apply(
                query_states, key_states, value_states,
                q2_packed, k2_packed, v2_packed,
                cu_q, cu_k, max(nq_lens), max(nk_lens),
                tuple(seg_meta),
                self if self.training else None,
                float(self.hd_lse_bias),
            )  # [B, Nq, H, D]
        else:
            out1, lse1 = _dat_attn_with_lse(
                query_states, key_states, value_states, causal=True,
            )
            if seg_q_list:
                out2_list, lse2_list = _dat_cross_attn_varlen(
                    seg_q_list, seg_k_list, seg_v_list,
                )
            else:
                out2_list, lse2_list = [], []

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

        # Visualization: record sampling locations (no attention map on the
        # two-pass path; the vis falls back to random point selection).
        if _want_vis and _dat_vis_entry is not None:
            self._dat_request_vis = False
            b_idx_v, slocs_v = _dat_vis_entry
            self._dat_vis_data = (slocs_v.detach(), None)
            self._dat_vis_b_idx = b_idx_v

        # [B, Nq, H, D] → [B, Nq, C], then apply the sigmoid output gate.
        # Gate is computed from Q-side hidden states only, so applying it after
        # the LSE merge is exactly equivalent to the official single-pass order.
        attn_output = out_final.reshape(*input_shape, -1).contiguous()
        self._fn_chk("out_final", out_final)
        self._fn_chk("out_gate", out_gate)
        attn_output = attn_output * torch.sigmoid(out_gate)
        attn_output = self.o_proj(attn_output)
        self._fn_chk("attn_output(post_o_proj)", attn_output)

        return attn_output, None


# ============================================================================
# DAT Decoder Layer
# ============================================================================

class Qwen3_5DecoderLayerDAT(Qwen3_5DecoderLayer):
    """Decoder layer with DAT attention.

    Only valid on 'full_attention' positions of the hybrid layer_types.
    The base class forward passes **kwargs to self_attn, so DAT-specific kwargs
    flow through automatically. GatedDeltaNet layers are never touched.
    """

    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int, dat_extra_args: dict):
        super().__init__(config, layer_idx)
        if self.layer_type != "full_attention":
            raise ValueError(
                f"DAT layer requested at index {layer_idx}, but layer_types[{layer_idx}] "
                f"= '{self.layer_type}'. DAT can only replace full_attention layers; "
                f"use build_dat_layers_string() to generate a valid pattern."
            )
        self.self_attn = Qwen3_5AttentionDAT(config, layer_idx, dat_extra_args)


# ============================================================================
# DAT ForConditionalGeneration (top-level model)
# ============================================================================

class Qwen3_5DATForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    """Qwen3.5 with DAT: uses the native vision encoder for both LR and HD features.

    Notes:
    - No deepstack in Qwen3.5 — the ViT emits one merged feature map.
    - MTP weights in official checkpoints are ignored on load (upstream
      _keys_to_ignore_on_load_unexpected = [r"^mtp.*"]).
    - GatedDeltaNet (linear attention) layers process the short LR-only
      sequence untouched; HD info reaches them via the residual stream.
    """
    config_class = Qwen3_5DATConfig

    def __init__(self, config: Qwen3_5DATConfig):
        super().__init__(config)

        dat_args = config.dat_extra_args
        layers_str = dat_args.get('layers', '')
        text_config = config.text_config

        if layers_str:
            assert len(layers_str) == text_config.num_hidden_layers, (
                f"Layer string length {len(layers_str)} != num_hidden_layers {text_config.num_hidden_layers}"
            )
            layer_types = list(getattr(text_config, 'layer_types', []))
            for i, layer_type in enumerate(layers_str):
                if layer_type == 'D':
                    if layer_types and layer_types[i] != "full_attention":
                        raise ValueError(
                            f"dat_layers has 'D' at index {i}, but layer_types[{i}] = "
                            f"'{layer_types[i]}'. DAT can only replace full_attention "
                            f"layers (hybrid Qwen3.5). Full-attention indices: "
                            f"{[j for j, lt in enumerate(layer_types) if lt == 'full_attention']}"
                        )
                    self.model.language_model.layers[i] = Qwen3_5DecoderLayerDAT(
                        text_config, i, dat_args
                    )
                elif layer_type == 'L':
                    pass
                else:
                    raise ValueError(f"Unknown layer type '{layer_type}' at index {i}")

            dat_count = sum(1 for c in layers_str if c == 'D')
            logger.info(f"Qwen3.5-DAT: {dat_count} DAT layers, "
                        f"{text_config.num_hidden_layers - dat_count} standard layers")

        self._patch_text_model_init_weights()

        # LR dropout: the mask is decided in forward() (where input_ids are
        # known) and applied on the language model's inputs_embeds, AFTER the
        # ViT features have been scattered in. A pre-hook covers all three ViT
        # paths (default / fused / shared) without touching HF's merge code.
        self._lr_drop_mask = None
        self._lr_drop_frac = 0.0
        self.model.language_model.register_forward_pre_hook(
            self._lr_drop_pre_hook, with_kwargs=True,
        )

    # ------------------------------------------------------------------
    # LR dropout
    # ------------------------------------------------------------------
    def _make_lr_drop_mask(self, input_ids):
        """Decide which LR image tokens to blank for this forward. Returns None
        when LR dropout is inactive. DAT_LR_DROP_FORCE=1 activates it regardless
        of training mode / lr_drop_prob (for the no-training leverage test)."""
        dat_args = self.config.dat_extra_args
        prob = float(dat_args.get('lr_drop_prob', 0.0))
        ratio = float(dat_args.get('lr_drop_ratio', 0.75))
        force = os.environ.get('DAT_LR_DROP_FORCE') == '1'
        if force:
            ratio = float(os.environ.get('DAT_LR_DROP_RATIO', ratio))
        if input_ids is None or ratio <= 0 or not (force or (self.training and prob > 0)):
            self._lr_drop_frac = 0.0
            return None
        img = input_ids == self.config.image_token_id                      # [B, L]
        if not img.any():
            self._lr_drop_frac = 0.0
            return None
        B = input_ids.shape[0]
        sel = torch.rand(B, device=input_ids.device) < (1.0 if force else prob)
        rnd = torch.rand(input_ids.shape, device=input_ids.device)
        drop = img & sel[:, None] & (rnd < ratio)
        self._lr_drop_frac = float(drop.sum()) / float(img.sum())
        return drop if bool(drop.any()) else None

    def _lr_drop_pre_hook(self, module, args, kwargs):
        drop = self._lr_drop_mask
        if drop is None:
            return None
        self._lr_drop_mask = None                     # one-shot
        emb = kwargs.get('inputs_embeds')
        if emb is None or emb.shape[:2] != drop.shape:
            return None
        img = self._lr_drop_img_mask                  # [B, L] all LR image tokens
        # per-sample mean LR embedding (detached: the fill carries no gradient)
        w = img.to(emb.dtype).unsqueeze(-1)
        fill = (emb.detach() * w).sum(1, keepdim=True) / w.sum(1, keepdim=True).clamp_min(1.0)
        kwargs['inputs_embeds'] = torch.where(drop.unsqueeze(-1), fill.to(emb.dtype), emb)
        return args, kwargs

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """Override to re-assert the on-disk DAT weights as the final load step.

        HF's post-load init pass (_initialize_weights -> _dat_init_weights) re-runs
        the full DAT init on every DAT attention module AFTER the checkpoint is
        loaded, clobbering already-loaded DAT params back to their init values
        (q_readout.layerscale -> 1e-2, hd_gate -> hd_gate_init, and — depending on
        HF's meta-materialization order — potentially the convs / hd_proj too).
        _manual_load_dat_raw_params copies the on-disk DAT tensors back in as the
        very last step, so a DAT checkpoint (stage-1 CPT -> stage-2 SFT) keeps its
        trained adapters. A fresh base conversion has no DAT keys on disk, so it is
        a no-op there and the fresh init stands.
        """
        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        cls._manual_load_dat_raw_params(model, pretrained_model_name_or_path)
        return model

    # substrings that mark a DAT-specific param that _dat_init_weights (re)inits;
    # base-attention keys (q_proj/k_proj/v_proj/o_proj/q_norm/k_norm) match none of
    # these, so a fresh base conversion collects nothing here.
    _DAT_REINIT_MARKERS = (
        '.conv_lr_dw.', '.conv_lr_proj.', '.conv_off_proj.',
        '.proj_intention.', '.proj_film.', '.spatial_gain',
        '.k_proj_hd.', '.v_proj_hd.', '.hd_input_layernorm.',
        '.hd_gate', '.q_readout.', '.ln_1.', '.ln_2.', '.conv_glob.',
    )

    @classmethod
    def _manual_load_dat_raw_params(cls, model, path):
        """Re-copy the on-disk DAT params over whatever HF's post-load init left.

        Needed because HF re-runs _dat_init_weights on the DAT attention modules
        after loading and re-initializes params it should have left alone (proven:
        q_readout.layerscale reverts to its 1e-2 init on every SFT reload even
        though the checkpoint holds the trained value and HF reports missing=0).
        Running last, this makes the checkpoint authoritative regardless of HF's
        init ordering. No-op for a fresh base conversion (no DAT keys on disk).
        """
        if not isinstance(path, str) or not os.path.isdir(path):
            return

        def _is_dat_key(k):
            return '.self_attn.' in k and any(m in k for m in cls._DAT_REINIT_MARKERS)

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
                        if _is_dat_key(k):
                            weights[k] = st.get_tensor(k)
            elif fn in ('pytorch_model.bin', 'model.bin'):
                sd = torch.load(full, map_location='cpu')
                for k, v in sd.items():
                    if _is_dat_key(k):
                        weights[k] = v

        if not weights:
            return

        targets = dict(model.named_parameters())
        targets.update(model.named_buffers())

        n_loaded = 0
        for k, v in weights.items():
            tgt = targets.get(k)
            if tgt is None or tuple(tgt.shape) != tuple(v.shape):
                continue
            with torch.no_grad():
                tgt.data.copy_(v.to(tgt.dtype).to(tgt.device))
            n_loaded += 1

        if n_loaded > 0:
            logger.info(
                f"[DAT post-load] re-asserted {n_loaded} on-disk DAT params "
                f"over HF's post-load re-init (incl. q_readout.layerscale, hd_gate)"
            )

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
            if isinstance(module, Qwen3_5AttentionDAT):
                if module.hd_gate is not None and hd_gate_init is not None:
                    module.hd_gate.data.fill_(float(hd_gate_init))
                    if hd_gate_freeze:
                        module.hd_gate.requires_grad = False
                nn.init.kaiming_normal_(module.conv_lr_dw.weight)
                nn.init.kaiming_normal_(module.conv_lr_proj.weight)
                nn.init.normal_(module.conv_off_proj.weight, std=OFF_PROJ_INIT_STD)
                if module.conv_lr_proj.bias is not None:
                    nn.init.zeros_(module.conv_lr_proj.bias)
                if isinstance(module.proj_intention, nn.Linear):
                    nn.init.xavier_uniform_(module.proj_intention.weight)
                    if module.proj_intention.bias is not None:
                        nn.init.zeros_(module.proj_intention.bias)
                module._init_hd_proj_weights()
                if module.q_readout is not None:
                    # QuestionReadout carries bare nn.Parameter (pos_emb,
                    # layerscale) and nn.LayerNorm that neither base _init_weights
                    # (Linear/RMSNorm only) nor the block above covers — on the
                    # meta-device from_pretrained flow these stay uninitialized
                    # (layerscale ~1e37 -> off_guide NaN). Init them explicitly.
                    module.q_readout.reset_parameters()
                if module.conv_glob is not None:
                    # zero = uniform relevance = identity on the grid; base
                    # _init_weights would give it a normal init (=> a random
                    # global shift at step 0). _manual_load_dat_raw_params
                    # restores trained values afterwards when present on disk.
                    nn.init.zeros_(module.conv_glob.weight)
                    nn.init.zeros_(module.conv_glob.bias)
            elif isinstance(module, _FP32WeightRMSNorm):
                nn.init.ones_(module.weight)

        text_model._init_weights = types.MethodType(_dat_init_weights, text_model)

    @torch.no_grad()
    def init_hd_proj_from_kv(self):
        """Warm-start the HD adapters from the layer's own k_proj / v_proj.

        HD features leave the same merger as the LR tokens, so the base
        projections already turn them into keys the layer's queries can match
        and values the layer knows how to read. The previous K=Kaiming, V=0
        start never escaped its own deadlock: with V~0 the attention pattern
        has no effect on the loss, so K receives only directionless gradient,
        so the attention stays random, so V only ever learns a bias. On the
        0901 / op10 / or10 / ee6 4B ckpts k_proj_hd sits at its kaiming init
        (rms 0.0198 = 1/sqrt(2560)) in every layer and v_proj_hd at ~25% of the
        base scale; disabling the branch at inference changes nothing.
        hd_input_layernorm is copied from the layer's input_layernorm for the
        same reason. Only for fresh conversions -- see convert_qwen3_5_to_dat.
        """
        n = 0
        for layer in self.model.language_model.layers:
            m = getattr(layer, 'self_attn', None)
            if not isinstance(m, Qwen3_5AttentionDAT) or m.k_proj_hd is None:
                continue
            m.k_proj_hd.weight.copy_(m.k_proj.weight)
            m.v_proj_hd.weight.copy_(m.v_proj.weight)
            if m.k_proj_hd.bias is not None and m.k_proj.bias is not None:
                m.k_proj_hd.bias.copy_(m.k_proj.bias)
            if m.v_proj_hd.bias is not None and m.v_proj.bias is not None:
                m.v_proj_hd.bias.copy_(m.v_proj.bias)
            if m.hd_input_layernorm is not None and hasattr(layer, 'input_layernorm'):
                m.hd_input_layernorm.weight.copy_(
                    layer.input_layernorm.weight.to(m.hd_input_layernorm.weight.dtype))
            n += 1
        logger.info(f"HD adapters warm-started from base k_proj / v_proj for {n} DAT layers")

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

    @contextlib.contextmanager
    def _hd_vit_truncated(self):
        """Temporarily truncate visual.blocks to the first k blocks (HD early exit).

        Port of the Qwen2.5-VL trick; structurally even simpler here. The
        Qwen3.5 ViT forward computes pos_embed / rotary / cu_seqlens before
        the block loop (all depth-independent), has no window permutation, no
        deepstack (deepstack_visual_indexes is empty), and the merger consumes
        the final hidden states directly — so running blocks[:k] + the stock
        merger is structurally identical to the full forward. All blocks are
        uniform full-attention, hence HD ViT runtime scales ~k/depth.
        """
        k = int(self.config.dat_extra_args.get('hd_early_exit_k', 0) or 0)
        visual = self.model.visual
        if k <= 0 or k >= len(visual.blocks):
            yield
            return
        full_blocks = visual.blocks
        if not getattr(self, '_hd_early_exit_logged', False):
            logger.info(
                f"[DAT] HD ViT early exit active: first {k}/{len(full_blocks)} blocks"
            )
            self._hd_early_exit_logged = True
        visual.blocks = full_blocks[:k]
        try:
            yield
        finally:
            visual.blocks = full_blocks

    def _generate_hd_features(self, pixel_values_hd, image_grid_thw_hd):
        """Generate HD feature maps from high-resolution pixel values (separate ViT call)."""
        with torch.no_grad(), self._hd_vit_truncated():
            pixel_values_hd = pixel_values_hd.type(self.model.visual.dtype)
            hd_output = self.model.visual(pixel_values_hd, grid_thw=image_grid_thw_hd, return_dict=True)
            hd_embeds = hd_output.pooler_output

        return self._parse_hd_embeds(hd_embeds, image_grid_thw_hd)

    def _parse_hd_embeds(self, hd_embeds, image_grid_thw_hd):
        """Parse flat HD embeddings into per-image [H_hd, W_hd, C] feature maps."""
        spatial_merge = self.config.vision_config.spatial_merge_size
        image_hd_features = []
        offset = 0

        if isinstance(hd_embeds, (list, tuple)):
            hd_embeds = torch.cat(list(hd_embeds), dim=0)

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

            if isinstance(combined_embeds, (list, tuple)):
                combined_embeds = torch.cat(list(combined_embeds), dim=0)

            split_sizes = (thw_combined.prod(-1) // spatial_merge ** 2).tolist()
            splits = torch.split(combined_embeds, [int(s) for s in split_sizes])
            n_lr = len(image_grid_thw)
            lr_embeds = torch.cat(list(splits[:n_lr]), dim=0)
            hd_embeds = torch.cat(list(splits[n_lr:]), dim=0)

        if isinstance(hd_embeds, (list, tuple)):
            hd_embeds = torch.cat(list(hd_embeds), dim=0)
        if isinstance(lr_embeds, (list, tuple)):
            lr_embeds = torch.cat(list(lr_embeds), dim=0)

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
            if isinstance(hd_embeds, (list, tuple)):
                hd_embeds = torch.cat(list(hd_embeds), dim=0)
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
        dat_force_window: Optional[List[Optional[torch.Tensor]]] = None,
        **kwargs,
    ) -> Union[Tuple, Qwen3_5CausalLMOutputWithPast]:
        """Forward pass with DAT HD feature injection.

        In addition to standard Qwen3.5 arguments, accepts:
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
        # Qwen3.5's get_rope_index returns [3, B, seq] (T/H/W; text uses the
        # shared running counter). Requires mm_token_type_ids — build it from
        # input_ids when the processor didn't supply it.
        mrope_position_ids = None
        if image_hd_features is not None and position_ids is None and input_ids is not None:
            if mm_token_type_ids is None:
                mm_token_type_ids = torch.zeros_like(input_ids)
                image_token_id = getattr(self.config, 'image_token_id', None)
                if image_token_id is not None:
                    mm_token_type_ids[input_ids == image_token_id] = 1
                video_token_id = getattr(self.config, 'video_token_id', None)
                if video_token_id is not None:
                    mm_token_type_ids[input_ids == video_token_id] = 2

            position_ids_3d, rope_deltas = self.model.get_rope_index(
                input_ids,
                mm_token_type_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
            )
            self.model.rope_deltas = rope_deltas
            mrope_position_ids = position_ids_3d  # [3, B, seq]
            position_ids = position_ids_3d
        elif position_ids is not None:
            if position_ids.shape[0] == 4:
                mrope_position_ids = position_ids[1:]
            elif position_ids.shape[0] == 3:
                mrope_position_ids = position_ids

        # === Step 3b: LR dropout mask (applied by the language_model pre-hook) ===
        self._lr_drop_mask = self._make_lr_drop_mask(input_ids)
        if self._lr_drop_mask is not None:
            self._lr_drop_img_mask = input_ids == self.config.image_token_id

        # === Step 3c: teacher-forced sampling windows (training data with bboxes) ===
        # A per-sample list; DAT attention modules read it per b_idx. Set on
        # EVERY forward (None when absent) rather than reset afterwards: with
        # gradient checkpointing the layer forward is recomputed during
        # backward, after this call returned, and must still see this batch's
        # windows. A fresh forward overwrites it, so nothing leaks across batches.
        if not hasattr(self, '_dat_attn_modules'):
            self._dat_attn_modules = [m for m in self.modules() if hasattr(m, '_dat_force_batch')]
        _win_locs = None
        if dat_force_window is not None and self._dat_attn_modules:
            # window -> [Ns, 2] grid once per forward (shared by all DAT layers)
            _dev = inputs_embeds.device if inputs_embeds is not None else input_ids.device
            _win_locs = [self._dat_attn_modules[0]._window_to_locs(w, _dev)
                         for w in dat_force_window]
        # Routing: with off_sup_weight > 0 the windows supervise the learned
        # grid (regression target, sampling stays the model's own); otherwise
        # they replace it (teacher forcing). Never both for the same sample.
        _supervise = bool(self._dat_attn_modules) and self._dat_attn_modules[0].off_sup_weight > 0
        for _m in self._dat_attn_modules:
            _m._dat_force_batch = None if _supervise else _win_locs
            _m._dat_off_target_batch = _win_locs if _supervise else None
        # fraction of samples carrying a window (forced or supervised)
        self._dat_tf_frac = 0.0 if dat_force_window is None else \
            sum(w is not None for w in dat_force_window) / max(1, len(dat_force_window))

        # === Step 4: Call base model with DAT kwargs ===
        # DAT kwargs flow through: Model → TextModel → DecoderLayer → Attention.
        # GatedDeltaNet layers receive and ignore them (**kwargs tolerant).
        # Qwen3_5Model accepts exactly one of input_ids / inputs_embeds, and the
        # fused and shared ViT paths above already materialised inputs_embeds.
        outputs = self.model(
            input_ids=None if inputs_embeds is not None else input_ids,
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
        # generate() prefill 传 logits_to_keep=1, 不切片会物化全序列 logits。
        hidden_states = outputs.last_hidden_state
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int) else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            logits = logits.float()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = shift_logits.view(-1, self.config.text_config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = F.cross_entropy(shift_logits, shift_labels)
            if os.environ.get("DAT_LOSS_PROBE"):
                _nv = int((shift_labels != -100).sum())
                _lf = bool(torch.isfinite(logits.detach()).all())
                _hf = bool(torch.isfinite(hidden_states.detach()).all())
                _lam = float(logits.detach().abs().max())
                _ham = float(hidden_states.detach().abs().max())
                print(f"[LOSSPROBE] loss={float(loss):.6g} finite_logits={_lf} "
                      f"finite_hidden={_hf} n_valid={_nv}/{shift_labels.numel()} "
                      f"logits_absmax={_lam:.4g} hidden_absmax={_ham:.4g}", flush=True)

        return Qwen3_5CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=getattr(outputs, 'rope_deltas', None),
        )


# ============================================================================
# Model Conversion Utility
# ============================================================================

DAT_KEYS_MATCH = [
    'conv_lr_dw', 'ln_1', 'conv_lr_proj', 'proj_intention',
    'ln_2', 'conv_off_proj', 'k_proj_hd', 'v_proj_hd',
    'hd_gate', 'hd_input_layernorm', 'q_readout',
    # intention_inject='film' extras (None unless enabled)
    'proj_film', 'spatial_gain',
    # use_global_offset extra (None unless enabled)
    'conv_glob',
]


def convert_qwen3_5_to_dat(base_model_or_path, dat_extra_args, torch_dtype=None,
                           attn_implementation=None):
    """Convert a pretrained Qwen3.5 to Qwen3.5-DAT.

    Args:
        base_model_or_path: path to pretrained Qwen3.5 checkpoint
        dat_extra_args: dict with DAT parameters. 'layers' may be an explicit
            D/L string or 'auto' / 'autoN' (anchored to full_attention positions).
        torch_dtype: optional torch dtype for loading
        attn_implementation: optional, e.g. "flash_attention_2". None keeps the
            transformers default (sdpa). Only affects HF-managed attention —
            i.e. the VISION tower: with dat_layers 'auto' every full_attention
            LLM layer becomes a DAT layer, whose attention runs through the
            module-level FA4/FA2 backend above regardless of this setting.

    Returns:
        Qwen3_5DATForConditionalGeneration with base weights + fresh DAT weights
    """
    if isinstance(base_model_or_path, str):
        base_config = Qwen3_5Config.from_pretrained(base_model_or_path)
    else:
        base_config = base_model_or_path.config

    dat_extra_args = dict(dat_extra_args)
    layers_str = dat_extra_args.get('layers', '')
    if layers_str.startswith('auto'):
        layers_str = build_dat_layers_string(base_config.text_config, layers_str)
        dat_extra_args['layers'] = layers_str
        logger.info(f"[DAT] resolved layers='{layers_str}' from layer_types")

    dat_config = Qwen3_5DATConfig(**base_config.to_dict())
    dat_config.dat_extra_args = dat_extra_args

    if isinstance(base_model_or_path, str):
        _extra_kwargs = {}
        if attn_implementation is not None:
            _extra_kwargs['attn_implementation'] = attn_implementation
        dat_model = Qwen3_5DATForConditionalGeneration.from_pretrained(
            base_model_or_path,
            config=dat_config,
            torch_dtype=torch_dtype,
            ignore_mismatched_sizes=False,
            **_extra_kwargs,
        )
    else:
        base_model_or_path.config = dat_config
        base_model_or_path.__class__ = Qwen3_5DATForConditionalGeneration
        text_config = dat_config.text_config
        if layers_str:
            assert len(layers_str) == text_config.num_hidden_layers
            for i, lt in enumerate(layers_str):
                if lt == 'D':
                    base_model_or_path.model.language_model.layers[i] = \
                        Qwen3_5DecoderLayerDAT(text_config, i, dat_extra_args)
        dat_model = base_model_or_path

    # Only a fresh conversion gets new adapters. A DAT checkpoint (stage-1 ->
    # stage-2 SFT, or a merged model) already carries trained k/v_hd; the
    # unconditional re-init that used to sit here silently threw stage 1's
    # adapters away at the start of every SFT run.
    # Qwen3_5Config.from_pretrained overwrites model_type with the class value, so
    # detect a DAT checkpoint from the raw config.json / the object's config.
    if isinstance(base_model_or_path, str):
        try:
            with open(os.path.join(base_model_or_path, 'config.json')) as _f:
                _raw = json.load(_f)
        except (OSError, ValueError):
            _raw = {}
        _is_dat_ckpt = _raw.get('model_type') == 'qwen3_5_dat' or 'dat_extra_args' in _raw
    else:
        _is_dat_ckpt = getattr(base_config, 'dat_extra_args', None) is not None
    if _is_dat_ckpt:
        logger.info("[DAT] loaded a DAT checkpoint: keeping its k/v_hd adapters (no re-init)")
    else:
        dat_model.init_hd_proj_from_kv()

    # Force fp32 storage for DAT scalar/near-unity params
    n_fixed = 0
    for m in dat_model.modules():
        if isinstance(m, Qwen3_5AttentionDAT):
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

    # --- env-gated NaN origin tracer (DAT_NAN_TRACE=1) --------------------
    # Registers a forward hook on every submodule; on the FIRST forward whose
    # output contains NaN/Inf, prints the module name + whether its inputs were
    # already non-finite. The first module printed with in_bad=False (in forward
    # execution order) is the ORIGIN. Zero overhead when the env var is unset.
    if os.environ.get('DAT_NAN_TRACE'):
        import sys as _sys
        _nt_seen = {}
        _nt_budget = [200]

        def _nt_bad(t):
            return torch.is_tensor(t) and t.is_floating_point() and \
                not torch.isfinite(t.detach()).all()

        def _nt_make(name, mod):
            def _hook(m, inp, out):
                if _nt_budget[0] <= 0 or name in _nt_seen:
                    return
                outs = out if isinstance(out, (tuple, list)) else (out,)
                if not any(_nt_bad(t) for t in outs if torch.is_tensor(t)):
                    return
                ins = inp if isinstance(inp, (tuple, list)) else (inp,)
                in_bad = any(_nt_bad(t) for t in ins if torch.is_tensor(t))
                _nt_seen[name] = 1
                _nt_budget[0] -= 1
                print(f"[NANTRACE] {name} <{type(m).__name__}> out=NON-FINITE "
                      f"in_bad={in_bad}", file=_sys.stderr, flush=True)
            return _hook

        n_hooks = 0
        for _name, _mod in dat_model.named_modules():
            if _name:
                _mod.register_forward_hook(_nt_make(_name, _mod))
                n_hooks += 1
        print(f"[NANTRACE] registered {n_hooks} forward hooks (DAT_NAN_TRACE on)",
              file=_sys.stderr, flush=True)

    # --- env-gated backward NaN origin tracer (DAT_BWD_TRACE=1) -----------
    # The forward is clean but the DAT-branch GRADIENT goes NaN at step 1.
    # register_full_backward_hook fires during the real backward (works with
    # use_reentrant=False checkpointing). The FIRST module (in backward order)
    # with grad_IN=NaN while grad_out is finite MANUFACTURED the NaN gradient.
    # Param hooks additionally flag which weight's .grad first goes NaN.
    if os.environ.get('DAT_BWD_TRACE'):
        import sys as _sys
        _bt_seen = {}
        _bt_budget = [400]

        def _bt_bad(t):
            return torch.is_tensor(t) and t.is_floating_point() and \
                not torch.isfinite(t.detach()).all()

        def _bt_make(name, mod):
            def _hook(m, grad_in, grad_out):
                if _bt_budget[0] <= 0 or name in _bt_seen:
                    return
                gis = grad_in if isinstance(grad_in, (tuple, list)) else (grad_in,)
                if not any(_bt_bad(t) for t in gis if torch.is_tensor(t)):
                    return
                gos = grad_out if isinstance(grad_out, (tuple, list)) else (grad_out,)
                go_bad = any(_bt_bad(t) for t in gos if torch.is_tensor(t))
                _bt_seen[name] = 1
                _bt_budget[0] -= 1
                print(f"[BWDTRACE] {name} <{type(m).__name__}> grad_IN=NaN "
                      f"grad_out_bad={go_bad}", file=_sys.stderr, flush=True)
            return _hook

        def _pt_make(pname):
            def _phook(g):
                key = "param:" + pname
                if _bt_budget[0] <= 0 or key in _bt_seen:
                    return
                if _bt_bad(g):
                    _bt_seen[key] = 1
                    _bt_budget[0] -= 1
                    print(f"[BWDTRACE] {key} grad=NaN", file=_sys.stderr, flush=True)
            return _phook

        n_bh = n_ph = 0
        for _name, _mod in dat_model.named_modules():
            if _name:
                _mod.register_full_backward_hook(_bt_make(_name, _mod))
                n_bh += 1
        for _pn, _p in dat_model.named_parameters():
            if _p.requires_grad:
                try:
                    _p.register_hook(_pt_make(_pn))
                    n_ph += 1
                except RuntimeError:
                    pass
        print(f"[BWDTRACE] registered {n_bh} bwd hooks + {n_ph} param hooks "
              "(DAT_BWD_TRACE on)", file=_sys.stderr, flush=True)

    # --- env-gated autograd anomaly detection (DAT_ANOMALY=1) -------------
    # Raises at the first backward op that emits NaN, with the FORWARD traceback
    # of the offending op. The most precise localizer (op + source line). Slow —
    # 1-step diagnostic only.
    if os.environ.get('DAT_ANOMALY'):
        import sys as _sys
        torch.autograd.set_detect_anomaly(True)
        print("[ANOMALY] set_detect_anomaly(True) — backward raises at first NaN op",
              file=_sys.stderr, flush=True)

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

    Note: on hybrid Qwen3.5, 'all' still only matches full_attention layers
    (linear_attention layers have no q/k/v/o_proj under self_attn).
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

    _qwen3_5_mapping = _get_ckpt_mapping("qwen3_5")
    if _qwen3_5_mapping is not None and _get_ckpt_mapping("qwen3_5_dat") is None:
        try:
            _register_ckpt_mapping("qwen3_5_dat", _qwen3_5_mapping, overwrite=False)
        except ValueError:
            pass
except ImportError:
    pass
