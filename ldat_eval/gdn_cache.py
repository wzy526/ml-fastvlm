"""Keep Qwen3.5 linear-attention state when several tokens are appended at once.

transformers 5.6.1 reads the cached conv state and recurrent state only when the
new chunk has length 1. A mosaic continuation is a few hundred tokens, so those
layers were recomputed as if the prefix had never run. Full-attention KV was
kept, which is why the letter stayed and the parentheses moved.
"""

import inspect

import torch
import torch.nn.functional as F

_NEEDLE = "has_previous_state(self.layer_idx) and seq_len == 1"
_CHUNK = 64
# While this is set, a fresh prefill longer than the boundary keeps the linear
# state at that boundary (a multiple of 64). The caller recomputes the tail
# together with the mosaic, so the chunks line up with one full forward.
_rewind_boundary = None
_rewind_failed = False
_snap_kernel_noted = False


def enable_multitoken_gdn():
    """Return ``patched``, ``upstream``, or ``broken``.

    ``upstream`` means this transformers already continues a multi-token chunk
    from the cached state. ``broken`` means the old gate is present and the
    patch could not be installed; the caller must not use the prefix cache.
    """
    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
    except Exception as exc:
        print(f"[ldat] linear-attention continuation unavailable ({type(exc).__name__}: {exc})", flush=True)
        return "broken"
    try:
        src = inspect.getsource(Qwen3_5GatedDeltaNet.forward)
    except (OSError, TypeError) as exc:
        print(f"[ldat] linear-attention continuation unavailable ({type(exc).__name__}: {exc})", flush=True)
        return "broken"
    if _NEEDLE not in src:
        return "upstream"
    if getattr(Qwen3_5GatedDeltaNet.forward, "_ldat_multitoken", False):
        return "patched"
    orig = Qwen3_5GatedDeltaNet.forward

    def forward(self, hidden_states, cache_params=None, attention_mask=None):
        seq_len = hidden_states.shape[1]
        continue_chunk = (
            cache_params is not None
            and seq_len > 1
            and cache_params.has_previous_state(self.layer_idx)
        )
        if not continue_chunk:
            boundary = _rewind_boundary
            if (
                boundary is not None
                and cache_params is not None
                and seq_len > boundary
                and not cache_params.has_previous_state(self.layer_idx)
            ):
                return _prefill_at_chunk(
                    self, hidden_states, cache_params, attention_mask, boundary, orig
                )
            return orig(self, hidden_states, cache_params=cache_params, attention_mask=attention_mask)
        return _continue(self, hidden_states, cache_params, attention_mask)

    forward._ldat_multitoken = True
    Qwen3_5GatedDeltaNet.forward = forward
    return "patched"


def keep_recurrent_state_fp32():
    """Keep the linear-attention summary in fp32 when it is written to the cache.

    The torch chunk kernel already returns that summary in fp32. The cache
    buffer is created in the conv state's dtype, bfloat16, and ``copy_`` rounds
    it. The mosaic then reads the rounded value. Full-attention keys and the
    conv tail stay in bfloat16. Returns ``patched`` or ``broken``.
    """
    try:
        from transformers.cache_utils import LinearAttentionLayer
    except Exception as exc:
        print(f"[ldat] fp32 linear-attention state unavailable ({type(exc).__name__}: {exc})", flush=True)
        return "broken"
    current = LinearAttentionLayer.update_recurrent_state
    if getattr(current, "_ldat_fp32", False):
        return "patched"

    def update_recurrent_state(self, recurrent_states, **kwargs):
        recurrent_states = recurrent_states.float()
        buf = self.recurrent_states
        if (
            not self.is_recurrent_states_initialized
            or buf is None
            or buf.dtype != torch.float32
            or buf.shape != recurrent_states.shape
            or buf.device != recurrent_states.device
        ):
            self.recurrent_states = torch.empty(
                recurrent_states.shape, dtype=torch.float32, device=recurrent_states.device
            )
            self.is_recurrent_states_initialized = True
        self.recurrent_states.copy_(recurrent_states)
        return self.recurrent_states

    update_recurrent_state._ldat_fp32 = True
    LinearAttentionLayer.update_recurrent_state = update_recurrent_state
    return "patched"


def arm_chunk_rewind(n_tokens):
    """Keep the linear state at the last multiple of 64 in the next fresh prefill.

    Returns that boundary, or ``None`` when ``n_tokens`` is already aligned.
    The prefill still returns the hidden states of one full call on the model
    kernel. The cached summary is that same kernel run on the tokens up to the
    boundary, and the conv tail is the frames ending there.
    """
    global _rewind_boundary, _rewind_failed
    n_tokens = int(n_tokens)
    boundary = (n_tokens // _CHUNK) * _CHUNK
    _rewind_failed = False
    _rewind_boundary = boundary if 0 < boundary < n_tokens else None
    return _rewind_boundary


def disarm_chunk_rewind():
    global _rewind_boundary
    _rewind_boundary = None


def rewind_failed():
    return _rewind_failed


def _boundary_state(self, hidden_states, attention_mask, boundary):
    """State after the first ``boundary`` tokens, using the model's own chunk kernel."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_mask_to_padding_states

    hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
    hidden_states = hidden_states[:, :boundary]
    batch_size, seq_len, _ = hidden_states.shape
    mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
    kernel = self.conv_kernel_size
    conv_state = F.pad(mixed_qkv, (kernel - mixed_qkv.shape[-1], 0))
    if self.causal_conv1d_fn is not None:
        mixed_qkv = self.causal_conv1d_fn(
            x=mixed_qkv,
            weight=self.conv1d.weight.squeeze(1),
            bias=self.conv1d.bias,
            activation=self.activation,
            seq_idx=None,
        )
    else:
        mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])
    mixed_qkv = mixed_qkv.transpose(1, 2)
    query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
    query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)
    beta = self.in_proj_b(hidden_states).sigmoid()
    a = self.in_proj_a(hidden_states)
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    if self.num_v_heads // self.num_k_heads > 1:
        repeat = self.num_v_heads // self.num_k_heads
        query = query.repeat_interleave(repeat, dim=2)
        key = key.repeat_interleave(repeat, dim=2)
    _, recurrent_state = self.chunk_gated_delta_rule(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    return conv_state, recurrent_state


def _prefill_at_chunk(self, hidden_states, cache_params, attention_mask, boundary, orig):
    """One full prefill on the model kernel. The cache keeps the state at ``boundary``.

    Hidden states come from the full call. The stored summary is the same kernel
    on the tokens up to ``boundary``, and the conv tail is the frames ending there.
    """
    global _rewind_failed, _snap_kernel_noted
    output = orig(self, hidden_states, cache_params=cache_params, attention_mask=attention_mask)
    fn = self.chunk_gated_delta_rule
    if not _snap_kernel_noted:
        print(
            f"[ldat] gdn boundary state read from {getattr(fn, '__name__', type(fn).__name__)}",
            flush=True,
        )
        _snap_kernel_noted = True
    try:
        conv_state, recurrent_state = _boundary_state(self, hidden_states, attention_mask, boundary)
    except Exception as exc:
        _rewind_failed = True
        if not getattr(_prefill_at_chunk, "_err_noted", False):
            print(f"[ldat] gdn boundary state failed ({type(exc).__name__}: {exc})", flush=True)
            _prefill_at_chunk._err_noted = True
        return output
    if recurrent_state is None:
        _rewind_failed = True
        return output
    cache_params.update_conv_state(conv_state, self.layer_idx)
    cache_params.update_recurrent_state(recurrent_state, self.layer_idx)
    return output


def _continue(self, hidden_states, cache_params, attention_mask):
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_mask_to_padding_states

    hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
    batch_size, seq_len, _ = hidden_states.shape
    layer = cache_params.layers[self.layer_idx]
    conv_state = layer.conv_states
    recurrent_state = layer.recurrent_states

    mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
    z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)

    # The cached tail is the last kernel frames of the prefix. Prepending it
    # gives the causal conv its left context; the oldest extra frame falls
    # outside a kernel of this width.
    mixed_qkv = torch.cat([conv_state, mixed_qkv], dim=-1)
    kernel = self.conv_kernel_size
    if mixed_qkv.shape[-1] >= kernel:
        new_conv_state = mixed_qkv[:, :, -kernel:].contiguous()
    else:
        new_conv_state = F.pad(mixed_qkv, (kernel - mixed_qkv.shape[-1], 0))
    cache_params.update_conv_state(new_conv_state, self.layer_idx)

    if self.causal_conv1d_fn is not None:
        conv_out = self.causal_conv1d_fn(
            x=mixed_qkv,
            weight=self.conv1d.weight.squeeze(1),
            bias=self.conv1d.bias,
            activation=self.activation,
            seq_idx=None,
        )
    else:
        conv_out = F.silu(self.conv1d(mixed_qkv)[:, :, : mixed_qkv.shape[-1]])
    mixed_qkv = conv_out[:, :, -seq_len:].transpose(1, 2)
    query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
    query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

    beta = b.sigmoid()
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    if self.num_v_heads // self.num_k_heads > 1:
        repeat = self.num_v_heads // self.num_k_heads
        query = query.repeat_interleave(repeat, dim=2)
        key = key.repeat_interleave(repeat, dim=2)

    core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=recurrent_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)

    core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
    z = z.reshape(-1, self.head_v_dim)
    core_attn_out = self.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
    return self.out_proj(core_attn_out)
