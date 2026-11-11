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
    The prefill still returns hidden states for every token. Only the cached
    summary and conv tail are rewound, so the next forward can recompute the
    leftover tokens together with the mosaic.
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


def _l2norm(x, dim=-1, eps=1e-6):
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def _chunk_copy_state(
    query, key, value, g, beta, keep_tokens, chunk_size=64, use_qk_l2norm_in_kernel=False,
):
    """One chunked prefill. Also return the fp32 summary after ``keep_tokens``.

    Same math as ``torch_chunk_gated_delta_rule``. The summary is the value the
    loop holds at that chunk boundary, before the tail of this call is applied.
    """
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]
    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    query = query * (1 / (query.shape[-1] ** 0.5))
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, device=value.device, dtype=value.dtype)
    core_attn_out = torch.zeros_like(value)
    saved = None
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn_i = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = k_cumdecay[:, :, i] @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn_i @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )
        if keep_tokens is not None and (i + 1) * chunk_size == keep_tokens:
            saved = last_recurrent_state.clone()
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, saved


def _prefill_at_chunk(self, hidden_states, cache_params, attention_mask, boundary, orig):
    """Run the whole prefix in one forward, and keep the linear cache at ``boundary``.

    The hidden states are the single-call result. Only the stored summary and
    the conv tail are the values from token ``boundary``.
    """
    global _rewind_failed
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_mask_to_padding_states

    global _snap_kernel_noted
    fn = self.chunk_gated_delta_rule
    if not _snap_kernel_noted:
        print(
            f"[ldat] gdn snapshot uses the torch chunk loop; model kernel is {getattr(fn, '__name__', type(fn).__name__)}",
            flush=True,
        )
        _snap_kernel_noted = True
    hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
    batch_size, seq_len, _ = hidden_states.shape
    mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
    kernel = self.conv_kernel_size
    if boundary < kernel or boundary >= seq_len:
        _rewind_failed = True
        return orig(self, hidden_states, cache_params=cache_params, attention_mask=attention_mask)
    conv_at = mixed_qkv[:, :, boundary - kernel:boundary].contiguous()
    cache_params.update_conv_state(conv_at, self.layer_idx)
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
    z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)
    beta = b.sigmoid()
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    if self.num_v_heads // self.num_k_heads > 1:
        repeat = self.num_v_heads // self.num_k_heads
        query = query.repeat_interleave(repeat, dim=2)
        key = key.repeat_interleave(repeat, dim=2)
    core_attn_out, saved = _chunk_copy_state(
        query, key, value, g=g, beta=beta, keep_tokens=boundary, use_qk_l2norm_in_kernel=True,
    )
    if saved is None:
        raise RuntimeError(f"gdn summary was not copied at token {boundary}")
    cache_params.update_recurrent_state(saved, self.layer_idx)
    core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
    z = z.reshape(-1, self.head_v_dim)
    core_attn_out = self.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
    return self.out_proj(core_attn_out)


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
