"""
Qwen2VL-DAT: Dynamic Attention Token extension for Qwen2VL.

Extends official Qwen2VLForConditionalGeneration with DAT mechanism:
- Offset-based sampling from high-resolution vision features
- Per-layer HD feature injection via modified attention
- Native mRoPE (3D position encoding) support

Architecture (transformers >= 5.2.0):
    Qwen2VLDATForConditionalGeneration(Qwen2VLForConditionalGeneration)
    ├── model: Qwen2VLModel
    │   ├── visual: Qwen2VisionTransformerPretrainedModel  (shared for LR & HR)
    │   └── language_model: Qwen2VLTextModel
    │       └── layers: mixed 'L' (standard) + 'D' (DAT) decoder layers
    └── lm_head: nn.Linear

HD kwargs flow: ForConditionalGeneration → Model(**kwargs) → TextModel(**kwargs)
    → DecoderLayerDAT(explicit HD args) → AttentionDAT(explicit HD args)
"""

import math
import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLModel,
    Qwen2VLDecoderLayer,
    Qwen2VLAttention,
    Qwen2VLRotaryEmbedding,
    Qwen2VLCausalLMOutputWithPast,
    apply_multimodal_rotary_pos_emb,
    rotate_half,
    repeat_kv,
)
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast


logger = logging.getLogger(__name__)

# Qwen2-VL special token IDs
IM_START_TOKEN_ID = 151644   # <|im_start|>


def _find_im_start_backward(ids, ans_start, im_start_token_id=IM_START_TOKEN_ID):
    """Scan backward from ans_start to find the nearest <|im_start|> token.

    Args:
        ids: 1D tensor of token IDs for one batch element
        ans_start: position to scan backward from (exclusive)
        im_start_token_id: token ID for <|im_start|>

    Returns:
        int: position of the <|im_start|> token

    Raises:
        ValueError: if no <|im_start|> found before ans_start
    """
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

class Qwen2VLDATConfig(Qwen2VLConfig):
    """Qwen2VL config extended with DAT parameters."""
    model_type = "qwen2vl_dat"

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
            'insert_kvhd_offset': 6,   # DEPRECATED: intention_idx is now computed dynamically
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
        cos: [3, batch, seq_len, head_dim]  (from Qwen2VLRotaryEmbedding)
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
    """Compute image_range_list from Qwen2VL-format inputs.

    Scans input_ids for image token regions and labels for answer regions.
    For each answer range, dynamically locates the preceding <|im_start|>
    token as the intention_idx (replaces the old hardcoded insert_kvhd_offset).

    Args:
        input_ids: [B, seq_len]
        labels: [B, seq_len] or None (inference mode)
        image_token_id: int
        im_start_token_id: int, token ID for <|im_start|> (default 151644)
        image_grid_thw: [num_images, 3] grid dimensions (t, h, w) per image
        spatial_merge_size: spatial merge factor (default 2)

    Returns:
        List per batch of:
            [(lr_start, lr_end, lr_h, lr_w), [ans1_start, ans1_end, intention_idx], ...]
        Empty list for batch items without images.
    """
    batch_size = input_ids.shape[0]
    result = []
    img_idx = 0  # index into image_grid_thw (tracks which image we're at)

    for b in range(batch_size):
        ids = input_ids[b]
        ranges = []

        # Find contiguous image token region
        image_mask = (ids == image_token_id)
        if not image_mask.any():
            result.append(ranges)
            continue

        image_indices = torch.where(image_mask)[0]
        lr_start = image_indices[0].item()
        lr_end = image_indices[-1].item() + 1

        # Get spatial dimensions from image_grid_thw
        if image_grid_thw is not None and img_idx < len(image_grid_thw):
            thw = image_grid_thw[img_idx]
            t, h, w = thw[0].item(), thw[1].item(), thw[2].item()
            lr_h = h // spatial_merge_size
            lr_w = w // spatial_merge_size
            img_idx += 1
        else:
            # Fallback: assume square
            lr_len = lr_end - lr_start
            lr_h = lr_w = int(lr_len ** 0.5)

        ranges.append((lr_start, lr_end, lr_h, lr_w))

        # Find answer ranges from labels
        if labels is not None:
            lab = labels[b]
            ans_mask = (lab != -100)
            if ans_mask.any():
                ans_indices = torch.where(ans_mask)[0]
                # Split into contiguous segments
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
            # Inference: answer starts at end of sequence
            seq_len = ids.shape[0]
            intention_idx = _find_im_start_backward(ids, seq_len, im_start_token_id)
            ranges.append([seq_len, -1, intention_idx])

        result.append(ranges)

    return result


def _build_causal_mask(seq_len, device, dtype):
    """Build a standard causal attention mask [1, 1, seq_len, seq_len]."""
    mask = torch.triu(
        torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=device, dtype=dtype),
        diagonal=1,
    )
    return mask.unsqueeze(0).unsqueeze(0)


# ============================================================================
# DAT Attention
# ============================================================================

class Qwen2VLAttentionDAT(Qwen2VLAttention):
    """
    Core DAT mechanism:
    1. Extract LR features from query -> generate sampling offsets
    2. Grid sample from HD features -> project to KV
    3. Insert HD KV into the sequence with custom attention mask
    4. Apply mRoPE with separate Q/KV position embeddings
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
            kernel_size=1, stride=1, padding=0, bias=False,
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

        # Rotary embedding for extended KV (separate Q/KV position encoding)
        self._dat_rotary_emb = Qwen2VLRotaryEmbedding(config=config)

        self._init_dat_weights()

    @torch.no_grad()
    def _init_dat_weights(self):
        _init = lambda w: nn.init.trunc_normal_(w, std=1e-4, a=-1e-2, b=1e-2)
        _init(self.conv_lr_dw.weight)
        _init(self.conv_lr_proj.weight)
        _init(self.conv_off_proj.weight)
        if self.k_proj_hd is not None:
            _init(self.k_proj_hd.weight)
            _init(self.v_proj_hd.weight)
            if self.k_proj_hd.bias is not None:
                nn.init.zeros_(self.k_proj_hd.bias)
            if self.v_proj_hd.bias is not None:
                nn.init.zeros_(self.v_proj_hd.bias)

    def _grid_generate(self, h, w, n_repeats, device, dtype):
        """Generate reference sampling grid in [-1, 1]."""
        grid_y = torch.linspace(-1, 1, h, device=device, dtype=dtype)
        grid_x = torch.linspace(-1, 1, w, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=0)  # [2, H, W]
        return grid.unsqueeze(0).repeat(n_repeats * self.off_grps, 1, 1, 1)

    @staticmethod
    def _flip_and_fill(mask):
        """Convert binary mask (0/1) to attention mask (0 -> -inf, 1 -> 0)."""
        return mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)

    def _generate_offsets_and_sample(self, query_states, image_hd_features, image_range_list, b_idx, hd_feat_idx):
        """Generate sampling offsets from LR queries and sample from HD features.

        Args:
            query_states: [B, Nq, hidden_size]
            image_hd_features: list of [H_hr, W_hr, C] per image (only images present)
            image_range_list: per-batch range info
            b_idx: batch index
            hd_feat_idx: index into image_hd_features for this batch element

        Returns:
            key_hd: [Lp, Ns, kv_dim]  - HD keys per answer
            value_hd: [Lp, Ns, kv_dim]  - HD values per answer
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

        # Intention branch
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
                off_guide = embed_lr_rep * (embed_intention.sigmoid() * 2.0)
            else:
                off_guide = torch.cat([embed_lr_rep, embed_intention], dim=1)
        else:
            off_guide = embed_lr_rep

        # 4. Predict offsets
        offsets = self.conv_off_proj(F.silu(self.ln_2(off_guide)))
        references = self._grid_generate(offsets.size(2), offsets.size(3), Lp, device, offsets.dtype)
        sample_locs = (references + offsets).clamp(-1., 1.).permute(0, 2, 3, 1)

        # 5. Grid sample from HD features
        hd_feat = image_hd_features[hd_feat_idx]  # [H_hr, W_hr, C]
        img_hr = einops.rearrange(
            hd_feat, 'h w (g c) -> g c h w',
            g=self.off_grps, c=self.off_dim,
        )
        img_hr = einops.repeat(img_hr, 'g c h w -> (l g) c h w', l=Lp)

        # Grid sample in fp32 for numerical stability
        orig_dtype = img_hr.dtype
        sampled_hr = F.grid_sample(
            img_hr.float(), sample_locs.float()[..., (1, 0)],
            mode='bilinear', align_corners=True,
        ).to(orig_dtype)

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

    def _reorganize_kv_and_mask(
        self, key_b, value_b, attn_b, key_hd, value_hd,
        image_range_list_b, Nq, Ns, device, dtype,
    ):
        """Insert HD KV pairs into the sequence and build attention mask.

        Args:
            key_b, value_b: [Nq, kv_dim] - original KV for one batch element
            attn_b: [1, Nq, Nq] - original attention mask (head dim = 1 for broadcast)
            key_hd, value_hd: [Lp, Ns, kv_dim] - HD KV per answer
            image_range_list_b: ranges for this batch element
            Nq: query sequence length
            Ns: number of HD tokens per answer

        Returns:
            key_ext: [kv_len, kv_dim]
            value_ext: [kv_len, kv_dim]
            attn_ext: [kv_len, 1, Nq] (permuted for pad_sequence)
            pos_q: [Nq_mapped] - Q position mapping into extended KV
        """
        answer_ranges = image_range_list_b[1:]
        k_split, v_split, attn_split = [], [], []
        pos_q = []
        insert_counter = 0
        pos_counter = 0

        for l_idx, answer_range in enumerate(answer_ranges):
            ans_start, ans_end, cur_intention_idx = answer_range[0], answer_range[1], answer_range[2]

            if ans_end > 0:
                # --- Training mode ---
                # Part 1: before insertion point (inclusive)
                seg = key_b[insert_counter:cur_intention_idx + 1]
                k_split.append(seg)
                v_split.append(value_b[insert_counter:cur_intention_idx + 1])
                seg_len = seg.size(0)
                pos_q.append(torch.arange(pos_counter, pos_counter + seg_len, device=device, dtype=torch.int64))
                pos_counter += seg_len

                # Part 2: HD KV
                k_split.append(key_hd[l_idx])
                v_split.append(value_hd[l_idx])
                pos_counter += Ns

                # Part 3: after insertion to answer end (inclusive)
                seg = key_b[cur_intention_idx + 1:ans_end + 1]
                k_split.append(seg)
                v_split.append(value_b[cur_intention_idx + 1:ans_end + 1])
                seg_len = seg.size(0)
                pos_q.append(torch.arange(pos_counter, pos_counter + seg_len, device=device, dtype=torch.int64))
                pos_counter += seg_len

                # Attention mask: original + HD visibility
                attn_split.append(attn_b[:, :, insert_counter:cur_intention_idx + 1])
                hd_vis = torch.cat([
                    torch.zeros(1, cur_intention_idx + 1, Ns, device=device, dtype=dtype),
                    torch.ones(1, Nq - cur_intention_idx - 1, Ns, device=device, dtype=dtype),
                ], dim=1)
                attn_split.append(self._flip_and_fill(hd_vis))
                attn_split.append(attn_b[:, :, cur_intention_idx + 1:ans_end + 1])

                insert_counter = ans_end + 1

            else:
                # --- Inference mode (ans_end == -1) ---
                seg = key_b[insert_counter:cur_intention_idx + 1]
                k_split.append(seg)
                v_split.append(value_b[insert_counter:cur_intention_idx + 1])
                seg_len = seg.size(0)
                pos_q.append(torch.arange(pos_counter, pos_counter + seg_len, device=device, dtype=torch.int64))
                pos_counter += seg_len

                k_split.append(key_hd[l_idx])
                v_split.append(value_hd[l_idx])
                pos_counter += Ns

                seg = key_b[cur_intention_idx + 1:ans_start]
                k_split.append(seg)
                v_split.append(value_b[cur_intention_idx + 1:ans_start])
                seg_len = seg.size(0)
                pos_q.append(torch.arange(pos_counter, pos_counter + seg_len, device=device, dtype=torch.int64))
                pos_counter += seg_len

                attn_split.append(attn_b[:, :, insert_counter:cur_intention_idx + 1])
                hd_vis = torch.cat([
                    torch.zeros(1, cur_intention_idx + 1, Ns, device=device, dtype=dtype),
                    torch.ones(1, Nq - cur_intention_idx - 1, Ns, device=device, dtype=dtype),
                ], dim=1)
                attn_split.append(self._flip_and_fill(hd_vis))
                attn_split.append(attn_b[:, :, cur_intention_idx + 1:ans_start])

        if len(k_split) == 0:
            return key_b, value_b, attn_b.permute(2, 0, 1).contiguous(), \
                   torch.arange(key_b.size(0), device=device, dtype=torch.int64)

        key_ext = torch.cat(k_split, dim=0)
        value_ext = torch.cat(v_split, dim=0)
        attn_ext = torch.cat(attn_split, dim=2).permute(2, 0, 1).contiguous()
        pos_q_cat = torch.cat(pos_q, dim=0)
        return key_ext, value_ext, attn_ext, pos_q_cat

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
        # DAT-specific (explicitly received from DecoderLayerDAT)
        image_hd_features: Optional[List[torch.Tensor]] = None,
        image_range_list: Optional[List[List]] = None,
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

        # --- DAT path ---
        B, Nq, C = hidden_states.size()
        dtype, device = hidden_states.dtype, hidden_states.device
        Ns = self.grid_size * self.grid_size
        mrope_section = self.config.rope_parameters["mrope_section"]

        # Ensure attention mask exists (SDPA may optimize it away)
        if attention_mask is None:
            attention_mask = _build_causal_mask(Nq, device, dtype)

        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Handle inference: add dummy answer range if only LR range exists
        if use_cache:
            for b in range(B):
                if len(image_range_list[b]) == 1:
                    # Fallback: <|im_start|> is 3 tokens before ans_start in ChatML
                    image_range_list[b].append([Nq, -1, Nq - 3])

        # Per-batch: generate offsets, sample HD, reorganize KV
        # Build mapping from batch index to image_hd_features index.
        # image_hd_features only contains entries for samples WITH images,
        # so b_idx cannot be used directly when some samples lack images.
        b_idx_to_hd_idx = {}
        hd_idx = 0
        for b in range(B):
            if len(image_range_list[b]) > 0:  # has image
                b_idx_to_hd_idx[b] = hd_idx
                hd_idx += 1

        keys_concat, values_concat, attns_concat, pos_q_concat = [], [], [], []

        for b_idx in range(B):
            if len(image_range_list[b_idx]) <= 1:
                # No answer ranges -> standard attention for this batch element
                keys_concat.append(key_states[b_idx])
                values_concat.append(value_states[b_idx])
                attns_concat.append(attention_mask[b_idx].permute(2, 0, 1).contiguous())
                pos_q_concat.append(torch.arange(Nq, device=device, dtype=torch.int64))
                continue

            # Generate offsets and sample from HD features
            hd_feat_idx = b_idx_to_hd_idx[b_idx]
            key_hd, value_hd, _ = self._generate_offsets_and_sample(
                query_states, image_hd_features, image_range_list, b_idx, hd_feat_idx,
            )

            # Reorganize KV sequence with HD insertion
            key_ext, value_ext, attn_ext, pos_q = self._reorganize_kv_and_mask(
                key_states[b_idx], value_states[b_idx], attention_mask[b_idx],
                key_hd, value_hd, image_range_list[b_idx], Nq, Ns, device, dtype,
            )

            keys_concat.append(key_ext)
            values_concat.append(value_ext)
            attns_concat.append(attn_ext)
            pos_q_concat.append(pos_q)

        # Pad batch to uniform KV length
        key_ext_padded = nn.utils.rnn.pad_sequence(keys_concat, batch_first=True, padding_value=0)
        value_ext_padded = nn.utils.rnn.pad_sequence(values_concat, batch_first=True, padding_value=0)
        attn_padded = nn.utils.rnn.pad_sequence(
            attns_concat, batch_first=True, padding_value=torch.finfo(dtype).min,
        )
        attn_mask_4d = attn_padded.permute(0, 2, 3, 1).contiguous()  # [B, heads, Nq, kv_len]

        kv_len = key_ext_padded.size(1)

        # Reshape to multi-head format
        query_bhnc = query_states.view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        key_bhnc = key_ext_padded.view(B, kv_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_bhnc = value_ext_padded.view(B, kv_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # --- mRoPE for extended sequence ---
        if Nq < kv_len:
            # KV is extended: compute separate position embeddings for Q and KV
            kv_pos_ids = torch.arange(kv_len, device=device, dtype=torch.long)
            kv_pos_ids = kv_pos_ids.view(1, 1, -1).expand(3, B, -1)
            cos_kv, sin_kv = self._dat_rotary_emb(value_bhnc, kv_pos_ids)

            # Q positions: use pos_q_concat to index into KV positions
            pos_q_padded = nn.utils.rnn.pad_sequence(pos_q_concat, batch_first=True, padding_value=0)
            pos_q_3d = pos_q_padded.unsqueeze(0).expand(3, -1, -1)  # [3, B, Nq]
            cos_q, sin_q = self._dat_rotary_emb(query_bhnc, pos_q_3d)

            query_bhnc = apply_multimodal_rotary_pos_emb_single(query_bhnc, cos_q, sin_q, mrope_section)
            key_bhnc = apply_multimodal_rotary_pos_emb_single(key_bhnc, cos_kv, sin_kv, mrope_section)

            cos, sin = cos_kv, sin_kv
        else:
            # No extension: use standard Qwen2VL path
            cos, sin = position_embeddings
            query_bhnc, key_bhnc = apply_multimodal_rotary_pos_emb(
                query_bhnc, key_bhnc, cos, sin, mrope_section,
            )

        # KV cache update
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_bhnc, value_bhnc = past_key_values.update(
                key_bhnc, value_bhnc, self.layer_idx, cache_kwargs,
            )

        # GQA: repeat KV heads to match Q heads
        key_bhnc = repeat_kv(key_bhnc, self.num_key_value_groups)
        value_bhnc = repeat_kv(value_bhnc, self.num_key_value_groups)

        # Ensure contiguous for SDPA
        if query_bhnc.device.type == "cuda":
            query_bhnc = query_bhnc.contiguous()
            key_bhnc = key_bhnc.contiguous()
            value_bhnc = value_bhnc.contiguous()

        # SDPA attention (disable cuDNN backend — it can fail with custom
        # attention masks on certain sequence-length combinations)
        with torch.nn.attention.sdpa_kernel([
            torch.nn.attention.SDPBackend.FLASH_ATTENTION,
            torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
            torch.nn.attention.SDPBackend.MATH,
        ]):
            attn_output = F.scaled_dot_product_attention(
                query_bhnc, key_bhnc, value_bhnc,
                attn_mask=attn_mask_4d,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=False,
            )

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, Nq, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, None


# ============================================================================
# DAT Decoder Layer
# ============================================================================

class Qwen2VLDecoderLayerDAT(Qwen2VLDecoderLayer):
    """Decoder layer with DAT attention (replaces standard attention).

    Explicitly receives and forwards image_hd_features and image_range_list
    to the DAT attention module.
    """

    def __init__(self, config, layer_idx: int, dat_extra_args: dict):
        super().__init__(config, layer_idx)
        # Replace standard attention with DAT attention
        self.self_attn = Qwen2VLAttentionDAT(config, layer_idx, dat_extra_args)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        # DAT-specific: explicitly received from kwargs flow
        image_hd_features: Optional[List[torch.Tensor]] = None,
        image_range_list: Optional[List[List]] = None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Explicitly pass HD inputs to DAT attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            image_hd_features=image_hd_features,
            image_range_list=image_range_list,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        return outputs


# ============================================================================
# DAT ForConditionalGeneration (top-level model)
# ============================================================================

class Qwen2VLDATForConditionalGeneration(Qwen2VLForConditionalGeneration):
    """Qwen2VL with DAT: uses native vision encoder for both LR and HD features.

    In transformers >= 5.2.0:
    - self.model = Qwen2VLModel (contains self.visual + self.language_model)
    - self.model.language_model = Qwen2VLTextModel (decoder stack)
    - HD kwargs flow through: self.model() -> language_model() -> decoder layers
    - DAT layers explicitly extract and handle HD kwargs
    """

    def __init__(self, config: Qwen2VLDATConfig):
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
                    self.model.language_model.layers[i] = Qwen2VLDecoderLayerDAT(
                        text_config, i, dat_args
                    )
                elif layer_type == 'L':
                    pass  # Keep standard layer
                else:
                    raise ValueError(f"Unknown layer type '{layer_type}' at index {i}")

            dat_count = sum(1 for c in layers_str if c == 'D')
            logger.info(f"Qwen2VL-DAT: {dat_count} DAT layers, "
                        f"{text_config.num_hidden_layers - dat_count} standard layers")

    def _generate_hd_features(self, pixel_values_hd, image_grid_thw_hd):
        """Generate HD feature maps from high-resolution pixel values.

        Uses the shared vision encoder (self.model.visual) to encode HD images.
        Returns a list of 2D spatial feature maps for grid_sample.

        Args:
            pixel_values_hd: HD pixel values (same format as Qwen2VL pixel_values)
            image_grid_thw_hd: [num_images, 3] grid dimensions for HD

        Returns:
            List of tensors [H_hr, W_hr, hidden_size] per image
        """
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
            # For single-frame images (t=1), reshape to 2D spatial map
            # For video (t>1), use first frame only (DAT designed for images)
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
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # DAT-specific
        pixel_values_hd: Optional[torch.Tensor] = None,
        image_grid_thw_hd: Optional[torch.LongTensor] = None,
        image_hd_features: Optional[List[torch.Tensor]] = None,
        image_range_list: Optional[List[List]] = None,
        **kwargs,
    ) -> Union[Tuple, Qwen2VLCausalLMOutputWithPast]:
        """Forward pass with DAT HD feature injection.

        In addition to standard Qwen2VL arguments, accepts:
            pixel_values_hd: High-resolution pixel values for HD features
            image_grid_thw_hd: Grid dimensions for HD images
            image_hd_features: Pre-computed HD features (alternative to pixel_values_hd)
            image_range_list: Pre-computed image/answer ranges (optional, auto-computed)
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

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

        # === Step 3: Call base model (handles LR vision, embedding, position IDs) ===
        # HD kwargs flow through: Model -> TextModel -> DecoderLayerDAT -> AttentionDAT
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
            return_dict=True,
            cache_position=cache_position,
            # DAT kwargs - flow through to DAT attention layers
            image_hd_features=image_hd_features,
            image_range_list=image_range_list,
            **kwargs,
        )

        # === Step 4: LM head + loss ===
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            logits = logits.float()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.text_config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return Qwen2VLCausalLMOutputWithPast(
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
]


def convert_qwen2vl_to_dat(base_model_or_path, dat_extra_args, torch_dtype=None):
    """Convert a pretrained Qwen2VL to Qwen2VL-DAT.

    Uses from_pretrained() directly with the DAT model class so that
    DeepSpeed ZeRO-3 parameter partitioning works correctly. The old
    two-model approach (create DAT + load base state_dict) fails under
    ZeRO-3 because state_dict() returns empty tensors for parameters
    on other ranks.

    Args:
        base_model_or_path: path to pretrained Qwen2VL checkpoint
        dat_extra_args: dict with DAT parameters
        torch_dtype: optional torch dtype for loading

    Returns:
        Qwen2VLDATForConditionalGeneration with base weights + fresh DAT weights
    """
    if isinstance(base_model_or_path, str):
        base_config = Qwen2VLConfig.from_pretrained(base_model_or_path)
    else:
        base_config = base_model_or_path.config

    # Create DAT config from base config
    dat_config = Qwen2VLDATConfig(**base_config.to_dict())
    dat_config.dat_extra_args = dat_extra_args

    # Load directly with DAT model class.
    # __init__ creates base layers then replaces D-layers with DAT layers.
    # from_pretrained() then loads checkpoint weights into matching params.
    # DAT-specific params (conv_lr_dw, k_proj_hd, etc.) are missing from
    # the checkpoint and keep their random init — this is expected.
    if isinstance(base_model_or_path, str):
        dat_model = Qwen2VLDATForConditionalGeneration.from_pretrained(
            base_model_or_path,
            config=dat_config,
            torch_dtype=torch_dtype,
            ignore_mismatched_sizes=False,
        )
    else:
        # Already-instantiated base model: swap class and reinit DAT layers
        base_model_or_path.config = dat_config
        base_model_or_path.__class__ = Qwen2VLDATForConditionalGeneration
        # Re-run DAT layer setup
        layers_str = dat_extra_args.get('layers', '')
        text_config = dat_config.text_config
        if layers_str:
            assert len(layers_str) == text_config.num_hidden_layers
            for i, lt in enumerate(layers_str):
                if lt == 'D':
                    base_model_or_path.model.language_model.layers[i] = \
                        Qwen2VLDecoderLayerDAT(text_config, i, dat_extra_args)
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
