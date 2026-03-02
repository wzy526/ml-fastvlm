"""
Qwen2.5VL-DAT: Dynamic Attention Token extension for Qwen2.5VL.

Extends official Qwen2_5_VLForConditionalGeneration with DAT mechanism:
- Offset-based sampling from high-resolution vision features
- Per-layer HD feature injection via modified attention
- Proper 3D mRoPE position encoding for extended KV sequences

Architecture (transformers >= 5.2.0):
    Qwen2_5_VLDATForConditionalGeneration(Qwen2_5_VLForConditionalGeneration)
    ├── model: Qwen2_5_VLModel (unmodified — kwargs flow natively)
    │   ├── visual: Qwen2_5_VisionTransformerPretrainedModel  (shared for LR & HR)
    │   └── language_model: Qwen2_5_VLTextModel (unmodified)
    │       └── layers: mixed 'L' (standard) + 'D' (DAT) decoder layers
    └── lm_head: nn.Linear

HD kwargs flow: ForConditionalGeneration → Model(**kwargs) → TextModel(**kwargs)
    → DecoderLayer(**kwargs) → AttentionDAT(**kwargs)

Key difference from Qwen2-VL DAT: uses proper 3D mRoPE coordinates for HD tokens
instead of simplified sequential 1D indices.
"""

import math
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

class Qwen2_5_VLAttentionDAT(Qwen2_5_VLAttention):
    """
    Core DAT mechanism for Qwen2.5-VL with proper 3D mRoPE:
    1. Extract LR features from query -> generate sampling offsets
    2. Grid sample from HD features -> project to KV
    3. Insert HD KV into the sequence with custom attention mask
    4. Build proper 3D position IDs for extended KV (using original coordinate system)
    5. Apply mRoPE with separate Q/KV position embeddings
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

    def _grid_generate(self, h, w, n_repeats, device):
        """Generate reference sampling grid in [-1, 1]."""
        grid_y = torch.linspace(-1, 1, h, device=device, dtype=torch.float32)
        grid_x = torch.linspace(-1, 1, w, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=0)  # [2, H, W]
        return grid.unsqueeze(0).repeat(n_repeats * self.off_grps, 1, 1, 1)

    @staticmethod
    def _flip_and_fill(mask):
        """Convert binary mask (0/1) to attention mask (0 -> -inf, 1 -> 0)."""
        return mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)

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

    def _compute_hd_kv_positions(self, image_range_list_b, Nq, Ns):
        """Compute HD token positions in the extended KV sequence.

        Mirrors the insertion logic of _reorganize_kv_mask_and_pos
        without constructing the actual tensors.
        """
        answer_ranges = image_range_list_b[1:]
        hd_positions = []
        cum = 0
        insert_counter = 0

        for ar in answer_ranges:
            ans_start, ans_end, cur_intention_idx = ar[0], ar[1], ar[2]

            if ans_end > 0:  # training
                cum += cur_intention_idx + 1 - insert_counter
                hd_positions.extend(range(cum, cum + Ns))
                cum += Ns
                cum += ans_end + 1 - (cur_intention_idx + 1)
                insert_counter = ans_end + 1
            else:  # inference
                cum += cur_intention_idx + 1 - insert_counter
                hd_positions.extend(range(cum, cum + Ns))
                cum += Ns
                cum += ans_start - (cur_intention_idx + 1)
                insert_counter = ans_start

        return hd_positions

    def _compute_hd_attention(self, b_idx, slocs, hd_pos,
                              query_bhnc, key_bhnc, attn_mask_4d):
        """Compute mean attention from all queries to HD KV tokens (one sample).

        Extracts the HD columns from the attention score matrix
        (Q @ K^T / sqrt(d)), applies the causal mask, softmaxes over
        HD positions, then averages over heads and query positions.

        Returns:
            (sampling_locs, attn_map) where attn_map is
            [Lp, grid_h, grid_w] or None.
        """
        if not hd_pos:
            return (slocs, None)

        with torch.no_grad():
            hd_idx = torch.tensor(
                hd_pos, device=query_bhnc.device, dtype=torch.long,
            )

            q_b = query_bhnc[b_idx]                       # [H, Nq, D]
            k_hd = key_bhnc[b_idx, :, hd_idx, :]          # [H, n_hd, D]

            scores = torch.matmul(
                q_b, k_hd.transpose(-1, -2),
            ) / math.sqrt(self.head_dim)                   # [H, Nq, n_hd]

            mask_hd = attn_mask_4d[b_idx, :, :, hd_idx]   # [1, Nq, n_hd]
            scores = scores + mask_hd

            attn = F.softmax(scores.float(), dim=-1)       # [H, Nq, n_hd]
            attn = attn.nan_to_num(0.0)

            attn_avg = attn.mean(dim=(0, 1))               # [n_hd]

            Lp = slocs.size(0)
            attn_map = attn_avg.reshape(
                Lp, self.grid_size, self.grid_size,
            )
            return (slocs, attn_map)

    def _reorganize_kv_mask_and_pos(
        self, key_b, value_b, attn_b, key_hd, value_hd,
        hd_pos_ids, orig_pos_b,
        image_range_list_b, Nq, Ns, device, dtype,
    ):
        """Insert HD KV pairs into the sequence and build attention mask + 3D position IDs.

        Enhanced version that also tracks proper 3D mRoPE position IDs for the
        extended KV sequence.

        Args:
            key_b, value_b: [Nq, kv_dim] - original KV for one batch element
            attn_b: [1, Nq, Nq] - original attention mask
            key_hd, value_hd: [Lp, Ns, kv_dim] - HD KV per answer
            hd_pos_ids: [3, Ns] — 3D position IDs for HD tokens
            orig_pos_b: [3, Nq] — original 3D position IDs for this batch element
            image_range_list_b: ranges for this batch element
            Nq: query sequence length
            Ns: number of HD tokens per answer

        Returns:
            key_ext: [kv_len, kv_dim]
            value_ext: [kv_len, kv_dim]
            attn_ext: [kv_len, 1, Nq] (permuted for pad_sequence)
            q_pos: [3, Nq_mapped] - Q position IDs in extended KV coordinate space
            kv_pos: [3, kv_len] - extended KV position IDs
        """
        answer_ranges = image_range_list_b[1:]
        k_split, v_split, attn_split = [], [], []
        kv_pos_split = []
        q_pos_indices = []  # indices into extended KV for Q positions
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
                kv_pos_split.append(orig_pos_b[:, insert_counter:cur_intention_idx + 1])
                seg_len = seg.size(0)
                q_pos_indices.append(torch.arange(pos_counter, pos_counter + seg_len, device=device, dtype=torch.int64))
                pos_counter += seg_len

                # Part 2: HD KV + HD position IDs
                k_split.append(key_hd[l_idx])
                v_split.append(value_hd[l_idx])
                kv_pos_split.append(hd_pos_ids)  # [3, Ns]
                pos_counter += Ns

                # Part 3: after insertion to answer end (inclusive)
                seg = key_b[cur_intention_idx + 1:ans_end + 1]
                k_split.append(seg)
                v_split.append(value_b[cur_intention_idx + 1:ans_end + 1])
                kv_pos_split.append(orig_pos_b[:, cur_intention_idx + 1:ans_end + 1])
                seg_len = seg.size(0)
                q_pos_indices.append(torch.arange(pos_counter, pos_counter + seg_len, device=device, dtype=torch.int64))
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
                kv_pos_split.append(orig_pos_b[:, insert_counter:cur_intention_idx + 1])
                seg_len = seg.size(0)
                q_pos_indices.append(torch.arange(pos_counter, pos_counter + seg_len, device=device, dtype=torch.int64))
                pos_counter += seg_len

                k_split.append(key_hd[l_idx])
                v_split.append(value_hd[l_idx])
                kv_pos_split.append(hd_pos_ids)
                pos_counter += Ns

                seg = key_b[cur_intention_idx + 1:ans_start]
                k_split.append(seg)
                v_split.append(value_b[cur_intention_idx + 1:ans_start])
                kv_pos_split.append(orig_pos_b[:, cur_intention_idx + 1:ans_start])
                seg_len = seg.size(0)
                q_pos_indices.append(torch.arange(pos_counter, pos_counter + seg_len, device=device, dtype=torch.int64))
                pos_counter += seg_len

                attn_split.append(attn_b[:, :, insert_counter:cur_intention_idx + 1])
                hd_vis = torch.cat([
                    torch.zeros(1, cur_intention_idx + 1, Ns, device=device, dtype=dtype),
                    torch.ones(1, Nq - cur_intention_idx - 1, Ns, device=device, dtype=dtype),
                ], dim=1)
                attn_split.append(self._flip_and_fill(hd_vis))
                attn_split.append(attn_b[:, :, cur_intention_idx + 1:ans_start])

                insert_counter = ans_start

        # Handle trailing tokens after the last answer range
        if insert_counter < Nq:
            seg = key_b[insert_counter:Nq]
            k_split.append(seg)
            v_split.append(value_b[insert_counter:Nq])
            kv_pos_split.append(orig_pos_b[:, insert_counter:Nq])
            seg_len = seg.size(0)
            q_pos_indices.append(torch.arange(pos_counter, pos_counter + seg_len, device=device, dtype=torch.int64))
            pos_counter += seg_len
            attn_split.append(attn_b[:, :, insert_counter:Nq])

        if len(k_split) == 0:
            return (key_b, value_b, attn_b.permute(2, 0, 1).contiguous(),
                    orig_pos_b, orig_pos_b)

        key_ext = torch.cat(k_split, dim=0)
        value_ext = torch.cat(v_split, dim=0)
        attn_ext = torch.cat(attn_split, dim=2).permute(2, 0, 1).contiguous()
        kv_pos = torch.cat(kv_pos_split, dim=1)

        # Q position IDs: extract from kv_pos using indices
        q_indices = torch.cat(q_pos_indices, dim=0)  # [Nq]
        q_pos = kv_pos[:, q_indices]  # [3, Nq]

        return key_ext, value_ext, attn_ext, q_pos, kv_pos

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

        # --- DAT path ---
        B, Nq, C = hidden_states.size()
        dtype, device = hidden_states.dtype, hidden_states.device
        Ns = self.grid_size * self.grid_size
        mrope_section = self.config.rope_parameters["mrope_section"]

        # Ensure attention mask exists
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
                    image_range_list[b].append([Nq, -1, Nq - 3])

        # Build mapping from batch index to image_hd_features index
        b_idx_to_hd_idx = {}
        hd_idx = 0
        for b in range(B):
            if len(image_range_list[b]) > 0:
                b_idx_to_hd_idx[b] = hd_idx
                hd_idx += 1

        keys_concat, values_concat, attns_concat = [], [], []
        kv_pos_concat = []   # [3, kv_len_b] per batch
        q_pos_concat = []    # [3, Nq] per batch
        _want_vis = (self.training
                     and getattr(self, '_dat_request_vis', False))
        _dat_vis = [] if _want_vis else None

        for b_idx in range(B):
            if len(image_range_list[b_idx]) <= 1:
                # No answer ranges -> standard attention
                keys_concat.append(key_states[b_idx])
                values_concat.append(value_states[b_idx])
                attns_concat.append(attention_mask[b_idx].permute(2, 0, 1).contiguous())
                if mrope_position_ids is not None:
                    kv_pos_concat.append(mrope_position_ids[:, b_idx, :])
                    q_pos_concat.append(mrope_position_ids[:, b_idx, :])
                else:
                    fallback = torch.arange(Nq, device=device, dtype=torch.long).unsqueeze(0).expand(3, -1)
                    kv_pos_concat.append(fallback)
                    q_pos_concat.append(fallback)
                continue

            # Get original 3D mRoPE positions for this batch element
            if mrope_position_ids is not None:
                orig_pos_b = mrope_position_ids[:, b_idx, :]  # [3, Nq]
            else:
                orig_pos_b = torch.arange(Nq, device=device, dtype=torch.long).unsqueeze(0).expand(3, -1)

            # Generate offsets and sample from HD features
            hd_feat_idx = b_idx_to_hd_idx[b_idx]
            key_hd, value_hd, _slocs = self._generate_offsets_and_sample(
                query_states, image_hd_features, image_range_list, b_idx, hd_feat_idx,
            )
            if _dat_vis is not None:
                _hd_pos = self._compute_hd_kv_positions(
                    image_range_list[b_idx], Nq, Ns,
                )
                _dat_vis.append((b_idx, _slocs, _hd_pos))

            # Construct HD position IDs from LR image coordinate range
            lr_start, lr_end, lr_h, lr_w = image_range_list[b_idx][0]
            hd_pos_ids = self._construct_hd_position_ids(
                orig_pos_b, lr_start, lr_end, lr_h, lr_w, device,
            )

            # Reorganize KV sequence with HD insertion + position tracking
            key_ext, value_ext, attn_ext, q_pos, kv_pos = self._reorganize_kv_mask_and_pos(
                key_states[b_idx], value_states[b_idx], attention_mask[b_idx],
                key_hd, value_hd, hd_pos_ids, orig_pos_b,
                image_range_list[b_idx], Nq, Ns, device, dtype,
            )

            keys_concat.append(key_ext)
            values_concat.append(value_ext)
            attns_concat.append(attn_ext)
            kv_pos_concat.append(kv_pos)
            q_pos_concat.append(q_pos)

        # Pad batch to uniform KV length
        key_ext_padded = nn.utils.rnn.pad_sequence(keys_concat, batch_first=True, padding_value=0)
        value_ext_padded = nn.utils.rnn.pad_sequence(values_concat, batch_first=True, padding_value=0)
        attn_padded = nn.utils.rnn.pad_sequence(
            attns_concat, batch_first=True, padding_value=torch.finfo(dtype).min,
        )
        attn_mask_4d = attn_padded.permute(0, 2, 3, 1).contiguous()  # [B, heads, Nq, kv_len]

        kv_len = key_ext_padded.size(1)

        # Pad 3D position IDs to uniform kv_len
        # kv_pos_concat[i] is [3, kv_len_i] — pad dim=1 to kv_len
        kv_pos_list = []
        for kv_p in kv_pos_concat:
            pad_size = kv_len - kv_p.size(1)
            if pad_size > 0:
                kv_p = F.pad(kv_p, (0, pad_size), value=0)
            kv_pos_list.append(kv_p)
        kv_pos_padded = torch.stack(kv_pos_list, dim=1)  # [3, B, kv_len]

        # Q positions: stack directly (all have same Nq)
        q_pos_padded = torch.stack(q_pos_concat, dim=1)  # [3, B, Nq]

        # Reshape to multi-head format
        query_bhnc = query_states.view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        key_bhnc = key_ext_padded.view(B, kv_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_bhnc = value_ext_padded.view(B, kv_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # --- mRoPE with proper 3D position IDs ---
        if Nq < kv_len:
            # KV is extended: compute separate position embeddings for Q and KV
            cos_kv, sin_kv = self._dat_rotary_emb(value_bhnc, kv_pos_padded)
            cos_q, sin_q = self._dat_rotary_emb(query_bhnc, q_pos_padded)

            query_bhnc = apply_multimodal_rotary_pos_emb_single(query_bhnc, cos_q, sin_q, mrope_section)
            key_bhnc = apply_multimodal_rotary_pos_emb_single(key_bhnc, cos_kv, sin_kv, mrope_section)

            cos, sin = cos_kv, sin_kv
        else:
            # No extension: use standard Qwen2.5-VL path
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
            
        # SDPA attention
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

        if _dat_vis:
            self._dat_request_vis = False
            pick = getattr(self, '_dat_vis_pick', 0) % len(_dat_vis)
            b_idx_v, slocs_v, hd_pos_v = _dat_vis[pick]
            self._dat_vis_data = self._compute_hd_attention(
                b_idx_v, slocs_v, hd_pos_v,
                query_bhnc, key_bhnc, attn_mask_4d,
            )
            self._dat_vis_b_idx = b_idx_v

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, Nq, self.hidden_size)
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
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.text_config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

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
