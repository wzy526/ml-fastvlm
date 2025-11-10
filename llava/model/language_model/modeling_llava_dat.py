from typing import List, Optional, Tuple, Union
import einops
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from torch.nn.attention import SDPBackend, sdpa_kernel
from timm.layers import LayerNorm2d
from transformers.activations import get_activation
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.utils import logging, is_torchdynamo_compiling
from transformers.models.llama.configuration_llama import LlamaConfig

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import (
    LlamaAttention, LlamaMLP, LlamaRMSNorm, LlamaRotaryEmbedding,
    LlamaPreTrainedModel, repeat_kv, apply_rotary_pos_emb,
    AttentionMaskConverter, rotate_half
)

logger = logging.get_logger(__name__) 

def str2bool(v):
    return v.lower() in ('true', '1')


def _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: torch.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
    cache_position: torch.Tensor,
    batch_size: int,
    **kwargs
):
        
    if attention_mask is not None and attention_mask.dim() == 4:
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        causal_mask = attention_mask
    else:
        min_dtype = torch.finfo(dtype).min
        causal_mask = torch.full(
            (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
        )
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )

    return causal_mask

@dataclass
class DATModelOutput(BaseModelOutputWithPast):
    sampling_locs: Optional[List[torch.Tensor]] = None

@dataclass
class DATCausalLMOutput(CausalLMOutputWithPast):
    sampling_locs: Optional[List[torch.Tensor]] = None


class LlamaAttentionEx(LlamaAttention):

    def __init__(self, config: LlamaConfig, layer_idx: int | None = None):
        super().__init__(config, layer_idx)
        self.use_sdpa = config.dat_extra_args['use_sdpa']

        # Add missing attributes that should be inherited from LlamaAttention by wzy
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_groups = getattr(config, 'num_key_value_heads', config.num_attention_heads)
        # rotary_emb fix by wzy
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

    @torch.no_grad()
    def init_conv_weights(self):
        pass

    def forward(
        self,
        hidden_states,
        attention_mask=None, # we assume this attention mask is 4D-padded, no KV-hd added in 
        position_ids=None,
        past_key_values=None,
        output_attentions=False,
        use_cache=False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        # additional args
        image_hd_features:List[torch.Tensor]=None, # [Batch, Lb C H W]
        image_range_list:List[List[int]]=None, # [Batch, Lb]
        **kwargs
    ):
        if self.use_sdpa:
            # Copy from LlamaSdpaAttention
            bsz, q_len, _ = hidden_states.size()
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.num_key_value_groups, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_key_value_groups, self.head_dim).transpose(1, 2)
            if position_embeddings is None:
                logger.warning_once(
                    "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                    "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                    "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                    "removed and `position_embeddings` will be mandatory."
                )
                cos, sin = self.rotary_emb(value_states, position_ids)
            else:
                cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            if past_key_values is not None:
                # sin and cos are specific to RoPE models; cache_position needed for the static cache
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
            key_states = repeat_kv(key_states, self.num_heads // self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_heads // self.num_key_value_groups)
            causal_mask = attention_mask
            if attention_mask is not None:
                causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]
            # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
            # Reference: https://github.com/pytorch/pytorch/issues/112577.
            if query_states.device.type == "cuda" and causal_mask is not None:
                query_states = query_states.contiguous()
                key_states = key_states.contiguous()
                value_states = value_states.contiguous()
            # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
            # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
            is_causal = True if causal_mask is None and q_len > 1 else False
            if use_cache:
                if causal_mask.shape[3] < key_states.shape[2]: 
                    n_pad = key_states.shape[2] - causal_mask.shape[3]
                    causal_mask = F.pad(causal_mask, (0, n_pad), mode="constant", value=0.0)
                attn_output = F.scaled_dot_product_attention(
                    query_states,
                    key_states,
                    value_states,
                    attn_mask=causal_mask,
                    dropout_p=self.attention_dropout if self.training else 0.0,
                    is_causal=False
                )
            else:
                attn_output = F.scaled_dot_product_attention(
                    query_states,
                    key_states,
                    value_states,
                    attn_mask=causal_mask,
                    dropout_p=self.attention_dropout if self.training else 0.0,
                    is_causal=is_causal
                )
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(bsz, q_len, -1)
            attn_output = self.o_proj(attn_output)
            return attn_output, None, past_key_values, None
            # End of copy
        else:
            # directly pass the args
            return super().forward(
                hidden_states,
                attention_mask,
                past_key_values,
                cache_position,
                position_embeddings,
                **kwargs
            )
    
class LlamaAttentionDAT(LlamaAttentionEx):

    def __init__(self, config, layer_idx):
        super().__init__(config=config, layer_idx=layer_idx)
        self.grid_size = config.dat_extra_args['grid_size']
        self.off_ksize = config.dat_extra_args['off_ksize']
        self.off_grps = config.dat_extra_args['off_grps']
        self.inter_size = config.dat_extra_args['inter_size']
        self.use_sdpa = config.dat_extra_args['use_sdpa']
        self.lr_size = config.dat_extra_args['lr_size']
        self.zoom_ratio = config.dat_extra_args['hr_image_size'] / config.dat_extra_args['lr_image_size']
        self.vit_enc_patchsize = config.dat_extra_args['lr_image_size'] // self.lr_size
        self.hr_size = config.dat_extra_args['hr_image_size'] // self.vit_enc_patchsize
        self.hd_proj = config.dat_extra_args['hd_proj']
        self.intention_as_gate = config.dat_extra_args['intention_as_gate']
        # self.max_kv_len = config.dat_extra_args.get('max_kv_len', 4096)
        self.hidden_size = config.hidden_size
        
        # Add missing attributes that should be inherited from LlamaAttention by wzy
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)

        self.off_dim = self.hidden_size // self.off_grps
        self.conv_lr_dw = nn.Conv2d(
            in_channels=self.off_dim,
            out_channels=self.off_dim,
            kernel_size=self.off_ksize,
            stride=1,
            padding=self.off_ksize // 2,
            groups=self.off_dim,
            bias=False
        )
        self.act = get_activation('silu')
        self.ln_1 = LayerNorm2d(self.off_dim, eps=1e-5)
        self.conv_lr_proj = nn.Conv2d(
            in_channels=self.off_dim,
            out_channels=self.inter_size,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.proj_intention = nn.Linear(
            in_features=self.off_dim,
            out_features=self.inter_size
        )
        if self.intention_as_gate:
            self.ln_2 = LayerNorm2d(self.inter_size, eps=1e-5)
            self.conv_off_proj = nn.Conv2d(
                in_channels=self.inter_size,
                out_channels=2,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            )
        else:
            self.ln_2 = LayerNorm2d(self.inter_size * 2, eps=1e-5)
            self.conv_off_proj = nn.Conv2d(
                in_channels=self.inter_size * 2,
                out_channels=2,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            )
        if self.hd_proj:
            self.k_proj_hd = nn.Linear(self.hidden_size, self.hidden_size)
            self.v_proj_hd = nn.Linear(self.hidden_size, self.hidden_size)
        else:
            self.k_proj_hd = nn.Identity()
            self.v_proj_hd = nn.Identity()
        
        # rotary_emb fix by wzy
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.insert_kvhd_offset = 6

        self.init_conv_weights() # Explicitly initialize the weights of the conv layers, no zero!


    @torch.no_grad()
    def init_conv_weights(self): 
        nn.init.kaiming_normal_(self.conv_lr_dw.weight, mode='fan_in', nonlinearity='linear')
        nn.init.kaiming_normal_(self.conv_lr_proj.weight)
        nn.init.kaiming_normal_(self.conv_off_proj.weight)
        
    @torch.no_grad()
    def _grid_generate(self, Hk, Wk, B, device, dtype):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, Hk - 0.5, Hk, dtype=dtype, device=device),
            torch.linspace(0.5, Wk - 0.5, Wk, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(Wk - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(Hk - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.off_grps, -1, -1, -1).permute(0, 3, 1, 2) # B * g H W 2
        return ref

    def _flip_and_fill(self, attention_mask):
        inverted_mask = 1.0 - attention_mask
        attention_mask = inverted_mask.masked_fill(
            inverted_mask.to(torch.bool), torch.finfo(inverted_mask.dtype).min
        )
        return attention_mask

    def forward(
        self,
        hidden_states,
        attention_mask=None, # we assume this attention mask is 4D-padded, no KV-hd added in 
        position_ids=None,
        past_key_values=None,
        output_attentions=False,
        use_cache=False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        # additional args
        image_hd_features:List[torch.Tensor]=None, # [B C H W]
        image_range_list:List[List[int]]=None, # [Batch: (), (), [], [], []]
        **kwargs
    ):
        # assert image_range_list is not None
        B, Nq, C = hidden_states.size()
        dtype, device = hidden_states.dtype, hidden_states.device
        assert C == self.off_grps * self.off_dim
        if use_cache and past_key_values.get_seq_length(self.layer_idx) == 0:
            # Here we manually add an extra intetion mark to the last token
            for b in range(B):
                if len(image_range_list[b]) <= 1:
                    image_range_list[b].append([Nq, -1])
            # assert b == B - 1 and B == 1, f"b={b},B={B},rng_list={image_range_list}"
        # 0. Upon inference, if no cache, we run a prefill to cache the corresponding high-resolution features to the question, 
        # and then do a regular inference which is same as the Llama.
        if use_cache and past_key_values.get_seq_length(self.layer_idx) > 0:
            # HD features are already processed during prefill, so they are not forwarded here.
            return super().forward(
                hidden_states,
                attention_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings
            )
        # Encode the hidden states to original q, k, v
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
       
        # We assume that each HD picture has the same spatial dimension, i.e., [L, C, H_hr, W_hr]
        # Convert image_range_list to index tensor
        # A new version for no image cases: for each sample in batch ... (parallel version has to deal with random batch index)
        keys_concat, values_concat, attns_concat = [], [], []
        sampling_locs = []
        pos_q_concat = []
        for b_idx in range(B):
            if len(image_range_list[b_idx]) == 0 or image_hd_features is None:
                keys_concat.append(key_states[b_idx])
                values_concat.append(value_states[b_idx])
                attns_concat.append(attention_mask[b_idx].permute(2, 0, 1).contiguous()) # head, Nq, Nkv -> Nkv, head, Nq
                continue
            # 1. We take the low-res image query features out.
            image_range_index = torch.arange(image_range_list[b_idx][0][0], image_range_list[b_idx][0][1], device=device, dtype=torch.int64)
            img_lr = einops.rearrange(
                query_states[b_idx, image_range_index],
                '(h w) (g c)  -> g c h w',
                g=self.off_grps, c=self.off_dim, h=self.lr_size, w=self.lr_size
            )
            local_embed_lr = self.conv_lr_dw(img_lr)
            if local_embed_lr.abs().max() < 1e-6:
                logger.warning("Warning, conv_embed_lr instability causes local_embed_lr to all 0.")
            normed_local_lr = self.ln_1(local_embed_lr)
            if normed_local_lr.abs().max() < 1e-6:
                logger.warning("Warning, ln_1 instability causes normed_local_lr to all 0.")
            local_embed_lr = self.act(normed_local_lr)
            if local_embed_lr.abs().max() < 1e-6:
                logger.warning("Warning, act instability causes local_embed_lr to all 0.")
            embed_lr = self.conv_lr_proj(local_embed_lr) # g, C_inter, H_lr, W_lr
            if embed_lr.abs().max() < 1e-6:
                logger.warning("Warning, conv_lr_proj instability causes embed_lr to all 0.")
            embed_lr = F.adaptive_avg_pool2d(embed_lr, (self.grid_size, self.grid_size))  # g, C_inter, H_grid, W_grid
            # 2. For each image, we mark the intention token for each question
            intention_index = [answer_range[0] - self.insert_kvhd_offset for answer_range in image_range_list[b_idx][1:]]
            Lp = len(intention_index)
            intention_tokens = query_states[b_idx, intention_index]
            intention_tokens_per_off_group = einops.rearrange(
                intention_tokens,
                'l (g c) -> l g c',
                l=Lp, g=self.off_grps, c=self.off_dim
            )
            embed_intention = self.proj_intention(intention_tokens_per_off_group) # Lp, group, C_off -> C_inter
            # 3. We combine them to guide offset generation as per channel gate
            embed_intention = einops.repeat(
                embed_intention,
                'l g c -> l g c h w',
                l=Lp, c=self.inter_size, g=self.off_grps, h=self.grid_size, w=self.grid_size
            )
            embed_intention = einops.rearrange(
                embed_intention,
                'l g c h w -> (l g) c h w',
                l=Lp, c=self.inter_size, g=self.off_grps, h=self.grid_size, w=self.grid_size
            )
            embed_lr = einops.repeat(
                embed_lr,
                'g c h w -> l g c h w',
                l=Lp, c=self.inter_size, g=self.off_grps, h=self.grid_size, w=self.grid_size
            )
            embed_lr = einops.rearrange(
                embed_lr,
                'l g c h w -> (l g) c h w',
                l=Lp, c=self.inter_size, g=self.off_grps, h=self.grid_size, w=self.grid_size
            )
            if self.intention_as_gate:
                off_guide = embed_lr.mul(embed_intention.sigmoid().mul(2.0))
            else:
                off_guide = torch.cat([embed_lr, embed_intention], dim=1) # Lp * g, 2C_inter, Hg, Wg
            # 4. We predict offsets
            offsets = self.conv_off_proj(self.act(self.ln_2(off_guide))) # Lp * g, 2, Hg, Wg
            # 5. We generate a grid then add offsets to them
            references = self._grid_generate(offsets.size(2), offsets.size(3), Lp, offsets.device, offsets.dtype)
            sample_locs = (references + offsets).clamp(-1., 1.).permute(0, 2, 3, 1) # Lp * g, 2, Hg, Wg
            img_hr = einops.repeat(
                image_hd_features[b_idx], # n_hr, C
                'n c -> l n c',
                n=self.hr_size * self.hr_size, c=C, l=Lp
            )
            img_hr = einops.rearrange(
                img_hr,
                'l (h w) (g c) -> (l g) c h w',
                l=Lp, h=self.hr_size, w=self.hr_size, g=self.off_grps, c=self.off_dim
            )
            # 6. We sample tokens from the hr image features and then 
            # Fix sampling to fp32
            dtype = img_hr.dtype
            img_hr = img_hr.to(torch.float32)
            sample_locs = sample_locs.to(torch.float32)
            sampled_hr = F.grid_sample(
                img_hr, 
                sample_locs[..., (1, 0)],
                mode='bilinear', align_corners=True
            ) # (Lp * g), Cg, H_grid, W_grid
            sampled_hr = sampled_hr.to(dtype)
            # end of precision fix
            sampled_hr = einops.rearrange(
                sampled_hr,
                '(l g) c h w -> l (h w) (g c)',
                l=Lp, g=self.off_grps, c=self.off_dim, h=self.grid_size, w=self.grid_size
            )
            # 7. Since we have obtained the sampled features, we use either the original Llama projection or a new set of projection to get key / value
            if self.hd_proj:
                key_hd, value_hd = self.k_proj_hd(sampled_hr), self.v_proj_hd(sampled_hr)
            else:
                key_hd, value_hd = self.k_proj(sampled_hr), self.v_proj(sampled_hr)
            Ns = self.grid_size * self.grid_size
            # 8. Let's pack each key_hd and value_hd into the key / value sequences! Let's make the causal / partial causal attention mask!
            # For one image case, the format is [SYS] [IMG_I_LR] [Q1] [IMG_I_HR1] [P1] [A1] [Q2] [IMG_I_HR2] [P2] [A2] ...
            # Q = question, SYS = system prompt, P = small tokens before answer, A = answer tokens
            sampling_locs.append(sample_locs.reshape(Lp, self.off_grps, self.grid_size, self.grid_size, 2).clone().detach())
            pos_q = []
            key_b, value_b, attn_b = key_states[b_idx], value_states[b_idx], attention_mask[b_idx]
            k_split, v_split, attn_split = [], [], []
            # We split the original kv, attn; then add the hd parts into them
            insert_counter, pos_counter, next_pos_counter = 0, 0, 0
            for l_idx, (this_answer_start, this_answer_end) in enumerate(image_range_list[b_idx][1:]):
                cur_intention_idx = this_answer_start - self.insert_kvhd_offset
                if this_answer_end > 0:
                    # Training mode, we split the input embeddings into three parts: before insert point, KV-hd, and after insert point
                    # Part 1: before insert point
                    k_split.append(key_b[insert_counter:cur_intention_idx + 1])
                    v_split.append(value_b[insert_counter:cur_intention_idx + 1])
                    next_pos_counter = pos_counter + k_split[-1].size(0)
                    pos_q.append(torch.arange(pos_counter, next_pos_counter, device=device, dtype=torch.int64))
                    pos_counter = next_pos_counter
                    # Part 2: KV-hd
                    k_split.append(key_hd[l_idx])
                    v_split.append(value_hd[l_idx])
                    next_pos_counter = pos_counter + k_split[-1].size(0)
                    pos_counter = next_pos_counter
                    # Part 3: after insert point
                    k_split.append(key_b[cur_intention_idx + 1:this_answer_end + 1])
                    v_split.append(value_b[cur_intention_idx + 1:this_answer_end + 1])
                    next_pos_counter = pos_counter + k_split[-1].size(0)
                    pos_q.append(torch.arange(pos_counter, next_pos_counter, device=device, dtype=torch.int64))
                    pos_counter = next_pos_counter
                    # Part 4: attention mask
                    # before insert point, no attention; after insert point, full attention, but before next question starts
                    attn_split.append(attn_b[:,:,insert_counter:cur_intention_idx + 1]) # [head, Nq, last_end to current insert point (inclusive)]
                    add_hd_attn_mask = torch.cat([
                        torch.zeros(1, cur_intention_idx + 1, Ns, device=device, dtype=attn_b.dtype), # Before the insert point, no tokens can see the hd features to avoid leakage
                        torch.ones(1, Nq - cur_intention_idx - 1, Ns, device=device, dtype=attn_b.dtype), # After the insert point, all successive tokens can see the hd features
                        # torch.zeros(1, Nq - this_answer_end, Ns, device=device, dtype=attn_b.dtype)
                    ], dim=1)
                    attn_split.append(self._flip_and_fill(add_hd_attn_mask)) # 0 -> -inf, 1 -> 0, in shape [head, Nq, n_kv_hd]
                    attn_split.append(attn_b[:,:,cur_intention_idx + 1:this_answer_end + 1]) # [head, Nq, this_start to this_end]
                    insert_counter = this_answer_end + 1 # From the last answer end point to the next answer start point
                else:
                    # Inference mode, no answer end actually, we pack the input embeddings with hd features
                    # Part 1: before insert point
                    k_split.append(key_b[insert_counter:cur_intention_idx + 1])
                    v_split.append(value_b[insert_counter:cur_intention_idx + 1])
                    next_pos_counter = pos_counter + k_split[-1].size(0)
                    pos_q.append(torch.arange(pos_counter, next_pos_counter, device=device, dtype=torch.int64))
                    pos_counter = next_pos_counter
                    # Part 2: KV-hd
                    k_split.append(key_hd[l_idx])
                    v_split.append(value_hd[l_idx])
                    next_pos_counter = pos_counter + k_split[-1].size(0)
                    pos_counter = next_pos_counter
                    # Part 3: after insert point
                    k_split.append(key_b[cur_intention_idx + 1:this_answer_start])
                    v_split.append(value_b[cur_intention_idx + 1:this_answer_start])
                    next_pos_counter = pos_counter + k_split[-1].size(0)
                    pos_q.append(torch.arange(pos_counter, next_pos_counter, device=device, dtype=torch.int64))
                    pos_counter = next_pos_counter
                    # Part 4: 
                    attn_split.append(attn_b[:,:,insert_counter:cur_intention_idx + 1]) # [head, Nq, last_end to current insert point (inclusive)]
                    add_hd_attn_mask = torch.cat([
                        torch.zeros(1, cur_intention_idx + 1, Ns, device=device, dtype=attn_b.dtype), # Before the insert point, no tokens can see the hd features to avoid leakage
                        torch.ones(1, Nq - cur_intention_idx - 1, Ns, device=device, dtype=attn_b.dtype), # After the insert point, all successive tokens can see the hd features
                    ], dim=1)
                    attn_split.append(self._flip_and_fill(add_hd_attn_mask)) # 0 -> -inf, 1 -> 0, in shape [head, Nq, n_kv_hd]
                    attn_split.append(attn_b[:,:,cur_intention_idx + 1:this_answer_start]) # [head, Nq, after insert point to output start (exclusive)]
                    # We need confirm this_answer_start is just Nq, since Nq is the length of the question tokens and cannot be indexed in tokens
                    assert this_answer_start == Nq, "this_answer_start should be just Nq, since Nq is the length of the question tokens and cannot be indexed in tokens"
                    # This time, insert counter should always be zero.
            # We add the packed dst segments into the batch kvs
            if len(k_split) == 0 or len(v_split) == 0 or len(attn_split) == 0:
                keys_concat.append(key_b)
                values_concat.append(value_b)
                attns_concat.append(attn_b.permute(2, 0, 1).contiguous()) # head, Nq, Nkv -> Nkv, head, Nq
            else:
                keys_concat.append(torch.cat(k_split, dim=0))
                values_concat.append(torch.cat(v_split, dim=0))
                attns_concat.append(torch.cat(attn_split, dim=2).permute(2, 0, 1).contiguous()) # head, Nq, Nkv -> Nkv, head, Nq
                pos_q_concat.append(torch.cat(pos_q, dim=0))
        # When no inputs are forwarded by offset_gen modules, or there are no questions
        # We pad each key, value, attn_mask with different lengths
        key_states_plus_hd = nn.utils.rnn.pad_sequence(keys_concat, batch_first=True, padding_value=0) # [B L_padded C]
        value_states_plus_hd = nn.utils.rnn.pad_sequence(values_concat, batch_first=True, padding_value=0) # [B L_padded C]
        attn_concat = nn.utils.rnn.pad_sequence(attns_concat, batch_first=True, padding_value=torch.finfo(dtype).min)
        attn_mask_4d = attn_concat.permute(0, 2, 3, 1).contiguous() # [B h N L_padded]
        # 9. Last step: compute vanilla attention or F.sdpa get output and done
        kv_len = key_states_plus_hd.size(1)
        # if kv_len > self.max_kv_len:
        #     logger.warning(f"Warning, kv_len {kv_len} is greater than max_kv_len {self.max_kv_len}, truncating to {self.max_kv_len}")
        #     kv_len = self.max_kv_len
        #     key_states_plus_hd = key_states_plus_hd[:, :self.max_kv_len, :]
        #     value_states_plus_hd = value_states_plus_hd[:, :self.max_kv_len, :]
        #     attn_mask_4d = attn_mask_4d[:, :, :, :self.max_kv_len]
        #     kv_len = self.max_kv_len
            
        query_bhnc = einops.rearrange(query_states, 'b n (h c) -> b h n c', b=B, n=Nq, h=self.num_heads, c=self.head_dim)
        key_bhnc = einops.rearrange(key_states_plus_hd, 'b n (h c) -> b h n c', b=B, n=kv_len, h=self.num_heads, c=self.head_dim)
        value_bhnc = einops.rearrange(value_states_plus_hd, 'b n (h c) -> b h n c', b=B, n=kv_len, h=self.num_heads, c=self.head_dim)

        if Nq < kv_len:
            kv_position_ids = torch.arange(kv_len, device=device, dtype=torch.int64)[None, :]
            cos, sin = self.rotary_emb(value_bhnc, kv_position_ids)
            position_ids_q = nn.utils.rnn.pad_sequence(pos_q_concat, batch_first=True, padding_value=0)
            position_ids_q_index = position_ids_q[..., None].expand(-1, -1, self.head_dim)
            cos_q = cos.expand(B, -1, -1).gather(1, position_ids_q_index)
            sin_q = sin.expand(B, -1, -1).gather(1, position_ids_q_index)
            query_bhnc = (query_bhnc * cos_q[:, None, ...]) + (rotate_half(query_bhnc) * sin_q[:, None, ...])
            key_bhnc = (key_bhnc * cos[:, None, ...]) + (rotate_half(key_bhnc) * sin[:, None, ...])
        else:
            assert Nq == kv_len, "Nq should be the same as kv_len"
            cos, sin = self.rotary_emb(value_states_plus_hd, position_ids)
            query_bhnc, key_bhnc = apply_rotary_pos_emb(query_bhnc, key_bhnc, cos, sin)

        # 10. Store bhnc version of K,V in the prefill of inference
        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_bhnc, value_bhnc = past_key_values.update(key_bhnc, value_bhnc, self.layer_idx, cache_kwargs)
        if self.use_sdpa:
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                attn_out = F.scaled_dot_product_attention(
                    query_bhnc, key_bhnc, value_bhnc, attn_mask_4d,
                    dropout_p=0.0, is_causal=False, scale=None
                )
            attn_weights = None
        else:
            attn_pre_act = torch.einsum('b h q c, b h k c -> b h q k', query_bhnc * (self.head_dim ** -0.5), key_bhnc)
            attn_pre_act = attn_pre_act + attn_mask_4d
            attn_weights = F.softmax(attn_pre_act, dim=3, dtype=torch.float32).to(query_bhnc.dtype)
            attn_out = torch.einsum('b h q k, b h k c -> b h q c', attn_weights, value_bhnc)
        attn_output = einops.rearrange(attn_out, 'b h n c -> b n (h c)', b=B, h=self.num_heads, n=Nq, c=self.head_dim)
        # Output projection
        attn_output = self.o_proj(attn_output)            

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_values, sampling_locs

ATTN_TYPE_MAPPING = {
    'L': LlamaAttentionEx,
    'D': LlamaAttentionDAT,
}

class LlamaDecoderLayerDAT(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()

        # Manually fix the args mismatch during different transformers versions
        if not hasattr(config, 'attention_dropout'):
            setattr(config, 'attention_dropout', 0.0)
        if not hasattr(config, 'rope_theta'):
            setattr(config, 'rope_theta', 10000.0)
        if not hasattr(config, 'attention_bias'):
            setattr(config, 'attention_bias', False)

        self.config = config
        assert hasattr(self.config, 'dat_extra_args'), 'dat_extra_args must be in the config'
        self.hidden_size = config.hidden_size
        this_layer_attn_type = self.config.dat_extra_args['layers'][layer_idx]
        self.self_attn = ATTN_TYPE_MAPPING[this_layer_attn_type](config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
        image_hd_features:List[torch.Tensor]=None, # [Batch, Lb C H W]
        image_range_list:List[List[int]]=None, # [Batch, Lb]
        **kwargs
    ):
        
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        # assert image_range_list is not None
        # Self Attention
        hidden_states, self_attn_weights, present_key_value, sampling_locs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            image_hd_features=image_hd_features,
            image_range_list=image_range_list,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        outputs += (sampling_locs,)

        return outputs

class LlamaDATModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayerDAT(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._use_sdpa = config._attn_implementation == "sdpa"
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            min_dtype=min_dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        image_hd_features:List[torch.Tensor]=None, # [B C H W]
        image_range_list:List[List[int]]=None # [Batch: (), (), [], [], []]
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                logger.warning_once(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                    "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for index, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                    image_hd_features,
                    image_range_list
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    image_hd_features=image_hd_features,
                    image_range_list=image_range_list
                )

            hidden_states = layer_outputs[0]
            sampling_locs = layer_outputs[-1]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return DATModelOutput(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            sampling_locs=sampling_locs,
        )

class LlamaDATForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        # Suppress attn impl to eager.
        config._attn_implementation = 'eager'
        super().__init__(config)
        self.model = LlamaDATModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        image_hd_features=None,
        image_range_list=None
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            image_hd_features=image_hd_features,
            image_range_list=image_range_list
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            if labels is None and not is_torchdynamo_compiling():
                logger.warning_once(
                    "Starting from v4.46, the `logits` model output will have the same type as the model (except at train time, where it will always be FP32)"
                )
            # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
            # TODO: remove the float() operation in v4.46
            logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :]).float()

        loss = None
        if labels is not None:
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        return DATCausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            sampling_locs=outputs.sampling_locs,
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
        num_logits_to_keep=None,
        image_hd_features=None,
        image_range_list=None,
        **kwargs,
    ):
        # assert image_range_list is not None
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            # The clone here is for the same reason as for `position_ids`.
            model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
                device = model_inputs["inputs_embeds"].device
            else:
                batch_size, sequence_length = model_inputs["input_ids"].shape
                device = model_inputs["input_ids"].device

            dtype = self.lm_head.weight.dtype
            min_dtype = torch.finfo(dtype).min

            attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_length(),
                dtype=dtype,
                device=device,
                min_dtype=min_dtype,
                cache_position=cache_position,
                batch_size=batch_size
            )

        if num_logits_to_keep is not None:
            model_inputs["num_logits_to_keep"] = num_logits_to_keep

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "image_hd_features": image_hd_features,
                "image_range_list": image_range_list
            }
        )
        return model_inputs

    
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
    
