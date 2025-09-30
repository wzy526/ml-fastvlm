#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.auto import AutoConfig, AutoModelForCausalLM
from transformers.models.llama import LlamaConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch_dat import LlavaDATMetaModel, LlavaDATMetaForCausalLM
from .modeling_llava_dat import LlamaDATModel, LlamaDATForCausalLM

class LlavaLlamaDATConfig(LlamaConfig):
    model_type = "llava_llama_dat"
    
    def to_dict(self):
        """Override to_dict to handle dat_extra_args serialization."""
        config_dict = super().to_dict()
        if hasattr(self, 'dat_extra_args') and self.dat_extra_args is not None:
            # Convert dat_extra_args to a serializable dictionary
            if hasattr(self.dat_extra_args, 'to_dict'):
                config_dict['dat_extra_args'] = self.dat_extra_args.to_dict()
            else:
                # Fallback: convert to dict manually
                config_dict['dat_extra_args'] = {
                    'lr_image_size': getattr(self.dat_extra_args, 'lr_image_size', None),
                    'hr_image_size': getattr(self.dat_extra_args, 'hr_image_size', None),
                    'grid_size': getattr(self.dat_extra_args, 'grid_size', None),
                    'off_ksize': getattr(self.dat_extra_args, 'off_ksize', None),
                    'off_grps': getattr(self.dat_extra_args, 'off_grps', None),
                    'inter_size': getattr(self.dat_extra_args, 'inter_size', None),
                    'lr_size': getattr(self.dat_extra_args, 'lr_size', None),
                    'hd_proj': getattr(self.dat_extra_args, 'hd_proj', None),
                    'layers': getattr(self.dat_extra_args, 'layers', None),
                    'use_sdpa': getattr(self.dat_extra_args, 'use_sdpa', None)
                }
        return config_dict


class LlavaLlamaDATModel(LlavaDATMetaModel, LlamaDATModel):
    config_class = LlavaLlamaDATConfig

    def __init__(self, config: LlamaConfig):
        super().__init__(config)


class LlavaLlamaDATForCausalLM(LlamaDATForCausalLM, LlavaDATMetaForCausalLM):
    config_class = LlavaLlamaDATConfig

    def __init__(self, config):
        if not hasattr(config, 'mlp_bias'):
            config.mlp_bias = False
        super().__init__(config)
        self.model = LlavaLlamaDATModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
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
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        ori_sizes: Optional[List] = None,
        image_hd_features=None, # Placeholder?
        image_range_list=None, # Placeholder?
        return_dict: Optional[bool] = None
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                image_hd_features,
                image_range_list # A list of [(Image), [begin_ans_1, end_ans_1], ..., [begin_ans_m, end_ans_m]]
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )
            # self.set_hd_info(hd_features, image_range_list, inputs_embeds, step=step, ori_sizes=ori_sizes)
            # We do not set a HD info or something
       
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            image_hd_features=image_hd_features,
            image_range_list=image_range_list
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        ori_sizes: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                hd_features,
                image_range_list
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
            # self.set_hd_info(hd_features, image_range_list, inputs_embeds, step=1.0, ori_sizes=ori_sizes)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        
        return super().generate(
            inputs_embeds=inputs_embeds,
            image_hd_features=hd_features,
            image_range_list=image_range_list,
            position_ids=position_ids,
            attention_mask=attention_mask,    
            **kwargs
        )

    def prepare_inputs_for_generation(
            self, 
            input_ids, 
            past_key_values=None,
            inputs_embeds=None, 
            **kwargs
        ):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_llama_dat", LlavaLlamaDATConfig)
AutoModelForCausalLM.register(LlavaLlamaDATConfig, LlavaLlamaDATForCausalLM)
