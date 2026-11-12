"""Backbone-specific boundary for where-head training and learned DAT inference.

The Qwen2.5 route uses the stock VLM: sampled visual tokens are placed in the
sequence, so it does not need the older DAT attention subclass or FlashAttention.
"""
import inspect

import torch


class VLMBackbone:
    def __init__(self, model, family):
        if family not in ("qwen2_5_vl", "qwen3_5"):
            raise ValueError(f"Unsupported learned DAT backbone: {family}")
        self.model, self.family = model, family
        cfg = model.config
        text = getattr(cfg, "text_config", cfg)
        self.hidden_size = text.hidden_size
        vis = cfg.vision_config
        self.patch_size = int(vis.patch_size)
        self.merge_size = int(vis.spatial_merge_size)
        self.factor = self.patch_size * self.merge_size
        self.visual = model.model.visual
        self.language_model = model.model.language_model
        self._uses_mm_types = "mm_token_type_ids" in inspect.signature(model.model.get_rope_index).parameters

    def forward(self, **kwargs):
        if not self._uses_mm_types:
            kwargs.pop("mm_token_type_ids", None)
        elif kwargs.get("mm_token_type_ids") is None and kwargs.get("image_grid_thw") is not None:
            ids = kwargs.get("input_ids")
            if ids is not None:
                kwargs["mm_token_type_ids"] = (ids == self.model.config.image_token_id).long()
        return self.model(**kwargs)

    def rope_index(self, ids, mm, image_grid_thw):
        kwargs = dict(input_ids=ids, image_grid_thw=image_grid_thw)
        if self._uses_mm_types:
            kwargs["mm_token_type_ids"] = mm
        return self.model.model.get_rope_index(**kwargs)

    @staticmethod
    def merged_visual(output):
        result = getattr(output, "pooler_output", output)
        if isinstance(result, (list, tuple)):
            result = torch.cat(list(result), 0)
        if not torch.is_tensor(result) or result.ndim != 2:
            raise RuntimeError("Expected merged ViT tokens with shape [tokens, hidden_size]")
        return result

    def visual_tokens(self, pixel_values, grid_thw):
        kwargs = dict(grid_thw=grid_thw)
        if "return_dict" in inspect.signature(self.visual.forward).parameters:
            kwargs["return_dict"] = True
        output = self.visual(pixel_values.to(self.visual.dtype), **kwargs)
        return self.merged_visual(output)

    def validate_layer(self, layer):
        if not 1 <= int(layer) < len(self.language_model.layers):
            raise ValueError("where_layer must select an intermediate decoder layer "
                             f"(1..{len(self.language_model.layers) - 1}), got {layer}")


def load_backbone(path, family="auto", *, device="cuda:0", dtype=torch.bfloat16, attn="sdpa"):
    from transformers import AutoConfig, Qwen2_5_VLForConditionalGeneration

    # Register the legacy Qwen3.5 config only when requested / present locally.
    import json
    from pathlib import Path
    config_path = Path(path).expanduser() / "config.json"
    model_type = json.loads(config_path.read_text()).get("model_type") if config_path.is_file() else None
    if model_type is None:
        model_type = AutoConfig.from_pretrained(path).model_type
    detected = {"qwen2_5_vl": "qwen2_5_vl", "qwen3_5": "qwen3_5",
                "qwen3_5_dat": "qwen3_5"}.get(model_type)
    if detected is None:
        raise ValueError(f"Unsupported checkpoint model_type={model_type!r}; this route needs "
                         "a native Qwen2.5-VL or Qwen3.5 backbone (not legacy Qwen2.5 DAT attention).")
    if family != "auto" and family != detected:
        raise ValueError(f"Requested {family}, checkpoint contains {detected}")
    if detected == "qwen3_5":
        from llava.model.language_model.modeling_qwen3_5_dat import Qwen3_5DATForConditionalGeneration
        cls = Qwen3_5DATForConditionalGeneration
    else:
        cls = Qwen2_5_VLForConditionalGeneration
    model = cls.from_pretrained(path, torch_dtype=dtype, device_map={"": device},
                               attn_implementation=attn).eval()
    return VLMBackbone(model, detected)


def where_prefix(processor, image, question, mode="train"):
    """Shared prompt for localization training/inference; no answer or bbox."""
    from llava.model.where_dat import WHERE_HINT
    if mode not in ("legacy", "train"):
        raise ValueError(f"Unknown where prefix: {mode}")
    content = [{"type": "image", "image": image}, {"type": "text", "text": question}]
    if mode == "train":
        content.append({"type": "text", "text": WHERE_HINT})
    messages = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False,
                                        add_generation_prompt=mode == "legacy", enable_thinking=False)
    if mode == "train":
        text = text.rstrip()
        if text.endswith("<|im_end|>"):
            text = text[:-len("<|im_end|>")]
    return text
