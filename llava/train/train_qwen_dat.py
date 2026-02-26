"""
Qwen2VL / Qwen2.5VL native training script (no LLaVA wrapper).
Trains Qwen2VLForConditionalGeneration (or DAT variants) directly using ChatML format.

Supports:
  - Baseline Qwen2VL / Qwen2.5VL fine-tuning
  - DAT (Dynamic Attention Token) training with HD image pipeline
  - Fine-grained freeze control (tune_mm_vision, tune_mm_mlp, tune_mm_llm)
  - Separate learning rates for vision tower, projector (merger), and DAT modules

Set --model_family "qwen2_5_vl" to use Qwen2.5-VL (default: "qwen2_vl").
"""

import os
import copy
import json
import logging
import math
import pathlib
import random
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

import torch
import transformers
from torch.utils.data import Dataset, Sampler
from PIL import Image

IGNORE_INDEX = -100

local_rank = None

# DAT parameter patterns (must match modeling_qwen2vl_dat.py)
DAT_KEYS_MATCH = [
    'conv_lr_dw', 'ln_1', 'conv_lr_proj', 'proj_intention',
    'ln_2', 'conv_off_proj', 'k_proj_hd', 'v_proj_hd',
]


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


# ---------------------------------------------------------------------------
# Token IDs (Qwen2-VL tokenizer, verified)
# ---------------------------------------------------------------------------
IM_START_TOKEN_ID = 151644  # <|im_start|>
IM_END_TOKEN_ID = 151645    # <|im_end|>
VISION_START_ID = 151652    # <|vision_start|>
VISION_END_ID = 151653      # <|vision_end|>
IMAGE_PAD_ID = 151655       # <|image_pad|>
ENDOFTEXT_ID = 151643       # <|endoftext|> (pad)
ASSISTANT_TOKEN_ID = 77091  # 'assistant'
NEWLINE_TOKEN_ID = 198      # '\n'

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="./Qwen2-VL-2B")
    model_family: str = field(
        default="qwen2_vl",
        metadata={"help": "Model family: 'qwen2_vl' or 'qwen2_5_vl'"}
    )
    # Legacy flag (kept for backward compat; overridden by tune_mm_* if set)
    freeze_vision: bool = field(default=False)
    # Fine-grained freeze control (Qwen3-VL style)
    tune_mm_vision: bool = field(
        default=True,
        metadata={"help": "Train visual encoder blocks (visual.blocks, excluding merger)"}
    )
    tune_mm_mlp: bool = field(
        default=True,
        metadata={"help": "Train merger/projector (visual.merger)"}
    )
    tune_mm_llm: bool = field(
        default=True,
        metadata={"help": "Train LLM decoder layers + lm_head"}
    )
    # --- DAT args ---
    use_dat: bool = field(default=False, metadata={"help": "Enable DAT model"})
    dat_layers: str = field(
        default="",
        metadata={"help": "Layer type string e.g. 'LLLDLLLDLLL...' "
                  "(L=standard, D=DAT). Must match num_hidden_layers."}
    )
    dat_grid_size: int = field(default=6)
    dat_off_ksize: int = field(default=3)
    dat_off_grps: int = field(default=1)
    dat_inter_size: int = field(default=64)
    dat_hr_scale: int = field(default=3)
    dat_hd_proj: bool = field(default=True)
    dat_use_intention_branch: bool = field(default=True)
    dat_intention_as_gate: bool = field(default=True)
    dat_insert_kvhd_offset: int = field(
        default=6,
        metadata={"help": "DEPRECATED: intention token position is now computed dynamically"}
    )
    dat_freeze_base: bool = field(
        default=True,
        metadata={"help": "When DAT is enabled, freeze all base params and only train DAT params"}
    )
    dat_warmup_steps: int = field(
        default=0,
        metadata={"help": "Two-phase DAT training: train only DAT params for this many steps, "
                  "then unfreeze DAT+LLM (ViT/connector stay frozen). 0 = disabled."}
    )


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to JSON training data."})
    image_folder: Optional[str] = field(default=None)
    system_message: str = field(default="You are a helpful assistant.")
    min_pixels: int = field(default=3136)       # 56*56
    max_pixels: int = field(default=1003520)    # 28*28*1280
    # HD data for DAT
    hd_min_pixels: int = field(default=28224, metadata={"help": "Min pixels for HD image (3^2 * 56*56, 3x linear scale)"})
    hd_max_pixels: int = field(default=9031680, metadata={"help": "Max pixels for HD image (3^2 * max_pixels, 3x linear scale)"})
    # Coupled LR-HD mode
    lr_pixels: int = field(default=112896, metadata={"help": "Target LR pixel count for coupled mode (default 336^2=112896)"})
    coupled_lr_hd: bool = field(default=False, metadata={"help": "Use coupled LR-HD processing: fixed HD target, LR = HD / hr_scale"})


@dataclass
class Qwen2VLTrainingArguments(transformers.TrainingArguments):
    model_max_length: int = field(default=4096)
    remove_unused_columns: bool = field(default=False)
    group_by_modality_length: bool = field(default=False)
    # Learning rate overrides
    vision_lr: Optional[float] = field(
        default=None,
        metadata={"help": "Legacy: single LR for all visual params. "
                  "Prefer mm_projector_lr + vision_tower_lr."}
    )
    mm_projector_lr: Optional[float] = field(
        default=None,
        metadata={"help": "LR for visual.merger (PatchMerger / projector)"}
    )
    vision_tower_lr: Optional[float] = field(
        default=None,
        metadata={"help": "LR for visual encoder blocks (visual.blocks)"}
    )
    dat_lr: Optional[float] = field(
        default=None,
        metadata={"help": "Separate LR for DAT-specific parameters"}
    )
    # LoRA
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05


# ---------------------------------------------------------------------------
# Utilities reused from train_dat.py
# ---------------------------------------------------------------------------
def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias.items():
            if k in lora_bias_names:
                to_return[k] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['visual', 'lm_head']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    rank0_print("Saving checkpoints! Please hold...")
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


# ---------------------------------------------------------------------------
# Model freeze / trainable parameter control (Qwen3-VL style)
# ---------------------------------------------------------------------------
def _get_visual_module(model):
    """Get the visual module from model, handling different transformers versions.

    Newer transformers: model.model.visual
    Older / Qwen3-VL: model.visual
    """
    if hasattr(model, 'visual'):
        return model.visual
    elif hasattr(model, 'model') and hasattr(model.model, 'visual'):
        return model.model.visual
    raise AttributeError("Cannot find visual module on model")


def _get_language_module(model):
    """Get the language module from model, handling different transformers versions.

    Newer transformers: model.model.language_model
    Older: model.model (with layers directly)
    """
    if hasattr(model, 'model'):
        inner = model.model
        if hasattr(inner, 'language_model'):
            return inner.language_model
        # Older structure: model.model has layers directly
        if hasattr(inner, 'layers'):
            return inner
    raise AttributeError("Cannot find language module on model")


def set_model(model, model_args):
    """Fine-grained freeze control following Qwen3-VL patterns.

    Qwen2VL architecture mapping:
        visual.* (excluding merger) -> vision encoder blocks
        visual.merger.*             -> projector / MLP adapter
        language_model.layers.*     -> LLM decoder
        lm_head.*                   -> LM head (follows tune_mm_llm)
    """
    # Handle legacy freeze_vision flag
    if model_args.freeze_vision:
        model_args.tune_mm_vision = False
        model_args.tune_mm_mlp = False

    visual = _get_visual_module(model)
    lang = _get_language_module(model)

    if model_args.tune_mm_vision:
        for n, p in visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in lang.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad_(True)
    else:
        for n, p in lang.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad_(False)


def print_trainable_parameters(model):
    """Print trainable parameter statistics (Qwen3-VL style)."""
    total_params = 0
    trainable_params = 0
    for _, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    rank0_print(f"Trainable params: {trainable_params:,} / {total_params:,} "
                f"({100 * trainable_params / total_params:.2f}%)")

    components = {"visual.blocks": 0, "visual.merger": 0,
                  "model.layers": 0, "lm_head": 0, "DAT": 0, "other": 0}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        n = param.numel()
        if any(k in name for k in DAT_KEYS_MATCH):
            components["DAT"] += n
        elif "merger" in name:
            components["visual.merger"] += n
        elif "visual" in name:
            components["visual.blocks"] += n
        elif name.startswith("lm_head"):
            components["lm_head"] += n
        elif "model." in name:
            components["model.layers"] += n
        else:
            components["other"] += n

    for comp, count in components.items():
        if count > 0:
            rank0_print(f"  {comp}: {count:,} trainable params")


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------

# Regex matching normalized [0,1] bbox: [0.52, 0.59, 0.82, 0.83]
import re
_NORM_BBOX_RE = re.compile(
    r'\[\s*'
    r'(0(?:\.\d+)?|1(?:\.0+)?)\s*,\s*'
    r'(0(?:\.\d+)?|1(?:\.0+)?)\s*,\s*'
    r'(0(?:\.\d+)?|1(?:\.0+)?)\s*,\s*'
    r'(0(?:\.\d+)?|1(?:\.0+)?)\s*'
    r'\]'
)


def _convert_bbox_to_qwen2vl(text):
    """Convert normalized [0,1] bbox to Qwen2-VL ×1000 integer format.

    [0.52, 0.59, 0.82, 0.83] → <|box_start|>(520,590),(820,830)<|box_end|>
    """
    def _replace(m):
        x1 = int(round(float(m.group(1)) * 1000))
        y1 = int(round(float(m.group(2)) * 1000))
        x2 = int(round(float(m.group(3)) * 1000))
        y2 = int(round(float(m.group(4)) * 1000))
        return f'<|box_start|>({x1},{y1}),({x2},{y2})<|box_end|>'
    return _NORM_BBOX_RE.sub(_replace, text)


def _convert_bbox_to_qwen2_5vl(text, image_w, image_h):
    """Convert normalized [0,1] bbox to Qwen2.5-VL absolute pixel format.

    [0.52, 0.59, 0.82, 0.83] → <|box_start|>(416,354),(656,498)<|box_end|>
    (absolute pixel coords based on original image size)
    """
    def _replace(m):
        x1 = int(round(float(m.group(1)) * image_w))
        y1 = int(round(float(m.group(2)) * image_h))
        x2 = int(round(float(m.group(3)) * image_w))
        y2 = int(round(float(m.group(4)) * image_h))
        return f'<|box_start|>({x1},{y1}),({x2},{y2})<|box_end|>'
    return _NORM_BBOX_RE.sub(_replace, text)


def _build_messages(item, image_folder, system_message, coord_format="qwen2_vl"):
    """
    Convert {image, conversations} → Qwen2VL chat messages format.
    Returns (messages, image_path_or_None).

    coord_format: "qwen2_vl" — ×1000 normalized coords
                  "qwen2_5_vl" — absolute pixel coords (requires reading image size)
                  "none" — no conversion (keep original format)
    """
    conversations = item["conversations"]
    has_image = "image" in item
    image_path = None

    if has_image:
        image_file = item["image"]
        if image_folder:
            image_path = os.path.join(image_folder, image_file)
        else:
            image_path = image_file
        # If the image file doesn't exist on disk, treat as text-only so
        # apply_chat_template won't insert a stray <|image_pad|> token.
        if not os.path.exists(image_path):
            has_image = False
            image_path = None

    # Pre-compute image size for absolute coord conversion
    image_w, image_h = None, None
    if coord_format == "qwen2_5_vl" and image_path is not None:
        with Image.open(image_path) as img_tmp:
            image_w, image_h = img_tmp.size

    messages = [{"role": "system", "content": system_message}]

    for conv in conversations:
        role_from = conv["from"]
        value = conv["value"]

        # Convert bbox coordinates
        if coord_format == "qwen2_vl":
            value = _convert_bbox_to_qwen2vl(value)
        elif coord_format == "qwen2_5_vl" and image_w is not None:
            value = _convert_bbox_to_qwen2_5vl(value, image_w, image_h)

        if role_from in ("human", "user"):
            content_parts = []
            if has_image and "<image>" in value:
                content_parts.append({"type": "image", "image": image_path})
                value = value.replace("<image>", "").strip()
                has_image = False  # only insert image once
            content_parts.append({"type": "text", "text": value})
            messages.append({"role": "user", "content": content_parts})
        elif role_from in ("gpt", "assistant"):
            messages.append({"role": "assistant", "content": value})

    return messages, image_path


def _apply_label_mask(input_ids, ignore_index=IGNORE_INDEX):
    """
    Mask everything except assistant responses in ChatML format.
    Unmasks: response content + <|im_end|> + \\n
    """
    labels = torch.full_like(input_ids, ignore_index)
    flat = input_ids.view(-1)
    L = flat.size(0)
    pos = 0
    while pos < L:
        if flat[pos].item() == ASSISTANT_TOKEN_ID:
            ans_start = pos + 2  # skip 'assistant' + '\n'
            ans_end = ans_start
            while ans_end < L and flat[ans_end].item() != IM_END_TOKEN_ID:
                ans_end += 1
            if ans_end < L:
                # unmask: response content + <|im_end|> + \n
                end_idx = min(ans_end + 2, L)
                labels.view(-1)[ans_start:end_idx] = input_ids.view(-1)[ans_start:end_idx]
                pos = ans_end
        pos += 1
    return labels


# ---------------------------------------------------------------------------
# Coupled LR-HD sizing
# ---------------------------------------------------------------------------
def compute_coupled_sizes(orig_h, orig_w, lr_pixels, hr_scale,
                          patch_size=14, merge_size=2):
    """Compute coupled (HD, LR) dimensions for DAT training.

    HD dims are multiples of (patch_size * merge_size * hr_scale) so that
    HD / hr_scale yields LR dims that are multiples of (patch_size * merge_size).
    """
    hd_pixels = lr_pixels * hr_scale * hr_scale
    lr_factor = patch_size * merge_size           # 28
    hd_factor = lr_factor * hr_scale              # 84 for hr_scale=3

    aspect = orig_w / orig_h
    hd_h = max(hd_factor, round(math.sqrt(hd_pixels / aspect) / hd_factor) * hd_factor)
    hd_w = max(hd_factor, round(math.sqrt(hd_pixels * aspect) / hd_factor) * hd_factor)

    lr_h = hd_h // hr_scale
    lr_w = hd_w // hr_scale
    return hd_h, hd_w, lr_h, lr_w


# ---------------------------------------------------------------------------
# Coupled LR-HD Dataset for DAT
# ---------------------------------------------------------------------------
class Qwen2VLCoupledDATDataset(Dataset):
    """Dataset with coupled LR-HD image processing for DAT training.

    HD resolution is determined by the processor's default min/max pixels
    (i.e. whatever smart_resize picks for the original image).  LR is
    derived as HD / hr_scale in each spatial dimension, rounded to the
    nearest patch-merge factor (28).
    """

    PATCH_SIZE = 14
    MERGE_SIZE = 2
    FACTOR = PATCH_SIZE * MERGE_SIZE  # 28

    def __init__(self, data_path, processor, data_args, model_args,
                 model_max_length, coord_format="qwen2_vl"):
        super().__init__()
        rank0_print(f"Loading data from {data_path} (coupled LR-HD mode)...")
        self.list_data_dict = json.load(open(data_path, "r"))
        rank0_print(f"Loaded {len(self.list_data_dict)} samples.")

        self.processor = processor
        self.data_args = data_args
        self.model_max_length = model_max_length
        self.coord_format = coord_format

        self.hr_scale = model_args.dat_hr_scale

        rank0_print(
            f"Coupled LR-HD: hr_scale={self.hr_scale}, "
            f"HD pixels by processor default, LR = HD / {self.hr_scale ** 2}"
        )

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i):
        MAX_RETRIES = 10
        for attempt in range(MAX_RETRIES):
            try:
                return self._get_item(i)
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    logging.warning(f"Error loading sample {i}: {e}. Retrying...")
                    i = random.randint(0, len(self) - 1)
                else:
                    raise RuntimeError(f"Failed after {MAX_RETRIES} attempts: {e}")

    def _get_item(self, i):
        item = self.list_data_dict[i]
        messages, image_path = _build_messages(
            item, self.data_args.image_folder, self.data_args.system_message,
            coord_format=self.coord_format,
        )

        has_image = image_path is not None and os.path.exists(image_path)
        img = None
        if has_image:
            img = Image.open(image_path).convert("RGB")

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        inputs_hd = None
        if has_image and img is not None:
            # Step 1: HD — let processor pick resolution with its defaults
            inputs_hd = self.processor(
                images=[img],
                text=["<|im_start|>"],
                return_tensors="pt",
                padding=False,
            )

            # Step 2: derive LR dims from HD actual dims
            thw = inputs_hd["image_grid_thw"][0]
            hd_h = thw[1].item() * self.FACTOR
            hd_w = thw[2].item() * self.FACTOR
            lr_h = max(self.FACTOR, (hd_h // self.hr_scale // self.FACTOR) * self.FACTOR)
            lr_w = max(self.FACTOR, (hd_w // self.hr_scale // self.FACTOR) * self.FACTOR)
            lr_total = lr_h * lr_w

            # Step 3: LR — force processor to use derived resolution
            inputs = self.processor(
                images=[img],
                text=[text],
                return_tensors="pt",
                padding=False,
                min_pixels=lr_total,
                max_pixels=lr_total,
            )
        else:
            inputs = self.processor(
                text=[text],
                return_tensors="pt",
                padding=False,
            )

        input_ids = inputs["input_ids"]
        if input_ids.size(1) > self.model_max_length:
            input_ids = input_ids[:, :self.model_max_length]

        labels = _apply_label_mask(input_ids)

        result = {
            "input_ids": input_ids.squeeze(0),
            "labels": labels.squeeze(0),
        }

        if "pixel_values" in inputs:
            result["pixel_values"] = inputs["pixel_values"]
            result["image_grid_thw"] = inputs["image_grid_thw"]

        if inputs_hd is not None:
            result["pixel_values_hd"] = inputs_hd["pixel_values"]
            result["image_grid_thw_hd"] = inputs_hd["image_grid_thw"]

        return result


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class Qwen2VLSupervisedDataset(Dataset):
    """Dataset for Qwen2VL supervised fine-tuning (with optional DAT HD data)."""

    def __init__(self, data_path, processor, data_args, model_max_length,
                 use_dat=False, processor_hd=None, coord_format="qwen2_vl"):
        super().__init__()
        rank0_print(f"Loading data from {data_path}...")
        self.list_data_dict = json.load(open(data_path, "r"))
        rank0_print(f"Loaded {len(self.list_data_dict)} samples.")
        self.processor = processor
        self.data_args = data_args
        self.model_max_length = model_max_length
        self.use_dat = use_dat
        self.processor_hd = processor_hd
        self.coord_format = coord_format

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        MAX_RETRIES = 10
        for attempt in range(MAX_RETRIES):
            try:
                return self._get_item(i)
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    logging.warning(f"Error loading sample {i}: {e}. Retrying with random index...")
                    i = random.randint(0, len(self) - 1)
                else:
                    raise RuntimeError(f"Failed to load sample after {MAX_RETRIES} attempts: {e}")

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        item = self.list_data_dict[i]
        messages, image_path = _build_messages(
            item, self.data_args.image_folder, self.data_args.system_message,
            coord_format=self.coord_format,
        )

        has_image = image_path is not None and os.path.exists(image_path)
        images = []
        if has_image:
            img = Image.open(image_path).convert("RGB")
            images.append(img)

        # Build ChatML text via processor's apply_chat_template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # Process through Qwen2VL processor
        if images:
            inputs = self.processor(
                images=images,
                text=[text],
                return_tensors="pt",
                padding=False,
                min_pixels=self.data_args.min_pixels,
                max_pixels=self.data_args.max_pixels,
            )
        else:
            inputs = self.processor(
                text=[text],
                return_tensors="pt",
                padding=False,
            )

        input_ids = inputs["input_ids"]  # [1, seq_len]

        # Truncate to model_max_length
        if input_ids.size(1) > self.model_max_length:
            input_ids = input_ids[:, :self.model_max_length]

        # Create labels with masking
        labels = _apply_label_mask(input_ids)

        result = {
            "input_ids": input_ids.squeeze(0),      # [seq_len]
            "labels": labels.squeeze(0),             # [seq_len]
        }

        if "pixel_values" in inputs:
            result["pixel_values"] = inputs["pixel_values"]          # [N, 1176]
            result["image_grid_thw"] = inputs["image_grid_thw"]      # [num_images, 3]

        # --- HD processing for DAT ---
        if self.use_dat and has_image and self.processor_hd is not None:
            inputs_hd = self.processor_hd(
                images=[img],  # reuse already-opened PIL image
                text=["<|im_start|>"],  # minimal text to satisfy API
                return_tensors="pt",
                padding=False,
                min_pixels=self.data_args.hd_min_pixels,
                max_pixels=self.data_args.hd_max_pixels,
            )
            result["pixel_values_hd"] = inputs_hd["pixel_values"]
            result["image_grid_thw_hd"] = inputs_hd["image_grid_thw"]

        return result


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------
class Qwen2VLDataCollator:
    """Collate for Qwen2VL: pad tokens, concatenate pixel_values/image_grid_thw."""

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [inst["input_ids"] for inst in instances]
        labels = [inst["labels"] for inst in instances]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        attention_mask = input_ids.ne(self.pad_token_id)

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

        # Concatenate pixel_values and image_grid_thw across the batch
        pixel_values_list = [inst["pixel_values"] for inst in instances if "pixel_values" in inst]
        image_grid_thw_list = [inst["image_grid_thw"] for inst in instances if "image_grid_thw" in inst]

        if pixel_values_list:
            batch["pixel_values"] = torch.cat(pixel_values_list, dim=0)
            batch["image_grid_thw"] = torch.cat(image_grid_thw_list, dim=0)
        else:
            batch["pixel_values"] = None
            batch["image_grid_thw"] = None

        # HD pixel values for DAT
        pv_hd_list = [inst["pixel_values_hd"] for inst in instances if "pixel_values_hd" in inst]
        grid_hd_list = [inst["image_grid_thw_hd"] for inst in instances if "image_grid_thw_hd" in inst]

        if pv_hd_list:
            batch["pixel_values_hd"] = torch.cat(pv_hd_list, dim=0)
            batch["image_grid_thw_hd"] = torch.cat(grid_hd_list, dim=0)

        return batch


# ---------------------------------------------------------------------------
# Grouped-modality sampler (from LLaVA trainer)
# ---------------------------------------------------------------------------
def get_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(mb, key=lambda i: lengths[i], reverse=True) for mb in megabatches]

    def split_to_even_chunks(idxs, lens, num_chunks):
        if len(idxs) % num_chunks != 0:
            return [idxs[i::num_chunks] for i in range(num_chunks)]
        n = len(idxs) // num_chunks
        chunks = [[] for _ in range(num_chunks)]
        chunks_lengths = [0] * num_chunks
        for idx in idxs:
            shortest = chunks_lengths.index(min(chunks_lengths))
            chunks[shortest].append(idx)
            chunks_lengths[shortest] += lens[idx]
            if len(chunks[shortest]) == n:
                chunks_lengths[shortest] = float("inf")
        return chunks

    megabatches = [split_to_even_chunks(mb, lengths, world_size) for mb in megabatches]
    return [i for mb in megabatches for batch in mb for i in batch]


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    assert all(l != 0 for l in lengths)
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for mb in megabatches for i in mb]


class LengthGroupedSampler(torch.utils.data.Sampler):
    def __init__(self, batch_size, world_size, lengths, generator=None, group_by_modality=False):
        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(
                self.lengths, self.batch_size, self.world_size, generator=self.generator
            )
        else:
            indices = get_length_grouped_indices(
                self.lengths, self.batch_size, self.world_size, generator=self.generator
            )
        return iter(indices)


# ---------------------------------------------------------------------------
# Two-phase DAT warmup callback
# ---------------------------------------------------------------------------
class DATWarmupCallback(transformers.TrainerCallback):
    """Two-phase DAT training schedule.

    Phase 1 (steps 0 .. warmup_steps-1): only DAT parameters are trainable.
    Phase 2 (steps warmup_steps .. end):  DAT + LLM are trainable.
    ViT and connector stay frozen throughout.

    Trick: before Trainer creation, DAT + LLM are all set requires_grad=True
    so that create_optimizer() includes both in param groups.  Then on_train_begin
    freezes LLM (params stay in optimizer but get no gradients), and on_step_begin
    unfreezes LLM at the transition point.
    """

    def __init__(self, warmup_steps: int, dat_keys: list):
        self.warmup_steps = warmup_steps
        self.dat_keys = dat_keys
        self.phase1_active = True

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        # Phase 1: freeze LLM, keep only DAT trainable
        lang = _get_language_module(model)
        for p in lang.parameters():
            p.requires_grad = False
        model.lm_head.requires_grad_(False)
        # Ensure DAT params stay unfrozen
        for name, param in model.named_parameters():
            if any(k in name for k in self.dat_keys):
                param.requires_grad = True
        rank0_print(f"[DATWarmup] Phase 1: DAT-only for {self.warmup_steps} steps")
        print_trainable_parameters(model)

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        if not self.phase1_active or model is None:
            return
        if state.global_step >= self.warmup_steps:
            # Phase 2: unfreeze LLM
            lang = _get_language_module(model)
            for p in lang.parameters():
                p.requires_grad = True
            model.lm_head.requires_grad_(True)
            self.phase1_active = False
            rank0_print(f"[DATWarmup] Step {state.global_step}: Phase 2 — unfroze LLM + DAT")
            print_trainable_parameters(model)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class Qwen2VLTrainer(transformers.Trainer):

    def _inner_training_loop(self, *args, **kwargs):
        """Override to disable gradient checkpointing on frozen vision encoder.

        ZeRO-3 partitions all parameters (including frozen ones).  When gradient
        checkpointing is active on frozen vision blocks, the recompute pass sees
        partitioned (shape-[0]) tensors instead of the gathered ones from the
        original forward, causing a CheckpointError.  Disabling gradient
        checkpointing for frozen visual blocks avoids this entirely.
        """
        try:
            visual = _get_visual_module(self.model)
        except AttributeError:
            visual = None

        vis_frozen = visual is not None and not any(p.requires_grad for p in visual.parameters())

        if vis_frozen:
            _orig_gc_enable = self.model.gradient_checkpointing_enable

            def _patched_gc_enable(**kw):
                _orig_gc_enable(**kw)
                for module in visual.modules():
                    if hasattr(module, 'gradient_checkpointing'):
                        module.gradient_checkpointing = False
                rank0_print("[Trainer] Disabled gradient checkpointing for frozen visual encoder (ZeRO-3 compat)")

            self.model.gradient_checkpointing_enable = _patched_gc_enable

        result = super()._inner_training_loop(*args, **kwargs)

        if vis_frozen:
            self.model.gradient_checkpointing_enable = _orig_gc_enable

        return result

    def _get_train_sampler(self, train_dataset=None):
        dataset = train_dataset or self.train_dataset
        if dataset is None or not transformers.trainer.has_length(dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler(train_dataset)

    def create_optimizer(self):
        """Optimizer with lr_mapper pattern (Qwen3-VL / LLaVATrainer style)."""
        if self.optimizer is not None:
            return self.optimizer

        opt_model = self.model
        from transformers.trainer import get_parameter_names

        ALL_LAYERNORM_LAYERS = [torch.nn.LayerNorm, torch.nn.BatchNorm2d]
        decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]

        # Build lr_mapper: keyword -> lr
        # Order matters: more specific keywords first (merger before visual)
        lr_mapper = {}
        if getattr(self.args, 'mm_projector_lr', None) is not None:
            lr_mapper["merger"] = self.args.mm_projector_lr
        if getattr(self.args, 'vision_tower_lr', None) is not None:
            lr_mapper["visual"] = self.args.vision_tower_lr
        # Legacy: vision_lr applies to everything under "visual"
        if getattr(self.args, 'vision_lr', None) is not None and "visual" not in lr_mapper:
            lr_mapper["visual"] = self.args.vision_lr
        # DAT-specific LR
        if getattr(self.args, 'dat_lr', None) is not None:
            for key in DAT_KEYS_MATCH:
                lr_mapper[key] = self.args.dat_lr

        if len(lr_mapper) > 0:
            special_lr_parameters = [
                name for name, _ in opt_model.named_parameters()
                if any(module_keyword in name for module_keyword in lr_mapper)
            ]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if (n in decay_parameters and n not in special_lr_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if (n not in decay_parameters and n not in special_lr_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]
            assigned = set()
            for module_keyword, lr in lr_mapper.items():
                module_parameters = [
                    name for name, _ in opt_model.named_parameters()
                    if module_keyword in name and name not in assigned
                ]
                assigned.update(module_parameters)
                optimizer_grouped_parameters.extend([
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and n in module_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and n in module_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": lr,
                    },
                ])
        else:
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

        optimizer_cls, optimizer_kwargs = transformers.Trainer.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer


# ---------------------------------------------------------------------------
# Main train function
# ---------------------------------------------------------------------------
def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, Qwen2VLTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    compute_dtype = (
        torch.float16 if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    # ----- Model -----
    is_qwen2_5 = model_args.model_family == "qwen2_5_vl"
    dat_warmup_callback = None

    if model_args.use_dat:
        if is_qwen2_5:
            from llava.model.language_model.modeling_qwen2_5vl_dat import (
                convert_qwen2_5vl_to_dat,
                freeze_base_unfreeze_dat,
                DAT_KEYS_MATCH as _DAT_KEYS,
            )
        else:
            from llava.model.language_model.modeling_qwen2vl_dat import (
                convert_qwen2vl_to_dat,
                freeze_base_unfreeze_dat,
                DAT_KEYS_MATCH as _DAT_KEYS,
            )

        dat_extra_args = {
            'grid_size': model_args.dat_grid_size,
            'off_ksize': model_args.dat_off_ksize,
            'off_grps': model_args.dat_off_grps,
            'inter_size': model_args.dat_inter_size,
            'hr_scale': model_args.dat_hr_scale,
            'hd_proj': model_args.dat_hd_proj,
            'layers': model_args.dat_layers,
            'use_intention_branch': model_args.dat_use_intention_branch,
            'intention_as_gate': model_args.dat_intention_as_gate,
        }

        rank0_print(f"Loading DAT model ({model_args.model_family}) from {model_args.model_name_or_path}...")
        rank0_print(f"DAT config: {dat_extra_args}")
        convert_fn = convert_qwen2_5vl_to_dat if is_qwen2_5 else convert_qwen2vl_to_dat
        model = convert_fn(
            model_args.model_name_or_path,
            dat_extra_args=dat_extra_args,
            torch_dtype=compute_dtype,
        )

        # Apply freeze strategy
        if model_args.dat_warmup_steps > 0:
            # Two-phase: DAT+LLM trainable initially (for optimizer creation),
            # ViT + connector frozen. Callback handles phase transition.
            rank0_print(f"Two-phase training: DAT-only for {model_args.dat_warmup_steps} steps, then DAT+LLM")
            visual = _get_visual_module(model)
            for p in visual.parameters():
                p.requires_grad = False
            lang = _get_language_module(model)
            for p in lang.parameters():
                p.requires_grad = True
            model.lm_head.requires_grad_(True)
            for name, param in model.named_parameters():
                if any(k in name for k in _DAT_KEYS):
                    param.requires_grad = True
            dat_warmup_callback = DATWarmupCallback(
                warmup_steps=model_args.dat_warmup_steps,
                dat_keys=_DAT_KEYS,
            )
        elif model_args.dat_freeze_base:
            rank0_print("Freezing base model, unfreezing DAT params only...")
            freeze_base_unfreeze_dat(model)
        else:
            set_model(model, model_args)
            # Ensure DAT params are always trainable
            for name, param in model.named_parameters():
                if any(k in name for k in _DAT_KEYS):
                    param.requires_grad = True

        # Validate dat_layers length
        if model_args.dat_layers:
            _text_cfg = getattr(model.config, 'text_config', None)
            num_layers = getattr(model.config, 'num_hidden_layers', None) or \
                         (_text_cfg.num_hidden_layers if _text_cfg else None)
            if num_layers and len(model_args.dat_layers) != num_layers:
                raise ValueError(
                    f"dat_layers length ({len(model_args.dat_layers)}) "
                    f"!= num_hidden_layers ({num_layers})"
                )
    else:
        rank0_print(f"Loading model ({model_args.model_family}) from {model_args.model_name_or_path}...")
        if is_qwen2_5:
            from transformers import Qwen2_5_VLForConditionalGeneration
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                torch_dtype=compute_dtype,
                attn_implementation="sdpa",
            )
        else:
            from transformers import Qwen2VLForConditionalGeneration
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                torch_dtype=compute_dtype,
                attn_implementation="sdpa",
            )

        # Apply fine-grained freeze control
        set_model(model, model_args)

    # ----- Processor -----
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        use_fast=False,
        min_pixels=data_args.min_pixels,
        max_pixels=data_args.max_pixels,
    )
    processor.tokenizer.model_max_length = training_args.model_max_length
    processor.tokenizer.padding_side = "right"

    # Align model config & generation_config with tokenizer special tokens
    # so the Trainer doesn't auto-override and emit a noisy warning.
    # Qwen2-VL: eos/pad on model.config directly
    # Qwen2.5-VL: eos/pad on model.config.text_config (nested config)
    _tok = processor.tokenizer
    _cfg = model.config
    _text_cfg = getattr(_cfg, 'text_config', None)  # Qwen2.5-VL only
    for _attr in ("pad_token_id", "bos_token_id", "eos_token_id"):
        _val = getattr(_tok, _attr, None)
        if hasattr(_cfg, _attr) and getattr(_cfg, _attr, None) != _val:
            setattr(_cfg, _attr, _val)
        if _text_cfg is not None and hasattr(_text_cfg, _attr) and getattr(_text_cfg, _attr, None) != _val:
            setattr(_text_cfg, _attr, _val)
        if hasattr(model, "generation_config") and getattr(model.generation_config, _attr, None) != _val:
            setattr(model.generation_config, _attr, _val)

    rank0_print(f"Model loaded. Vocab size: {len(processor.tokenizer)}")
    # print_trainable_parameters(model)

    # ----- Gradient checkpointing -----
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # ----- LoRA -----
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        if training_args.bf16:
            model.to(torch.bfloat16)
        if training_args.fp16:
            model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)
        rank0_print(f"LoRA target modules: {lora_config.target_modules}")

    # ----- HD Processor for DAT (only for uncoupled mode) -----
    processor_hd = None
    if model_args.use_dat and not data_args.coupled_lr_hd:
        processor_hd = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
            use_fast=False,
            min_pixels=data_args.hd_min_pixels,
            max_pixels=data_args.hd_max_pixels,
        )

    # ----- Dataset -----
    coord_format = model_args.model_family  # "qwen2_vl" or "qwen2_5_vl"
    rank0_print(f"Bbox coordinate format: {coord_format}")
    if model_args.use_dat and data_args.coupled_lr_hd:
        train_dataset = Qwen2VLCoupledDATDataset(
            data_path=data_args.data_path,
            processor=processor,
            data_args=data_args,
            model_args=model_args,
            model_max_length=training_args.model_max_length,
            coord_format=coord_format,
        )
    else:
        train_dataset = Qwen2VLSupervisedDataset(
            data_path=data_args.data_path,
            processor=processor,
            data_args=data_args,
            model_max_length=training_args.model_max_length,
            use_dat=model_args.use_dat,
            processor_hd=processor_hd,
            coord_format=coord_format,
        )
    data_collator = Qwen2VLDataCollator(pad_token_id=ENDOFTEXT_ID)

    # ----- Trainer -----
    callbacks = []
    if dat_warmup_callback is not None:
        callbacks.append(dat_warmup_callback)

    trainer = Qwen2VLTrainer(
        model=model,
        processing_class=processor.tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
        callbacks=callbacks if callbacks else None,
    )

    # ----- Train -----
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    # ----- Save -----
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), "none"
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    # Save processor (Qwen3-VL pattern)
    if training_args.local_rank in (0, -1):
        processor.save_pretrained(training_args.output_dir)



if __name__ == "__main__":
    train()
