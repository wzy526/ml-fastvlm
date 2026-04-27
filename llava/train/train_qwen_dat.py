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
import torch.nn.functional as F
import transformers
from torch.utils.data import Dataset, Sampler
from PIL import Image

try:
    import wandb
except ImportError:
    wandb = None


# Module-level guard so that ``wandb.define_metric`` is called at most once per
# process even if multiple callbacks try to register it independently.
_WANDB_STEP_METRIC_DEFINED = False


def _define_wandb_step_metric() -> None:
    """Idempotently register ``train/global_step`` as the common step metric.

    Without this, HF's own ``WandbCallback.on_log`` calls ``wandb.log`` *without*
    a ``step=`` argument, which makes wandb auto-increment its internal
    ``run.step`` on every log. Our custom DAT callbacks, on the other hand,
    call ``wandb.log(..., step=state.global_step)`` using HF's global_step.
    The two step sequences drift apart — one optimization step triggers
    ``on_log`` 2-3 times (``compute_loss`` self.log + ``_maybe_log_save_evaluate``
    self.log + sub-step callbacks), so wandb's internal step advances 2-3× per
    global_step. Eventually our callbacks try to log at a ``step`` below
    ``run.step``, producing::

        wandb: WARNING Tried to log to step 1 that is less than the current step 5.

    The wandb-recommended fix (see https://wandb.me/define-metric) is to let
    wandb keep auto-incrementing its internal step *and* tell it to use a
    specific data field as the X-axis for every metric.
    """
    global _WANDB_STEP_METRIC_DEFINED
    if _WANDB_STEP_METRIC_DEFINED:
        return
    if wandb is None or wandb.run is None:
        return
    try:
        wandb.define_metric("train/global_step")
        wandb.define_metric("*", step_metric="train/global_step")
        _WANDB_STEP_METRIC_DEFINED = True
    except Exception:
        # Best-effort; if it fails we just keep the wandb auto step.
        pass


IGNORE_INDEX = -100

local_rank = None

# DAT parameter patterns (must match modeling_qwen2vl_dat.py)
DAT_KEYS_MATCH = [
    'conv_lr_dw', 'ln_1', 'conv_lr_proj', 'proj_intention',
    'ln_2', 'conv_off_proj', 'k_proj_hd', 'v_proj_hd',
    'hd_gate',
]


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


# ---------------------------------------------------------------------------
# Model unwrapping helper
# ---------------------------------------------------------------------------
def _unwrap_to_base_dat_model(model):
    """Peel back DDP / DeepSpeed / PEFT / LoRA wrappers to reach the raw
    ``Qwen2VLForConditionalGeneration`` / ``Qwen2_5_VLForConditionalGenerationDAT``
    instance that hosts the ``model.language_model.layers`` tree and receives
    attribute writes such as ``_batch_image_paths`` / ``_dat_vis_input_ids``.

    Handles (in order) the typical wrapping chain::

        DistributedDataParallel / DeepSpeedEngine   →  .module
        PeftModel (peft)                            →  .base_model
        LoraModel (peft)                            →  .model

    Stops when the chain stabilises (no more recognised wrappers).  Always
    returns a non-None module (the input itself if no wrapper found).
    """
    if model is None:
        return None
    seen = set()
    cur = model
    while id(cur) not in seen:
        seen.add(id(cur))
        # DDP / DeepSpeed / Accelerator wrap the module as .module
        if hasattr(cur, 'module') and isinstance(
            getattr(cur, 'module', None), torch.nn.Module,
        ) and type(cur).__name__ in (
            'DistributedDataParallel', 'DataParallel', 'DeepSpeedEngine',
            'FullyShardedDataParallel',
        ):
            cur = cur.module
            continue
        # PEFT's PeftModel  →  .base_model  (which is a LoraModel / LycorisModel / etc.)
        if type(cur).__name__.startswith('Peft') and hasattr(cur, 'base_model'):
            cur = cur.base_model
            continue
        # peft.tuners.lora.LoraModel (and siblings) keep the real model at .model
        if type(cur).__name__ in (
            'LoraModel', 'LycorisModel', 'AdaLoraModel', 'IA3Model',
            'BoneModel', 'OFTModel',
        ) and hasattr(cur, 'model'):
            cur = cur.model
            continue
        break
    return cur


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
    dat_use_spatial_attn_guide: bool = field(
        default=True,
        metadata={"help": "Enable Q_intention x Q_lr spatial attention guidance when predicting DAT offsets."}
    )
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
    dat_hd_gate_init: Optional[float] = field(
        default=None,
        metadata={"help": "Learnable HD gate init value (nn.Parameter). Applied as "
                  "lse2 += log(sigmoid(hd_gate)). Use e.g. -10.0 so sigmoid ≈ 0 "
                  "at the start, preventing HD from dominating before offset modules "
                  "learn meaningful sampling. None = disabled."}
    )
    dat_fused_vit: bool = field(
        default=False,
        metadata={"help": "Fuse LR+HD into one ViT call (saves kernel launches; "
                  "costs ~2× activation memory). Default False = separate ViT calls."}
    )
    dat_shared_vit: bool = field(
        default=False,
        metadata={"help": "Shared-ViT path: single HD ViT call; LR tokens are "
                  "adaptive-pooled from HD features (no LR ViT at all). ViT cost "
                  "≈ HD-only baseline. Overrides dat_fused_vit when True. "
                  "Semantics differ from the LR-ViT baseline → requires retraining."}
    )
    dat_manual_attn: bool = field(
        default=False,
        metadata={"help": "Use manual mask-based attention implementation instead of two-pass LSE merge."}
    )


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to JSON training data."})
    image_folder: Optional[str] = field(default=None)
    system_message: str = field(default="You are a helpful assistant.")
    hd_max_pixels: int = field(
        default=3211264,
        metadata={"help": "[Legacy LR-first mode] Hard cap on HD pixel count "
                  "(default ~4096 HD tokens). Ignored when use_hr_first_resize=True."}
    )
    coupled_lr_hd: bool = field(
        default=True,
        metadata={"help": "DEPRECATED: coupled LR-HD is now the default for DAT. Kept for backward compat."}
    )
    # [Legacy LR-first mode] LR pixel range used by the LR processor.
    lr_min_pixels: int = field(default=200704)    # 256*28*28
    lr_max_pixels: int = field(default=501760)    # 640*28*28 (~640 tokens)

    # ---- HR-first resize (aligned with lmms-eval qwen2_5_dat_vl.py inference) ----
    # When True, HR is the anchor: HR processor smart_resizes to [hr_min, hr_max] first,
    # then LR_thw is derived as floor(HR_thw / dat_hr_scale) aligned to spatial_merge=2,
    # and the LR processor is forced to that exact size (min_pixels=max_pixels=lr_pixels).
    # This matches the test-time semantics so shared-ViT / DAT geometry is identical
    # between training and inference.
    use_hr_first_resize: bool = field(
        default=False,
        metadata={"help": "If True, use HR-anchored resize: LR=HR/hr_scale strict ratio. "
                  "Recommended for shared_vit and aligned with lmms-eval default."}
    )
    hr_min_pixels: int = field(
        default=28224,
        metadata={"help": "Minimum HR pixel count when use_hr_first_resize=True. "
                  "Default 28224 matches lmms-eval qwen2_5_dat_vl.py default."}
    )
    hr_max_pixels: int = field(
        default=9031680,
        metadata={"help": "Maximum HR pixel count when use_hr_first_resize=True. "
                  "Default 9031680 matches lmms-eval qwen2_5_dat_vl.py default; "
                  "yields LR<=1280 tokens at hr_scale=3 (LR<=2880 at hr_scale=2)."}
    )

    
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
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_layers: str = field(
        default="all",
        metadata={"help": "LoRA target scope: 'dat' (QKVO in DAT layers only) "
                  "or 'all' (QKVO in all decoder layers)"}
    )
    lora_lr: Optional[float] = field(
        default=None,
        metadata={"help": "Separate learning rate for LoRA adapter parameters"}
    )
    visualization_every_n_steps: int = 10
    # ---- Online layer-wise Knowledge Distillation (KD) ----
    kd_on: bool = field(
        default=False,
        metadata={"help": "Enable online layer-wise KD that aligns student's per-layer "
                  "output logits distribution to a frozen base VLM snapshot. "
                  "Teacher is a pure base Qwen2VL / Qwen2.5VL (no DAT, no LoRA)."}
    )
    kd_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight of the aggregated layer-wise KL distillation loss."}
    )
    kd_temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature used for softmax/log_softmax when computing KD KL divergence."}
    )
    kd_layer_stride: int = field(
        default=1,
        metadata={"help": "Deprecated for current KD implementation. Kept for CLI "
                  "backward compatibility; final-logit KD always uses the last layer."}
    )
    kd_last_layer_only: bool = field(
        default=True,
        metadata={"help": "Deprecated for current KD implementation. Kept for CLI "
                  "backward compatibility; final-logit KD always uses the last layer."}
    )
    kd_base_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the pure base VLM checkpoint used as KD teacher. "
                  "If None, re-uses model_name_or_path. Loaded fresh without DAT/LoRA wrapping."}
    )
    kd_teacher_min_pixels: Optional[int] = field(
        default=None,
        metadata={"help": "min_pixels used when feeding images to the KD teacher. "
                  "Default None -> use processor default (base VLM recommended)."}
    )
    kd_teacher_max_pixels: Optional[int] = field(
        default=None,
        metadata={"help": "max_pixels used when feeding images to the KD teacher. "
                  "Default None -> use processor default (base VLM recommended). "
                  "Teacher typically takes the original HR resolution, while the student's "
                  "vision pathway takes LR (lr_min_pixels/lr_max_pixels)."}
    )


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


def get_lora_target_modules(dat_layers_str, target_layers="all"):
    """Build regex pattern for PEFT LoRA targeting QKVO projections.

    Args:
        dat_layers_str: Layer type string e.g. 'DLLLLLDLLLLL...'
        target_layers: 'dat' applies LoRA only to DAT layer QKVO,
            'all' applies LoRA to every decoder layer's QKVO.

    Returns:
        Regex pattern string for ``LoraConfig(target_modules=...)``.
    """
    qkvo = r"(q_proj|k_proj|v_proj|o_proj)"
    if target_layers == "dat" and dat_layers_str:
        dat_indices = [str(i) for i, c in enumerate(dat_layers_str) if c == 'D']
        layer_pattern = "|".join(dat_indices)
        return rf"model\.language_model\.layers\.({layer_pattern})\.self_attn\.{qkvo}"
    else:
        return rf"model\.language_model\.layers\.\d+\.self_attn\.{qkvo}"


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


def _set_merger_trainable_after_lora(model, trainable: bool) -> int:
    """Set ``visual.merger`` trainability on a PEFT-wrapped model.

    In LoRA mode, PEFT freezes all base parameters. We selectively re-enable
    merger params when ``tune_mm_mlp=True`` so projector tuning still works.
    """
    touched = 0
    for name, param in model.named_parameters():
        if "visual.merger" in name:
            param.requires_grad = trainable
            touched += 1
    return touched


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
                  "model.layers": 0, "lm_head": 0, "DAT": 0, "LoRA": 0, "other": 0}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        n = param.numel()
        if any(k in name for k in DAT_KEYS_MATCH):
            components["DAT"] += n
        elif "lora_" in name:
            components["LoRA"] += n
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
        if flat[pos].item() == ASSISTANT_TOKEN_ID and pos > 0 and flat[pos - 1].item() == IM_START_TOKEN_ID:
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
# HR-first geometry helpers (aligned with lmms-eval qwen2_5_dat_vl.py)
# ---------------------------------------------------------------------------
PATCH_SIZE = 14
SPATIAL_MERGE = 2


def _derive_lr_geometry_from_hr_thw(hd_h_patches: int, hd_w_patches: int, hr_scale: int):
    """Derive LR (h, w) in 14-px patches as floor(HR / hr_scale) aligned to SPATIAL_MERGE.

    Output guarantees:
        lr_h_patches and lr_w_patches are multiples of SPATIAL_MERGE (= 2),
        so that after spatial-merge they yield integer merged-token counts and
        ``adaptive_avg_pool2d`` in shared-ViT path has a well-defined target.

    Returns (lr_h_patches, lr_w_patches, lr_pixels).
    ``lr_pixels = lr_h_patches * lr_w_patches * PATCH_SIZE * PATCH_SIZE`` is
    always a multiple of (SPATIAL_MERGE * PATCH_SIZE)^2 = 28^2 so the
    Qwen2.5-VL processor's smart_resize will round-trip exactly.
    """
    lr_h_patches = max(SPATIAL_MERGE, (hd_h_patches // hr_scale // SPATIAL_MERGE) * SPATIAL_MERGE)
    lr_w_patches = max(SPATIAL_MERGE, (hd_w_patches // hr_scale // SPATIAL_MERGE) * SPATIAL_MERGE)
    lr_pixels = lr_h_patches * lr_w_patches * PATCH_SIZE * PATCH_SIZE
    return lr_h_patches, lr_w_patches, lr_pixels


def hr_first_lr_resize(img: Image.Image, processor, hr_scale: int,
                       hr_min_pixels: int, hr_max_pixels: int):
    """HR-anchored LR/HR pair, identical semantics to lmms-eval inference.

    Pipeline:
        1. Run HR processor on the original PIL image with ``[hr_min, hr_max]``
           to get ``hd_thw`` (in 14-px patches).
        2. Derive ``lr_h_patches, lr_w_patches = floor(hd_thw / hr_scale)``,
           snapped to SPATIAL_MERGE multiples.
        3. Compute ``lr_pixels = lr_h_patches * lr_w_patches * PATCH_SIZE^2``;
           since this is exactly a multiple of ``(SPATIAL_MERGE * PATCH_SIZE)^2 = 28^2``,
           the processor's ``smart_resize`` will round-trip to the exact target
           size when called with ``min_pixels = max_pixels = lr_pixels``.

    Critical: we do **not** pre-resize the PIL image ourselves. The image
    processor's ``smart_resize`` uses ``BICUBIC`` interpolation by default
    (PILImageResampling.BICUBIC), which is what base Qwen2.5-VL was
    pretrained on. Pre-resizing here with PIL/LANCZOS would produce a
    sharper-than-pretrained signal and shift the ViT input distribution
    (loss climbs from ~0.2 to ~0.7 when this happens).

    The caller is expected to feed ``(img, min_pixels=lr_pixels,
    max_pixels=lr_pixels)`` directly to the processor for the LR forward,
    and to use ``inputs_hr`` (returned here) directly as the HD input.

    Returns (lr_pixels, inputs_hr, hd_thw).
    """
    inputs_hr = processor(
        images=[img], text=["<|im_start|>"],
        return_tensors="pt", padding=False,
        min_pixels=hr_min_pixels, max_pixels=hr_max_pixels,
    )
    hd_thw = inputs_hr["image_grid_thw"][0]
    hd_h_patches = int(hd_thw[1].item())
    hd_w_patches = int(hd_thw[2].item())

    _, _, lr_pixels = _derive_lr_geometry_from_hr_thw(
        hd_h_patches, hd_w_patches, hr_scale,
    )
    return lr_pixels, inputs_hr, hd_thw


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
                 model_max_length, coord_format="qwen2_vl",
                 kd_enabled: bool = False,
                 kd_teacher_min_pixels: Optional[int] = None,
                 kd_teacher_max_pixels: Optional[int] = None):
        super().__init__()
        rank0_print(f"Loading data from {data_path} (coupled LR-HD mode)...")
        self.list_data_dict = json.load(open(data_path, "r"))
        rank0_print(f"Loaded {len(self.list_data_dict)} samples.")

        self.processor = processor
        self.data_args = data_args
        self.model_max_length = model_max_length
        self.coord_format = coord_format

        self.hr_scale = model_args.dat_hr_scale
        self.hd_max_pixels = data_args.hd_max_pixels

        # Knowledge distillation teacher image resolution configuration
        self.kd_enabled = kd_enabled
        self.kd_teacher_min_pixels = kd_teacher_min_pixels
        self.kd_teacher_max_pixels = kd_teacher_max_pixels

        rank0_print(
            f"Coupled LR-HD: hr_scale={self.hr_scale}, "
            f"LR pixels=[{data_args.lr_min_pixels}, {data_args.lr_max_pixels}], "
            f"HD = LR * {self.hr_scale}² "
            f"(capped at orig_pixels and hd_max_pixels={self.hd_max_pixels})"
        )
        if self.kd_enabled:
            rank0_print(
                f"[KD-data] Teacher image resolution: min_pixels={self.kd_teacher_min_pixels}, "
                f"max_pixels={self.kd_teacher_max_pixels} (None => processor default)"
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
            if getattr(self.data_args, "use_hr_first_resize", False):
                # ---- HR-first mode (matches lmms-eval inference) ---------------
                # HR drives the geometry: we smart_resize the original image to
                # [hr_min, hr_max], then derive LR_thw = floor(HR_thw / hr_scale)
                # and force the LR processor to that exact size. ``shared_vit``
                # then sees the same pool ratio at train and at test time.
                # Note: pass the original PIL to the LR processor (NOT a
                # pre-resized one) so smart_resize uses BICUBIC matching
                # base Qwen2.5-VL pretraining. See hr_first_lr_resize docstring.
                lr_pixels, inputs_hd, _hd_thw = hr_first_lr_resize(
                    img, self.processor, self.hr_scale,
                    self.data_args.hr_min_pixels, self.data_args.hr_max_pixels,
                )
                inputs = self.processor(
                    images=[img], text=[text],
                    return_tensors="pt", padding=False,
                    min_pixels=lr_pixels, max_pixels=lr_pixels,
                )
            else:
                # ---- Legacy LR-first mode (kept for backward compat) -----------
                # Step 1: LR — processor 默认（不覆盖 min/max）
                inputs = self.processor(
                    images=[img],
                    text=[text],
                    return_tensors="pt",
                    padding=False,
                    min_pixels=self.data_args.lr_min_pixels,
                    max_pixels=self.data_args.lr_max_pixels,
                )

                # Step 2: HD — 尽可能高，但有上限
                orig_pixels = img.width * img.height
                thw = inputs["image_grid_thw"][0]
                lr_h = thw[1].item() * self.FACTOR
                lr_w = thw[2].item() * self.FACTOR
                lr_pixels = lr_h * lr_w

                # HD 目标：LR 的 hr_scale² 倍，但不超过原图，也不超过显存上限
                hd_target = lr_pixels * (self.hr_scale ** 2)
                hd_target = min(hd_target, orig_pixels)     # 不超过原图
                hd_target = min(hd_target, self.hd_max_pixels)  # 显存上限

                # 保持宽高比，对齐到 FACTOR
                aspect = img.width / img.height
                hd_h = int(math.sqrt(hd_target / aspect))
                hd_w = int(hd_h * aspect)
                hd_h = max(self.FACTOR, (hd_h // self.FACTOR) * self.FACTOR)
                hd_w = max(self.FACTOR, (hd_w // self.FACTOR) * self.FACTOR)
                hd_total = hd_h * hd_w

                # Step 3: HD processing
                inputs_hd = self.processor(
                    images=[img],
                    text=["<|im_start|>"],
                    return_tensors="pt",
                    padding=False,
                    min_pixels=hd_total,
                    max_pixels=hd_total,
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
            "image_path": image_path,
        }

        if "pixel_values" in inputs:
            result["pixel_values"] = inputs["pixel_values"]
            result["image_grid_thw"] = inputs["image_grid_thw"]

        if inputs_hd is not None:
            result["pixel_values_hd"] = inputs_hd["pixel_values"]
            result["image_grid_thw_hd"] = inputs_hd["image_grid_thw"]

        # ---- Optional KD teacher input: higher resolution for the pure base VLM ----
        # Teacher typically sees the full HR image (processor default min/max pixels),
        # while the student LR pathway uses lr_min_pixels/lr_max_pixels and relies on
        # DAT HD features for detail. We reuse the same chat-template text so the
        # assistant-answer token content (and therefore labels != IGNORE_INDEX count)
        # is identical on both sides -- only their absolute positions differ because
        # the two sides end up with different image_pad counts.
        if self.kd_enabled:
            teacher_kwargs = {
                "text": [text],
                "return_tensors": "pt",
                "padding": False,
            }
            if has_image and img is not None:
                teacher_kwargs["images"] = [img]
                if self.kd_teacher_min_pixels is not None:
                    teacher_kwargs["min_pixels"] = self.kd_teacher_min_pixels
                if self.kd_teacher_max_pixels is not None:
                    teacher_kwargs["max_pixels"] = self.kd_teacher_max_pixels

            inputs_teacher = self.processor(**teacher_kwargs)
            t_ids = inputs_teacher["input_ids"]
            if t_ids.size(1) > self.model_max_length:
                t_ids = t_ids[:, :self.model_max_length]
            t_labels = _apply_label_mask(t_ids)

            result["input_ids_teacher"] = t_ids.squeeze(0)
            result["labels_teacher"] = t_labels.squeeze(0)
            if "pixel_values" in inputs_teacher:
                result["pixel_values_teacher"] = inputs_teacher["pixel_values"]
                result["image_grid_thw_teacher"] = inputs_teacher["image_grid_thw"]

        return result


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class Qwen2VLSupervisedDataset(Dataset):
    """Dataset for Qwen2VL/Qwen2.5VL supervised fine-tuning (baseline, no DAT)."""

    def __init__(self, data_path, processor, data_args, model_max_length,
                 coord_format="qwen2_vl",
                 kd_enabled: bool = False,
                 kd_teacher_min_pixels: Optional[int] = None,
                 kd_teacher_max_pixels: Optional[int] = None):
        super().__init__()
        rank0_print(f"Loading data from {data_path}...")
        self.list_data_dict = json.load(open(data_path, "r"))
        rank0_print(f"Loaded {len(self.list_data_dict)} samples.")
        self.processor = processor
        self.data_args = data_args
        self.model_max_length = model_max_length
        self.coord_format = coord_format
        self.kd_enabled = kd_enabled
        self.kd_teacher_min_pixels = kd_teacher_min_pixels
        self.kd_teacher_max_pixels = kd_teacher_max_pixels

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

        if images:
            # In non-DAT baseline, LR == HR (LLM sees the full-resolution view).
            # When use_hr_first_resize=True, we use [hr_min, hr_max] so the
            # baseline sees the same resolution band as the DAT student would
            # encode at HR-first inference time (single-tower fairness).
            if getattr(self.data_args, "use_hr_first_resize", False):
                _min_p = self.data_args.hr_min_pixels
                _max_p = self.data_args.hr_max_pixels
            else:
                _min_p = self.data_args.lr_min_pixels
                _max_p = self.data_args.lr_max_pixels
            inputs = self.processor(
                images=images,
                text=[text],
                return_tensors="pt",
                padding=False,
                min_pixels=_min_p,
                max_pixels=_max_p,
            )
        else:
            inputs = self.processor(
                text=[text],
                return_tensors="pt",
                padding=False,
            )

        input_ids = inputs["input_ids"]  # [1, seq_len]

        if input_ids.size(1) > self.model_max_length:
            input_ids = input_ids[:, :self.model_max_length]

        labels = _apply_label_mask(input_ids)

        result = {
            "input_ids": input_ids.squeeze(0),
            "labels": labels.squeeze(0),
            "image_path": image_path,
        }

        if "pixel_values" in inputs:
            result["pixel_values"] = inputs["pixel_values"]
            result["image_grid_thw"] = inputs["image_grid_thw"]

        # ---- Optional KD teacher input (same text, possibly different image resolution) ----
        if self.kd_enabled:
            teacher_kwargs = {
                "text": [text],
                "return_tensors": "pt",
                "padding": False,
            }
            if has_image and images:
                teacher_kwargs["images"] = images
                if self.kd_teacher_min_pixels is not None:
                    teacher_kwargs["min_pixels"] = self.kd_teacher_min_pixels
                if self.kd_teacher_max_pixels is not None:
                    teacher_kwargs["max_pixels"] = self.kd_teacher_max_pixels

            inputs_teacher = self.processor(**teacher_kwargs)
            t_ids = inputs_teacher["input_ids"]
            if t_ids.size(1) > self.model_max_length:
                t_ids = t_ids[:, :self.model_max_length]
            t_labels = _apply_label_mask(t_ids)

            result["input_ids_teacher"] = t_ids.squeeze(0)
            result["labels_teacher"] = t_labels.squeeze(0)
            if "pixel_values" in inputs_teacher:
                result["pixel_values_teacher"] = inputs_teacher["pixel_values"]
                result["image_grid_thw_teacher"] = inputs_teacher["image_grid_thw"]

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

        # Image paths for visualization (list of str, not a tensor)
        image_paths = [inst.get("image_path") for inst in instances]
        if any(p is not None for p in image_paths):
            batch["image_paths"] = image_paths

        # HD pixel values for DAT
        pv_hd_list = [inst["pixel_values_hd"] for inst in instances if "pixel_values_hd" in inst]
        grid_hd_list = [inst["image_grid_thw_hd"] for inst in instances if "image_grid_thw_hd" in inst]

        if pv_hd_list:
            batch["pixel_values_hd"] = torch.cat(pv_hd_list, dim=0)
            batch["image_grid_thw_hd"] = torch.cat(grid_hd_list, dim=0)

        # ---- KD teacher: separate inputs with possibly different image resolution ----
        has_teacher = any("input_ids_teacher" in inst for inst in instances)
        if has_teacher:
            input_ids_t = [inst["input_ids_teacher"] for inst in instances]
            labels_t = [inst["labels_teacher"] for inst in instances]
            input_ids_t = torch.nn.utils.rnn.pad_sequence(
                input_ids_t, batch_first=True, padding_value=self.pad_token_id
            )
            labels_t = torch.nn.utils.rnn.pad_sequence(
                labels_t, batch_first=True, padding_value=IGNORE_INDEX
            )
            attention_mask_t = input_ids_t.ne(self.pad_token_id)
            batch["input_ids_teacher"] = input_ids_t
            batch["labels_teacher"] = labels_t
            batch["attention_mask_teacher"] = attention_mask_t

            pv_t_list = [inst["pixel_values_teacher"] for inst in instances if "pixel_values_teacher" in inst]
            gt_t_list = [inst["image_grid_thw_teacher"] for inst in instances if "image_grid_thw_teacher" in inst]
            if pv_t_list:
                batch["pixel_values_teacher"] = torch.cat(pv_t_list, dim=0)
                batch["image_grid_thw_teacher"] = torch.cat(gt_t_list, dim=0)
            else:
                batch["pixel_values_teacher"] = None
                batch["image_grid_thw_teacher"] = None

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
# WandB DAT Monitor Callback
# ---------------------------------------------------------------------------
class WandbDATMonitorCallback(transformers.TrainerCallback):
    """Monitors DAT-specific grad norms, weight norms, and actual LR.

    Grad norm is captured via param.register_hook, which fires during the
    backward pass—before DeepSpeed's engine.step() internally calls zero_grad.
    This is the only reliable capture point under ZeRO-2/3.

    Per-micro-step norms are buffered and averaged over the logging interval,
    then reported to WandB and the console at every on_log call.

    Also collects per-forward DAT diagnostics:
      - offset mean / std (deformable sampling offsets)
      - gate value mean / std (intention_as_gate sigmoid output)
    These are stored on each DAT attention module during forward and harvested
    at on_step_end.
    """

    KVHD_KEYS = ('k_proj_hd', 'v_proj_hd')

    def __init__(self, use_kvhd: bool = False):
        self._model     = None
        self._optimizer = None
        self._hooks     = []
        self._step_sq   = 0.0
        self._step_cnt  = 0
        self._norm_buf  = []
        # Buffers for forward-pass diagnostics (flushed at on_log)
        self._offset_mean_buf = []
        self._offset_std_buf  = []
        self._gate_mean_buf   = []
        self._gate_std_buf    = []
        self._hd_gate_buf     = []
        # KVHD-specific gradient monitoring (only when hd_proj is enabled)
        self._use_kvhd       = use_kvhd
        self._kvhd_step_sq   = 0.0
        self._kvhd_step_cnt  = 0
        self._kvhd_norm_buf  = []
        # Trainer-captured grad norms (after backward, before zero_grad).
        # Reliable fallback when register_hook silently fails under PEFT/DDP.
        self._trainer_grad_buf      = []
        self._trainer_kvhd_grad_buf = []

    # ------------------------------------------------------------------
    def on_train_begin(self, args, state, control, model=None, optimizer=None, **kwargs):
        if model is not None:
            self._model = model
            self._register_hooks(model)
        if optimizer is not None:
            self._optimizer = optimizer

    def _register_hooks(self, model):
        """Attach a backward hook to every DAT parameter."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        for name, param in model.named_parameters():
            if any(k in name for k in DAT_KEYS_MATCH) and param.requires_grad:
                is_kvhd = any(k in name for k in self.KVHD_KEYS)
                def _h(grad, _self=self, _is_kvhd=is_kvhd):
                    if local_rank == 0:
                        gnorm_sq = grad.detach().norm(2).item() ** 2
                        _self._step_sq  += gnorm_sq
                        _self._step_cnt += 1
                        if _is_kvhd and _self._use_kvhd:
                            _self._kvhd_step_sq  += gnorm_sq
                            _self._kvhd_step_cnt += 1
                self._hooks.append(param.register_hook(_h))
        rank0_print(f"[DATMonitor] Registered {len(self._hooks)} backward hooks on DAT params")

    def capture_grad_norms(self, model):
        """Capture DAT grad norms after backward, before zero_grad.

        Called by Trainer.training_step() so that grad norms are available
        even when register_hook silently fails under PEFT/DDP wrapping.
        With gradient_accumulation_steps=1, on_substep_end never fires and
        on_step_end fires AFTER zero_grad, making the old param.grad fallback
        ineffective.  This method runs at the right timing (post-backward,
        pre-zero_grad) regardless of accumulation settings.
        """
        if local_rank != 0:
            return
        dat_sq, dat_cnt = 0.0, 0
        kvhd_sq, kvhd_cnt = 0.0, 0
        for name, param in model.named_parameters():
            if not (any(k in name for k in DAT_KEYS_MATCH) and param.requires_grad):
                continue
            if param.grad is not None:
                gsq = param.grad.detach().float().norm(2).item() ** 2
                dat_sq += gsq
                dat_cnt += 1
                if self._use_kvhd and any(k in name for k in self.KVHD_KEYS):
                    kvhd_sq += gsq
                    kvhd_cnt += 1
        if dat_cnt > 0:
            self._trainer_grad_buf.append(dat_sq ** 0.5)
            if not hasattr(self, '_capture_confirmed'):
                rank0_print(f"[DATMonitor] capture_grad_norms OK: "
                            f"{dat_cnt} DAT params with grad, norm={dat_sq**0.5:.4f}")
                self._capture_confirmed = True
        if kvhd_cnt > 0:
            self._trainer_kvhd_grad_buf.append(kvhd_sq ** 0.5)

    def _flush_step(self, state):
        """Commit the current micro-step's accumulated grad norm to the buffer.

        Priority order:
          1. register_hook accumulator (fires during backward on each param)
          2. Trainer-captured grad norms (capture_grad_norms, post-backward pre-zero_grad)
          3. Direct param.grad read (only works if called before zero_grad,
             e.g. from on_substep_end with gradient_accumulation_steps > 1)
        """
        if state.is_world_process_zero:
            if self._step_cnt > 0:
                self._norm_buf.append(self._step_sq ** 0.5)
                if not hasattr(self, '_flush_path_logged'):
                    rank0_print("[DATMonitor] Grad norm source: backward hooks")
                    self._flush_path_logged = True
            elif self._trainer_grad_buf:
                self._norm_buf.extend(self._trainer_grad_buf)
                if not hasattr(self, '_flush_path_logged'):
                    rank0_print("[DATMonitor] Grad norm source: Trainer post-backward capture")
                    self._flush_path_logged = True
            elif self._model is not None:
                dat_sq, dat_cnt = 0.0, 0
                kvhd_sq, kvhd_cnt = 0.0, 0
                for name, param in self._model.named_parameters():
                    if not (any(k in name for k in DAT_KEYS_MATCH) and param.requires_grad):
                        continue
                    if param.grad is not None:
                        gsq = param.grad.detach().float().norm(2).item() ** 2
                        dat_sq += gsq
                        dat_cnt += 1
                        if self._use_kvhd and any(k in name for k in self.KVHD_KEYS):
                            kvhd_sq += gsq
                            kvhd_cnt += 1
                if dat_cnt > 0:
                    self._norm_buf.append(dat_sq ** 0.5)
                if kvhd_cnt > 0:
                    self._kvhd_norm_buf.append(kvhd_sq ** 0.5)

            if self._kvhd_step_cnt > 0:
                self._kvhd_norm_buf.append(self._kvhd_step_sq ** 0.5)
            elif self._trainer_kvhd_grad_buf:
                self._kvhd_norm_buf.extend(self._trainer_kvhd_grad_buf)

        self._trainer_grad_buf.clear()
        self._trainer_kvhd_grad_buf.clear()
        self._step_sq  = 0.0
        self._step_cnt = 0
        self._kvhd_step_sq  = 0.0
        self._kvhd_step_cnt = 0

    def _harvest_forward_diagnostics(self, model):
        """Collect offset / gate stats stashed by DAT attention layers during forward.

        ``model`` may be DDP- / PEFT- / LoRA-wrapped.  We always traverse
        ``model.modules()`` (which recurses through every wrapper) *and* also
        explicitly walk the unwrapped base model — iterating both yields the
        same module set but gives us a reliable fallback in pathological
        wrapper scenarios.
        """
        if model is None or local_rank != 0:
            return

        # ``model.modules()`` recurses through every submodule including
        # wrappers — this is the primary traversal path.  We also unwrap
        # for diagnostics so we can report the exact DAT-attention count.
        base_model = _unwrap_to_base_dat_model(model)

        off_means, off_stds = [], []
        gate_means, gate_stds = [], []
        hd_gate_vals = []

        dat_attn_count = 0
        for module in model.modules():
            is_dat_attn = 'DAT' in type(module).__name__
            if is_dat_attn:
                dat_attn_count += 1
            if hasattr(module, '_dat_offset_stats'):
                m, s = module._dat_offset_stats
                off_means.append(m)
                off_stds.append(s)
                del module._dat_offset_stats
            if hasattr(module, '_dat_gate_stats'):
                m, s = module._dat_gate_stats
                gate_means.append(m)
                gate_stds.append(s)
                del module._dat_gate_stats
            if hasattr(module, '_dat_hd_gate_value'):
                hd_gate_vals.append(module._dat_hd_gate_value)
                del module._dat_hd_gate_value

        # One-time diagnostic: report what the callback sees on its very
        # first harvest call.  Helps pin down wrapper / training-mode issues
        # (e.g. zero DAT-attn modules found → wrapper hides them; modules
        # found but no stats → forward never set them, meaning self.training
        # was False or the DAT path was skipped).
        if not getattr(self, '_harvest_diag_logged', False):
            rank0_print(
                f"[DATMonitor] _harvest_forward_diagnostics first call: "
                f"modules traversed via model={type(model).__name__} "
                f"(base={type(base_model).__name__}), "
                f"DAT-attention modules found={dat_attn_count}, "
                f"offset_stats={len(off_means)}, "
                f"gate_stats={len(gate_means)}, "
                f"hd_gate_stats={len(hd_gate_vals)}"
            )
            self._harvest_diag_logged = True

        if hd_gate_vals:
            self._hd_gate_buf.append(sum(hd_gate_vals) / len(hd_gate_vals))
        if off_means:
            self._offset_mean_buf.append(sum(off_means) / len(off_means))
            self._offset_std_buf.append(sum(off_stds) / len(off_stds))
            if not getattr(self, '_harvest_offset_confirmed', False):
                rank0_print(
                    f"[DATMonitor] First successful harvest of offset/gate "
                    f"stats (#offset={len(off_means)}, #gate={len(gate_means)})"
                )
                self._harvest_offset_confirmed = True
        if gate_means:
            self._gate_mean_buf.append(sum(gate_means) / len(gate_means))
            self._gate_std_buf.append(sum(gate_stds) / len(gate_stds))

    def on_substep_end(self, args, state, control, model=None, **kwargs):
        self._flush_step(state)
        self._harvest_forward_diagnostics(model or self._model)

    def on_step_end(self, args, state, control, model=None, optimizer=None, **kwargs):
        if model is not None:
            self._model = model
        if optimizer is not None:
            self._optimizer = optimizer
        self._flush_step(state)
        self._harvest_forward_diagnostics(model or self._model)

    # ------------------------------------------------------------------
    def _compute_weight_norms(self, model):
        """Compute fp32 L2 weight norms for DAT params (and kvhd subset) in one pass.

        Under ZeRO-2 + bf16 mixed precision, param.data is the bf16 copy.
        Uses DeepSpeed's safe_get_full_fp32_param() to retrieve the actual
        fp32 master weights (involves an all-gather across ranks).  Only
        called at logging intervals so the communication cost is acceptable.
        Falls back to param.data.float() when the API is unavailable.

        Returns (dat_weight_norm, kvhd_weight_norm,
                 hd_gate_raw, hd_gate_sigmoid).  Any may be None.
        """
        try:
            from deepspeed.utils import safe_get_full_fp32_param
            _get_fp32 = safe_get_full_fp32_param
        except ImportError:
            _get_fp32 = None

        dat_sq, dat_cnt = 0.0, 0
        kvhd_sq, kvhd_cnt = 0.0, 0
        hd_gate_sum, hd_gate_cnt = 0.0, 0
        for name, param in model.named_parameters():
            if not (any(k in name for k in DAT_KEYS_MATCH) and param.requires_grad):
                continue
            try:
                fp32_tensor = _get_fp32(param) if _get_fp32 is not None else None
                # Scalar params: report value directly, don't fold into L2 norm
                if 'hd_gate' in name and param.numel() == 1:
                    val = fp32_tensor.item() if fp32_tensor is not None else param.data.float().item()
                    hd_gate_sum += val
                    hd_gate_cnt += 1
                    continue
                if fp32_tensor is not None:
                    norm_sq = fp32_tensor.norm(2).item() ** 2
                else:
                    norm_sq = param.data.float().norm(2).item() ** 2
                dat_sq += norm_sq
                dat_cnt += 1
                if self._use_kvhd and any(k in name for k in self.KVHD_KEYS):
                    kvhd_sq += norm_sq
                    kvhd_cnt += 1
            except Exception:
                pass
        dat_wn = dat_sq ** 0.5 if dat_cnt > 0 else None
        kvhd_wn = kvhd_sq ** 0.5 if kvhd_cnt > 0 else None
        hd_gate_raw = hd_gate_sum / hd_gate_cnt if hd_gate_cnt > 0 else None
        hd_gate_sig = torch.sigmoid(torch.tensor(hd_gate_raw)).item() if hd_gate_raw is not None else None
        return dat_wn, kvhd_wn, hd_gate_raw, hd_gate_sig

    def on_log(self, args, state, control, logs=None, model=None, optimizer=None, **kwargs):
        """Flush buffered metrics to WandB and console."""
        _model     = model     or self._model
        _optimizer = optimizer or self._optimizer

        # Under ZeRO-2/3, safe_get_full_fp32_param does an all-gather so ALL
        # ranks must call _compute_weight_norms.  Under DDP (no ZeRO), each rank
        # has the full model — only rank 0 needs to compute.
        _uses_zero = _model is not None and any(
            hasattr(p, 'ds_id') for p in _model.parameters()
        )
        if _uses_zero or state.is_world_process_zero:
            wn, kvhd_wn, hd_gate_raw, hd_gate_sig = (
                self._compute_weight_norms(_model) if _model is not None
                else (None, None, None, None)
            )
        else:
            wn = kvhd_wn = hd_gate_raw = hd_gate_sig = None

        if not state.is_world_process_zero:
            return

        metrics = {}

        if wn is not None:
            metrics["dat/weight_norm"] = wn

        # 2. Average DAT grad norm over the logging interval
        if self._norm_buf:
            metrics["dat/grad_norm"] = sum(self._norm_buf) / len(self._norm_buf)
            self._norm_buf.clear()

        # 3. Real DAT LR from the optimizer param group
        if _optimizer is not None and _model is not None:
            try:
                probe = next(
                    p for n, p in _model.named_parameters()
                    if any(k in n for k in DAT_KEYS_MATCH) and p.requires_grad
                )
                for g in _optimizer.param_groups:
                    if any(id(p) == id(probe) for p in g['params']):
                        metrics["dat/lr"] = g['lr']
                        break
            except StopIteration:
                pass

        # 4. Offset statistics (mean / std of predicted offsets)
        if self._offset_mean_buf:
            metrics["dat/offset_mean"] = sum(self._offset_mean_buf) / len(self._offset_mean_buf)
            metrics["dat/offset_std"]  = sum(self._offset_std_buf)  / len(self._offset_std_buf)
            self._offset_mean_buf.clear()
            self._offset_std_buf.clear()

        # 5. Gate value statistics (intention_as_gate sigmoid output)
        if self._gate_mean_buf:
            metrics["dat/gate_mean"] = sum(self._gate_mean_buf) / len(self._gate_mean_buf)
            metrics["dat/gate_std"]  = sum(self._gate_std_buf)  / len(self._gate_std_buf)
            self._gate_mean_buf.clear()
            self._gate_std_buf.clear()

        # 6. KVHD-specific metrics (only when hd_proj is enabled)
        if self._use_kvhd:
            if kvhd_wn is not None:
                metrics["dat/kvhd_weight_norm"] = kvhd_wn
            if self._kvhd_norm_buf:
                metrics["dat/kvhd_grad_norm"] = sum(self._kvhd_norm_buf) / len(self._kvhd_norm_buf)
                self._kvhd_norm_buf.clear()

        # 7. Learnable HD gate (from weight norm pass, averaged across layers)
        if hd_gate_raw is not None:
            metrics["dat/hd_gate_raw"] = hd_gate_raw
        if hd_gate_sig is not None:
            metrics["dat/hd_gate_sigmoid"] = hd_gate_sig

        # 8. HD gate sigmoid from forward pass (averaged across layers and steps)
        if self._hd_gate_buf:
            metrics["dat/hd_gate"] = sum(self._hd_gate_buf) / len(self._hd_gate_buf)
            self._hd_gate_buf.clear()

        if not metrics:
            return

        if wandb is not None and wandb.run is not None:
            _define_wandb_step_metric()
            # Don't pass ``step=`` here; HF's own WandbCallback logs without a
            # step argument, letting wandb auto-increment its internal step.
            # We piggyback ``train/global_step`` as a data field so the
            # previously-registered step_metric lines up our curves with the
            # real training step in the wandb UI.
            wandb.log(
                {**metrics, "train/global_step": state.global_step},
                commit=False,
            )
        parts = [f"{k}={v:.6g}" for k, v in sorted(metrics.items())]
        rank0_print(f"[DATMonitor] step {state.global_step}: {', '.join(parts)}")

    def on_train_end(self, args, state, control, **kwargs):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ---------------------------------------------------------------------------
# WandB Sampling-Point Visualization Callback
# ---------------------------------------------------------------------------
class WandbSamplingVisCallback(transformers.TrainerCallback):
    """Visualise DAT sampling-point positions on WandB at logging time.

    For one randomly chosen sample in the batch, draws the sampling grid of
    every DAT layer: points coloured by offset-group (H in HSV) with
    brightness proportional to a per-point attention proxy (V in HSV).
    Thin arrows show offsets from the uniform reference grid.
    """

    def __init__(self, tokenizer=None, vis_every_n_logs: int = 10):
        self._model = None
        self._tokenizer = tokenizer
        self._vis_every_n_logs = vis_every_n_logs
        self._log_count = 0

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is not None:
            self._model = model
            vis_pick = random.randint(0, 2**31)
            # ``model.modules()`` recurses through DDP / PEFT / LoRA wrappers
            # and finds the DAT attention instances regardless.
            for m in model.modules():
                if hasattr(m, 'grid_size'):
                    m._dat_request_vis = True
                    m._dat_vis_pick = vis_pick

    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        _model = model or self._model
        self._log_count += 1

        is_vis_step = (
            _model is not None
            and state.is_world_process_zero
            and self._log_count % self._vis_every_n_logs == 0
            and wandb is not None and wandb.run is not None
        )

        # Harvest results computed during the most recent forward
        if is_vis_step:
            try:
                import matplotlib.pyplot as plt
                _define_wandb_step_metric()
                figs = self._create_sampling_vis(_model, self._tokenizer)
                for key, fig in figs.items():
                    # No ``step=`` — the registered step_metric
                    # (``train/global_step`` in the payload) is what wandb
                    # will use as the X-axis for this image panel.
                    wandb.log(
                        {key: wandb.Image(fig),
                         "train/global_step": state.global_step},
                        commit=False,
                    )
                    plt.close(fig)
            except Exception as e:
                rank0_print(f"[SamplingVis] Error: {e}")

        # Cleanup stored vis data on ALL ranks.  Per-layer flags live on
        # the DAT attention submodules (found via modules()); the vis
        # metadata (_dat_vis_input_ids / _dat_vis_image_path /
        # _batch_image_paths) is written onto the *base* DAT model by
        # forward(), so we must unwrap wrappers (DDP / PEFT / LoRA) to
        # locate and delete them.
        if _model is not None:
            for m in _model.modules():
                if hasattr(m, '_dat_vis_data'):
                    del m._dat_vis_data
            base_model = _unwrap_to_base_dat_model(_model)
            for _attr in ('_dat_vis_input_ids', '_dat_vis_image_path',
                          '_batch_image_paths'):
                if base_model is not None and hasattr(base_model, _attr):
                    delattr(base_model, _attr)
                # Also clean up on the outer wrapper in case an older code
                # path set it there (safety net, keeps state consistent).
                if hasattr(_model, _attr):
                    delattr(_model, _attr)

        # Request vis computation for the NEXT forward pass
        # (one logging interval before the next vis step)
        next_vis = (
            _model is not None
            and self._log_count % self._vis_every_n_logs
                == self._vis_every_n_logs - 1
        )
        if next_vis:
            vis_pick = random.randint(0, 2**31)
            for m in _model.modules():
                if hasattr(m, 'grid_size'):
                    m._dat_request_vis = True
                    m._dat_vis_pick = vis_pick

    @staticmethod
    def _create_sampling_vis(model, tokenizer=None):
        """Return dict of {wandb_key: matplotlib_figure}, one per DAT layer.

        Each figure shows the original image as background with the top-20
        sampling points (by attention) overlaid per group.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import numpy as np
        import textwrap
        from PIL import Image
        TOP_K = 10

        layer_data = []
        for name, module in model.named_modules():
            if hasattr(module, '_dat_vis_data') and module._dat_vis_data:
                layer_data.append((name, module._dat_vis_data))
        if not layer_data:
            return {}

        # ``_dat_vis_input_ids`` / ``_dat_vis_image_path`` are set by the
        # *base* DAT model's forward on ``self``.  Under DDP/PEFT/LoRA the
        # outer ``model`` is a wrapper and does not carry those attrs, so
        # unwrap before reading.
        attr_host = _unwrap_to_base_dat_model(model)
        if attr_host is None:
            attr_host = model

        # --- Decode question text ---
        conv_text = ''
        if tokenizer is not None and hasattr(attr_host, '_dat_vis_input_ids'):
            try:
                full = tokenizer.decode(
                    attr_host._dat_vis_input_ids, skip_special_tokens=True,
                )
                if 'user\n' in full:
                    question = full.split('user\n')[-1].split('assistant')[0].strip()
                else:
                    question = full[-300:]
                conv_text = textwrap.shorten(question, width=90, placeholder=' ...')
            except Exception:
                pass

        # --- Load original image ---
        bg_img = None
        if hasattr(attr_host, '_dat_vis_image_path') and attr_host._dat_vis_image_path:
            try:
                bg_img = np.asarray(Image.open(attr_host._dat_vis_image_path).convert('RGB'))
            except Exception:
                pass

        # --- One figure per layer ---
        figs = {}
        for name, (locs, attn) in layer_data:
            # locs: [Lp, off_grps, grid_h, grid_w, 2]
            # attn: [Lp, grid_h, grid_w] or None
            lp_idx = random.randint(0, locs.size(0) - 1) if locs.size(0) > 1 else 0
            locs_0 = locs[lp_idx].cpu().numpy()
            attn_0 = attn[lp_idx].cpu().numpy() if attn is not None else None
            n_grps, gh, gw = locs_0.shape[0], locs_0.shape[1], locs_0.shape[2]

            fig, ax = plt.subplots(1, 1, figsize=(5, 5))

            if bg_img is not None:
                ax.imshow(bg_img, extent=[-1, 1, 1, -1], aspect='auto', alpha=0.85)
            else:
                ax.set_facecolor('#f0f0f0')

            for g in range(n_grps):
                hue = g / max(n_grps, 1)
                pts = locs_0[g]               # [gh, gw, 2]
                x_all = pts[:, :, 0].flatten()
                y_all = pts[:, :, 1].flatten()

                if attn_0 is not None:
                    a_all = attn_0.flatten()
                    top_idx = np.argsort(a_all)[-TOP_K:]
                    x_pts = x_all[top_idx]
                    y_pts = y_all[top_idx]
                    a_sel = a_all[top_idx]
                    a_min, a_max = a_sel.min(), a_sel.max()
                    if a_max > a_min:
                        vals = (a_sel - a_min) / (a_max - a_min)
                    else:
                        vals = np.ones_like(a_sel)
                    vals = 0.5 + 0.5 * vals
                else:
                    n_pts = gh * gw
                    sel = np.random.choice(n_pts, min(TOP_K, n_pts), replace=False)
                    x_pts, y_pts = x_all[sel], y_all[sel]
                    vals = np.ones(len(sel))

                colors = [mcolors.hsv_to_rgb([hue, 1.0, float(v)]) for v in vals]
                ax.scatter(
                    x_pts, y_pts, c=colors, s=60,
                    edgecolors='white', linewidths=0.8, zorder=3,
                )

            ax.set_xlim(-1.05, 1.05)
            ax.set_ylim(1.05, -1.05)
            ax.set_aspect('equal')
            ax.tick_params(labelsize=6)

            # Layer label
            layer_label = name
            for k, part in enumerate(name.split('.')):
                if part == 'layers' and k + 1 < len(name.split('.')):
                    layer_label = f'L{name.split(".")[k + 1]}'
                    break

            title = layer_label
            if conv_text:
                title += f'\nQ: {conv_text}'
            fig.suptitle(title, fontsize=8, y=0.99)
            fig.tight_layout(rect=[0, 0, 1, 0.93])

            figs[f'dat/sampling_{layer_label}'] = fig

        return figs


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class Qwen2VLTrainer(transformers.Trainer):

    # ---- Keys that are DAT-/trainer-specific and must be stripped before
    # feeding the pure base VLM teacher model ----
    _TEACHER_DROP_KEYS = (
        'pixel_values_hd', 'image_grid_thw_hd',
        'image_hd_features', 'image_range_list',
        'image_paths',
    )

    def __init__(self, *args, kd_teacher: Optional[torch.nn.Module] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.kd_teacher = kd_teacher
        if self.kd_teacher is not None:
            try:
                device = next(self.model.parameters()).device
                self.kd_teacher.to(device)
                rank0_print(f"[KD] Teacher moved to device {device}")
            except Exception as e:
                rank0_print(f"[KD] Failed to move teacher: {e}")

    def training_step(self, model, inputs, num_items_in_batch=None, **kwargs):
        image_paths = inputs.pop('image_paths', None)
        if image_paths is not None:
            # Must store on the innermost base DAT model so that
            # Qwen2_5_VLDATForConditionalGeneration.forward (which accesses
            # self._batch_image_paths) can find it under LoRA/PEFT wrapping.
            # Peeling only one .module level is correct for DDP but leaves
            # PeftModel/LoraModel in front, hiding the attribute.
            inner = _unwrap_to_base_dat_model(model)
            inner._batch_image_paths = image_paths
        result = super().training_step(model, inputs, num_items_in_batch, **kwargs)

        # After backward (before zero_grad): capture DAT grad norms and
        # forward diagnostics.  Under DDP+PEFT, register_hook may silently
        # fail and on_step_end fires after zero_grad, so this is the only
        # reliable capture point.
        if hasattr(self, '_dat_monitor_callback') and self._dat_monitor_callback is not None:
            self._dat_monitor_callback.capture_grad_norms(model)
            self._dat_monitor_callback._harvest_forward_diagnostics(model)

        return result

    def save_model(self, output_dir=None, _internal_call=False):
        """Save LoRA adapter *and* non-LoRA trainables (DAT weights) + config.

        The default Trainer only calls PeftModel.save_pretrained() which stores
        LoRA adapter weights.  DAT modules (conv_lr_dw, hd_gate, …) are
        trainable but not part of LoRA, so we persist them separately as
        ``non_lora_trainables.bin`` — the same format used by the end-of-
        training save in :func:`train`.
        """
        super().save_model(output_dir, _internal_call=_internal_call)

        if output_dir is None:
            output_dir = self.args.output_dir
        if not self.args.should_save:
            return
        if not getattr(self.args, 'lora_enable', False):
            return

        peft_model = self.accelerator.unwrap_model(self.model)

        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            peft_model.named_parameters()
        )
        if non_lora_state_dict:
            torch.save(non_lora_state_dict,
                       os.path.join(output_dir, 'non_lora_trainables.bin'))
            rank0_print(f"  Saved {len(non_lora_state_dict)} non-LoRA tensors "
                        f"to {output_dir}/non_lora_trainables.bin")

        if hasattr(peft_model, 'config'):
            peft_model.config.save_pretrained(output_dir)

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
                # Merge use_reentrant=False so FlexAttention is compatible with GC
                kw.setdefault("gradient_checkpointing_kwargs", {})
                if isinstance(kw["gradient_checkpointing_kwargs"], dict):
                    kw["gradient_checkpointing_kwargs"].setdefault("use_reentrant", False)
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
        # LoRA adapter LR
        if getattr(self.args, 'lora_lr', None) is not None:
            lr_mapper["lora_"] = self.args.lora_lr

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

    # ------------------------------------------------------------------
    # Online layer-wise Knowledge Distillation
    # ------------------------------------------------------------------
    @staticmethod
    def _get_output_embeddings_unwrapped(m):
        """Fetch ``lm_head`` from a possibly DDP/Accelerate/PEFT-wrapped model.

        PeftModel forwards get_output_embeddings() to the underlying base model,
        so for most cases a single call is enough.  We still drill through
        ``.module`` / ``.get_base_model()`` defensively for older wrappers.
        """
        cur = m
        for _ in range(6):
            if hasattr(cur, 'get_output_embeddings'):
                try:
                    head = cur.get_output_embeddings()
                    if head is not None:
                        return head
                except Exception:
                    pass
            if hasattr(cur, 'get_base_model'):
                try:
                    cur = cur.get_base_model()
                    continue
                except Exception:
                    pass
            if hasattr(cur, 'base_model'):
                inner = cur.base_model
                if hasattr(inner, 'model'):
                    cur = inner.model
                else:
                    cur = inner
                continue
            if hasattr(cur, 'module'):
                cur = cur.module
                continue
            break
        return None

    def _compute_layerwise_kd_loss(
        self,
        student_hidden_states,
        teacher_hidden_states,
        student_lm_head,
        teacher_lm_head,
        student_labels,
        teacher_labels,
        temperature: float,
        layer_stride: int,
    ):
        """Compute final-logit KL(p_teacher || p_student) on **answer tokens only**,
        with correct alignment when student and teacher sequences have *different*
        lengths (e.g. when teacher consumes HR resolution while student consumes LR,
        so the ``<|image_pad|>`` count differs).

        Because (a) student/teacher share the same chat-template text and tokenizer,
        and (b) ``_apply_label_mask`` only unmasks assistant response tokens, the
        *number* and *content* of ``labels != IGNORE_INDEX`` tokens are identical
        per sample across student/teacher -- only their absolute indices differ.
        We therefore:

          1. Shift both labels for next-token alignment.
          2. Flatten answer-only hidden states via boolean indexing (batch-major,
             seq-major) on each side -> shape ``(N_answer, H)``.
          3. Per-sample count check: abort KD for this step if counts mismatch
             (safeguard against truncation or tokenizer drift).
          4. Project only the extracted answer tokens through each side's
             ``lm_head`` -> ``(N_answer, V)`` logits, which is drastically cheaper
             than projecting the entire ``(B, seq, H)`` tensor.
          5. Project only the last decoder layer through each side's ``lm_head``.
          6. Compute KL(p_teacher || p_student) in fp32 and mean over answer tokens.
        """
        if student_hidden_states is None or teacher_hidden_states is None:
            return None
        if student_labels is None or teacher_labels is None:
            return None

        num_layers = min(len(student_hidden_states), len(teacher_hidden_states))
        if num_layers == 0:
            return None

        last_idx = num_layers - 1

        T = float(temperature) if temperature and temperature > 1e-6 else 1.0

        # ---- Per-side shifted masks (bool) for answer-only positions ----
        s_shift = student_labels[..., 1:]
        t_shift = teacher_labels[..., 1:]
        s_mask = (s_shift != IGNORE_INDEX)
        t_mask = (t_shift != IGNORE_INDEX)

        s_counts = s_mask.sum(dim=-1)  # (B,)
        t_counts = t_mask.sum(dim=-1)  # (B,)
        if not torch.equal(s_counts, t_counts):
            # Token-count mismatch typically indicates truncation on one side only
            # (student's LR input is shorter so rarely truncated, but teacher HR
            # input may exceed model_max_length). Fallback to no-KD for this step.
            try:
                rank0_print(
                    f"[KD] answer-token count mismatch (student={s_counts.tolist()} "
                    f"vs teacher={t_counts.tolist()}); skipping KD this step"
                )
            except Exception:
                pass
            return None

        total_answer_tokens = int(s_counts.sum().item())
        if total_answer_tokens == 0:
            return None

        s_head_w = getattr(student_lm_head, 'weight', None)
        t_head_w = getattr(teacher_lm_head, 'weight', None)
        s_proj_dtype = s_head_w.dtype if isinstance(s_head_w, torch.Tensor) else student_hidden_states[0].dtype
        t_proj_dtype = t_head_w.dtype if isinstance(t_head_w, torch.Tensor) else teacher_hidden_states[0].dtype
        s_h = student_hidden_states[last_idx]
        t_h = teacher_hidden_states[last_idx]
        if s_h.shape[-1] != t_h.shape[-1]:
            return None

        # Shift hidden states (skip last position)
        s_h = s_h[..., :-1, :]
        t_h = t_h[..., :-1, :]

        # Extract answer-only tokens: result is (N, H), batch-major + seq-major.
        s_ans = s_h[s_mask]
        t_ans = t_h[t_mask]

        if s_ans.dtype != s_proj_dtype:
            s_ans = s_ans.to(s_proj_dtype)
        if t_ans.dtype != t_proj_dtype:
            t_ans = t_ans.to(t_proj_dtype)

        s_logits = student_lm_head(s_ans).float()
        t_logits = teacher_lm_head(t_ans).float()
        log_p_s = F.log_softmax(s_logits / T, dim=-1)
        log_p_t = F.log_softmax(t_logits / T, dim=-1)
        per_elem_kl = F.kl_div(log_p_s, log_p_t, log_target=True, reduction='none')
        per_token_kl = per_elem_kl.sum(dim=-1)  # (N,)
        kd_loss = per_token_kl.mean() * (T * T)

        del s_logits, t_logits, log_p_s, log_p_t, per_elem_kl, per_token_kl, s_ans, t_ans
        return kd_loss

    @staticmethod
    def _split_student_teacher_inputs(inputs):
        """Split a collated batch into student / teacher input dicts.

        The dataset may emit a set of keys suffixed with ``_teacher`` that carry
        a separately-processed (typically higher-resolution) copy of the sample
        for the KD teacher.  Everything else belongs to the student.
        """
        teacher_extras = {}
        student_inputs = {}
        for k, v in inputs.items():
            if k.endswith('_teacher'):
                base_k = k[:-len('_teacher')]
                teacher_extras[base_k] = v
            else:
                student_inputs[k] = v
        return student_inputs, teacher_extras

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        """Override HF default compute_loss to optionally inject online KD from a
        pure base-VLM teacher (``self.kd_teacher``).  If KD is disabled, the
        behavior is identical to transformers.Trainer.compute_loss aside from
        harmless stripping of any ``_teacher`` suffixed fields the dataset may
        have produced.
        """
        # Always strip teacher fields before the student sees inputs, regardless
        # of kd_on -- otherwise unknown kwargs would crash model.forward().
        student_inputs, teacher_extras = self._split_student_teacher_inputs(inputs)

        kd_on = getattr(self.args, 'kd_on', False)
        if (not kd_on) or (self.kd_teacher is None):
            try:
                return super().compute_loss(
                    model, student_inputs,
                    return_outputs=return_outputs,
                    num_items_in_batch=num_items_in_batch,
                )
            except TypeError:
                return super().compute_loss(
                    model, student_inputs,
                    return_outputs=return_outputs,
                )

        # ---------- KD path ----------
        # Student forward: accepts all DAT-specific kwargs (pixel_values_hd, etc.)
        student_outputs = model(
            **student_inputs,
            output_hidden_states=True,
            return_dict=True,
        )
        ce_loss = student_outputs.loss
        if ce_loss is None:
            zero = torch.zeros((), device=next(model.parameters()).device)
            return (zero, student_outputs) if return_outputs else zero

        student_hidden_states = student_outputs.hidden_states
        student_labels = student_inputs.get('labels', None)

        # Build teacher inputs.  Preferred path: dataset emitted ``_teacher``
        # fields (HR images processed with teacher's min/max pixels).
        # Fallback path: reuse the LR student inputs (stripped of DAT extras)
        # so KD still runs on batches/samples that didn't carry a teacher copy
        # (e.g. text-only, or datasets not updated to emit teacher fields).
        if teacher_extras:
            teacher_inputs = dict(teacher_extras)
            teacher_labels = teacher_inputs.pop('labels', None)
        else:
            teacher_inputs = {
                k: v for k, v in student_inputs.items()
                if k not in self._TEACHER_DROP_KEYS and k != 'labels'
            }
            teacher_labels = student_labels

        with torch.no_grad():
            teacher_outputs = self.kd_teacher(
                **teacher_inputs,
                output_hidden_states=True,
                return_dict=True,
            )
        teacher_hidden_states = teacher_outputs.hidden_states

        student_lm_head = self._get_output_embeddings_unwrapped(model)
        teacher_lm_head = self._get_output_embeddings_unwrapped(self.kd_teacher)
        if student_lm_head is None or teacher_lm_head is None:
            rank0_print("[KD] lm_head unavailable on one side; skipping KD this step.")
            return (ce_loss, student_outputs) if return_outputs else ce_loss

        kd_loss = self._compute_layerwise_kd_loss(
            student_hidden_states=student_hidden_states,
            teacher_hidden_states=teacher_hidden_states,
            student_lm_head=student_lm_head,
            teacher_lm_head=teacher_lm_head,
            student_labels=student_labels,
            teacher_labels=teacher_labels,
            temperature=float(getattr(self.args, 'kd_temperature', 1.0)),
            layer_stride=int(getattr(self.args, 'kd_layer_stride', 1)),
        )

        if kd_loss is None:
            total_loss = ce_loss
        else:
            kd_w = float(getattr(self.args, 'kd_loss_weight', 1.0))
            total_loss = ce_loss + kd_w * kd_loss.to(ce_loss.dtype)

        # Lightweight rank-0 logging; avoid touching the HF log_history pipeline.
        try:
            step = getattr(self.state, 'global_step', 0) if self.state is not None else 0
            log_every = max(1, int(getattr(self.args, 'logging_steps', 1) or 1))
            if step % log_every == 0 and local_rank in (0, -1):
                ce_val = float(ce_loss.detach().float().mean().cpu().item())
                kd_val = float(kd_loss.detach().float().mean().cpu().item()) if kd_loss is not None else 0.0
                total_val = float(total_loss.detach().float().mean().cpu().item())
                self.log({
                    "ce_loss": ce_val,
                    "kd_loss": kd_val,
                    "total_loss": total_val,
                    "kd_applied": 1.0 if kd_loss is not None else 0.0,
                })
                rank0_print(f"[KD] step={step} ce_loss={ce_val:.4f} kd_loss={kd_val:.4f}")
        except Exception:
            pass

        return (total_loss, student_outputs) if return_outputs else total_loss


# ---------------------------------------------------------------------------
# Main train function
# ---------------------------------------------------------------------------
def train():
    global local_rank

    torch.backends.cuda.enable_cudnn_sdp(False)
    rank0_print("[train] Disabled cuDNN SDPA backend to avoid mha_graph execution failures")

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
        if is_qwen2_5 and model_args.dat_manual_attn:
            from llava.model.language_model.modeling_qwen2_5vl_dat_manual import (
                convert_qwen2_5vl_to_dat,
                freeze_base_unfreeze_dat,
                DAT_KEYS_MATCH as _DAT_KEYS,
            )
        elif is_qwen2_5:
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
            'use_spatial_attn_guide': model_args.dat_use_spatial_attn_guide,
            'hd_gate_init': model_args.dat_hd_gate_init,
            'use_fused_vit': model_args.dat_fused_vit,
            'use_shared_vit': model_args.dat_shared_vit,
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
        if training_args.lora_enable:
            # LoRA mode: PEFT will freeze all base params and add adapters.
            # DAT-specific params will be unfrozen after PEFT wrapping.
            rank0_print("LoRA mode: deferring freeze control to PEFT")
        elif model_args.dat_warmup_steps > 0:
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

        if model_args.use_dat:
            target_modules = get_lora_target_modules(
                model_args.dat_layers,
                target_layers=training_args.lora_target_layers,
            )
        else:
            target_modules = find_all_linear_names(model)

        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=training_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        if training_args.bf16:
            model.to(torch.bfloat16)
        if training_args.fp16:
            model.to(torch.float16)
        rank0_print(f"Adding LoRA adapters (target_layers={training_args.lora_target_layers})...")
        model = get_peft_model(model, lora_config)
        rank0_print(f"LoRA target modules pattern: {target_modules}")

        # After PEFT wrapping (which freezes all base params), explicitly restore
        # selected non-LoRA trainables controlled by tune flags / DAT logic.
        if model_args.use_dat:
            dat_unfrozen = 0
            for name, param in model.named_parameters():
                if any(k in name for k in DAT_KEYS_MATCH):
                    param.requires_grad = True
                    dat_unfrozen += 1
            rank0_print(f"Unfroze {dat_unfrozen} DAT parameters after LoRA wrapping")

        # Respect tune_mm_mlp under LoRA: merger/projector should remain trainable
        # when requested, and stay frozen otherwise.
        merger_touched = _set_merger_trainable_after_lora(
            model, trainable=bool(getattr(model_args, "tune_mm_mlp", False))
        )
        rank0_print(
            f"LoRA + tune_mm_mlp={bool(getattr(model_args, 'tune_mm_mlp', False))}: "
            f"{'unfroze' if bool(getattr(model_args, 'tune_mm_mlp', False)) else 'kept frozen'} "
            f"{merger_touched} merger parameters"
        )

        print_trainable_parameters(model)

    # ----- KD teacher (pure base VLM, no DAT, no LoRA) -----
    kd_teacher = None
    if getattr(training_args, 'kd_on', False):
        teacher_path = getattr(training_args, 'kd_base_model_path', None) or model_args.model_name_or_path
        rank0_print(
            f"[KD] kd_on=True | loss_weight={training_args.kd_loss_weight} | "
            f"T={training_args.kd_temperature} | layer_stride={training_args.kd_layer_stride}"
        )
        rank0_print(f"[KD] Loading pure base VLM as teacher from: {teacher_path}")
        if is_qwen2_5:
            from transformers import Qwen2_5_VLForConditionalGeneration as _KD_TeacherCls
        else:
            from transformers import Qwen2VLForConditionalGeneration as _KD_TeacherCls
        kd_teacher = _KD_TeacherCls.from_pretrained(
            teacher_path,
            torch_dtype=compute_dtype,
            attn_implementation="sdpa",
        )
        kd_teacher.eval()
        for p in kd_teacher.parameters():
            p.requires_grad_(False)
        try:
            kd_teacher.config.use_cache = False
        except Exception:
            pass
        try:
            if hasattr(kd_teacher, 'gradient_checkpointing_disable'):
                kd_teacher.gradient_checkpointing_disable()
        except Exception:
            pass
        try:
            t_dtype = next(kd_teacher.parameters()).dtype
            rank0_print(
                f"[KD] Teacher ready | type={type(kd_teacher).__name__} | dtype={t_dtype}"
            )
        except Exception:
            pass

    # ----- Dataset -----
    coord_format = model_args.model_family  # "qwen2_vl" or "qwen2_5_vl"
    rank0_print(f"Bbox coordinate format: {coord_format}")
    _kd_enabled_for_dataset = bool(getattr(training_args, 'kd_on', False)) and (kd_teacher is not None)
    if model_args.use_dat:
        train_dataset = Qwen2VLCoupledDATDataset(
            data_path=data_args.data_path,
            processor=processor,
            data_args=data_args,
            model_args=model_args,
            model_max_length=training_args.model_max_length,
            coord_format=coord_format,
            kd_enabled=_kd_enabled_for_dataset,
            kd_teacher_min_pixels=getattr(training_args, 'kd_teacher_min_pixels', None),
            kd_teacher_max_pixels=getattr(training_args, 'kd_teacher_max_pixels', None),
        )
    else:
        train_dataset = Qwen2VLSupervisedDataset(
            data_path=data_args.data_path,
            processor=processor,
            data_args=data_args,
            model_max_length=training_args.model_max_length,
            coord_format=coord_format,
            kd_enabled=_kd_enabled_for_dataset,
            kd_teacher_min_pixels=getattr(training_args, 'kd_teacher_min_pixels', None),
            kd_teacher_max_pixels=getattr(training_args, 'kd_teacher_max_pixels', None),
        )
    data_collator = Qwen2VLDataCollator(pad_token_id=ENDOFTEXT_ID)

    # ----- Trainer -----
    callbacks = []
    dat_monitor = None
    if dat_warmup_callback is not None:
        callbacks.append(dat_warmup_callback)
    if model_args.use_dat:
        dat_monitor = WandbDATMonitorCallback(use_kvhd=model_args.dat_hd_proj)
        callbacks.append(dat_monitor)
    # Both the LSE two-pass path (modeling_qwen2_5vl_dat.py) and the manual-
    # attention path (modeling_qwen2_5vl_dat_manual.py) store _dat_vis_data on
    # DAT attention modules when _dat_request_vis=True.  The LSE path omits
    # the per-point attention map but still yields sampling locations — fine
    # for the viz.  Hence we register the callback whenever DAT is enabled
    # and visualisation is requested, regardless of the attention backend.
    if model_args.use_dat and training_args.visualization_every_n_steps > 0:
        callbacks.append(
            WandbSamplingVisCallback(
                tokenizer=processor.tokenizer, vis_every_n_logs=training_args.visualization_every_n_steps,
            )
        )

    trainer = Qwen2VLTrainer(
        model=model,
        processing_class=processor.tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
        callbacks=callbacks if callbacks else None,
        kd_teacher=kd_teacher,
    )
    trainer._dat_monitor_callback = dat_monitor

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
