#!/usr/bin/env python3
"""Merge LoRA adapters + trained DAT weights into a single full-weight checkpoint.

Usage:
    python scripts/merge_lora_dat_weights.py \
        --model_base /path/to/Qwen2.5-VL-3B-Instruct \
        --lora_path  /path/to/lora_checkpoint \
        --output_dir /path/to/merged_output

Supports all DAT families; the family is auto-detected from the checkpoint's
config.json model_type (qwen2_5_vl_dat / qwen3_vl_dat / qwen3_5_dat), or can
be forced with --family.

The script:
  1. Loads the base model
  2. Converts it to DAT (using dat_extra_args from the checkpoint config)
  3. Loads trained DAT weights from non_lora_trainables.bin
  4. Loads LoRA adapters and merges them into the base weights
  5. Saves the full merged model as a standard HF checkpoint
"""

import argparse
import importlib
import json
import os
import sys

import torch
from transformers import AutoProcessor, AutoConfig

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

_FAMILIES = {
    'qwen2_5_vl': ('llava.model.language_model.modeling_qwen2_5vl_dat',
                   'convert_qwen2_5vl_to_dat',
                   'Qwen2_5_VLDATConfig',
                   'Qwen2_5_VLDATForConditionalGeneration'),
    'qwen3_vl':   ('llava.model.language_model.modeling_qwen3_vl_dat',
                   'convert_qwen3_vl_to_dat',
                   'Qwen3VLDATConfig',
                   'Qwen3VLDATForConditionalGeneration'),
    'qwen3_5':    ('llava.model.language_model.modeling_qwen3_5_dat',
                   'convert_qwen3_5_to_dat',
                   'Qwen3_5DATConfig',
                   'Qwen3_5DATForConditionalGeneration'),
}


def _detect_family(lora_path: str) -> str:
    """Map the checkpoint's config.json model_type to a DAT family."""
    cfg_path = os.path.join(lora_path, 'config.json')
    with open(cfg_path, 'r', encoding='utf-8') as f:
        model_type = json.load(f).get('model_type', '')
    for fam in ('qwen3_5', 'qwen3_vl', 'qwen2_5_vl'):
        if model_type.startswith(fam):
            return fam
    raise ValueError(
        f"Cannot infer DAT family from model_type={model_type!r} in {cfg_path}; "
        f"pass --family explicitly."
    )


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA + DAT weights into a full checkpoint")
    parser.add_argument("--model_base", required=True,
                        help="Path to the base pretrained model (e.g. Qwen2.5-VL-3B-Instruct)")
    parser.add_argument("--lora_path", required=True,
                        help="Path to the LoRA checkpoint (containing adapter_model, non_lora_trainables.bin)")
    parser.add_argument("--output_dir", required=True,
                        help="Where to save the merged full-weight model")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"],
                        help="Torch dtype for loading and saving")
    parser.add_argument("--family", default=None, choices=sorted(_FAMILIES),
                        help="DAT model family. Default: auto-detect from the checkpoint config.")
    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.torch_dtype]

    family = args.family or _detect_family(args.lora_path)
    mod_name, convert_name, config_name, cls_name = _FAMILIES[family]
    mod = importlib.import_module(mod_name)
    convert_fn = getattr(mod, convert_name)
    config_cls = getattr(mod, config_name)
    print(f"[family] {family}  ({mod_name})")

    # --- Step 1: Read DAT config from checkpoint ---
    print(f"[1/5] Loading config from {args.lora_path} ...")
    dat_config = config_cls.from_pretrained(args.lora_path)
    print(f"  dat_extra_args: {dat_config.dat_extra_args}")

    # --- Step 2: Load base model and convert to DAT ---
    print(f"[2/5] Loading base model from {args.model_base} and converting to DAT ...")
    model = convert_fn(
        args.model_base,
        dat_extra_args=dat_config.dat_extra_args,
        torch_dtype=torch_dtype,
    )
    print(f"  Model type: {type(model).__name__}")

    # --- Step 3: Load trained DAT weights ---
    nlt_path = os.path.join(args.lora_path, 'non_lora_trainables.bin')
    if os.path.exists(nlt_path):
        print(f"[3/5] Loading DAT weights from {nlt_path} ...")
        non_lora_trainables = torch.load(nlt_path, map_location='cpu')

        non_lora_trainables = {
            (k[11:] if k.startswith('base_model.') else k): v
            for k, v in non_lora_trainables.items()
        }
        if any(k.startswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {
                (k[6:] if k.startswith('model.') else k): v
                for k, v in non_lora_trainables.items()
            }

        info = model.load_state_dict(non_lora_trainables, strict=False)
        print(f"  Loaded {len(non_lora_trainables)} tensors")
        if info.unexpected_keys:
            print(f"  WARNING: unexpected keys: {info.unexpected_keys[:5]}...")
    else:
        print(f"[3/5] No non_lora_trainables.bin found, skipping DAT weight loading")

    # --- Step 4: Load and merge LoRA ---
    adapter_path = os.path.join(args.lora_path, 'adapter_config.json')
    if os.path.exists(adapter_path):
        from peft import PeftModel
        print(f"[4/5] Loading LoRA adapters from {args.lora_path} ...")
        model = PeftModel.from_pretrained(model, args.lora_path)
        print("  Merging LoRA weights into base model ...")
        model = model.merge_and_unload()
        print("  LoRA merged successfully")
    else:
        print(f"[4/5] No adapter_config.json found, skipping LoRA merge")

    # --- Step 5: Save merged model ---
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[5/5] Saving merged model to {args.output_dir} ...")
    model.save_pretrained(args.output_dir, safe_serialization=True)

    # Qwen3.5 ships a fast-only tokenizer; legacy use_fast=False elsewhere.
    processor = AutoProcessor.from_pretrained(
        args.model_base, trust_remote_code=True, use_fast=(family == 'qwen3_5'),
    )
    processor.save_pretrained(args.output_dir)

    # Newer transformers' ProcessorMixin.save_pretrained() only writes a unified
    # `processor_config.json` + `chat_template.jinja` and tokenizer files, skipping
    # per-component configs. Persist legacy-format files so downstream tooling that
    # still looks for `preprocessor_config.json` / `video_preprocessor_config.json`
    # / `chat_template.json` (e.g. evaluation scripts) works without silently
    # falling back to the base model on the hub.
    for attr_name in ("image_processor", "video_processor", "feature_extractor"):
        sub = getattr(processor, attr_name, None)
        if sub is not None and hasattr(sub, "save_pretrained"):
            sub.save_pretrained(args.output_dir)

    if isinstance(getattr(processor, "chat_template", None), str):
        with open(os.path.join(args.output_dir, "chat_template.json"), "w", encoding="utf-8") as f:
            json.dump({"chat_template": processor.chat_template}, f, ensure_ascii=False, indent=2)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nDone! Saved merged model ({total_params:,} params) to {args.output_dir}")
    print(f"Load with:")
    print(f"  from {mod_name} import {cls_name}")
    print(f"  model = {cls_name}.from_pretrained('{args.output_dir}')")


if __name__ == "__main__":
    main()
