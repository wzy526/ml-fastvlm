#!/usr/bin/env python3
"""Merge LoRA adapters + trained DAT weights into a single full-weight checkpoint.

Usage:
    python scripts/merge_lora_dat_weights.py \
        --model_base /path/to/Qwen2.5-VL-3B-Instruct \
        --lora_path  /path/to/lora_checkpoint \
        --output_dir /path/to/merged_output

The script:
  1. Loads the base Qwen2.5-VL model
  2. Converts it to DAT (using dat_extra_args from the checkpoint config)
  3. Loads trained DAT weights from non_lora_trainables.bin
  4. Loads LoRA adapters and merges them into the base weights
  5. Saves the full merged model as a standard HF checkpoint

The output can be loaded directly with:
    Qwen2_5_VLDATForConditionalGeneration.from_pretrained(output_dir)
"""

import argparse
import json
import os
import sys

import torch
from transformers import AutoProcessor, AutoConfig

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


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
    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.torch_dtype]

    from llava.model.language_model.modeling_qwen2_5vl_dat import (
        convert_qwen2_5vl_to_dat,
        Qwen2_5_VLDATConfig,
    )

    # --- Step 1: Read DAT config from checkpoint ---
    print(f"[1/5] Loading config from {args.lora_path} ...")
    dat_config = Qwen2_5_VLDATConfig.from_pretrained(args.lora_path)
    print(f"  dat_extra_args: {dat_config.dat_extra_args}")

    # --- Step 2: Load base model and convert to DAT ---
    print(f"[2/5] Loading base model from {args.model_base} and converting to DAT ...")
    model = convert_qwen2_5vl_to_dat(
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

    processor = AutoProcessor.from_pretrained(args.model_base, trust_remote_code=True, use_fast=False)
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
    print(f"  from llava.model.language_model.modeling_qwen2_5vl_dat import Qwen2_5_VLDATForConditionalGeneration")
    print(f"  model = Qwen2_5_VLDATForConditionalGeneration.from_pretrained('{args.output_dir}')")


if __name__ == "__main__":
    main()
