#!/usr/bin/env python3
"""Merge native Qwen SFT adapters and merger weights (no DAT attention conversion)."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoProcessor

from ldat_eval.backbone import load_backbone


def merge(model_path, lora_path, output_dir, dtype=torch.bfloat16):
    from peft import PeftModel
    lora_path, output_dir = Path(lora_path).expanduser(), Path(output_dir).expanduser()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output directory is not empty: {output_dir}")
    if not (lora_path / "adapter_config.json").is_file():
        raise ValueError(f"No LoRA adapter under {lora_path}")
    model = load_backbone(model_path, "qwen2_5_vl", device="cpu", dtype=dtype).model
    trainables = lora_path / "non_lora_trainables.bin"
    if trainables.is_file():
        state = torch.load(trainables, map_location="cpu", weights_only=True)
        state = {k.removeprefix("base_model."): v for k, v in state.items()}
        if any(k.startswith("model.model.") for k in state):
            state = {k.removeprefix("model."): v for k, v in state.items()}
        info = model.load_state_dict(state, strict=False)
        if info.unexpected_keys:
            raise ValueError(f"Unrecognized trained weights: {info.unexpected_keys[:8]}")
    model = PeftModel.from_pretrained(model, str(lora_path)).merge_and_unload()
    model.save_pretrained(output_dir, safe_serialization=True)
    AutoProcessor.from_pretrained(model_path).save_pretrained(output_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_path", required=True)
    p.add_argument("--lora_path", required=True)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()
    merge(args.model_path, args.lora_path, args.output_dir)
