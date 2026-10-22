"""Build a fresh DAT checkpoint with **random-initialized DAT layers** on top of
the un-fine-tuned Qwen2.5-VL-3B-Instruct base. Used as a "before training"
baseline for offset diagnostics: if the trained Run B and the random-init
DAT produce qualitatively similar offset behavior (e.g., both prompt-blind),
then training did not actively *break* the offset network — it simply never
learned anything beyond what kaiming-normal weights already gave us.

The DAT config matches Run B exactly:
    layers = DLLLLLDLLLLLDLLLLLDLLLLLDLLLLLDLLLLL  (D at 0/6/12/18/24/30)
    grid_size=20, off_grps=8, inter_size=128, hr_scale=3
    hd_proj=True, use_intention_branch=True, intention_as_gate=True,
    use_spatial_attn_guide=True, hd_gate_init=-1.0,
    fused_vit=False, shared_vit=False
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        default="/root/autodl-tmp/models_data/Qwen2.5-VL-3B-Instruct",
        help="Path to base (un-fine-tuned) Qwen2.5-VL-3B-Instruct",
    )
    parser.add_argument(
        "--out",
        default="/root/autodl-tmp/vldat_experiments/_random_init_dat_runB_cfg",
        help="Output directory for the random-init DAT checkpoint",
    )
    args = parser.parse_args()

    from transformers import AutoProcessor, AutoTokenizer
    from llava.model.language_model.modeling_qwen2_5vl_dat import (
        convert_qwen2_5vl_to_dat,
    )

    dat_extra_args = {
        "grid_size": 20,
        "off_ksize": 3,
        "off_grps": 8,
        "inter_size": 128,
        "hr_scale": 3,
        "hd_proj": True,
        "layers": "DLLLLLDLLLLLDLLLLLDLLLLLDLLLLLDLLLLL",
        "use_intention_branch": True,
        "intention_as_gate": True,
        "use_spatial_attn_guide": True,
        "hd_gate_init": -1.0,
        "hd_gate_freeze": False,
        "use_fused_vit": False,
        "use_shared_vit": False,
    }
    print(f"[1/3] Converting {args.base} to DAT with config:")
    for k, v in dat_extra_args.items():
        print(f"  {k}: {v}")

    torch.manual_seed(42)
    model = convert_qwen2_5vl_to_dat(
        args.base,
        dat_extra_args=dat_extra_args,
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[2/3] Saving to {out_dir}")
    model.save_pretrained(out_dir, safe_serialization=True)

    print("[3/3] Copying processor + tokenizer assets")
    proc = AutoProcessor.from_pretrained(args.base)
    proc.save_pretrained(out_dir)
    tok = AutoTokenizer.from_pretrained(args.base)
    tok.save_pretrained(out_dir)

    print(f"\nDone. Random-init DAT ckpt at:\n  {out_dir}")


if __name__ == "__main__":
    main()
