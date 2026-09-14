#!/usr/bin/env python3
"""Print the LR / HR geometry lmms-eval's DAT wrapper will actually use, per
LR pixel budget, on a real benchmark image. No model is loaded — only the two
image processors and the wrapper's LR-first HR derivation, so it runs on CPU
in seconds.

    python scripts/check_eval_token_geometry.py \
        --ckpt ~/vldat_experiments/0908_sft_qwen35_2b_dat_exactgrad_genvs-merged \
        --hrbench_parquet ~/.cache/huggingface/hub/datasets--DreamMr--HR-Bench/snapshots/*/hr_bench_4k.parquet \
        --pixels 262144 655360 1310720 1806336 2621440 6553600 11796480 \
        --hr_cap 5017600

Columns: LR tokens fed to the LLM, HR side / pixels seen by the HD ViT, the
number of merged HD patch features the 400 DAT samples are drawn from, and the
effective HR/LR edge ratio (should be hr_scale=3 unless hr_cap or the image's
native size binds).
"""
import argparse
import glob
import io
import math
import os
import sys

from PIL import Image

sys.path.insert(0, os.environ.get("LMMS_EVAL_DIR", os.path.expanduser("~/lmms-eval")))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="DAT ckpt dir (needs preprocessor_config.json)")
    ap.add_argument("--image", default=None, help="any image; default = first HR-Bench row")
    ap.add_argument("--hrbench_parquet", default=os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--DreamMr--HR-Bench/snapshots/*/hr_bench_4k.parquet"))
    ap.add_argument("--pixels", type=int, nargs="+",
                    default=[262144, 655360, 1310720, 1806336, 2621440, 6553600, 11796480],
                    help="lr_max_pixels grid (eval_pixel_sweep.sh default = tok x 1024)")
    ap.add_argument("--min_pixels", type=int, default=28224)
    ap.add_argument("--hr_cap", type=int, default=5017600, help="eval_pixel_sweep.sh HR_CAP")
    ap.add_argument("--hr_scale", type=int, default=3)
    ap.add_argument("--grid_size", type=int, default=20, help="DAT grid -> grid^2 sampled HD tokens")
    args = ap.parse_args()

    from transformers import AutoProcessor
    from lmms_eval.models.simple.qwen3_5_dat import Qwen3_5_DAT

    if args.image:
        img = Image.open(args.image).convert("RGB")
    else:
        import pandas as pd
        hits = sorted(glob.glob(os.path.expanduser(args.hrbench_parquet)))
        if not hits:
            sys.exit(f"no parquet matches {args.hrbench_parquet}; pass --image")
        row = pd.read_parquet(hits[0]).iloc[0]
        blob = row["image"]
        blob = blob["bytes"] if isinstance(blob, dict) else blob
        img = Image.open(io.BytesIO(blob)).convert("RGB")
    W, H = img.size
    print(f"image {W}x{H} = {W*H/1e6:.2f} MP")

    # Stand-in for the wrapper: only the attributes _derive_hr_size_from_lr_first reads.
    w = Qwen3_5_DAT.__new__(Qwen3_5_DAT)
    w._patch_size = Qwen3_5_DAT.PATCH_SIZE
    w._spatial_merge = Qwen3_5_DAT.SPATIAL_MERGE
    w._factor = w._patch_size * w._spatial_merge
    w.hr_scale = args.hr_scale
    w.hr_max_pixels = args.hr_cap
    tok_px = w._factor * w._factor
    hr_proc = AutoProcessor.from_pretrained(args.ckpt, min_pixels=tok_px, max_pixels=100_000_000)

    n_hd = args.grid_size ** 2
    print(f"hr_scale={args.hr_scale}  hr_cap={args.hr_cap} ({int(math.sqrt(args.hr_cap))}^2)  "
          f"DAT samples {n_hd} HD tokens per layer\n")
    print(f"{'lr_max_px':>10} | {'LR side':>9} | {'LR tok':>6} | {'HR side':>10} | {'HR MP':>6} | "
          f"{'HD feats':>8} | {'400/feats':>9} | {'HR/LR':>5}")
    print("-" * 90)
    for px in args.pixels:
        w.processor = AutoProcessor.from_pretrained(args.ckpt, min_pixels=args.min_pixels, max_pixels=px)
        lr = w.processor.image_processor(images=[img], return_tensors="pt")["image_grid_thw"][0]
        lr_h, lr_w = int(lr[1]) * w._patch_size, int(lr[2]) * w._patch_size
        lr_tok = (int(lr[1]) // w._spatial_merge) * (int(lr[2]) // w._spatial_merge)
        hd_h, hd_w = w._derive_hr_size_from_lr_first(img)
        hr = hr_proc.image_processor(images=[img.resize((hd_w, hd_h))], return_tensors="pt")["image_grid_thw"][0]
        hd_feats = (int(hr[1]) // w._spatial_merge) * (int(hr[2]) // w._spatial_merge)
        ratio = math.sqrt((hd_h * hd_w) / (lr_h * lr_w))
        flag = "" if abs(ratio - args.hr_scale) < 0.05 else "  <- cap/native binds"
        print(f"{px:>10} | {lr_w:>4}x{lr_h:<4} | {lr_tok:>6} | {hd_w:>5}x{hd_h:<4} | "
              f"{hd_h*hd_w/1e6:>6.2f} | {hd_feats:>8} | {n_hd/hd_feats:>8.1%} | {ratio:>5.2f}{flag}")


if __name__ == "__main__":
    main()
