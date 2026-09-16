#!/usr/bin/env python3
"""Synthetic "HD-necessary" teacher-forcing data: small text pasted onto
high-resolution images, readable in the HD pathway but not in LR.

Why: the VG "describe this region" data used for the first teacher-forcing SFT
was a no-op — VG images are ~500 px, so HD == LR (hd_target is capped at the
original pixels) and a 20-cell minimum window covered the whole image. The
readout never received detail that LR lacked. This generator guarantees it:
for every sample the text height is chosen so that

    height * s_lr  <= --lr_max_px_text   (unreadable at the LR budget)
    height * s_hd  >= --hd_min_px_text   (readable at the HD size)

where s_lr / s_hd are the exact LR-first resize scales used by
Qwen2VLCoupledDATDataset (lr_min/max_pixels, hr_scale, hd_max_pixels, cap at
the original image). Images that cannot satisfy both (too small) are skipped.

Each sample: one short text (a word or an alphanumeric code) rendered with a
contrasting outline at a random location; the question names the region
([x0, y0, x1, y1], fractions, 2 decimals, same style as VG/RefCOCO); the
top-level ``bbox`` drives teacher forcing in the trainer.

Outputs:
  <out_json>            LLaVA-format train samples (+ bbox), optionally mixed
                        with --mix_n random samples from --mix_from
  <out_json>.heldout.json   eval set for probe_whd_qwen35.py --dataset synth
  <out_img_dir>/*.jpg   rendered images; symlink the dir into the train_split
                        farm as ``synth_hd`` (the json's image field prefix)

Usage:
  python scripts/build_synth_hd_text_data.py \
      --image_dir /data/oss_bucket_0/wangziyi/models_data/sa1b_images \
      --out_img_dir /data/oss_bucket_0/wangziyi/models_data/synth_hd_images \
      --out_json /data/oss_bucket_0/wangziyi/models_data/synth_hd_text_50k.json \
      --n 50000 --heldout 500 --repeat 2 \
      --mix_from /data/oss_bucket_0/wangziyi/models_data/llava_hr_gen_vs_0817.json --mix_n 50000
"""
import argparse
import glob
import json
import math
import os
import random
import string
import sys

from PIL import Image, ImageDraw, ImageFont

FACTOR = 32          # Qwen3.5: patch 16 x merge 2 -> one LLM token / HD feature cell
WORDS = ("apple river stone cloud tiger lemon piano candle bridge forest garden "
         "silver window rocket basket pepper violet marble copper eagle falcon "
         "harbor island jacket kettle ladder magnet needle orange pocket quartz "
         "rabbit saddle tunnel velvet walnut yellow zipper anchor button castle "
         "dragon engine feather goblet helmet igloo jungle kitten lantern mirror "
         "nutmeg oyster parrot quiver ribbon spider turtle umbrella vessel whistle "
         "north south east west exit stop open closed sale menu taxi hotel bank "
         "cafe park gate dock lane road hill lake pond barn mill farm shop").split()
CODE_CHARS = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"        # no 0/O, 1/I ambiguity
QUESTIONS = (
    "What is written in the region {box}?",
    "Read the text inside the box {box}.",
    "What does the small text at {box} say?",
    "There is a piece of text in the region {box}. What is it?",
)
ANSWERS = (
    'The text in that region is "{t}".',
    'It says "{t}".',
    'The text reads "{t}".',
)


def lr_scale(W, H, lr_min, lr_max):
    px = W * H
    if px > lr_max:
        return math.sqrt(lr_max / px)
    if px < lr_min:
        return math.sqrt(lr_min / px)
    return 1.0


def hd_scale(W, H, s_lr, hr_scale, hd_max):
    lr_px = (W * s_lr) * (H * s_lr)
    hd_px = min(lr_px * hr_scale * hr_scale, W * H, hd_max)
    return math.sqrt(hd_px / (W * H))


def load_font(path, size):
    if path:
        return ImageFont.truetype(path, size)
    for cand in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"):
        if os.path.isfile(cand):
            return ImageFont.truetype(cand, size)
    hits = glob.glob("/usr/share/fonts/**/*.ttf", recursive=True)
    if hits:
        return ImageFont.truetype(sorted(hits)[0], size)
    return ImageFont.load_default(size=size)          # Pillow >= 10.1


def luminance(img, box):
    x0, y0, x1, y1 = [int(v) for v in box]
    crop = img.crop((max(x0, 0), max(y0, 0), min(x1, img.width), min(y1, img.height))).convert("L")
    crop = crop.resize((8, 8))
    return sum(crop.getdata()) / 64.0


def render(img, text, height_px, font_path, rng):
    """Paste `text` at a random spot; returns (image, bbox_px [x0,y0,x1,y1])."""
    W, H = img.size
    font = load_font(font_path, height_px)
    draw = ImageDraw.Draw(img)
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    tw, th = r - l, b - t
    if tw >= W * 0.9 or th >= H * 0.9:
        return None
    margin = int(0.03 * min(W, H))
    x = rng.randint(margin, max(margin, W - tw - margin))
    y = rng.randint(margin, max(margin, H - th - margin))
    lum = luminance(img, (x - 4, y - 4, x + tw + 4, y + th + 4))
    fill, stroke = ((255, 255, 255), (0, 0, 0)) if lum < 128 else ((0, 0, 0), (255, 255, 255))
    draw.text((x - l, y - t), text, font=font, fill=fill,
              stroke_width=max(1, height_px // 12), stroke_fill=stroke)
    return img, (x, y, x + tw, y + th)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir", required=True, help="dir of high-res source images (e.g. sa1b)")
    ap.add_argument("--image_glob", default="**/*.jpg")
    ap.add_argument("--out_img_dir", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--img_prefix", default="synth_hd",
                    help="image field prefix in the json (= symlink name in train_split)")
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--heldout", type=int, default=500)
    ap.add_argument("--repeat", type=int, default=1,
                    help="duplicate each train sample this many times (short answers carry "
                         "little token-mean loss; repetition is the code-free up-weighting)")
    ap.add_argument("--mix_from", default=None, help="json to draw extra plain samples from")
    ap.add_argument("--mix_n", type=int, default=0)
    ap.add_argument("--lr_min_pixels", type=int, default=200704)
    ap.add_argument("--lr_max_pixels", type=int, default=501760)
    ap.add_argument("--hd_max_pixels", type=int, default=5017600)
    ap.add_argument("--hr_scale", type=float, default=3.0)
    ap.add_argument("--lr_max_px_text", type=float, default=11.0,
                    help="text height at the LR size must be <= this (unreadable)")
    ap.add_argument("--hd_min_px_text", type=float, default=26.0,
                    help="text height at the HD size must be >= this (readable)")
    ap.add_argument("--min_cells", type=int, default=20, help="trainer's dat_tf_min_cells (for stats)")
    ap.add_argument("--font", default=None)
    ap.add_argument("--jpeg_quality", type=int, default=92)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    files = sorted(glob.glob(os.path.join(args.image_dir, args.image_glob), recursive=True))
    if not files:
        sys.exit(f"no images under {args.image_dir}/{args.image_glob}")
    rng.shuffle(files)
    os.makedirs(args.out_img_dir, exist_ok=True)
    print(f"{len(files)} source images; need {args.n + args.heldout}")

    train, held = [], []
    skipped_small = skipped_render = 0
    lr_px_list, hd_px_list, win_list, ratio_list = [], [], [], []
    target = args.n + args.heldout
    fi = 0
    while len(train) + len(held) < target and fi < len(files):
        path = files[fi]; fi += 1
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            continue
        W, H = img.size
        s_lr = lr_scale(W, H, args.lr_min_pixels, args.lr_max_pixels)
        s_hd = hd_scale(W, H, s_lr, args.hr_scale, args.hd_max_pixels)
        h_min = args.hd_min_px_text / s_hd            # readable at HD
        h_max = args.lr_max_px_text / s_lr            # unreadable at LR
        if h_min > h_max:
            skipped_small += 1
            continue
        height = int(rng.uniform(h_min, h_max))
        text = rng.choice(WORDS) if rng.random() < 0.5 else \
            "".join(rng.choice(CODE_CHARS) for _ in range(rng.randint(4, 6)))
        out = render(img, text, height, args.font, rng)
        if out is None:
            skipped_render += 1
            continue
        img, (x0, y0, x1, y1) = out
        pad_x, pad_y = 0.25 * (x1 - x0), 0.35 * (y1 - y0)
        bx = [max(0.0, (x0 - pad_x) / W), max(0.0, (y0 - pad_y) / H),
              min(1.0, (x1 + pad_x) / W), min(1.0, (y1 + pad_y) / H)]
        bx = [round(v, 4) for v in bx]
        box_str = "[" + ", ".join(f"{v:.2f}" for v in bx) + "]"

        idx = len(train) + len(held)
        name = f"synth_{idx:06d}.jpg"
        img.save(os.path.join(args.out_img_dir, name), quality=args.jpeg_quality)
        rel = f"{args.img_prefix}/{name}"

        lr_px_list.append(height * s_lr); hd_px_list.append(height * s_hd)
        ratio_list.append((s_hd / s_lr) ** 2)
        hd_w = W * s_hd
        win_list.append(max(bx[2] - bx[0], args.min_cells * FACTOR / hd_w))

        if len(held) < args.heldout:
            held.append({"id": f"synthhd_{idx:06d}", "image": rel,
                         "question": rng.choice(QUESTIONS).format(box=box_str),
                         "answer": text, "bbox": bx, "text_height_px": height,
                         "image_size": [W, H]})
        else:
            q = rng.choice(QUESTIONS).format(box=box_str)
            a = rng.choice(ANSWERS).format(t=text)
            train.append({"id": f"synthhd_{idx:06d}", "image": rel, "source": "synth_hd",
                          "conversations": [{"from": "human", "value": "<image>\n" + q},
                                            {"from": "gpt", "value": a}],
                          "bbox": bx})
        if idx % 1000 == 0:
            print(f"  {idx} done  (skipped small={skipped_small} render={skipped_render})", flush=True)

    out = train * max(1, args.repeat)
    n_mix = 0
    if args.mix_from and args.mix_n > 0:
        pool = json.load(open(args.mix_from))
        rng.shuffle(pool)
        out += pool[: args.mix_n]
        n_mix = min(args.mix_n, len(pool))
    rng.shuffle(out)
    json.dump(out, open(args.out_json, "w"), ensure_ascii=False)
    json.dump(held, open(args.out_json + ".heldout.json", "w"), ensure_ascii=False)

    def med(v):
        v = sorted(v); return v[len(v) // 2] if v else float("nan")
    print(f"\nwrote {len(out)} train samples ({len(train)} synth x{args.repeat} + {n_mix} mixed) -> {args.out_json}")
    print(f"wrote {len(held)} heldout -> {args.out_json}.heldout.json")
    print(f"skipped: too small for the LR/HD gap = {skipped_small}, render = {skipped_render}")
    print(f"text height at LR: median {med(lr_px_list):.1f} px  (max allowed {args.lr_max_px_text})")
    print(f"text height at HD: median {med(hd_px_list):.1f} px  (min required {args.hd_min_px_text})")
    print(f"HD/LR pixel ratio: median {med(ratio_list):.2f}x   forced window side: median {med(win_list):.2f} of image")


if __name__ == "__main__":
    main()
