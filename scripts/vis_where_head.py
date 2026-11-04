#!/usr/bin/env python3
"""Render the where-head's per-cell target map (the Fig used for the layer-8 claim).

For each of --n V* questions: low-res forward of the frozen LM, the where-head on
layer 8's image-token states, its softmax map overlaid on the image. Green = GT
box, cyan = the window the pipeline would crop (top cell, padded x2).

    python scripts/vis_where_head.py --model_path $E/checkpoint-1000 \
        --where_head_pt $E/loc/where_lm8_ce20.pt --vstar_root ~/data/vstar_bench \
        --out ~/vldat_experiments/where_vis/layer8.png
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llava.model.language_model.modeling_qwen3_5_dat as dat_mod  # noqa: E402
from gdino_probe import load_vstar, gt_box_frac  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--where_head_pt", required=True)
    ap.add_argument("--vstar_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--lr_tok", type=int, default=640)
    ap.add_argument("--thumb", type=int, default=420, help="thumbnail width in the contact sheet")
    args = ap.parse_args()

    dev = torch.device("cuda")
    model = dat_mod.Qwen3_5DATForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    sd = torch.load(os.path.expanduser(args.where_head_pt), map_location="cpu")
    sd = {k[len("where_head."):] if k.startswith("where_head.") else k: v for k, v in sd.items()}
    model.where_head.load_state_dict(sd)

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model_path, min_pixels=1024, max_pixels=args.lr_tok * 1024)
    image_id = model.config.image_token_id

    thumbs = []
    with torch.no_grad():
        for s in load_vstar(args.vstar_root, args.n):
            image, q = s["image"], s["prompt"]
            W, H = image.size
            thw = processor.image_processor(images=[image], return_tensors="pt")["image_grid_thw"][0]
            lr = image.resize((int(thw[2]) * 16, int(thw[1]) * 16), Image.BICUBIC)
            msgs = [{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": [{"type": "image", "image": lr}, {"type": "text", "text": q},
                                                 {"type": "text", "text": dat_mod.WHERE_HINT}]}]
            text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False,
                                                 enable_thinking=False).rstrip()
            if text.endswith("<|im_end|>"):
                text = text[:-len("<|im_end|>")]
            inp = processor(text=[text], images=[lr], return_tensors="pt")
            inp = {k: v.to(dev) for k, v in inp.items() if torch.is_tensor(v)}
            inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
            out = model(**inp, output_hidden_states=True, return_dict=True, use_cache=False)
            ids = inp["input_ids"][0]
            is_img = ids == image_id
            gh, gw = int(thw[1]) // 2, int(thw[2]) // 2
            hs = out.hidden_states[8][0]
            cells = hs[is_img]
            if cells.shape[0] != gh * gw:
                print(f"skip {s.get('category')}: {cells.shape[0]} cells != {gh}x{gw}", flush=True)
                continue
            logits = model.where_head(cells.reshape(gh, gw, -1), hs[int(torch.nonzero(is_img)[-1]) + 1:])
            p = torch.softmax(logits.float().flatten(), 0).reshape(gh, gw).cpu().numpy()
            hd_w, hd_h = dat_mod._where_hd_size(image, thw, 3, 5017600)
            k = int(np.argmax(p))
            cy, cx = divmod(k, gw)
            box = ((cx + 0.5 - 2) / gw, (cy + 0.5 - 2) / gh, (cx + 0.5 + 2) / gw, (cy + 0.5 + 2) / gh)
            win = dat_mod._where_pad_window(box, 2.0, 20, hd_w, hd_h)
            thumbs.append(render(image, p, gt_box_frac(s), win, q))
            print(f"{s.get('category')}: peak {p.max():.3f} at cell ({cy},{cx})", flush=True)

    sheet = contact(thumbs)
    os.makedirs(os.path.dirname(os.path.expanduser(args.out)) or ".", exist_ok=True)
    sheet.save(os.path.expanduser(args.out))
    print(f"saved {args.out} ({sheet.size[0]}x{sheet.size[1]}, {len(thumbs)} rows)", flush=True)


def render(image, p, gt, win, question, width=420):
    """Image with the probability map overlaid; green = GT box, cyan = crop window."""
    W, H = image.size
    s = width / W
    im = image.resize((width, max(1, int(H * s))), Image.BICUBIC).convert("RGB")
    heat = Image.fromarray((plt_cmap(p / p.max()) * 255).astype("uint8"), "RGB")
    heat = heat.resize(im.size, Image.BILINEAR)
    blended = Image.blend(im, heat, 0.45)
    d = ImageDraw.Draw(blended)
    w, h = blended.size

    def rect(b, color, t=3):
        d.rectangle([b[0] * w, b[1] * h, b[2] * w, b[3] * h], outline=color, width=t)

    if gt is not None:
        rect(gt, (0, 255, 0))
    rect(win, (0, 255, 255))
    cap = Image.new("RGB", (w, 36), (0, 0, 0))
    ImageDraw.Draw(cap).text((6, 8), question[:90], fill=(255, 255, 255))
    out = Image.new("RGB", (w, blended.size[1] + 36))
    out.paste(blended, (0, 0))
    out.paste(cap, (0, blended.size[1]))
    return out


def plt_cmap(a):
    a = np.clip(a, 0, 1)[..., None]
    lo = np.array([10, 10, 60])
    hi = np.array([255, 230, 40])
    return lo * (1 - a) + hi * a


def contact(thumbs, cols=4, gap=8):
    if not thumbs:
        raise SystemExit("no rows rendered")
    rows = (len(thumbs) + cols - 1) // cols
    cw = max(t.size[0] for t in thumbs)
    ch = max(t.size[1] for t in thumbs)
    sheet = Image.new("RGB", (cols * cw + (cols + 1) * gap, rows * ch + (rows + 1) * gap), (255, 255, 255))
    for i, t in enumerate(thumbs):
        r, c = divmod(i, cols)
        sheet.paste(t, (gap + c * (cw + gap), gap + r * (ch + gap)))
    return sheet


if __name__ == "__main__":
    main()
