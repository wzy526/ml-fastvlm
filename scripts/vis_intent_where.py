#!/usr/bin/env python
"""Same image, two questions: does the where-head move?

The layer-8 where-head is the only place the question can move the window.
Image tokens come first, so those states have not seen the question; the head's
cross-attention lets each cell read the question tokens. This script looks for
that on GQA, where one image has several questions and each question has its
own boxes.

It keeps pairs whose boxes are far apart, runs the same where-head forward as
ldat_eval (LR image, question, WHERE_HINT, no sampling head), and writes the
pairs whose peaks land in different boxes.

    python scripts/vis_intent_where.py \
        --model_path $E/checkpoint-1000 \
        --where_head_pt $E/loc/where_lm8_ce20.pt \
        --json $OSS_DATA/extra_1023/gqa_multibox.json \
        --image_root ~/sft_data/train_split \
        --out $X/vis_intent
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llava.model.language_model.modeling_qwen3_5_dat as dat_mod


def question_of(s):
    conv = s.get("conversations") or []
    q = conv[0].get("value", "") if conv else ""
    return " ".join(q.replace("<image>", " ").split())


def boxes_of(s):
    raw = s.get("bboxes") or ([s["bbox"]] if s.get("bbox") else [])
    out = []
    for b in raw:
        if not (isinstance(b, (list, tuple)) and len(b) == 4):
            continue
        b = [float(v) for v in b]
        if b[2] > b[0] and b[3] > b[1] and max(b) <= 1.5:
            out.append(b)
    return out


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def center(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def l1(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def inside(px, py, boxes, margin=0.02):
    hit = []
    for b in boxes:
        if b[0] - margin <= px <= b[2] + margin and b[1] - margin <= py <= b[3] + margin:
            hit.append(b)
    return hit


def pair_sep(ba, bb):
    """Largest center distance between a box of A and a box of B, if the
    closest pair still does not overlap. None when the two questions share a region."""
    if not ba or not bb:
        return None
    closest = min(l1(center(a), center(b)) for a in ba for b in bb)
    overlap = max(iou(a, b) for a in ba for b in bb)
    if closest < 0.28 or overlap > 0.05:
        return None
    return max(l1(center(a), center(b)) for a in ba for b in bb)


def load_pairs(path, image_root, n_try):
    data = json.load(open(os.path.expanduser(path)))
    groups = defaultdict(list)
    for s in data:
        im = s.get("image")
        im = im[0] if isinstance(im, list) else im
        if not im:
            continue
        q = question_of(s)
        bs = boxes_of(s)
        if not q or not bs:
            continue
        groups[im].append({"q": q, "boxes": bs, "src": s.get("source", "")})
    cands = []
    for im, rows in groups.items():
        # one row per distinct question
        uniq = []
        seen = set()
        for r in rows:
            if r["q"] in seen:
                continue
            seen.add(r["q"])
            uniq.append(r)
        path = im if os.path.isabs(im) else os.path.join(image_root, im)
        if not os.path.isfile(path):
            continue
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                sep = pair_sep(uniq[i]["boxes"], uniq[j]["boxes"])
                if sep is None:
                    continue
                cands.append((sep, path, uniq[i], uniq[j]))
    cands.sort(key=lambda t: -t[0])
    return cands[:n_try], len(groups), len(cands)


class WhereMap:
    """Layer-8 where-head, same prompt prefix as ldat_eval.infer.LDatRunner.prepare."""

    def __init__(self, model_path, where_pt, lr_tok, device):
        from transformers import AutoProcessor

        self.dev = device
        self.where_layer = 8
        self.model = dat_mod.Qwen3_5DATForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map={"": device}, attn_implementation="sdpa"
        ).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        sd = torch.load(os.path.expanduser(where_pt), map_location="cpu")
        sd = {k[len("where_head."):] if k.startswith("where_head.") else k: v for k, v in sd.items()}
        hidden = self.model.config.text_config.hidden_size
        self.model.where_head = dat_mod._WhereXAttnHead(hidden, hidden, d=384, layers=2, heads=6, max_hw=512)
        self.model.where_head.load_state_dict(sd)
        self.model.where_head.float().to(device).eval()
        self.processor = AutoProcessor.from_pretrained(model_path, min_pixels=1024, max_pixels=int(lr_tok) * 1024)
        self.image_id = self.model.config.image_token_id

    @torch.inference_mode()
    def run(self, image, question):
        image = image.convert("RGB")
        thw = self.processor.image_processor(images=[image], return_tensors="pt")["image_grid_thw"][0]
        lr_img = image.resize((int(thw[2]) * 16, int(thw[1]) * 16), Image.BICUBIC)
        msgs = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [{"type": "image", "image": lr_img},
                                             {"type": "text", "text": question},
                                             {"type": "text", "text": dat_mod.WHERE_HINT}]}]
        text = self.processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False)
        text = text.rstrip()
        if text.endswith("<|im_end|>"):
            text = text[:-len("<|im_end|>")]
        inp = self.processor(text=[text], images=[lr_img], return_tensors="pt")
        inp = {k: v.to(self.dev) for k, v in inp.items() if torch.is_tensor(v)}
        inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
        out = self.model(**inp, output_hidden_states=True, return_dict=True, use_cache=False)
        ids = inp["input_ids"][0]
        is_img = ids == self.image_id
        gh, gw = int(thw[1]) // 2, int(thw[2]) // 2
        hs = out.hidden_states[self.where_layer][0]
        cells = hs[is_img]
        if cells.shape[0] != gh * gw:
            return None
        logits = self.model.where_head(cells.reshape(gh, gw, -1), hs[int(torch.nonzero(is_img)[-1]) + 1:])
        p = dat_mod._wdat_softmax(logits.float().tolist())
        flat = int(logits.detach().float().argmax())
        cy, cx = divmod(flat, gw)
        box = ((cx + 0.5 - 2) / gw, (cy + 0.5 - 2) / gh, (cx + 0.5 + 2) / gw, (cy + 0.5 + 2) / gh)
        hd_w, hd_h = dat_mod._where_hd_size(image, thw, 3, 5017600)
        win = dat_mod._where_pad_window(box, 2.0, 20, hd_w, hd_h)
        win = dat_mod._wdat_adaptive_window(p, box, win, 0.6)[0]
        peak = ((cx + 0.5) / gw, (cy + 0.5) / gh)
        return {"p": p, "peak": peak, "win": win}


def font(size):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        if os.path.isfile(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def wrap(text, fnt, width, max_lines):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        nxt = w if not cur else cur + " " + w
        if fnt.getlength(nxt) <= width:
            cur = nxt
        else:
            if cur:
                lines.append(cur)
            cur = w
        if len(lines) == max_lines:
            break
    if len(lines) < max_lines and cur:
        lines.append(cur)
    if len(lines) == max_lines and (cur not in lines[-1]):
        lines[-1] = lines[-1].rstrip(".") + "…"
    return lines


def rect(dr, box, size, color, width):
    W, H = size
    dr.rectangle([box[0] * W, box[1] * H, box[2] * W, box[3] * H], outline=color, width=width)


def cross(dr, pt, size, color, r=7):
    x, y = pt[0] * size[0], pt[1] * size[1]
    dr.ellipse([x - r, y - r, x + r, y + r], outline=color, width=3)
    dr.line([x - r - 4, y, x + r + 4, y], fill=color, width=2)
    dr.line([x, y - r - 4, x, y + r + 4], fill=color, width=2)


def overlay(image, p, color):
    arr = np.asarray(p, np.float32)
    arr = arr / max(float(arr.max()), 1e-8)
    arr = np.sqrt(arr)
    W, H = image.size
    heat = np.asarray(Image.fromarray((arr * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR), np.float32) / 255.0
    base = np.asarray(image.convert("RGB"), np.float32)
    a = np.clip((heat - 0.22) / 0.78, 0, 1)[..., None] * 0.72
    col = np.array(color, np.float32)
    return Image.fromarray((base * (1 - a) + col * a).astype(np.uint8))


def fit(im, w, h):
    im = im.copy()
    im.thumbnail((w, h), Image.BICUBIC)
    canvas = Image.new("RGB", (w, h), (248, 248, 248))
    canvas.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    return canvas


def draw_example(image, a, b, hit):
    """Three panels: both boxes, question A map, question B map."""
    green, orange, cyan = (30, 160, 60), (230, 120, 20), (0, 190, 220)
    base = image.convert("RGB")
    left = base.copy()
    d = ImageDraw.Draw(left)
    for box in a["boxes"]:
        rect(d, box, left.size, green, 4)
    for box in b["boxes"]:
        rect(d, box, left.size, orange, 4)

    mid = overlay(base, a["map"]["p"], (210, 40, 40))
    d = ImageDraw.Draw(mid)
    for box in a["boxes"]:
        rect(d, box, mid.size, green, 4)
    rect(d, a["map"]["win"], mid.size, cyan, 4)
    cross(d, a["map"]["peak"], mid.size, (255, 255, 255))

    right = overlay(base, b["map"]["p"], (40, 90, 210))
    d = ImageDraw.Draw(right)
    for box in b["boxes"]:
        rect(d, box, right.size, orange, 4)
    rect(d, b["map"]["win"], right.size, cyan, 4)
    cross(d, b["map"]["peak"], right.size, (255, 255, 255))

    pw, ph, cap_h = 460, 360, 108
    panels = [fit(left, pw, ph), fit(mid, pw, ph), fit(right, pw, ph)]
    fnt = font(16)
    small = font(14)
    titles = [
        "image  ·  green = question A boxes, orange = question B",
        "question A  ·  red = where-head, cyan = window",
        "question B  ·  blue = where-head, cyan = window",
    ]
    bodies = ["", a["q"], b["q"]]
    sheet = Image.new("RGB", (pw * 3 + 16, ph + cap_h + 8), (255, 255, 255))
    for i, panel in enumerate(panels):
        sheet.paste(panel, (8 + i * pw, 4))
    dr = ImageDraw.Draw(sheet)
    for i in range(3):
        x = 8 + i * pw
        dr.text((x, ph + 8), titles[i], fill=(40, 40, 40), font=small)
        color = (20, 20, 20) if i == 0 else (green if i == 1 else orange)
        y = ph + 30
        for line in wrap(bodies[i], fnt, pw - 12, 3):
            dr.text((x, y), line, fill=color, font=fnt)
            y += 22
    tag = "both peaks in their own boxes" if hit else "peaks did not both land in their boxes"
    dr.text((8, ph + cap_h - 16), tag, fill=(20, 120, 40) if hit else (160, 40, 40), font=small)
    return sheet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--where_head_pt", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--image_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr_tok", type=int, default=640)
    ap.add_argument("--n_try", type=int, default=36, help="separated pairs to run, farthest first")
    ap.add_argument("--n_show", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    image_root = os.path.expanduser(args.image_root)
    pairs, n_img, n_sep = load_pairs(args.json, image_root, args.n_try)
    print(f"[vis] {n_img} images, {n_sep} question pairs with boxes >= 0.28 apart, running {len(pairs)}",
          flush=True)
    if not pairs:
        raise SystemExit("[vis] no separated pairs with an image file on disk")

    where = WhereMap(args.model_path, args.where_head_pt, args.lr_tok, args.device)
    done = []
    for k, (sep, path, a, b) in enumerate(pairs):
        image = Image.open(path).convert("RGB")
        ma, mb = where.run(image, a["q"]), where.run(image, b["q"])
        if ma is None or mb is None:
            print(f"[vis] {k} grid mismatch {path}", flush=True)
            continue
        ha = inside(*ma["peak"], a["boxes"])
        hb = inside(*mb["peak"], b["boxes"])
        cross_a = inside(*ma["peak"], b["boxes"])
        cross_b = inside(*mb["peak"], a["boxes"])
        dist = l1(ma["peak"], mb["peak"])
        hit = bool(ha) and bool(hb) and not cross_a and not cross_b and dist >= 0.22
        rec = {"path": path, "sep": sep, "dist": dist, "hit": hit,
               "a": {**a, "map": ma, "hit_boxes": ha},
               "b": {**b, "map": mb, "hit_boxes": hb},
               "image": image}
        done.append(rec)
        print(f"[vis] {k + 1}/{len(pairs)} hit={int(hit)} peak_l1={dist:.2f} box_sep={sep:.2f} "
              f"| {a['q'][:70]} || {b['q'][:70]}", flush=True)

    hits = [r for r in done if r["hit"]]
    hits.sort(key=lambda r: -r["dist"])
    show = hits[:args.n_show]
    kind = "hits"
    if len(show) < args.n_show:
        rest = [r for r in done if not r["hit"]]
        rest.sort(key=lambda r: -(int(bool(r["a"]["hit_boxes"])) + int(bool(r["b"]["hit_boxes"])) + r["dist"]))
        show = (hits + rest)[:args.n_show]
        kind = "hits+nearmiss" if hits else "nearmiss"
    if not show:
        raise SystemExit("[vis] no forward succeeded")

    os.makedirs(args.out, exist_ok=True)
    sheets = [draw_example(r["image"], r["a"], r["b"], r["hit"]) for r in show]
    w = max(s.width for s in sheets)
    gap = 10
    canvas = Image.new("RGB", (w, sum(s.height for s in sheets) + gap * (len(sheets) - 1)), (255, 255, 255))
    y = 0
    meta = []
    for i, (s, r) in enumerate(zip(sheets, show)):
        canvas.paste(s, (0, y))
        y += s.height + gap
        one = os.path.join(args.out, f"ex{i:02d}_{'hit' if r['hit'] else 'miss'}.png")
        s.save(one)
        meta.append({
            "file": one, "image": r["path"], "hit": r["hit"],
            "peak_l1": round(r["dist"], 3), "box_sep": round(r["sep"], 3),
            "q1": r["a"]["q"], "boxes1": r["a"]["boxes"], "peak1": [round(v, 4) for v in r["a"]["map"]["peak"]],
            "win1": [round(v, 4) for v in r["a"]["map"]["win"]],
            "q2": r["b"]["q"], "boxes2": r["b"]["boxes"], "peak2": [round(v, 4) for v in r["b"]["map"]["peak"]],
            "win2": [round(v, 4) for v in r["b"]["map"]["win"]],
        })
    sheet_path = os.path.join(args.out, "intent_where.png")
    canvas.save(sheet_path)
    json.dump({"kind": kind, "n_tried": len(done), "n_hit": len(hits), "rows": meta},
              open(os.path.join(args.out, "intent_where.json"), "w"), indent=2)
    print(f"[vis] tried {len(done)}  both-peaks-in-own-boxes {len(hits)}  wrote {kind} -> {sheet_path}", flush=True)


if __name__ == "__main__":
    main()
