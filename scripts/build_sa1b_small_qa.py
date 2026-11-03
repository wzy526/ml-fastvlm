#!/usr/bin/env python3
"""SA-1B small-object multiple-choice QA for the learned sampler (stage 1) and joint tuning (stage 2).

Per image: noun phrases from its InternVL caption -> Grounding DINO (whole image + tiles) -> keep phrases
detected exactly once whose box is small at the LR view (<= --max_lr_side px of the ~640-token image) ->
match the SA-1B mask with the best IoU -> questions whose answers come from the pixels, V*-style:

  color   "What is the color of the {phrase}?"  4 options, answer = dominant colour of the mask pixels
  relpos  "Is the {a} on the left or right side of the {b}?"  2 options, from the box centres

Rows are LLaVA json with "bbox" (normalised x0 y0 x1 y1 of the target / the pair's union) and "mc": true
(offset_conv_train.load_rows then keeps the whole prompt with its options). V* images are excluded by dHash.

  # 8 GPUs, one shard each, then merge
  python scripts/build_sa1b_small_qa.py --shard $i --shards 8 --out_dir $D
  python scripts/build_sa1b_small_qa.py --merge --out_dir $D --out $D/sa1b_small_qa.json
"""
import argparse
import colorsys
import glob
import json
import math
import os
import random
import re

import numpy as np
from PIL import Image

LR_PIXELS = 640 * 1024
COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white", "gray"]
CONFUSE = {"red": {"orange", "pink", "brown", "purple"}, "orange": {"red", "yellow", "brown"},
           "yellow": {"orange", "green"}, "green": {"yellow", "blue"}, "blue": {"green", "purple"},
           "purple": {"blue", "pink", "red"}, "pink": {"red", "purple"}, "brown": {"orange", "red", "black"},
           "black": {"gray", "brown"}, "white": {"gray"}, "gray": {"white", "black"}}
COLOR_WORDS = set(COLORS) | {"grey", "golden", "gold", "silver", "beige", "tan", "maroon", "navy", "dark", "light",
                             "colorful", "colored", "multicolored", "bright", "pale"}
DET = r"(?:a|an|the|one|two|three|four|several|some|many|another)"
STOP = {"of", "with", "on", "in", "and", "that", "which", "is", "are", "was", "were", "to", "at", "by", "from",
        "near", "behind", "under", "above", "over", "into", "while", "who", "for", "as", "its", "their", "his",
        "her", "has", "have", "can", "be", "or", "but", "this", "these", "those", "there", "it", "they"}
ABSTRACT = {"image", "scene", "picture", "photo", "photograph", "background", "foreground", "view", "left",
            "right", "side", "top", "bottom", "center", "centre", "middle", "area", "part", "front", "back",
            "distance", "setting", "atmosphere", "overall", "sense", "variety", "number", "lot", "group", "mix",
            "array", "row", "line", "edge", "corner", "space", "surface", "shot", "frame", "composition", "day",
            "time", "light", "lighting", "sky", "ground", "floor", "wall", "road", "street", "water", "grass"}
PROMPT_TAIL = "Answer with the option's letter from the given choices directly."


def noun_phrases(caption, k):
    out, seen = [], set()
    for m in re.finditer(DET + r"\s+([a-z][a-z\-]*(?:\s+[a-z][a-z\-]*){0,3})", caption.lower()):
        words = []
        for w in m.group(1).split():
            if w in STOP:
                break
            words.append(w)
        while words and (words[-1] in ABSTRACT or words[-1].endswith(("ing", "ly"))
                         or (len(words) > 1 and len(words[-1]) >= 6 and words[-1].endswith("ed"))):
            words.pop()
        if not words or len(words) > 3 or words[-1] in ABSTRACT or len(words[-1]) < 3:
            continue
        p = " ".join(words)
        if p not in seen:
            seen.add(p)
            out.append(p)
        if len(out) >= k:
            break
    return out


def rle_counts(s):
    cnts, p = [], 0
    while p < len(s):
        x, k, more = 0, 0, 1
        while more:
            c = ord(s[p]) - 48
            x |= (c & 0x1F) << (5 * k)
            more = c & 0x20
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if len(cnts) > 2:
            x += cnts[-2]
        cnts.append(x)
    return cnts


def rle_mask(seg):
    h, w = seg["size"]
    c = seg["counts"]
    cnts = rle_counts(c) if isinstance(c, str) else list(c)
    vals = np.zeros(len(cnts), np.uint8)
    vals[1::2] = 1
    return np.repeat(vals, cnts)[: h * w].reshape(w, h).T


def color_name(rgb):
    out = []
    for r, g, b in rgb:
        h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        h *= 360
        if v < 0.2:
            out.append("black")
        elif s < 0.15:
            out.append("white" if v > 0.8 else ("black" if v < 0.3 else "gray"))
        elif h < 15 or h >= 340:
            out.append("red")
        elif h < 45:
            out.append("brown" if v < 0.6 else "orange")
        elif h < 70:
            out.append("yellow")
        elif h < 165:
            out.append("green")
        elif h < 255:
            out.append("blue")
        elif h < 290:
            out.append("purple")
        else:
            out.append("pink")
    return out


def dhash(img):
    g = img.convert("L")
    g.thumbnail((256, 256))
    a = np.asarray(g.resize((9, 8), Image.LANCZOS), dtype=np.int16)
    return int("".join("1" if v else "0" for v in (a[:, 1:] > a[:, :-1]).flatten()), 2)


def vstar_hashes(root):
    root = os.path.expanduser(root)
    hs = []
    for line in open(os.path.join(root, "test_questions.jsonl")):
        hs.append(dhash(Image.open(os.path.join(root, json.loads(line)["image"]))))
    return hs


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter, 1e-9)


def same_obj(a, b):
    """Same detection seen from the whole image and from overlapping tiles: high IoU, or one box mostly inside
    the other (a tile cuts the object)."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return iou(a, b) > 0.5 or ix * iy > 0.6 * max(small, 1e-9)


def tiles_of(W, H, tile, overlap):
    if tile <= 0 or (W <= tile and H <= tile):
        return [(0, 0, W, H)]
    step = max(1, int(tile * (1 - overlap)))
    xs = list(range(0, max(W - tile, 0) + 1, step))
    ys = list(range(0, max(H - tile, 0) + 1, step))
    if xs[-1] + tile < W:
        xs.append(W - tile)
    if ys[-1] + tile < H:
        ys.append(H - tile)
    return [(0, 0, W, H)] + [(x, y, min(x + tile, W), min(y + tile, H)) for y in ys for x in xs]


def label_to_phrase(label, phrases):
    label = label.strip().lower()
    hits = [p for p in phrases if p == label]
    if not hits:
        hits = [p for p in phrases if label and (label in p or p in label)]
    return hits[0] if len(hits) == 1 else None


def detect(model, processor, img, phrases, device, thr, text_thr):
    import torch
    text = " . ".join(phrases) + " ."
    inp = processor(images=img, text=text, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model(**inp)
    kw = dict(text_threshold=text_thr, target_sizes=[(img.height, img.width)])
    try:
        res = processor.post_process_grounded_object_detection(out, inp["input_ids"], threshold=thr, **kw)[0]
    except TypeError:
        res = processor.post_process_grounded_object_detection(out, inp["input_ids"], box_threshold=thr, **kw)[0]
    labels = res.get("text_labels", res.get("labels"))
    return [(float(s), [float(v) for v in b], str(l))
            for s, b, l in zip(res["scores"].tolist(), res["boxes"].tolist(), labels)]


def mc(question, options, gt, rng):
    opts = list(options)
    rng.shuffle(opts)
    letters = "ABCD"[: len(opts)]
    body = "\n".join(f"({c}) {o}" for c, o in zip(letters, opts))
    return f"{question}\n{body}\n{PROMPT_TAIL}", letters[opts.index(gt)]


def build_shard(args):
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    rng = random.Random(args.seed * 1000 + args.shard)
    data = json.load(open(os.path.expanduser(args.caption_json)))
    random.Random(args.seed).shuffle(data)
    data = data[: args.n_images][args.shard:: args.shards]
    vh = vstar_hashes(args.vstar_root) if args.vstar_root else []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(args.gdino)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.gdino).to(device).eval()
    root = os.path.expanduser(args.image_root)
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"shard_{args.shard}.jsonl")
    done = set()
    if os.path.isfile(out_path):
        for line in open(out_path):
            done.add(json.loads(line)["image"])
    fout = open(out_path, "a")
    stats = {"img": 0, "skip_vstar": 0, "no_mask": 0, "no_np": 0, "raw": 0, "mapped": 0, "multi": 0, "low": 0,
             "big": 0, "tiny": 0, "small": 0, "no_mask_match": 0, "targets": 0, "color": 0, "relpos": 0,
             "labels": []}
    for n, s in enumerate(data):
        im = s.get("image")
        im = im[0] if isinstance(im, list) else im
        conv = s.get("conversations") or []
        if not im or len(conv) < 2 or im in done:
            continue
        path = os.path.join(root, im)
        side = os.path.splitext(path)[0] + ".json"
        if not (os.path.isfile(path) and os.path.isfile(side)):
            stats["no_mask"] += 1
            continue
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            continue
        stats["img"] += 1
        if vh:
            h = dhash(img)
            if min(bin(h ^ v).count("1") for v in vh) <= 8:
                stats["skip_vstar"] += 1
                continue
        phrases = noun_phrases(str(conv[1].get("value", "")), args.max_phrases)
        if not phrases:
            stats["no_np"] += 1
            continue
        W, H = img.size
        dets = []
        for (x0, y0, x1, y1) in tiles_of(W, H, args.tile, 0.2):
            crop = img if (x0, y0, x1, y1) == (0, 0, W, H) else img.crop((x0, y0, x1, y1))
            for sc, b, lab in detect(model, processor, crop, phrases, device, args.thr, args.text_thr):
                stats["raw"] += 1
                if len(stats["labels"]) < 12:
                    stats["labels"].append(str(lab))
                p = label_to_phrase(lab, phrases)
                if p:
                    dets.append((sc, [b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0], p))
        stats["mapped"] += len(dets)
        dets.sort(key=lambda d: -d[0])
        keep = []
        for d in dets:
            if all(not (d[2] == k[2] and same_obj(d[1], k[1])) for k in keep):
                keep.append(d)
        s_lr = math.sqrt(min(1.0, LR_PIXELS / float(W * H)))
        per = {}
        for d in keep:
            per.setdefault(d[2], []).append(d)
        cands = []
        for p, ds in per.items():
            if len(ds) != 1:
                stats["multi"] += 1
                continue
            if ds[0][0] < args.min_score:
                stats["low"] += 1
                continue
            b = ds[0][1]
            side_lr = max(b[2] - b[0], b[3] - b[1]) * s_lr
            if args.min_lr_side <= side_lr <= args.max_lr_side:
                cands.append((p, b))
            else:
                stats["big" if side_lr > args.max_lr_side else "tiny"] += 1
        if not cands:
            continue
        stats["small"] += len(cands)
        ann = json.load(open(side))
        iw, ih = ann.get("image", {}).get("width", W), ann.get("image", {}).get("height", H)
        sx, sy = iw / W, ih / H
        img_m = img if (iw, ih) == (W, H) else img.resize((iw, ih))
        rows = []
        for p, b in cands:
            bb = [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]
            best, bi = None, 0.0
            for a in ann.get("annotations", []):
                x, y, w, h = a["bbox"]
                v = iou(bb, [x, y, x + w, y + h])
                if v > bi:
                    best, bi = a, v
            if best is None or bi < args.min_mask_iou:
                stats["no_mask_match"] += 1
                continue
            x, y, w, h = best["bbox"]
            tb = [x / iw, y / ih, (x + w) / iw, (y + h) / ih]
            stats["targets"] += 1
            if not (set(p.split()) & COLOR_WORDS):
                m = rle_mask(best["segmentation"])
                x0, y0 = int(x), int(y)
                x1, y1 = max(int(math.ceil(x + w)), x0 + 1), max(int(math.ceil(y + h)), y0 + 1)
                sub = m[y0:y1, x0:x1].astype(bool)
                if sub.sum() >= 30:
                    pix = np.asarray(img_m.crop((x0, y0, x1, y1)), dtype=np.uint8)[sub]
                    names = color_name(pix[:: max(1, len(pix) // 2000)])
                    top = max(set(names), key=names.count)
                    if names.count(top) / len(names) >= args.color_frac:
                        pool = [c for c in COLORS if c != top and c not in CONFUSE[top]]
                        q, a = mc(f"What is the color of the {p}?", [top] + rng.sample(pool, 3), top, rng)
                        rows.append({"q": q, "a": a, "bbox": tb, "qtype": "color", "phrase": p})
            rows.append({"_target": (p, tb)})
        targets = [r["_target"] for r in rows if "_target" in r]
        rows = [r for r in rows if "_target" not in r]
        rows = [r for r in rows if r["qtype"] != "color"] + rng.sample(
            [r for r in rows if r["qtype"] == "color"], min(2, sum(r["qtype"] == "color" for r in rows)))
        pairs = [(t1, t2) for i, t1 in enumerate(targets) for t2 in targets[i + 1:]
                 if abs((t1[1][0] + t1[1][2]) - (t2[1][0] + t2[1][2])) / 2 > args.relpos_sep]
        if pairs:
            (pa, ba), (pb, bb2) = rng.choice(pairs)
            if rng.random() < 0.5:
                (pa, ba), (pb, bb2) = (pb, bb2), (pa, ba)
            gt = "left" if (ba[0] + ba[2]) < (bb2[0] + bb2[2]) else "right"
            q, a = mc(f"Is the {pa} on the left or right side of the {pb}?", ["left", "right"], gt, rng)
            u = [min(ba[0], bb2[0]), min(ba[1], bb2[1]), max(ba[2], bb2[2]), max(ba[3], bb2[3])]
            rows.append({"q": q, "a": a, "bbox": u, "qtype": "relpos", "phrase": f"{pa} | {pb}"})
        for r in rows:
            stats[r["qtype"]] += 1
            fout.write(json.dumps({"image": im, "conversations": [
                {"from": "human", "value": "<image>\n" + r["q"]}, {"from": "gpt", "value": r["a"]}],
                "bbox": [round(v, 5) for v in r["bbox"]], "mc": True, "qtype": r["qtype"],
                "phrase": r["phrase"], "src": "sa1b_small"}) + "\n")
        fout.flush()
        if n % 200 == 0:
            print(f"[sa1b_qa] shard {args.shard}: {n}/{len(data)} {stats}", flush=True)
    print(f"[sa1b_qa] shard {args.shard} done: {stats}", flush=True)


def merge(args):
    rows = []
    for f in sorted(glob.glob(os.path.join(args.out_dir, "shard_*.jsonl"))):
        rows += [json.loads(line) for line in open(f)]
    random.Random(args.seed).shuffle(rows)
    json.dump(rows, open(os.path.expanduser(args.out), "w"))
    by = {}
    for r in rows:
        by[r["qtype"]] = by.get(r["qtype"], 0) + 1
    letters = {}
    for r in rows:
        letters[r["conversations"][1]["value"]] = letters.get(r["conversations"][1]["value"], 0) + 1
    print(f"[sa1b_qa] {len(rows)} rows from {len({r['image'] for r in rows})} images -> {args.out}; "
          f"{by}; answers {dict(sorted(letters.items()))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caption_json", default="/data/oss_bucket_0/wangziyi/models_data/sft_data/llava_sa1b_caption_pretrain.json")
    ap.add_argument("--image_root", default="~/sft_data/train_split")
    ap.add_argument("--vstar_root", default="~/data/vstar_bench", help="'' to skip the V* exclusion")
    ap.add_argument("--gdino", default="IDEA-Research/grounding-dino-base")
    ap.add_argument("--n_images", type=int, default=60000, help="images considered over all shards")
    ap.add_argument("--max_phrases", type=int, default=16)
    ap.add_argument("--tile", type=int, default=800, help="also detect on tiles of this side (0 = whole only)")
    ap.add_argument("--thr", type=float, default=0.3)
    ap.add_argument("--text_thr", type=float, default=0.25)
    ap.add_argument("--min_score", type=float, default=0.4)
    ap.add_argument("--min_lr_side", type=float, default=6.0)
    ap.add_argument("--max_lr_side", type=float, default=24.0, help="box side in px at the 640-token LR view")
    ap.add_argument("--min_mask_iou", type=float, default=0.5)
    ap.add_argument("--color_frac", type=float, default=0.5)
    ap.add_argument("--relpos_sep", type=float, default=0.08, help="min |centre x| gap, image-width fraction")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.merge:
        merge(args)
    else:
        build_shard(args)


if __name__ == "__main__":
    main()
