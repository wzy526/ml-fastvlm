#!/usr/bin/env python3
"""Natural-image boxes for the localisation head (where_head.py), from the
HF RefCOCO / RefCOCO+ / RefCOCOg mirrors that embed the COCO images
(lmms-lab/RefCOCO*, every split; no local COCO needed).

Writes images to <out_dir>/images/<name>.jpg and a LLaVA-format json whose
"bbox" is [x0, y0, x1, y1] as image fractions and whose human turn is a
V*-style question built from the referring expression, e.g.
  "What is the colour of the man in the red shirt?"
Only a probe-data builder; nothing in the model or trainer changes.

  python scripts/build_refcoco_box_data.py --out_dir $OSS_DATA/extra_1018/refcoco \
      --out $OSS_DATA/extra_1018/refcoco_bbox.json
"""
import argparse
import io
import json
import os
import random
import sys

from PIL import Image

SPECS = [  # (name, HF repo)
    ("refcoco", "lmms-lab/RefCOCO"),
    ("refcocoplus", "lmms-lab/RefCOCOplus"),
    ("refcocog", "lmms-lab/RefCOCOg"),
]
TEMPLATES = [
    "What is the color of {e}?",
    "What color is {e}?",
    "Is {e} facing left or right?",
    "What is {e} holding?",
    "What material is {e} made of?",
    "Is {e} closer to the camera than the other objects?",
    "What is written on {e}?",
    "How many stripes does {e} have?",
]


def expressions(ex):
    for k in ("answer", "sentences", "sentence", "sent", "expressions", "question"):
        v = ex.get(k)
        if v is None:
            continue
        if isinstance(v, str):
            v = [v]
        if isinstance(v, (list, tuple)):
            out = []
            for s in v:
                if isinstance(s, dict):
                    s = s.get("sent") or s.get("raw") or s.get("sentence")
                if isinstance(s, str) and s.strip():
                    out.append(s.strip().rstrip(".").lower())
            if out:
                return out
    return []


def box_frac(ex, w, h):
    bb = ex.get("bbox")
    if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
        return None
    x, y, bw, bh = [float(v) for v in bb]
    # lmms-lab RefCOCO stores COCO xywh pixels; fall back to xyxy if xywh overflows
    x0, y0, x1, y1 = x / w, y / h, (x + bw) / w, (y + bh) / h
    if x1 > 1.01 or y1 > 1.01:
        x0, y0, x1, y1 = x / w, y / h, bw / w, bh / h
    x1, y1 = min(x1, 1.0), min(y1, 1.0)
    if not (0 <= x0 < x1 and 0 <= y0 < y1):
        return None
    return [round(x0, 5), round(y0, 5), round(x1, 5), round(y1, 5)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, help="images go to <out_dir>/images")
    ap.add_argument("--out", required=True, help="output json")
    ap.add_argument("--max_per_dataset", type=int, default=20000)
    ap.add_argument("--max_rel_area", type=float, default=0.15,
                    help="keep boxes <= this fraction of image area (V*-ish); 1.0 = all")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    from datasets import get_dataset_split_names, load_dataset

    img_dir = os.path.join(args.out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    rows, seen = [], set()
    for name, repo in SPECS:
        try:
            splits = get_dataset_split_names(repo)
        except Exception as e:  # noqa: BLE001
            print(f"[refcoco] {repo}: cannot list splits ({e}); trying val", flush=True)
            splits = ["val"]
        kept = skipped_big = 0
        for split in splits:
            if kept >= args.max_per_dataset:
                break
            try:
                ds = load_dataset(repo, split=split)
            except Exception as e:  # noqa: BLE001
                print(f"[refcoco] {repo}/{split}: load failed ({e})", flush=True)
                continue
            print(f"[refcoco] {repo}/{split}: {len(ds)} rows, columns {ds.column_names}", flush=True)
            for i, ex in enumerate(ds):
                if kept >= args.max_per_dataset:
                    break
                im = ex.get("image")
                if im is None:
                    continue
                if isinstance(im, dict):
                    im = Image.open(io.BytesIO(im["bytes"]))
                w, h = im.size
                b = box_frac(ex, w, h)
                if b is None:
                    continue
                if (b[2] - b[0]) * (b[3] - b[1]) > args.max_rel_area:
                    skipped_big += 1
                    continue
                exprs = expressions(ex)
                if not exprs:
                    continue
                fname = ex.get("file_name") or f"{name}_{split}_{i}.jpg"
                fname = os.path.basename(str(fname))
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    fname += ".jpg"
                path = os.path.join(img_dir, fname)
                if fname not in seen:
                    im.convert("RGB").save(path, quality=92)
                    seen.add(fname)
                e = random.choice(exprs)
                q = random.choice(TEMPLATES).format(e=e)
                rows.append({
                    "id": f"{name}_{split}_{i}",
                    "image": path,
                    "bbox": b,
                    "target": e,
                    "conversations": [{"from": "human", "value": f"<image>\n{q}"},
                                      {"from": "gpt", "value": ""}],
                })
                kept += 1
        print(f"[refcoco] {name}: kept {kept}, skipped {skipped_big} boxes > {args.max_rel_area:.0%} area", flush=True)

    random.shuffle(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(rows, open(args.out, "w"))
    areas = sorted((r["bbox"][2] - r["bbox"][0]) * (r["bbox"][3] - r["bbox"][1]) for r in rows)
    if areas:
        med = areas[len(areas) // 2]
        print(f"[refcoco] wrote {len(rows)} rows, {len(seen)} images -> {args.out}; "
              f"box area median {med:.1%}, p10 {areas[len(areas) // 10]:.1%}")
    else:
        print("[refcoco] nothing written", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
