#!/usr/bin/env python3
"""Natural-image SMALL-object boxes for the localisation head (where_head.py)
from the local COCO instances annotations (no download, no teacher).

For every image, a category that occurs exactly once (so "the {category}" is
unambiguous) with a box of <= --max_rel_area of the image gives one row; the
smallest such boxes are preferred (V*-like). Human turn is a V*-style question
about the category, "bbox" is [x0, y0, x1, y1] as image fractions.
Probe-data builder only; nothing in the model or trainer changes.

  python scripts/build_coco_box_data.py --coco_root ~/sft_data/train_split/coco \
      --out $OSS_DATA/extra_1018/coco_small_bbox.json
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict

TEMPLATES = [
    "What is the color of the {c}?",
    "What color is the {c}?",
    "Is the {c} facing left or right?",
    "What is the {c} next to?",
    "What material is the {c} made of?",
    "Is the {c} on the left or the right side of the image?",
    "What is on top of the {c}?",
    "Is the {c} open or closed?",
]


def find_image_dir(coco_root, file_name, split):
    for cand in (os.path.join(coco_root, "images", split), os.path.join(coco_root, split),
                 os.path.join(coco_root, "images"), coco_root):
        if os.path.isfile(os.path.join(cand, file_name)):
            return cand
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_root", default="~/sft_data/train_split/coco")
    ap.add_argument("--splits", default="train2017", help="comma list, e.g. train2017,val2017")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_rel_area", type=float, default=0.05,
                    help="keep boxes <= this fraction of image area (0.05 ~ <=140 px side on 640)")
    ap.add_argument("--min_side_px", type=float, default=10, help="drop boxes smaller than this (px)")
    ap.add_argument("--per_image", type=int, default=2, help="rows per image (smallest unique categories first)")
    ap.add_argument("--max_rows", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    root = os.path.expanduser(args.coco_root)

    rows = []
    for split in args.splits.split(","):
        ann_path = os.path.join(root, "annotations", f"instances_{split}.json")
        if not os.path.isfile(ann_path):
            print(f"[coco] missing {ann_path}", file=sys.stderr)
            continue
        print(f"[coco] loading {ann_path}", flush=True)
        d = json.load(open(ann_path))
        cats = {c["id"]: c["name"] for c in d["categories"]}
        imgs = {im["id"]: im for im in d["images"]}
        by_img = defaultdict(list)
        for a in d["annotations"]:
            if a.get("iscrowd", 0):
                continue
            by_img[a["image_id"]].append(a)
        img_dir = None
        n_img = n_rows = 0
        for iid, anns in by_img.items():
            im = imgs[iid]
            w, h = im["width"], im["height"]
            count = defaultdict(int)
            for a in anns:
                count[a["category_id"]] += 1
            cands = []
            for a in anns:
                if count[a["category_id"]] != 1:
                    continue
                x, y, bw, bh = a["bbox"]
                if bw < args.min_side_px or bh < args.min_side_px:
                    continue
                rel = (bw * bh) / float(w * h)
                if rel > args.max_rel_area:
                    continue
                cands.append((rel, a))
            if not cands:
                continue
            if img_dir is None:
                img_dir = find_image_dir(root, im["file_name"], split)
                if img_dir is None:
                    print(f"[coco] cannot find image {im['file_name']} under {root}", file=sys.stderr)
                    sys.exit(1)
                print(f"[coco] images at {img_dir}", flush=True)
            cands.sort(key=lambda t: t[0])
            n_img += 1
            for rel, a in cands[: args.per_image]:
                x, y, bw, bh = a["bbox"]
                c = cats[a["category_id"]]
                q = random.choice(TEMPLATES).format(c=c)
                rows.append({
                    "id": f"coco_{split}_{a['id']}",
                    "image": os.path.join(img_dir, im["file_name"]),
                    "bbox": [round(x / w, 5), round(y / h, 5), round((x + bw) / w, 5), round((y + bh) / h, 5)],
                    "target": c,
                    "rel_area": round(rel, 5),
                    "conversations": [{"from": "human", "value": f"<image>\n{q}"},
                                      {"from": "gpt", "value": ""}],
                })
                n_rows += 1
        print(f"[coco] {split}: {n_img} images -> {n_rows} rows", flush=True)

    random.shuffle(rows)
    rows = rows[: args.max_rows]
    if not rows:
        print("[coco] nothing written", file=sys.stderr)
        sys.exit(1)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(rows, open(args.out, "w"))
    areas = sorted(r["rel_area"] for r in rows)
    sides640 = [640 * (a ** 0.5) for a in areas]  # target side in px when long side ~640
    print(f"[coco] wrote {len(rows)} rows -> {args.out}; box area median {areas[len(areas) // 2]:.2%}, "
          f"p10 {areas[len(areas) // 10]:.2%}; side @640: median {sides640[len(sides640) // 2]:.0f} px, "
          f"p10 {sides640[len(sides640) // 10]:.0f} px; <16 px: {sum(s < 16 for s in sides640) / len(sides640):.0%}")


if __name__ == "__main__":
    main()
