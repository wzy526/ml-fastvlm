#!/usr/bin/env python
"""Two-box (multi-object) where-head samples from GQA.

GQA questions carry `annotations` that map question/answer words to scene-graph
object ids; the scene graph has each object's box. A question that references
>= 2 distinct objects ("Is the cup to the left of the laptop?") is a natural
cross-target sample: the where-head should put one peak on each object.

Output: the viscot_bbox.json layout (image / conversations / source) with
`bboxes` = list of normalised xyxy boxes (2..--max_boxes), one per referenced
object. There is deliberately no single `bbox` key, so the current where_head
(single box only) skips these rows until it learns the multi-box target.

    python scripts/build_gqa_multibox_data.py \
        --questions $OSS_DATA/gqa_ann/train_balanced_questions.json \
        --scene_graphs $OSS_DATA/gqa_ann/train_sceneGraphs.json \
        --image_prefix gqa/images \
        --out $OSS_DATA/extra_1023/gqa_multibox.json --max_samples 60000

image_prefix is relative to the --image_root given to where_head.py
($OSS_DATA/train_split -> train_split/gqa/images/<imageId>.jpg).
"""
import argparse
import json
import os
import random
from collections import Counter


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def referenced_objects(q):
    """Object ids referenced by the question text (and answer / fullAnswer).
    Values are object ids or comma-separated id lists."""
    ids = []
    ann = q.get("annotations") or {}
    for part in ("question", "answer", "fullAnswer"):
        for v in (ann.get(part) or {}).values():
            for oid in str(v).split(","):
                oid = oid.strip()
                if oid and oid not in ids:
                    ids.append(oid)
    return ids


def boxes_for(ids, sg, min_frac, max_frac, max_iou):
    """Normalised, de-duplicated boxes for the object ids; None if the sample
    does not qualify (too few distinct boxes, a box too small/large, or two
    boxes overlapping)."""
    W, H = float(sg["width"]), float(sg["height"])
    if W <= 0 or H <= 0:
        return None, "bad_size"
    objs = sg.get("objects") or {}
    boxes = []
    for oid in ids:
        o = objs.get(oid)
        if o is None:
            continue
        b = [o["x"] / W, o["y"] / H, (o["x"] + o["w"]) / W, (o["y"] + o["h"]) / H]
        b = [min(max(v, 0.0), 1.0) for v in b]
        if not (b[2] > b[0] and b[3] > b[1]):
            continue
        if any(iou(b, c) > 0.8 for c in boxes):        # same object listed twice
            continue
        boxes.append(b)
    if len(boxes) < 2:
        return None, "lt2_boxes"
    for b in boxes:
        fr = (b[2] - b[0]) * (b[3] - b[1])
        if fr < min_frac:
            return None, "box_too_small"
        if fr > max_frac:
            return None, "box_too_large"
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if iou(boxes[i], boxes[j]) > max_iou:
                return None, "boxes_overlap"
    return boxes, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--scene_graphs", required=True)
    ap.add_argument("--image_prefix", default="gqa/images")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_samples", type=int, default=60000)
    ap.add_argument("--max_per_image", type=int, default=2)
    ap.add_argument("--max_boxes", type=int, default=3)
    ap.add_argument("--min_frac", type=float, default=0.002, help="min box area fraction")
    ap.add_argument("--max_frac", type=float, default=0.25, help="max box area fraction")
    ap.add_argument("--max_iou", type=float, default=0.1, help="max IoU between two target boxes")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    print(f"[gqa] loading {args.scene_graphs}")
    sgs = json.load(open(os.path.expanduser(args.scene_graphs)))
    print(f"[gqa] {len(sgs)} scene graphs; loading {args.questions}")
    qs = json.load(open(os.path.expanduser(args.questions)))
    print(f"[gqa] {len(qs)} questions")

    c, per_img, out = Counter(), Counter(), []
    nbox, types = Counter(), Counter()
    keys = list(qs.keys())
    random.shuffle(keys)
    for qid in keys:
        q = qs[qid]
        img = str(q.get("imageId"))
        sg = sgs.get(img)
        if sg is None:
            c["no_scene_graph"] += 1
            continue
        ids = referenced_objects(q)
        if len(ids) < 2:
            c["lt2_refs"] += 1
            continue
        boxes, why = boxes_for(ids, sg, args.min_frac, args.max_frac, args.max_iou)
        if boxes is None:
            c[why] += 1
            continue
        if len(boxes) > args.max_boxes:
            c["gt_max_boxes"] += 1
            continue
        if per_img[img] >= args.max_per_image:
            c["per_image_cap"] += 1
            continue
        text = str(q.get("question", "")).strip()
        ans = str(q.get("fullAnswer") or q.get("answer") or "").strip()
        if not text or not ans:
            c["empty_qa"] += 1
            continue
        per_img[img] += 1
        nbox[len(boxes)] += 1
        types[(q.get("types") or {}).get("structural", "?")] += 1
        out.append({
            "id": f"gqa_{qid}",
            "image": f"{args.image_prefix}/{img}.jpg",
            "conversations": [
                {"from": "human", "value": f"<image>\n{text}"},
                {"from": "gpt", "value": ans},
            ],
            "bboxes": [[round(v, 4) for v in b] for b in boxes],
            "source": "gqa_multibox",
        })
        c["kept"] += 1
        if len(out) >= args.max_samples:
            break

    os.makedirs(os.path.dirname(os.path.expanduser(args.out)) or ".", exist_ok=True)
    json.dump(out, open(os.path.expanduser(args.out), "w"))
    print(f"[gqa] wrote {len(out)} samples -> {args.out}")
    print("[gqa] counters:", dict(c))
    print("[gqa] boxes per sample:", dict(nbox))
    print("[gqa] structural types:", dict(types.most_common()))
    if out:
        frs = sorted((b[2] - b[0]) * (b[3] - b[1]) for s in out for b in s["bboxes"])
        dist = sorted(abs((s["bboxes"][0][0] + s["bboxes"][0][2]) - (s["bboxes"][1][0] + s["bboxes"][1][2])) / 2
                      + abs((s["bboxes"][0][1] + s["bboxes"][0][3]) - (s["bboxes"][1][1] + s["bboxes"][1][3])) / 2
                      for s in out)
        pct = lambda a, p: a[min(len(a) - 1, int(p * len(a)))]
        print(f"[gqa] box area frac p10/p50/p90: {pct(frs, .1):.4f} / {pct(frs, .5):.4f} / {pct(frs, .9):.4f}")
        print(f"[gqa] centre L1 distance box1-box2 p10/p50/p90: {pct(dist, .1):.3f} / {pct(dist, .5):.3f} / {pct(dist, .9):.3f}")
        for s in out[:3]:
            print("   ", s["conversations"][0]["value"].split("\n")[1], "->", s["conversations"][1]["value"], s["bboxes"])


if __name__ == "__main__":
    main()
