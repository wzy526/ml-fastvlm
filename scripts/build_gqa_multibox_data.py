#!/usr/bin/env python
"""Two-box (multi-object) where-head samples from GQA.

GQA questions carry `annotations` that map question/answer words to scene-graph
object ids; the scene graph has each object's box. A question that references
>= 2 distinct objects ("Is the cup to the left of the laptop?") is a natural
cross-target sample: the where-head should put one peak on each object.

Output: the viscot_bbox.json layout (image / conversations / source) with
`bboxes` = list of normalised xyxy boxes (2..--max_boxes), one per referenced
object. Multi-box rows deliberately have no single `bbox` key, so a
single-box where_head skips them. --single_ratio adds one-object questions on
the same images (`bboxes` of length 1 plus `bbox`, source gqa_singlebox), so
the peak count is tied to the question rather than to the image source.

    python scripts/build_gqa_multibox_data.py \
        --questions $OSS_DATA/gqa_ann/train_balanced_instructions.parquet \
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


def _plain(v):
    """numpy arrays / structs from a parquet row -> python lists / dicts."""
    if hasattr(v, "tolist") and not isinstance(v, (str, bytes)):
        v = v.tolist()
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return v


def load_questions(path):
    """{qid: question dict} from the official JSON or the lmms-lab parquet
    (train_balanced_instructions). The parquet stores each annotation part as
    [{'objectId': <word index>, 'value': <scene-graph object id>}]; convert it
    back to the JSON's {word index: object id}."""
    path = os.path.expanduser(path)
    if not path.endswith(".parquet"):
        return json.load(open(path))
    import pandas as pd
    d = pd.read_parquet(path, columns=["id", "imageId", "question", "answer", "fullAnswer",
                                       "types", "annotations"])
    qs = {}
    for r in d.itertuples(index=False):
        ann = _plain(r.annotations) or {}
        ann = {part: {str(e["objectId"]): str(e["value"]) for e in (ann.get(part) or [])}
               for part in ("question", "answer", "fullAnswer")}
        qs[str(r.id)] = {"imageId": r.imageId, "question": r.question, "answer": r.answer,
                         "fullAnswer": r.fullAnswer, "types": _plain(r.types) or {},
                         "annotations": ann}
    return qs


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


STUFF = {"grass", "sky", "wall", "ground", "floor", "road", "water", "field", "street",
         "sidewalk", "snow", "sand", "ocean", "sea", "beach", "dirt", "pavement", "background",
         "ceiling", "air", "hill", "mountain", "mountains", "trees", "tree", "bushes", "leaves",
         "cloud", "clouds", "shadow", "lake", "river", "wave", "waves", "track", "runway", "court",
         "carpet", "rug", "zoo", "park", "city", "forest", "area", "picture", "photo", "scene"}


def boxes_for(ids, sg, min_frac, max_frac, max_iou, min_boxes=2):
    """Normalised, de-duplicated boxes for the object ids (background "stuff"
    objects dropped: one patch of grass/sky is not a target); None if the
    sample does not qualify (fewer than min_boxes, a box too small/large, or
    two boxes overlapping)."""
    W, H = float(sg["width"]), float(sg["height"])
    if W <= 0 or H <= 0:
        return None, "bad_size"
    objs = sg.get("objects") or {}
    boxes = []
    for oid in ids:
        o = objs.get(oid)
        if o is None:
            continue
        if str(o.get("name", "")).strip().lower() in STUFF:
            continue
        b = [o["x"] / W, o["y"] / H, (o["x"] + o["w"]) / W, (o["y"] + o["h"]) / H]
        b = [min(max(v, 0.0), 1.0) for v in b]
        if not (b[2] > b[0] and b[3] > b[1]):
            continue
        if any(iou(b, c) > 0.8 for c in boxes):        # same object listed twice
            continue
        boxes.append(b)
    if len(boxes) < min_boxes:
        return None, f"lt{min_boxes}_boxes"
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
    ap.add_argument("--single_ratio", type=float, default=1.0,
                    help="one-object questions from the same images, per multi-box sample")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    print(f"[gqa] loading {args.scene_graphs}")
    sgs = json.load(open(os.path.expanduser(args.scene_graphs)))
    print(f"[gqa] {len(sgs)} scene graphs; loading {args.questions}")
    qs = load_questions(args.questions)
    print(f"[gqa] {len(qs)} questions")

    keys = list(qs.keys())
    random.shuffle(keys)

    def row(qid, q, img, boxes, src):
        text = str(q.get("question", "")).strip()
        ans = str(q.get("fullAnswer") or q.get("answer") or "").strip()
        if not text or not ans:
            return None
        r = {"id": f"gqa_{qid}", "image": f"{args.image_prefix}/{img}.jpg",
             "conversations": [{"from": "human", "value": f"<image>\n{text}"},
                               {"from": "gpt", "value": ans}],
             "bboxes": [[round(v, 4) for v in b] for b in boxes], "source": src}
        if len(boxes) == 1:
            r["bbox"] = r["bboxes"][0]
        return r

    # pass 1: questions referencing >= 2 objects
    c, per_img, out = Counter(), Counter(), []
    nbox, types = Counter(), Counter()
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
        r = row(qid, q, img, boxes, "gqa_multibox")
        if r is None:
            c["empty_qa"] += 1
            continue
        per_img[img] += 1
        nbox[len(boxes)] += 1
        types[(q.get("types") or {}).get("structural", "?")] += 1
        out.append(r)
        c["kept"] += 1
        if len(out) >= args.max_samples:
            break
    n_multi = len(out)

    # pass 2: one-object questions on the SAME images, so the number of peaks
    # can only be read from the question, not from the image source
    n_single_cap = int(round(args.single_ratio * n_multi))
    multi_imgs = set(per_img)
    per_img_s, cs = Counter(), Counter()
    for qid in keys if n_single_cap > 0 else []:
        q = qs[qid]
        img = str(q.get("imageId"))
        if img not in multi_imgs or per_img_s[img] >= 1:
            continue
        sg = sgs.get(img)
        ids = referenced_objects(q)
        if not ids:
            continue
        boxes, why = boxes_for(ids, sg, args.min_frac, args.max_frac, args.max_iou, min_boxes=1)
        if boxes is None or len(boxes) != 1:
            cs[why or "not_one_box"] += 1
            continue
        r = row(qid, q, img, boxes, "gqa_singlebox")
        if r is None:
            continue
        per_img_s[img] += 1
        nbox[1] += 1
        out.append(r)
        cs["kept"] += 1
        if cs["kept"] >= n_single_cap:
            break
    random.shuffle(out)

    os.makedirs(os.path.dirname(os.path.expanduser(args.out)) or ".", exist_ok=True)
    json.dump(out, open(os.path.expanduser(args.out), "w"))
    print(f"[gqa] wrote {len(out)} samples -> {args.out}")
    print("[gqa] multi-box counters:", dict(c))
    print("[gqa] single-box counters (same images):", dict(cs))
    print("[gqa] boxes per sample:", dict(sorted(nbox.items())))
    print("[gqa] structural types (multi):", dict(types.most_common()))
    multi = [s for s in out if len(s["bboxes"]) >= 2]
    if multi:
        frs = sorted((b[2] - b[0]) * (b[3] - b[1]) for s in multi for b in s["bboxes"])
        dist = sorted(abs((s["bboxes"][0][0] + s["bboxes"][0][2]) - (s["bboxes"][1][0] + s["bboxes"][1][2])) / 2
                      + abs((s["bboxes"][0][1] + s["bboxes"][0][3]) - (s["bboxes"][1][1] + s["bboxes"][1][3])) / 2
                      for s in multi)
        pct = lambda a, p: a[min(len(a) - 1, int(p * len(a)))]
        print(f"[gqa] box area frac p10/p50/p90: {pct(frs, .1):.4f} / {pct(frs, .5):.4f} / {pct(frs, .9):.4f}")
        print(f"[gqa] centre L1 distance box1-box2 p10/p50/p90: {pct(dist, .1):.3f} / {pct(dist, .5):.3f} / {pct(dist, .9):.3f}")
        for s in out[:4]:
            print("   ", s["conversations"][0]["value"].split("\n")[1], "->", s["conversations"][1]["value"], s["bboxes"])


if __name__ == "__main__":
    main()
