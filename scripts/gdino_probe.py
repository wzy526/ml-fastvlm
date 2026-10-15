"""Can an open-vocabulary detector (Grounding DINO) be the OFFLINE box labeller
for the localisation SFT? Same three numbers as probe_whd_qwen35.py's
ground_window FINAL line, on V* with the sidecar target names as the phrase:
top-1 box centre in the GT window (>= --min_cells HD cells, 32 px each),
centre in the GT box, IoU with the GT box. --tile runs the detector on
overlapping tiles of that many px (small targets in 2-4k images) and keeps the
highest-scoring box over all tiles.

  python scripts/gdino_probe.py --vstar_root ~/data/vstar_bench                 # whole image
  python scripts/gdino_probe.py --vstar_root ~/data/vstar_bench --tile 800      # 800 px tiles
"""
import argparse
import json
import math
import os

import torch
from PIL import Image

FACTOR = 32
HD_CAP = 5017600


def load_vstar(root, max_samples):
    root = os.path.expanduser(root)
    samples = []
    with open(os.path.join(root, "test_questions.jsonl")) as f:
        for line in f:
            if len(samples) >= max_samples:
                break
            doc = json.loads(line)
            img_path = os.path.join(root, doc["image"])
            side = os.path.splitext(img_path)[0] + ".json"
            sd = json.load(open(side)) if os.path.isfile(side) else {}
            samples.append({
                "image": Image.open(img_path).convert("RGB"),
                "category": str(doc.get("category", "n/a")),
                "bboxes": [list(map(float, b)) for b in (sd.get("bbox") or [])],
                "targets": [str(t) for t in (sd.get("target_object") or [])],
            })
    return samples


def gt_box_frac(sample):
    boxes = sample["bboxes"]
    if not boxes:
        return None
    W, H = sample["image"].size
    return (min(b[0] for b in boxes) / W, min(b[1] for b in boxes) / H,
            max(b[0] + b[2] for b in boxes) / W, max(b[1] + b[3] for b in boxes) / H)


def hd_size(image):
    """probe_whd_qwen35.hd_target_size with the LR term irrelevant (tok 640 x 9 > cap)."""
    total = min(image.width * image.height, HD_CAP)
    aspect = image.width / image.height
    h = int(math.sqrt(total / aspect))
    w = int(h * aspect)
    return max(FACTOR, w // FACTOR * FACTOR), max(FACTOR, h // FACTOR * FACTOR)


def gt_window(gt, hd_w, hd_h, min_cells):
    x0, y0, x1, y1 = gt
    ww = min(max(x1 - x0, min_cells * FACTOR / hd_w), 1.0)
    wh = min(max(y1 - y0, min_cells * FACTOR / hd_h), 1.0)
    cx = min(max((x0 + x1) / 2, ww / 2), 1 - ww / 2)
    cy = min(max((y0 + y1) / 2, wh / 2), 1 - wh / 2)
    return cx - ww / 2, cy - wh / 2, cx + ww / 2, cy + wh / 2


def box_hits(box, gt, win):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    ix = max(0.0, min(box[2], gt[2]) - max(box[0], gt[0]))
    iy = max(0.0, min(box[3], gt[3]) - max(box[1], gt[1]))
    inter = ix * iy
    union = (box[2] - box[0]) * (box[3] - box[1]) + (gt[2] - gt[0]) * (gt[3] - gt[1]) - inter
    return ((win[0] <= cx <= win[2]) and (win[1] <= cy <= win[3]),
            (gt[0] <= cx <= gt[2]) and (gt[1] <= cy <= gt[3]),
            inter / max(union, 1e-9))


def detect(model, processor, img, text, device, thr, text_thr):
    inp = processor(images=img, text=text, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model(**inp)
    try:
        res = processor.post_process_grounded_object_detection(
            out, inp["input_ids"], threshold=thr, text_threshold=text_thr,
            target_sizes=[(img.height, img.width)])[0]
    except TypeError:                                     # older signature
        res = processor.post_process_grounded_object_detection(
            out, inp["input_ids"], box_threshold=thr, text_threshold=text_thr,
            target_sizes=[(img.height, img.width)])[0]
    return [(float(s), [float(v) for v in b])
            for s, b in zip(res["scores"].tolist(), res["boxes"].tolist())]


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
    return [(x, y, min(x + tile, W), min(y + tile, H)) for y in ys for x in xs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vstar_root", default="~/data/vstar_bench")
    ap.add_argument("--max_samples", type=int, default=300)
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-base")
    ap.add_argument("--tile", type=int, default=0, help="tile side in px (0 = whole image)")
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--thr", type=float, default=0.2, help="box score threshold")
    ap.add_argument("--text_thr", type=float, default=0.2)
    ap.add_argument("--min_cells", type=int, default=20)
    ap.add_argument("--out", default=None, help="json: per-sample top-1 box (fractions) + score")
    args = ap.parse_args()

    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model).to(device).eval()
    samples = load_vstar(args.vstar_root, args.max_samples)
    print(f"[gdino] {args.model}; {len(samples)} V* questions; tile={args.tile or 'whole'} "
          f"overlap={args.overlap} thr={args.thr}")

    n = n_det = n_win = n_box = 0
    n_win3 = n_box3 = 0
    ious = []
    per_cat = {}
    rows = []
    for i, s in enumerate(samples):
        gt = gt_box_frac(s)
        if gt is None or not s["targets"]:
            continue
        n += 1
        img = s["image"]
        W, H = img.size
        hd_w, hd_h = hd_size(img)
        win = gt_window(gt, hd_w, hd_h, args.min_cells)
        text = ". ".join(t.lower().strip() for t in s["targets"]) + "."
        dets = []
        for (x0, y0, x1, y1) in tiles_of(W, H, args.tile, args.overlap):
            crop = img if (x0, y0, x1, y1) == (0, 0, W, H) else img.crop((x0, y0, x1, y1))
            for sc, b in detect(model, processor, crop, text, device, args.thr, args.text_thr):
                dets.append((sc, [(b[0] + x0) / W, (b[1] + y0) / H, (b[2] + x0) / W, (b[3] + y0) / H]))
        dets.sort(key=lambda d: -d[0])
        cat = per_cat.setdefault(s["category"], [0, 0, 0])
        cat[0] += 1
        if not dets:
            rows.append({"i": i, "box": None, "score": None})
            continue
        n_det += 1
        in_win, in_box, iou = box_hits(dets[0][1], gt, win)
        n_win += in_win
        n_box += in_box
        ious.append(iou)
        cat[1] += in_win
        cat[2] += in_box
        hits3 = [box_hits(b, gt, win) for _, b in dets[:3]]
        n_win3 += any(h[0] for h in hits3)
        n_box3 += any(h[1] for h in hits3)
        rows.append({"i": i, "box": dets[0][1], "score": dets[0][0], "n_det": len(dets),
                     "boxes": [b for _, b in dets[:3]], "scores": [sc for sc, _ in dets[:3]],
                     "in_win": in_win, "in_box": in_box, "iou": iou})
        if (n % 25) == 0:
            print(f"  {n}: window {n_win}/{n}, in box {n_box}/{n}, IoU {sum(ious)/max(len(ious),1):.3f}")

    print(f"\n[gdino] FINAL on {n} questions: any detection {n_det}/{n}; top-1 box centre in GT window "
          f"{n_win}/{n}, in GT box {n_box}/{n}; mean IoU (detected) {sum(ious)/max(len(ious),1):.3f}; "
          f"top-3 any: window {n_win3}/{n}, box {n_box3}/{n}")
    for c, (m, w, b) in sorted(per_cat.items()):
        print(f"  {c}: n={m} window {w} box {b}")
    if args.out:
        json.dump(rows, open(os.path.expanduser(args.out), "w"))
        print(f"[gdino] saved -> {args.out}")


if __name__ == "__main__":
    main()
