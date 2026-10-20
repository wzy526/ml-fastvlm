#!/usr/bin/env python3
"""Does the LM feature that already localises coarsely (where_head, lm:8)
localise finely once the coarse window is cropped and re-encoded?

The frozen HD cells do not (fine_peak_probe: score mass == chance). This
probe zooms into one window, runs the same LM prefill the coarse head was
trained on, and reads two maps on the crop grid:
  cosine        max cosine between the layer-L image cells and the question
                tokens of that same forward (no learned head)
  where-head    the trained coarse head, zero-shot on the crop

The crop-centre baseline is high on an oracle window by construction (the
window is placed on the target). The informative comparisons are:
  oracle crop   cosine / head score mass against chance (area of the GT box)
  real crop     head top-1 inside the GT box, against the coarse box's own
                centre (the 14 % this probe has to beat) and against chance

  torchrun --nproc_per_node 8 scripts/crop_lm_probe.py \
      --model_path $E/checkpoint-1000 --load $E/loc/where_lm8_mix.pt \
      --box_json $E/loc/where_lm8_mix.json \
      --train_json $OSS/extra_0916/viscot_bbox.train.json --max_doc 100 \
      --out_boxes $E/loc/crop_lm.json
"""
import argparse
import json
import math
import os
import random
import sys
import time

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import where_head as wh
from fine_head import window_at, window_hd_cells
from fine_peak_probe import cosine_map
from gdino_probe import gt_box_frac, load_vstar


def clamp_window(box, min_side=0.05):
    """Rectangular window containing `box`, each side at least min_side, inside the image."""
    x0, y0, x1, y1 = [float(v) for v in box]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = max(x1 - x0, min_side), max(y1 - y0, min_side)
    x0, x1 = cx - w / 2, cx + w / 2
    y0, y1 = cy - h / 2, cy + h / 2
    if x0 < 0:
        x1 -= x0
        x0 = 0.0
    if y0 < 0:
        y1 -= y0
        y0 = 0.0
    if x1 > 1:
        x0 -= x1 - 1
        x1 = 1.0
    if y1 > 1:
        y0 -= y1 - 1
        y1 = 1.0
    return max(0.0, x0), max(0.0, y0), min(1.0, x1), min(1.0, y1)


def crop_image(img, win):
    W, H = img.size
    box = (int(math.floor(win[0] * W)), int(math.floor(win[1] * H)),
           int(math.ceil(win[2] * W)), int(math.ceil(win[3] * H)))
    box = (max(0, box[0]), max(0, box[1]),
           min(W, max(box[0] + 2, box[2])), min(H, max(box[1] + 2, box[3])))
    return img.crop(box)


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(union, 1e-9)


def peak_stats(score, gt, win):
    """Top-1 cell of `score` (crop grid) mapped back through `win`.
    Returns top_in, chance, mass, cell box in image fractions, centre_in."""
    h, w = score.shape
    ys = torch.arange(h, device=score.device).float()
    xs = torch.arange(w, device=score.device).float()
    cy = win[1] + (ys + 0.5) / h * (win[3] - win[1])
    cx = win[0] + (xs + 0.5) / w * (win[2] - win[0])
    inside = ((cy[:, None] >= gt[1]) & (cy[:, None] <= gt[3])
              & (cx[None, :] >= gt[0]) & (cx[None, :] <= gt[2]))
    flat = int(score.reshape(-1).argmax())
    iy, ix = divmod(flat, w)
    cell = (ix / w, iy / h, (ix + 1) / w, (iy + 1) / h)
    ww, wh = win[2] - win[0], win[3] - win[1]
    box = (win[0] + cell[0] * ww, win[1] + cell[1] * wh,
           win[0] + cell[2] * ww, win[1] + cell[3] * wh)
    prob = torch.softmax(score.float().reshape(-1), 0).reshape(h, w)
    mass = float(prob[inside].sum()) if bool(inside.any()) else 0.0
    centre_in = bool(inside[h // 2, w // 2])
    return bool(inside[iy, ix]), float(inside.float().mean()), mass, box, centre_in


def centre_in_box(box, gt):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return gt[0] <= cx <= gt[2] and gt[1] <= cy <= gt[3]


def gt_in_window(gt, win):
    cx, cy = (gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2
    return win[0] <= cx <= win[2] and win[1] <= cy <= win[3]


@torch.no_grad()
def read_crop(frozen, head, img, gt, q, win):
    cells, txt = frozen.lm_features(crop_image(img, win), q)
    cos = cosine_map(cells, txt)
    if cos is None:
        raise SystemExit(f"[crop] cell C={cells.shape[-1]} txt C={txt.shape[-1]}")
    logits = head(cells, txt)
    return peak_stats(cos, gt, win), peak_stats(logits, gt, win)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--load", required=True, help="trained where_head (lm:8) weights")
    ap.add_argument("--box_json", default=None, help="coarse boxes; enables the real-crop numbers")
    ap.add_argument("--feat", default="lm:8")
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--lr_tok", type=int, default=640)
    ap.add_argument("--win_frac", type=float, default=0.2, help="oracle crop side, fraction of each axis")
    ap.add_argument("--train_json", nargs="*", default=[])
    ap.add_argument("--image_root", default="~/sft_data/train_split")
    ap.add_argument("--max_doc", type=int, default=100)
    ap.add_argument("--min_hd_cells", type=int, default=8)
    ap.add_argument("--txt_layer", type=int, default=15)
    ap.add_argument("--hd_cap", type=int, default=5017600)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--vstar_root", default="~/data/vstar_bench")
    ap.add_argument("--vstar_n", type=int, default=300)
    ap.add_argument("--out_boxes", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stagger", type=float, default=8.0)
    args = ap.parse_args()
    random.seed(args.seed)

    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    if world > 1:
        import torch.distributed as dist
        dist.init_process_group("nccl")
        cache_root = os.environ.get("TRITON_CACHE_ROOT", f"/tmp/triton_cache_{os.environ.get('USER', 'u')}")
        os.environ["TRITON_CACHE_DIR"] = os.path.join(cache_root, f"rank{local_rank}")
        os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)
    main_rank = rank == 0
    say = print if main_rank else (lambda *a, **k: None)

    layer, tok = args.feat.split(":")
    assert layer == "lm", args.feat
    frozen = wh.Frozen(args.model_path, args.txt_layer, args.hd_cap, args.attn, device, local_rank)
    time.sleep(args.stagger * local_rank)
    frozen.set_lm_view(args.model_path, int(tok), args.lr_tok)
    vstar = load_vstar(args.vstar_root, args.vstar_n)
    cells0, txt0 = frozen.lm_features(vstar[0]["image"], "what colour is the cup?")
    head = wh.XAttnHead(cells0.shape[-1], txt0.shape[-1], args.d).to(device)
    head.load_state_dict(torch.load(os.path.expanduser(args.load), map_location=device))
    head.eval()
    say(f"[crop] {args.feat} grid {tuple(cells0.shape[:2])} C={cells0.shape[-1]} "
        f"loaded {args.load} on {world} GPU(s)")

    boxes = {}
    if args.box_json:
        for r in json.load(open(os.path.expanduser(args.box_json))):
            if r.get("box"):
                boxes[int(r["i"])] = r["box"]
        say(f"[crop] coarse boxes: {len(boxes)}")

    # oracle: n, cos_hit, head_hit, centre_hit, chance, cos_mass, head_mass, head_iou
    oracle = torch.zeros(8, device=device)
    # real:   n, contains, coarse_hit, cos_hit, head_hit, head_hit_given, chance, cos_mass, head_mass, head_iou
    real = torch.zeros(10, device=device)
    out_rows = []
    for i, s in enumerate(vstar):
        if i % world != rank:
            continue
        gt = gt_box_frac(s)
        if gt is None:
            continue
        q = wh.vstar_query(s, "question")
        own = window_at((gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2, args.win_frac)
        (c_top, chance, c_mass, _, c_mid), (h_top, _, h_mass, h_box, _) = read_crop(
            frozen, head, s["image"], gt, q, own)
        oracle += torch.tensor([1, c_top, h_top, c_mid, chance, c_mass, h_mass, iou(h_box, gt)],
                               device=device, dtype=torch.float32)
        if i in boxes:
            win = clamp_window(boxes[i])
            contains = gt_in_window(gt, win)
            (rc_top, rchance, rc_mass, _, _), (rh_top, _, rh_mass, rh_box, _) = read_crop(
                frozen, head, s["image"], gt, q, win)
            real += torch.tensor([1, contains, centre_in_box(boxes[i], gt), rc_top, rh_top,
                                  contains and rh_top, rchance, rc_mass, rh_mass, iou(rh_box, gt)],
                                 device=device, dtype=torch.float32)
            out_rows.append({"i": i, "box": list(rh_box), "score": 1.0,
                             "boxes": [list(rh_box)], "scores": [1.0],
                             "in_win": bool(contains), "in_box": bool(rh_top)})

    doc = torch.zeros(8, device=device)
    if args.train_json:
        rows = wh.load_train(args.train_json, os.path.expanduser(args.image_root), 20000,
                             say if main_rank else (lambda *a, **k: None))
        random.shuffle(rows)
        picked = []
        for r in rows:
            if window_hd_cells(r["path"], args.win_frac) >= args.min_hd_cells:
                picked.append(r)
            if len(picked) >= args.max_doc:
                break
        say(f"[crop] doc rows: {len(picked)}")
        for j, r in enumerate(picked):
            if j % world != rank:
                continue
            try:
                img = Image.open(r["path"]).convert("RGB")
            except Exception:
                continue
            gt = r["box"]
            own = window_at((gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2, args.win_frac)
            (c_top, chance, c_mass, _, c_mid), (h_top, _, h_mass, h_box, _) = read_crop(
                frozen, head, img, gt, r["q"], own)
            doc += torch.tensor([1, c_top, h_top, c_mid, chance, c_mass, h_mass, iou(h_box, gt)],
                                device=device, dtype=torch.float32)

    if world > 1:
        dist.all_reduce(oracle)
        dist.all_reduce(real)
        dist.all_reduce(doc)
        gathered = [None] * world
        dist.all_gather_object(gathered, out_rows)
        out_rows = [r for part in gathered for r in part]
    n = max(float(oracle[0]), 1.0)
    say(f"[crop] V* oracle-crop n={int(oracle[0])}: "
        f"cosine top1 {int(oracle[1])}/{int(oracle[0])} ({100 * oracle[1] / n:.1f}%), "
        f"where-head top1 {int(oracle[2])}/{int(oracle[0])} ({100 * oracle[2] / n:.1f}%), "
        f"crop-centre {int(oracle[3])}/{int(oracle[0])} ({100 * oracle[3] / n:.1f}%), "
        f"chance {100 * oracle[4] / n:.1f}%")
    say(f"[crop] V* oracle-crop mass on GT cells: cosine {100 * oracle[5] / n:.1f}%, "
        f"where-head {100 * oracle[6] / n:.1f}% (chance {100 * oracle[4] / n:.1f}%); "
        f"where-head IoU {oracle[7] / n:.3f}")
    if float(real[0]) > 0:
        m = float(real[0])
        say(f"[crop] V* real-crop n={int(real[0])}: coarse window contains target {int(real[1])}/{int(real[0])}; "
            f"coarse centre in GT {int(real[2])}/{int(real[0])} ({100 * real[2] / m:.1f}%); "
            f"cosine top1 {int(real[3])}/{int(real[0])} ({100 * real[3] / m:.1f}%), "
            f"where-head top1 {int(real[4])}/{int(real[0])} ({100 * real[4] / m:.1f}%)")
        say(f"[crop] V* real-crop refine|hit {int(real[5])}/{int(real[1])} "
            f"({100 * real[5] / max(float(real[1]), 1):.1f}%); "
            f"mass cosine {100 * real[7] / m:.1f}%, where-head {100 * real[8] / m:.1f}% "
            f"(chance {100 * real[6] / m:.1f}%); where-head IoU {real[9] / m:.3f}")
        if args.out_boxes and main_rank:
            json.dump(out_rows, open(os.path.expanduser(args.out_boxes), "w"))
            say(f"[crop] boxes -> {args.out_boxes}")
    if float(doc[0]) > 0:
        k = float(doc[0])
        say(f"[crop] doc oracle-crop n={int(doc[0])}: "
            f"cosine top1 {100 * doc[1] / k:.1f}%, where-head top1 {100 * doc[2] / k:.1f}%, "
            f"crop-centre {100 * doc[3] / k:.1f}%, chance {100 * doc[4] / k:.1f}%; "
            f"mass cosine {100 * doc[5] / k:.1f}%, where-head {100 * doc[6] / k:.1f}%")


if __name__ == "__main__":
    main()
