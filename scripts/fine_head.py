#!/usr/bin/env python3
"""Step 2: FINE localisation inside the coarse window, on frozen HD features.

The coarse head (where_head.py, 76% V* window hit) only lands inside the GT
box 14% of the time. This head takes the HD cells (32 px) inside one window
and regresses a tight box. Everything is frozen except the head.

Three numbers, so a low score says WHICH thing failed:
  held-out (doc, oracle window)  in-domain: can HD features refine at all
  V* oracle window               same window centred on the GT: do the
                                 features transfer from doc boxes to V*
  V* real window, refine|hit     window = the coarse head's own prediction:
                                 of the cases it actually contains the target,
                                 how often the tight box lands inside the GT

Training windows are GT-centred with jitter (the target is always inside),
because refinement only matters when the coarse window is right. Rows whose
window is under --min_hd_cells HD cells are dropped: a 640 px COCO image has
~4 cells in the window, nothing to refine (viscot docs, 89% >= 1500 px, pass).

  torchrun --nproc_per_node 8 scripts/fine_head.py --model_path $E/checkpoint-1000 \
      --coarse $E/loc/where_lm8_mix.pt --coarse_feat lm:8 \
      --train_json $OSS/extra_0916/viscot_bbox.train.json \
                   $OSS/extra_1018/refcoco_bbox.json $OSS/extra_1018/coco_small_bbox.json \
      --max_per_json 40000 40000 60000 --steps 1000 --accum 4 \
      --save $E/loc/fine_mix.pt --out_boxes $E/loc/fine_mix.json
"""
import argparse
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from PIL import Image

import where_head as wh
from gdino_probe import FACTOR, box_hits, gt_box_frac, gt_window, hd_size, load_vstar  # noqa: E402


def window_at(cx, cy, frac):
    """Square window of side `frac` (fraction of each axis) centred on (cx, cy)."""
    cx = min(max(cx, frac / 2), 1 - frac / 2)
    cy = min(max(cy, frac / 2), 1 - frac / 2)
    return cx - frac / 2, cy - frac / 2, cx + frac / 2, cy + frac / 2


def sample_window(gt, frac, rng):
    """Window of side >= max(frac, box side) that contains the GT box, jittered."""
    bw, bh = gt[2] - gt[0], gt[3] - gt[1]
    ww, wh = max(frac, bw), max(frac, bh)
    cx, cy = (gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2
    jx, jy = max(0.0, (ww - bw) / 2), max(0.0, (wh - bh) / 2)
    cx = min(max(cx + rng.uniform(-jx, jx), ww / 2), 1 - ww / 2)
    cy = min(max(cy + rng.uniform(-jy, jy), wh / 2), 1 - wh / 2)
    return cx - ww / 2, cy - wh / 2, cx + ww / 2, cy + wh / 2


def window_hd_cells(path, frac):
    """HD cells along the shorter side of a `frac` window, or 0 if unreadable."""
    try:
        w, h = Image.open(path).size
    except Exception:
        return 0
    hd_w, hd_h = hd_size(type("S", (), {"width": w, "height": h})())
    return min(frac * hd_w, frac * hd_h) / FACTOR


def crop_cells(feat, win):
    """HD feature cells (H, W, C) inside the fractional window -> (h, w, C)."""
    H, W, _ = feat.shape
    x0, x1 = int(math.floor(win[0] * W)), int(math.ceil(win[2] * W))
    y0, y1 = int(math.floor(win[1] * H)), int(math.ceil(win[3] * H))
    x0, x1 = max(0, x0), min(W, max(x1, x0 + 1))
    y0, y1 = max(0, y0), min(H, max(y1, y0 + 1))
    return feat[y0:y1, x0:x1]


def to_window(box, win):
    ww, wh = win[2] - win[0], win[3] - win[1]
    return ((box[0] - win[0]) / ww, (box[1] - win[1]) / wh,
            (box[2] - win[0]) / ww, (box[3] - win[1]) / wh)


def to_image(box, win):
    ww, wh = win[2] - win[0], win[3] - win[1]
    b = (win[0] + box[0] * ww, win[1] + box[1] * wh, win[0] + box[2] * ww, win[1] + box[3] * wh)
    return tuple(min(1.0, max(0.0, v)) for v in b)


class FineHead(nn.Module):
    """HD cells inside one window + the question -> one box, as fractions of
    the window (centre x/y, width, height). Same question<->cell blocks as the
    coarse head, then the question pooled against the cells picks the box."""

    def __init__(self, c_img, c_txt, d=256, layers=2, heads=4, max_hw=64):
        super().__init__()
        self.img_in = nn.Sequential(nn.LayerNorm(c_img), nn.Linear(c_img, d))
        self.txt_in = nn.Sequential(nn.LayerNorm(c_txt), nn.Linear(c_txt, d))
        self.row, self.col = nn.Embedding(max_hw, d), nn.Embedding(max_hw, d)
        nn.init.normal_(self.row.weight, std=0.02)
        nn.init.normal_(self.col.weight, std=0.02)
        self.blocks = nn.ModuleList(wh.XBlock(d, heads) for _ in range(layers))
        self.out = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 4))

    def forward(self, feat, txt):
        H, W, _ = feat.shape
        x = self.img_in(feat.float())
        n = self.row.num_embeddings
        ri = torch.arange(H, device=x.device).clamp_max(n - 1)
        ci = torch.arange(W, device=x.device).clamp_max(n - 1)
        cells = (x + self.row(ri)[:, None] + self.col(ci)[None]).reshape(1, H * W, -1)
        t = self.txt_in(txt.float())
        tt = t[None]
        for blk in self.blocks:
            cells, tt = blk(cells, tt, H, W)
        q = t.mean(0)
        attn = torch.softmax(cells[0] @ q / math.sqrt(q.numel()), 0)
        pooled = (attn[:, None] * cells[0]).sum(0)
        cx, cy, bw, bh = torch.sigmoid(self.out(torch.cat([pooled, q])))
        return cx, cy, bw.clamp(min=1e-3), bh.clamp(min=1e-3)


def giou(a, b):
    """a, b: (4,) tensors xyxy."""
    ix = (torch.minimum(a[2], b[2]) - torch.maximum(a[0], b[0])).clamp(min=0)
    iy = (torch.minimum(a[3], b[3]) - torch.maximum(a[1], b[1])).clamp(min=0)
    inter = ix * iy
    ua = (a[2] - a[0]).clamp(min=0) * (a[3] - a[1]).clamp(min=0) \
        + (b[2] - b[0]).clamp(min=0) * (b[3] - b[1]).clamp(min=0) - inter
    iou = inter / ua.clamp(min=1e-6)
    enc = (torch.maximum(a[2], b[2]) - torch.minimum(a[0], b[0])).clamp(min=0) \
        * (torch.maximum(a[3], b[3]) - torch.minimum(a[1], b[1])).clamp(min=0)
    return iou - (enc - ua) / enc.clamp(min=1e-6)


def box_loss(pred, gt):
    """pred, gt: (cx, cy, w, h) tensors in window fractions -> L1 + (1 - GIoU)."""
    def xyxy(z):
        return torch.stack((z[0] - z[2] / 2, z[1] - z[3] / 2, z[0] + z[2] / 2, z[1] + z[3] / 2)).clamp(0, 1)
    p, g = xyxy(pred), xyxy(gt)
    return (p - g).abs().sum() + (1 - giou(p, g))


def pred_box_image(pred, win):
    cx, cy, bw, bh = (float(v) for v in pred)
    return to_image((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), win)


@torch.no_grad()
def evaluate(frozen, head, samples, frac, coarse, coarse_cells, out_path, tag, say):
    """samples: training rows (path/box/q) or V* dicts (image/prompt/bboxes).
    Reports oracle-window numbers always; real-window numbers when `coarse`
    (the loaded coarse head) is given."""
    head.eval()
    if coarse is not None:
        coarse.eval()
    n = o_box = r_hit = r_box = r_both = 0
    o_ious, r_ious, rows = [], [], []
    for i, s in enumerate(samples):
        vstar = "image" in s and not isinstance(s.get("image"), str)
        if vstar:
            img, gt, q = s["image"], gt_box_frac(s), wh.vstar_query(s, "question")
        else:
            try:
                img = Image.open(s["path"]).convert("RGB")
            except Exception:
                continue
            gt, q = s["box"], s["q"]
        if gt is None:
            continue
        n += 1
        hd = frozen.hd_features(img)
        txt = frozen.text_states(q)
        own = window_at((gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2, frac)
        pb = pred_box_image(head(crop_cells(hd, own), txt), own)
        h = box_hits(pb, gt, gt_window(gt, *hd_size(img), 20))
        o_box += h[1]
        o_ious.append(h[2])
        rec = {"i": i, "box": list(pb)}
        if coarse is not None and vstar:
            feat, ctxt = frozen.lm_features(img, q)
            cb = wh.top_boxes(coarse(feat, ctxt), 1, coarse_cells)[0][1]
            win = window_at((cb[0] + cb[2]) / 2, (cb[1] + cb[3]) / 2, frac)
            contains = win[0] <= (gt[0] + gt[2]) / 2 <= win[2] and win[1] <= (gt[1] + gt[3]) / 2 <= win[3]
            rb = pred_box_image(head(crop_cells(hd, win), txt), win)
            rh = box_hits(rb, gt, gt_window(gt, *hd_size(img), 20))
            r_hit += contains
            r_box += rh[1]
            r_both += contains and rh[1]
            r_ious.append(rh[2])
            rec = {"i": i, "box": list(rb), "score": 1.0, "boxes": [list(rb)], "scores": [1.0],
                   "in_win": bool(contains), "in_box": bool(rh[1])}
        rows.append(rec)
    say(f"[fine]{tag} oracle-win {n}: in-box {o_box}/{n} ({100 * o_box / max(n, 1):.1f}%), "
        f"IoU {sum(o_ious) / max(len(o_ious), 1):.3f}")
    if r_ious:
        say(f"[fine]{tag} real-win: coarse window contains target {r_hit}/{n}; "
            f"refine|hit {r_both}/{r_hit} ({100 * r_both / max(r_hit, 1):.1f}%); "
            f"overall in-box {r_box}/{n} ({100 * r_box / max(n, 1):.1f}%), "
            f"IoU {sum(r_ious) / len(r_ious):.3f}")
        if out_path:
            import json
            json.dump(rows, open(os.path.expanduser(out_path), "w"))
            say(f"[fine] boxes -> {out_path}")
    head.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--train_json", nargs="*", default=[])
    ap.add_argument("--image_root", default="~/sft_data/train_split")
    ap.add_argument("--max_per_json", type=int, nargs="+", default=[20000])
    ap.add_argument("--coarse", default=None, help="coarse where_head weights; enables the real-window eval")
    ap.add_argument("--coarse_feat", default="lm:8")
    ap.add_argument("--coarse_d", type=int, default=384)
    ap.add_argument("--coarse_cells", type=int, default=4, help="square side the coarse head was evaluated with")
    ap.add_argument("--win_frac", type=float, default=0.2, help="window side as a fraction of each image axis")
    ap.add_argument("--min_hd_cells", type=int, default=8, help="drop rows whose window is smaller than this")
    ap.add_argument("--heldout", type=int, default=200)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--txt_layer", type=int, default=15)
    ap.add_argument("--hd_cap", type=int, default=5017600)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--vstar_root", default="~/data/vstar_bench")
    ap.add_argument("--vstar_n", type=int, default=300)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--save", default=None)
    ap.add_argument("--load", default=None)
    ap.add_argument("--out_boxes", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stagger", type=float, default=8.0)
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    if world > 1:
        import torch.distributed as dist
        dist.init_process_group("nccl")
        os.environ["TRITON_CACHE_DIR"] = os.path.expanduser(f"~/.triton_cache/rank{local_rank}")
        os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)
    main_rank = rank == 0
    say = print if main_rank else (lambda *a, **k: None)
    barrier = (lambda: dist.barrier()) if world > 1 else (lambda: None)

    frozen = wh.Frozen(args.model_path, args.txt_layer, args.hd_cap, args.attn, device, local_rank)
    time.sleep(args.stagger * local_rank)
    vstar = load_vstar(args.vstar_root, args.vstar_n) if main_rank else []

    img0 = vstar[0]["image"] if main_rank else Image.new("RGB", (640, 480))
    hd0 = frozen.hd_features(img0)
    txt0 = frozen.text_states("what colour is the cup?")
    head = FineHead(hd0.shape[-1], txt0.shape[-1], args.d, args.layers, args.heads).to(device)
    say(f"[fine] head d={args.d}: {sum(p.numel() for p in head.parameters()) / 1e6:.2f} M params; "
        f"HD C={hd0.shape[-1]} grid {tuple(hd0.shape[:2])}, txt C={txt0.shape[-1]}; "
        f"window {args.win_frac:.0%} of each axis; "
        f"{world} GPU(s) -> {args.steps} x {args.accum} x {world} = {args.steps * args.accum * world} samples")

    coarse = None
    if args.coarse and main_rank:
        layer, tok = args.coarse_feat.split(":")
        assert layer == "lm"
        frozen.set_lm_view(args.model_path, int(tok), 640)
        f, t = frozen.lm_features(img0, "what colour is the cup?")
        coarse = wh.XAttnHead(f.shape[-1], t.shape[-1], args.coarse_d).to(device)
        coarse.load_state_dict(torch.load(os.path.expanduser(args.coarse), map_location=device))
        coarse.eval()
        say(f"[fine] coarse head loaded from {args.coarse} ({args.coarse_feat}, "
            f"grid {tuple(f.shape[:2])})")

    if args.load:
        head.load_state_dict(torch.load(os.path.expanduser(args.load), map_location=device))
        say(f"[fine] loaded {args.load}")
        if not args.train_json:
            if main_rank:
                evaluate(frozen, head, vstar, args.win_frac, coarse, args.coarse_cells,
                         args.out_boxes, " [loaded]", say)
            return

    rows = wh.load_train(args.train_json, os.path.expanduser(args.image_root), args.max_per_json, say)
    if main_rank:
        kept = []
        for n, r in enumerate(rows):
            if n % 10000 == 0:
                say(f"[fine] scanning resolution {n}/{len(rows)}", flush=True)
            if window_hd_cells(r["path"], args.win_frac) >= args.min_hd_cells:
                kept.append(r)
        say(f"[fine] resolution: kept {len(kept)}/{len(rows)} "
            f"(window >= {args.min_hd_cells} HD cells; 640 px images are dropped)")
    else:
        kept = None
    if world > 1:
        box = [kept]
        dist.broadcast_object_list(box, src=0)
        kept = box[0]
    random.shuffle(kept)
    held, rows = kept[: args.heldout], kept[args.heldout:]
    rows = rows[rank::world]
    say(f"[fine] {len(rows)} train rows per rank, {len(held)} held-out")
    if not rows:
        raise SystemExit("[fine] nothing to train on after the resolution filter")

    if main_rank:
        evaluate(frozen, head, held, args.win_frac, None, args.coarse_cells, None, " [untrained] held-out", say)
        evaluate(frozen, head, vstar, args.win_frac, coarse, args.coarse_cells, None, " [untrained]", say)
    barrier()

    train_head = head
    if world > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        train_head = DDP(head, device_ids=[local_rank])
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)))
    rng, t0, seen, run_loss, skipped = random.Random(args.seed + rank), time.time(), 0, 0.0, 0
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        for _ in range(args.accum):
            while True:
                r = rows[seen % len(rows)]
                seen += 1
                try:
                    img = Image.open(r["path"]).convert("RGB")
                    break
                except Exception:
                    skipped += 1
            win = sample_window(r["box"], args.win_frac, rng)
            pred = train_head(crop_cells(frozen.hd_features(img), win), frozen.text_states(r["q"]))
            gt_w = to_window(r["box"], win)
            inside = 0.0 <= (gt_w[0] + gt_w[2]) / 2 <= 1.0 and 0.0 <= (gt_w[1] + gt_w[3]) / 2 <= 1.0
            if inside:
                gt_t = torch.tensor(((gt_w[0] + gt_w[2]) / 2, (gt_w[1] + gt_w[3]) / 2,
                                     gt_w[2] - gt_w[0], gt_w[3] - gt_w[1]),
                                    device=pred[0].device, dtype=torch.float32).clamp(1e-3, 1)
                loss = box_loss(torch.stack(pred), gt_t) / args.accum
            else:
                skipped += 1
                loss = torch.stack(pred).sum() * 0.0
            loss.backward()
            run_loss += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 50 == 0:
            say(f"[fine] step {step}/{args.steps} loss {run_loss / 50:.3f} "
                f"{(time.time() - t0) / step:.2f}s/step skipped {skipped}", flush=True)
            run_loss = 0.0
        if step % args.eval_every == 0 or step == args.steps:
            if main_rank:
                evaluate(frozen, head, held, args.win_frac, None, args.coarse_cells, None,
                         f" [step {step}] held-out", say)
                evaluate(frozen, head, vstar, args.win_frac, coarse, args.coarse_cells,
                         args.out_boxes if step == args.steps else None, f" [step {step}]", say)
                if args.save:
                    torch.save(head.state_dict(), os.path.expanduser(args.save))
            barrier()


if __name__ == "__main__":
    main()
