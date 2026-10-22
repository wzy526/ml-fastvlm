#!/usr/bin/env python3
"""No training: inside a window centred on the target, does any per-cell score
peak on the target, or is the map flat?

Two scores, same window (side --win_frac, centred on the GT box):
  cosine        max cosine between the frozen HD cell and the question tokens.
                The two vectors were never aligned, so this is a lower bound.
  trained-attn  the attention map inside the trained fine head (fine_head.py),
                i.e. what the box regression actually looked at.

Each is compared with two baselines: the centre cell of the window, and chance
(the fraction of window cells whose centre lies inside the GT box). Also the
share of the score mass sitting on those GT cells: above chance means the map
knows where the target is even if the single highest cell misses it.

  torchrun --nproc_per_node 8 scripts/fine_peak_probe.py \
      --model_path $E/checkpoint-1000 --load $E/loc/fine_mix.pt \
      --train_json $OSS/extra_0916/viscot_bbox.train.json --max_doc 100
"""
import argparse
import os
import random
import sys
import time

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # gdino_probe lives in scripts/

import fine_head as fh
import where_head as wh
from gdino_probe import gt_box_frac, load_vstar


def cosine_map(feat, txt):
    """(H, W) max cosine of each cell against the question tokens, or None."""
    if feat.shape[-1] != txt.shape[-1]:
        return None
    c = feat.float()
    c = c / c.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    t = txt.float()
    t = t / t.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return torch.einsum("hwc,tc->hwt", c, t).max(-1).values


def cell_hits(score, gt, origin, full_hw):
    """Top-1 cell centre inside the GT box, centre-cell inside, chance
    (fraction of cells inside), and the share of softmax mass inside."""
    h, w = score.shape
    y0, x0 = origin
    H, W = full_hw
    ys = torch.arange(h, device=score.device)
    xs = torch.arange(w, device=score.device)
    cy = (y0 + ys.float() + 0.5) / H
    cx = (x0 + xs.float() + 0.5) / W
    inside = (cy[:, None] >= gt[1]) & (cy[:, None] <= gt[3]) & (cx[None, :] >= gt[0]) & (cx[None, :] <= gt[2])
    iy, ix = divmod(int(score.argmax()), w)
    top_in = bool(inside[iy, ix])
    centre_in = bool(inside[h // 2, w // 2])
    chance = float(inside.float().mean())
    prob = torch.softmax(score.float().reshape(-1), 0).reshape(h, w)
    mass = float(prob[inside].sum()) if bool(inside.any()) else 0.0
    return top_in, centre_in, chance, mass


@torch.no_grad()
def accumulate(frozen, head, items, frac, device):
    """items: iterable of (image, gt box fractions, question). Returns a
    (n, cos_hit, attn_hit, centre_hit, chance_sum, cos_mass, attn_mass) tensor."""
    acc = torch.zeros(7, device=device)
    for img, gt, q in items:
        if gt is None:
            continue
        hd = frozen.hd_features(img)
        txt = frozen.text_states(q)
        H, W, _ = hd.shape
        win = fh.window_at((gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2, frac)
        y0, y1, x0, x1 = fh.crop_bounds(H, W, win)
        crop = hd[y0:y1, x0:x1]
        cos = cosine_map(crop, txt)
        if cos is None:
            raise SystemExit(f"[peak] HD C={crop.shape[-1]} txt C={txt.shape[-1]}: cosine is undefined")
        cos_hit, centre_hit, chance, cos_mass = cell_hits(cos, gt, (y0, x0), (H, W))
        attn = head.cell_scores(crop, txt)
        attn_hit, _, _, attn_mass = cell_hits(attn, gt, (y0, x0), (H, W))
        acc += torch.tensor([1, cos_hit, attn_hit, centre_hit, chance, cos_mass, attn_mass],
                            device=device, dtype=torch.float32)
    return acc


def report(name, acc, say):
    n = max(float(acc[0]), 1.0)
    say(f"[peak] {name} n={int(acc[0])}: "
        f"cosine top1 {int(acc[1])}/{int(acc[0])} ({100 * acc[1] / n:.1f}%), "
        f"trained-attn top1 {int(acc[2])}/{int(acc[0])} ({100 * acc[2] / n:.1f}%), "
        f"centre cell {int(acc[3])}/{int(acc[0])} ({100 * acc[3] / n:.1f}%), "
        f"chance {100 * acc[4] / n:.1f}%")
    say(f"[peak] {name} score mass on the GT cells: cosine {100 * acc[5] / n:.1f}%, "
        f"trained-attn {100 * acc[6] / n:.1f}% (chance {100 * acc[4] / n:.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--load", required=True, help="trained fine_head weights")
    ap.add_argument("--train_json", nargs="*", default=[], help="doc boxes, for the in-domain number")
    ap.add_argument("--image_root", default="~/sft_data/train_split")
    ap.add_argument("--max_doc", type=int, default=100)
    ap.add_argument("--win_frac", type=float, default=0.2)
    ap.add_argument("--min_hd_cells", type=int, default=8)
    ap.add_argument("--txt_layer", type=int, default=15)
    ap.add_argument("--hd_cap", type=int, default=5017600)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--vstar_root", default="~/data/vstar_bench")
    ap.add_argument("--vstar_n", type=int, default=300)
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

    frozen = wh.Frozen(args.model_path, args.txt_layer, args.hd_cap, args.attn, device, local_rank)
    time.sleep(args.stagger * local_rank)
    vstar = load_vstar(args.vstar_root, args.vstar_n)
    hd0 = frozen.hd_features(vstar[0]["image"])
    txt0 = frozen.text_states("what colour is the cup?")
    head = fh.FineHead(hd0.shape[-1], txt0.shape[-1], args.d).to(device)
    head.load_state_dict(torch.load(os.path.expanduser(args.load), map_location=device))
    head.eval()
    say(f"[peak] HD C={hd0.shape[-1]} txt C={txt0.shape[-1]} window {args.win_frac:.0%} "
        f"loaded {args.load} on {world} GPU(s)")

    items = [(s["image"], gt_box_frac(s), wh.vstar_query(s, "question")) for s in vstar]
    acc = accumulate(frozen, head, items[rank::world], args.win_frac, device)
    if world > 1:
        dist.all_reduce(acc)
    report("V*", acc, say)

    if args.train_json:
        rows = wh.load_train(args.train_json, os.path.expanduser(args.image_root), 20000,
                             say if main_rank else (lambda *a, **k: None))
        random.shuffle(rows)
        picked = []
        for r in rows:
            if fh.window_hd_cells(r["path"], args.win_frac) >= args.min_hd_cells:
                picked.append(r)
            if len(picked) >= args.max_doc:
                break
        say(f"[peak] doc rows with a >= {args.min_hd_cells}-cell window: {len(picked)}")
        doc = []
        for r in picked[rank::world]:
            try:
                doc.append((Image.open(r["path"]).convert("RGB"), r["box"], r["q"]))
            except Exception:
                continue
        acc = accumulate(frozen, head, doc, args.win_frac, device)
        if world > 1:
            dist.all_reduce(acc)
        report("doc", acc, say)


if __name__ == "__main__":
    main()
