#!/usr/bin/env python3
"""Train the coarse where-head on CROPPED windows, then test fine localisation.

Zero-shot on a crop failed (V* in-box 14% -> 24%, refine|hit 31%). The head
had only ever seen full images. Here each training sample is a window that
contains the GT box (jittered, side --win_frac), re-encoded with the same LM
prefill, and the head is trained to put its mass on the GT cells inside that
crop. Nothing in the model changes.

The bar is the same one the zero-shot run missed: on V*, using the coarse
head's own window as the crop, refine|hit > 60%. Step 0 must reproduce the
zero-shot numbers (~35% oracle top-1, ~24% real top-1); if it does not, the
crop or the loaded weights are wrong.

  torchrun --nproc_per_node 8 scripts/crop_where_train.py \
      --model_path $E/checkpoint-1000 --load $E/loc/where_lm8_mix.pt \
      --box_json $E/loc/where_lm8_mix.json \
      --train_json $OSS/extra_0916/viscot_bbox.train.json \
                   $OSS/extra_1018/refcoco_bbox.json $OSS/extra_1018/coco_small_bbox.json \
      --max_per_json 40000 40000 60000 --steps 1000 --accum 4 \
      --save $E/loc/crop_where.pt --out_boxes $E/loc/crop_where.json
"""
import argparse
import json
import math
import os
import random
import sys
import time
from datetime import timedelta

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import where_head as wh
from crop_lm_probe import centre_in_box, clamp_window, crop_image, gt_in_window, iou, read_crop
from fine_head import sample_window, window_at
from gdino_probe import gt_box_frac, load_vstar


def box_in_window(gt, win):
    """GT box as fractions of the window, clamped to the window."""
    ww, wh = max(win[2] - win[0], 1e-6), max(win[3] - win[1], 1e-6)
    return tuple(min(1.0, max(0.0, v)) for v in (
        (gt[0] - win[0]) / ww, (gt[1] - win[1]) / wh, (gt[2] - win[0]) / ww, (gt[3] - win[1]) / wh))


@torch.no_grad()
def evaluate(frozen, head, vstar, boxes, held, frac, say, tag):
    head.eval()
    n = c_hit = h_hit = centre = 0
    chance = c_mass = h_mass = 0.0
    rn = contains = coarse_hit = rh_hit = rh_given = 0
    r_chance = r_mass = 0.0
    rows = []
    for i, s in enumerate(vstar):
        gt = gt_box_frac(s)
        if gt is None:
            continue
        q = wh.vstar_query(s, "question")
        own = window_at((gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2, frac)
        (ct, ch, cm, _, mid), (ht, _, hm, hb, _) = read_crop(frozen, head, s["image"], gt, q, own)
        n += 1
        c_hit += ct
        h_hit += ht
        centre += mid
        chance += ch
        c_mass += cm
        h_mass += hm
        if i in boxes:
            win = clamp_window(boxes[i])
            inside = gt_in_window(gt, win)
            (_, _, _, _, _), (rht, rch, rhm, rhb, _) = read_crop(frozen, head, s["image"], gt, q, win)
            rn += 1
            contains += inside
            coarse_hit += centre_in_box(boxes[i], gt)
            rh_hit += rht
            rh_given += inside and rht
            r_chance += rch
            r_mass += rhm
            rows.append({"i": i, "box": list(rhb), "score": 1.0, "boxes": [list(rhb)], "scores": [1.0],
                         "in_win": bool(inside), "in_box": bool(rht), "iou": iou(rhb, gt)})
    say(f"[ctrain]{tag} V* oracle-crop n={n}: cosine {c_hit}/{n} ({100 * c_hit / max(n, 1):.1f}%), "
        f"where-head {h_hit}/{n} ({100 * h_hit / max(n, 1):.1f}%), "
        f"crop-centre {centre}/{n} ({100 * centre / max(n, 1):.1f}%), "
        f"chance {100 * chance / max(n, 1):.1f}%, mass {100 * h_mass / max(n, 1):.1f}%")
    if rn:
        say(f"[ctrain]{tag} V* real-crop n={rn}: contains {contains}/{rn}, "
            f"coarse centre {coarse_hit}/{rn} ({100 * coarse_hit / rn:.1f}%), "
            f"where-head {rh_hit}/{rn} ({100 * rh_hit / rn:.1f}%), "
            f"refine|hit {rh_given}/{contains} ({100 * rh_given / max(contains, 1):.1f}%), "
            f"mass {100 * r_mass / rn:.1f}% (chance {100 * r_chance / rn:.1f}%)")
    hn = hh = 0
    for r in held:
        try:
            img = Image.open(r["path"]).convert("RGB")
        except Exception:
            continue
        gt = r["box"]
        own = window_at((gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2, frac)
        (_, _, _, _, _), (ht, _, _, _, _) = read_crop(frozen, head, img, gt, r["q"], own)
        hn += 1
        hh += ht
    if hn:
        say(f"[ctrain]{tag} held-out oracle-crop {hn}: where-head {hh}/{hn} ({100 * hh / hn:.1f}%)")
    head.train()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--train_json", nargs="*", default=[])
    ap.add_argument("--image_root", default="~/sft_data/train_split")
    ap.add_argument("--max_per_json", type=int, nargs="+", default=[20000])
    ap.add_argument("--load", default=None)
    ap.add_argument("--box_json", default=None)
    ap.add_argument("--feat", default="lm:8")
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--lr_tok", type=int, default=640)
    ap.add_argument("--win_frac", type=float, default=0.2)
    ap.add_argument("--dilate", type=int, default=1)
    ap.add_argument("--heldout", type=int, default=200)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--txt_layer", type=int, default=15)
    ap.add_argument("--hd_cap", type=int, default=5017600)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--vstar_root", default="~/data/vstar_bench")
    ap.add_argument("--vstar_n", type=int, default=300)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--save", default=None)
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
        dist.init_process_group("nccl", timeout=timedelta(hours=3))
        cache_root = os.environ.get("TRITON_CACHE_ROOT", f"/tmp/triton_cache_{os.environ.get('USER', 'u')}")
        os.environ["TRITON_CACHE_DIR"] = os.path.join(cache_root, f"rank{local_rank}")
        os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)
    main_rank = rank == 0
    say = print if main_rank else (lambda *a, **k: None)
    barrier = (lambda: dist.barrier()) if world > 1 else (lambda: None)

    layer, tok = args.feat.split(":")
    assert layer == "lm", args.feat
    frozen = wh.Frozen(args.model_path, args.txt_layer, args.hd_cap, args.attn, device, local_rank)
    time.sleep(args.stagger * local_rank)
    frozen.set_lm_view(args.model_path, int(tok), args.lr_tok)
    vstar = load_vstar(args.vstar_root, args.vstar_n) if main_rank else []
    probe_img = vstar[0]["image"] if main_rank else Image.new("RGB", (640, 480))
    cells0, txt0 = frozen.lm_features(probe_img, "what colour is the cup?")
    head = wh.XAttnHead(cells0.shape[-1], txt0.shape[-1], args.d).to(device)
    if args.load:
        head.load_state_dict(torch.load(os.path.expanduser(args.load), map_location=device))
        say(f"[ctrain] loaded {args.load}")
    say(f"[ctrain] {args.feat} grid {tuple(cells0.shape[:2])} C={cells0.shape[-1]} "
        f"window {args.win_frac:.0%}; {world} GPU(s) -> {args.steps} x {args.accum} x {world} samples")

    boxes = {}
    if args.box_json and main_rank:
        for r in json.load(open(os.path.expanduser(args.box_json))):
            if r.get("box"):
                boxes[int(r["i"])] = r["box"]
        say(f"[ctrain] coarse boxes: {len(boxes)}")

    rows = wh.load_train(args.train_json, os.path.expanduser(args.image_root), args.max_per_json, say)
    random.shuffle(rows)
    held, rows = rows[: args.heldout], rows[args.heldout:]
    rows = rows[rank::world]
    say(f"[ctrain] {len(rows)} train rows per rank, {len(held)} held-out")
    if not rows:
        raise SystemExit("[ctrain] no training rows")

    if main_rank:
        evaluate(frozen, head, vstar, boxes, held[:16], args.win_frac, say, " [untrained]")
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
            logits = train_head(*frozen.lm_features(crop_image(img, win), r["q"]))
            mask = wh.cell_targets(box_in_window(r["box"], win), *logits.shape, args.dilate)
            if float(mask.sum()) == 0:
                skipped += 1
                loss = logits.sum() * 0.0
            else:
                loss = wh.head_loss(logits, mask) / args.accum
            loss.backward()
            run_loss += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 50 == 0:
            say(f"[ctrain] step {step}/{args.steps} loss {run_loss / 50:.3f} "
                f"{(time.time() - t0) / step:.2f}s/step skipped {skipped}", flush=True)
            run_loss = 0.0
        if step % args.eval_every == 0 or step == args.steps:
            if main_rank:
                out = evaluate(frozen, head, vstar, boxes, held, args.win_frac, say, f" [step {step}]")
                if args.save:
                    torch.save(head.state_dict(), os.path.expanduser(args.save))
                if args.out_boxes and step == args.steps:
                    json.dump(out, open(os.path.expanduser(args.out_boxes), "w"))
                    say(f"[ctrain] boxes -> {args.out_boxes}")
            barrier()


if __name__ == "__main__":
    main()
