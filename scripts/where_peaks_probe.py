#!/usr/bin/env python3
"""Does the where-head's heat map have a usable SECOND peak?

Multi-group sampling (one DAT grid per peak) only pays off if, on questions
about two objects, the head puts a second local maximum on the second object.
The head was trained on single boxes (V* two-object questions used the union
box), so this is measured, not assumed. Zero extra GPU work: the head's
heat map is captured from the live prefill (where_crop_generate's own prompt
and LR view) and the forward is aborted right after the head.

Per question: NMS peaks of the 25x25 softmax map (3x3 local maxima, Chebyshev
distance >= --min_dist cells, prob >= --tau x main peak, at most --max_peaks).
V* has GT boxes: a peak "covers" a box when the 0.32-side window around the
peak contains the box centre; we report whether peak 2 covers a box peak 1
does not (the second object).

  for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i setsid nohup python scripts/where_peaks_probe.py \
      --model_path $E/checkpoint-1000 --where_head_pt $E/loc/where_lm8_ce20.pt \
      --dataset hrbench4k --max_samples 800 --shard $i --shards 4 \
      --out $E/loc/peaks/hr4k_$i.json > $E/loc/peaks/hr4k_$i.log 2>&1 < /dev/null & disown; done
  python scripts/where_peaks_probe.py --summarize "$E/loc/peaks/hr4k_*.json"
"""
import argparse
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

WIN_SIDE = 0.32          # the live crop window (out_cells 4 x ground_pad 2 / 25)


def softmax2d(logits):
    flat = [v for row in logits for v in row]
    mx = max(flat)
    ex = [math.exp(v - mx) for v in flat]
    z = sum(ex)
    gw = len(logits[0])
    return [[ex[i * gw + j] / z for j in range(gw)] for i in range(len(logits))]


def nms_peaks(p, min_dist, tau, max_peaks):
    """3x3 local maxima -> greedy NMS. Returns [(prob, cy, cx)] sorted by prob,
    all with prob >= tau * main and Chebyshev distance >= min_dist from every
    kept peak."""
    gh, gw = len(p), len(p[0])
    cands = []
    for i in range(gh):
        for j in range(gw):
            v = p[i][j]
            ok = True
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == dj == 0:
                        continue
                    ii, jj = i + di, j + dj
                    if 0 <= ii < gh and 0 <= jj < gw and p[ii][jj] > v:
                        ok = False
            if ok:
                cands.append((v, i, j))
    cands.sort(reverse=True)
    kept = []
    for v, i, j in cands:
        if kept and v < tau * kept[0][0]:
            break
        if all(max(abs(i - ki), abs(j - kj)) >= min_dist for _, ki, kj in kept):
            kept.append((v, i, j))
        if len(kept) >= max_peaks:
            break
    return kept


def window_mass(p, cy, cx, side_cells):
    gh, gw = len(p), len(p[0])
    h = side_cells / 2.0
    return sum(p[i][j] for i in range(gh) for j in range(gw)
               if abs(i + 0.5 - (cy + 0.5)) <= h and abs(j + 0.5 - (cx + 0.5)) <= h)


def covers(peak, box_c, gh, gw):
    """Window of WIN_SIDE around the peak's cell centre contains the box centre."""
    _, cy, cx = peak
    px, py = (cx + 0.5) / gw, (cy + 0.5) / gh
    return abs(px - box_c[0]) <= WIN_SIDE / 2 and abs(py - box_c[1]) <= WIN_SIDE / 2


def summarize(pattern, taus=(0.2, 0.3, 0.5)):
    rows = []
    for f in sorted(glob.glob(os.path.expanduser(pattern))):
        rows.extend(json.load(open(f)))
    if not rows:
        raise SystemExit(f"no json matched {pattern}")
    cats = sorted({r["cat"] for r in rows})
    md = rows[0]["min_dist"]
    print(f"{len(rows)} questions; peaks: 3x3 maxima, min_dist {md} cells, max {rows[0]['max_peaks']}")
    print(f"{'category':20s} {'n':>5s} " + " ".join(f"{'>=2pk@' + str(t):>9s}" for t in taus)
          + f" {'p2/p1 med':>10s} {'dist med':>9s} {'mass1 win':>10s}")
    for cat in cats + ["ALL"]:
        rs = rows if cat == "ALL" else [r for r in rows if r["cat"] == cat]
        n = len(rs)
        frac = []
        for t in taus:
            frac.append(sum(1 for r in rs if len(r["peaks"]) >= 2 and r["peaks"][1][0] >= t * r["peaks"][0][0]) / max(n, 1))
        ratios = sorted(r["peaks"][1][0] / r["peaks"][0][0] for r in rs if len(r["peaks"]) >= 2)
        dists = sorted(math.hypot(r["peaks"][1][1] - r["peaks"][0][1], r["peaks"][1][2] - r["peaks"][0][2])
                       for r in rs if len(r["peaks"]) >= 2)
        m1 = sorted(r["mass1_win"] for r in rs)
        med = lambda a: a[len(a) // 2] if a else float("nan")
        print(f"{cat:20s} {n:5d} " + " ".join(f"{100 * x:8.1f}%" for x in frac)
              + f" {med(ratios):10.2f} {med(dists):9.1f} {med(m1):10.2f}")
    if any("cover" in r for r in rows):
        print("\nV* with GT boxes: does peak 2 cover a box that peak 1 misses?")
        for cat in cats:
            rs = [r for r in rows if r["cat"] == cat and "cover" in r]
            two = [r for r in rs if len(r["boxes_c"]) >= 2]
            miss1 = [r for r in two if not all(r["cover"][0])]
            rescued = [r for r in miss1 if len(r["cover"]) >= 2
                       and any(c2 and not c1 for c1, c2 in zip(r["cover"][0], r["cover"][1]))]
            print(f"  {cat:20s} n={len(rs)} two-box {len(two)}; peak1 misses a box {len(miss1)}; "
                  f"peak2 covers that box {len(rescued)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summarize", default=None)
    ap.add_argument("--model_path")
    ap.add_argument("--where_head_pt", default=None)
    ap.add_argument("--dataset", choices=["vstar", "hrbench4k", "hrbench8k"], default="hrbench4k")
    ap.add_argument("--vstar_root", default="~/data/vstar_bench")
    ap.add_argument("--hrbench_parquet", default=None)
    ap.add_argument("--max_samples", type=int, default=800)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--lr_tok", type=int, default=640)
    ap.add_argument("--min_dist", type=int, default=4, help="cells between peaks (window = 8 cells)")
    ap.add_argument("--tau", type=float, default=0.05, help="record peaks down to this ratio; summarize re-thresholds")
    ap.add_argument("--max_peaks", type=int, default=4)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.summarize:
        summarize(args.summarize)
        return

    import torch
    from transformers import AutoProcessor
    from probe_whd_qwen35 import load_samples
    import llava.model.language_model.modeling_qwen3_5_dat as dat_mod

    model = dat_mod.Qwen3_5DATForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation=args.attn).eval()
    if not model._where_head_loaded and not args.where_head_pt:
        raise SystemExit("no where-head weights")
    if args.where_head_pt:
        sd = torch.load(os.path.expanduser(args.where_head_pt), map_location="cpu")
        sd = {k[len("where_head."):] if k.startswith("where_head.") else k: v for k, v in sd.items()}
        model.where_head.load_state_dict(sd)
        model._where_head_loaded = True
    processor = AutoProcessor.from_pretrained(args.model_path, min_pixels=1024, max_pixels=args.lr_tok * 1024)
    hr_processor = AutoProcessor.from_pretrained(args.model_path, min_pixels=1024, max_pixels=100_000_000)

    probe_args = argparse.Namespace(dataset=args.dataset, vstar_root=args.vstar_root,
                                    hrbench_parquet=args.hrbench_parquet, image_folder=None,
                                    max_samples=args.max_samples)
    samples = load_samples(probe_args)
    mine = [(i, s) for i, s in enumerate(samples) if i % args.shards == args.shard]
    print(f"[peaks] {args.dataset} shard {args.shard}/{args.shards}: {len(mine)} questions", flush=True)

    class _Stop(Exception):
        pass

    cap = {}
    orig_top_box = dat_mod._where_top_box

    def capture(logits, cells):
        cap["logits"] = logits.float().tolist()
        raise _Stop()

    dat_mod._where_top_box = capture
    rows = []
    for gi, s in mine:
        try:
            model.where_crop_generate(s["image"], s["prompt"], processor, hr_processor,
                                      where_question=s["prompt"], head_prefix="train", single_pass=False)
        except _Stop:
            pass
        p = softmax2d(cap["logits"])
        gh, gw = len(p), len(p[0])
        peaks = nms_peaks(p, args.min_dist, args.tau, args.max_peaks)
        row = {"i": gi, "cat": s["category"], "grid": [gh, gw],
               "peaks": [[round(v, 5), cy, cx] for v, cy, cx in peaks],
               "mass1_win": round(window_mass(p, peaks[0][1], peaks[0][2], WIN_SIDE * gw), 4),
               "min_dist": args.min_dist, "max_peaks": args.max_peaks}
        if s.get("bboxes"):
            W, H = s["image"].size
            boxes_c = [((b[0] + b[2] / 2) / W, (b[1] + b[3] / 2) / H) for b in s["bboxes"]]
            row["boxes_c"] = boxes_c
            row["cover"] = [[covers(pk, c, gh, gw) for c in boxes_c] for pk in peaks]
        rows.append(row)
        pk = row["peaks"]
        print(f"[peaks] {gi} cat={row['cat']} n_peaks={len(pk)} p1={pk[0][0]:.3f}"
              + (f" p2/p1={pk[1][0] / pk[0][0]:.2f} dist={math.hypot(pk[1][1] - pk[0][1], pk[1][2] - pk[0][2]):.1f}"
                 if len(pk) > 1 else "")
              + f" mass1_win={row['mass1_win']:.2f}", flush=True)
    dat_mod._where_top_box = orig_top_box
    if args.out:
        os.makedirs(os.path.dirname(os.path.expanduser(args.out)) or ".", exist_ok=True)
        json.dump(rows, open(os.path.expanduser(args.out), "w"))
        print(f"[peaks] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
