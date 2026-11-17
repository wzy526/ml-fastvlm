#!/usr/bin/env python
"""Where the final Q-LAS sampler puts its tokens on HR-Bench.

One row per question: the full image with the window, the learned density
inside the window, and the token centres of the 400-token mosaic. Everything
comes from ldat_eval.infer.LDatRunner.prepare with the main-table settings,
so the window and the points are the ones the answer was computed from.

``--mode pairs`` (the default) keeps images that carry two questions about
different places, and draws those questions one under the other.

``--n_win 4`` runs the multi-window setting (four 20x20 windows) and draws
(a) the image, (b) the where-head relevance map with the numbered windows,
(c) each window's learned density and token centres.

    python scripts/vis_hrbench_sampling.py \
        --model_path $E/checkpoint-1000 \
        --where_head_pt $E/loc/where_lm8_ce20.pt \
        --ldat_pt $X/hinge_noratio_ldat.pt \
        --split hrbench_8k --out /tmp/vis_hrbench
"""
import argparse
import hashlib
import io
import json
import os
import random
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.expanduser(os.environ.get("LMMS_EVAL_DIR", "~/lmms-eval-oldvstar")))

from ldat_eval.infer import LDatRunner  # noqa: E402
from offset_feat_train import inv_cdf  # noqa: E402
from vis_intent_where import fit, font, wrap  # noqa: E402

CYAN, RED, DOT = (0, 190, 220), (210, 40, 40), (255, 230, 0)
WIN_COLORS = [(70, 185, 205), (205, 120, 52), (55, 142, 66), (140, 90, 185), (90, 90, 90), (190, 190, 190)]
SAMPLE_DOT = (45, 90, 195)

# Questions that name different objects in one photograph. Each entry is a
# list of substrings, and each substring must hit a different question.
PAIRS = [
    ("signs_towns", ["Zaragoza", "Aloc"]),
    ("two_signs", ["direction arrows", "rectangular white sign"]),
    ("two_flags", ["color of the flag on the left", "flag on the right"]),
    ("bench_table", ["bench on the left", "striped shirt"]),
    ("dog_bin", ["color of the dog", "green trash bin"]),
    ("bike_sign", ["color of the bike", "text written on the signboard"]),
    ("two_devices", ["black rectangular device", "square white adapter"]),
    ("sleeve_bin", ["jacket's sleeve", "recycle bin"]),
    ("clock_river", ["time shown on the clock", "river flowing"]),
    ("mailbox_entrance", ["color of the mailbox", "above the entrance"]),
    ("door_backpack", ["signboard above the left door", "backpack"]),
    ("flag_totem", ["below the American flag", "totem pole"]),
    ("plate_bird", ["bird positioned", "license plate number of the car"]),
    ("barge_building", ["name of the barge", "small white building"]),
    ("camera_church", ["Surveillance camera", "main entrance of the church"]),
    ("sign_structure", ["message written on the sign", "type of structure"]),
]


def save_png(im, path):
    # PIL opens files "w+b", which the OSS mount rejects; write the bytes once.
    buf = io.BytesIO()
    im.save(buf, "PNG")
    with open(path, "wb") as f:
        f.write(buf.getvalue())


def _dataset(split):
    from datasets import load_dataset
    from lmms_eval.tasks.hrbench.utils import hrbench_doc_to_text, hrbench_doc_to_visual

    def prompt_of(d):
        try:
            return hrbench_doc_to_text(d, None)
        except (TypeError, KeyError):
            return hrbench_doc_to_text(d, {"pre_prompt": "", "post_prompt": ""})

    return load_dataset("DreamMr/HR-Bench", "hrbench_version_split", split=split), prompt_of, hrbench_doc_to_visual


def _row(d, prompt_of, visual):
    return {"cat": d["category"], "question": d["question"].strip(), "answer": d["answer"],
            "options": {k: str(d[k]).strip() for k in "ABCD" if d.get(k) not in (None, "")},
            "prompt": prompt_of(d), "image": visual(d)[0], "index": int(d["index"])}


def load_pairs(split, only=None):
    """One group per shared image, with the questions named by PAIRS.

    Only the chosen rows are decoded. The scan keeps dataset indices, so several
    processes can share the cache without each holding every 8K image.
    """
    ds, prompt_of, visual = _dataset(split)
    groups = {}
    for i in range(len(ds)):
        d = ds[i]
        if str(d["cycle_category"]) != "0":
            continue
        groups.setdefault(hashlib.md5(d["image"].encode()).hexdigest(), []).append(i)
    chosen = []
    for name, spec in PAIRS:
        if only and name not in only:
            continue
        hit = None
        for idxs in groups.values():
            questions = [ds[i]["question"] for i in idxs]
            if all(any(s.lower() in q.lower() for q in questions) for s in spec):
                hit = idxs
                break
        if hit is None:
            print(f"[vis] no image contains {spec}", flush=True)
            continue
        picked = []
        for s in spec:
            for i in hit:
                d = ds[i]
                if s.lower() in d["question"].lower() and all(d["question"] != r["question"] for r in picked):
                    picked.append(_row(d, prompt_of, visual))
                    break
        chosen.append((name, picked))
    return chosen


def merge_pairs(out):
    """Stack the per-image sheets once every rank has written its pngs."""
    blocks, meta = [], []
    for name, _spec in PAIRS:
        path = os.path.join(out, f"pair_{name}.png")
        if not os.path.exists(path):
            print(f"[vis] missing {path}", flush=True)
            continue
        blocks.append(Image.open(path).convert("RGB"))
        jpath = os.path.join(out, f"pair_{name}.json")
        if os.path.exists(jpath):
            with open(jpath) as f:
                meta.extend(json.load(f))
    if not blocks:
        raise SystemExit(f"no pair_*.png in {out}")
    gap = 18
    canvas = Image.new("RGB", (max(s.width for s in blocks), sum(s.height for s in blocks) + gap * (len(blocks) - 1)),
                       (255, 255, 255))
    y = 0
    for s in blocks:
        canvas.paste(s, (0, y))
        y += s.height + gap
    save_png(canvas, os.path.join(out, "hrbench_sampling.png"))
    with open(os.path.join(out, "hrbench_sampling.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[vis] merged {len(blocks)} pairs -> {out}", flush=True)


def load_rows(split, n_try, seed):
    from datasets import load_dataset
    from lmms_eval.tasks.hrbench.utils import hrbench_doc_to_text, hrbench_doc_to_visual

    def prompt_of(d):
        try:
            return hrbench_doc_to_text(d, None)
        except (TypeError, KeyError):
            return hrbench_doc_to_text(d, {"pre_prompt": "", "post_prompt": ""})

    ds = load_dataset("DreamMr/HR-Bench", "hrbench_version_split", split=split)
    # Each question appears once per option rotation; keep the first rotation.
    seen, by_cat = set(), {}
    for i in range(len(ds)):
        d = ds[i]
        key = (d["category"], d["question"].strip())
        if key in seen:
            continue
        seen.add(key)
        by_cat.setdefault(d["category"], []).append(i)
    rng = random.Random(seed)
    rows = []
    for cat in sorted(by_cat):
        idx = by_cat[cat]
        rng.shuffle(idx)
        for i in idx[:n_try]:
            d = ds[i]
            rows.append({"cat": cat, "question": d["question"].strip(), "answer": d["answer"],
                         "prompt": prompt_of(d), "image": hrbench_doc_to_visual(d)[0],
                         "index": int(d["index"])})
    print(f"[vis] {split}: {len(seen)} distinct questions, running {len(rows)}", flush=True)
    print(f"[vis] prompt of the first row:\n{rows[0]['prompt']}", flush=True)
    return rows


def token_grid(w):
    """Cell edges and centres of the mosaic, in window pixels."""
    lx0, ly0, lx1, ly1 = w["lpx"]
    gw, gh = w["gw"], w["gh"]
    dev = w["dx"].device
    ex = inv_cdf(w["dx"], torch.arange(gw + 1, device=dev) / gw).cpu().numpy() * (lx1 - lx0)
    ey = inv_cdf(w["dy"], torch.arange(gh + 1, device=dev) / gh).cpu().numpy() * (ly1 - ly0)
    cx = inv_cdf(w["dx"], (torch.arange(gw, device=dev) + 0.5) / gw).cpu().numpy() * (lx1 - lx0)
    cy = inv_cdf(w["dy"], (torch.arange(gh, device=dev) + 0.5) / gh).cpu().numpy() * (ly1 - ly0)
    return ex, ey, cx, cy


def heat(crop, q):
    arr = q.detach().float().cpu().numpy()
    arr = np.sqrt(arr / max(float(arr.max()), 1e-12))
    W, H = crop.size
    h = np.asarray(Image.fromarray((arr * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC), np.float32) / 255.0
    base = np.asarray(crop.convert("RGB"), np.float32)
    a = (h ** 1.4)[..., None] * 0.72
    return Image.fromarray((base * (1 - a) + np.array(RED, np.float32) * a).astype(np.uint8))


def gamma_correct(image, g):
    """Display only: out = in ** (1 / g), so g > 1 lifts the shadows."""
    if g == 1.0:
        return image
    lut = [min(255, round(255 * (i / 255) ** (1.0 / g))) for i in range(256)]
    return image.point(lut * 3)


def draw_row(r, w, pw=520, ph=520, cap_h=150, gamma=1.0):
    image = gamma_correct(r["image"].convert("RGB"), gamma)
    lx0, ly0, lx1, ly1 = w["lpx"]
    crop = image.crop(w["lpx"])

    full = image.copy()
    full.thumbnail((2400, 2400), Image.BICUBIC)
    s = full.width / image.width
    d = ImageDraw.Draw(full)
    lw = max(4, full.width // 160)
    d.rectangle([lx0 * s, ly0 * s, lx1 * s, ly1 * s], outline=CYAN, width=lw)

    dens = heat(crop, w["q"])
    d = ImageDraw.Draw(dens)
    d.rectangle([0, 0, dens.width - 1, dens.height - 1], outline=CYAN, width=6)

    pts = crop.copy()
    pts.thumbnail((1600, 1600), Image.BICUBIC)
    k = pts.width / crop.width
    ex, ey, cx, cy = token_grid(w)
    d = ImageDraw.Draw(pts)
    for x in ex:
        d.line([x * k, 0, x * k, pts.height], fill=(255, 255, 255), width=1)
    for y in ey:
        d.line([0, y * k, pts.width, y * k], fill=(255, 255, 255), width=1)
    rad = max(3, pts.width // 180)
    for y in cy:
        for x in cx:
            d.ellipse([x * k - rad, y * k - rad, x * k + rad, y * k + rad], fill=DOT, outline=(0, 0, 0))
    d.rectangle([0, 0, pts.width - 1, pts.height - 1], outline=CYAN, width=6)

    panels = [fit(full, pw, ph), fit(dens, pw, ph), fit(pts, pw, ph)]
    sheet = Image.new("RGB", (pw * 3 + 16, ph + cap_h + 8), (255, 255, 255))
    for i, p in enumerate(panels):
        sheet.paste(p, (8 + i * pw, 4))
    dr = ImageDraw.Draw(sheet)
    small, fnt = font(14), font(16)
    titles = [f"{r['cat']}  ·  {image.width}x{image.height}  ·  cyan = window",
              "learned density inside the window",
              f"{w['gw']}x{w['gh']} token centres (yellow) and cell edges"]
    for i, t in enumerate(titles):
        dr.text((8 + i * pw, ph + 8), t, fill=(40, 40, 40), font=small)
    y = ph + 32
    for line in wrap(r["question"], fnt, pw * 3 - 16, 3):
        dr.text((8, y), line, fill=(20, 20, 20), font=fnt)
        y += 22
    ok = r["pred"][:1].upper() == str(r["answer"]).strip()[:1].upper()
    dr.text((8, y + 4), f"answer {r['answer']}  ·  Q-LAS {r['pred'][:8]!r}  ·  {'correct' if ok else 'wrong'}",
            fill=(20, 120, 40) if ok else (160, 40, 40), font=small)
    return sheet, ok


def window_tile(image, w, color, tw, th, dot=4.5, ss=2, dot_color=SAMPLE_DOT, stride=2):
    """One window at tw x th: learned density under its token centres (dot = radius in output px).

    stride > 1 draws every stride-th centre along each axis; the drawn points are real
    sample positions, so their spacing still follows the density.
    """
    crop = image.crop(w["lpx"])
    tile = heat(crop, w["q"])
    tile.thumbnail((tw * ss, th * ss), Image.BICUBIC)
    k = tile.width / crop.width
    _ex, _ey, cx, cy = token_grid(w)
    d = ImageDraw.Draw(tile)
    rad = dot * ss
    off = stride // 2
    for y in cy[off::stride]:
        for x in cx[off::stride]:
            d.ellipse([x * k - rad, y * k - rad, x * k + rad, y * k + rad], fill=dot_color)
    d.rectangle([0, 0, tile.width - 1, tile.height - 1], outline=color, width=4 * ss)
    return fit(tile, tw, th)


def caption(dr, r, x, y, width):
    """Question, options, ground truth and the Q-LAS answer; returns whether Q-LAS is right."""
    opts = r.get("options") or {}

    def opt(v):
        k = str(v).strip()[:1].upper()
        return f"{k}. {opts[k]}" if k in opts else str(v).strip()

    fnt = font(22)
    for line in wrap("Question: " + r["question"], fnt, width, 2):
        dr.text((x, y), line, fill=(20, 20, 20), font=fnt)
        y += 28
    if opts:
        for line in wrap("Options: " + "  ·  ".join(f"{k}. {v}" for k, v in opts.items()), fnt, width, 2):
            dr.text((x, y), line, fill=(60, 60, 60), font=fnt)
            y += 28
    ok = r["pred"][:1].upper() == str(r["answer"]).strip()[:1].upper()
    dr.text((x, y + 4), f"GT: {opt(r['answer'])}", fill=(20, 20, 20), font=fnt)
    gx = x + max(fnt.getlength(f"GT: {opt(r['answer'])}") + 60, width // 3)
    dr.text((gx, y + 4), f"Q-LAS: {opt(r['pred'])}  ({'correct' if ok else 'wrong'})",
            fill=(20, 120, 40) if ok else (170, 40, 40), font=fnt)
    return ok


def draw_multi(r, wins, pw=520, ph=520, cap_h=190, gamma=1.0, dot=4.5, dot_color=SAMPLE_DOT, stride=2):
    """(a) image, (b) relevance map with the numbered windows, (c) each window's density and samples."""
    image = gamma_correct(r["image"].convert("RGB"), gamma)
    full = image.copy()
    full.thumbnail((2400, 2400), Image.BICUBIC)
    s = full.width / image.width

    rel = heat(full, wins[0]["head_p"])
    d = ImageDraw.Draw(rel)
    lw = max(4, full.width // 160)
    lab = font(max(20, full.width // 30))
    for i, w in enumerate(wins):
        c = WIN_COLORS[i % len(WIN_COLORS)]
        x0, y0, x1, y1 = (v * s for v in w["lpx"])
        d.rectangle([x0, y0, x1, y1], outline=c, width=lw)
        t = str(i + 1)
        tw, th = lab.getlength(t), lab.size
        d.rectangle([x0, y0, x0 + tw + 12, y0 + th + 10], fill=c)
        d.text((x0 + 6, y0 + 3), t, fill=(0, 0, 0), font=lab)

    cols = 2 if len(wins) > 1 else 1
    rows = (len(wins) + cols - 1) // cols
    tw, th = (pw - 4 * (cols - 1)) // cols, (ph - 4 * (rows - 1)) // rows
    grid = Image.new("RGB", (pw, ph), (248, 248, 248))
    gd = ImageDraw.Draw(grid)
    small = font(18)
    for i, w in enumerate(wins):
        c = WIN_COLORS[i % len(WIN_COLORS)]
        x, y = (i % cols) * (tw + 4), (i // cols) * (th + 4)
        grid.paste(window_tile(image, w, c, tw, th, dot=dot, dot_color=dot_color, stride=stride), (x, y))
        gd.rectangle([x, y, x + 26, y + 24], fill=c)
        gd.text((x + 7, y + 2), str(i + 1), fill=(0, 0, 0), font=small)

    panels = [fit(full, pw, ph), fit(rel, pw, ph), grid]
    sheet = Image.new("RGB", (pw * 3 + 16, ph + cap_h + 8), (255, 255, 255))
    for i, p in enumerate(panels):
        sheet.paste(p, (8 + i * pw, 4))
    dr = ImageDraw.Draw(sheet)
    side = f"{wins[0]['gw']}x{wins[0]['gh']}"
    titles = [f"(a) {r['cat']}  ·  {image.width}x{image.height}",
              f"(b) relevance map  ·  {len(wins)} windows",
              f"(c) learned density + {side} samples per window" if stride == 1 else
              f"(c) learned density + samples (every {({2: '2nd', 3: '3rd'}).get(stride, f'{stride}th')} of {side})"]
    for i, t in enumerate(titles):
        dr.text((8 + i * pw, ph + 8), t, fill=(40, 40, 40), font=small)
    ok = caption(dr, r, 8, ph + 40, pw * 3 - 16)
    return sheet, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path")
    ap.add_argument("--where_head_pt")
    ap.add_argument("--ldat_pt")
    ap.add_argument("--split", default="hrbench_8k")
    ap.add_argument("--out", default="/tmp/vis_hrbench")
    ap.add_argument("--mode", choices=("pairs", "random"), default="pairs")
    ap.add_argument("--rank", type=int, default=0, help="this process's share of the pairs")
    ap.add_argument("--world", type=int, default=1, help="how many processes split the pairs")
    ap.add_argument("--merge", action="store_true", help="stack pair_*.png and exit")
    ap.add_argument("--n_try", type=int, default=12, help="random mode: questions per category")
    ap.add_argument("--n_show", type=int, default=4, help="random mode: rows per category, correct ones first")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", nargs="*", help="pairs mode: draw only these PAIRS names, e.g. plate_bird")
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="gamma of the drawn image only (> 1 brightens); the model sees the original")
    ap.add_argument("--n_win", type=int, default=1,
                    help="> 1: that many 20x20 windows (the multi-window eval setting), drawn as (a)/(b)/(c)")
    ap.add_argument("--dot", type=float, default=4.5, help="n_win > 1: sample dot radius in output pixels")
    ap.add_argument("--dot_stride", type=int, default=2,
                    help="n_win > 1: draw every k-th sample along each axis (1 = all 20x20)")
    ap.add_argument("--dot_color", default="2d5ac3", help="n_win > 1: sample dot colour, hex RGB")
    ap.add_argument("--keep", nargs="*", help="pairs mode: draw only questions containing one of these substrings")
    args = ap.parse_args()
    if args.merge:
        merge_pairs(args.out)
        return
    if not (args.model_path and args.where_head_pt and args.ldat_pt):
        ap.error("--model_path, --where_head_pt and --ldat_pt are required")
    if args.world < 1 or not 0 <= args.rank < args.world:
        ap.error("--rank must be in 0 .. --world-1")

    groups = load_pairs(args.split, args.only) if args.mode == "pairs" else [(None, load_rows(args.split, args.n_try, args.seed))]
    if args.keep:
        groups = [(name, [r for r in rows if any(s.lower() in r["question"].lower() for s in args.keep)])
                  for name, rows in groups]
        groups = [g for g in groups if g[1]]
    groups = groups[args.rank::args.world]
    print(f"[vis] rank {args.rank}/{args.world}: " + ", ".join(name or "random" for name, _rows in groups), flush=True)
    runner = LDatRunner(args.model_path, args.where_head_pt, args.ldat_pt, args.device,
                        crop_tok=400, lr_tok=640, dat_alpha=0.8, dat_ratio=0.0, dat_k=0.6,
                        src_scale=4.0, where_layer=8, fast_prefix=True, n_win=args.n_win)
    os.makedirs(args.out, exist_ok=True)
    meta, blocks = [], []
    n_ok = n_ran = 0
    for name, rows in groups:
        sheets = []
        for r in rows:
            with torch.inference_mode():
                pr = runner.prepare(r["image"], r["prompt"])
                if pr is None:
                    print(f"[vis] no window | {r['question'][:90]}", flush=True)
                    continue
                r["pred"] = runner.finish(pr, 16).strip()
            if args.n_win > 1:
                sheet, ok = draw_multi(r, pr["wins"], gamma=args.gamma, dot=args.dot, stride=max(1, args.dot_stride),
                                       dot_color=tuple(int(args.dot_color.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)))
            else:
                sheet, ok = draw_row(r, pr["wins"][0], gamma=args.gamma)
            sheets.append(sheet)
            n_ran += 1
            n_ok += int(ok)
            meta.append({"group": name, "category": r["cat"], "index": r["index"], "question": r["question"],
                         "options": r.get("options"), "answer": r["answer"], "pred": r["pred"], "correct": ok,
                         "windows": [list(w["lpx"]) for w in pr["wins"]]})
            print(f"[vis] {n_ran} {name or r['cat']} ok={int(ok)} pred={r['pred'][:8]!r} gt={r['answer']} "
                  f"| {r['question'][:90]}", flush=True)
        if not sheets:
            continue
        if args.mode == "pairs":
            gap = 8
            block = Image.new("RGB", (sheets[0].width, sum(s.height for s in sheets) + gap * (len(sheets) - 1)),
                              (255, 255, 255))
            y = 0
            for s in sheets:
                block.paste(s, (0, y))
                y += s.height + gap
            save_png(block, os.path.join(args.out, f"pair_{name}.png"))
            with open(os.path.join(args.out, f"pair_{name}.json"), "w") as f:
                json.dump([m for m in meta if m["group"] == name], f, indent=2)
            blocks.append(block)
        else:
            blocks.extend(sheets)
    if args.mode == "random":
        kept = []
        for cat in sorted({m["category"] for m in meta}):
            rs = [b for b, m in zip(blocks, meta) if m["category"] == cat]
            ms = [m for m in meta if m["category"] == cat]
            order = sorted(range(len(ms)), key=lambda i: not ms[i]["correct"])
            kept += [rs[i] for i in order[:args.n_show]]
        blocks = kept
    if blocks:
        gap = 18
        canvas = Image.new("RGB", (max(s.width for s in blocks), sum(s.height for s in blocks) + gap * (len(blocks) - 1)),
                           (255, 255, 255))
        y = 0
        for s in blocks:
            canvas.paste(s, (0, y))
            y += s.height + gap
        tag = "" if args.world == 1 else f".r{args.rank}"
        save_png(canvas, os.path.join(args.out, f"hrbench_sampling{tag}.png"))
    tag = "" if args.world == 1 else f".r{args.rank}"
    with open(os.path.join(args.out, f"hrbench_sampling{tag}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[vis] ran {n_ran}  correct {n_ok}  wrote {len(blocks)} blocks -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
