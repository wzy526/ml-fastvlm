#!/usr/bin/env python3
"""Patch-level deformable sampling, zero training, V* with oracle localisation.

Question: can the frozen ViT read a 2nd image whose BLOCKS are undistorted
32x32 pixel patches but whose ARRANGEMENT is non-uniform (dense on the target,
sparse elsewhere)? Pixel-level warps failed (viscot: warp_oracle 74.4/76.8 vs
plain whole image 80.6 vs crop 87.8) because the ViT saw stretched strokes;
here every block is isotropically scaled, only the block layout follows the
density. If this holds, DAT's reference grid + density/offsets can drive the
sampling with the exit that works (ViT re-encode -> tokens in the sequence).

The 2nd image is a MOSAIC: the sampled blocks tiled into a regular gh x gw
image, so the stock processor / ViT / prompt / single-pass path are untouched
(slot positions; real-coordinate RoPE is a later step). Configs:

  crop      the live pipeline's crop (oracle window) — reference, 84.3 on V*
  uni_win   uniform blocks tiling the same window, same grid as the crop's
            resize -> must agree with `crop` (implementation check)
  uni_full  uniform blocks over the whole image (= whole image at 400 tok)
  qw0.5     rectilinear quantile grid over the whole image, 50% / 75% of the
  qw0.75    sampling mass on the oracle WINDOW (crop's content at ~0.75x /
            ~0.9x the crop's pixel density), the rest spread as context
  qb0.25    25% of the mass on the GT BOX itself (the pixel warp_oracle's
            density; over-zooms V*'s tiny boxes, kept for that comparison)
  --block square (default): isotropic blocks; aniso: cell resized = piecewise warp
  dw0.5     DAT inside the head's window (dat_window): reference = crop's grid,
            offsets toward the head map (alpha 0.5), neighbour scale ratio
            <= 1.5 (r<ratio>), k<mass> = adaptive window scale, sq = square
            blocks. dw0 = the crop exactly; dw0k0.7 = crop of the adaptive window.
  --summarize --dataset hrbench4k: also paired per-question comparison vs crop
  mg2       multi-group: one uniform window per head peak (NMS, >= 4 cells apart),
            stacked into one mosaic; mg2 = forced 2 groups, 400 tok split;
            mg2x2 = 400 tok each; mg2a0.3 = adaptive (2nd group only if its
            peak >= 0.3 x main, else identical to crop). Needs --head real.

  for i in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$i setsid nohup python scripts/patch_sample_probe.py \
      --model_path $E/checkpoint-1000 --shard $i --shards 8 \
      --out $E/loc/psp/psp_$i.json --dump_dir $E/loc/psp/img > $E/loc/psp/psp_$i.log 2>&1 < /dev/null & disown; done
  python scripts/patch_sample_probe.py --summarize "$E/loc/psp/psp_*.json"
"""
import argparse
import bisect
import glob
import json
import math
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BLOCK = 32


# ───────────────────────── geometry (no torch) ─────────────────────────

def inv_cdf_edges(marg, n):
    """marg: G non-negative weights = piecewise-constant density on [0, 1].
    -> n + 1 edges in [0, 1] at the quantiles k / n (each of the n cells gets
    1 / n of the mass). Flat marg -> uniform edges."""
    G = len(marg)
    tot = sum(marg) or 1.0
    cdf = [0.0]
    for v in marg:
        cdf.append(cdf[-1] + v / tot)
    cdf[-1] = 1.0
    edges = []
    for k in range(n + 1):
        u = k / n
        c = min(max(bisect.bisect_right(cdf, u) - 1, 0), G - 1)
        span = cdf[c + 1] - cdf[c]
        frac = (u - cdf[c]) / span if span > 1e-12 else 0.0
        edges.append((c + min(max(frac, 0.0), 1.0)) / G)
    edges[0], edges[-1] = 0.0, 1.0
    return edges


def density_marginals(box, mass, G=200):
    """(1 - mass) uniform + mass uniform on `box` (normalised x0,y0,x1,y1);
    separable marginals, which is all a rectilinear grid uses (the 2-D
    map's marginal is exactly this)."""
    def axis(a0, a1):
        m = [(1.0 - mass) / G] * G
        lo = min(max(int(a0 * G), 0), G - 1)
        hi = min(max(int(math.ceil(a1 * G)), lo + 1), G)
        for c in range(lo, hi):
            m[c] += mass / (hi - lo)
        return m
    return axis(box[0], box[2]), axis(box[1], box[3])


def grid_shape(aspect, n):
    gw = max(1, int(round(math.sqrt(n * aspect))))
    gh = max(1, n // gw)
    return gh, gw


def crop_geometry(image, win, crop_tok):
    """modeling._where_crop_image's pixel box and its resized grid (gh, gw)."""
    W, H = image.size
    x0, y0, x1, y1 = win
    px0, py0 = int(x0 * W), int(y0 * H)
    px1 = min(max(int(math.ceil(x1 * W)), px0 + 1), W)
    py1 = min(max(int(math.ceil(y1 * H)), py0 + 1), H)
    cw, ch = px1 - px0, py1 - py0
    sc = math.sqrt(min(crop_tok * 1024, cw * ch) / float(cw * ch))
    nw = max(BLOCK, int(cw * sc) // BLOCK * BLOCK)
    nh = max(BLOCK, int(ch * sc) // BLOCK * BLOCK)
    return px0, py0, px1, py1, nh // BLOCK, nw // BLOCK


def linspace(a, b, n):
    return [a + (b - a) * k / n for k in range(n + 1)]


def mosaic(image, xe, ye, block, box=None):
    """xe (gw + 1), ye (gh + 1) pixel edges -> (gw*32 x gh*32) mosaic.
    block 'aniso': the cell itself resized to 32x32 (= piecewise pixel warp);
    'square': an isotropic square of side sqrt(w*h) at the cell centre, so
    strokes keep their shape and only the zoom differs per block.
    Returns (mosaic, stats): blocks whose centre is in `box`, their mean side."""
    from PIL import Image
    W, H = image.size
    gw, gh = len(xe) - 1, len(ye) - 1
    out = Image.new("RGB", (gw * BLOCK, gh * BLOCK))
    n_in, side_in, side_all = 0, 0.0, 0.0
    for i in range(gh):
        for j in range(gw):
            x0, x1, y0, y1 = xe[j], xe[j + 1], ye[i], ye[i + 1]
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            if block == "square":
                s = math.sqrt(max(x1 - x0, 1e-3) * max(y1 - y0, 1e-3))
                x0, x1 = cx - s / 2, cx + s / 2
                y0, y1 = cy - s / 2, cy + s / 2
                if x0 < 0:
                    x0, x1 = 0.0, min(s, W)
                if y0 < 0:
                    y0, y1 = 0.0, min(s, H)
                if x1 > W:
                    x0, x1 = max(W - s, 0.0), float(W)
                if y1 > H:
                    y0, y1 = max(H - s, 0.0), float(H)
            bx0, by0 = int(math.floor(x0)), int(math.floor(y0))
            bx1 = max(int(math.ceil(min(x1, W))), bx0 + 1)
            by1 = max(int(math.ceil(min(y1, H))), by0 + 1)
            tile = image.crop((bx0, by0, bx1, by1)).resize((BLOCK, BLOCK), Image.BICUBIC)
            out.paste(tile, (j * BLOCK, i * BLOCK))
            side = math.sqrt((bx1 - bx0) * (by1 - by0))
            side_all += side
            if box is not None and box[0] * W <= cx <= box[2] * W and box[1] * H <= cy <= box[3] * H:
                n_in += 1
                side_in += side
    stats = {"blocks": gw * gh, "in_box": n_in,
             "side_in_box": side_in / n_in if n_in else None,
             "side_mean": side_all / (gw * gh)}
    return out, stats


def gt_box(s):
    """Normalised union GT box, or None when the dataset has no boxes (HR-Bench)."""
    W, H = s["image"].size
    bb = s.get("bboxes") or []
    if not bb:
        return None
    x0, y0 = min(b[0] for b in bb), min(b[1] for b in bb)
    x1, y1 = max(b[0] + b[2] for b in bb), max(b[1] + b[3] for b in bb)
    return (x0 / W, y0 / H, x1 / W, y1 / H)


def map_marginals(p2d, mass, G=200):
    """p2d: (gh, gw) non-negative map (the head's cell probabilities), used as
    a sampling density: mass * p + (1 - mass) * uniform. Separable marginals
    on a G-bin axis (cells are resampled to G bins so all densities share the
    same resolution as density_marginals)."""
    gh, gw = len(p2d), len(p2d[0])
    tot = sum(sum(r) for r in p2d) or 1.0
    col = [sum(p2d[i][j] for i in range(gh)) / tot for j in range(gw)]
    row = [sum(p2d[i]) / tot for i in range(gh)]

    def axis(marg):
        n = len(marg)
        out = []
        for b in range(G):
            c = min(int(b * n / G), n - 1)
            out.append(mass * marg[c] * n / G + (1.0 - mass) / G)
        return out
    return axis(col), axis(row)


def head_map(logits, temp=1.0):
    """(gh, gw) list of logits -> softmax(logits / temp) as nested lists."""
    flat = [v / temp for row in logits for v in row]
    mx = max(flat)
    ex = [math.exp(v - mx) for v in flat]
    z = sum(ex)
    gw = len(logits[0])
    return [[ex[i * gw + j] / z for j in range(gw)] for i in range(len(logits))]


def multi_group(image, win, crop_tok, cfg, box, head, orig_crop):
    """mg<K>[a<tau>][x<B>]: one uniform group (window of the live window's size,
    centred on a head peak) per NMS peak of the head map, stacked vertically
    into one mosaic. K = max groups; with a<tau> the count is adaptive (peaks
    with prob >= tau * main), else forced to K; the token budget is crop_tok
    (split over the groups) or B * crop_tok with x<B>. One group -> the live
    crop, so adaptive K = 1 is identical to `crop`."""
    from PIL import Image
    from where_peaks_probe import nms_peaks, softmax2d
    spec = cfg[2:]
    budget_mult = 1.0
    if "x" in spec:
        spec, b = spec.split("x")
        budget_mult = float(b)
    tau = 0.0
    if "a" in spec:
        spec, t = spec.split("a")
        tau = float(t)
    K = int(spec)
    p = softmax2d(head["logits"])
    gh_m, gw_m = len(p), len(p[0])
    peaks = nms_peaks(p, 4, tau, K)
    if len(peaks) <= 1:
        return orig_crop(image, win, crop_tok), {"groups": 1, "peaks": peaks, "in_box": 0, "blocks": 0,
                                                 "side_in_box": None, "side_mean": 0.0}
    W, H = image.size
    ww, wh = win[2] - win[0], win[3] - win[1]
    tok_g = int(round(crop_tok * budget_mult / len(peaks)))
    gh, gw = grid_shape((ww * W) / max(wh * H, 1e-6), tok_g)
    parts, stats = [], {"groups": len(peaks), "peaks": peaks, "in_box": 0, "blocks": 0, "side_in_box": None}
    side_in, n_in, side_all = 0.0, 0, 0.0
    for _, cy, cx in peaks:
        cxn, cyn = (cx + 0.5) / gw_m, (cy + 0.5) / gh_m
        x0 = min(max(cxn - ww / 2, 0.0), 1.0 - ww)
        y0 = min(max(cyn - wh / 2, 0.0), 1.0 - wh)
        px0, py0, px1, py1 = x0 * W, y0 * H, (x0 + ww) * W, (y0 + wh) * H
        img, st = mosaic(image, linspace(px0, px1, gw), linspace(py0, py1, gh), "aniso", box)
        parts.append(img)
        stats["blocks"] += st["blocks"]
        n_in += st["in_box"]
        side_in += (st["side_in_box"] or 0.0) * st["in_box"]
        side_all += st["side_mean"]
    stats["in_box"] = n_in
    stats["side_in_box"] = side_in / n_in if n_in else None
    stats["side_mean"] = side_all / len(parts)
    out = Image.new("RGB", (gw * BLOCK, gh * BLOCK * len(parts)))
    for k, img in enumerate(parts):
        out.paste(img, (0, k * gh * BLOCK))
    return out, stats


DW_RE = re.compile(r"^dw([\d.]+)(?:r([\d.]+))?(?:k([\d.]+))?(sq)?$")


def _overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def window_mass(p2d, win):
    """Head probability inside the normalised window (cells weighted by overlap)."""
    gh, gw = len(p2d), len(p2d[0])
    tot = 0.0
    for i in range(gh):
        oy = _overlap(i / gh, (i + 1) / gh, win[1], win[3]) * gh
        if oy <= 0:
            continue
        for j in range(gw):
            ox = _overlap(j / gw, (j + 1) / gw, win[0], win[2]) * gw
            if ox > 0:
                tot += p2d[i][j] * ox * oy
    return tot


def adaptive_window(head, win, k, smin=0.6, smax=1.8):
    """DAT's scale term s: the live window (same aspect) rescaled by the
    smallest sigma in [smin, smax] whose window around the head's top box
    holds >= k of the head's probability. Returns (window, sigma)."""
    p = head_map(head["logits"])
    hb = head["box"]
    cx, cy = (hb[0] + hb[2]) / 2, (hb[1] + hb[3]) / 2
    hw, hh = (win[2] - win[0]) / 2, (win[3] - win[1]) / 2
    best = None
    steps = int(round((smax - smin) / 0.1))
    for t in range(steps + 1):
        sg = smin + 0.1 * t
        w2, h2 = min(sg * hw, 0.5), min(sg * hh, 0.5)
        x0 = min(max(cx - w2, 0.0), 1.0 - 2 * w2)
        y0 = min(max(cy - h2, 0.0), 1.0 - 2 * h2)
        best = ((x0, y0, x0 + 2 * w2, y0 + 2 * h2), sg)
        if window_mass(p, best[0]) >= k:
            break
    return best


def window_marginals(p2d, win, alpha, ratio, G=64):
    """Sampling density inside the window, per axis: (1 - alpha) uniform +
    alpha * the head map's marginal over the window (linearly interpolated
    between cell centres, so the grid spacing changes smoothly), then floored
    so max / min <= ratio: neighbouring tokens differ in scale by little and
    the warp stays mild. alpha = 0 -> uniform = the crop's grid."""
    gh, gw = len(p2d), len(p2d[0])
    col = [sum(p2d[i][j] * _overlap(i / gh, (i + 1) / gh, win[1], win[3]) * gh for i in range(gh))
           for j in range(gw)]
    row = [sum(p2d[i][j] * _overlap(j / gw, (j + 1) / gw, win[0], win[2]) * gw for j in range(gw))
           for i in range(gh)]

    def axis(vals, a0, a1):
        n = len(vals)
        samp = []
        for b in range(G):
            u = a0 + (b + 0.5) / G * (a1 - a0)
            f = min(max(u * n - 0.5, 0.0), n - 1.0)
            lo = int(f)
            hi = min(lo + 1, n - 1)
            samp.append(vals[lo] * (1 - (f - lo)) + vals[hi] * (f - lo))
        tot = sum(samp)
        d = [(1.0 - alpha) / G + (alpha * v / tot if tot > 0 else alpha / G) for v in samp]
        if ratio and ratio > 1:
            fl = max(d) / ratio
            d = [max(v, fl) for v in d]
        return d
    return axis(col, win[0], win[2]), axis(row, win[1], win[3])


def dat_window(image, win, crop_tok, cfg, box, head, orig_crop):
    """dw<alpha>[r<ratio>][k<mass>][sq]: DAT inside the localisation window.
    x = c + s * reference + offset: c = the head's window centre, s = the live
    window size (or, with k<mass>, adaptive_window), reference = the crop's
    uniform gh x gw grid, offset = the inverse-CDF shift of the grid lines
    toward the head's mass (alpha), bounded by ratio (default 1.5). Blocks are
    the cells themselves (contiguous piecewise-linear warp, no seams); `sq`
    uses isotropic squares instead. alpha = 0 -> exactly the crop of the
    (possibly adaptive) window; nothing is spent outside the window."""
    m = DW_RE.match(cfg)
    if not m:
        raise ValueError(cfg)
    alpha = float(m.group(1))
    ratio = float(m.group(2)) if m.group(2) else 1.5
    k = float(m.group(3)) if m.group(3) else None
    blk = "square" if m.group(4) else "aniso"
    stats = {}
    if k is not None:
        win, sg = adaptive_window(head, win, k)
        stats["sigma"] = sg
    if alpha == 0 and blk == "aniso":
        return orig_crop(image, win, crop_tok), stats
    px0, py0, px1, py1, gh, gw = crop_geometry(image, win, crop_tok)
    nwin = (px0 / image.width, py0 / image.height, px1 / image.width, py1 / image.height)
    dx, dy = window_marginals(head_map(head["logits"]), nwin, alpha, ratio)
    xe = [px0 + e * (px1 - px0) for e in inv_cdf_edges(dx, gw)]
    ye = [py0 + e * (py1 - py0) for e in inv_cdf_edges(dy, gh)]
    img, st = mosaic(image, xe, ye, blk, box)
    widths = [xe[j + 1] - xe[j] for j in range(gw)]
    st["scale_ratio"] = max(widths) / max(min(widths), 1e-6)
    st.update(stats)
    return img, st


def second_image(image, win, crop_tok, cfg, box, block, orig_crop, head=None):
    """The 2nd image for config `cfg`. `box` is the GT box (stats, qb*);
    `head` = {"box": the head's top box, "map": (gh, gw) cell probabilities}
    for the hb* / hm* configs. Returns (PIL, stats)."""
    W, H = image.size
    if cfg == "crop":
        return orig_crop(image, win, crop_tok), {}
    if cfg.startswith("mg"):
        return multi_group(image, win, crop_tok, cfg, box, head, orig_crop)
    if cfg.startswith("dw"):
        return dat_window(image, win, crop_tok, cfg, box, head, orig_crop)
    if cfg == "uni_win":
        px0, py0, px1, py1, gh, gw = crop_geometry(image, win, crop_tok)
        return mosaic(image, linspace(px0, px1, gw), linspace(py0, py1, gh), "aniso", box)
    gh, gw = grid_shape(W / H, crop_tok)
    if cfg == "uni_full":
        mx, my = density_marginals(win, 0.0)
    elif cfg.startswith("qw"):             # mass on the WINDOW (what crop shows), rest context
        mx, my = density_marginals(win, float(cfg[2:]))
    elif cfg.startswith("qb"):             # mass on the GT BOX itself (oracle density)
        if box is None:
            raise ValueError(f"{cfg} needs GT boxes (V* only)")
        mx, my = density_marginals(box, float(cfg[2:]))
    elif cfg.startswith("hb"):             # mass on the HEAD's top box (4 cells), rest context
        hb = [min(max(v, 0.0), 1.0) for v in head["box"]]
        mx, my = density_marginals(hb, float(cfg[2:]))
    elif cfg.startswith("hm"):             # the head's probability map as the density; hm<m>t<T>
        spec = cfg[2:].split("t")           # sharpens it: softmax(logits / T), T < 1 (zero training)
        mass, temp = float(spec[0]), float(spec[1]) if len(spec) > 1 else 1.0
        mx, my = map_marginals(head_map(head["logits"], temp), mass)
    else:
        raise ValueError(cfg)
    xe = [e * W for e in inv_cdf_edges(mx, gw)]
    ye = [e * H for e in inv_cdf_edges(my, gh)]
    return mosaic(image, xe, ye, block, box)


# ───────────────────────── summary ─────────────────────────

def prompt_key(prompt):
    """HR-Bench repeats each question with its options permuted: the question
    line + the SET of option texts identifies it."""
    lines = [l.strip() for l in str(prompt).split("\n") if l.strip()]
    opts = frozenset(re.sub(r"^[A-Z]\.\s*", "", l) for l in lines[1:] if re.match(r"^[A-Z]\.", l))
    return (lines[0] if lines else "", opts)


def question_groups(rows, dataset=None, parquet=None):
    """{row index i: group id}. Rows carry `prompt` (new runs); older jsons get
    it from the HR-Bench parquet (--dataset). Same key only merges indices
    within 8 of the group's first one (the repeats sit next to each other;
    'How many people are there?' on another image is another question)."""
    prompts = {r["i"]: r["prompt"] for r in rows if r.get("prompt")}
    if not prompts and dataset and dataset.startswith("hrbench"):
        import pandas as pd
        from probe_whd_qwen35 import _local_hrbench_parquet
        split = {"hrbench4k": "hrbench_4k", "hrbench8k": "hrbench_8k"}[dataset]
        d = pd.read_parquet(parquet or _local_hrbench_parquet(split), columns=["question"] + list("ABCD"))
        for i in range(len(d)):
            r = d.iloc[i]
            opts = "".join(f"{c}. {r[c]}\n" for c in "ABCD" if isinstance(r[c], str))
            prompts[i] = f"{str(r['question']).strip()}\n{opts}"
    if not prompts:
        return {r["i"]: r["i"] for r in rows}
    groups, first = {}, {}
    for i in sorted({r["i"] for r in rows}):
        k = prompt_key(prompts.get(i, i))
        if k in first and i - first[k][0] <= 8:
            groups[i] = first[k][1]
        else:
            first[k] = (i, i)
            groups[i] = i
    return groups


def sign_test(w, l):
    n = w + l
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, j) for j in range(min(w, l) + 1)) / 2.0 ** n
    return min(1.0, 2 * tail)


def paired_vs_crop(runs, groups):
    """Per question (repeats averaged): how many questions a config gets more /
    less right than crop, and the two-sided sign-test p."""
    if "crop" not in runs:
        return
    def per_q(rows):
        acc = {}
        for r in rows:
            acc.setdefault(groups.get(r["i"], r["i"]), []).append(r["ok"])
        return {g: sum(v) / len(v) for g, v in acc.items()}
    ref = per_q(runs["crop"])
    print(f"paired vs crop over {len(ref)} questions (repeats averaged): better / worse / sign-test p")
    for cfg, rows in runs.items():
        if cfg == "crop":
            continue
        q = per_q(rows)
        common = [g for g in q if g in ref]
        w = sum(q[g] > ref[g] for g in common)
        l = sum(q[g] < ref[g] for g in common)
        net = sum(q[g] - ref[g] for g in common)
        print(f"  {cfg:12s} {w:4d} / {l:<4d} p={sign_test(w, l):.3f}   net {net:+.2f} questions")


def summarize(pattern, dataset=None, parquet=None):
    runs = {}
    for f in sorted(glob.glob(os.path.expanduser(pattern))):
        for cfg, rows in json.load(open(f)).items():
            runs.setdefault(cfg, []).extend(rows)
    if not runs:
        raise SystemExit(f"no json matched {pattern}")
    ref = {r["i"]: r for r in runs.get("crop", [])}
    print(f"{'config':10s} {'acc':>12s} {'agree w/ crop':>14s} {'tok':>6s} {'blocks in box':>14s} {'side in/all px':>16s}")
    for cfg, rows in runs.items():
        n = len(rows)
        ok = sum(r["ok"] for r in rows)
        agree = sum(r["pred"] == ref[r["i"]]["pred"] for r in rows if r["i"] in ref)
        tok = sum(r.get("n_tokens") or 0 for r in rows) / max(n, 1)
        st = [r["stats"] for r in rows if r.get("stats")]
        inb = sum(s["in_box"] for s in st) / len(st) if st else float("nan")
        sin = [s["side_in_box"] for s in st if s.get("side_in_box")]
        sin = sum(sin) / len(sin) if sin else float("nan")
        sall = sum(s["side_mean"] for s in st) / len(st) if st else float("nan")
        grp = [s["groups"] for s in st if "groups" in s]
        grp = f"  groups>1: {sum(g > 1 for g in grp)}/{len(grp)}" if grp else ""
        sg = [s["sigma"] for s in st if "sigma" in s]
        if sg:
            grp += f"  sigma mean {sum(sg) / len(sg):.2f} (<1: {sum(v < 0.95 for v in sg)}, >1: {sum(v > 1.05 for v in sg)})"
        sr = [s["scale_ratio"] for s in st if "scale_ratio" in s]
        if sr:
            grp += f"  col scale ratio {sum(sr) / len(sr):.2f}"
        print(f"{cfg:10s} {ok:4d}/{n:<4d}{100 * ok / max(n, 1):5.1f} {agree:7d}/{n:<6d} {tok:6.0f} "
              f"{inb:14.1f} {sin:7.1f}/{sall:<8.1f}{grp}")
    cats = sorted({r["cat"] for rows in runs.values() for r in rows})
    print("per category: " + "  ".join(cats))
    for cfg, rows in runs.items():
        parts = []
        for c in cats:
            rs = [r for r in rows if r["cat"] == c]
            parts.append(f"{sum(r['ok'] for r in rs)}/{len(rs)}")
        print(f"  {cfg:10s} " + "  ".join(parts))
    any_rows = next(iter(runs.values()))
    paired_vs_crop(runs, question_groups(any_rows, dataset, parquet))


# ───────────────────────── main ─────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summarize", default=None, help="glob of --out jsons; no GPU")
    ap.add_argument("--model_path")
    ap.add_argument("--dataset", choices=["vstar", "hrbench4k", "hrbench8k"], default="vstar",
                    help="HR-Bench has no GT boxes: --head real with crop / hm* / hb* / uni_* only")
    ap.add_argument("--vstar_root", default="~/data/vstar_bench")
    ap.add_argument("--hrbench_parquet", default=None, help="see probe_whd_qwen35.py")
    ap.add_argument("--max_samples", type=int, default=300)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--configs", default="crop,uni_win,uni_full,qw0.5,qw0.75,qb0.25",
                    help="also hb<m> (head's top box) / hm<m>[t<T>] (head's softmax(logits/T) map) with --head real")
    ap.add_argument("--head", choices=["oracle", "real"], default="oracle",
                    help="oracle: GT-centre one-hot head (window + qb* are oracle); real: the checkpoint's head")
    ap.add_argument("--where_head_pt", default=None, help="--head real: load this state_dict instead of where_head.pt")
    ap.add_argument("--block", choices=["square", "aniso"], default="square",
                    help="q*/uni_full blocks: isotropic squares (default) or the cell resized (= piecewise warp)")
    ap.add_argument("--crop_tok", type=int, default=400)
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--min_cells", type=int, default=20)
    ap.add_argument("--ground_pad", type=float, default=2.0)
    ap.add_argument("--lr_tok", type=int, default=640)
    ap.add_argument("--head_prefix", default="train", choices=["legacy", "train"])
    ap.add_argument("--single_pass", action="store_true")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--dump_dir", default=None, help="save the 2nd images of the first --dump_n samples")
    ap.add_argument("--dump_n", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.summarize:
        summarize(args.summarize, args.dataset if args.dataset != "vstar" else None, args.hrbench_parquet)
        return
    if not args.model_path:
        raise SystemExit("--model_path required")

    import torch
    from transformers import AutoProcessor
    from probe_whd_qwen35 import extract_letter, load_samples
    from where_crop_eval import OracleHead
    import llava.model.language_model.modeling_qwen3_5_dat as dat_mod

    model = dat_mod.Qwen3_5DATForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation=args.attn).eval()
    oracle = None
    if args.head == "oracle":
        oracle = OracleHead()
        model.where_head = oracle
    elif not model._where_head_loaded:
        raise SystemExit(f"--head real: no where_head.pt in {args.model_path}")
    if args.where_head_pt:
        sd = torch.load(os.path.expanduser(args.where_head_pt), map_location="cpu")
        sd = {k[len("where_head."):] if k.startswith("where_head.") else k: v for k, v in sd.items()}
        model.where_head.load_state_dict(sd)
        print(f"[psp] head weights <- {args.where_head_pt}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_path, min_pixels=1024, max_pixels=args.lr_tok * 1024)
    hr_processor = AutoProcessor.from_pretrained(args.model_path, min_pixels=1024, max_pixels=100_000_000)

    if args.head == "oracle" and args.dataset != "vstar":
        raise SystemExit("--head oracle needs GT boxes; only V* has them")
    probe_args = argparse.Namespace(dataset=args.dataset, vstar_root=args.vstar_root,
                                    hrbench_parquet=args.hrbench_parquet,
                                    image_folder=None, max_samples=args.max_samples)
    samples = load_samples(probe_args)
    mine = [(i, s) for i, s in enumerate(samples) if i % args.shards == args.shard
            and (args.head == "real" or s.get("bboxes"))]
    configs = [c for c in args.configs.split(",") if c]
    print(f"[psp] {args.dataset} shard {args.shard}/{args.shards}: {len(mine)} samples x {configs}; head={args.head} block={args.block} "
          f"crop_tok={args.crop_tok} single_pass={args.single_pass}", flush=True)

    state = {"cfg": None, "box": None, "i": None, "stats": {}, "dumped": 0, "head": None}
    orig_crop = dat_mod._where_crop_image
    orig_top_box = dat_mod._where_top_box

    def top_box_capture(logits, cells):
        box = orig_top_box(logits, cells)
        state["head"] = {"box": tuple(float(v) for v in box), "logits": logits.float().tolist()}
        return box

    def patched(image, win, crop_tok):
        img, st = second_image(image, win, crop_tok, state["cfg"], state["box"], args.block, orig_crop,
                               head=state["head"])
        state["stats"] = st
        if args.dump_dir and state["dumped"] < args.dump_n * len(configs):
            os.makedirs(os.path.expanduser(args.dump_dir), exist_ok=True)
            img.save(os.path.join(os.path.expanduser(args.dump_dir), f"{state['i']}_{state['cfg']}.png"))
            state["dumped"] += 1
        return img

    dat_mod._where_crop_image = patched
    dat_mod._where_top_box = top_box_capture

    out = {}
    for cfg in configs:
        rows, n_ok, t0 = [], 0, time.time()
        for gi, s in mine:
            if oracle is not None:
                oracle.set_sample(s)
            state.update(cfg=cfg, box=gt_box(s), i=gi, stats={})
            where_q = s["prompt"] if args.single_pass else s["prompt"].split("\n")[0]
            rec = model.where_crop_generate(
                s["image"], s["prompt"], processor, hr_processor, where_question=where_q,
                layer=args.layer, crop_tok=args.crop_tok, min_cells=args.min_cells,
                ground_pad=args.ground_pad, head_prefix=args.head_prefix, single_pass=args.single_pass)
            pred = extract_letter(rec["text"])
            ok = pred == s["gt"]
            n_ok += ok
            rows.append({"i": gi, "pred": pred, "gt": s["gt"], "ok": bool(ok), "cat": s["category"],
                         "prompt": s["prompt"],
                         "window": list(rec["window"]), "n_tokens": rec.get("n_tokens"),
                         "stats": state["stats"]})
            st = state["stats"]
            print(f"[psp] {cfg} {gi} pred={pred} gt={s['gt']} ok={int(ok)} tok={rec.get('n_tokens')}"
                  + (f" in_box={st['in_box']}/{st['blocks']} side_in={st['side_in_box'] and round(st['side_in_box'], 1)}"
                     if st else ""), flush=True)
        print(f"[psp] {cfg}: {n_ok}/{len(rows)} ({100 * n_ok / max(len(rows), 1):.1f}%) "
              f"{(time.time() - t0) / max(len(rows), 1):.2f} s/q", flush=True)
        out[cfg] = rows
    dat_mod._where_crop_image = orig_crop
    dat_mod._where_top_box = orig_top_box
    if args.out:
        os.makedirs(os.path.dirname(os.path.expanduser(args.out)) or ".", exist_ok=True)
        json.dump(out, open(os.path.expanduser(args.out), "w"))
        print(f"[psp] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
