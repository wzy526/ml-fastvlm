"""Shared where-head and pixel-DAT geometry; no model-family imports.

32px defaults preserve existing Qwen3.5 checkpoints. Callers for other
backbones pass processor-derived patch / merged-token sizes explicitly.
"""
import bisect
import math

import torch
from torch import nn

def _where_top_box(logits, cells):
    """Top cell -> square of `cells` cells, as fractions of the LM grid.
    Same square scripts/where_head.py top_boxes writes into the box json."""
    H, W = logits.shape
    cy, cx = divmod(int(logits.reshape(-1).argmax()), W)
    half = cells / 2.0
    return ((cx + 0.5 - half) / W, (cy + 0.5 - half) / H,
            (cx + 0.5 + half) / W, (cy + 0.5 + half) / H)


def _where_hd_size(image, lr_thw, hr_scale, hd_cap, *, patch_size=16, merge_size=2):
    """probe_whd_qwen35.hd_target_size, so the min-cells floor matches the probe."""
    factor = patch_size * merge_size
    lr_px = int(lr_thw[1]) * patch_size * int(lr_thw[2]) * patch_size
    total = min(lr_px * hr_scale * hr_scale, image.width * image.height, hd_cap)
    aspect = image.width / image.height
    hd_h = int(math.sqrt(total / aspect))
    hd_w = int(hd_h * aspect)
    return max(factor, (hd_w // factor) * factor), max(factor, (hd_h // factor) * factor)


def _where_pad_window(box, pad, min_cells, hd_w, hd_h, *, factor=32):
    """probe_whd_qwen35.pad_window."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ww = min(max((x1 - x0) * pad, min_cells * factor / hd_w), 1.0)
    wh = min(max((y1 - y0) * pad, min_cells * factor / hd_h), 1.0)
    cx = min(max(cx, ww / 2), 1 - ww / 2)
    cy = min(max(cy, wh / 2), 1 - wh / 2)
    return cx - ww / 2, cy - wh / 2, cx + ww / 2, cy + wh / 2


def _where_crop_image(image, win, crop_tok, *, factor=32):
    """probe_whd_qwen35.make_crop_image."""
    from PIL import Image
    W, H = image.size
    x0, y0, x1, y1 = win
    px0, py0 = int(x0 * W), int(y0 * H)
    px1 = max(int(math.ceil(x1 * W)), px0 + 1)
    py1 = max(int(math.ceil(y1 * H)), py0 + 1)
    crop = image.crop((px0, py0, min(px1, W), min(py1, H)))
    cw, ch = crop.size
    sc = math.sqrt(min(crop_tok * factor * factor, cw * ch) / float(cw * ch))
    nw = max(factor, int(cw * sc) // factor * factor)
    nh = max(factor, int(ch * sc) // factor * factor)
    return crop.resize((nw, nh), Image.BICUBIC)


# ---- DAT sampling inside the localisation window (sample_mode='dat') ----
# Ported from scripts/patch_sample_probe.py (dat_window, config dw<a>r<r>k<k>);
# keep the two identical so probe numbers carry over.
# x = c + s * reference + offset: c = the head's window centre, s = the window
# scale (adaptive: smallest that holds `k` of the head's mass), reference = the
# crop's uniform grid, offset = inverse-CDF shift of the grid lines toward the
# head map (strength alpha, neighbour scale ratio <= ratio). alpha = 0 and no
# adaptive scale -> exactly _where_crop_image.

def _wdat_overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def _wdat_softmax(logits):
    flat = [v for row in logits for v in row]
    mx = max(flat)
    ex = [math.exp(v - mx) for v in flat]
    z = sum(ex)
    gw = len(logits[0])
    return [[ex[i * gw + j] / z for j in range(gw)] for i in range(len(logits))]


def _wdat_window_mass(p, win):
    gh, gw = len(p), len(p[0])
    tot = 0.0
    for i in range(gh):
        oy = _wdat_overlap(i / gh, (i + 1) / gh, win[1], win[3]) * gh
        if oy <= 0:
            continue
        for j in range(gw):
            ox = _wdat_overlap(j / gw, (j + 1) / gw, win[0], win[2]) * gw
            if ox > 0:
                tot += p[i][j] * ox * oy
    return tot


def _wdat_adaptive_window(p, box, win, k, smin=0.6, smax=1.8):
    """The window (same aspect as `win`) around the head's top box, rescaled
    by the smallest sigma in [smin, smax] (0.1 steps) holding >= k of p."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    hw, hh = (win[2] - win[0]) / 2, (win[3] - win[1]) / 2
    best = None
    for t in range(int(round((smax - smin) / 0.1)) + 1):
        sg = smin + 0.1 * t
        w2, h2 = min(sg * hw, 0.5), min(sg * hh, 0.5)
        x0 = min(max(cx - w2, 0.0), 1.0 - 2 * w2)
        y0 = min(max(cy - h2, 0.0), 1.0 - 2 * h2)
        best = ((x0, y0, x0 + 2 * w2, y0 + 2 * h2), sg)
        if _wdat_window_mass(p, best[0]) >= k:
            break
    return best


def _wdat_marginals(p, win, alpha, ratio, G=64):
    gh, gw = len(p), len(p[0])
    col = [sum(p[i][j] * _wdat_overlap(i / gh, (i + 1) / gh, win[1], win[3]) * gh for i in range(gh))
           for j in range(gw)]
    row = [sum(p[i][j] * _wdat_overlap(j / gw, (j + 1) / gw, win[0], win[2]) * gw for j in range(gw))
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


def _wdat_inv_cdf_edges(marg, n):
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


def _where_dat_image(image, win, crop_tok, logits, box, alpha=0.5, ratio=1.5, k=0.6, *, factor=32):
    """sample_mode='dat': the 2nd image sampled by a DAT grid inside the
    head's window. Same pixel box / gh x gw grid as _where_crop_image on the
    (adaptive) window, so the token count matches the crop; each grid cell
    is resized to 32x32 and the cells tile contiguously (piecewise-linear
    warp, no seams). Returns (PIL image, info dict)."""
    from PIL import Image
    p = _wdat_softmax(logits.float().tolist())
    info = {"window": tuple(win)}
    if k is not None:
        win, sg = _wdat_adaptive_window(p, box, win, k)
        info.update(window=win, sigma=sg)
    if not alpha:
        return _where_crop_image(image, win, crop_tok, factor=factor), info
    W, H = image.size
    px0, py0 = int(win[0] * W), int(win[1] * H)
    px1 = min(max(int(math.ceil(win[2] * W)), px0 + 1), W)
    py1 = min(max(int(math.ceil(win[3] * H)), py0 + 1), H)
    cw, ch = px1 - px0, py1 - py0
    sc = math.sqrt(min(crop_tok * factor * factor, cw * ch) / float(cw * ch))
    gw = max(factor, int(cw * sc) // factor * factor) // factor
    gh = max(factor, int(ch * sc) // factor * factor) // factor
    dx, dy = _wdat_marginals(p, (px0 / W, py0 / H, px1 / W, py1 / H), alpha, ratio)
    xe = [px0 + e * cw for e in _wdat_inv_cdf_edges(dx, gw)]
    ye = [py0 + e * ch for e in _wdat_inv_cdf_edges(dy, gh)]
    out = Image.new("RGB", (gw * factor, gh * factor))
    for i in range(gh):
        for j in range(gw):
            bx0, by0 = int(math.floor(xe[j])), int(math.floor(ye[i]))
            bx1 = max(int(math.ceil(min(xe[j + 1], W))), bx0 + 1)
            by1 = max(int(math.ceil(min(ye[i + 1], H))), by0 + 1)
            out.paste(image.crop((bx0, by0, bx1, by1)).resize((factor, factor), Image.BICUBIC), (j * factor, i * factor))
    widths = [xe[j + 1] - xe[j] for j in range(gw)]
    info["scale_ratio"] = max(widths) / max(min(widths), 1e-6)
    return out, info


class _WhereXBlock(nn.Module):
    """Same block as scripts/where_head.py XBlock. Parameter names must match
    where_lm8_mix.pt so the checkpoint sidecar loads without a key rename."""

    def __init__(self, d, heads):
        super().__init__()
        self.t_norm = nn.LayerNorm(d)
        self.t_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.c_norm = nn.LayerNorm(d)
        self.c_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.conv = nn.Conv2d(d, d, 3, padding=1, groups=d)
        self.f_norm = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, cells, txt, H, W):
        tq = self.t_norm(txt)
        txt = txt + self.t_attn(tq, cells, cells, need_weights=False)[0]
        cq = self.c_norm(cells)
        cells = cells + self.c_attn(cq, txt, txt, need_weights=False)[0]
        g = cells.transpose(1, 2).reshape(1, -1, H, W)
        cells = cells + self.conv(g).flatten(2).transpose(1, 2)
        cells = cells + self.ffn(self.f_norm(cells))
        return cells, txt


class _WhereXAttnHead(nn.Module):
    """Same module as scripts/where_head.py XAttnHead (d=384, 2 layers, 6 heads)."""

    def __init__(self, c_img, c_txt, d=384, layers=2, heads=6, max_hw=512):
        super().__init__()
        self.img_in = nn.Sequential(nn.LayerNorm(c_img), nn.Linear(c_img, d))
        self.txt_in = nn.Sequential(nn.LayerNorm(c_txt), nn.Linear(c_txt, d))
        self.row = nn.Embedding(max_hw, d)
        self.col = nn.Embedding(max_hw, d)
        nn.init.normal_(self.row.weight, std=0.02)
        nn.init.normal_(self.col.weight, std=0.02)
        self.blocks = nn.ModuleList(_WhereXBlock(d, heads) for _ in range(layers))
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, feat, txt):
        H, W, _ = feat.shape
        wd = self.img_in[1].weight.dtype            # fp32 when kept there, else the model dtype
        x = self.img_in(feat.to(wd))
        n_pos = self.row.num_embeddings
        ri = torch.arange(H, device=x.device).clamp_max(n_pos - 1)
        ci = torch.arange(W, device=x.device).clamp_max(n_pos - 1)
        pos = self.row(ri)[:, None] + self.col(ci)[None]
        cells = (x + pos).reshape(1, H * W, -1)
        t = self.txt_in(txt.to(wd))[None]
        for blk in self.blocks:
            cells, t = blk(cells, t, H, W)
        return self.out(cells)[0, :, 0].reshape(H, W)


# Text placed between the question and the crop, in training and in
# where_crop_generate. Keep the two identical.
WHERE_HINT = "Zoomed-in view of the relevant region:"


def _where_head_loss(logits, mask, focal_w=0.5):
    """scripts/where_head.py head_loss: soft CE (uniform over the GT cells) +
    focal BCE. logits / mask (H, W)."""
    flat, m = logits.reshape(-1).float(), mask.reshape(-1).to(logits.device).float()
    ce = -(torch.log_softmax(flat, -1) * (m / m.sum().clamp(min=1))).sum()
    p = torch.sigmoid(flat)
    pt = torch.where(m > 0, p, 1 - p)
    focal = (-(1 - pt) ** 2 * torch.log(pt.clamp(min=1e-6))).mean()
    return ce + focal_w * focal * 100.0


def _where_token_runs(ids, image_token_id):
    """Contiguous runs of image tokens in a 1-D id tensor -> [(start, end)]."""
    idx = torch.nonzero(ids == image_token_id).flatten().tolist()
    if not idx:
        return []
    runs, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i != prev + 1:
            runs.append((start, prev + 1))
            start = i
        prev = i
    runs.append((start, prev + 1))
    return runs
