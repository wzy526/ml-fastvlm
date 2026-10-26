"""Quick check: can a small SUPERVISED localisation head on the frozen DAT
model's HD features (the whole image at native resolution, one cell per 32 px,
= what the HD branch already computes) find the question's target better than
the LM's own grounding (V*: 77 % window hit, IoU 0.01) or Grounding DINO (79 %)?

Everything is frozen except the head; nothing in the model or trainer changes.
  question  -> LM text-only forward, hidden states of one layer   (T, C)
  HD image  -> model._generate_hd_features                        (H, W, C)
  head      -> per-cell logit = max over question tokens of <conv(cell), token>
  loss      -> soft CE over cells (uniform on GT-box cells) + focal BCE
  eval (V*) -> top-1 cell centre in the GT window / in the GT box, top-3 any;
               writes a --box_json file for probe_whd_qwen35.py (ground_window)

  torchrun --nproc_per_node 8 scripts/where_head.py --model_path $E/checkpoint-1000 \
      --train_json $OSS_DATA/extra_0916/viscot_bbox.train.json \
                   $OSS_DATA/extra_0920/viscot_nat_bbox.json \
      --image_root ~/sft_data/train_split --steps 3000 \
      --vstar_root ~/data/vstar_bench --save $E/loc/where_head.pt --out_boxes $E/loc/where_vstar.json
"""
import argparse
import json
import math
import os
import random
import re
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gdino_probe import load_vstar, gt_box_frac, gt_window, box_hits, hd_size, FACTOR  # noqa: E402

BOX_RE = re.compile(r"\[\s*\d(?:\.\d+)?\s*,\s*\d(?:\.\d+)?\s*,\s*\d(?:\.\d+)?\s*,\s*\d(?:\.\d+)?\s*\]")


# ───────────────────────── data ─────────────────────────

def question_of(s):
    conv = s.get("conversations") or []
    q = conv[0].get("value", "") if conv else ""
    q = q.replace("<image>", "").strip()
    q = BOX_RE.sub("", q).strip()                 # never leak a box through the text
    return q.split("\n")[0].strip() or q


def load_train(jsons, image_root, max_per_json, say=print):
    """max_per_json: one int for all jsons, or one per json (last value repeats)."""
    caps = list(max_per_json) if isinstance(max_per_json, (list, tuple)) else [max_per_json]
    rows = []
    for ji, j in enumerate(jsons):
        cap = caps[min(ji, len(caps) - 1)]
        data = json.load(open(os.path.expanduser(j)))
        random.shuffle(data)
        k = 0
        for s in data:
            boxes = s.get("bboxes") or [s.get("bbox")]
            ok = []
            for b in boxes:
                if not (isinstance(b, (list, tuple)) and len(b) == 4):
                    break
                b = [float(v) for v in b]
                if not (b[2] > b[0] and b[3] > b[1] and max(b) <= 1.0):
                    break
                ok.append(tuple(b))
            if not ok or len(ok) != len(boxes):
                continue
            im = s.get("image")
            im = im[0] if isinstance(im, list) else im
            if not im:
                continue
            q = question_of(s)
            if not q:
                continue
            path = im if os.path.isabs(im) else os.path.join(image_root, im)
            rows.append({"path": path, "box": ok[0], "boxes": ok, "q": q, "src": os.path.basename(j)})
            k += 1
            if k >= cap:
                break
        n_multi = sum(len(r["boxes"]) > 1 for r in rows[len(rows) - k:])
        say(f"[where] {os.path.basename(j)}: {k} samples with a box ({n_multi} with >= 2 boxes)")
    random.shuffle(rows)
    return rows


# ───────────────────────── frozen model features ─────────────────────────

class Frozen:
    def __init__(self, model_path, txt_layer, hd_cap, attn, device, local_rank=0):
        from transformers import AutoProcessor
        from llava.model.language_model.modeling_qwen3_5_dat import Qwen3_5DATForConditionalGeneration
        self.model = Qwen3_5DATForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map={"": local_rank},
            attn_implementation=attn).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.proc = AutoProcessor.from_pretrained(model_path, min_pixels=FACTOR * FACTOR, max_pixels=100_000_000)
        self.tok = self.proc.tokenizer
        self.txt_layer = txt_layer
        self.hd_cap = hd_cap
        self.device = device
        self.dtype = torch.bfloat16
        self.feat_cap = None        # --feat lr: tok x 1024 px (the LM's LR view), else the HD cap
        self.lm_layer = None        # --feat lm:L: cells = LM layer-L states of the LR image tokens
        self.lr_proc = None
        self.image_token_id = getattr(self.model.config, "image_token_id", None)
        self.lm_calls = 0

    def set_lm_view(self, model_path, lm_layer, lr_tok):
        from transformers import AutoProcessor
        self.lm_layer = lm_layer
        self.lr_proc = AutoProcessor.from_pretrained(model_path, min_pixels=FACTOR * FACTOR,
                                                     max_pixels=lr_tok * FACTOR * FACTOR)

    def feats(self, img):
        return self.hd_features(img, self.feat_cap)

    def pair(self, img, q):
        """(cells (H, W, C), text (T, C)) for the head."""
        if self.lm_layer is not None:
            return self.lm_features(img, q)
        return self.feats(img), self.text_states(q)

    @torch.no_grad()
    def lm_features(self, img, q):
        """One LM forward on [LR image, question] (the student's own prefill):
        cells = layer-L hidden states at the image-token positions, reshaped to
        the merged grid; text = layer-L states of everything after the image
        (question + template tail). These cells have attended to the question
        for L layers -- the representation the LM's own text grounding (77 %)
        and SD-RPN's RPN read from; the frozen ViT cells (48 %) have not."""
        messages = [{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": [{"type": "image", "image": img},
                                                 {"type": "text", "text": q}]}]
        text = self.lr_proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=False)
        inp = self.lr_proc(text=[text], images=[img], return_tensors="pt")
        ids = inp["input_ids"].to(self.device)
        kw = {}                                   # everything the processor made (mm_token_type_ids too)
        for k, v in inp.items():
            if isinstance(v, torch.Tensor):
                kw[k] = v.to(device=self.device, dtype=self.dtype) if k == "pixel_values" else v.to(self.device)
        out = self.model(**kw, output_hidden_states=True, return_dict=True)
        hs = out.hidden_states[self.lm_layer][0]              # (S, C)
        is_img = ids[0] == self.image_token_id
        thw = inp["image_grid_thw"][0]
        gh, gw = int(thw[1]) // 2, int(thw[2]) // 2
        cells = hs[is_img]
        assert cells.shape[0] == gh * gw, (cells.shape, gh, gw)
        last_img = int(torch.nonzero(is_img)[-1])
        txt = hs[last_img + 1:]
        self.lm_calls += 1
        return cells.reshape(gh, gw, -1), txt

    @torch.no_grad()
    def hd_features(self, img, cap=None):
        """(H, W, C) at the probe's HD geometry: native resolution capped at hd_cap
        (or `cap`: e.g. the LR token budget x 1024 px = what the LM's LR view is
        built from), 32 px cells."""
        total = min(img.width * img.height, cap or self.hd_cap)
        aspect = img.width / img.height
        h = int(math.sqrt(total / aspect))
        w = int(h * aspect)
        hd_w, hd_h = max(FACTOR, w // FACTOR * FACTOR), max(FACTOR, h // FACTOR * FACTOR)
        hd_img = img.resize((hd_w, hd_h), Image.BICUBIC)
        hr = self.proc.image_processor(images=[hd_img], return_tensors="pt")
        feat = self.model._generate_hd_features(
            hr["pixel_values"].to(device=self.device, dtype=self.dtype),
            hr["image_grid_thw"].to(self.device))[0]
        return feat                                           # (H, W, C) bf16

    @torch.no_grad()
    def text_states(self, q):
        ids = self.tok(q, return_tensors="pt", truncation=True, max_length=96).to(self.device)
        out = self.model(input_ids=ids["input_ids"], attention_mask=ids["attention_mask"],
                         output_hidden_states=True, return_dict=True)
        return out.hidden_states[self.txt_layer][0]           # (T, C) bf16


# ───────────────────────── the head ─────────────────────────

class WhereHead(nn.Module):
    def __init__(self, c_img, c_txt, d=256):
        super().__init__()
        self.img_in = nn.Sequential(nn.LayerNorm(c_img), nn.Linear(c_img, d))
        self.ctx = nn.Sequential(nn.Conv2d(d, d, 3, padding=1), nn.GELU(), nn.Conv2d(d, d, 3, padding=1))
        self.txt_in = nn.Sequential(nn.LayerNorm(c_txt), nn.Linear(c_txt, d), nn.GELU(), nn.Linear(d, d))
        self.scale = nn.Parameter(torch.tensor(1.0 / math.sqrt(d)))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, feat, txt):
        """feat (H, W, C) ; txt (T, C_txt) -> logits (H, W)"""
        x = self.img_in(feat.float())                          # H W d
        x = x + self.ctx(x.permute(2, 0, 1)[None])[0].permute(1, 2, 0)
        t = self.txt_in(txt.float())                           # T d
        s = torch.einsum("hwd,td->hwt", x, t) * self.scale     # H W T
        return s.max(-1).values + self.bias


class XBlock(nn.Module):
    """One round of question <-> cells interaction: the question tokens read
    the cells, the cells read the (updated) question, then a 3x3 conv (local
    context) and an FFN on the cells."""

    def __init__(self, d, heads):
        super().__init__()
        self.t_norm = nn.LayerNorm(d)
        self.t_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.c_norm = nn.LayerNorm(d)
        self.c_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.conv = nn.Conv2d(d, d, 3, padding=1, groups=d)            # depthwise, cheap
        self.f_norm = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, cells, txt, H, W):
        """cells (1, N, d), txt (1, T, d)"""
        tq = self.t_norm(txt)
        txt = txt + self.t_attn(tq, cells, cells, need_weights=False)[0]
        cq = self.c_norm(cells)
        cells = cells + self.c_attn(cq, txt, txt, need_weights=False)[0]
        g = cells.transpose(1, 2).reshape(1, -1, H, W)
        cells = cells + self.conv(g).flatten(2).transpose(1, 2)
        cells = cells + self.ffn(self.f_norm(cells))
        return cells, txt


class XAttnHead(nn.Module):
    """--head xattn: a small question<->cell transformer (what the dot-product
    head lacks: the cells can see which words were already matched elsewhere,
    the words can see the whole map) + a learned 2-D position embedding so
    'left of' / 'second row' style phrases can be used. Per-cell logit at the end."""

    def __init__(self, c_img, c_txt, d=384, layers=2, heads=6, max_hw=512):
        super().__init__()
        self.img_in = nn.Sequential(nn.LayerNorm(c_img), nn.Linear(c_img, d))
        self.txt_in = nn.Sequential(nn.LayerNorm(c_txt), nn.Linear(c_txt, d))
        self.row = nn.Embedding(max_hw, d)
        self.col = nn.Embedding(max_hw, d)
        nn.init.normal_(self.row.weight, std=0.02)
        nn.init.normal_(self.col.weight, std=0.02)
        self.blocks = nn.ModuleList(XBlock(d, heads) for _ in range(layers))
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, feat, txt):
        H, W, _ = feat.shape
        x = self.img_in(feat.float())
        n_pos = self.row.num_embeddings                        # 5 Mpx panoramas reach ~140 cells a side
        ri = torch.arange(H, device=x.device).clamp_max(n_pos - 1)
        ci = torch.arange(W, device=x.device).clamp_max(n_pos - 1)
        pos = self.row(ri)[:, None] + self.col(ci)[None]
        cells = (x + pos).reshape(1, H * W, -1)
        t = self.txt_in(txt.float())[None]
        for blk in self.blocks:
            cells, t = blk(cells, t, H, W)
        return self.out(cells)[0, :, 0].reshape(H, W)


def cell_targets(box, H, W, dilate):
    """Binary (H, W) mask of the cells the GT box (fractions) touches, dilated."""
    x0 = max(0, int(box[0] * W) - dilate)
    y0 = max(0, int(box[1] * H) - dilate)
    x1 = min(W - 1, int(math.ceil(box[2] * W)) - 1 + dilate)
    y1 = min(H - 1, int(math.ceil(box[3] * H)) - 1 + dilate)
    m = torch.zeros(H, W)
    m[y0:y1 + 1, x0:x1 + 1] = 1.0
    return m


def multi_targets(boxes, H, W, dilate):
    """(union mask, CE target) for one or more GT boxes. Each box gets 1/K of the
    mass, uniform on its own cells: uniform on the union would let a large box
    take nearly all of it (GQA box areas span 40x) and the small object's peak
    would never be learned. One box -> the original target."""
    masks = [cell_targets(b, H, W, dilate) for b in boxes]
    union = torch.stack(masks).amax(0)
    tgt = sum(m / m.sum().clamp(min=1) for m in masks) / len(masks)
    return union, tgt


def head_loss(logits, mask, focal_w=0.5, ce_weight=1.0, target=None):
    """ce_weight 1 is the original loss. The focal term (~x100) swamps the CE, so
    the softmax spreads even when the argmax is right; --ce_weight raises the CE,
    whose target (uniform on the GT-box cells, 0 elsewhere) is the sampling density.
    `target` (multi_targets) replaces that uniform target for multi-box rows."""
    flat, m = logits.reshape(-1), mask.reshape(-1).to(logits.device)
    t = (m / m.sum().clamp(min=1)) if target is None else target.reshape(-1).to(logits.device)
    ce = -(torch.log_softmax(flat, -1) * t).sum()
    p = torch.sigmoid(flat)
    pt = torch.where(m > 0, p, 1 - p)
    focal = (-(1 - pt) ** 2 * torch.log(pt.clamp(min=1e-6))).mean()
    return ce_weight * ce + focal_w * focal * 100.0


def softmax_mass(logits, mask):
    """Fraction of the softmax sitting on the positive cells. This, not the argmax,
    is what a density-driven DAT grid samples from."""
    p = torch.softmax(logits.detach().float().reshape(-1), -1)
    m = (mask.reshape(-1) > 0).to(p)
    return float((p * m).sum())


def peak_stats(logits, boxes, tau=0.3, min_dist=4):
    """NMS peaks of the head map (where_peaks_probe) vs the GT boxes (xyxy
    fractions): (n_peaks, missed: peak 1's window misses some box centre,
    recovered: every box peak 1 misses is covered by another peak)."""
    from where_peaks_probe import nms_peaks, softmax2d, covers
    p = softmax2d(logits.detach().float().tolist())
    gh, gw = len(p), len(p[0])
    peaks = nms_peaks(p, min_dist, tau, 4)
    centres = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]
    miss = [c for c in centres if not covers(peaks[0], c, gh, gw)]
    rec = bool(miss) and all(any(covers(pk, c, gh, gw) for pk in peaks[1:]) for c in miss)
    return len(peaks), bool(miss), rec


class PeakTally:
    """>= 2-peak rate for one- vs multi-box questions (should be low / high),
    and how often a box missed by peak 1 is picked up by a later peak."""

    def __init__(self):
        self.n = {1: 0, 2: 0}
        self.two = {1: 0, 2: 0}
        self.miss = self.rec = 0
        self.m_hi = self.m_lo = 0.0
        self.lo_small = 0

    def add(self, logits, boxes):
        k = 1 if len(boxes) == 1 else 2
        n_pk, miss, rec = peak_stats(logits, boxes)
        self.n[k] += 1
        self.two[k] += n_pk >= 2
        if k == 2:
            self.miss += miss
            self.rec += rec
            H, W = logits.shape
            ms = sorted(softmax_mass(logits, cell_targets(b, H, W, 0)) for b in boxes)
            self.m_hi += ms[-1]
            self.m_lo += ms[0]
            self.lo_small += ms[0] < 0.05

    def line(self):
        f = lambda k: f"{self.two[k]}/{self.n[k]} ({100 * self.two[k] / max(self.n[k], 1):.0f}%)"
        n2 = max(self.n[2], 1)
        return (f">=2 peaks@0.3: one-box {f(1)}, multi-box {f(2)}; "
                f"multi-box: peak 1 misses a box {self.miss}, a later peak covers it {self.rec}; "
                f"per-box mass hi {self.m_hi / n2:.3f} lo {self.m_lo / n2:.3f}, lo < 0.05 in {self.lo_small}/{self.n[2]}")


def top_boxes(logits, k, cells):
    """k highest cells -> (score, box in fractions) squares of `cells` cells."""
    H, W = logits.shape
    probs = torch.softmax(logits.reshape(-1).float(), -1)
    vals, idx = probs.topk(min(k, H * W))
    out = []
    for v, i in zip(vals.tolist(), idx.tolist()):
        cy, cx = divmod(i, W)
        half = cells / 2
        out.append((v, ((cx + 0.5 - half) / W, (cy + 0.5 - half) / H,
                        (cx + 0.5 + half) / W, (cy + 0.5 + half) / H)))
    return out


# ───────────────────────── eval ─────────────────────────

def vstar_query(s, mode):
    """question: first line of the prompt (what the head was trained on).
    target: the sidecar object names ('cup, glove') — an upper bound that
    separates 'the features cannot find it' from 'the head cannot work out
    WHAT to find from the question'."""
    if mode == "target" and s.get("targets"):
        return ", ".join(s["targets"])
    return s["prompt"].split("\n")[0]


@torch.no_grad()
def eval_vstar(frozen, head, samples, min_cells, out_cells, out_path=None, tag="", query="question"):
    head.eval()
    n = n_win = n_box = n_win3 = n_box3 = 0
    mass_sum = 0.0
    ious, rows, per_cat = [], [], {}
    peaks = PeakTally()
    for i, s in enumerate(samples):
        gt = gt_box_frac(s)
        if gt is None:
            continue
        n += 1
        feat, txt = frozen.pair(s["image"], vstar_query(s, query))
        logits = head(feat, txt)
        H, W = logits.shape
        mass_sum += softmax_mass(logits, cell_targets(gt, H, W, 0))
        iw, ih = s["image"].size
        peaks.add(logits, [(b[0] / iw, b[1] / ih, (b[0] + b[2]) / iw, (b[1] + b[3]) / ih) for b in s["bboxes"]])
        hd_w, hd_h = hd_size(s["image"])
        win = gt_window(gt, hd_w, hd_h, min_cells)
        tops = top_boxes(logits, 3, out_cells)
        hits = [box_hits(b, gt, win) for _, b in tops]
        n_win += hits[0][0]
        n_box += hits[0][1]
        ious.append(hits[0][2])
        n_win3 += any(h[0] for h in hits)
        n_box3 += any(h[1] for h in hits)
        c = per_cat.setdefault(s["category"], [0, 0, 0])
        c[0] += 1
        c[1] += hits[0][0]
        c[2] += hits[0][1]
        rows.append({"i": i, "box": list(tops[0][1]), "score": tops[0][0],
                     "boxes": [list(b) for _, b in tops], "scores": [v for v, _ in tops],
                     "grid": [H, W], "in_win": bool(hits[0][0]), "in_box": bool(hits[0][1])})
    print(f"[where]{tag} V* {n} (query={query}): top-1 cell centre in GT window {n_win}/{n} ({100 * n_win / max(n, 1):.1f}%), "
          f"in GT box {n_box}/{n}; top-3 any: window {n_win3}/{n}, box {n_box3}/{n}; "
          f"IoU of a {out_cells}-cell square {sum(ious) / max(len(ious), 1):.3f}; "
          f"softmax mass on the GT-box cells {mass_sum / max(n, 1):.3f}")
    for c, (m, w, b) in sorted(per_cat.items()):
        print(f"         {c}: n={m} window {w} box {b}")
    print(f"         V* {peaks.line()}")
    if out_path:
        json.dump(rows, open(os.path.expanduser(out_path), "w"))
        print(f"[where] boxes -> {out_path}  (probe_whd_qwen35.py --hd_source ground_window --box_json ...)")
    head.train()
    return n_win / max(n, 1)


@torch.no_grad()
def eval_heldout(frozen, head, rows, min_cells, out_cells, tag=""):
    head.eval()
    n = n1 = n_win = n_box = 0
    mass_sum = mass_multi = 0.0
    peaks = PeakTally()
    for r in rows:
        try:
            img = Image.open(r["path"]).convert("RGB")
        except Exception:
            continue
        n += 1
        logits = head(*frozen.pair(img, r["q"]))
        boxes = r.get("boxes") or [r["box"]]
        union, _ = multi_targets(boxes, *logits.shape, 0)
        m = softmax_mass(logits, union)
        mass_sum += m
        peaks.add(logits, boxes)
        if len(boxes) > 1:
            mass_multi += m
            continue
        n1 += 1
        hd_w, hd_h = hd_size(img)
        win = gt_window(r["box"], hd_w, hd_h, min_cells)
        h = box_hits(top_boxes(logits, 1, out_cells)[0][1], r["box"], win)
        n_win += h[0]
        n_box += h[1]
    nm = n - n1
    print(f"[where]{tag} held-out {n} ({n1} one-box): one-box window {n_win}/{n1} ({100 * n_win / max(n1, 1):.1f}%), "
          f"in box {n_box}/{n1}; softmax mass on the GT-box cells {mass_sum / max(n, 1):.3f} "
          f"(multi-box {mass_multi / max(nm, 1):.3f})")
    print(f"         held-out {peaks.line()}")
    head.train()


# ───────────────────────── main ─────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--train_json", nargs="*", default=[])
    ap.add_argument("--image_root", default="~/sft_data/train_split")
    ap.add_argument("--max_per_json", type=int, nargs="+", default=[20000],
                    help="cap per --train_json (one value for all, or one per json)")
    ap.add_argument("--heldout", type=int, default=200, help="training rows kept aside for an in-domain check")
    ap.add_argument("--steps", type=int, default=3000, help="optimizer steps (x --accum samples each)")
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--dilate", type=int, default=1)
    ap.add_argument("--ce_weight", type=float, default=1.0,
                    help="weight of the softmax CE. 1 = original loss; 20 makes the softmax "
                         "track the GT-box density instead of being set by the focal term")
    ap.add_argument("--txt_layer", type=int, default=15)
    ap.add_argument("--hd_cap", type=int, default=5017600)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--min_cells", type=int, default=20, help="GT window size for the hit metric")
    ap.add_argument("--out_cells", type=int, default=4, help="side of the square written as the box")
    ap.add_argument("--vstar_root", default="~/data/vstar_bench")
    ap.add_argument("--vstar_n", type=int, default=300)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--save", default=None)
    ap.add_argument("--load", default=None, help="init head from this file; with --train_json continue training, else eval only")
    ap.add_argument("--out_boxes", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--head", choices=["dot", "xattn"], default="dot",
                    help="dot: word-cell dot product, max over words (v1). xattn: --xlayers rounds of "
                         "question<->cell cross-attention + depthwise conv + FFN, per-cell logit")
    ap.add_argument("--xlayers", type=int, default=2)
    ap.add_argument("--xheads", type=int, default=6)
    ap.add_argument("--feat", default="hd",
                    help="hd: cells from the frozen ViT on the HD grid (native res, hd_cap); "
                         "lr: same ViT on the LR view (--lr_tok x 1024 px); "
                         "lm:L: LM layer-L hidden states of the LR image tokens after one "
                         "prefill of [image, question] (text = the same layer's question states)")
    ap.add_argument("--lr_tok", type=int, default=640)
    ap.add_argument("--eval_query", default="question",
                    help="V* eval text, comma list of question|target (target = sidecar object names)")
    ap.add_argument("--stagger", type=float, default=8.0,
                    help="seconds x local_rank to wait before the first forward (Triton compile)")
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # data parallel over the GPUs torchrun gives us: every rank runs the frozen
    # model on its own shard of the rows, DDP averages the (tiny) head gradients
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    if world > 1:
        import torch.distributed as dist
        dist.init_process_group("nccl")
    main_rank = rank == 0
    say = print if main_rank else (lambda *a, **k: None)
    barrier = (lambda: dist.barrier()) if world > 1 else (lambda: None)
    if world > 1:
        # 8 ranks compiling the same Triton kernels (fla linear attention in the
        # Qwen3.5 trunk) against one cache dir deadlocked on the first HD
        # forward (py-spy: all ranks inside _generate_hd_features): give every
        # rank its own cache and stagger the first forward
        # Local disk, not the quota'd JuiceFS home (1020: rank 1/4 died with
        # 'Disk quota exceeded' writing a new kernel). TRITON_CACHE_ROOT overrides.
        cache_root = os.environ.get("TRITON_CACHE_ROOT", f"/tmp/triton_cache_{os.environ.get('USER', 'u')}")
        os.environ["TRITON_CACHE_DIR"] = os.path.join(cache_root, f"rank{local_rank}")
        os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)

    frozen = Frozen(args.model_path, args.txt_layer, args.hd_cap, args.attn, device, local_rank)
    time.sleep(args.stagger * local_rank)
    if args.feat == "lr":
        frozen.feat_cap = args.lr_tok * FACTOR * FACTOR
    elif args.feat.startswith("lm:"):
        frozen.set_lm_view(args.model_path, int(args.feat.split(":")[1]), args.lr_tok)
    elif args.feat != "hd":
        raise SystemExit(f"--feat {args.feat}: expected hd | lr | lm:L")
    vstar = load_vstar(args.vstar_root, args.vstar_n)
    f0, t0_ = frozen.pair(vstar[0]["image"], "what colour is the cup?")
    c_img, c_txt = f0.shape[-1], t0_.shape[-1]
    if args.head == "xattn":
        head = XAttnHead(c_img, c_txt, args.d, args.xlayers, args.xheads).to(device)
    else:
        head = WhereHead(c_img, c_txt, args.d).to(device)
    say(f"[where] head={args.head} d={args.d}: {sum(p.numel() for p in head.parameters()) / 1e6:.2f} M params; "
        f"feat={args.feat} (sample 0 grid {tuple(f0.shape[:2])}) C_img={c_img} C_txt={c_txt} (LM layer {args.txt_layer}); "
        f"{world} GPU(s) -> {args.steps} steps x {args.accum} x {world} = {args.steps * args.accum * world} samples")

    if args.load:
        head.load_state_dict(torch.load(os.path.expanduser(args.load), map_location=device))
        say(f"[where] loaded head weights from {args.load}")
        if not args.train_json:                                # eval-only
            if main_rank:
                for q in args.eval_query.split(","):
                    eval_vstar(frozen, head, vstar, args.min_cells, args.out_cells,
                               args.out_boxes if q == "question" else None, " [loaded]", query=q)
            return

    rows = load_train(args.train_json, os.path.expanduser(args.image_root), args.max_per_json, say)
    held, rows = rows[: args.heldout], rows[args.heldout:]
    rows = rows[rank::world]                                   # same shuffle on every rank (seeded)
    say(f"[where] {len(rows)} train rows per rank, {len(held)} held-out")
    if args.steps == 0:                                        # eval-only on the same held-out split
        if main_rank:
            eval_heldout(frozen, head, held, args.min_cells, args.out_cells, " [loaded]")
            eval_vstar(frozen, head, vstar, args.min_cells, args.out_cells, None, " [loaded]")
        return
    if main_rank:
        eval_vstar(frozen, head, vstar, args.min_cells, args.out_cells, None, " [untrained]")
    barrier()
    train_head = head
    if world > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        train_head = DDP(head, device_ids=[local_rank])

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)))
    t0, seen, run_loss, run_mass, run_n, skipped = time.time(), 0, 0.0, 0.0, 0, 0
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        for _ in range(args.accum):
            while True:                                        # every rank does exactly accum backwards
                r = rows[seen % len(rows)]
                seen += 1
                try:
                    img = Image.open(r["path"]).convert("RGB")
                    break
                except Exception:
                    skipped += 1
            feat, txt = frozen.pair(img, r["q"])
            logits = train_head(feat, txt)
            mask, tgt = multi_targets(r["boxes"], *logits.shape, args.dilate)
            if mask.sum() == 0:
                skipped += 1
                loss = logits.sum() * 0.0                      # keep DDP's collective in step
            else:
                loss = head_loss(logits, mask, ce_weight=args.ce_weight, target=tgt) / args.accum
                run_mass += softmax_mass(logits, mask)
                run_n += 1
            loss.backward()
            run_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 50 == 0:
            say(f"[where] step {step}/{args.steps} loss {run_loss / 50:.3f} "
                f"mass_in {run_mass / max(run_n, 1):.3f} lr {sched.get_last_lr()[0]:.2e} "
                f"{(time.time() - t0) / step:.2f}s/step skipped {skipped}", flush=True)
            run_loss, run_mass, run_n = 0.0, 0.0, 0
        if step % args.eval_every == 0 or step == args.steps:
            if main_rank:
                eval_heldout(frozen, head, held, args.min_cells, args.out_cells, f" [step {step}]")
                eval_vstar(frozen, head, vstar, args.min_cells, args.out_cells,
                           args.out_boxes if step == args.steps else None, f" [step {step}]")
                if args.save:
                    torch.save(head.state_dict(), os.path.expanduser(args.save))
            barrier()


if __name__ == "__main__":
    main()
