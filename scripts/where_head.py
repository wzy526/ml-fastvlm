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

  python scripts/where_head.py --model_path $E/checkpoint-1000 \
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


def load_train(jsons, image_root, max_per_json):
    rows = []
    for j in jsons:
        data = json.load(open(os.path.expanduser(j)))
        random.shuffle(data)
        k = 0
        for s in data:
            b = s.get("bbox")
            if not (isinstance(b, (list, tuple)) and len(b) == 4):
                continue
            b = [float(v) for v in b]
            if not (b[2] > b[0] and b[3] > b[1] and max(b) <= 1.0):
                continue
            im = s.get("image")
            im = im[0] if isinstance(im, list) else im
            if not im:
                continue
            q = question_of(s)
            if not q:
                continue
            path = im if os.path.isabs(im) else os.path.join(image_root, im)
            rows.append({"path": path, "box": tuple(b), "q": q, "src": os.path.basename(j)})
            k += 1
            if k >= max_per_json:
                break
        print(f"[where] {os.path.basename(j)}: {k} samples with a box")
    random.shuffle(rows)
    return rows


# ───────────────────────── frozen model features ─────────────────────────

class Frozen:
    def __init__(self, model_path, txt_layer, hd_cap, attn, device):
        from transformers import AutoProcessor
        from llava.model.language_model.modeling_qwen3_5_dat import Qwen3_5DATForConditionalGeneration
        self.model = Qwen3_5DATForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map={"": 0}, attn_implementation=attn).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.proc = AutoProcessor.from_pretrained(model_path, min_pixels=FACTOR * FACTOR, max_pixels=100_000_000)
        self.tok = self.proc.tokenizer
        self.txt_layer = txt_layer
        self.hd_cap = hd_cap
        self.device = device
        self.dtype = torch.bfloat16

    @torch.no_grad()
    def hd_features(self, img):
        """(H, W, C) at the probe's HD geometry: native resolution capped at hd_cap, 32 px cells."""
        total = min(img.width * img.height, self.hd_cap)
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


def cell_targets(box, H, W, dilate):
    """Binary (H, W) mask of the cells the GT box (fractions) touches, dilated."""
    x0 = max(0, int(box[0] * W) - dilate)
    y0 = max(0, int(box[1] * H) - dilate)
    x1 = min(W - 1, int(math.ceil(box[2] * W)) - 1 + dilate)
    y1 = min(H - 1, int(math.ceil(box[3] * H)) - 1 + dilate)
    m = torch.zeros(H, W)
    m[y0:y1 + 1, x0:x1 + 1] = 1.0
    return m


def head_loss(logits, mask, focal_w=0.5):
    flat, m = logits.reshape(-1), mask.reshape(-1).to(logits.device)
    ce = -(torch.log_softmax(flat, -1) * (m / m.sum().clamp(min=1))).sum()
    p = torch.sigmoid(flat)
    pt = torch.where(m > 0, p, 1 - p)
    focal = (-(1 - pt) ** 2 * torch.log(pt.clamp(min=1e-6))).mean()
    return ce + focal_w * focal * 100.0


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

@torch.no_grad()
def eval_vstar(frozen, head, samples, min_cells, out_cells, out_path=None, tag=""):
    head.eval()
    n = n_win = n_box = n_win3 = n_box3 = 0
    ious, rows, per_cat = [], [], {}
    for i, s in enumerate(samples):
        gt = gt_box_frac(s)
        if gt is None:
            continue
        n += 1
        feat = frozen.hd_features(s["image"])
        txt = frozen.text_states(s["prompt"].split("\n")[0])
        logits = head(feat, txt)
        H, W = logits.shape
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
    print(f"[where]{tag} V* {n}: top-1 cell centre in GT window {n_win}/{n} ({100 * n_win / max(n, 1):.1f}%), "
          f"in GT box {n_box}/{n}; top-3 any: window {n_win3}/{n}, box {n_box3}/{n}; "
          f"IoU of a {out_cells}-cell square {sum(ious) / max(len(ious), 1):.3f}")
    for c, (m, w, b) in sorted(per_cat.items()):
        print(f"         {c}: n={m} window {w} box {b}")
    if out_path:
        json.dump(rows, open(os.path.expanduser(out_path), "w"))
        print(f"[where] boxes -> {out_path}  (probe_whd_qwen35.py --hd_source ground_window --box_json ...)")
    head.train()
    return n_win / max(n, 1)


@torch.no_grad()
def eval_heldout(frozen, head, rows, min_cells, out_cells, tag=""):
    head.eval()
    n = n_win = n_box = 0
    for r in rows:
        try:
            img = Image.open(r["path"]).convert("RGB")
        except Exception:
            continue
        n += 1
        logits = head(frozen.hd_features(img), frozen.text_states(r["q"]))
        hd_w, hd_h = hd_size(img)
        win = gt_window(r["box"], hd_w, hd_h, min_cells)
        h = box_hits(top_boxes(logits, 1, out_cells)[0][1], r["box"], win)
        n_win += h[0]
        n_box += h[1]
    print(f"[where]{tag} held-out {n}: window {n_win}/{n} ({100 * n_win / max(n, 1):.1f}%), in box {n_box}/{n}")
    head.train()


# ───────────────────────── main ─────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--train_json", nargs="*", default=[])
    ap.add_argument("--image_root", default="~/sft_data/train_split")
    ap.add_argument("--max_per_json", type=int, default=20000)
    ap.add_argument("--heldout", type=int, default=200, help="training rows kept aside for an in-domain check")
    ap.add_argument("--steps", type=int, default=3000, help="optimizer steps (x --accum samples each)")
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--dilate", type=int, default=1)
    ap.add_argument("--txt_layer", type=int, default=15)
    ap.add_argument("--hd_cap", type=int, default=5017600)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--min_cells", type=int, default=20, help="GT window size for the hit metric")
    ap.add_argument("--out_cells", type=int, default=4, help="side of the square written as the box")
    ap.add_argument("--vstar_root", default="~/data/vstar_bench")
    ap.add_argument("--vstar_n", type=int, default=300)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--save", default=None)
    ap.add_argument("--load", default=None, help="skip training, evaluate this head")
    ap.add_argument("--out_boxes", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda"
    frozen = Frozen(args.model_path, args.txt_layer, args.hd_cap, args.attn, device)
    vstar = load_vstar(args.vstar_root, args.vstar_n)
    c_img = frozen.hd_features(vstar[0]["image"]).shape[-1]
    c_txt = frozen.text_states("what colour is the cup?").shape[-1]
    head = WhereHead(c_img, c_txt, args.d).to(device)
    print(f"[where] head: {sum(p.numel() for p in head.parameters()) / 1e6:.2f} M params; "
          f"C_img={c_img} C_txt={c_txt} (LM layer {args.txt_layer}); HD cap {args.hd_cap} px")

    if args.load:
        head.load_state_dict(torch.load(os.path.expanduser(args.load), map_location=device))
        eval_vstar(frozen, head, vstar, args.min_cells, args.out_cells, args.out_boxes, " [loaded]")
        return

    rows = load_train(args.train_json, os.path.expanduser(args.image_root), args.max_per_json)
    held, rows = rows[: args.heldout], rows[args.heldout:]
    print(f"[where] {len(rows)} train rows, {len(held)} held-out; {args.steps} steps x {args.accum}")
    eval_vstar(frozen, head, vstar, args.min_cells, args.out_cells, None, " [untrained]")

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)))
    t0, seen, run_loss, skipped = time.time(), 0, 0.0, 0
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        for _ in range(args.accum):
            r = rows[seen % len(rows)]
            seen += 1
            try:
                img = Image.open(r["path"]).convert("RGB")
            except Exception:
                skipped += 1
                continue
            feat = frozen.hd_features(img)
            txt = frozen.text_states(r["q"])
            logits = head(feat, txt)
            mask = cell_targets(r["box"], *logits.shape, args.dilate)
            if mask.sum() == 0:
                skipped += 1
                continue
            loss = head_loss(logits, mask) / args.accum
            loss.backward()
            run_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 50 == 0:
            print(f"[where] step {step}/{args.steps} loss {run_loss / 50:.3f} lr {sched.get_last_lr()[0]:.2e} "
                  f"{(time.time() - t0) / step:.2f}s/step skipped {skipped}", flush=True)
            run_loss = 0.0
        if step % args.eval_every == 0 or step == args.steps:
            eval_heldout(frozen, head, held, args.min_cells, args.out_cells, f" [step {step}]")
            eval_vstar(frozen, head, vstar, args.min_cells, args.out_cells,
                       args.out_boxes if step == args.steps else None, f" [step {step}]")
            if args.save:
                torch.save(head.state_dict(), os.path.expanduser(args.save))


if __name__ == "__main__":
    main()
