#!/usr/bin/env python3
"""Coarse-to-fine DAT on HR ViT tokens: question only in the coarse stage, a plain
(text-free) DAT offset head inside each window, sampling on the HR feature map.

  LR image + question -> LM layer 8 -> where-head map -> K windows (NMS peaks, default 4)
  whole image (<= --hr_full_cap tokens) -> frozen ViT -> HR map; X_k = the map under window k
  (--hr_source window: each window's pixels through the ViT on their own instead)
  offset head (shared over windows, one offset group, all C channels, no FiLM):
      X_k -> dwconv kxk -> LayerNorm -> GELU -> adaptive pool to (gh, gw) -> 1x1 (no bias,
      zero init) -> tanh * max_off            (offsets in reference-cell units)
  P_k = R_k + 2 * off / (gw, gh)   (R_k = cell centres of a uniform gh x gw grid, window-local
      [-1, 1] coords, clamped to the window)  ->  Z_k = grid_sample(X_k, P_k)  (gh*gw, C)
  Z_1..Z_K replace the image tokens of K second images in [LR, q, hint, img_1..img_K, q];
  --pe global: their M-RoPE h / w come from the sample points mapped back to full-image
  coordinates (global_positions); loss = CE over the GT answer tokens; only the head trains.

Budget: --crop_tok total, split evenly over the K windows (each window's grid keeps its
aspect, crop_geometry), or --grid N for N x N per window.

Held-out (training) and accuracy (--eval_dataset) both score the same windows four ways:
  crop   window pixels resized to gh x gw tokens -> ViT (the current pixel path)
  fpool  X_k average-pooled to gh x gw
  fzero  X_k bilinearly sampled at R_k (the head at step 0)
  net    X_k sampled at R_k + learned offsets
Accuracy uses option-letter scoring (argmax over the option letters' logits at the first
answer position), the same for all four.

    torchrun --nproc_per_node 8 scripts/offset_feat_train.py \
        --model_path $E/checkpoint-1000 --where_head_pt $E/loc/where_lm8_ce20.pt \
        --train_json $OSS_DATA/extra_0916/viscot_bbox.train.json --held_from_train 3000 \
        --crop_tok 400 --n_win 1 --min_win_ratio 2 --steps 400 --save $E/loc/offset_feat.pt
"""
import argparse
import json
import math
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from offset_conv_train import ChannelNorm, load_rows, sign_p


class OffsetHead(nn.Module):
    def __init__(self, c, k=7, max_off=2.0):
        super().__init__()
        self.max_off = max_off
        self.dw = nn.Conv2d(c, c, k, padding=k // 2, groups=c)
        self.norm = ChannelNorm(c)
        self.act = nn.GELU()
        self.pw = nn.Conv2d(c, 2, 1, bias=False)
        nn.init.zeros_(self.pw.weight)

    def forward(self, x, gh, gw):
        """x (1, C, Hh, Wh) HR window tokens -> (1, 2, gh, gw) offsets in reference cells."""
        h = self.act(self.norm(self.dw(x)))
        h = F.adaptive_avg_pool2d(h, (gh, gw))
        return torch.tanh(self.pw(h)) * self.max_off


def ref_grid(gh, gw, dev):
    xs = (torch.arange(gw, device=dev, dtype=torch.float32) + 0.5) / gw * 2 - 1
    ys = (torch.arange(gh, device=dev, dtype=torch.float32) + 0.5) / gh * 2 - 1
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx, gy], -1)


def sample_tokens(src, off, sbox=(0.0, 0.0, 1.0, 1.0)):
    """src (1, C, Hs, Ws) feature map, sbox = the window inside src as [0, 1] fractions,
    off (1, 2, gh, gw) -> tokens (gh*gw, C), window-local points (gh, gw, 2) in [-1, 1]."""
    gh, gw = off.shape[-2:]
    shift = torch.stack([2 * off[0, 0] / gw, 2 * off[0, 1] / gh], -1)
    P = (ref_grid(gh, gw, src.device) + shift).clamp(-1, 1)
    x0, y0, x1, y1 = sbox
    G = torch.stack([(x0 + (P[..., 0] + 1) / 2 * (x1 - x0)) * 2 - 1,
                     (y0 + (P[..., 1] + 1) / 2 * (y1 - y0)) * 2 - 1], -1)
    z = F.grid_sample(src, G[None], mode="bilinear", padding_mode="border", align_corners=False)
    return z[0].flatten(1).t(), P


def global_positions(pos3d, blocks, wins, pts):
    """M-RoPE (t, h, w) of each window's tokens from where its points sit in the full image:
    h / w = start + the point's coordinate on a full-image grid at that window's reference
    spacing (zero offsets on a window at the image origin = the stock grid). Later positions
    shift so text keeps following the image's largest position."""
    pos = pos3d.clone().float()
    for (a, b), w, P in zip(blocks, wins, pts):
        s = float(pos[0, 0, a])
        px0, py0, px1, py1 = w["px"]
        P = P.detach().float()
        xg = px0 / (px1 - px0) * w["gw"] + (P[..., 0] + 1) / 2 * w["gw"] - 0.5
        yg = py0 / (py1 - py0) * w["gh"] + (P[..., 1] + 1) / 2 * w["gh"] - 0.5
        old_max = float(pos[:, 0, a:b].max())
        pos[1, 0, a:b] = s + yg.flatten()
        pos[2, 0, a:b] = s + xg.flatten()
        pos[:, 0, b:] += float(pos[:, 0, a:b].max()) - old_max
    return pos


def nms_peaks(logits, k, dist_cells, tau):
    """Up to k (row, col) peaks of the head map, >= dist_cells apart (Chebyshev); a later
    peak needs >= tau x the top cell's probability."""
    p = torch.softmax(logits.float().reshape(-1), -1).reshape(logits.shape)
    H, W = p.shape
    q = p.clone()
    out, top = [], float(p.max())
    for _ in range(k):
        v, i = q.reshape(-1).max(0)
        if out and float(v) < tau * top:
            break
        cy, cx = divmod(int(i), W)
        out.append((cy, cx))
        q[max(0, cy - dist_cells + 1):cy + dist_cells, max(0, cx - dist_cells + 1):cx + dist_cells] = -1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--where_head_pt", required=True)
    ap.add_argument("--train_json", default=None)
    ap.add_argument("--held_from_train", type=int, default=3000)
    ap.add_argument("--image_root", default="~/sft_data/train_split")
    ap.add_argument("--max_train", type=int, default=20000)
    ap.add_argument("--eval_dataset", default=None, choices=["vstar", "hrbench4k", "hrbench8k"],
                    help="accuracy mode: crop / fpool / fzero (/ net with --load), psp-style json to --out")
    ap.add_argument("--load", default=None)
    ap.add_argument("--vstar_root", default="~/data/vstar_bench")
    ap.add_argument("--hrbench_parquet", default=None)
    ap.add_argument("--max_samples", type=int, default=1000)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--crop_tok", type=int, default=400, help="total sampled tokens, split over the windows")
    ap.add_argument("--grid", type=int, default=0, help=">0: N x N tokens per window instead of the budget split")
    ap.add_argument("--n_win", type=int, default=4)
    ap.add_argument("--peak_dist", type=int, default=4, help="NMS distance between window centres, LR cells")
    ap.add_argument("--peak_tau", type=float, default=0.0, help="window k>1 needs >= tau x top cell probability")
    ap.add_argument("--hr_source", choices=["full", "window"], default="full",
                    help="full: one ViT pass on the whole image (<= --hr_full_cap tokens), windows cut from that "
                         "map; window: each window's pixels through the ViT on their own (<= --hr_cap tokens)")
    ap.add_argument("--hr_full_cap", type=int, default=16384)
    ap.add_argument("--hr_cap", type=int, default=4096, help="max HR tokens per window for --hr_source window")
    ap.add_argument("--pe", choices=["global", "local"], default="global",
                    help="global: window tokens' M-RoPE h/w from their sample points in full-image coordinates; "
                         "local: each window its own stock grid")
    ap.add_argument("--min_win_ratio", type=float, default=0.0,
                    help="skip samples whose window holds < ratio x its sampled tokens natively")
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--max_off", type=float, default=2.0)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr_tok", type=int, default=640)
    ap.add_argument("--max_ans_tok", type=int, default=32)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()
    if not args.eval_dataset and not args.train_json:
        raise SystemExit("training needs --train_json (or --eval_dataset)")

    ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if ddp:
        dist.init_process_group("nccl")
    rank = dist.get_rank() if ddp else 0
    world = dist.get_world_size() if ddp else 1
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    torch.manual_seed(args.seed + rank)

    def say(*a):
        if rank == 0:
            print(*a, flush=True)

    from transformers import AutoProcessor
    import llava.model.language_model.modeling_qwen3_5_dat as dat_mod
    from patch_sample_probe import crop_geometry

    model = dat_mod.Qwen3_5DATForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map={"": local}, attn_implementation=args.attn).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    sd = torch.load(os.path.expanduser(args.where_head_pt), map_location="cpu")
    sd = {k[len("where_head."):] if k.startswith("where_head.") else k: v for k, v in sd.items()}
    model.where_head.load_state_dict(sd)
    processor = AutoProcessor.from_pretrained(args.model_path, min_pixels=1024, max_pixels=args.lr_tok * 1024)
    hr_processor = AutoProcessor.from_pretrained(args.model_path, min_pixels=1024, max_pixels=100_000_000)
    tok = hr_processor.tokenizer
    ip = hr_processor.image_processor
    ps, tps, ms = ip.patch_size, ip.temporal_patch_size, ip.merge_size
    mean = torch.tensor(ip.image_mean, device=dev).view(3, 1, 1)
    std = torch.tensor(ip.image_std, device=dev).view(3, 1, 1)
    image_id = model.config.image_token_id
    C = model.config.text_config.hidden_size
    embed = model.get_input_embeddings()

    head = OffsetHead(C, k=args.k, max_off=args.max_off).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.0)
    skips = {"small": 0, "other": 0}
    say(f"[oft] offset head {sum(p.numel() for p in head.parameters()) / 1e6:.2f}M params, world {world}, "
        f"{args.n_win} window(s), {args.crop_tok} tokens")

    def patchify(x):
        _, H, W = x.shape
        x = ((x - mean) / std)[None].repeat(tps, 1, 1, 1)
        gh, gw = H // ps, W // ps
        x = x.reshape(1, tps, 3, gh // ms, ms, ps, gw // ms, ms, ps).permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        return x.reshape(gh * gw, 3 * tps * ps * ps)

    def vit_pv(pv, thw):
        e = model.model.visual(pv.to(torch.bfloat16), grid_thw=thw, return_dict=True).pooler_output
        return torch.cat(list(e), 0) if isinstance(e, (list, tuple)) else e

    def vit_image(pil):
        """PIL (sides multiples of 32) -> merged tokens (h, w, C)."""
        x = torch.from_numpy(np.asarray(pil, dtype=np.float32) / 255.0).permute(2, 0, 1).to(dev)
        h, w = x.shape[1] // ps, x.shape[2] // ps
        e = vit_pv(patchify(x), torch.tensor([[1, h, w]], device=dev))
        return e.reshape(h // ms, w // ms, -1)

    @torch.no_grad()
    def prepare(row):
        image = row["image"] if row.get("image") is not None else Image.open(row["path"])
        W, H = image.size
        per_win = args.grid * args.grid if args.grid else args.crop_tok // args.n_win
        if args.min_win_ratio and W * H * 0.25 / 1024 < args.min_win_ratio * per_win:
            skips["small"] += 1
            return None
        image = image.convert("RGB")
        thw = processor.image_processor(images=[image], return_tensors="pt")["image_grid_thw"][0]
        lr_img = image.resize((int(thw[2]) * 16, int(thw[1]) * 16), Image.BICUBIC)
        q = row["q"]
        msgs = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [{"type": "image", "image": lr_img}, {"type": "text", "text": q},
                                             {"type": "text", "text": dat_mod.WHERE_HINT}]}]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False)
        text = text.rstrip()
        if text.endswith("<|im_end|>"):
            text = text[:-len("<|im_end|>")]
        inp = processor(text=[text], images=[lr_img], return_tensors="pt")
        inp = {k: v.to(dev) for k, v in inp.items() if torch.is_tensor(v)}
        inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
        out = model(**inp, output_hidden_states=True, return_dict=True, use_cache=False)
        ids = inp["input_ids"][0]
        is_img = ids == image_id
        gh_l, gw_l = int(thw[1]) // 2, int(thw[2]) // 2
        hs = out.hidden_states[8][0]
        cells = hs[is_img]
        if cells.shape[0] != gh_l * gw_l:
            return None
        logits = model.where_head(cells.reshape(gh_l, gw_l, -1), hs[int(torch.nonzero(is_img)[-1]) + 1:])
        hd_w, hd_h = dat_mod._where_hd_size(image, thw, 3, 5017600)
        full = None
        if args.hr_source == "full":
            sc = math.sqrt(min(args.hr_full_cap * 1024, W * H) / float(W * H))
            Wf, Hf = max(32, int(W * sc) // 32 * 32), max(32, int(H * sc) // 32 * 32)
            full = vit_image(image.resize((Wf, Hf), Image.BICUBIC)).permute(2, 0, 1)[None].float()

        wins = []
        for cy, cx in nms_peaks(logits, args.n_win, args.peak_dist, args.peak_tau):
            box = ((cx + 0.5 - 2) / gw_l, (cy + 0.5 - 2) / gh_l, (cx + 0.5 + 2) / gw_l, (cy + 0.5 + 2) / gh_l)
            win = dat_mod._where_pad_window(box, 2.0, 20, hd_w, hd_h)
            px0, py0, px1, py1, gh, gw = crop_geometry(image, win, per_win)
            if args.grid:
                gh = gw = args.grid
            cw, ch = px1 - px0, py1 - py0
            if args.min_win_ratio and cw * ch / 1024 < args.min_win_ratio * gh * gw:
                skips["small"] += 1
                return None
            crop = image.crop((px0, py0, px1, py1))
            if full is not None:
                Hg, Wg = full.shape[-2:]
                ty0, tx0 = min(int(py0 / H * Hg), Hg - 1), min(int(px0 / W * Wg), Wg - 1)
                ty1, tx1 = max(ty0 + 1, math.ceil(py1 / H * Hg)), max(tx0 + 1, math.ceil(px1 / W * Wg))
                hr, src, sbox = full[..., ty0:ty1, tx0:tx1], full, (px0 / W, py0 / H, px1 / W, py1 / H)
            else:
                sc = math.sqrt(min(args.hr_cap * 1024, cw * ch) / float(cw * ch))
                hw_, hh_ = max(32, int(cw * sc) // 32 * 32), max(32, int(ch * sc) // 32 * 32)
                hr = vit_image(crop.resize((hw_, hh_), Image.BICUBIC)).permute(2, 0, 1)[None].float()
                src, sbox = hr, (0.0, 0.0, 1.0, 1.0)
            lo = vit_image(crop.resize((gw * 32, gh * 32), Image.BICUBIC)).reshape(gh * gw, -1)
            wins.append(dict(px=(px0, py0, px1, py1), gh=gh, gw=gw, hr=hr, src=src, sbox=sbox, crop=lo))

        dummies = [Image.new("RGB", (w["gw"] * 32, w["gh"] * 32)) for w in wins]
        qd = {"type": "text", "text": q}
        content = [{"type": "image", "image": lr_img}, qd, {"type": "text", "text": dat_mod.WHERE_HINT}] + \
            [{"type": "image", "image": d} for d in dummies] + [dict(qd)]
        msgs = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": content}]
        text = hr_processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        enc = hr_processor(text=[text], images=[lr_img] + dummies, return_tensors="pt")
        thw2 = enc["image_grid_thw"].to(dev)
        for k, w in enumerate(wins):
            if tuple(thw2[1 + k].tolist()) != (1, 2 * w["gh"], 2 * w["gw"]):
                return None
        a_ids = tok.encode(row["a"], add_special_tokens=False)[: args.max_ans_tok] + \
            tok.encode("<|im_end|>", add_special_tokens=False) if row["a"] else []
        P = int(enc["input_ids"].shape[1])
        ids = torch.cat([enc["input_ids"][0], torch.tensor(a_ids, dtype=torch.long)]).to(dev)[None]
        mm = torch.zeros_like(ids)
        mm[ids == image_id] = 1
        pos3d, rope_deltas = model.model.get_rope_index(ids, mm, image_grid_thw=thw2)
        n_lr = int(torch.prod(thw2[0]))
        lr_emb = vit_pv(enc["pixel_values"][:n_lr].to(dev), thw2[:1])
        img_pos = torch.nonzero(ids[0] == image_id).flatten()
        if img_pos.numel() != lr_emb.shape[0] + sum(w["gh"] * w["gw"] for w in wins):
            return None
        blocks = dat_mod._where_token_runs(ids[0], image_id)[1:]
        if len(blocks) != len(wins) or any(b - a != w["gh"] * w["gw"] for (a, b), w in zip(blocks, wins)):
            return None
        return dict(wins=wins, ids=ids, mm=mm, pos3d=pos3d, rope_deltas=rope_deltas, P=P, blocks=blocks,
                    emb=embed(ids), lr_emb=lr_emb, img_pos=img_pos, W=W, H=H, box=row.get("box"))

    def run_lm(pr, feats, pts, prompt_only=False):
        e = pr["emb"].clone()
        e[0, pr["img_pos"]] = torch.cat([pr["lr_emb"]] + list(feats), 0).to(e.dtype)
        n = pr["P"] if prompt_only else e.shape[1]
        pos = global_positions(pr["pos3d"], pr["blocks"], pr["wins"], pts) if args.pe == "global" else pr["pos3d"]
        model.model.rope_deltas = pr["rope_deltas"]
        o = model(inputs_embeds=e[:, :n], position_ids=pos[..., :n], mm_token_type_ids=pr["mm"][:, :n],
                  use_cache=False, return_dict=True, logits_to_keep=1 if prompt_only else 0)
        return o.logits

    def answer_loss(pr, feats, pts):
        lg = run_lm(pr, feats, pts)[0, pr["P"] - 1:-1].float()
        return F.cross_entropy(lg, pr["ids"][0, pr["P"]:])

    def feats_for(pr, cfg):
        """-> (list of (gh*gw, C) token blocks, list of window-local points, mean |off|).
        crop / fpool sit on the reference grid."""
        if cfg in ("crop", "fpool"):
            pts = [ref_grid(w["gh"], w["gw"], dev) for w in pr["wins"]]
            if cfg == "crop":
                return [w["crop"] for w in pr["wins"]], pts, 0.0
            return [F.adaptive_avg_pool2d(w["hr"], (w["gh"], w["gw"]))[0].flatten(1).t() for w in pr["wins"]], pts, 0.0
        zs, ps_, mo = [], [], 0.0
        for w in pr["wins"]:
            if cfg == "net":
                off = head(w["hr"], w["gh"], w["gw"])
            else:
                off = torch.zeros(1, 2, w["gh"], w["gw"], device=dev)
            z, P = sample_tokens(w["src"], off, w["sbox"])
            zs.append(z)
            ps_.append(P)
            mo += float(off.detach().abs().mean()) / len(pr["wins"])
        return zs, ps_, mo

    def inbox(pr, pts):
        if pr["box"] is None or pts is None:
            return 0.0
        b, W, H = pr["box"], pr["W"], pr["H"]
        hit = tot = 0.0
        for w, P in zip(pr["wins"], pts):
            px0, py0, px1, py1 = w["px"]
            u = px0 + (P[..., 0] + 1) / 2 * (px1 - px0)
            v = py0 + (P[..., 1] + 1) / 2 * (py1 - py0)
            m = (u >= b[0] * W) & (u <= b[2] * W) & (v >= b[1] * H) & (v <= b[3] * H)
            hit += float(m.float().sum())
            tot += m.numel()
        return hit / max(tot, 1.0)

    # ───────────────────────── accuracy mode ─────────────────────────
    if args.eval_dataset:
        from probe_whd_qwen35 import load_samples
        if args.load:
            head.load_state_dict(torch.load(os.path.expanduser(args.load), map_location="cpu"))
        head.eval()
        samples = load_samples(argparse.Namespace(dataset=args.eval_dataset, vstar_root=args.vstar_root,
                                                  hrbench_parquet=args.hrbench_parquet, image_folder=None,
                                                  max_samples=args.max_samples))
        mine = [(i, s) for i, s in enumerate(samples) if i % args.shards == args.shard]
        cfgs = ["crop", "fpool", "fzero"] + (["net"] if args.load else [])
        out = {c: [] for c in cfgs}
        print(f"[oft-eval] {args.eval_dataset} shard {args.shard}/{args.shards}: {len(mine)} samples, configs {cfgs}",
              flush=True)
        for gi, s in mine:
            pr = prepare({"image": s["image"], "q": s["prompt"], "a": "", "box": None})
            if pr is None:
                print(f"[oft-eval] {gi}: skipped", flush=True)
                continue
            letters = re.findall(r"^([A-Z])\. ", s["prompt"], re.M) or list("ABCD")
            lid = torch.tensor([tok.encode(L, add_special_tokens=False)[0] for L in letters], device=dev)
            line = []
            with torch.no_grad():
                for c in cfgs:
                    feats, pts, mo = feats_for(pr, c)
                    lg = run_lm(pr, feats, pts, prompt_only=True)[0, -1].float()
                    pred = letters[int(lg[lid].argmax())]
                    ok = pred == s["gt"]
                    out[c].append({"i": gi, "pred": pred, "gt": s["gt"], "ok": bool(ok), "cat": s["category"],
                                   "prompt": s["prompt"], "window": [list(w["px"]) for w in pr["wins"]],
                                   "n_tokens": int(pr["P"]), "stats": {"off": mo}})
                    line.append(f"{c}={pred}{'+' if ok else '-'}")
            print(f"[oft-eval] {gi} gt={s['gt']} " + " ".join(line), flush=True)
        if args.out:
            os.makedirs(os.path.dirname(os.path.expanduser(args.out)) or ".", exist_ok=True)
            json.dump(out, open(os.path.expanduser(args.out), "w"))
        for c in cfgs:
            print(f"[oft-eval] {c}: {sum(r['ok'] for r in out[c])}/{len(out[c])}", flush=True)
        return

    # ───────────────────────── training ─────────────────────────
    rows = load_rows(args.train_json, args.image_root, args.max_train + args.held_from_train, args.seed)
    train, held = rows[:-args.held_from_train], rows[-args.held_from_train:]
    say(f"[oft] train {len(train)}  held-out {len(held)}")
    mine_held = held[rank::world]
    cache = {}

    def evaluate(step):
        head.eval()
        # n, L crop, L pool, L zero, L net, net<zero, net>zero, net<crop, net>crop, ib zero, ib net, |off|
        acc = torch.zeros(12, device=dev)
        for hi, row in enumerate(mine_held):
            try:
                pr = prepare(row)
            except Exception:
                pr = None
            if pr is None:
                continue
            with torch.no_grad():
                if hi not in cache:
                    lc = float(answer_loss(pr, *feats_for(pr, "crop")[:2]))
                    lp = float(answer_loss(pr, *feats_for(pr, "fpool")[:2]))
                    fz, pz, _ = feats_for(pr, "fzero")
                    cache[hi] = (lc, lp, float(answer_loss(pr, fz, pz)), inbox(pr, pz))
                lc, lp, lz, ibz = cache[hi]
                fn, pn, mo = feats_for(pr, "net")
                ln = float(answer_loss(pr, fn, pn))
            acc += torch.tensor([1, lc, lp, lz, ln, ln < lz - 1e-4, ln > lz + 1e-4, ln < lc - 1e-4, ln > lc + 1e-4,
                                 ibz, inbox(pr, pn), mo], device=dev)
        if ddp:
            dist.all_reduce(acc)
        n = max(float(acc[0]), 1.0)
        a = [int(acc[i]) for i in (5, 6, 7, 8)]
        say(f"[oft] EVAL step {step}: n={int(acc[0])}  loss crop {acc[1] / n:.4f}  fpool {acc[2] / n:.4f}  "
            f"fzero {acc[3] / n:.4f}  net {acc[4] / n:.4f}  net vs fzero {a[0]}/{a[1]} p={sign_p(a[0], a[1]):.3f}  "
            f"net vs crop {a[2]}/{a[3]} p={sign_p(a[2], a[3]):.3f}  inbox fzero {acc[9] / n:.3f} net {acc[10] / n:.3f}  "
            f"|off| {acc[11] / n:.3f} cells")
        head.train()

    evaluate(0)
    rng = random.Random(args.seed + 17 * rank)
    t0 = time.time()
    run = torch.zeros(3, device=dev)
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        n_ok = 0
        for _ in range(args.accum):
            pr = None
            for _try in range(40):
                row = train[rng.randrange(len(train))]
                try:
                    pr = prepare(row)
                except Exception:
                    skips["other"] += 1
                    pr = None
                if pr is not None:
                    break
            if pr is None:
                continue
            loss = answer_loss(pr, *feats_for(pr, "net")[:2])
            (loss / args.accum).backward()
            with torch.no_grad():
                l0 = answer_loss(pr, *feats_for(pr, "fzero")[:2])
            run += torch.tensor([1.0, float(loss), float(l0)], device=dev)
            n_ok += 1
        cnt = torch.tensor([float(n_ok)], device=dev)
        if ddp:
            for p in head.parameters():
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                dist.all_reduce(p.grad)
            dist.all_reduce(cnt)
        if float(cnt) > 0:
            for p in head.parameters():
                if p.grad is not None:
                    p.grad /= max(float(cnt) / args.accum, 1.0)
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
        if step % 10 == 0:
            r = run.clone()
            if ddp:
                dist.all_reduce(r)
            n = max(float(r[0]), 1.0)
            say(f"[oft] step {step}: train loss net {r[1] / n:.4f} fzero {r[2] / n:.4f} (delta {(r[1] - r[2]) / n:+.4f})  "
                f"{(time.time() - t0) / step:.2f} s/step  skipped small-window {skips['small']} (rank 0)")
            run.zero_()
        if step % args.eval_every == 0 or step == args.steps:
            evaluate(step)
            if rank == 0 and args.save:
                torch.save(head.state_dict(), os.path.expanduser(args.save))
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
