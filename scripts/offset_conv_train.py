#!/usr/bin/env python3
"""Can a DAT conv offset network LEARN offsets from the answer loss alone?

Reference = the crop's uniform gh x gw grid in the head's live window. An
offset net (DAT's: depthwise conv -> GELU -> 1x1 conv) reads the LM layer-8
states of the LR cells under the window (resampled to gh x gw, FiLM'd by the
question), outputs (dx, dy) per reference point, tanh-bounded to --max_off
tokens, last conv zero-init (step 0 = the crop exactly). The 2nd image is
drawn with grid_sample (as in offset_grad_probe.py) and goes through the
frozen ViT + LM on the live single-pass prompt [LR, q, hint, img2, q]; loss =
CE over the GT answer tokens. Only the offset net trains.

The test is generalisation: per-sample gradients do lower the loss at 1-3 px
scale (offset_grad_probe --step_cells 0.1) without moving toward the target,
which is what an adversarial perturbation looks like. On held-out samples,
every --eval_every steps, three offset fields are scored on the same metric:
  zero  the crop (reference grid)
  net   the learned offsets
  dat   the current analytic DAT warp (dw0.5: inverse-CDF of the head map
        inside the same window, ratio 1.5), as the bar to beat
Reported: mean answer loss, paired wins/losses vs zero (sign test), the share
of img2 pixels from inside the GT box, mean |offset|. The train-side
delta vs zero is logged too: train down + held-out flat = not learnable.

    torchrun --nproc_per_node 8 scripts/offset_conv_train.py \
        --model_path $E/checkpoint-1000 --where_head_pt $E/loc/where_lm8_ce20.pt \
        --train_json $OSS_DATA/extra_0916/viscot_bbox.train.json \
        --held_json $OSS_DATA/extra_0916/viscot_bbox.heldout.json --image_root ~/sft_data/train_split \
        --steps 400 --accum 1 --save $E/loc/offset_conv.pt > $E/loc/offset_conv.log 2>&1

v2 (DAT's offset net + fixed square grid, only samples with room to gain):
--k 7 (dwconv 7x7 -> LayerNorm -> GELU -> 1x1 no bias; --d 0 = on all LM channels,
one offset group), --grid 10 (every window -> 10 x 10 = 100
tokens, anisotropic), --min_win_ratio 2 (window >= 200 native tokens, so moving points
can add information), --held_from_train 3000 (held-out rows disjoint from training).
--eval_dataset with the same --grid / --k scores crop / net / dat on HR-Bench.
"""
import argparse
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

def sign_p(w, l):
    n = w + l
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, j) for j in range(min(w, l) + 1)) / 2 ** n)


class ChannelNorm(nn.Module):
    """LayerNorm over the channel dim of (B, C, H, W) (DAT's LayerNormProxy)."""

    def __init__(self, d):
        super().__init__()
        self.ln = nn.LayerNorm(d)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class OffsetNet(nn.Module):
    def __init__(self, c_in, d=0, k=7, max_off=2.0):
        """d <= 0: no channel reduction, the conv runs on all c_in channels (DAT with one
        offset group; the windows play the role of DAT's groups)."""
        super().__init__()
        self.max_off = max_off
        self.ln = nn.LayerNorm(c_in)
        if d <= 0 or d == c_in:
            d = c_in
            self.proj = nn.Identity()
        else:
            self.proj = nn.Linear(c_in, d)
        self.ln_t = nn.LayerNorm(c_in)
        self.film = nn.Linear(c_in, 2 * d)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.dw = nn.Conv2d(d, d, k, padding=k // 2, groups=d)
        self.norm = ChannelNorm(d)
        self.act = nn.GELU()
        self.pw = nn.Conv2d(d, 2, 1, bias=False)
        nn.init.zeros_(self.pw.weight)

    def forward(self, cells, txt, nwin, gh, gw):
        """cells (h, w, C) LR states, txt (T, C), nwin = window in [0,1] -> (1, 2, gh, gw) in tokens."""
        x = self.proj(self.ln(cells.float())).permute(2, 0, 1)[None]          # (1, d, h, w)
        xs = nwin[0] + (torch.arange(gw, device=x.device) + 0.5) / gw * (nwin[2] - nwin[0])
        ys = nwin[1] + (torch.arange(gh, device=x.device) + 0.5) / gh * (nwin[3] - nwin[1])
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([2 * gx - 1, 2 * gy - 1], -1)[None].float()
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
        g, b = self.film(self.ln_t(txt.float().mean(0))).chunk(2)
        x = x * (1 + g[None, :, None, None]) + b[None, :, None, None]
        return torch.tanh(self.pw(self.act(self.norm(self.dw(x))))) * self.max_off


def draw(geo, off=None, edges=None):
    """-> (1, 3, Ho, Wo) image, source x / y in ORIGINAL px. off (1, 2, gh, gw) tokens, or
    edges = (xe, ye) cell edges in original px (piecewise-linear, the analytic DAT)."""
    Ho, Wo, sx, sy = geo["Ho"], geo["Wo"], geo["sx"], geo["sy"]
    if edges is None:
        xs = geo["px0"] * sx + torch.arange(Wo, device=geo["img_t"].device, dtype=torch.float32) + 0.5
        ys = geo["py0"] * sy + torch.arange(Ho, device=geo["img_t"].device, dtype=torch.float32) + 0.5
        by, bx = torch.meshgrid(ys, xs, indexing="ij")
        up = F.interpolate(off, size=(Ho, Wo), mode="bilinear", align_corners=False)[0] * 32.0
        srcx, srcy = bx + up[0], by + up[1]
    else:
        def axis(e, n_out):
            e = torch.tensor(e, device=geo["img_t"].device, dtype=torch.float32)
            n = e.numel() - 1
            t = (torch.arange(n_out, device=geo["img_t"].device, dtype=torch.float32) + 0.5) / n_out * n
            j = t.floor().clamp(max=n - 1).long()
            return e[j] + (t - j) * (e[j + 1] - e[j])
        sxo = axis(edges[0], Wo) * sx
        syo = axis(edges[1], Ho) * sy
        srcy, srcx = torch.meshgrid(syo, sxo, indexing="ij")
    grid = torch.stack([2 * srcx / geo["Wr"] - 1, 2 * srcy / geo["Hr"] - 1], -1)[None]
    img = F.grid_sample(geo["img_t"][None], grid, mode="bilinear", padding_mode="border", align_corners=False)
    return img, srcx / sx, srcy / sy


def inbox(geo, srcx, srcy, box):
    W, H = geo["W"], geo["H"]
    m = (srcx >= box[0] * W) & (srcx <= box[2] * W) & (srcy >= box[1] * H) & (srcy <= box[3] * H)
    return float(m.float().mean())


def load_rows(path, image_root, cap, seed):
    from where_head import question_of
    data = json.load(open(os.path.expanduser(path)))
    random.Random(seed).shuffle(data)
    rows = []
    for s in data:
        b = s.get("bbox")
        conv = s.get("conversations") or []
        if not (isinstance(b, (list, tuple)) and len(b) == 4) or len(conv) < 2:
            continue
        b = [float(v) for v in b]
        if not (b[2] > b[0] and b[3] > b[1] and max(b) <= 1.0):
            continue
        q, a = question_of(s), str(conv[1].get("value", "")).strip()
        im = s.get("image")
        im = im[0] if isinstance(im, list) else im
        if not q or not a or not im:
            continue
        rows.append({"path": im if os.path.isabs(im) else os.path.join(os.path.expanduser(image_root), im),
                     "q": q, "a": a, "box": b})
        if len(rows) >= cap:
            break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--where_head_pt", required=True)
    ap.add_argument("--train_json", default=None)
    ap.add_argument("--held_json", default=None)
    ap.add_argument("--eval_dataset", default=None, choices=["vstar", "hrbench4k", "hrbench8k"],
                    help="accuracy mode (no training): crop (zero offsets) / net (--load) / dat (dw0.5), "
                         "same window and pipeline; writes a patch_sample_probe-style json (--out)")
    ap.add_argument("--load", default=None, help="offset net state_dict for --eval_dataset")
    ap.add_argument("--vstar_root", default="~/data/vstar_bench")
    ap.add_argument("--hrbench_parquet", default=None)
    ap.add_argument("--max_samples", type=int, default=1000)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--image_root", default="~/sft_data/train_split")
    ap.add_argument("--max_train", type=int, default=20000)
    ap.add_argument("--n_held", type=int, default=320)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--accum", type=int, default=1, help="samples per rank per step")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max_off", type=float, default=2.0, help="offset bound, in tokens")
    ap.add_argument("--d", type=int, default=0, help="conv channels; 0 = all LM channels, no reduction")
    ap.add_argument("--k", type=int, default=7, help="depthwise conv kernel")
    ap.add_argument("--grid", type=int, default=0,
                    help=">0: fixed N x N reference grid (N*N tokens) on the window, any aspect; 0: crop's grid")
    ap.add_argument("--min_win_ratio", type=float, default=0.0,
                    help="skip samples whose window holds < ratio x (grid tokens) native tokens (no room to gain)")
    ap.add_argument("--held_from_train", type=int, default=0,
                    help=">0: held-out = the last N rows of --train_json (disjoint from training), not --held_json")
    ap.add_argument("--crop_tok", type=int, default=400)
    ap.add_argument("--lr_tok", type=int, default=640)
    ap.add_argument("--max_ans_tok", type=int, default=32)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    if not args.eval_dataset and not (args.train_json and (args.held_json or args.held_from_train)):
        raise SystemExit("training needs --train_json and --held_json / --held_from_train (or --eval_dataset)")
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
    from patch_sample_probe import crop_geometry, head_map, inv_cdf_edges, window_marginals

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

    net = OffsetNet(C, d=args.d, k=args.k, max_off=args.max_off).to(dev)
    skips = {"small": 0, "other": 0}
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.0)
    say(f"[oct] offset net {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M params, world {world}")

    def patchify(x):
        _, H, W = x.shape
        x = ((x - mean) / std)[None].repeat(tps, 1, 1, 1)
        gh, gw = H // ps, W // ps
        x = x.reshape(1, tps, 3, gh // ms, ms, ps, gw // ms, ms, ps).permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        return x.reshape(gh * gw, 3 * tps * ps * ps)

    @torch.no_grad()
    def prepare(row):
        """Head prefill on the LR view -> window, LR states, answer-prompt tensors. None on skip."""
        image = row["image"] if row.get("image") is not None else Image.open(row["path"])
        W, H = image.size
        n_pts = args.grid * args.grid if args.grid else args.crop_tok
        if args.min_win_ratio and W * H * 0.25 / 1024 < args.min_win_ratio * n_pts:
            skips["small"] += 1                        # even a generous window is too small
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
        cells = cells.reshape(gh_l, gw_l, -1)
        txt = hs[int(torch.nonzero(is_img)[-1]) + 1:]
        logits = model.where_head(cells, txt)
        box = dat_mod._where_top_box(logits, 4.0)
        hd_w, hd_h = dat_mod._where_hd_size(image, thw, 3, 5017600)
        win = dat_mod._where_pad_window(box, 2.0, 20, hd_w, hd_h)
        px0, py0, px1, py1, gh, gw = crop_geometry(image, win, args.crop_tok)
        if args.grid:
            gh = gw = args.grid
        if args.min_win_ratio and (px1 - px0) * (py1 - py0) / 1024 < args.min_win_ratio * gh * gw:
            skips["small"] += 1
            return None
        Ho, Wo = gh * 32, gw * 32
        sx, sy = Wo / (px1 - px0), Ho / (py1 - py0)
        Wr, Hr = max(1, round(W * sx)), max(1, round(H * sy))
        import numpy as np
        img_t = torch.from_numpy(np.asarray(image.resize((Wr, Hr), Image.BICUBIC), dtype="float32") / 255.0
                                 ).permute(2, 0, 1).to(dev)
        geo = dict(px0=px0, py0=py0, px1=px1, py1=py1, gh=gh, gw=gw, Ho=Ho, Wo=Wo, sx=sx, sy=sy,
                   Wr=Wr, Hr=Hr, W=W, H=H, img_t=img_t)
        # answer prompt, built on the zero-offset image (fixes ids / grid)
        img0 = draw(geo, torch.zeros(1, 2, gh, gw, device=dev))[0]
        pil0 = Image.fromarray((img0[0].clamp(0, 1).permute(1, 2, 0) * 255).round().byte().cpu().numpy())
        qd = {"type": "text", "text": q}
        content = [{"type": "image", "image": lr_img}, qd, {"type": "text", "text": dat_mod.WHERE_HINT},
                   {"type": "image", "image": pil0}, dict(qd)]
        msgs = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": content}]
        text = hr_processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        enc = hr_processor(text=[text], images=[lr_img, pil0], return_tensors="pt")
        a_ids = tok.encode(row["a"], add_special_tokens=False)[: args.max_ans_tok] + \
            tok.encode("<|im_end|>", add_special_tokens=False)
        P = int(enc["input_ids"].shape[1])
        ids = torch.cat([enc["input_ids"][0], torch.tensor(a_ids)]).to(dev)[None]
        thw2 = enc["image_grid_thw"].to(dev)
        if tuple(thw2[1].tolist()) != (1, Ho // ps, Wo // ps):
            return None
        mm = torch.zeros_like(ids)
        mm[ids == image_id] = 1
        pos3d, rope_deltas = model.model.get_rope_index(ids, mm, image_grid_thw=thw2)
        pv = enc["pixel_values"].to(dev, torch.float32)
        n_lr = int(torch.prod(thw2[0]))
        nwin = (px0 / W, py0 / H, px1 / W, py1 / H)
        # analytic DAT (dw0.5) edges inside the same window
        dx, dy = window_marginals(head_map(logits.float().tolist()), nwin, 0.5, 1.5)
        xe = [px0 + e * (px1 - px0) for e in inv_cdf_edges(dx, gw)]
        ye = [py0 + e * (py1 - py0) for e in inv_cdf_edges(dy, gh)]
        return dict(geo=geo, cells=cells, txt=txt, nwin=nwin, ids=ids, mm=mm, pos3d=pos3d, rope_deltas=rope_deltas,
                    thw=thw2, pv_lr=pv[:n_lr], P=P, box=row["box"], xe=xe, ye=ye)

    def answer_loss(pr, img):
        pvals = torch.cat([pr["pv_lr"], patchify(img[0].clamp(0, 1))], 0).to(torch.bfloat16)
        model.model.rope_deltas = pr["rope_deltas"]
        o = model(input_ids=pr["ids"], position_ids=pr["pos3d"], mm_token_type_ids=pr["mm"], pixel_values=pvals,
                  image_grid_thw=pr["thw"], use_cache=False, return_dict=True)
        P = pr["P"]
        lg = o.logits[0, P - 1:-1].float()
        return F.cross_entropy(lg, pr["ids"][0, P:])

    if args.eval_dataset:
        from probe_whd_qwen35 import extract_letter, load_samples
        if args.load:
            net.load_state_dict(torch.load(os.path.expanduser(args.load), map_location="cpu"))
        net.eval()
        samples = load_samples(argparse.Namespace(dataset=args.eval_dataset, vstar_root=args.vstar_root,
                                                  hrbench_parquet=args.hrbench_parquet, image_folder=None,
                                                  max_samples=args.max_samples))
        mine = [(i, s) for i, s in enumerate(samples) if i % args.shards == args.shard]
        cfgs = ["crop", "net", "dat"] if args.load else ["crop", "dat"]
        out = {c: [] for c in cfgs}
        print(f"[oct-eval] {args.eval_dataset} shard {args.shard}/{args.shards}: {len(mine)} samples, "
              f"crop_tok {args.crop_tok}, configs {cfgs}", flush=True)

        @torch.no_grad()
        def generate(pr, img):
            pvals = torch.cat([pr["pv_lr"], patchify(img[0].clamp(0, 1))], 0).to(torch.bfloat16)
            ids = pr["ids"][:, :pr["P"]]
            model.model.rope_deltas = None
            g = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=pvals,
                               image_grid_thw=pr["thw"], max_new_tokens=8, do_sample=False)
            return tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True).strip()

        for gi, s in mine:
            pr = prepare({"image": s["image"], "q": s["prompt"], "a": "", "box": None})
            if pr is None:
                print(f"[oct-eval] {gi}: skipped", flush=True)
                continue
            g = pr["geo"]
            with torch.no_grad():
                off = net(pr["cells"], pr["txt"], pr["nwin"], g["gh"], g["gw"])
            imgs = {"crop": draw(g, torch.zeros_like(off))[0], "dat": draw(g, edges=(pr["xe"], pr["ye"]))[0]}
            if "net" in cfgs:
                imgs["net"] = draw(g, off)[0]
            line = []
            for c in cfgs:
                pred = extract_letter(generate(pr, imgs[c]))
                ok = pred == s["gt"]
                out[c].append({"i": gi, "pred": pred, "gt": s["gt"], "ok": bool(ok), "cat": s["category"],
                               "prompt": s["prompt"], "window": list(pr["nwin"]), "n_tokens": int(pr["P"]),
                               "stats": {}})
                line.append(f"{c}={pred}{'+' if ok else '-'}")
            print(f"[oct-eval] {gi} gt={s['gt']} " + " ".join(line) + f" |off| {float(off.abs().mean()):.2f}",
                  flush=True)
        if args.out:
            os.makedirs(os.path.dirname(os.path.expanduser(args.out)) or ".", exist_ok=True)
            json.dump(out, open(os.path.expanduser(args.out), "w"))
        for c in cfgs:
            print(f"[oct-eval] {c}: {sum(r['ok'] for r in out[c])}/{len(out[c])}", flush=True)
        return

    if args.held_from_train:
        rows = load_rows(args.train_json, args.image_root, args.max_train + args.held_from_train, args.seed)
        train, held = rows[:-args.held_from_train], rows[-args.held_from_train:]
    else:
        train = load_rows(args.train_json, args.image_root, args.max_train, args.seed)
        held = load_rows(args.held_json, args.image_root, args.n_held, args.seed)
    say(f"[oct] train {len(train)}  held-out {len(held)}")
    mine_held = held[rank::world]
    zero_cache = {}

    def evaluate(step):
        net.eval()
        acc = torch.zeros(12, device=dev)   # n, L0, Ln, Ld, win_n, lose_n, win_d, lose_d, ib0, ibn, ibd, |off|
        for hi, row in enumerate(mine_held):
            try:
                pr = prepare(row)
            except Exception as e:                      # unreadable image etc.
                pr = None
            if pr is None:
                continue
            with torch.no_grad():
                g = pr["geo"]
                off = net(pr["cells"], pr["txt"], pr["nwin"], g["gh"], g["gw"])
                if hi not in zero_cache:
                    img0, s0x, s0y = draw(g, torch.zeros_like(off))
                    imgd, sdx, sdy = draw(g, edges=(pr["xe"], pr["ye"]))
                    zero_cache[hi] = (float(answer_loss(pr, img0)), float(answer_loss(pr, imgd)),
                                      inbox(g, s0x, s0y, pr["box"]), inbox(g, sdx, sdy, pr["box"]))
                l0, ld, ib0, ibd = zero_cache[hi]
                imgn, snx, sny = draw(g, off)
                ln = float(answer_loss(pr, imgn))
                ibn = inbox(g, snx, sny, pr["box"])
            acc += torch.tensor([1, l0, ln, ld, ln < l0 - 1e-4, ln > l0 + 1e-4, ld < l0 - 1e-4, ld > l0 + 1e-4,
                                 ib0, ibn, ibd, float(off.abs().mean())], device=dev)
        if ddp:
            dist.all_reduce(acc)
        n = max(float(acc[0]), 1.0)
        w_n, l_n, w_d, l_d = (int(acc[i]) for i in (4, 5, 6, 7))
        say(f"[oct] EVAL step {step}: n={int(acc[0])}  loss zero {acc[1] / n:.4f}  net {acc[2] / n:.4f} "
            f"(vs zero {w_n}/{l_n} p={sign_p(w_n, l_n):.3f})  dat {acc[3] / n:.4f} (vs zero {w_d}/{l_d} p={sign_p(w_d, l_d):.3f})  "
            f"inbox zero {acc[8] / n:.3f} net {acc[9] / n:.3f} dat {acc[10] / n:.3f}  |off| {acc[11] / n:.3f} tok")
        net.train()

    evaluate(0)
    rng = random.Random(args.seed + 17 * rank)
    t0 = time.time()
    run = torch.zeros(3, device=dev)        # n, loss net, loss zero
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        n_ok = 0
        for _ in range(args.accum):
            pr = None
            for _try in range(40):                     # --min_win_ratio rejects most small images
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
            g = pr["geo"]
            off = net(pr["cells"], pr["txt"], pr["nwin"], g["gh"], g["gw"])
            img, _, _ = draw(g, off)
            loss = answer_loss(pr, img)
            (loss / args.accum).backward()
            with torch.no_grad():
                l0 = answer_loss(pr, draw(g, torch.zeros_like(off))[0])
            run += torch.tensor([1.0, float(loss), float(l0)], device=dev)
            n_ok += 1
        cnt = torch.tensor([float(n_ok)], device=dev)
        if ddp:
            for p in net.parameters():
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                dist.all_reduce(p.grad)
            dist.all_reduce(cnt)
        if float(cnt) > 0:
            for p in net.parameters():
                p.grad /= max(float(cnt) / args.accum, 1.0)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        if step % 10 == 0:
            r = run.clone()
            if ddp:
                dist.all_reduce(r)
            n = max(float(r[0]), 1.0)
            say(f"[oct] step {step}: train loss net {r[1] / n:.4f} zero {r[2] / n:.4f} (delta {(r[1] - r[2]) / n:+.4f})  "
                f"{(time.time() - t0) / step:.2f} s/step  skipped small-window {skips['small']} (rank 0)")
            run.zero_()
        if step % args.eval_every == 0 or step == args.steps:
            evaluate(step)
            if rank == 0 and args.save:
                torch.save(net.state_dict(), os.path.expanduser(args.save))
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
