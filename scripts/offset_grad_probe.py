#!/usr/bin/env python3
"""Would LEARNED DAT offsets get a useful signal? Gradient of the answer loss
w.r.t. the 2nd image's sampling positions, through the frozen ViT + LM (V*).

Per sample: the live window (real head) gives the crop's gh x gw grid. The
2nd image is drawn with grid_sample from the image (pre-resized by the crop's
scale, so the base grid IS the crop up to the resampling kernel), with a
per-token offset field (2, gh, gw) in token units, bilinearly upsampled to
pixels, initialised at 0. The answer prompt is the live single-pass prompt
[LR, q, hint, 2nd image, q]; loss = CE over the option letters at the first
answer token. Backprop -> d loss / d offset. Reported per sample:

  cos    |g|-weighted mean cosine between -grad and the direction from the
         token's sampling point to the GT box centre (tokens outside the box);
         > 0 = descending the loss pulls samples toward the target
  inbox  fraction of the 2nd image's pixels drawn from inside the GT box,
         at offset 0 and after one step of max --step_cells token
  dloss  loss change after that step vs after a random step of the same norm

    for i in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$i setsid nohup python scripts/offset_grad_probe.py \
        --model_path $E/checkpoint-1000 --where_head_pt $E/loc/where_lm8_ce20.pt --n 96 --shard $i --shards 8 \
        --out $E/loc/ogp/ogp_$i.json > $E/loc/ogp/ogp_$i.log 2>&1 < /dev/null & disown; done
    python scripts/offset_grad_probe.py --summarize "$E/loc/ogp/ogp_*.json"
"""
import argparse
import glob
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def summarize(pattern):
    rows = []
    for f in sorted(glob.glob(os.path.expanduser(pattern))):
        rows.extend(json.load(open(f)))
    if not rows:
        raise SystemExit(f"no json matched {pattern}")
    print(f"{len(rows)} samples; patchify check max |diff| {max(r['patch_diff'] for r in rows):.4f}")

    def line(name, rs):
        if not rs:
            return
        n = len(rs)
        cos = [r["cos"] for r in rs if r["cos"] is not None]
        print(f"  {name:14s} n={n:3d}  cos>0 {sum(c > 0 for c in cos):3d}/{len(cos):<3d} mean cos {sum(cos) / max(len(cos), 1):+.3f}  "
              f"inbox {sum(r['inbox0'] for r in rs) / n:.3f} -> {sum(r['inbox1'] for r in rs) / n:.3f}  "
              f"(up in {sum(r['inbox1'] > r['inbox0'] + 1e-4 for r in rs)})  "
              f"dloss grad {sum(r['dloss_grad'] for r in rs) / n:+.4f} rand {sum(r['dloss_rand'] for r in rs) / n:+.4f}  "
              f"loss0 {sum(r['loss0'] for r in rs) / n:.3f}")
    line("all", rows)
    line("answer right", [r for r in rows if r["ok0"]])
    line("answer wrong", [r for r in rows if not r["ok0"]])
    for c in sorted({r["cat"] for r in rows}):
        line(c, [r for r in rows if r["cat"] == c])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summarize", default=None)
    ap.add_argument("--model_path")
    ap.add_argument("--where_head_pt", default=None)
    ap.add_argument("--vstar_root", default="~/data/vstar_bench")
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--crop_tok", type=int, default=400)
    ap.add_argument("--lr_tok", type=int, default=640)
    ap.add_argument("--step_cells", type=float, default=0.5, help="max offset of the one test step, in tokens")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.summarize:
        summarize(args.summarize)
        return

    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import AutoProcessor
    from probe_whd_qwen35 import load_samples
    from patch_sample_probe import crop_geometry, gt_box
    import llava.model.language_model.modeling_qwen3_5_dat as dat_mod

    torch.manual_seed(args.seed)
    dev = "cuda"
    model = dat_mod.Qwen3_5DATForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map={"": 0}, attn_implementation=args.attn).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if args.where_head_pt:
        sd = torch.load(os.path.expanduser(args.where_head_pt), map_location="cpu")
        sd = {k[len("where_head."):] if k.startswith("where_head.") else k: v for k, v in sd.items()}
        model.where_head.load_state_dict(sd)
    processor = AutoProcessor.from_pretrained(args.model_path, min_pixels=1024, max_pixels=args.lr_tok * 1024)
    hr_processor = AutoProcessor.from_pretrained(args.model_path, min_pixels=1024, max_pixels=100_000_000)
    ip = hr_processor.image_processor
    ps, tps, ms = ip.patch_size, ip.temporal_patch_size, ip.merge_size
    mean = torch.tensor(ip.image_mean, device=dev).view(3, 1, 1)
    std = torch.tensor(ip.image_std, device=dev).view(3, 1, 1)
    tok = hr_processor.tokenizer

    def patchify(x):
        """(3, H, W) in [0, 1] -> Qwen2-VL pixel_values rows (differentiable)."""
        _, H, W = x.shape
        x = ((x - mean) / std)[None].repeat(tps, 1, 1, 1)
        gh, gw = H // ps, W // ps
        x = x.reshape(1, tps, 3, gh // ms, ms, ps, gw // ms, ms, ps)
        x = x.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        return x.reshape(gh * gw, 3 * tps * ps * ps)

    samples = load_samples(argparse.Namespace(dataset="vstar", vstar_root=args.vstar_root,
                                              hrbench_parquet=None, image_folder=None, max_samples=300))
    mine = [(i, s) for i, s in enumerate(samples) if s.get("bboxes") and i % args.shards == args.shard]
    mine = mine[: max(1, args.n // args.shards)]
    print(f"[ogp] shard {args.shard}/{args.shards}: {len(mine)} samples", flush=True)

    rows = []
    for gi, s in mine:
        image, prompt = s["image"], s["prompt"]
        W, H = image.size
        rec = model.where_crop_generate(image, prompt, processor, hr_processor, where_question=prompt,
                                        crop_tok=args.crop_tok, head_prefix="train", single_pass=True,
                                        max_new_tokens=1)
        win = rec["window"]
        box = gt_box(s)
        px0, py0, px1, py1, gh, gw = crop_geometry(image, win, args.crop_tok)
        Ho, Wo = gh * 32, gw * 32
        sx, sy = Wo / (px1 - px0), Ho / (py1 - py0)
        Wr, Hr = max(1, round(W * sx)), max(1, round(H * sy))
        img_t = torch.from_numpy(__import__("numpy").asarray(
            image.resize((Wr, Hr), Image.BICUBIC), dtype="float32") / 255.0).permute(2, 0, 1).to(dev)
        # base grid in resized pixels: output pixel centre xo + 0.5 <- px0 * sx + xo + 0.5
        xs = px0 * sx + torch.arange(Wo, device=dev, dtype=torch.float32) + 0.5
        ys = py0 * sy + torch.arange(Ho, device=dev, dtype=torch.float32) + 0.5
        base_y, base_x = torch.meshgrid(ys, xs, indexing="ij")

        def draw(off):
            """off (1, 2, gh, gw) in token units -> (1, 3, Ho, Wo), source px (Ho, Wo) x2."""
            up = F.interpolate(off, size=(Ho, Wo), mode="bilinear", align_corners=False)[0] * 32.0
            sxp, syp = base_x + up[0], base_y + up[1]
            grid = torch.stack([2 * sxp / Wr - 1, 2 * syp / Hr - 1], -1)[None]
            img = F.grid_sample(img_t[None], grid, mode="bilinear", padding_mode="border", align_corners=False)
            return img, sxp / sx, syp / sy                     # source in ORIGINAL pixels

        def inbox(srcx, srcy):
            m = (srcx >= box[0] * W) & (srcx <= box[2] * W) & (srcy >= box[1] * H) & (srcy <= box[3] * H)
            return float(m.float().mean())

        off0 = torch.zeros(1, 2, gh, gw, device=dev)
        img0, sx0, sy0 = draw(off0)
        pil0 = Image.fromarray((img0[0].clamp(0, 1).permute(1, 2, 0) * 255).round().byte().cpu().numpy())
        thw_lr = processor.image_processor(images=[image], return_tensors="pt")["image_grid_thw"][0]
        lr_img = image.resize((int(thw_lr[2]) * 16, int(thw_lr[1]) * 16), Image.BICUBIC)
        q = {"type": "text", "text": prompt}
        content = [{"type": "image", "image": lr_img}, q, {"type": "text", "text": dat_mod.WHERE_HINT},
                   {"type": "image", "image": pil0}, dict(q)]
        msgs = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": content}]
        text = hr_processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inp = hr_processor(text=[text], images=[lr_img, pil0], return_tensors="pt")
        ids = inp["input_ids"].to(dev)
        thw = inp["image_grid_thw"].to(dev)
        pv = inp["pixel_values"].to(dev, torch.float32)
        n_lr = int(torch.prod(thw[0]))
        if tuple(thw[1].tolist()) != (1, Ho // ps, Wo // ps):
            print(f"[ogp] {gi}: processor resized the 2nd image {tuple(thw[1].tolist())} vs {(1, Ho // ps, Wo // ps)}; skip")
            continue
        patch_diff = float((patchify(img0[0].clamp(0, 1)) - pv[n_lr:]).abs().max())
        mm = torch.zeros_like(ids)
        mm[ids == model.config.image_token_id] = 1
        pos3d, rope_deltas = model.model.get_rope_index(ids, mm, image_grid_thw=thw)
        letters = re.findall(r"^\(?([A-Z])[.)]", prompt, flags=re.M) or ["A", "B"]
        letter_ids = [tok.encode(l, add_special_tokens=False)[0] for l in letters]
        gt_idx = letters.index(s["gt"]) if s["gt"] in letters else 0

        def loss_of(off, grad=False):
            with torch.set_grad_enabled(grad):
                img, _, _ = draw(off)
                pvals = torch.cat([pv[:n_lr], patchify(img[0].clamp(0, 1))], 0).to(torch.bfloat16)
                model.model.rope_deltas = rope_deltas
                o = model(input_ids=ids, position_ids=pos3d, mm_token_type_ids=mm, pixel_values=pvals,
                          image_grid_thw=thw, use_cache=False, return_dict=True)
                lg = o.logits[0, -1, letter_ids].float()
                return -torch.log_softmax(lg, -1)[gt_idx], int(lg.argmax()) == gt_idx

        off = off0.clone().requires_grad_(True)
        loss0, ok0 = loss_of(off, grad=True)
        loss0.backward()
        g = off.grad.detach()[0]                                  # (2, gh, gw), token units
        # per-token sampling centres in original px, direction to the GT box centre
        cxs = px0 + (torch.arange(gw, device=dev) + 0.5) * (px1 - px0) / gw
        cys = py0 + (torch.arange(gh, device=dev) + 0.5) * (py1 - py0) / gh
        cy, cx = torch.meshgrid(cys, cxs, indexing="ij")
        bx, by = (box[0] + box[2]) / 2 * W, (box[1] + box[3]) / 2 * H
        ux, uy = bx - cx, by - cy
        outside = ~((cx >= box[0] * W) & (cx <= box[2] * W) & (cy >= box[1] * H) & (cy <= box[3] * H))
        gn = torch.sqrt(g[0] ** 2 + g[1] ** 2)
        un = torch.sqrt(ux ** 2 + uy ** 2).clamp(min=1e-6)
        cosv = (-(g[0] * ux + g[1] * uy)) / (gn.clamp(min=1e-12) * un)
        wsum = float((gn * outside).sum())
        cos = float((cosv * gn * outside).sum()) / wsum if wsum > 0 else None

        step = -args.step_cells * g / gn.max().clamp(min=1e-12)
        rnd = torch.randn_like(g)
        rnd = rnd * (step.norm() / rnd.norm().clamp(min=1e-12))
        l1, _ = loss_of(step[None])
        lr_, _ = loss_of(rnd[None])
        _, sx1, sy1 = draw(step[None])
        r = {"i": gi, "cat": s["category"], "ok0": bool(ok0), "loss0": float(loss0),
             "cos": cos, "inbox0": inbox(sx0, sy0), "inbox1": inbox(sx1, sy1),
             "dloss_grad": float(l1) - float(loss0), "dloss_rand": float(lr_) - float(loss0),
             "patch_diff": patch_diff, "grid": [gh, gw], "grad_max": float(gn.max())}
        rows.append(r)
        print(f"[ogp] {gi} {s['category'][:8]} ok0={int(ok0)} loss0={r['loss0']:.3f} cos={cos if cos is None else round(cos, 3)} "
              f"inbox {r['inbox0']:.3f}->{r['inbox1']:.3f} dloss grad {r['dloss_grad']:+.4f} rand {r['dloss_rand']:+.4f} "
              f"patch_diff {patch_diff:.4f}", flush=True)
        del off, loss0
        torch.cuda.empty_cache()

    if args.out:
        os.makedirs(os.path.dirname(os.path.expanduser(args.out)) or ".", exist_ok=True)
        json.dump(rows, open(os.path.expanduser(args.out), "w"))
        print(f"[ogp] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
