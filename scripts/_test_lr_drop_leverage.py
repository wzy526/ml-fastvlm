#!/usr/bin/env python3
"""LR-dropout leverage test (no training): does blanking LR tokens move the
training pressure onto the HD path?

For N training samples (LLaVA-style json: image + conversations) we run one
forward+backward under four settings and compare loss and gradient norms:

    A  full LR,  HD on      (normal training step)
    B  LR drop,  HD on      (what LR-dropout training would see)
    C  LR drop,  HD off     (same blanked LR, no HD)   -> loss(C)-loss(B) = HD's help under starvation
    D  full LR,  HD off     ->  loss(D)-loss(A)  = HD's help today

The lever exists if grad_norm(k/v_proj_hd) under B is several x that under A.
HD currently carries content iff loss(C) > loss(B) by a margin.

Usage (OSS pod):
    DAT_ATTN_BACKEND=fa2 python scripts/_test_lr_drop_leverage.py \
        --model_path ~/vldat_experiments/0908_pretrain_qwen35_2b_dat_exactgrad \
        --data_json ~/sft_data/xxx.json --image_folder ~/sft_data \
        --n 32 --ratio 0.75 --out /tmp/lr_drop_leverage.json
"""

import argparse
import json
import math
import os
import random
import sys

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.probe_whd_qwen35 import FACTOR, TOK_PX, SYSTEM_PROMPT, hd_target_size  # noqa: E402

KVHD_KEYS = ("k_proj_hd", "v_proj_hd")
KVLR_KEYS = ("self_attn.k_proj.", "self_attn.v_proj.")
OFF_KEYS = ("conv_off_proj", "conv_lr_dw", "conv_lr_proj")


def load_items(path, image_folder, n, seed):
    with open(path) as f:
        data = json.load(f)
    rng = random.Random(seed)
    rng.shuffle(data)
    items = []
    for it in data:
        img = it.get("image")
        conv = it.get("conversations") or []
        if not img or isinstance(img, list) or len(conv) < 2:
            continue
        if conv[0].get("from") != "human" or "<image>" not in conv[0].get("value", ""):
            continue
        p = img if os.path.isabs(img) else os.path.join(image_folder, img)
        if not os.path.exists(p):
            continue
        q = conv[0]["value"].replace("<image>", "").strip()
        a = conv[1]["value"].strip()
        if not q or not a:
            continue
        items.append((p, q, a))
        if len(items) >= n:
            break
    return items


def build(item, processor, hr_processor, args, device, dtype):
    path, q, a = item
    img = Image.open(path).convert("RGB")
    user = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "image", "image": img},
                                         {"type": "text", "text": q}]}]
    prompt = processor.apply_chat_template(user, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
    full = prompt + a + "<|im_end|>\n"
    inputs = processor(text=[full], images=[img], return_tensors="pt")
    n_prompt = processor(text=[prompt], images=[img], return_tensors="pt")["input_ids"].shape[1]
    labels = inputs["input_ids"].clone()
    labels[:, :n_prompt] = -100
    if args.max_answer_tokens > 0 and labels.shape[1] - n_prompt > args.max_answer_tokens:
        cut = n_prompt + args.max_answer_tokens
        inputs = {k: (v[:, :cut] if k in ("input_ids", "attention_mask") else v)
                  for k, v in inputs.items()}
        labels = labels[:, :cut]

    lr_thw = inputs["image_grid_thw"][0]
    hd_w, hd_h = hd_target_size(img, lr_thw, args.hr_scale, args.hd_cap)
    hr = hr_processor.image_processor(images=[img.resize((hd_w, hd_h), Image.BICUBIC)],
                                      return_tensors="pt")
    base = {}
    for k, v in inputs.items():
        if not isinstance(v, torch.Tensor):
            continue
        base[k] = v.to(device=device, dtype=dtype) if k == "pixel_values" else v.to(device)
    base["labels"] = labels.to(device)
    hd = {"pixel_values_hd": hr["pixel_values"].to(device=device, dtype=dtype),
          "image_grid_thw_hd": hr["image_grid_thw"].to(device)}
    n_ans = int((labels != -100).sum())
    return base, hd, int(lr_thw[1] * lr_thw[2] // 4), n_ans


def grad_norm(model, keys):
    tot = 0.0
    for n, p in model.named_parameters():
        if p.grad is not None and any(k in n for k in keys):
            tot += float(p.grad.float().norm() ** 2)
    return math.sqrt(tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--processor_path", default=None)
    ap.add_argument("--data_json", required=True)
    ap.add_argument("--image_folder", default="")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--ratio", type=float, default=0.75, help="LR tokens blanked in B/C")
    ap.add_argument("--tok_budget", type=int, default=512, help="LR token budget (train range 256-640)")
    ap.add_argument("--min_pixels", type=int, default=200704)
    ap.add_argument("--hd_cap", type=int, default=5017600)
    ap.add_argument("--hr_scale", type=int, default=3)
    ap.add_argument("--max_answer_tokens", type=int, default=128)
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoProcessor
    from llava.model.language_model.modeling_qwen3_5_dat import Qwen3_5DATForConditionalGeneration

    items = load_items(args.data_json, args.image_folder, args.n, args.seed)
    print(f"[lrdrop] {len(items)} samples from {args.data_json}")

    model = Qwen3_5DATForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation=args.attn,
    )
    model.train()
    # grads only where we measure them (base k/v of DAT layers as the reference)
    dat_layer_prefixes = set()
    for n, _ in model.named_modules():
        if "self_attn" in n and hasattr(model.get_submodule(n), "k_proj_hd"):
            dat_layer_prefixes.add(n)
    for n, p in model.named_parameters():
        p.requires_grad = any(n.startswith(pre) for pre in dat_layer_prefixes)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[lrdrop] measuring grads on {len(dat_layer_prefixes)} DAT attention modules "
          f"({n_train/1e6:.1f}M params)")

    ppath = args.processor_path or args.model_path
    processor = AutoProcessor.from_pretrained(ppath, min_pixels=args.min_pixels,
                                              max_pixels=args.tok_budget * TOK_PX)
    hr_processor = AutoProcessor.from_pretrained(ppath, min_pixels=TOK_PX, max_pixels=100_000_000)
    device = next(model.parameters()).device
    dtype = torch.bfloat16

    settings = [("A full LR, HD on", False, True),
                ("B LR drop, HD on", True, True),
                ("C LR drop, HD off", True, False),
                ("D full LR, HD off", False, False)]
    rec = {s[0]: {"loss": [], "g_kvhd": [], "g_kvlr": [], "g_off": [], "lr_drop_frac": []}
           for s in settings}
    os.environ["DAT_LR_DROP_RATIO"] = str(args.ratio)

    for i, item in enumerate(tqdm(items, desc="samples")):
        try:
            base, hd, n_lr, n_ans = build(item, processor, hr_processor, args, device, dtype)
        except Exception as e:  # noqa: BLE001
            print(f"[lrdrop] skip {item[0]}: {e}")
            continue
        for name, drop, hd_on in settings:
            os.environ["DAT_LR_DROP_FORCE"] = "1" if drop else "0"
            torch.manual_seed(args.seed * 100003 + i)      # identical mask for B and C
            model.zero_grad(set_to_none=True)
            inputs = {**base, **hd} if hd_on else dict(base)
            out = model(**inputs)
            out.loss.backward()
            r = rec[name]
            r["loss"].append(float(out.loss))
            r["g_kvhd"].append(grad_norm(model, KVHD_KEYS))
            r["g_kvlr"].append(grad_norm(model, KVLR_KEYS))
            r["g_off"].append(grad_norm(model, OFF_KEYS))
            r["lr_drop_frac"].append(float(getattr(model, "_lr_drop_frac", 0.0)))
        model.zero_grad(set_to_none=True)
    os.environ.pop("DAT_LR_DROP_FORCE", None)

    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    print(f"\n==== LR-dropout leverage  (n={len(rec[settings[0][0]]['loss'])}, "
          f"ratio={args.ratio}, tok_budget={args.tok_budget}) ====")
    print(f"{'setting':>20} | {'loss':>7} | {'grad k/v_hd':>11} | {'grad k/v_lr':>11} | "
          f"{'grad offset':>11} | {'hd/lr':>6} | {'lr_drop':>7}")
    print("-" * 95)
    summ = {}
    for name, _, hd_on in settings:
        r = rec[name]
        s = {k: mean(v) for k, v in r.items()}
        s["hd_over_lr"] = s["g_kvhd"] / s["g_kvlr"] if s["g_kvlr"] > 0 else float("nan")
        summ[name] = s
        kv = f"{s['g_kvhd']:>11.4f}" if hd_on else f"{'—':>11}"
        ratio = f"{s['hd_over_lr']:>6.3f}" if hd_on else f"{'—':>6}"
        print(f"{name:>20} | {s['loss']:>7.4f} | {kv} | {s['g_kvlr']:>11.4f} | "
              f"{s['g_off']:>11.4f} | {ratio} | {s['lr_drop_frac']:>7.3f}")

    A, B, C, D = (summ[s[0]] for s in settings)
    lever = B["g_kvhd"] / A["g_kvhd"] if A["g_kvhd"] > 0 else float("nan")
    pa = [c - b for b, c in zip(rec[settings[1][0]]["loss"], rec[settings[2][0]]["loss"])]
    pd = [d - a for a, d in zip(rec[settings[0][0]]["loss"], rec[settings[3][0]]["loss"])]
    print("\n==== read-out ====")
    print(f"lever  grad(k/v_hd) B/A            : {lever:6.2f}x   (>3x = blanking LR pushes pressure onto HD)")
    print(f"HD help under starvation loss(C)-(B): {mean(pa):+.4f}  (paired mean; >0 = HD recovers blanked content)")
    print(f"HD help today        loss(D)-(A)   : {mean(pd):+.4f}  (paired mean; ~0 = HD contributes nothing now)")
    print(f"starvation cost      loss(B)-(A)   : {B['loss']-A['loss']:+.4f}")

    if args.out:
        json.dump({"args": vars(args), "summary": summ, "per_sample": rec,
                   "lever": lever, "hd_help_starved": mean(pa), "hd_help_today": mean(pd)},
                  open(args.out, "w"), indent=2)
        print(f"[lrdrop] saved -> {args.out}")


if __name__ == "__main__":
    main()
