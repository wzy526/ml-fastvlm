#!/usr/bin/env python3
"""LR-dropout leverage test (no training): does blanking LR tokens move the
training pressure onto the HD path?

For N training samples (LLaVA-style json: image + conversations) we run one
forward+backward under four settings and compare loss and gradient norms:

    A  full LR,  HD on      (normal training step)
    B  LR drop,  HD on      (what LR-dropout training would see)
    C  LR drop,  HD off     (same blanked LR, no HD)   -> loss(C)-loss(B) = HD's help under starvation
    D  full LR,  HD off     ->  loss(D)-loss(A)  = HD's help today

When the json carries a top-level `bbox` ([x0, y0, x1, y1] image fractions,
scripts/build_viscot_bbox_data.py) three more settings isolate the READOUT
from localisation (the learned grid in B is ~uniform, so HD may not even hold
the answer there):

    O  LR drop,  HD oracle  (grid laid on the GT box: HD holds the answer)
    S  LR drop,  HD shuffle (another sample's image, same grid: wrong content)
    P  full LR,  HD oracle  (loss-level version of the probe's oracle test)

    loss(S)-loss(O) > 0  -> the LM reads HD content when starved (readout works,
                            training just never needed it -> LR-dropout SFT)
    loss(S) ~ loss(O)    -> nothing readable comes out of k/v_proj_hd today; then
                            grad(k/v_hd) under O vs A says whether a training
                            signal exists at all (x-several = learnable).

The lever exists if grad_norm(k/v_proj_hd) under B is several x that under A.
HD currently carries content iff loss(C) > loss(B) by a margin.

Usage (OSS pod):
    DAT_ATTN_BACKEND=fa2 python scripts/_test_lr_drop_leverage.py \
        --model_path ~/vldat_experiments/0908_pretrain_qwen35_2b_dat_exactgrad \
        --data_json ~/sft_data/xxx.json --image_folder ~/sft_data \
        --n 32 --ratio 0.75 --out /tmp/lr_drop_leverage.json
    # readout test with oracle windows (bbox json from build_viscot_bbox_data.py):
    python scripts/_test_lr_drop_leverage.py --model_path <v2-merged> \
        --data_json $OSS_DATA/extra_0916/viscot_bbox.json --image_folder ~/sft_data/train_split \
        --n 64 --tok_budget 256 --out <v2-merged>/readout_leverage.json
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
from scripts.probe_whd_qwen35 import (  # noqa: E402
    FACTOR, TOK_PX, SYSTEM_PROMPT, hd_target_size, oracle_locs, set_force_locs,
)

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
        bbox = it.get("bbox")                     # [x0, y0, x1, y1] fractions, or None
        items.append((p, q, a, bbox))
        if len(items) >= n:
            break
    n_box = sum(1 for it in items if it[3])
    if 0 < n_box < len(items):
        print(f"[lrdrop] {len(items) - n_box} samples without bbox dropped (oracle settings need it)")
        items = [it for it in items if it[3]]
    return items


def hd_inputs(img, hd_w, hd_h, hr_processor, device, dtype):
    hr = hr_processor.image_processor(images=[img.resize((hd_w, hd_h), Image.BICUBIC)],
                                      return_tensors="pt")
    return {"pixel_values_hd": hr["pixel_values"].to(device=device, dtype=dtype),
            "image_grid_thw_hd": hr["image_grid_thw"].to(device)}


def build(item, idx, processor, hr_processor, args, device, dtype, grid_size):
    path, q, a, bbox = item
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
        seq_keys = ("input_ids", "attention_mask", "mm_token_type_ids", "position_ids")
        inputs = {k: (v[:, :cut] if k in seq_keys else v) for k, v in inputs.items()}
        labels = labels[:, :cut]

    lr_thw = inputs["image_grid_thw"][0]
    hd_w, hd_h = hd_target_size(img, lr_thw, args.hr_scale, args.hd_cap)
    base = {}
    for k, v in inputs.items():
        if not isinstance(v, torch.Tensor):
            continue
        base[k] = v.to(device=device, dtype=dtype) if k == "pixel_values" else v.to(device)
    base["labels"] = labels.to(device)
    hd = hd_inputs(img, hd_w, hd_h, hr_processor, device, dtype)
    locs = None
    if bbox:
        # same window construction as the probe's --hd_source oracle
        W, H = img.size
        x0, y0, x1, y1 = bbox
        sample = {"image": img, "bboxes": [[x0 * W, y0 * H, (x1 - x0) * W, (y1 - y0) * H]]}
        locs = oracle_locs(sample, idx, hd_w, hd_h, grid_size, args.oracle_min_cells, "oracle")
    n_ans = int((labels != -100).sum())
    return base, hd, (hd_w, hd_h), locs, int(lr_thw[1] * lr_thw[2] // 4), n_ans


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
    ap.add_argument("--question_hd", action="store_true",
                    help="add setting E: question tokens also read (image-conditioned) HD K/V "
                         "(dat_image_hd_for_question=True at runtime; needs intention_as_gate)")
    ap.add_argument("--lse_bias", type=float, default=0.0,
                    help="add setting F: constant added to lse_hd (training-time curriculum knob)")
    ap.add_argument("--oracle_min_cells", type=int, default=20,
                    help="oracle window >= this many HD cells per side (probe default)")
    ap.add_argument("--no_oracle", action="store_true",
                    help="skip settings O/S/P even when the json has bboxes")
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

    # (name, lr_drop, hd: off|real|oracle|shuffle, question_readers, lse_bias)
    settings = [("A full LR, HD on", False, "real", False, 0.0),
                ("B LR drop, HD on", True, "real", False, 0.0),
                ("C LR drop, HD off", True, "off", False, 0.0),
                ("D full LR, HD off", False, "off", False, 0.0)]
    has_bbox = bool(items) and all(it[3] for it in items) and not args.no_oracle
    if has_bbox:
        settings += [("O LR drop, HD oracle", True, "oracle", False, 0.0),
                     ("S LR drop, HD shuffle", True, "shuffle", False, 0.0),
                     ("P full LR, HD oracle", False, "oracle", False, 0.0)]
    if args.question_hd:
        settings.append(("E +question readers", False, "real", True, 0.0))
    if args.lse_bias != 0.0:
        settings.append((f"F lse_bias={args.lse_bias:+g}", False, "real", False, args.lse_bias))
    if args.question_hd and args.lse_bias != 0.0:
        settings.append(("G question+bias", False, "real", True, args.lse_bias))
    rec = {s[0]: {"loss": [], "g_kvhd": [], "g_kvlr": [], "g_off": [], "lr_drop_frac": []}
           for s in settings}
    os.environ["DAT_LR_DROP_RATIO"] = str(args.ratio)
    grid_size = int(model.config.dat_extra_args.get("grid_size", 20))
    print(f"[lrdrop] settings: {[s[0] for s in settings]}  grid_size={grid_size}")

    dat_mods = [model.get_submodule(n) for n in sorted(dat_layer_prefixes)]

    def set_knobs(q_hd, bias):
        for m in dat_mods:
            m.dat_image_hd_for_question = bool(q_hd)
            m.hd_lse_bias = float(bias)

    for i, item in enumerate(tqdm(items, desc="samples")):
        try:
            base, hd, (hd_w, hd_h), locs, n_lr, n_ans = build(
                item, i, processor, hr_processor, args, device, dtype, grid_size)
        except Exception as e:  # noqa: BLE001
            print(f"[lrdrop] skip {item[0]}: {e}")
            continue
        # HD ViT once per sample (frozen, no grad); the settings share the features
        with torch.no_grad():
            hd_feats = model._generate_hd_features(hd["pixel_values_hd"], hd["image_grid_thw_hd"])
            hd_feats_shuf = None
            if any(s[2] == "shuffle" for s in settings):
                # another sample's image at the SAME HD geometry (probe's shuffle)
                other = items[(i + len(items) // 2) % len(items)][0]
                hd_o = hd_inputs(Image.open(other).convert("RGB"), hd_w, hd_h,
                                 hr_processor, device, dtype)
                hd_feats_shuf = model._generate_hd_features(hd_o["pixel_values_hd"],
                                                            hd_o["image_grid_thw_hd"])
        for name, drop, hd_mode, q_hd, bias in settings:
            hd_on = hd_mode != "off"
            os.environ["DAT_LR_DROP_FORCE"] = "1" if drop else "0"
            set_knobs(q_hd, bias)
            set_force_locs(model, locs if hd_mode in ("oracle", "shuffle") else None)
            torch.manual_seed(args.seed * 100003 + i)      # identical mask for B/C/O/S
            model.zero_grad(set_to_none=True)
            feats = hd_feats_shuf if hd_mode == "shuffle" else hd_feats
            inputs = {**base, "image_hd_features": feats} if hd_on else dict(base)
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
    set_knobs(False, 0.0)
    set_force_locs(model, None)

    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    print(f"\n==== LR-dropout leverage  (n={len(rec[settings[0][0]]['loss'])}, "
          f"ratio={args.ratio}, tok_budget={args.tok_budget}) ====")
    print(f"{'setting':>20} | {'loss':>7} | {'grad k/v_hd':>11} | {'grad k/v_lr':>11} | "
          f"{'grad offset':>11} | {'hd/lr':>6} | {'lr_drop':>7}")
    print("-" * 95)
    summ = {}
    for name, _, hd_mode, _, _ in settings:
        hd_on = hd_mode != "off"
        r = rec[name]
        s = {k: mean(v) for k, v in r.items()}
        s["hd_over_lr"] = s["g_kvhd"] / s["g_kvlr"] if s["g_kvlr"] > 0 else float("nan")
        summ[name] = s
        kv = f"{s['g_kvhd']:>11.4f}" if hd_on else f"{'—':>11}"
        ratio = f"{s['hd_over_lr']:>6.3f}" if hd_on else f"{'—':>6}"
        print(f"{name:>20} | {s['loss']:>7.4f} | {kv} | {s['g_kvlr']:>11.4f} | "
              f"{s['g_off']:>11.4f} | {ratio} | {s['lr_drop_frac']:>7.3f}")

    A, B, C, D = (summ[s[0]] for s in settings[:4])
    lever = B["g_kvhd"] / A["g_kvhd"] if A["g_kvhd"] > 0 else float("nan")
    pa = [c - b for b, c in zip(rec[settings[1][0]]["loss"], rec[settings[2][0]]["loss"])]
    pd = [d - a for a, d in zip(rec[settings[0][0]]["loss"], rec[settings[3][0]]["loss"])]
    print("\n==== read-out ====")
    print(f"lever  grad(k/v_hd) B/A            : {lever:6.2f}x   (>3x = blanking LR pushes pressure onto HD)")
    print(f"HD help under starvation loss(C)-(B): {mean(pa):+.4f}  (paired mean; >0 = HD recovers blanked content)")
    print(f"HD help today        loss(D)-(A)   : {mean(pd):+.4f}  (paired mean; ~0 = HD contributes nothing now)")
    print(f"starvation cost      loss(B)-(A)   : {B['loss']-A['loss']:+.4f}")
    if has_bbox:
        L = lambda k: rec[k]["loss"]
        so = [b_ - a_ for a_, b_ in zip(L("O LR drop, HD oracle"), L("S LR drop, HD shuffle"))]
        co = [b_ - a_ for a_, b_ in zip(L("O LR drop, HD oracle"), L("C LR drop, HD off"))]
        dp = [b_ - a_ for a_, b_ in zip(L("P full LR, HD oracle"), L("D full LR, HD off"))]
        O = summ["O LR drop, HD oracle"]
        print("\n==== read-out: readout vs localisation (bbox settings) ====")
        print(f"content read, starved  loss(S)-(O) : {mean(so):+.4f}  (>0 = oracle HD beats wrong-image HD: the LM READS HD content)")
        print(f"oracle help, starved   loss(C)-(O) : {mean(co):+.4f}  (>0 = HD on the answer recovers blanked LR)")
        print(f"oracle help today      loss(D)-(P) : {mean(dp):+.4f}  (~0 = with LR intact the LM ignores even perfect HD)")
        print(f"signal for training  grad(k/v_hd) O/A : {O['g_kvhd']/A['g_kvhd'] if A['g_kvhd'] > 0 else float('nan'):6.2f}x  "
              f"(hd/lr {O['hd_over_lr']:.3f} vs A {A['hd_over_lr']:.3f}; x-several = learnable under LR-drop + TF)")
    for s in settings[4 + (3 if has_bbox else 0):]:
        X = summ[s[0]]
        print(f"{s[0]:>20}  grad(k/v_hd) x{X['g_kvhd']/A['g_kvhd']:5.2f} vs A   hd/lr {X['hd_over_lr']:.3f} "
              f"(A {A['hd_over_lr']:.3f})   loss {X['loss']-A['loss']:+.4f} vs A")

    if args.out:
        extra = {}
        if has_bbox:
            extra = {"content_read_starved": mean(so), "oracle_help_starved": mean(co),
                     "oracle_help_today": mean(dp)}
        json.dump({"args": vars(args), "summary": summ, "per_sample": rec,
                   "lever": lever, "hd_help_starved": mean(pa), "hd_help_today": mean(pd), **extra},
                  open(args.out, "w"), indent=2)
        print(f"[lrdrop] saved -> {args.out}")


if __name__ == "__main__":
    main()
