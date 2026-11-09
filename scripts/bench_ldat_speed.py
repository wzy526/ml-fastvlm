#!/usr/bin/env python
"""Speed of the finalized learnable sampler. One GPU.

LR tokens follow the accuracy sweep: 640 1280 2560 6400. The window budget
stays at 400. Prefill is everything through the first token (where-head,
window mosaic, LR ViT, LM prefill). Decode is the next fixed number of
tokens; end-of-sequence does not stop the loop, so every sample has the
same decode length.

Each LR size is warmed up before it is timed. The warmup runs are not
recorded. Shapes change with the LR size, so the warmup is per size.

    python -u scripts/bench_ldat_speed.py \
        --model_path $E/checkpoint-1000 \
        --where_head_pt $E/loc/where_lm8_ce20.pt \
        --ldat_pt $X/hinge_noratio_ldat.pt \
        --out $X/speed_ldat/speed.json
"""
import argparse
import base64
import glob
import io
import json
import math
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from ldat_eval.infer import LDatRunner  # noqa: E402
from offset_feat_train import global_positions  # noqa: E402


def _as_image(raw):
    if isinstance(raw, Image.Image):
        return raw.convert("RGB")
    if isinstance(raw, dict):
        raw = raw.get("bytes") or raw.get("path")
    if isinstance(raw, str) and os.path.isfile(raw):
        return Image.open(raw).convert("RGB")
    if isinstance(raw, str):
        raw = base64.b64decode(raw)
    if isinstance(raw, (bytes, bytearray)):
        return Image.open(io.BytesIO(raw)).convert("RGB")
    raise TypeError(f"unsupported image value {type(raw)}")


def _find_parquet(which):
    name = f"hr_bench_{which}.parquet"
    roots = [
        os.environ.get("HF_HUB_CACHE", ""),
        os.path.expanduser("~/.cache/huggingface/hub"),
        os.path.expanduser("~/.cache/huggingface/datasets"),
    ]
    patterns = [
        os.path.join("datasets--DreamMr--HR-Bench", "snapshots", "*", name),
        os.path.join("DreamMr___HR-Bench", "**", name),
    ]
    hits = []
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        for pat in patterns:
            hits.extend(glob.glob(os.path.join(root, pat), recursive=True))
    return sorted(set(hits))


def load_samples(args):
    n = args.max_samples
    if args.image_folder:
        exts = {".jpg", ".jpeg", ".png", ".webp"}
        files = sorted(
            os.path.join(args.image_folder, f)
            for f in os.listdir(os.path.expanduser(args.image_folder))
            if os.path.splitext(f)[1].lower() in exts
        )
        if not files:
            raise SystemExit(f"[speed] no images in {args.image_folder}")
        files = (files * n)[:n]
        q = "What is shown in the image?"
        return [{"image": Image.open(f).convert("RGB"), "question": q, "path": f} for f in files]

    hits = _find_parquet(args.dataset)
    if not hits:
        raise SystemExit(
            "[speed] no local HR-Bench parquet. Pass --image_folder, or check "
            "HF_HUB_CACHE for hr_bench_4k.parquet")
    path = hits[-1]
    print(f"[speed] reading {n} rows from {path}", flush=True)
    import pyarrow.parquet as pq
    rows = []
    pf = pq.ParquetFile(path)
    names = list(getattr(pf.schema_arrow, "names", None) or pf.schema.names)
    cols = [c for c in ("image", "question") if c in names]
    if "image" not in cols:
        raise SystemExit(f"[speed] {path} has no image column ({names})")
    for batch in pf.iter_batches(batch_size=n, columns=cols):
        data = batch.to_pydict()
        qs = data.get("question") or ["What is shown in the image?"] * len(data["image"])
        for raw, q in zip(data["image"], qs):
            rows.append({"image": _as_image(raw), "question": str(q), "path": path})
            if len(rows) >= n:
                return rows
    if not rows:
        raise SystemExit(f"[speed] {path} produced no images")
    return rows


def set_lr_tok(runner, model_path, lr_tok):
    from transformers import AutoProcessor
    runner.lr_tok = int(lr_tok)
    runner.processor = AutoProcessor.from_pretrained(
        model_path, min_pixels=1024, max_pixels=runner.lr_tok * 1024)


@torch.inference_mode()
def time_one(runner, image, question, decode_len):
    """prepare + first LM forward, then exactly decode_len more forwards."""
    dev = runner.dev
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    pr = runner.prepare(image, question)
    torch.cuda.synchronize()
    t_prep = time.perf_counter()
    if pr is None:
        return None
    e = pr["emb"].clone()
    e[0, pr["img_pos"]] = torch.cat([pr["lr_emb"]] + pr["feats"], 0).to(e.dtype)
    pos = global_positions(pr["pos3d"], pr["blocks"], pr["wins"], pr["pts"])
    runner.model.model.rope_deltas = pr["rope_deltas"]
    n = e.shape[1]
    o = runner.model(
        inputs_embeds=e, position_ids=pos, mm_token_type_ids=pr["mm"],
        cache_position=torch.arange(n, device=dev),
        use_cache=True, return_dict=True, logits_to_keep=1)
    torch.cuda.synchronize()
    t_pre = time.perf_counter()
    cache = o.past_key_values
    next_pos = int(math.floor(float(pos.max()))) + 1
    tok = int(o.logits[0, -1].argmax())
    for k in range(int(decode_len)):
        o = runner.model(
            input_ids=torch.tensor([[tok]], device=dev),
            position_ids=torch.full((3, 1, 1), next_pos + k, device=dev, dtype=torch.long),
            past_key_values=cache,
            cache_position=torch.tensor([n + k], device=dev),
            use_cache=True, return_dict=True)
        cache = o.past_key_values
        tok = int(o.logits[0, -1].argmax())
    torch.cuda.synchronize()
    t_dec = time.perf_counter()
    return {
        "prepare_ms": (t_prep - t0) * 1000.0,
        "lm_prefill_ms": (t_pre - t_prep) * 1000.0,
        "prefill_ms": (t_pre - t0) * 1000.0,
        "decode_ms": (t_dec - t_pre) * 1000.0,
        "prompt_tokens": int(pr["n_tokens"]),
        "lr_tokens": int(pr["lr_emb"].shape[0]),
        "win_tokens": int(sum(w["gh"] * w["gw"] for w in pr["wins"])),
    }


def _stat(xs):
    a = np.asarray(xs, dtype=np.float64)
    return float(a.mean()), float(np.median(a))


def run_point(runner, samples, lr_tok, args):
    set_lr_tok(runner, args.model_path, lr_tok)
    warm = samples[:max(args.warmup, 1)]
    print(f"[speed] lr={lr_tok} warmup x{len(warm)}", flush=True)
    for s in warm:
        time_one(runner, s["image"], s["question"], args.decode_len)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    rows = []
    for i, s in enumerate(samples):
        rec = time_one(runner, s["image"], s["question"], args.decode_len)
        if rec is None:
            print(f"[speed] lr={lr_tok} sample {i} prepare failed", flush=True)
            continue
        rows.append(rec)
        print(f"[speed] lr={lr_tok} {i + 1}/{len(samples)}  "
              f"prefill {rec['prefill_ms']:.0f} ms  decode {rec['decode_ms']:.0f} ms  "
              f"lr_tok {rec['lr_tokens']}  win {rec['win_tokens']}", flush=True)
    if not rows:
        raise SystemExit(f"[speed] lr={lr_tok} had no successful samples")

    pre_m, pre_p50 = _stat([r["prefill_ms"] for r in rows])
    dec_m, dec_p50 = _stat([r["decode_ms"] for r in rows])
    prep_m, _ = _stat([r["prepare_ms"] for r in rows])
    lm_m, _ = _stat([r["lm_prefill_ms"] for r in rows])
    lr_m, _ = _stat([r["lr_tokens"] for r in rows])
    win_m, _ = _stat([r["win_tokens"] for r in rows])
    prompt_m, _ = _stat([r["prompt_tokens"] for r in rows])
    point = {
        "lr_tok": int(lr_tok),
        "crop_tok": int(args.crop_tok),
        "decode_len": int(args.decode_len),
        "num_samples": len(rows),
        "avg_prefill_ms": pre_m,
        "median_prefill_ms": pre_p50,
        "avg_prepare_ms": prep_m,
        "avg_lm_prefill_ms": lm_m,
        "avg_decode_ms": dec_m,
        "median_decode_ms": dec_p50,
        "ms_per_decode_tok": dec_m / args.decode_len,
        "avg_lr_tokens": lr_m,
        "avg_win_tokens": win_m,
        "avg_prompt_tokens": prompt_m,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
    }
    if lr_m < 0.9 * lr_tok:
        print(f"[speed] lr budget {lr_tok} but images averaged {lr_m:.0f} LR tokens", flush=True)
    return point


def main():
    e = os.environ.get("E", os.path.expanduser("~/vldat_experiments/0924_miniE1_2k"))
    x = os.environ.get("X", "/data/oss_bucket_0/wangziyi/vldat_experiments/sa1b_ldat")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=os.path.join(e, "checkpoint-1000"))
    ap.add_argument("--where_head_pt", default=os.path.join(e, "loc", "where_lm8_ce20.pt"))
    ap.add_argument("--ldat_pt", default=os.path.join(x, "hinge_noratio_ldat.pt"))
    ap.add_argument("--tokens", type=int, nargs="+", default=[640, 1280, 2560, 6400])
    ap.add_argument("--crop_tok", type=int, default=400)
    ap.add_argument("--decode_len", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--max_samples", type=int, default=8)
    ap.add_argument("--dataset", choices=["4k", "8k"], default="4k",
                    help="HR-Bench parquet already in the HF cache")
    ap.add_argument("--image_folder", default=None)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=os.path.join(x, "speed_ldat", "speed.json"))
    args = ap.parse_args()

    samples = load_samples(args)
    print(f"[speed] {len(samples)} images  window={args.crop_tok}  "
          f"decode_len={args.decode_len}  warmup={args.warmup}  tokens={args.tokens}", flush=True)
    print(f"[speed] loading {args.model_path}", flush=True)
    runner = LDatRunner(
        args.model_path, args.where_head_pt, args.ldat_pt, args.device,
        crop_tok=args.crop_tok, lr_tok=args.tokens[0],
        dat_alpha=0.8, dat_ratio=0.0, src_scale=4.0, ldat_d=256,
        n_win=1, grid=0, attn=args.attn)

    points = []
    for tok in args.tokens:
        pt = run_point(runner, samples, tok, args)
        points.append(pt)
        print(f"[speed] lr={tok}  prefill avg {pt['avg_prefill_ms']:.0f} ms "
              f"(p50 {pt['median_prefill_ms']:.0f}, prepare {pt['avg_prepare_ms']:.0f} + "
              f"lm {pt['avg_lm_prefill_ms']:.0f})  "
              f"decode {pt['avg_decode_ms']:.0f} ms  {pt['ms_per_decode_tok']:.1f} ms/tok  "
              f"lr_tok {pt['avg_lr_tokens']:.0f}  win {pt['avg_win_tokens']:.0f}  "
              f"vram {pt['peak_vram_gb']:.1f}G", flush=True)

    result = {
        "model_path": args.model_path,
        "where_head_pt": args.where_head_pt,
        "ldat_pt": args.ldat_pt,
        "crop_tok": args.crop_tok,
        "decode_len": args.decode_len,
        "warmup": args.warmup,
        "dataset": args.image_folder or f"hrbench{args.dataset}",
        "attn": args.attn,
        "gpu": torch.cuda.get_device_name(0),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "points": points,
    }
    out = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[speed] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
