#!/usr/bin/env python
"""Pure language-model speed. No ViT, no where-head, no window mosaic.

The transformer is Qwen3.5-2B. DAT and the baseline share that shape; the
extra DAT cost inside the language model is the longer sequence (the visual
tokens plus about 400 window tokens). This script times both lengths.

Prefill is one transformer forward on already-built embeddings. Decode is
32 single-token steps with the KV cache. Each length is warmed up first.

    python -u scripts/bench_llm_speed.py --out $X/speed_ldat/speed_llm.json
"""
import argparse
import json
import os
import time

import numpy as np
import torch


def _stat(xs):
    a = np.asarray(xs, dtype=np.float64)
    return float(a.mean()), float(np.median(a))


@torch.inference_mode()
def time_llm(model, embeds, decode_len, device):
    """embeds: (1, seq, hidden), already on device. ViT is not called."""
    n = embeds.shape[1]
    pos = torch.arange(n, device=device).view(1, 1, n).expand(3, 1, n)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model(
        inputs_embeds=embeds,
        position_ids=pos,
        cache_position=torch.arange(n, device=device),
        use_cache=True,
        return_dict=True,
        logits_to_keep=1,
    )
    torch.cuda.synchronize()
    t_pre = time.perf_counter()
    cache = out.past_key_values
    tok = int(out.logits[0, -1].argmax())
    for k in range(int(decode_len)):
        out = model(
            input_ids=torch.tensor([[tok]], device=device),
            position_ids=torch.full((3, 1, 1), n + k, device=device, dtype=torch.long),
            past_key_values=cache,
            cache_position=torch.tensor([n + k], device=device),
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )
        cache = out.past_key_values
        tok = int(out.logits[0, -1].argmax())
    torch.cuda.synchronize()
    t_dec = time.perf_counter()
    return {
        "prefill_ms": (t_pre - t0) * 1000.0,
        "decode_ms": (t_dec - t_pre) * 1000.0,
        "seq": n,
    }


def make_embeds(embed, n, device):
    vocab = embed.weight.shape[0]
    ids = torch.arange(n, device=device) % vocab
    return embed(ids)[None]


def run_point(model, embed, n, args):
    warm = make_embeds(embed, n, args.device)
    print(f"[llm] seq={n} warmup x{args.warmup}", flush=True)
    for _ in range(args.warmup):
        time_llm(model, warm, args.decode_len, args.device)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    rows = []
    embeds = make_embeds(embed, n, args.device)
    for i in range(args.iters):
        rec = time_llm(model, embeds, args.decode_len, args.device)
        rows.append(rec)
        print(f"[llm] seq={n} {i + 1}/{args.iters}  "
              f"prefill {rec['prefill_ms']:.0f} ms  decode {rec['decode_ms']:.0f} ms", flush=True)
    pre_m, pre_p50 = _stat([r["prefill_ms"] for r in rows])
    dec_m, dec_p50 = _stat([r["decode_ms"] for r in rows])
    return {
        "seq": n,
        "decode_len": int(args.decode_len),
        "iters": len(rows),
        "avg_prefill_ms": pre_m,
        "median_prefill_ms": pre_p50,
        "avg_decode_ms": dec_m,
        "median_decode_ms": dec_p50,
        "ms_per_decode_tok": dec_m / args.decode_len,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
    }


def main():
    x = os.environ.get("X", "/data/oss_bucket_0/wangziyi/vldat_experiments/sa1b_ldat")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="/data/oss_bucket_0/wangziyi/official_ckpt/Qwen3.5-2B")
    ap.add_argument("--tokens", type=int, nargs="+", default=[640, 1280, 2560, 6400])
    ap.add_argument("--window_tokens", type=int, default=400,
                    help="extra sequence length for the DAT window; 0 skips those rows")
    ap.add_argument("--text_tokens", type=int, default=64)
    ap.add_argument("--decode_len", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=os.path.join(x, "speed_ldat", "speed_llm.json"))
    args = ap.parse_args()

    from transformers import Qwen3_5ForConditionalGeneration
    print(f"[llm] loading {args.model_path}", flush=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map={"": args.device},
        attn_implementation=args.attn).eval()
    embed = model.get_input_embeddings()

    lengths = []
    for tok in args.tokens:
        lengths.append((tok, 0, tok + args.text_tokens))
        if args.window_tokens > 0:
            lengths.append((tok, args.window_tokens, tok + args.window_tokens + args.text_tokens))

    points = []
    for vis, win, n in lengths:
        pt = run_point(model, embed, n, args)
        pt["vis_tokens"] = vis
        pt["window_tokens"] = win
        pt["text_tokens"] = args.text_tokens
        points.append(pt)
        tag = "dat" if win else "base"
        print(f"[llm] {tag} vis={vis} win={win} seq={n}  "
              f"prefill avg {pt['avg_prefill_ms']:.0f} ms (p50 {pt['median_prefill_ms']:.0f})  "
              f"decode {pt['avg_decode_ms']:.0f} ms  {pt['ms_per_decode_tok']:.1f} ms/tok  "
              f"vram {pt['peak_vram_gb']:.1f}G", flush=True)

    result = {
        "model_path": args.model_path,
        "text_tokens": args.text_tokens,
        "window_tokens": args.window_tokens,
        "decode_len": args.decode_len,
        "warmup": args.warmup,
        "iters": args.iters,
        "attn": args.attn,
        "gpu": torch.cuda.get_device_name(0),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "embeddings are built before the timer; ViT is not called",
        "points": points,
    }
    out = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[llm] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
