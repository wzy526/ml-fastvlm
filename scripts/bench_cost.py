#!/usr/bin/env python
"""Latency, FLOPs and visual / generated tokens per method. One GPU, batch 1.

The same images (evenly spaced rows of V* / HR-Bench 4K / 8K, prompts built by
the lmms-eval task code) go through every method. Each sample runs twice:

  timed    latency from the PIL image and prompt string to the last token,
           split into prepare / prefill / decode, plus per-module GPU time.
  counted  the same call under torch's FlopCounterMode (not timed). Greedy
           decoding repeats the timed tokens.

FLOPs are 2 x multiply-adds of every matmul / conv / SDPA the counter sees.
Qwen3.5's Gated DeltaNet core runs as fla Triton kernels the counter cannot see;
that part is added from its shapes (see gdn_core_flops) and reported as
gdn_est. A check sums every nn.Linear from hooks; the counter must not be
below it.

    python -u scripts/bench_cost.py --name "Native (LR)" --model qwen_native \
        --model_args pretrained=$M4,max_tok=640 --out $OUT/cost.jsonl
    python -u scripts/bench_cost.py --name "Q-LAS" --model ldat \
        --model_args pretrained=$M4B,where_head_pt=$M4B/where_lm8_ce20.pt,ldat_pt=$M4B/ldat_zero.pt,fast_prefix=1 \
        --out $OUT/cost.jsonl
    python scripts/bench_cost_table.py $OUT/cost.jsonl

--model is a key of bench_cost_models.MODELS or package.module:Class.
"""
import argparse
import importlib
import json
import os
import sys
import time
from collections import defaultdict
from functools import partial

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

VSTAR_POST = "\nAnswer with the option's letter from the given choices directly."
GDN_CHUNK = 64


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Clock:
    """Wall time and FLOPs between marks. flops_now() -> FLOPs counted so far."""

    def __init__(self, flops_now):
        self.flops_now = flops_now

    def start(self):
        _sync()
        self.t, self.f = time.perf_counter(), self.flops_now()
        self.ms, self.flops = {}, {}

    def mark(self, phase):
        _sync()
        t, f = time.perf_counter(), self.flops_now()
        self.ms[phase] = self.ms.get(phase, 0.0) + (t - self.t) * 1000.0
        self.flops[phase] = self.flops.get(phase, 0.0) + (f - self.f)
        self.t, self.f = t, f


class GroupProbe:
    """GPU time (inclusive) and FLOPs (exclusive of nested groups) per module group."""

    def __init__(self, groups, flops_now):
        self.flops_now = flops_now
        self.cuda = torch.cuda.is_available()
        self.handles = []
        for name, mod in groups.items():
            if mod is None:
                continue
            self.handles.append(mod.register_forward_pre_hook(partial(self._enter, name)))
            self.handles.append(mod.register_forward_hook(partial(self._exit, name)))
        self.reset()

    def reset(self):
        self.spans, self.flops, self.stack = defaultdict(list), defaultdict(float), []

    def _now(self):
        if self.cuda:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            return ev
        return time.perf_counter()

    def _enter(self, name, mod, args):
        self.stack.append([name, self._now(), self.flops_now(), 0.0])

    def _exit(self, name, mod, args, out):
        _, t0, f0, child = self.stack.pop()
        self.spans[name].append((t0, self._now()))
        d = self.flops_now() - f0
        self.flops[name] += d - child
        if self.stack:
            self.stack[-1][3] += d

    def result(self):
        _sync()
        ms = {}
        for name, spans in self.spans.items():
            if self.cuda:
                ms[name] = sum(a.elapsed_time(b) for a, b in spans)
            else:
                ms[name] = sum((b - a) * 1000.0 for a, b in spans)
        return ms, dict(self.flops)


def gdn_core_flops(tokens, seq, heads, dk, dv):
    """Gated delta rule core, per call. Chunked (seq > 1, chunk C): per token and head
    2 * (3 C dk + 2 C dv + 3 dk dv) -- K K^T, the W / U products and Q K^T inside the
    chunk, plus the query, update and correction terms against the dk x dv state.
    Recurrent (seq == 1): 2 * 3 dk dv per token and head."""
    if seq > 1:
        c = min(GDN_CHUNK, seq)
        return float(heads * tokens * 2 * (3 * c * dk + 2 * c * dv + 3 * dk * dv))
    return float(heads * tokens * 2 * 3 * dk * dv)


def _torch_visible(fn):
    mod = getattr(fn, "__module__", None) or ""
    return mod.startswith("transformers") or mod.startswith("torch")


class FlopExtras:
    """Shape-based FLOPs for kernels FlopCounterMode cannot see, and the nn.Linear check."""

    def __init__(self, model):
        self.active = False
        self.gdn = 0.0
        self.linear = 0.0
        self.handles = []
        for m in model.modules():
            if type(m).__name__.endswith("GatedDeltaNet"):
                self.handles.append(m.register_forward_pre_hook(self._gdn, with_kwargs=True))
            elif isinstance(m, torch.nn.Linear):
                self.handles.append(m.register_forward_hook(self._lin))

    def reset(self):
        self.gdn = self.linear = 0.0

    def _gdn(self, m, args, kwargs):
        if not self.active:
            return
        hs = args[0] if args else kwargs.get("hidden_states")
        if hs is None:
            return
        b, seq = int(hs.shape[0]), int(hs.shape[1])
        name = "chunk_gated_delta_rule" if seq > 1 else "recurrent_gated_delta_rule"
        fn = getattr(m, name, None) or getattr(m, "chunk_gated_delta_rule", None)
        if not _torch_visible(fn):
            self.gdn += gdn_core_flops(b * seq, seq, int(m.num_v_heads), int(m.head_k_dim), int(m.head_v_dim))
        conv = getattr(m, "conv1d", None)
        if getattr(m, "causal_conv1d_fn", None) is not None and conv is not None:
            self.gdn += 2.0 * b * seq * conv.weight.shape[0] * conv.weight.shape[-1]

    def _lin(self, m, args, out):
        if self.active:
            self.linear += 2.0 * (out.numel() // m.out_features) * m.in_features * m.out_features


def sdpa_check(device):
    """Counted / expected FLOPs of one SDPA on this device and dtype. The counter
    only knows some SDPA kernels (the CPU one is not among them)."""
    from torch.utils.flop_counter import FlopCounterMode

    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    q = torch.randn(1, 4, 256, 64, device=device, dtype=dtype)
    with torch.inference_mode(), FlopCounterMode(display=False) as fc:
        torch.nn.functional.scaled_dot_product_attention(q, q, q)
    return fc.get_total_flops() / (4.0 * 4 * 256 * 256 * 64)


def _parse_args_string(s):
    out = {}
    for item in filter(None, (s or "").split(",")):
        k, _, v = item.partition("=")
        v = v.strip()
        for cast in (int, float):
            try:
                v = cast(v)
                break
            except ValueError:
                pass
        if isinstance(v, str) and v.lower() in {"true", "false"}:
            v = v.lower() == "true"
        out[k.strip()] = v
    return out


def build_model(spec, model_args, device, attn):
    if ":" in spec:
        mod, cls = spec.split(":", 1)
        factory = getattr(importlib.import_module(mod), cls)
    else:
        from bench_cost_models import MODELS
        factory = MODELS[spec]
    return factory(device=device, attn=attn, **model_args)


def _rows(n_total, n):
    if n <= 0 or n >= n_total:
        return list(range(n_total))
    return sorted(set(int(round(x)) for x in np.linspace(0, n_total - 1, n)))


def load_bench(name, n):
    """[{image, question, id}] for evenly spaced rows, prompts from the lmms-eval tasks."""
    if name == "vstar":
        from datasets import load_dataset
        from lmms_eval.tasks.vstar_bench.utils import vstar_doc_to_text, vstar_doc_to_visual
        ds = load_dataset("lmms-lab/vstar-bench", split="test")
        rows = _rows(len(ds), n)
        kw = {"pre_prompt": "", "post_prompt": VSTAR_POST}
        return [{"image": vstar_doc_to_visual(ds[i])[0], "question": vstar_doc_to_text(ds[i], kw), "id": i}
                for i in rows]
    if name in ("hrbench4k", "hrbench8k"):
        from datasets import load_dataset
        from lmms_eval.tasks.hrbench.utils import hrbench_doc_to_text, hrbench_doc_to_visual
        split = "hrbench_4k" if name == "hrbench4k" else "hrbench_8k"
        ds = load_dataset("DreamMr/HR-Bench", "hrbench_version_split", split=split)
        rows = _rows(len(ds), n)
        return [{"image": hrbench_doc_to_visual(ds[i])[0], "question": hrbench_doc_to_text(ds[i]), "id": i}
                for i in rows]
    if os.path.isdir(os.path.expanduser(name)):
        from PIL import Image
        d = os.path.expanduser(name)
        files = sorted(f for f in os.listdir(d) if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png", ".webp"})
        files = [files[i] for i in _rows(len(files), n)]
        return [{"image": Image.open(os.path.join(d, f)).convert("RGB"),
                 "question": "What is shown in the image?", "id": f} for f in files]
    raise SystemExit(f"[cost] unknown dataset {name} (vstar, hrbench4k, hrbench8k, or an image folder)")


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def summarize(recs):
    keys_ms = sorted({k for r in recs for k in r["phase_ms"]})
    keys_gms = sorted({k for r in recs for k in r["group_ms"]})
    out = {
        "n": len(recs),
        "latency_s": _mean([r["latency_s"] for r in recs]),
        "latency_s_p50": float(np.median([r["latency_s"] for r in recs])),
        "phase_ms": {k: _mean([r["phase_ms"].get(k, 0.0) for r in recs]) for k in keys_ms},
        "group_ms": {k: _mean([r["group_ms"].get(k, 0.0) for r in recs]) for k in keys_gms},
        "visual_tokens": _mean([r["visual_tokens"] for r in recs]),
        "gen_tokens": _mean([r["gen_tokens"] for r in recs]),
        "prompt_tokens": _mean([r["prompt_tokens"] for r in recs]),
    }
    for k in ("lr_tokens", "win_tokens"):
        if any(k in r for r in recs):
            out[k] = _mean([r.get(k) for r in recs])
    dec = [r["phase_ms"].get("decode", 0.0) / max(r["gen_tokens"] - 1, 1) for r in recs if r["gen_tokens"] > 1]
    out["ms_per_decode_tok"] = _mean(dec)
    if any("flops_T" in r for r in recs):
        fr = [r for r in recs if "flops_T" in r]
        out["flops_T"] = _mean([r["flops_T"] for r in fr])
        out["gdn_est_T"] = _mean([r["gdn_est_T"] for r in fr])
        out["phase_flops_T"] = {k: _mean([r["phase_flops_T"].get(k, 0.0) for r in fr])
                                for k in sorted({k for r in fr for k in r["phase_flops_T"]})}
        out["group_flops_T"] = {k: _mean([r["group_flops_T"].get(k, 0.0) for r in fr])
                                for k in sorted({k for r in fr for k in r["group_flops_T"]})}
        out["linear_check_min"] = min(r["linear_check"] for r in fr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="row label in the table, e.g. 'Native (LR)'")
    ap.add_argument("--model", required=True)
    ap.add_argument("--model_args", default="")
    ap.add_argument("--datasets", nargs="+", default=["vstar", "hrbench4k", "hrbench8k"])
    ap.add_argument("--max_samples", type=int, default=50, help="rows per dataset, evenly spaced; <= 0: all")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=2, help="untimed runs per dataset before timing")
    ap.add_argument("--no_flops", action="store_true")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True, help="jsonl; one line per run is appended")
    args = ap.parse_args()
    if args.max_new_tokens < 1:
        raise SystemExit("--max_new_tokens must be >= 1")
    model_args = _parse_args_string(args.model_args)

    from torch.utils.flop_counter import FlopCounterMode

    counter = {"fc": None}
    print(f"[cost] {args.name}: {args.model} {model_args}", flush=True)
    attn_seen = None
    if not args.no_flops:
        attn_seen = sdpa_check(args.device)
        print(f"[cost] SDPA FLOPs check on {args.device}: counter saw {attn_seen:.3f} of the expected", flush=True)
        if attn_seen < 0.99:
            print("[cost] WARNING: attention FLOPs are not counted on this device; FLOPs miss attention",
                  flush=True)
    model = build_model(args.model, model_args, args.device, args.attn)
    roots = list({id(m): m for m in model.groups.values() if m is not None}.values())
    extras = [FlopExtras(m) for m in roots]

    def flops_now():
        fc = counter["fc"]
        if fc is None:
            return 0.0
        return float(fc.get_total_flops()) + sum(e.gdn for e in extras)

    clock = Clock(flops_now)
    probe = GroupProbe(model.groups, flops_now)

    def call(s, count=False):
        for e in extras:
            e.reset()
            e.active = count
        probe.reset()
        if count:
            with FlopCounterMode(display=False) as fc:
                counter["fc"] = fc
                try:
                    clock.start()
                    res = model.run(s["image"], s["question"], args.max_new_tokens, clock)
                    total = flops_now()
                finally:
                    counter["fc"] = None
        else:
            clock.start()
            res = model.run(s["image"], s["question"], args.max_new_tokens, clock)
            total = 0.0
        gms, gfl = probe.result()
        for e in extras:
            e.active = False
        return res, dict(clock.ms), dict(clock.flops), gms, gfl, total

    per_ds, all_recs = {}, []
    t_run = time.time()
    for ds in args.datasets:
        samples = load_bench(ds, args.max_samples)
        print(f"[cost] {ds}: {len(samples)} samples, warmup {args.warmup}", flush=True)
        for s in samples[:max(args.warmup, 0)]:
            call(s)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        recs = []
        for i, s in enumerate(samples):
            res, ms, _, gms, _, _ = call(s)
            if res is None:
                print(f"[cost] {ds} {s['id']}: no output, skipped", flush=True)
                continue
            rec = {"dataset": ds, "id": s["id"], "latency_s": sum(ms.values()) / 1000.0,
                   "phase_ms": ms, "group_ms": gms,
                   "image_size": list(s["image"].size), "text": str(res.get("text", ""))[:200]}
            rec.update({k: v for k, v in res.items() if k != "text"})
            if not args.no_flops:
                res_c, _, pfl, _, gfl, total = call(s, count=True)
                lin = sum(e.linear for e in extras)
                gdn = sum(e.gdn for e in extras)
                rec.update(flops_T=total / 1e12, gdn_est_T=gdn / 1e12,
                           phase_flops_T={k: v / 1e12 for k, v in pfl.items()},
                           group_flops_T={k: v / 1e12 for k, v in gfl.items()},
                           linear_check=(total - gdn) / lin if lin else 1.0)
                if res_c and res_c.get("gen_tokens") != res.get("gen_tokens"):
                    print(f"[cost] {ds} {s['id']}: counted run generated {res_c.get('gen_tokens')} "
                          f"tokens, timed run {res.get('gen_tokens')}", flush=True)
                if rec["linear_check"] < 0.99:
                    print(f"[cost] {ds} {s['id']}: counter saw {rec['linear_check']:.3f} of the nn.Linear "
                          f"FLOPs; some kernel is invisible to it", flush=True)
            recs.append(rec)
            fl = f"  {rec['flops_T']:.2f} TFLOPs" if "flops_T" in rec else ""
            print(f"[cost] {ds} {i + 1}/{len(samples)}  {rec['latency_s']:.2f} s  "
                  f"V {rec['visual_tokens']}  G {rec['gen_tokens']}{fl}", flush=True)
        if not recs:
            raise SystemExit(f"[cost] {ds}: no successful samples")
        summ = summarize(recs)
        if torch.cuda.is_available():
            summ["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 1024 ** 3
        per_ds[ds] = summ
        all_recs += recs
        fl = f"  {summ['flops_T']:.2f} TFLOPs" if "flops_T" in summ else ""
        print(f"[cost] {ds} done: {summ['latency_s']:.2f} s (p50 {summ['latency_s_p50']:.2f})  "
              f"V {summ['visual_tokens']:.0f}  G {summ['gen_tokens']:.1f}{fl}", flush=True)

    try:
        import transformers
        tf_version = transformers.__version__
    except ImportError:
        tf_version = None
    result = {
        "name": args.name, "model": args.model, "model_args": model_args,
        "datasets": per_ds, "max_samples": args.max_samples, "max_new_tokens": args.max_new_tokens,
        "warmup": args.warmup, "attn": args.attn, "dtype": "bfloat16",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__, "transformers": tf_version,
        "flops_counted": not args.no_flops, "sdpa_check": attn_seen, "wall_min": (time.time() - t_run) / 60.0,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "a") as f:
        f.write(json.dumps(result) + "\n")
    samples_path = out[:-len(".jsonl")] if out.endswith(".jsonl") else out
    safe = "".join(c if c.isalnum() else "_" for c in args.name).strip("_")
    with open(f"{samples_path}.{safe}.samples.jsonl", "w") as f:
        for r in all_recs:
            f.write(json.dumps(r) + "\n")
    print(f"[cost] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
