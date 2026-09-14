#!/usr/bin/env python3
"""Offset / deformable-sampling behavioural probe for Qwen3.5-DAT (inference-only).

Settles the ONE thing weights + logs + code-audit could not: when only the
QUESTION changes (same image), do the sampling points move, and is that motion
SEMANTIC or just an untrained-random projection of the question embedding?

Supports both question-conditioning paths: the legacy proj_intention gate and
the xattn QuestionReadout residual. The null condition hooks whichever path is
active and zeroes exactly its question-driven output.

For each image we run the SAME real HD image under text variants and record the
realised sampling locations per DAT layer:
  real      the sample's real question + options
  shuf      same question with its WORDS shuffled (semantics destroyed, lexicon kept)
  rand      a different sample's question glued onto the same options (unrelated)
  randfull  the entire neighbouring prompt (question and options)
  null      real prompt with the active question->offset module output zeroed

Per-layer metrics (all in units of the reference-grid cell pitch):
  off_mag        mean |slocs_real - reference|         (is there any offset at all)
  d(real,null)   mean per-point ||real - null||        (TOTAL question effect)
  d(real,shuf)   mean per-point ||real - shuf||         (semantic effect, upper bnd)
  d(real,rand)   mean per-point ||real - rand||
  d(shuf,null), d(rand,null)                            (reference legs)

Verdict logic (the DECISIVE leg is content sensitivity d(real,randfull), NOT
the shuffle leg — swapping to an unrelated prompt is what a semantic pointer
must react to):
  d(real,null) ~ 0                         -> points frozen wrt question signal.
  d(real,null) > 0 AND
     d(real,randfull) ~ d(real,rand) ~ 0 << d(real,null)
                                           -> moves mainly with question PRESENCE,
                                              blind to CONTENT.
  d(real,randfull) ~ d(real,null) (large)  -> an unrelated prompt moves points as
                                              much as removing the question =
                                              genuine semantic conditioning.

Reuses probe_whd_qwen35's proven data / geometry / build_inputs harness.

Usage (single GPU, on the eval cluster; offline pod: source scripts/eval_pod_env.sh):
  CUDA_VISIBLE_DEVICES=0 python scripts/probe_offset_qwen35.py \
      --model_path .../0901_sft_qwen35_4b_dat_ivcap_ee6_op10_kv_g20-merged \
      --dataset hrbench4k --max_samples 64 --tok_budget 1764 \
      --out _hr_path_check/offset_probe_g20.json
"""

import argparse
import json
import math
import random
from collections import defaultdict

import numpy as np
import torch

# Reuse the exact, proven harness from the sibling probe (same dir on sys.path[0]).
from probe_whd_qwen35 import (
    TOK_PX, load_samples, extract_letter, build_inputs,
)


# ──────────────────────────────────────────────────────────────
# Hooks: (1) record realised sampling locations per DAT layer;
#        (2) optionally zero the question->offset path (ablation).
#            intention-branch ckpts: zero proj_intention (gate=0.5).
#            xattn ckpts: zero the QuestionReadout residual (its output),
#            which is the ONLY question->offset path when
#            use_intention_branch=False.
# ──────────────────────────────────────────────────────────────
SLOCS = {}                                    # layer_idx -> np.array [gs, gs, 2]  (first call this gen)
STATE = {"record": False, "ablate_intention": False}


def _reduce_slocs(slocs):
    """[Lp, off_grps, gs, gs, 2] -> [gs, gs, 2] averaged over answer-range & group."""
    a = slocs.detach().float().cpu().numpy()
    gs = a.shape[-2]
    return a.reshape(-1, gs, gs, 2).mean(axis=0)


def install_hooks(model):
    from llava.model.language_model.modeling_qwen3_5_dat import Qwen3_5AttentionDAT

    # (1) capture the slocs returned by _generate_offsets_and_sample (first call
    #     per layer per generation = the prefill/answer-position offsets).
    orig = Qwen3_5AttentionDAT._generate_offsets_and_sample

    def patched(self, *a, **kw):
        ret = orig(self, *a, **kw)
        slocs = ret[2]                        # slocs_first
        if STATE["record"] and slocs is not None and self.layer_idx not in SLOCS:
            SLOCS[self.layer_idx] = _reduce_slocs(slocs)
        return ret

    Qwen3_5AttentionDAT._generate_offsets_and_sample = patched

    # (2) forward-hook the question->offset path: zero its output on demand.
    #     intention-branch ckpts: zero proj_intention (gate=sigmoid(0)=0.5).
    #     xattn ckpts: zero the q_readout residual (the returned tensor IS
    #     layerscale*delta, i.e. exactly the question-driven part of off_guide).
    n_hooked = 0
    for mod in model.modules():
        if isinstance(mod, Qwen3_5AttentionDAT):
            if getattr(mod, "q_readout", None) is not None:
                mod.q_readout.register_forward_hook(
                    lambda m, inp, out: torch.zeros_like(out) if STATE["ablate_intention"] else out
                )
                n_hooked += 1
            elif getattr(mod, "proj_intention", None) is not None:
                mod.proj_intention.register_forward_hook(
                    lambda m, inp, out: torch.zeros_like(out) if STATE["ablate_intention"] else out
                )
                n_hooked += 1
    return n_hooked


# ──────────────────────────────────────────────────────────────
# Question variants (same image; only the text prompt changes)
# ──────────────────────────────────────────────────────────────
def _split_prompt(prompt):
    """Return (question_line, rest) so we can rewrite just the question."""
    if "\n" in prompt:
        q, rest = prompt.split("\n", 1)
        return q, "\n" + rest
    return prompt, ""


def make_prompt(samples, idx, variant, rng):
    p = samples[idx]["prompt"]
    if variant == "real":
        return p
    q, rest = _split_prompt(p)
    if variant == "shuf":
        words = q.split()
        rng.shuffle(words)
        return " ".join(words) + rest
    if variant == "rand":
        oq, _ = _split_prompt(samples[(idx + 1) % len(samples)]["prompt"])
        return oq + rest                       # unrelated question, SAME options
    if variant == "randfull":
        # entire neighbour prompt (question AND options) — kills the "shared
        # options common-mode" caveat: if points still don't move, the sampling
        # is blind to the whole textual query, not just its question line.
        return samples[(idx + 1) % len(samples)]["prompt"]
    raise ValueError(variant)


# ──────────────────────────────────────────────────────────────
# Reference grid + cell pitch (mirror _grid_generate, half-cell margin)
# ──────────────────────────────────────────────────────────────
def reference_grid(gs):
    m = 1.0 / max(gs - 1, 1)
    ax = np.linspace(-1.0 + m, 1.0 - m, gs, dtype=np.float64)
    gy, gx = np.meshgrid(ax, ax, indexing="ij")
    ref = np.stack([gx, gy], axis=-1)          # [gs, gs, 2] = (x, y), matches slocs last dim
    pitch = float(ax[1] - ax[0]) if gs > 1 else 1.0
    return ref, pitch


def _per_point_dist(a, b, pitch):
    """mean over grid points of ||a-b|| in cell-pitch units."""
    return float(np.linalg.norm(a - b, axis=-1).mean() / pitch)


# ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--processor_path", default=None)
    ap.add_argument("--dataset", choices=["hrbench4k", "hrbench8k"], default="hrbench4k")
    ap.add_argument("--image_folder", default=None)
    ap.add_argument("--tok_budget", type=int, default=1764,
                    help="LR LLM-visual-token budget (lr_max_pixels = tok*1024)")
    ap.add_argument("--min_pixels", type=int, default=28224)
    ap.add_argument("--hd_cap", type=int, default=5017600)
    ap.add_argument("--hr_scale", type=int, default=3)
    ap.add_argument("--hd_source", default="real")     # offsets probe always uses the real HD image
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--max_samples", type=int, default=64)
    ap.add_argument("--max_new_tokens", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoProcessor
    from llava.model.language_model.modeling_qwen3_5_dat import (
        Qwen3_5DATForConditionalGeneration,
    )

    samples = load_samples(args)
    has_gt = bool(samples) and samples[0]["gt"] is not None
    print(f"[offset] {len(samples)} samples  tok_budget={args.tok_budget}  "
          f"dataset={args.image_folder or args.dataset}")

    model = Qwen3_5DATForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation=args.attn,
    ).eval()
    ppath = args.processor_path or args.model_path
    processor = AutoProcessor.from_pretrained(
        ppath, min_pixels=args.min_pixels, max_pixels=args.tok_budget * TOK_PX)
    hr_processor = AutoProcessor.from_pretrained(ppath, min_pixels=TOK_PX,
                                                 max_pixels=100_000_000)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    n_hooked = install_hooks(model)
    # config sanity: which offset paths are structurally live in this ckpt?
    dat_mod = next(m for m in model.modules()
                   if type(m).__name__ == "Qwen3_5AttentionDAT")
    cfg = {k: getattr(dat_mod, k, None) for k in
           ("grid_size", "off_grps", "off_range", "off_penalty",
            "use_intention_branch", "intention_as_gate",
            "use_spatial_attn_guide", "intention_inject", "question_inject")}
    print(f"[offset] hooked {n_hooked} DAT layers | dat cfg: {cfg}")
    if not (cfg["use_intention_branch"] and cfg["intention_as_gate"]) \
            and cfg.get("question_inject") != "xattn":
        print("[offset][warn] no question->offset path active in this ckpt; "
              "the 'null' ablation is a no-op.")

    gen_kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=False,
                      temperature=None, top_p=None, top_k=None)
    rng = random.Random(args.seed)

    VARIANTS = ["real", "shuf", "rand", "randfull"]  # 'null' = real prompt + intention ablation
    # per-layer accumulators for pairwise per-point distances (cell units)
    PAIRS = ["off_mag", "d_real_null", "d_real_shuf", "d_real_rand", "d_real_randfull",
             "d_shuf_null", "d_rand_null", "d_randfull_null"]
    acc = defaultdict(lambda: defaultdict(list))   # layer -> metric -> [per-image value]
    pred = defaultdict(list)                        # variant -> [letters]
    ref_cache = {}

    def run(prompt_text, ablate):
        """Generate once for a (prompt, ablation) and return {layer: slocs[gs,gs,2]}, letter."""
        SLOCS.clear()
        s = dict(samples[i]); s["prompt"] = prompt_text
        base_inputs, hd_extra = build_inputs(
            s, i, samples, processor, hr_processor, args, device, dtype)
        STATE["record"] = True
        STATE["ablate_intention"] = ablate
        with torch.inference_mode():
            out = model.generate(**{**base_inputs, **hd_extra}, **gen_kwargs)
        STATE["record"] = False
        STATE["ablate_intention"] = False
        n_in = base_inputs["input_ids"].shape[1]
        letter = extract_letter(
            processor.tokenizer.decode(out[0][n_in:], skip_special_tokens=True))
        return dict(SLOCS), letter

    from tqdm import tqdm
    for i in tqdm(range(len(samples)), desc="offset"):
        slocs = {}
        for v in VARIANTS:
            slocs[v], letter = run(make_prompt(samples, i, v, rng), ablate=False)
            pred[v].append(letter)
        slocs["null"], letter = run(samples[i]["prompt"], ablate=True)
        pred["null"].append(letter)

        # layers present in ALL conditions
        conds = ("real", "shuf", "rand", "randfull", "null")
        layers = set.intersection(*(set(slocs[v]) for v in conds)) \
            if all(slocs[v] for v in conds) else set()
        for lid in layers:
            gs = slocs["real"][lid].shape[0]
            if gs not in ref_cache:
                ref_cache[gs] = reference_grid(gs)
            ref, pitch = ref_cache[gs]
            R, S, D, F, N = (slocs["real"][lid], slocs["shuf"][lid],
                             slocs["rand"][lid], slocs["randfull"][lid],
                             slocs["null"][lid])
            acc[lid]["off_mag"].append(_per_point_dist(R, ref, pitch))
            acc[lid]["d_real_null"].append(_per_point_dist(R, N, pitch))
            acc[lid]["d_real_shuf"].append(_per_point_dist(R, S, pitch))
            acc[lid]["d_real_rand"].append(_per_point_dist(R, D, pitch))
            acc[lid]["d_real_randfull"].append(_per_point_dist(R, F, pitch))
            acc[lid]["d_shuf_null"].append(_per_point_dist(S, N, pitch))
            acc[lid]["d_rand_null"].append(_per_point_dist(D, N, pitch))
            acc[lid]["d_randfull_null"].append(_per_point_dist(F, N, pitch))

    # ── report ────────────────────────────────────────────────
    print(f"\n==== offset motion per DAT layer  (cell-pitch units, N={len(samples)}) ====")
    hdr = f"{'layer':>5} | " + " ".join(f"{m:>12}" for m in PAIRS)
    print(hdr); print("-" * len(hdr))
    layer_stats = {}
    for lid in sorted(acc):
        row = {m: (float(np.mean(acc[lid][m])) if acc[lid][m] else float("nan"))
               for m in PAIRS}
        layer_stats[lid] = row
        print(f"{lid:>5} | " + " ".join(f"{row[m]:>12.4f}" for m in PAIRS))

    # aggregate verdict across layers
    if layer_stats:
        agg = {m: float(np.mean([layer_stats[l][m] for l in layer_stats])) for m in PAIRS}
        print("\n==== aggregate (mean over layers) ====")
        for m in PAIRS:
            print(f"  {m:>12} = {agg[m]:.4f} cell")
        moves = agg["d_real_null"]            # question ON/OFF switch effect
        content = agg["d_real_randfull"]      # content sensitivity (whole prompt swapped)
        print("\n---- read ----")
        if moves < 0.02:
            print(f"  points ~FROZEN wrt question signal (d_real_null={moves:.4f} cell).")
        else:
            print(f"  points MOVE with question SIGNAL (d_real_null={moves:.4f} cell).")
            # DECISIVE discriminator is CONTENT sensitivity, not the shuffle leg:
            # genuine semantic conditioning REQUIRES that swapping the whole prompt
            # to an unrelated one moves the points (d_real_randfull large). If a
            # different prompt barely moves them, the motion is a bare ON/OFF
            # switch = semantics-blind (consistent with untrained proj_intention).
            if content >= 0.6 * moves:
                print(f"  d_real_randfull({content:.4f}) ~ d_real_null({moves:.4f}): "
                      f"an unrelated prompt moves points as much as removing the "
                      f"question -> genuine SEMANTIC conditioning.")
            else:
                print(f"  d_real_randfull({content:.4f}) << d_real_null({moves:.4f}): "
                      f"an unrelated prompt barely moves points -> SEMANTICS-BLIND "
                      f"(points respond to question PRESENCE, not CONTENT).")

    if args.out:
        json.dump({
            "model_path": args.model_path,
            "dataset": args.image_folder or args.dataset,
            "tok_budget": args.tok_budget,
            "num_samples": len(samples),
            "dat_cfg": {k: (v if isinstance(v, (int, float, bool, str)) or v is None
                            else str(v)) for k, v in cfg.items()},
            "layer_stats": layer_stats,
            "aggregate": (agg if layer_stats else None),
            "pred_agreement": {
                v: (sum(pred[v][k] == pred["real"][k] for k in range(len(samples)))
                    / max(len(samples), 1))
                for v in ("shuf", "rand", "randfull", "null")
            },
        }, open(args.out, "w"), indent=2, ensure_ascii=False)
        print(f"[offset] saved -> {args.out}")


if __name__ == "__main__":
    main()
