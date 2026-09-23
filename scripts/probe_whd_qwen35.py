#!/usr/bin/env python3
"""HD-pathway diagnostics for Qwen3.5-DAT checkpoints (inference-only).

Runs each HR-Bench sample under several HD configurations and reports
accuracy (overall + per category FSP/FCP), answer flips vs HD-off, and the
per-layer LSE merge weight w_hd = exp(lse_hd - logaddexp(lse_lr, lse_hd)).

Configs (all share the same LR input; only the HD side changes):
  off               no pixel_values_hd at all (vanilla single-pass attention)
  <source> b=<bias> HD pass on, with
      --hd_source  real     the real image at HD resolution (default)
                   lr_up    LR-resized image upsampled back to HD size
                            (same key count, ZERO extra information)
                   shuffle  a different sample's image (wrong content)
                   noise    uniform random noise
      --hd_bias    constant(s) added to lse_hd before the merge; >0 pushes
                   attention toward HD, <0 away. Pass several values to
                   sweep, e.g. --hd_bias -6 -3 0 1.5 3

Diagnosis A (bias sweep, real source): does any b>0 help? -> HD content is
  useful but mis-weighted (gate fix suffices). Does only b<0 help? -> HD
  content is noise (feature-side problem).
Diagnosis B (source ablation, b=0): lr_up/shuffle/noise ~= real -> the model
  treats HD as a content-free attention sink, w_hd is scale-driven.

Prompt / option formatting / GT match lmms-eval's hrbench task exactly.

Usage (single GPU, on the eval cluster):
  export HF_ENDPOINT=https://hf-mirror.com
  CUDA_VISIBLE_DEVICES=0 python scripts/probe_whd_qwen35.py \
      --model_path /data/oss_bucket_0/wangziyi/vldat_experiments/0826_sft_qwen35_2b_dat_genvs-merged \
      --dataset hrbench4k --max_samples 200 --tok_budget 256 \
      --hd_source real --hd_bias -6 -3 0 1.5 3 --out diagA_hr4k_tok256.json
"""

import argparse
import base64
import glob
import io
import json
import math
import os
import re
import string
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

PATCH = 16
MERGE = 2
FACTOR = PATCH * MERGE          # 32
TOK_PX = FACTOR * FACTOR        # 1024

SYSTEM_PROMPT = "You are a helpful assistant."
LETTER_RE = re.compile(r"\b([A-H])\b")


# ──────────────────────────────────────────────────────────────
# Data (mirrors lmms_eval/tasks/hrbench/utils.py)
# ──────────────────────────────────────────────────────────────

def _is_nan(v):
    return v is None or (isinstance(v, float) and math.isnan(v))


def load_samples(args):
    samples = []
    if args.image_folder:
        exts = {".jpg", ".jpeg", ".png", ".webp"}
        files = sorted(f for f in os.listdir(args.image_folder)
                       if os.path.splitext(f)[1].lower() in exts)[: args.max_samples]
        for f in files:
            samples.append({
                "image": Image.open(os.path.join(args.image_folder, f)).convert("RGB"),
                "prompt": "Describe this image briefly.",
                "gt": None, "category": "n/a",
            })
        return samples

    if args.dataset == "vstar":
        return load_vstar(args)
    if args.dataset == "synth":
        return load_synth(args)

    from datasets import load_dataset
    split = {"hrbench4k": "hrbench_4k", "hrbench8k": "hrbench_8k"}[args.dataset]
    # Prefer the parquet already sitting in the HF hub cache (or an explicit
    # --hrbench_parquet): load_dataset("DreamMr/HR-Bench") always phones the
    # Hub to resolve the repo first, which hangs for HF_HUB_DOWNLOAD_TIMEOUT
    # when the mirror is slow and fails outright under HF_HUB_OFFLINE=1 once
    # the ~/.cache/huggingface/datasets arrow cache has been wiped.
    pq = args.hrbench_parquet or _local_hrbench_parquet(split)
    if pq:
        print(f"[probe] HR-Bench from local parquet: {pq}")
        ds = load_dataset("parquet", data_files=pq, split="train")
    else:
        print("[probe] HR-Bench parquet not in local hub cache; resolving via the Hub")
        ds = load_dataset("DreamMr/HR-Bench", "hrbench_version_split", split=split)
    for doc in ds:
        if len(samples) >= args.max_samples:
            break
        img = Image.open(io.BytesIO(base64.b64decode(doc["image"]))).convert("RGB")
        options = {c: doc[c] for c in string.ascii_uppercase
                   if c in doc and not _is_nan(doc[c])}
        opt_txt = "".join(f"{k}. {v}\n" for k, v in options.items())
        prompt = f"{doc['question'].strip()}\n{opt_txt}Answer the option letter directly."
        samples.append({
            "image": img, "prompt": prompt,
            "gt": str(doc["answer"]).strip().upper(),
            "category": str(doc.get("category", "n/a")),
        })
    return samples


def load_vstar(args):
    """V* Bench from the craigwu/vstar_bench layout: <root>/test_questions.jsonl
    plus <root>/<category>/<name>.jpg and a sidecar <name>.json holding
    ``target_object`` and ``bbox`` ([x, y, w, h] in original-image pixels).
    The bboxes are what the oracle sampling sources use."""
    root = os.path.expanduser(args.vstar_root)
    qfile = os.path.join(root, "test_questions.jsonl")
    if not os.path.isfile(qfile):
        raise FileNotFoundError(
            f"{qfile} not found. Download with:\n"
            "  HF_ENDPOINT=https://hf-mirror.com huggingface-cli download --repo-type dataset "
            f"craigwu/vstar_bench --local-dir {root}")
    samples = []
    with open(qfile) as f:
        for line in f:
            if len(samples) >= args.max_samples:
                break
            doc = json.loads(line)
            img_path = os.path.join(root, doc["image"])
            side = os.path.splitext(img_path)[0] + ".json"
            bboxes = []
            if os.path.isfile(side):
                bboxes = json.load(open(side)).get("bbox", []) or []
            samples.append({
                "image": Image.open(img_path).convert("RGB"),
                "prompt": doc["text"].strip(),
                "gt": str(doc["label"]).strip().upper(),
                "category": str(doc.get("category", "n/a")),
                "bboxes": [list(map(float, b)) for b in bboxes],
            })
    n_box = sum(1 for s in samples if s["bboxes"])
    print(f"[probe] V* Bench: {len(samples)} questions, {n_box} with target bboxes")
    return samples


def load_synth(args):
    """Held-out split written by scripts/build_synth_hd_text_data.py
    (<train>.json.heldout.json): small text pasted on high-res images, with
    `bbox` in image fractions. gt is the text; scoring = normalized containment.
    The text is sized to be unreadable at the LR budget and readable at HD, so
    HD-on vs off here is a direct unit test of the readout; oracle vs
    oracle_rand tests whether it reads the *window* or just any HD content."""
    if not args.synth_json:
        raise SystemExit("--dataset synth needs --synth_json <...heldout.json>")
    root = os.path.expanduser(args.synth_image_root or "")
    samples = []
    for doc in json.load(open(os.path.expanduser(args.synth_json)))[: args.max_samples]:
        img = Image.open(os.path.join(root, doc["image"])).convert("RGB")
        W, H = img.size
        x0, y0, x1, y1 = doc["bbox"]
        if "conversations" in doc:      # LLaVA-format bbox data (build_viscot_bbox_data.py)
            question = doc["conversations"][0]["value"].replace("<image>", "").strip()
            answer = doc["conversations"][1]["value"]
            category = doc.get("source", "bbox")
        else:                            # build_synth_hd_text_data.py heldout format
            question, answer = doc["question"].strip(), doc["answer"]
            category = "code" if any(ch.isdigit() for ch in answer) else "word"
        samples.append({
            "image": img,
            "prompt": question,
            "gt": norm_text(answer),
            "category": category,
            "bboxes": [[x0 * W, y0 * H, (x1 - x0) * W, (y1 - y0) * H]],   # V* style [x, y, w, h] px
        })
    print(f"[probe] synth text: {len(samples)} samples from {args.synth_json}")
    return samples


def norm_text(t):
    return re.sub(r"[^a-z0-9]", "", t.lower())


def _local_hrbench_parquet(split):
    """Locate hr_bench_{4k,8k}.parquet inside the HF hub cache, if downloaded."""
    fname = {"hrbench_4k": "hr_bench_4k.parquet", "hrbench_8k": "hr_bench_8k.parquet"}[split]
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    hits = sorted(glob.glob(os.path.join(hub, "datasets--DreamMr--HR-Bench", "snapshots", "*", fname)),
                  key=os.path.getmtime)
    return hits[-1] if hits else None


def extract_letter(text):
    m = LETTER_RE.search(text.strip().upper())
    return m.group(1) if m else "Z"


# ──────────────────────────────────────────────────────────────
# Geometry (LR-first, identical to training / lmms-eval wrapper)
# ──────────────────────────────────────────────────────────────

def hd_target_size(image, lr_grid_thw, hr_scale, hd_cap):
    lr_px = int(lr_grid_thw[1]) * FACTOR * int(lr_grid_thw[2]) * FACTOR
    hd_total = min(lr_px * hr_scale * hr_scale, image.width * image.height, hd_cap)
    aspect = image.width / image.height
    hd_h = int(math.sqrt(hd_total / aspect))
    hd_w = int(hd_h * aspect)
    hd_h = max(FACTOR, (hd_h // FACTOR) * FACTOR)
    hd_w = max(FACTOR, (hd_w // FACTOR) * FACTOR)
    return hd_w, hd_h


def make_hd_image(sample, idx, samples, lr_thw, hd_w, hd_h, source):
    """Build the HD-side image under the requested ablation source."""
    img = sample["image"]
    if source in ("real", "oracle", "oracle_rand"):
        return img.resize((hd_w, hd_h), Image.BICUBIC)
    if source == "lr_up":
        # exactly what the LR branch sees (patch grid * 16 px), blown back up
        lr_w_px = int(lr_thw[2]) * PATCH
        lr_h_px = int(lr_thw[1]) * PATCH
        return (img.resize((lr_w_px, lr_h_px), Image.BICUBIC)
                   .resize((hd_w, hd_h), Image.BICUBIC))
    if source == "shuffle":
        # HR-Bench stores the 4 cyclic option permutations of one question as
        # CONSECUTIVE rows sharing the same image, so idx+1 returns the SAME
        # image 3 times out of 4. Jump half-way round and skip identical images.
        n = len(samples)
        for step in range(n // 2, n // 2 + 8):
            other = samples[(idx + step) % n]["image"]
            if other.size != img.size or other.tobytes()[:4096] != img.tobytes()[:4096]:
                return other.resize((hd_w, hd_h), Image.BICUBIC)
        raise RuntimeError(f"shuffle: could not find a different image for sample {idx}")
    if source == "noise":
        rng = np.random.RandomState(idx)
        arr = rng.randint(0, 256, size=(hd_h, hd_w, 3), dtype=np.uint8)
        return Image.fromarray(arr, "RGB")
    raise ValueError(source)


def oracle_locs(sample, idx, hd_w, hd_h, grid_size, min_cells, source):
    """Forced sampling grid for the oracle sources, as [Ns, 2] (x, y) in [-1, 1].

    oracle:      grid_size x grid_size uniform points over a window centred on
                 the union of the GT target bboxes, at least `min_cells` HD
                 feature cells (32 px of the HD image) per side so that a tiny
                 target still comes with context and the window is sampled at
                 full HD resolution.
    oracle_rand: a window of the SAME size at a random location whose window
                 does not overlap the GT bbox — controls for "any focused
                 window helps" vs "the right location helps".
    Returns None when the sample has no bbox (falls back to learned offsets).
    """
    boxes = sample.get("bboxes") or []
    if not boxes:
        return None
    W, H = sample["image"].size
    x0 = min(b[0] for b in boxes) / W
    y0 = min(b[1] for b in boxes) / H
    x1 = max(b[0] + b[2] for b in boxes) / W
    y1 = max(b[1] + b[3] for b in boxes) / H
    # window size in normalised [0, 1] image fractions
    ww = max(x1 - x0, min_cells * FACTOR / hd_w)
    wh = max(y1 - y0, min_cells * FACTOR / hd_h)
    ww, wh = min(ww, 1.0), min(wh, 1.0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if source == "oracle_rand":
        rng = np.random.RandomState(1000 + idx)
        best, best_d = None, -1.0
        for _ in range(64):
            rx, ry = rng.uniform(ww / 2, 1 - ww / 2), rng.uniform(wh / 2, 1 - wh / 2)
            # no overlap between the random window and the GT union box
            if (abs(rx - cx) >= (ww + (x1 - x0)) / 2) or (abs(ry - cy) >= (wh + (y1 - y0)) / 2):
                best = (rx, ry)
                break
            d = (rx - cx) ** 2 + (ry - cy) ** 2
            if d > best_d:
                best, best_d = (rx, ry), d
        cx, cy = best
    # clamp the window inside the image
    cx = min(max(cx, ww / 2), 1 - ww / 2)
    cy = min(max(cy, wh / 2), 1 - wh / 2)
    gx = torch.linspace(cx - ww / 2, cx + ww / 2, grid_size)
    gy = torch.linspace(cy - wh / 2, cy + wh / 2, grid_size)
    gy, gx = torch.meshgrid(gy, gx, indexing="ij")           # row-major like _grid_generate
    locs = torch.stack([gx, gy], dim=-1).reshape(-1, 2) * 2.0 - 1.0
    return locs.clamp(-1.0, 1.0)


def set_force_locs(model, locs):
    from llava.model.language_model.modeling_qwen3_5_dat import Qwen3_5AttentionDAT
    for m in model.modules():
        if isinstance(m, Qwen3_5AttentionDAT):
            m._dat_force_locs = locs


def build_inputs(sample, idx, samples, processor, hr_processor, args, device, dtype):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": sample["image"]},
            {"type": "text", "text": sample["prompt"]},
        ]},
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = processor(text=[text], images=[sample["image"]], return_tensors="pt")
    lr_thw = inputs["image_grid_thw"][0]

    hd_w, hd_h = hd_target_size(sample["image"], lr_thw, args.hr_scale, args.hd_cap)
    hd_img = make_hd_image(sample, idx, samples, lr_thw, hd_w, hd_h, args.hd_source)
    hr_inputs = hr_processor.image_processor(images=[hd_img], return_tensors="pt")

    moved = {}
    for k, v in inputs.items():
        if not isinstance(v, torch.Tensor):
            continue
        if k == "pixel_values":
            moved[k] = v.to(device=device, dtype=dtype)
        elif k in ("input_ids", "attention_mask", "image_grid_thw"):
            moved[k] = v.to(device)
    hd_extra = {
        "pixel_values_hd": hr_inputs["pixel_values"].to(device=device, dtype=dtype),
        "image_grid_thw_hd": hr_inputs["image_grid_thw"].to(device),
    }
    locs = None
    if args.hd_source.startswith("oracle"):
        locs = oracle_locs(sample, idx, hd_w, hd_h, args.grid_size,
                           args.oracle_min_cells, args.hd_source)
    # Localisation target for the learned grid (any source): the oracle window
    # grid (= the training target of --dat_off_sup_weight) and the raw GT box
    # in [-1, 1] coords. None when the sample has no bbox.
    STATE["target"] = None
    if sample.get("bboxes"):
        W, H = sample["image"].size
        bs = sample["bboxes"]
        box = torch.tensor([min(b[0] for b in bs) / W, min(b[1] for b in bs) / H,
                            max(b[0] + b[2] for b in bs) / W, max(b[1] + b[3] for b in bs) / H])
        STATE["target"] = (oracle_locs(sample, idx, hd_w, hd_h, args.grid_size,
                                       args.oracle_min_cells, "oracle"),
                           box * 2.0 - 1.0)
    return moved, hd_extra, locs


# ──────────────────────────────────────────────────────────────
# Merge hook: w_hd recorder + lse_hd bias injector
# ──────────────────────────────────────────────────────────────
WHD = defaultdict(list)          # layer_idx -> [(mean, p90, max), ...]
STATE = {"record": False, "bias": 0.0}


def install_merge_hook():
    from llava.model.language_model.modeling_qwen3_5_dat import Qwen3_5AttentionDAT
    import torch.nn.functional as F

    orig = Qwen3_5AttentionDAT._merge_two_pass_lse

    def patched(self, out1, lse1, out2, lse2, ans_start, ans_end):
        if STATE["bias"] != 0.0:
            lse2 = lse2 + STATE["bias"]
        if STATE["record"]:
            with torch.no_grad():
                l1 = lse1[:, :, ans_start:ans_end].float()
                l2 = lse2.float()
                if self.hd_gate is not None:
                    l2 = l2 + F.logsigmoid(self.hd_gate)
                w = (l2 - torch.logaddexp(l1, l2)).exp().flatten()
                WHD[self.layer_idx].append(
                    (w.mean().item(), w.quantile(0.9).item(), w.max().item()))
        return orig(self, out1, lse1, out2, lse2, ans_start, ans_end)

    Qwen3_5AttentionDAT._merge_two_pass_lse = patched

    # The trunk's OWN attention as a relevance map (diagnostic 2): at each DAT
    # layer, the intention token's attention over the LR image tokens (post-
    # RoPE q/k of the base path, all heads averaged, softmax restricted to the
    # LR tokens), pooled to the offset grid. Parameter-free reference for what
    # the learned glob_q/glob_k relevance could at best read off these features.
    import llava.model.language_model.modeling_qwen3_5_dat as _mod
    orig_rope = _mod.apply_rotary_pos_emb

    def patched_rope(q, k, cos, sin, *a, **kw):
        q, k = orig_rope(q, k, cos, sin, *a, **kw)
        if STATE["record"]:
            STATE["last_qk"] = (q, k)
        return q, k

    _mod.apply_rotary_pos_emb = patched_rope
    orig_gen = Qwen3_5AttentionDAT._generate_offsets_and_sample

    def patched_gen(self, query_states, image_hd_features, image_range_list, b_idx, hd_feat_idxs, *a, **kw):
        self._probe_attn_map = None
        self._probe_attn_heads = None
        self._probe_attn_alt = None
        qk = STATE.get("last_qk") if STATE["record"] else None
        if qk is not None and len(image_range_list[b_idx]) > 1 and getattr(self, "use_global_offset", False):
            with torch.no_grad():
                q, k = qk
                lr_start, lr_end, lr_h, lr_w = image_range_list[b_idx][0][0]
                idx = [ar[2] for ar in image_range_list[b_idx][1:]]
                kl = k[b_idx, :, lr_start:lr_end].float()                  # [Hk, N, D]
                kl = kl.repeat_interleave(q.size(1) // kl.size(0), dim=0)   # [H, N, D]
                gs = self.grid_size
                T = q.size(2)

                def heads_for(pos):
                    pos = [p for p in pos if 0 <= p < T]
                    if not pos:
                        return None
                    qi = q[b_idx, :, pos].float()                          # [H, L, D]
                    att = torch.einsum('hld,hnd->hln', qi, kl) / math.sqrt(qi.size(-1))
                    att = att.softmax(-1).mean(1)                           # [H, N] over LR tokens
                    hh = F.adaptive_avg_pool2d(att.view(-1, 1, lr_h, lr_w), (gs, gs)).flatten(1)
                    return hh / hh.sum(-1, keepdim=True).clamp_min(1e-9)    # [H, gs*gs], rows sum to 1

                heads = heads_for(idx)
                self._probe_attn_heads = heads
                self._probe_attn_map = heads.mean(0)                        # head-averaged, sums to 1
                # Diagnostic 4: which token should ask? Same maps with the query
                # taken at other positions around the assistant <|im_start|> (t):
                # the prompt ends "...question<|im_end|>\n<|im_start|>assistant\n",
                # so t-3 = last question token, t-2 = <|im_end|>, t-1 = \n,
                # t+1 = 'assistant', t+2 = its \n; q_mean = every question token.
                t = idx[-1]
                ar = image_range_list[b_idx][-1]
                q_start = ar[3] if len(ar) > 3 else lr_end
                alt = {"q_last": [t - 3], "im_end": [t - 2], "nl": [t - 1],
                       "asst": [t + 1], "asst_nl": [t + 2],
                       "q_mean": list(range(max(q_start, lr_end), max(q_start, lr_end, t - 2)))}
                self._probe_attn_alt = {kname: heads_for(p) for kname, p in alt.items()}
        return orig_gen(self, query_states, image_hd_features, image_range_list, b_idx, hd_feat_idxs, *a, **kw)

    Qwen3_5AttentionDAT._generate_offsets_and_sample = patched_gen

    # K_hd "sink-ness": how much of every HD key is a direction shared by all
    # Ns tokens (content-free, attracts the same logit for every query) versus
    # token-specific residual. ratio >> 1 means the HD keys are near-identical
    # and act as one big sink; ratio << 1 means they carry per-token content.
    # Also records how far sampling points moved off the uniform reference grid.
    orig_sample = Qwen3_5AttentionDAT._sample_hd_from_off_guide

    # Diagnostic 5 (--glob_readout / --glob_layers): swap the map -> (c, s)
    # readout at inference WITHOUT retraining. The model computes
    # c = E_p[g], s = std_p/std_u over the whole map; a map with 60% mass in
    # the window and 40% flat background gives s ~ 0.9 and a centroid pulled
    # to the image centre. 'floor:<lam>' subtracts lam/N (lam = 1: the uniform
    # level) from p and renormalises before the moments; layers not in
    # --glob_layers get a flat map (c = 0, s = 1, i.e. the global term off).
    # Implemented by intercepting the single torch.softmax call of
    # _sample_hd_from_off_guide (the others there are F.softmax / .softmax()).
    orig_softmax = torch.softmax

    def glob_softmax_for(layer_idx, gs):
        mode = STATE.get("glob_readout")
        layers = STATE.get("glob_layers")
        N = gs * gs

        def floor(p, lam):
            q = (p - lam / N).clamp_min(0.0)
            z = q.sum(-1, keepdim=True)
            return torch.where(z > 1e-9, q / z.clamp_min(1e-9), p)

        def argmax_window(p, k):
            # [R, N] mask of the k x k cell window centred on each row's argmax
            # (clipped at the borders, so c moves inward there: the grid must
            # stay inside the image anyway)
            am = p.argmax(-1)                                   # [R]
            ay, ax = am // gs, am % gs
            ii = torch.arange(gs, device=p.device)
            r = k // 2
            my = ((ii[None, :] - ay[:, None]).abs() <= r)       # [R, gs]
            mx = ((ii[None, :] - ax[:, None]).abs() <= r)
            return (my[:, :, None] & mx[:, None, :]).flatten(1).float()   # [R, N]

        def f(x, dim=-1, *aa, **kk):
            p = orig_softmax(x, dim, *aa, **kk)
            if x.dim() != 2 or x.size(-1) != N:
                return p
            if layers is not None and layer_idx not in layers:
                return torch.full_like(p, 1.0 / N)
            if not mode:
                return p
            kind, _, rest = mode.partition(":")
            if kind == "floor":                     # floor:<lam>
                return floor(p, float(rest))
            if kind == "win":                       # win:<k>  c = argmax cell, s ~ k/gs
                m = argmax_window(p, int(rest))
                return m / m.sum(-1, keepdim=True)
            if kind == "local":                     # local:<k>:<lam>  moments inside k x k around argmax
                k, lam = rest.split(":")
                q = floor(p, float(lam)) * argmax_window(p, int(k))
                z = q.sum(-1, keepdim=True)
                return torch.where(z > 1e-9, q / z.clamp_min(1e-9), p)
            raise ValueError(f"unknown --glob_readout {mode}")
        return f

    def patched_sample(self, *a, **kw):
        if STATE.get("glob_readout") or STATE.get("glob_layers") is not None:
            torch.softmax = glob_softmax_for(self.layer_idx, self.grid_size)
            try:
                key_hd, value_hd, locs = orig_sample(self, *a, **kw)
            finally:
                torch.softmax = orig_softmax
        else:
            key_hd, value_hd, locs = orig_sample(self, *a, **kw)
        if STATE["record"]:
            with torch.no_grad():
                k = key_hd.float()                       # [Lp, Ns, D]
                mu = k.mean(1, keepdim=True)
                shared = mu.norm(dim=-1).mean().item()
                resid = (k - mu).norm(dim=-1).mean().item()
                gh, gw = locs.shape[2], locs.shape[3]
                my, mx = 1.0 / max(gh - 1, 1), 1.0 / max(gw - 1, 1)
                gy, gx = torch.meshgrid(
                    torch.linspace(-1 + my, 1 - my, gh, device=locs.device),
                    torch.linspace(-1 + mx, 1 - mx, gw, device=locs.device),
                    indexing="ij")                       # same half-cell-margin grid as _grid_generate
                l = locs.float()
                ref = torch.stack([gx, gy], -1)              # [gh, gw, 2] (x, y)
                off = min((l - ref).abs().mean().item(),
                          (l - torch.stack([gy, gx], -1)).abs().mean().item())
                KHD[self.layer_idx].append((shared / max(resid, 1e-6), off))
                # dat_use_global_offset: (mean |grid centroid|, mean grid scale)
                # of this call; 0 / 1 = the global term is still the identity.
                gl = getattr(self, "_dat_glob_last", None)
                if gl is not None:
                    GLOB[self.layer_idx].append(tuple(gl.float().tolist()))
                # Localisation: did the learned grid move toward the GT region?
                #   dist   = mean point->target-grid distance (training's off_sup_dist)
                #   in_box = fraction of sampling points inside the raw GT box
                # each with the uniform reference grid as the "did not move" baseline.
                if STATE.get("target") is not None:
                    tgt, box = STATE["target"]
                    tgt = tgt.to(l.device).view(gh, gw, 2)
                    box = box.to(l.device)
                    pts = l.reshape(-1, gh, gw, 2)           # [Lp*G, gh, gw, 2]

                    def in_box(p):
                        return ((p[..., 0] >= box[0]) & (p[..., 0] <= box[2]) &
                                (p[..., 1] >= box[1]) & (p[..., 1] <= box[3])).float().mean().item()
                    LOC[self.layer_idx].append((
                        (pts - tgt).norm(dim=-1).mean().item(),
                        (ref - tgt).norm(dim=-1).mean().item(),
                        in_box(pts), in_box(ref)))
                    # Is the global term sample-dependent or a constant prior?
                    # Per sample: predicted grid centroid / scale (question slot
                    # rows only) vs the GT window's centre / scale.
                    cs = getattr(self, "_dat_glob_last_cs", None)
                    if cs is not None:
                        c, s = cs
                        G = self.off_grps
                        if c.size(0) > G:                      # drop the image-only lead slot
                            c, s = c[G:], s[G:]
                        t2 = tgt.reshape(-1, 2)
                        r2 = ref.reshape(-1, 2)
                        tc = t2.mean(0)
                        ts = t2.std(0, unbiased=False) / r2.std(0, unbiased=False)
                        GLOBC[self.layer_idx].append(
                            c.mean(0).tolist() + s.mean(0).tolist() + tc.tolist() + ts.tolist())
                    # The relevance MAP itself (before the soft-argmax): per
                    # sample, the question-slot rows' mean map over the N cells
                    # and the mask of cells inside the GT window (training's
                    # supervision region, +-half a cell like the trainer).
                    pg = getattr(self, "_dat_glob_last_p", None)
                    if pg is not None:
                        pm, gc = pg                               # [R, N], [2, N]
                        G = self.off_grps
                        if pm.size(0) > G:
                            pm = pm[G:]
                        t2 = tgt.reshape(-1, 2)
                        lo, hi = t2.min(0).values, t2.max(0).values
                        hx, hy = 1.0 / gw, 1.0 / gh
                        m = ((gc[0] >= lo[0] - hx) & (gc[0] <= hi[0] + hx) &
                             (gc[1] >= lo[1] - hy) & (gc[1] <= hi[1] + hy))
                        mb = ((gc[0] >= box[0]) & (gc[0] <= box[2]) &
                              (gc[1] >= box[1]) & (gc[1] <= box[3]))
                        am = getattr(self, "_probe_attn_map", None)
                        ah = getattr(self, "_probe_attn_heads", None)
                        alt = getattr(self, "_probe_attn_alt", None)
                        GLOBP[self.layer_idx].append((pm.mean(0).cpu().numpy(),
                                                      m.cpu().numpy(), mb.cpu().numpy(), (gh, gw),
                                                      None if am is None else am.cpu().numpy(),
                                                      None if ah is None else ah.cpu().numpy(),
                                                      None if alt is None else
                                                      {kk: (None if vv is None else vv.cpu().numpy())
                                                       for kk, vv in alt.items()}))
        return key_hd, value_hd, locs

    Qwen3_5AttentionDAT._sample_hd_from_off_guide = patched_sample


KHD = defaultdict(list)          # layer_idx -> [(shared/resid ratio, mean |offset|), ...]
KLR = defaultdict(list)          # layer_idx -> [shared/resid ratio of the LR image keys, ...]
LOC = defaultdict(list)          # layer_idx -> [(dist, dist_uniform, in_box, in_box_uniform), ...]
GLOB = defaultdict(list)         # layer_idx -> [(mean |centroid|, mean scale), ...]  (global offset term)
GLOBC = defaultdict(list)        # layer_idx -> [(cx, cy, sx, sy, tcx, tcy, tsx, tsy), ...] per sample
GLOBP = defaultdict(list)        # layer_idx -> [(relevance map [N], in-window mask [N], in-box mask [N]), ...]


def install_lr_key_hook(model):
    """Baseline for KHD: the same shared/resid ratio on the base k_proj output
    restricted to LR image tokens of the prefill, so the HD number has a
    same-layer reference instead of being read in a vacuum."""
    from llava.model.language_model.modeling_qwen3_5_dat import Qwen3_5AttentionDAT

    def make_hook(layer_idx):
        def hook(mod, inp, out):
            mask = STATE.get("img_mask")
            if not STATE["record"] or mask is None or out.shape[1] != mask.numel():
                return
            with torch.no_grad():
                k = out[0][mask.to(out.device)].float()      # [N_lr, D]
                mu = k.mean(0, keepdim=True)
                shared = mu.norm().item()
                resid = (k - mu).norm(dim=-1).mean().item()
                KLR[layer_idx].append(shared / max(resid, 1e-6))
        return hook

    for m in model.modules():
        if isinstance(m, Qwen3_5AttentionDAT) and getattr(m, "hd_proj", False):
            m.k_proj.register_forward_hook(make_hook(m.layer_idx))


# ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--processor_path", default=None)
    ap.add_argument("--dataset", choices=["hrbench4k", "hrbench8k", "vstar", "synth"], default="hrbench4k")
    ap.add_argument("--synth_json", default=None,
                    help="--dataset synth: the *.heldout.json from build_synth_hd_text_data.py")
    ap.add_argument("--synth_image_root", default=None,
                    help="--dataset synth: dir that contains the json's image paths (train_split farm)")
    ap.add_argument("--vstar_root", default="~/data/vstar_bench",
                    help="craigwu/vstar_bench checkout (test_questions.jsonl + <category>/*.jpg|.json)")
    ap.add_argument("--oracle_min_cells", type=int, default=20,
                    help="oracle window >= this many HD feature cells (32 px) per side; "
                         "20 = the 20x20 grid samples the window at full HD resolution")
    ap.add_argument("--image_folder", default=None)
    ap.add_argument("--hrbench_parquet", default=None,
                    help="explicit path to hr_bench_4k/8k.parquet (skips the Hub entirely)")
    ap.add_argument("--tok_budget", type=int, default=256,
                    help="LR LLM-visual-token budget (lr_max_pixels = tok*1024)")
    ap.add_argument("--min_pixels", type=int, default=28224)
    ap.add_argument("--hd_cap", type=int, default=5017600)
    ap.add_argument("--hr_scale", type=int, default=3)
    ap.add_argument("--hd_source",
                    choices=["real", "lr_up", "shuffle", "noise", "oracle", "oracle_rand"],
                    default="real",
                    help="oracle/oracle_rand need --dataset vstar (GT bboxes): HD is the real "
                         "image but the 400 sampling points are forced onto a window around the "
                         "target (oracle) or a same-size window elsewhere (oracle_rand)")
    ap.add_argument("--hd_bias", type=float, nargs="+", default=[0.0],
                    help="constant(s) added to lse_hd; several values = sweep")
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--max_samples", type=int, default=200)
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump_maps", default=None,
                    help="directory: write <layer>.png contact sheets (first --dump_n samples) with the "
                         "learned qk map, the de-sinked trunk attention (query = prompt end) and the "
                         "de-sinked attention from <|im_start|>, each over the image with GT box / window")
    ap.add_argument("--dump_n", type=int, default=12)
    ap.add_argument("--dump_layers", default="7,11,15,19")
    ap.add_argument("--dump_select", choices=["first", "miss", "hit"], default="first",
                    help="which samples go on the sheet: the first --dump_n, or only those whose "
                         "model-map peak is > 0.5 from the GT centre (miss) / within it (hit)")
    ap.add_argument("--glob_readout", default=None,
                    help="inference-only swap of the relevance-map -> (c, s) readout: "
                         "'floor:<lam>' subtracts lam/N from p and renormalises before the "
                         "moments (lam=1 removes the uniform background); 'win:<k>' = uniform over the "
                         "k x k cells around the argmax (c = argmax, s ~ k/gs); 'local:<k>:<lam>' = "
                         "floor then moments inside the k x k window around the argmax. "
                         "Default: model as trained")
    ap.add_argument("--glob_sink_x", type=float, default=None,
                    help="inference-only override of the model's glob_sink_x (cells whose EMA prior "
                         "> x/N are masked in the attn relevance map); default: as trained")
    ap.add_argument("--glob_layers", default=None,
                    help="comma list of DAT layers that keep the global term at inference; the "
                         "others get a flat map (c=0, s=1). Default: all")
    args = ap.parse_args()
    STATE["glob_readout"] = args.glob_readout
    STATE["glob_layers"] = (None if args.glob_layers is None
                            else {int(x) for x in args.glob_layers.split(",") if x.strip()})
    if args.glob_readout or args.glob_layers:
        print(f"[probe] glob readout override: {args.glob_readout}  layers={args.glob_layers}  "
              f"(relevance-map table then shows the post-floor map)")

    from transformers import AutoProcessor
    from llava.model.language_model.modeling_qwen3_5_dat import (
        Qwen3_5DATForConditionalGeneration,
    )

    install_merge_hook()
    samples = load_samples(args)
    has_gt = samples and samples[0]["gt"] is not None
    print(f"[probe] {len(samples)} samples  tok_budget={args.tok_budget}  "
          f"hd_source={args.hd_source}  biases={args.hd_bias}")

    model = Qwen3_5DATForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation=args.attn,
    ).eval()
    install_lr_key_hook(model)
    if args.glob_sink_x is not None:
        from llava.model.language_model.modeling_qwen3_5_dat import Qwen3_5AttentionDAT as _DAT
        n_set = 0
        for m in model.modules():
            if isinstance(m, _DAT) and hasattr(m, "glob_sink_x"):
                m.glob_sink_x = float(args.glob_sink_x); n_set += 1
        print(f"[probe] glob_sink_x override -> {args.glob_sink_x} on {n_set} DAT layers")
    image_token_id = getattr(model.config, "image_token_id", None)
    args.grid_size = int((getattr(model.config, "dat_extra_args", None) or {}).get("grid_size", 20))
    if args.hd_source.startswith("oracle") and args.dataset not in ("vstar", "synth"):
        raise SystemExit("--hd_source oracle/oracle_rand needs --dataset vstar or synth (GT bboxes)")
    # Free-text datasets: normalized containment instead of MCQ letter matching,
    # and room for a short sentence answer.
    free_text = args.dataset == "synth"
    extract = norm_text if free_text else extract_letter
    hit = (lambda p, g: bool(g) and g in p) if free_text else (lambda p, g: p == g)
    if free_text and args.max_new_tokens < 24:
        args.max_new_tokens = 24
    ppath = args.processor_path or args.model_path
    processor = AutoProcessor.from_pretrained(
        ppath, min_pixels=args.min_pixels, max_pixels=args.tok_budget * TOK_PX)
    hr_processor = AutoProcessor.from_pretrained(ppath, min_pixels=TOK_PX,
                                                 max_pixels=100_000_000)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    gen_kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=False,
                      temperature=None, top_p=None, top_k=None)

    # config name -> list of predicted letters (parallel to samples)
    configs = ["off"] + [f"{args.hd_source} b={b:+g}" for b in args.hd_bias]
    preds = {c: [] for c in configs}
    raw = {c: [] for c in configs}
    record_bias = 0.0 if 0.0 in args.hd_bias else args.hd_bias[0]

    def gen(inputs):
        with torch.inference_mode():
            out = model.generate(**inputs, **gen_kwargs)
        n_in = inputs["input_ids"].shape[1]
        return processor.tokenizer.decode(out[0][n_in:], skip_special_tokens=True).strip()

    n_forced = 0
    for i, s in enumerate(tqdm(samples, desc="probe")):
        base_inputs, hd_extra, locs = build_inputs(
            s, i, samples, processor, hr_processor, args, device, dtype)

        STATE["img_mask"] = (base_inputs["input_ids"][0] == image_token_id) \
            if image_token_id is not None else None

        STATE["bias"] = 0.0
        set_force_locs(model, None)
        t = gen(base_inputs)
        raw["off"].append(t); preds["off"].append(extract(t))

        set_force_locs(model, locs)          # None -> learned offsets (non-oracle sources)
        n_forced += locs is not None
        for b, cname in zip(args.hd_bias, configs[1:]):
            STATE["bias"] = float(b)
            STATE["record"] = (b == record_bias)
            t = gen({**base_inputs, **hd_extra})
            STATE["record"] = False
            raw[cname].append(t); preds[cname].append(extract(t))
        set_force_locs(model, None)
    STATE["bias"] = 0.0
    if args.hd_source.startswith("oracle"):
        print(f"[probe] forced sampling window on {n_forced}/{len(samples)} samples "
              f"(min {args.oracle_min_cells} HD cells per side)")

    # ── report ────────────────────────────────────────────────
    cats = sorted({s["category"] for s in samples})
    gts = [s["gt"] for s in samples]
    n = len(samples)

    def acc(letters, mask=None):
        idx = [k for k in range(n) if mask is None or mask[k]]
        if not idx:
            return float("nan")
        return 100.0 * sum(hit(letters[k], gts[k]) for k in idx) / len(idx)

    print(f"\n==== accuracy on {n} samples  (tok_budget={args.tok_budget}, "
          f"hd_source={args.hd_source}) ====")
    hdr = f"{'config':>16} | {'acc':>6} | " + " ".join(f"{c[:8]:>8}" for c in cats) \
        + f" | {'flip_vs_off':>11} | {'unres':>5}"
    print(hdr); print("-" * len(hdr))
    summary = {}
    for c in configs:
        row = {"acc": acc(preds[c]) if has_gt else None}
        for cat in cats:
            mask = [s["category"] == cat for s in samples]
            row[f"acc_{cat}"] = acc(preds[c], mask) if has_gt else None
        row["flip_vs_off"] = sum(p != q for p, q in zip(preds[c], preds["off"]))
        row["unresolved"] = sum(p == "Z" for p in preds[c]) if not free_text else 0
        summary[c] = row
        cat_cells = " ".join(
            f"{row[f'acc_{cat}']:>8.2f}" if has_gt else f"{'—':>8}" for cat in cats)
        acc_cell = f"{row['acc']:>6.2f}" if has_gt else f"{'—':>6}"
        print(f"{c:>16} | {acc_cell} | {cat_cells} | {row['flip_vs_off']:>11} | "
              f"{row['unresolved']:>5}")

    if has_gt and len(configs) > 1:
        # right->wrong / wrong->right decomposition for each HD config vs off
        print("\n==== flips vs off (HD-on made it ...) ====")
        for c in configs[1:]:
            fixed = sum(hit(preds[c][k], gts[k]) and not hit(preds["off"][k], gts[k]) for k in range(n))
            broke = sum(hit(preds["off"][k], gts[k]) and not hit(preds[c][k], gts[k]) for k in range(n))
            summary[c]["fixed"] = fixed; summary[c]["broke"] = broke
            print(f"{c:>16} : fixed {fixed:>3}   broke {broke:>3}   net {fixed - broke:+d}")

    print(f"\n==== w_hd per DAT layer  (source={args.hd_source}, bias={record_bias:+g}) ====")
    print(f"{'layer':>6} | {'mean':>8} | {'p90':>8} | {'max':>8} | {'Khd shared/resid':>16} | "
          f"{'Klr shared/resid':>16} | {'|off|':>6}")
    print("-" * 90)
    layer_whd = {}
    for lid in sorted(WHD):
        ms, p9, mx = zip(*WHD[lid])
        layer_whd[lid] = {"mean": sum(ms) / len(ms), "p90": sum(p9) / len(p9), "max": max(mx)}
        st = layer_whd[lid]
        extra = f"{'n/a':>16} | {'n/a':>16} | {'n/a':>6}"
        if KHD.get(lid):
            rs, offs = zip(*KHD[lid])
            st["k_shared_over_resid"] = sum(rs) / len(rs)
            st["mean_abs_offset"] = sum(offs) / len(offs)
            lr_cell = f"{'n/a':>16}"
            if KLR.get(lid):
                st["k_lr_shared_over_resid"] = sum(KLR[lid]) / len(KLR[lid])
                lr_cell = f"{st['k_lr_shared_over_resid']:>16.3f}"
            extra = f"{st['k_shared_over_resid']:>16.3f} | {lr_cell} | {st['mean_abs_offset']:>6.3f}"
        print(f"{lid:>6} | {st['mean']:>8.4f} | {st['p90']:>8.4f} | {st['max']:>8.4f} | {extra}")

    layer_loc = {}
    if LOC:
        # Learned-grid localisation against the GT box (bbox datasets only).
        # dist/uniform < 1: the grid moved toward the target; in_box above the
        # uniform value: more sampling points actually land on the target.
        print(f"\n==== learned grid vs GT box  (source={args.hd_source}; uniform grid = no movement) ====")
        print(f"{'layer':>6} | {'dist':>7} | {'uniform':>7} | {'ratio':>6} | "
              f"{'in_box':>7} | {'uniform':>7} | {'n':>5}")
        print("-" * 66)
        for lid in sorted(LOC):
            d, du, ib, ibu = (sum(v) / len(v) for v in zip(*LOC[lid]))
            layer_loc[lid] = {"dist": d, "dist_uniform": du, "in_box": ib,
                              "in_box_uniform": ibu, "n": len(LOC[lid])}
            print(f"{lid:>6} | {d:>7.3f} | {du:>7.3f} | {d / max(du, 1e-6):>6.3f} | "
                  f"{ib:>7.3f} | {ibu:>7.3f} | {len(LOC[lid]):>5}")

    layer_glob = {}
    if GLOB:
        # dat_use_global_offset: how much of the movement is the global term.
        # shift = mean |grid centroid| (0 = centred as at init; a supervised
        # window centre sits ~0.3-0.6 away on average); scale = mean grid scale
        # (1 = full image as at init; windows are ~0.3-0.4 of the image side).
        print(f"\n==== global offset term  (shift 0 / scale 1 = identity, i.e. legacy grid) ====")
        print(f"{'layer':>6} | {'shift':>7} | {'scale':>7} | {'n':>5}")
        print("-" * 36)
        for lid in sorted(GLOB):
            sh, sc = (sum(v) / len(v) for v in zip(*GLOB[lid]))
            layer_glob[lid] = {"shift": sh, "scale": sc, "n": len(GLOB[lid])}
            print(f"{lid:>6} | {sh:>7.3f} | {sc:>7.3f} | {len(GLOB[lid]):>5}")

    if GLOBC:
        # Sample-dependence of the global term. r = Pearson correlation across
        # samples between the predicted centroid (scale) and the GT window's
        # centre (scale) per axis; |c-t| = mean centroid->window-centre distance
        # of the prediction vs. of the best CONSTANT centroid (sample mean, the
        # question-blind prior). std(c) = spread of the prediction across samples.
        # r ~ 0 and |c-t| ~ const -> the head learned a fixed prior, it does not
        # read the question/image; r >> 0 and |c-t| < const -> it localises.
        def _corr(a, b):
            a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
            if len(a) < 3 or a.std() < 1e-9 or b.std() < 1e-9:
                return float("nan")
            return float(np.corrcoef(a, b)[0, 1])

        print(f"\n==== global term: sample-dependence  (r = corr(pred, GT) across samples) ====")
        print(f"{'layer':>6} | {'r_cx':>6} {'r_cy':>6} | {'r_sx':>6} {'r_sy':>6} | "
              f"{'|c-t|':>6} {'const':>6} | {'std(c)':>6} | {'mean s':>6} {'GT s':>6}")
        print("-" * 84)
        for lid in sorted(GLOBC):
            cols = list(zip(*GLOBC[lid]))
            cx, cy, sx, sy, tcx, tcy, tsx, tsy = cols
            n_ = len(cx)
            mcx, mcy = sum(cx) / n_, sum(cy) / n_
            d_pred = sum(math.hypot(a - b, c - d) for a, b, c, d in zip(cx, tcx, cy, tcy)) / n_
            d_const = sum(math.hypot(mcx - b, mcy - d) for b, d in zip(tcx, tcy)) / n_
            std_c = math.hypot(float(np.std(cx)), float(np.std(cy)))
            layer_glob.setdefault(lid, {}).update({
                "r_cx": _corr(cx, tcx), "r_cy": _corr(cy, tcy),
                "r_sx": _corr(sx, tsx), "r_sy": _corr(sy, tsy),
                "centroid_err": d_pred, "centroid_err_const": d_const, "centroid_std": std_c,
                "scale_mean": (sum(sx) + sum(sy)) / (2 * n_), "scale_gt": (sum(tsx) + sum(tsy)) / (2 * n_),
            })
            g = layer_glob[lid]
            print(f"{lid:>6} | {g['r_cx']:>6.3f} {g['r_cy']:>6.3f} | {g['r_sx']:>6.3f} {g['r_sy']:>6.3f} | "
                  f"{d_pred:>6.3f} {d_const:>6.3f} | {std_c:>6.3f} | {g['scale_mean']:>6.3f} {g['scale_gt']:>6.3f}")

    if GLOBP:
        # The relevance map before the soft-argmax collapses it to centroid /
        # spread. Separates "the map never learned content matching" from "the
        # map is on target but the centroid aggregation loses it":
        #   mass    = softmax mass inside the GT window (pred) vs the same for
        #             the PRIOR map (mean map over all samples, question-blind)
        #             vs a uniform map (base = window's share of the cells)
        #   argmax  = fraction of samples whose peak cell is inside the window
        #             (pred / prior / chance = base); box = inside the raw GT box
        #   peak    = max p * N (1 = flat map)
        # pred ~ prior ~ base, peak ~ 1  -> flat map + prior, W_q/W_k learned nothing:
        #                                    the map needs its own dense signal
        # pred ~ prior >> base           -> a peaky but constant map (prior)
        # pred >> prior, but r_c low     -> map localises, soft-argmax aggregation
        #                                    loses it (multi-modal): change the aggregation
        print(f"\n==== relevance map vs GT window  (pred / prior=mean map / base=uniform) ====")
        print(f"{'layer':>6} | {'mass':>6} {'prior':>6} {'base':>6} | {'argmax':>6} {'prior':>6} {'box':>6} | "
              f"{'peak':>6} | {'n':>4}")
        print("-" * 78)
        for lid in sorted(GLOBP):
            maps = np.stack([v[0] for v in GLOBP[lid]]).astype(np.float64)   # [n, N]
            win = np.stack([v[1] for v in GLOBP[lid]]).astype(np.float64)    # [n, N]
            bx = np.stack([v[2] for v in GLOBP[lid]]).astype(np.float64)
            prior = maps.mean(0, keepdims=True)                             # [1, N]
            n_, N = maps.shape
            mass = float((maps * win).sum(1).mean())
            mass_prior = float((prior * win).sum(1).mean())
            base = float(win.mean())
            am = maps.argmax(1)
            argmax_in = float(win[np.arange(n_), am].mean())
            argmax_prior = float(win[:, int(prior.argmax())].mean())
            argmax_box = float(bx[np.arange(n_), am].mean())
            peak = float((maps.max(1) * N).mean())
            layer_glob.setdefault(lid, {}).update({
                "map_mass": mass, "map_mass_prior": mass_prior, "map_mass_base": base,
                "map_argmax_in": argmax_in, "map_argmax_prior": argmax_prior, "map_argmax_box": argmax_box,
                "map_peak": peak,
            })
            print(f"{lid:>6} | {mass:>6.3f} {mass_prior:>6.3f} {base:>6.3f} | "
                  f"{argmax_in:>6.3f} {argmax_prior:>6.3f} {argmax_box:>6.3f} | {peak:>6.2f} | {n_:>4}")

        # Diagnostic 2: the trunk's own intention->LR attention as the map, same
        # columns. attn >> prior  -> the features carry the match, the learned
        # qk head just did not pick it up (init glob_q/glob_k from q_proj/k_proj
        # or use the attention directly); attn ~ prior -> the intention token
        # does not know where the answer is at this layer, single-token qk is
        # the wrong source.
        # Coordinate check: the same window mass with the map mirrored in x, in
        # y, or transposed. Any of these beating the identity by a clear margin
        # means the map's cell order and the target's (x, y) disagree.
        def _var(maps, hw, how):
            gh_, gw_ = hw
            m = maps.reshape(-1, gh_, gw_)
            if how == "flipx":
                m = m[:, :, ::-1]
            elif how == "flipy":
                m = m[:, ::-1, :]
            elif how == "T":
                m = m.transpose(0, 2, 1) if gh_ == gw_ else m
            return m.reshape(maps.shape[0], -1)

        print(f"\n==== trunk attention (intention -> LR cells) as relevance map;  coord check on the qk map ====")
        print(f"{'layer':>6} | {'a.mass':>6} {'prior':>6} {'base':>6} | {'a.amax':>6} {'prior':>6} {'box':>6} | "
              f"{'a.peak':>6} | {'qk:id':>6} {'flipx':>6} {'flipy':>6} {'T':>6} | {'att:id':>6} {'flipx':>6} {'flipy':>6} {'T':>6}")
        print("-" * 128)
        for lid in sorted(GLOBP):
            rows = GLOBP[lid]
            maps = np.stack([v[0] for v in rows]).astype(np.float64)
            win = np.stack([v[1] for v in rows]).astype(np.float64)
            bx = np.stack([v[2] for v in rows]).astype(np.float64)
            hw = rows[0][3]
            n_, N = maps.shape
            qk_var = {h: float((_var(maps, hw, h) * win).sum(1).mean()) for h in ("id", "flipx", "flipy", "T")}
            att_rows = [v[4] for v in rows if v[4] is not None]
            if len(att_rows) == n_:
                att = np.stack(att_rows).astype(np.float64)
                prior = att.mean(0, keepdims=True)
                a_mass = float((att * win).sum(1).mean())
                a_prior = float((prior * win).sum(1).mean())
                am = att.argmax(1)
                a_amax = float(win[np.arange(n_), am].mean())
                a_amax_prior = float(win[:, int(prior.argmax())].mean())
                a_box = float(bx[np.arange(n_), am].mean())
                a_peak = float((att.max(1) * N).mean())
                att_var = {h: float((_var(att, hw, h) * win).sum(1).mean()) for h in ("id", "flipx", "flipy", "T")}
                layer_glob.setdefault(lid, {}).update({
                    "attn_mass": a_mass, "attn_mass_prior": a_prior, "attn_argmax_in": a_amax,
                    "attn_argmax_prior": a_amax_prior, "attn_argmax_box": a_box, "attn_peak": a_peak,
                    "attn_mass_flip": att_var, "map_mass_flip": qk_var,
                })
                print(f"{lid:>6} | {a_mass:>6.3f} {a_prior:>6.3f} {float(win.mean()):>6.3f} | "
                      f"{a_amax:>6.3f} {a_amax_prior:>6.3f} {a_box:>6.3f} | {a_peak:>6.2f} | "
                      f"{qk_var['id']:>6.3f} {qk_var['flipx']:>6.3f} {qk_var['flipy']:>6.3f} {qk_var['T']:>6.3f} | "
                      f"{att_var['id']:>6.3f} {att_var['flipx']:>6.3f} {att_var['flipy']:>6.3f} {att_var['T']:>6.3f}")
            else:
                layer_glob.setdefault(lid, {}).update({"map_mass_flip": qk_var})
                print(f"{lid:>6} | {'(no attention map recorded)':<42} | "
                      f"{qk_var['id']:>6.3f} {qk_var['flipx']:>6.3f} {qk_var['flipy']:>6.3f} {qk_var['T']:>6.3f} |")

        # Diagnostic 3: per HEAD. The head-averaged attention is sink-dominated
        # (one fixed cell holds 20-50% of the mass), which hides any single
        # "grounding head". Per head and per layer, with the sink cells removed
        # (cells whose across-sample mean exceeds `sink_x` x uniform are constant,
        # question-blind, and are zeroed before renormalising):
        #   mean   = de-sinked head-averaged window mass
        #   best   = the head with the highest de-sinked window mass: its mass,
        #            the mass of ITS constant prior map (content = mass - prior),
        #            argmax-in-window, argmax-in-box
        #   n>base = heads whose de-sinked mass beats uniform by >= 0.05
        # best mass >= 0.5 with box well above ~0.07 -> grounding heads exist:
        # a learned per-head weighting over the trunk's attention (GLOB_REL=attn)
        # is a near-parameter-free relevance source. best ~ base -> the frozen
        # trunk has no readable question->cell match at all in this form.
        sink_x = 5.0

        def _desink(m):                                    # m: [n, N] rows sum to 1
            prior = m.mean(0)
            if m.shape[0] < 50:                            # too few samples: a single sharp
                return m, 0                                # peak would look like a sink
            keep = (prior <= sink_x / m.shape[1]).astype(np.float64)
            m = m * keep
            return m / np.clip(m.sum(1, keepdims=True), 1e-9, None), int((keep == 0).sum())

        def _head_stats(heads, win, bx):
            """heads [n, H, N] (rows sum to 1) -> de-sinked per-head window stats."""
            n_, H, N = heads.shape
            base = float(win.mean())
            per_head, n_sinks = [], 0
            for h in range(H):
                m, ns = _desink(heads[:, h])
                n_sinks = max(n_sinks, ns)
                prior = m.mean(0, keepdims=True)
                am = m.argmax(1)
                per_head.append({
                    "mass": float((m * win).sum(1).mean()),
                    "prior": float((prior * win).sum(1).mean()),
                    "amax": float(win[np.arange(n_), am].mean()),
                    "box": float(bx[np.arange(n_), am].mean()),
                })
            mean_ds, _ = _desink(heads.mean(1))
            b = int(np.argmax([d["mass"] for d in per_head]))
            return {
                "sinks": n_sinks, "mean_mass": float((mean_ds * win).sum(1).mean()), "base": base,
                "best_head": b, "best": per_head[b],
                "n_above_base": sum(d["mass"] >= base + 0.05 for d in per_head),
                "per_head_mass": [d["mass"] for d in per_head], "H": H,
            }

        def _print_head_table(title, get_heads, key):
            print(f"\n==== {title}  (de-sinked: cells with mean > {sink_x:g}x uniform removed) ====")
            print(f"{'layer':>6} | {'sinks':>5} | {'mean':>6} {'base':>6} | {'best':>4} {'mass':>6} {'prior':>6} "
                  f"{'amax':>6} {'box':>6} | {'n>base':>6}/{'H':<3}")
            print("-" * 92)
            for lid in sorted(GLOBP):
                rows = GLOBP[lid]
                hr = [get_heads(v) for v in rows]
                if any(h is None for h in hr) or not hr:
                    continue
                st = _head_stats(np.stack(hr).astype(np.float64),
                                 np.stack([v[1] for v in rows]).astype(np.float64),
                                 np.stack([v[2] for v in rows]).astype(np.float64))
                layer_glob.setdefault(lid, {})[key] = st
                bb = st["best"]
                print(f"{lid:>6} | {st['sinks']:>5} | {st['mean_mass']:>6.3f} {st['base']:>6.3f} | {st['best_head']:>4} "
                      f"{bb['mass']:>6.3f} {bb['prior']:>6.3f} {bb['amax']:>6.3f} {bb['box']:>6.3f} | "
                      f"{st['n_above_base']:>6}/{st['H']:<3}")

        _print_head_table("trunk attention per head, query = intention token <|im_start|>",
                          lambda v: v[5] if len(v) > 5 else None, "attn_heads")
        # Diagnostic 4: the same table with the query taken at other prompt
        # positions. If one of them localises where <|im_start|> does not, the
        # qk head asks the wrong token, not a trunk without the information.
        for kname, desc in (("q_last", "last question token"), ("im_end", "<|im_end|>"),
                            ("nl", "\\n after <|im_end|>"), ("asst", "'assistant'"),
                            ("asst_nl", "\\n after 'assistant'"), ("q_mean", "mean over question tokens")):
            _print_head_table(f"trunk attention per head, query = {desc} [{kname}]",
                              lambda v, kn=kname: (v[6] or {}).get(kn) if len(v) > 6 else None,
                              f"attn_heads_{kname}")

        # Diagnostic 7: are the misses (peak far from the target, ~30% of the
        # samples on miniD-attn) shared across layers / heads, or independent?
        # If independent, fusing layers (mean / product of the maps) or picking
        # heads cuts the miss rate without any new capacity. Per map set:
        #   mass / amax = as above; |c-t| = soft-centroid error to the GT window
        #   centre (the model's own c on the single-layer rows); miss = fraction
        #   of samples whose peak is > 0.5 (a quarter of the image) from the
        #   target centre.  Ceilings: any = some layer/head peaks in the window;
        #   agree = layers' peaks within 2 cells of each other (and how often
        #   an agreed peak is in the window).
        def _grid_xy(hw):
            gh_, gw_ = hw                                                 # same grid as patched_sample's ref
            mx, my = 1.0 / max(gw_ - 1, 1), 1.0 / max(gh_ - 1, 1)
            gx = np.linspace(-1 + mx, 1 - mx, gw_)
            gy = np.linspace(-1 + my, 1 - my, gh_)
            X, Y = np.meshgrid(gx, gy)                                   # [gh, gw]
            return X.ravel(), Y.ravel()                                   # [N] each

        def _set_stats(maps, win, bx, tc, hw):
            n_, N = maps.shape
            X, Y = _grid_xy(hw)
            am = maps.argmax(1)
            c = np.stack([(maps * X).sum(1), (maps * Y).sum(1)], 1)     # [n, 2]
            cerr = float(np.linalg.norm(c - tc, axis=1).mean())
            pk = np.stack([X[am], Y[am]], 1)
            miss = float((np.linalg.norm(pk - tc, axis=1) > 0.5).mean())
            return {"mass": float((maps * win).sum(1).mean()),
                    "amax": float(win[np.arange(n_), am].mean()),
                    "box": float(bx[np.arange(n_), am].mean()),
                    "cerr": cerr, "miss": miss}

        lids = [l for l in sorted(GLOBP) if l in GLOBC and len(GLOBC[l]) == len(GLOBP[l])]
        if len(lids) >= 2:
            n_ = len(GLOBP[lids[0]])
            hw = GLOBP[lids[0]][0][3]
            win = np.stack([v[1] for v in GLOBP[lids[0]]]).astype(np.float64)
            bx = np.stack([v[2] for v in GLOBP[lids[0]]]).astype(np.float64)
            tc = np.array([[r[4], r[5]] for r in GLOBC[lids[0]]], dtype=np.float64)   # GT window centre
            per = {l: np.stack([v[0] for v in GLOBP[l]]).astype(np.float64) for l in lids}
            base = float(win.mean())
            good = [l for l in lids if float((per[l] * win).sum(1).mean()) >= base + 0.1]
            if len(good) < 2:
                good = lids[-max(2, len(lids) // 2):]
            X, Y = _grid_xy(hw)

            print(f"\n==== cross-layer fusion of the model's map  (layers used for fusion: {good};  "
                  f"miss = peak > 0.5 from GT centre) ====")
            print(f"{'map':>14} | {'mass':>6} {'amax':>6} {'box':>6} | {'|c-t|':>6} {'miss':>6}")
            print("-" * 60)
            fusion = {}
            for l in lids:
                s_ = _set_stats(per[l], win, bx, tc, hw)
                fusion[f"layer{l}"] = s_
                print(f"{'layer ' + str(l):>14} | {s_['mass']:>6.3f} {s_['amax']:>6.3f} {s_['box']:>6.3f} | "
                      f"{s_['cerr']:>6.3f} {s_['miss']:>6.3f}")
            mean_map = np.mean([per[l] for l in good], 0)
            mean_map /= mean_map.sum(1, keepdims=True)
            prod_map = np.exp(np.mean([np.log(np.clip(per[l], 1e-9, None)) for l in good], 0))
            prod_map /= prod_map.sum(1, keepdims=True)
            for name, m in (("mean", mean_map), ("geo-mean", prod_map)):
                s_ = _set_stats(m, win, bx, tc, hw)
                fusion[name] = s_
                print(f"{name:>14} | {s_['mass']:>6.3f} {s_['amax']:>6.3f} {s_['box']:>6.3f} | "
                      f"{s_['cerr']:>6.3f} {s_['miss']:>6.3f}")
            # ceilings / agreement over the fused layers
            ams = np.stack([per[l].argmax(1) for l in good])                        # [L, n]
            in_win = np.stack([win[np.arange(n_), ams[i]] for i in range(len(good))])   # [L, n]
            any_in = float(in_win.max(0).mean())
            all_in = float(in_win.min(0).mean())
            pk = np.stack([np.stack([X[ams[i]], Y[ams[i]]], 1) for i in range(len(good))])   # [L, n, 2]
            spread = np.linalg.norm(pk - pk.mean(0, keepdims=True), axis=2).max(0)           # [n]
            agree = spread <= 2.0 * (2.0 / hw[1])
            agree_in = float(in_win.min(0)[agree].mean()) if agree.any() else float("nan")
            fusion["ceiling"] = {"any_in": any_in, "all_in": all_in, "agree_frac": float(agree.mean()),
                                 "agree_in": agree_in}
            print(f"  peak in window: any layer {any_in:.3f} | all layers {all_in:.3f} | "
                  f"layers agree (<=2 cells) on {agree.mean():.3f} of samples, of which in window {agree_in:.3f}")
            layer_glob.setdefault(-1, {})["fusion"] = fusion

            # per-head oracle at the ans_prev query: the model's glob_query_pos
            # 'ans_prev' is the token right before the answer = the '\n' after
            # 'assistant' = alt key 'asst_nl' (t+2), NOT 'nl' (t-1, which has no
            # signal). Does SOME head peak in the window far more often than the
            # head mean? (=> learn head weights)
            QPOS_KEY = "asst_nl"
            for l in good:
                hr = [(v[6] or {}).get(QPOS_KEY) if len(v) > 6 else None for v in GLOBP[l]]
                if any(h is None for h in hr):
                    continue
                heads = np.stack(hr).astype(np.float64)                     # [n, H, N]
                H = heads.shape[1]
                hin = []
                for h in range(H):
                    m, _ = _desink(heads[:, h])
                    hin.append(win[np.arange(n_), m.argmax(1)])
                hin = np.stack(hin)                                          # [H, n]
                mean_h = heads.mean(1)
                mean_h, _ = _desink(mean_h)
                s_mean = float(win[np.arange(n_), mean_h.argmax(1)].mean())
                print(f"  layer {l:>2} heads@{QPOS_KEY}: peak-in-window  mean-of-heads {s_mean:.3f} | "
                      f"best single head {hin.mean(1).max():.3f} | any head {hin.max(0).mean():.3f} | "
                      f">= half the heads {(hin.mean(0) >= 0.5).mean():.3f}")

            # Diagnostic 8: how to POOL the heads (raw trunk attention at the
            # ans_prev query, each head de-sinked, no tau / cell_bias):
            #   mean   = what the model does now
            #   max    = per cell max over heads, renormalised (picks the sharpest head)
            #   best   = the single head with the highest peak-in-window rate (static oracle)
            #   conf   = per-sample convex mix, head weight = its peak value (sharper = trusted)
            #   lse    = logsumexp over heads with temperature 0.1 (soft max)
            # Same columns as the fusion table; "model" = the trained map for reference.
            pools = ("mean", "max", "best", "conf", "lse")
            print(f"\n==== head pooling (raw attention @ {QPOS_KEY} (= ans_prev), per-head de-sinked; base amax = {base:.3f}) ====")
            print(f"{'layer':>6} {'pool':>6} | {'mass':>6} {'amax':>6} {'box':>6} | {'|c-t|':>6} {'miss':>6}")
            print("-" * 60)
            pool_stats = {}
            pooled_by_layer = {}
            for l in good:
                hr = [(v[6] or {}).get(QPOS_KEY) if len(v) > 6 else None for v in GLOBP[l]]
                if any(h is None for h in hr):
                    continue
                heads = np.stack(hr).astype(np.float64)                     # [n, H, N]
                H = heads.shape[1]
                ds = np.stack([_desink(heads[:, h])[0] for h in range(H)], 1)   # [n, H, N]
                hin = np.stack([win[np.arange(n_), ds[:, h].argmax(1)] for h in range(H)])   # [H, n]
                best_h = int(hin.mean(1).argmax())
                conf_w = ds.max(2)                                           # [n, H]
                conf_w = conf_w / conf_w.sum(1, keepdims=True)
                lse = np.log(np.exp(np.log(np.clip(ds, 1e-9, None)) / 0.1).sum(1))   # [n, N]
                cands = {
                    "mean": ds.mean(1),
                    "max": ds.max(1),
                    "best": ds[:, best_h],
                    "conf": (ds * conf_w[:, :, None]).sum(1),
                    "lse": np.exp(lse - lse.max(1, keepdims=True)),
                }
                s_model = _set_stats(per[l], win, bx, tc, hw)
                print(f"{l:>6} {'model':>6} | {s_model['mass']:>6.3f} {s_model['amax']:>6.3f} {s_model['box']:>6.3f} | "
                      f"{s_model['cerr']:>6.3f} {s_model['miss']:>6.3f}")
                pooled_by_layer[l] = {}
                for name in pools:
                    m = cands[name]
                    m = m / np.clip(m.sum(1, keepdims=True), 1e-9, None)
                    pooled_by_layer[l][name] = m
                    s_ = _set_stats(m, win, bx, tc, hw)
                    pool_stats[f"layer{l}_{name}"] = s_
                    tag = f"{name}[{best_h}]" if name == "best" else name
                    print(f"{'':>6} {tag:>6} | {s_['mass']:>6.3f} {s_['amax']:>6.3f} {s_['box']:>6.3f} | "
                          f"{s_['cerr']:>6.3f} {s_['miss']:>6.3f}")
            # each pooling fused (mean) over the good layers
            if pooled_by_layer:
                print("-" * 60)
                for name in pools:
                    ms = [pooled_by_layer[l][name] for l in pooled_by_layer]
                    m = np.mean(ms, 0)
                    m = m / m.sum(1, keepdims=True)
                    s_ = _set_stats(m, win, bx, tc, hw)
                    pool_stats[f"fused_{name}"] = s_
                    print(f"{'fused':>6} {name:>6} | {s_['mass']:>6.3f} {s_['amax']:>6.3f} {s_['box']:>6.3f} | "
                          f"{s_['cerr']:>6.3f} {s_['miss']:>6.3f}")
            layer_glob.setdefault(-1, {})["head_pool"] = pool_stats

        # Diagnostic 10: split the samples by whether the model's map PEAK is
        # inside the GT window at the reference layer (hit) or not (miss) and
        # look at everything downstream per subset: readout accuracy off/on,
        # grid scale s, in_box, dist ratio. Hits ~ oracle & misses ~ shuffle
        # => the pipeline works and localisation is the only limit; hits not
        # better than misses => the readout / grid scale is the limit.
        # Also splits by GT-box height at LR (<10 px: text unreadable at LR).
        if has_gt and len(GLOBP) >= 1:
            ref_l = max(GLOBP, key=lambda l: layer_glob.get(l, {}).get("map_argmax_in", -1.0))
            rows = GLOBP[ref_l]
            if len(rows) == n:
                maps = np.stack([v[0] for v in rows]).astype(np.float64)
                win = np.stack([v[1] for v in rows]).astype(np.float64)
                am = maps.argmax(1)
                hitm = win[np.arange(n), am] > 0.5                          # peak inside window
                lr_h = []
                for s in samples:
                    W_, H_ = s["image"].size
                    lr_h.append(s["bboxes"][0][3] * math.sqrt(args.tok_budget * TOK_PX / max(W_ * H_, 1)))
                lr_h = np.array(lr_h)
                small = lr_h < 10.0
                subsets = [("peak in win", hitm), ("peak out", ~hitm),
                           ("box<10px@LR", small), ("box>=10px", ~small),
                           ("in&>=10px", hitm & ~small), ("out&>=10px", ~hitm & ~small)]
                on_c = configs[1] if len(configs) > 1 else None
                print(f"\n==== by localisation outcome at layer {ref_l}  (peak in GT window = hit)  "
                      f"and by GT box height at LR ====")
                print(f"{'subset':>12} | {'n':>4} | {'off':>6} {'on':>6} {'delta':>6} | "
                      f"{'s':>6} {'in_box':>6} {'ratio':>6}")
                print("-" * 70)
                subset_out = {}
                for name, m in subsets:
                    idx = [k for k in range(n) if m[k]]
                    if not idx:
                        continue
                    a_off = acc(preds["off"], m)
                    a_on = acc(preds[on_c], m) if on_c else float("nan")
                    s_mean = float(np.mean([(GLOBC[ref_l][k][2] + GLOBC[ref_l][k][3]) / 2 for k in idx])) \
                        if ref_l in GLOBC and len(GLOBC[ref_l]) == n else float("nan")
                    if ref_l in LOC and len(LOC[ref_l]) == n:
                        ib = float(np.mean([LOC[ref_l][k][2] for k in idx]))
                        ratio = float(np.mean([LOC[ref_l][k][0] / max(LOC[ref_l][k][1], 1e-9) for k in idx]))
                    else:
                        ib = ratio = float("nan")
                    subset_out[name] = {"n": len(idx), "off": a_off, "on": a_on, "s": s_mean,
                                        "in_box": ib, "ratio": ratio}
                    print(f"{name:>12} | {len(idx):>4} | {a_off:>6.2f} {a_on:>6.2f} {a_on - a_off:>+6.2f} | "
                          f"{s_mean:>6.3f} {ib:>6.3f} {ratio:>6.3f}")
                layer_glob.setdefault(-1, {})["by_outcome"] = {"ref_layer": ref_l, **subset_out}

        if args.dump_maps:
            # Visual check: per layer one contact sheet, one row per sample,
            # three panels: learned qk map | de-sinked trunk attention with the
            # query at the prompt end (asst_nl) | de-sinked attention from
            # <|im_start|> (the current intention token). Green = GT box,
            # yellow = supervision window. Heat = p / max(p) of that panel.
            from PIL import ImageDraw
            os.makedirs(args.dump_maps, exist_ok=True)
            PW = 300
            want = [int(x) for x in args.dump_layers.split(",") if x.strip()]

            def _heat(img, pmap, hw, win, box_px, label):
                gh_, gw_ = hw
                W, H = img.size
                sc = PW / W
                base_img = img.resize((PW, int(H * sc))).convert("RGB")
                heat = np.asarray(pmap, dtype=np.float64).reshape(gh_, gw_) * (gh_ * gw_) - 1.0
                heat = np.clip(heat / max(float(heat.max()), 1e-9), 0.0, 1.0)   # 0 = uniform level
                hm = Image.fromarray((heat * 255).astype(np.uint8)).resize(base_img.size, Image.BILINEAR)
                red = Image.new("RGB", base_img.size, (255, 40, 0))
                out = Image.composite(red, base_img, hm.point(lambda v: int(v * 0.75)))
                d = ImageDraw.Draw(out)
                # window cells
                wm = np.asarray(win, dtype=bool).reshape(gh_, gw_)
                ys, xs = np.where(wm)
                if len(xs):
                    d.rectangle([xs.min() / gw_ * out.width, ys.min() / gh_ * out.height,
                                 (xs.max() + 1) / gw_ * out.width, (ys.max() + 1) / gh_ * out.height],
                                outline=(255, 230, 0), width=2)
                x, y, w, h = box_px
                d.rectangle([x * sc, y * sc, (x + w) * sc, (y + h) * sc], outline=(0, 255, 0), width=2)
                d.text((4, 2), label, fill=(255, 255, 255))
                return out

            for lid in want:
                rows = GLOBP.get(lid)
                if not rows or len(rows) != len(samples):
                    print(f"[probe] dump_maps: layer {lid} has {0 if not rows else len(rows)} entries "
                          f"for {len(samples)} samples, skipped")
                    continue
                hw = rows[0][3]
                qk_all = np.stack([v[0] for v in rows]).astype(np.float64)
                att_end = [((v[6] or {}).get("asst_nl") if len(v) > 6 else None) for v in rows]
                att_im = [v[5] for v in rows]
                if any(a is None for a in att_end) or any(a is None for a in att_im):
                    print(f"[probe] dump_maps: layer {lid} lacks attention maps, skipped")
                    continue
                end_ds, _ = _desink(np.stack([a.mean(0) for a in att_end]).astype(np.float64))
                im_ds, _ = _desink(np.stack([a.mean(0) for a in att_im]).astype(np.float64))
                # which samples: first N, or only the model-map MISSES (peak
                # > 0.5 from the GT window centre) / HITS, to see what the
                # localiser fails on (diag 9: legible at LR or not?)
                pk_d = None
                if lid in GLOBC and len(GLOBC[lid]) == len(rows):
                    Xg, Yg = _grid_xy(hw)
                    am = qk_all.argmax(1)
                    tcl = np.array([[r[4], r[5]] for r in GLOBC[lid]], dtype=np.float64)
                    pk_d = np.linalg.norm(np.stack([Xg[am], Yg[am]], 1) - tcl, axis=1)
                if args.dump_select == "first" or pk_d is None:
                    sel = list(range(min(args.dump_n, len(rows))))
                else:
                    mask = pk_d > 0.5 if args.dump_select == "miss" else pk_d <= 0.5
                    sel = [int(i) for i in np.where(mask)[0][:args.dump_n]]
                    print(f"[probe] dump_maps: layer {lid} {args.dump_select}: {int(mask.sum())}/{len(rows)} samples")
                panels = []
                for i in sel:
                    s = samples[i]
                    bx = s["bboxes"][0]
                    win = rows[i][1]
                    m_qk = float((qk_all[i] * win).sum())
                    m_end = float((end_ds[i] * win).sum())
                    m_im = float((im_ds[i] * win).sum())
                    # GT box size in LR pixels (LR area ~ tok_budget * 1024 px):
                    # text < ~10 px tall is unreadable at LR
                    W_, H_ = s["image"].size
                    lr_sc = math.sqrt(args.tok_budget * TOK_PX / max(W_ * H_, 1))
                    box_lr = f"box@LR {bx[2] * lr_sc:.0f}x{bx[3] * lr_sc:.0f}px"
                    dtxt = "" if pk_d is None else f"  peak-dist={pk_d[i]:.2f}"
                    row = [
                        _heat(s["image"], qk_all[i], hw, win, bx, f"model map  mass={m_qk:.2f}{dtxt}"),
                        _heat(s["image"], end_ds[i], hw, win, bx, f"attn@prompt-end  mass={m_end:.2f}"),
                        _heat(s["image"], im_ds[i], hw, win, bx, f"attn@<|im_start|>  mass={m_im:.2f}"),
                    ]
                    rh = max(p.height for p in row) + 14
                    strip = Image.new("RGB", (PW * 3 + 8, rh), (20, 20, 20))
                    for j, p in enumerate(row):
                        strip.paste(p, (j * (PW + 4), 14))
                    gt_txt = str(s.get("gt", ""))[:40]
                    ImageDraw.Draw(strip).text(
                        (4, 1), f"[{i}] {box_lr}  gt='{gt_txt}'  {s['prompt'][:90]}", fill=(200, 200, 200))
                    panels.append(strip)
                if not panels:
                    print(f"[probe] dump_maps: layer {lid}: nothing selected")
                    continue
                sheet = Image.new("RGB", (panels[0].width, sum(p.height for p in panels)), (20, 20, 20))
                yy = 0
                for p in panels:
                    sheet.paste(p, (0, yy))
                    yy += p.height
                fn = os.path.join(args.dump_maps, f"layer{lid}.png")
                sheet.save(fn)
                print(f"[probe] dump_maps: wrote {fn}  ({len(panels)} samples)")

    if args.out:
        json.dump({
            "model_path": args.model_path,
            "dataset": args.image_folder or args.dataset,
            "tok_budget": args.tok_budget,
            "hd_source": args.hd_source,
            "hd_bias": args.hd_bias,
            "num_samples": n,
            "summary": summary,
            "layer_whd": layer_whd,
            "layer_loc": layer_loc,
            "layer_glob": layer_glob,
            "per_sample": [
                {"gt": gts[k], "category": samples[k]["category"],
                 **{c: raw[c][k] for c in configs}}
                for k in range(n)
            ],
        }, open(args.out, "w"), indent=2, ensure_ascii=False)
        print(f"[probe] saved -> {args.out}")


if __name__ == "__main__":
    main()
