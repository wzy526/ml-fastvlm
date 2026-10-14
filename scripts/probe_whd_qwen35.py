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
                   crop_tokens  READOUT CEILING, no HD pass: the oracle window
                            cropped from the original image goes in as a 2nd
                            image of <= --crop_tok ordinary LLM tokens
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
import torch.nn.functional as Fnn
from PIL import Image
from tqdm import tqdm

if __package__:
    from .hd_probe_utils import different_image_indices, image_fingerprint
else:
    from hd_probe_utils import different_image_indices, image_fingerprint

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
            bboxes, targets = [], []
            if os.path.isfile(side):
                sd = json.load(open(side))
                bboxes = sd.get("bbox", []) or []
                targets = sd.get("target_object", []) or []
            samples.append({
                "image": Image.open(img_path).convert("RGB"),
                "prompt": doc["text"].strip(),
                "gt": str(doc["label"]).strip().upper(),
                "category": str(doc.get("category", "n/a")),
                "bboxes": [list(map(float, b)) for b in bboxes],
                "targets": [str(t) for t in targets],      # names of the queried objects
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
    lr_px = int(lr_grid_thw[1]) * PATCH * int(lr_grid_thw[2]) * PATCH
    hd_total = min(lr_px * hr_scale * hr_scale, image.width * image.height, hd_cap)
    aspect = image.width / image.height
    hd_h = int(math.sqrt(hd_total / aspect))
    hd_w = int(hd_h * aspect)
    hd_h = max(FACTOR, (hd_h // FACTOR) * FACTOR)
    hd_w = max(FACTOR, (hd_w // FACTOR) * FACTOR)
    return hd_w, hd_h


def prepare_shuffle_sources(samples):
    """Record content-checked donors once, before loading the model."""
    fingerprints = [image_fingerprint(sample["image"]) for sample in samples]
    donors = different_image_indices(fingerprints)
    for i, (sample, donor) in enumerate(zip(samples, donors)):
        sample["image_sha256"] = fingerprints[i]
        sample["shuffle_source_index"] = donor
        sample["shuffle_source_sha256"] = fingerprints[donor]


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
        if "shuffle_source_index" not in sample:
            prepare_shuffle_sources(samples)
        other = samples[sample["shuffle_source_index"]]["image"]
        # LR/prompt and the HD geometry still belong to the target sample.
        return other.resize((hd_w, hd_h), Image.BICUBIC)
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


def gt_box_frac(sample):
    """Union of the GT boxes as (x0, y0, x1, y1) image fractions, or None."""
    boxes = sample.get("bboxes") or []
    if not boxes:
        return None
    W, H = sample["image"].size
    return (min(b[0] for b in boxes) / W, min(b[1] for b in boxes) / H,
            max(b[0] + b[2] for b in boxes) / W, max(b[1] + b[3] for b in boxes) / H)


def crop_window(sample, hd_w, hd_h, min_cells, pad=1.0):
    """The 'oracle' window as (x0, y0, x1, y1) image fractions — same box
    expansion / min-cells / clamping as oracle_locs(source='oracle'); pad > 1
    scales the box about its centre first (looser window, less zoom)."""
    gt = gt_box_frac(sample)
    if gt is None:
        return None
    x0, y0, x1, y1 = gt
    ww = min(max((x1 - x0) * pad, min_cells * FACTOR / hd_w), 1.0)
    wh = min(max((y1 - y0) * pad, min_cells * FACTOR / hd_h), 1.0)
    cx = min(max((x0 + x1) / 2, ww / 2), 1 - ww / 2)
    cy = min(max((y0 + y1) / 2, wh / 2), 1 - wh / 2)
    return cx - ww / 2, cy - wh / 2, cx + ww / 2, cy + wh / 2


def corrupt_window(win, gt, rng, hit, shift):
    """Controlled localisation error for the hit-rate / precision curves.
    hit=False: a window of the SAME size at a random place not overlapping the
    GT box (a confidently wrong localiser). shift>0 (hit=True): the window
    centre moved by shift x window size in a random direction (an imprecise
    localiser), clamped into the image."""
    x0, y0, x1, y1 = win
    ww, wh = x1 - x0, y1 - y0
    if not hit:
        best, best_d = None, -1.0
        gcx, gcy = (gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2
        for _ in range(64):
            rx, ry = rng.uniform(ww / 2, max(ww / 2, 1 - ww / 2)), rng.uniform(wh / 2, max(wh / 2, 1 - wh / 2))
            if abs(rx - gcx) >= (ww + gt[2] - gt[0]) / 2 or abs(ry - gcy) >= (wh + gt[3] - gt[1]) / 2:
                best = (rx, ry)
                break
            d = (rx - gcx) ** 2 + (ry - gcy) ** 2
            if d > best_d:
                best, best_d = (rx, ry), d
        cx, cy = best
    else:
        ang = rng.uniform(0, 2 * math.pi)
        cx = (x0 + x1) / 2 + shift * ww * math.cos(ang)
        cy = (y0 + y1) / 2 + shift * wh * math.sin(ang)
    cx = min(max(cx, ww / 2), 1 - ww / 2)
    cy = min(max(cy, wh / 2), 1 - wh / 2)
    return cx - ww / 2, cy - wh / 2, cx + ww / 2, cy + wh / 2


def make_crop_image(sample, win, crop_tok):
    """Crop the window from the ORIGINAL image and size it to <= crop_tok LLM
    tokens (multiples of FACTOR, never upsampled beyond the original pixels)."""
    W, H = sample["image"].size
    x0, y0, x1, y1 = win
    px0, py0 = int(x0 * W), int(y0 * H)
    px1, py1 = max(int(math.ceil(x1 * W)), px0 + 1), max(int(math.ceil(y1 * H)), py0 + 1)
    crop = sample["image"].crop((px0, py0, min(px1, W), min(py1, H)))
    cw, ch = crop.size
    sc = math.sqrt(min(crop_tok * TOK_PX, cw * ch) / (cw * ch))
    nw = max(FACTOR, int(cw * sc) // FACTOR * FACTOR)
    nh = max(FACTOR, int(ch * sc) // FACTOR * FACTOR)
    return crop.resize((nw, nh), Image.BICUBIC)


def build_two_image_inputs(sample, lr_thw, second_img, processor, hr_processor, args,
                           device, dtype):
    """[LR image (exactly the LR branch's pixels), second_img] as a native
    two-image prompt, no HD side. --crop_pos before: the 2nd image precedes
    the question (question tokens can attend to it); after: it FOLLOWS the
    question (= single-pass KV-cache continuation; only the answer sees it).
    Returns the model kwargs that REPLACE the LR-only ones."""
    lr_img = sample["image"].resize((int(lr_thw[2]) * PATCH, int(lr_thw[1]) * PATCH),
                                    Image.BICUBIC)
    seconds = list(second_img) if isinstance(second_img, (list, tuple)) else [second_img]
    hint = {"type": "text", "text": "Zoomed-in view of the relevant region:" if len(seconds) == 1
            else "Zoomed-in views of candidate regions:"}
    second = [{"type": "image", "image": im} for im in seconds]
    question = {"type": "text", "text": sample["prompt"]}
    content = [{"type": "image", "image": lr_img}]
    if args.crop_pos == "before":
        content += [hint] + second + [question]
    elif args.crop_pos == "after":
        content += [question, hint] + second
    else:                                   # after_restate: question repeated after the 2nd image
        content += [question, hint] + second + [dict(question)]
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    # hr_processor keeps any FACTOR-aligned size in [TOK_PX, 100 Mpx] as is, so the
    # first image reproduces the LR grid exactly and the second keeps its budget.
    inp = hr_processor(text=[text], images=[lr_img] + seconds, return_tensors="pt")
    thw = inp["image_grid_thw"]
    st = STATE.setdefault("crop_stats", {"n": 0, "lr_same": 0, "tok": [], "area": []})
    st["n"] += 1
    st["lr_same"] += bool(torch.equal(thw[0].cpu(), lr_thw.cpu()))
    st["tok"].append(int(sum(int(t[1] * t[2]) for t in thw[1:])) // (MERGE * MERGE))
    st["area"].append(STATE.pop("crop_stats_area", float("nan")))   # window area fraction
    out = {}
    for k, v in inp.items():
        if not isinstance(v, torch.Tensor):
            continue
        if k == "pixel_values":
            out[k] = v.to(device=device, dtype=dtype)
        elif k in ("input_ids", "attention_mask", "image_grid_thw"):
            out[k] = v.to(device)
    return out


def build_crop_token_inputs(sample, idx, lr_thw, hd_w, hd_h, processor, hr_processor, args,
                            device, dtype):
    """--hd_source crop_tokens: the ORACLE window crop as the 2nd image, or
    None when the sample has no bbox. plain_tokens: the WHOLE image resized to
    the same token budget (control: 'any 2nd image' vs 'the right window').
    --crop_pad / --crop_hit_rate / --crop_shift degrade the oracle window in a
    controlled way (how good must a localiser be?)."""
    if args.hd_source == "plain_tokens":
        STATE["crop_stats_area"] = 1.0
        return build_two_image_inputs(sample, lr_thw,
                                      make_crop_image(sample, (0.0, 0.0, 1.0, 1.0), args.crop_tok),
                                      processor, hr_processor, args, device, dtype)
    win = crop_window(sample, hd_w, hd_h, args.oracle_min_cells, pad=args.crop_pad)
    if win is None:
        return None
    if args.crop_hit_rate < 1.0 or args.crop_shift > 0:
        rng = np.random.RandomState(2000 + idx)
        hit = bool(rng.uniform() < args.crop_hit_rate)
        win = corrupt_window(win, gt_box_frac(sample), rng, hit, args.crop_shift)
        cs = STATE.setdefault("crop_corrupt", {"hit": 0, "n": 0})
        cs["n"] += 1
        cs["hit"] += hit
    STATE["crop_stats_area"] = (win[2] - win[0]) * (win[3] - win[1])
    return build_two_image_inputs(sample, lr_thw, make_crop_image(sample, win, args.crop_tok),
                                  processor, hr_processor, args, device, dtype)


GROUND_PROMPT = ("To answer the question below, which part of the image has to be examined "
                 "closely? Output only the bounding box of that region as [x1, y1, x2, y2].\n"
                 "Question: {q}")
GROUND_PROMPT_TGT = ("Locate the following in the image and output only its bounding box as "
                     "[x1, y1, x2, y2]: {t}")


GROUND_PROMPT_TOPK = ("Locate the following in the image. Output up to {k} candidate bounding boxes, "
                      "most likely first, each as [x1, y1, x2, y2], one per line: {t}")
EXTRACT_PROMPT = ("Which object or region in the image must be looked at closely to answer the "
                  "question? Reply with only a short noun phrase, no explanation.\nQuestion: {q}")


def ground_prompt(sample, query, k=1):
    """target: the sidecar target_object names (V* only; an upper bound, the
    names are not available in the wild); question: the question's first line
    (no answer-format instructions, which the model would obey instead of
    producing a box); extract: a phrase the model itself produced from the
    question (sample['_extracted']) — the realistic single-model pipeline."""
    q = sample["prompt"].strip().split("\n")[0]
    if query == "target" and sample.get("targets"):
        t = ", ".join(sample["targets"])
    elif query == "extract" and sample.get("_extracted"):
        t = sample["_extracted"]
    else:
        return GROUND_PROMPT.format(q=q)
    return GROUND_PROMPT_TGT.format(t=t) if k <= 1 else GROUND_PROMPT_TOPK.format(t=t, k=k)


def lm_text(img, prompt_text, model, processor, hr_processor, device, dtype, max_new_tokens):
    """Greedy answer of the frozen model to (img, prompt_text); img is used at
    its own size (FACTOR-aligned by the caller)."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [{"type": "image", "image": img},
                                             {"type": "text", "text": prompt_text}]}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inp = hr_processor(text=[text], images=[img], return_tensors="pt")
    moved = {k: (v.to(device=device, dtype=dtype) if k == "pixel_values" else v.to(device))
             for k, v in inp.items()
             if isinstance(v, torch.Tensor) and k in ("input_ids", "attention_mask",
                                                      "pixel_values", "image_grid_thw")}
    with torch.inference_mode():
        out = model.generate(**moved, max_new_tokens=max_new_tokens, do_sample=False,
                             temperature=None, top_p=None, top_k=None)
    return processor.tokenizer.decode(out[0][moved["input_ids"].shape[1]:],
                                      skip_special_tokens=True).strip()


def parse_boxes(g, img_size, coords, max_boxes):
    """All bounding boxes in the model's text as (x0, y0, x1, y1) fractions of
    the image the model saw (plus the alternative coordinate reading of the
    first box, for the report). Explicit 4-number lists / point pairs first;
    otherwise the first four numbers after dropping key names like bbox_2d."""
    Wl, Hl = img_size
    num = r"(-?\d+(?:\.\d+)?)"
    quads = re.findall(rf"\[\s*{num}\s*,\s*{num}\s*,\s*{num}\s*,\s*{num}\s*\]", g) \
        or re.findall(rf"\(\s*{num}\s*,\s*{num}\s*\)\s*,\s*\(\s*{num}\s*,\s*{num}\s*\)", g)
    if not quads:
        nums = re.findall(r"-?\d+(?:\.\d+)?", re.sub(r"[A-Za-z_]+\d[A-Za-z_]*", " ", g))
        quads = [tuple(nums[:4])] if len(nums) >= 4 else []
    boxes, alt = [], None
    for q in quads[:max_boxes]:
        x1, y1, x2, y2 = (float(v) for v in q)
        if max(x1, y1, x2, y2) <= 1.0:
            fr, a = (x1, y1, x2, y2), None
        elif coords == "norm1000":
            fr, a = (x1 / 1000, y1 / 1000, x2 / 1000, y2 / 1000), (x1 / Wl, y1 / Hl, x2 / Wl, y2 / Hl)
        else:
            fr, a = (x1 / Wl, y1 / Hl, x2 / Wl, y2 / Hl), (x1 / 1000, y1 / 1000, x2 / 1000, y2 / 1000)
        fr = tuple(min(max(v, 0.0), 1.0) for v in fr)
        if fr[2] > fr[0] and fr[3] > fr[1]:
            boxes.append(fr)
            if alt is None:
                alt = a
    return boxes, alt


def box_hits(box, sample, hd_w, hd_h, min_cells):
    """(centre in GT window, centre in GT box, IoU with GT box) or None without GT."""
    gt = gt_box_frac(sample)
    if gt is None:
        return None
    gw = crop_window(sample, hd_w, hd_h, min_cells)
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    ix = max(0.0, min(box[2], gt[2]) - max(box[0], gt[0]))
    iy = max(0.0, min(box[3], gt[3]) - max(box[1], gt[1]))
    inter = ix * iy
    union = (box[2] - box[0]) * (box[3] - box[1]) + (gt[2] - gt[0]) * (gt[3] - gt[1]) - inter
    return ((gw[0] <= cx <= gw[2]) and (gw[1] <= cy <= gw[3]),
            (gt[0] <= cx <= gt[2]) and (gt[1] <= cy <= gt[3]),
            inter / max(union, 1e-9))


def pad_window(box, pad, min_cells, hd_w, hd_h):
    """(x0, y0, x1, y1) fractions -> window scaled by `pad` about its centre,
    at least min_cells HD cells per side, clamped into the image."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ww = min(max((x1 - x0) * pad, min_cells * FACTOR / hd_w), 1.0)
    wh = min(max((y1 - y0) * pad, min_cells * FACTOR / hd_h), 1.0)
    cx = min(max(cx, ww / 2), 1 - ww / 2)
    cy = min(max(cy, wh / 2), 1 - wh / 2)
    return cx - ww / 2, cy - wh / 2, cx + ww / 2, cy + wh / 2


def ground_window_inputs(sample, lr_thw, hd_w, hd_h, model, processor, hr_processor, args,
                         device, dtype):
    """--hd_source ground_window: ask the (frozen) model, on the LR image, for
    the bbox of the region it needs to look at; crop that (padded) window as
    the 2nd image. Zero-training localisation from the model's own grounding
    ability instead of its attention. Falls back to the whole image when no
    box can be parsed."""
    lr_img = sample["image"].resize((int(lr_thw[2]) * PATCH, int(lr_thw[1]) * PATCH),
                                    Image.BICUBIC)
    st = STATE.setdefault("ground_stats", {"n": 0, "parsed": 0, "in_win": 0, "in_box": 0,
                                           "iou": [], "raw": [], "alt_in_win": 0,
                                           "s1_in_win": 0, "any_in_win": 0, "n_boxes": [],
                                           "extracted": []})
    st["n"] += 1
    gen_ = lambda img, p, n: lm_text(img, p, model, processor, hr_processor, device, dtype, n)
    # --ground_query extract: the model names what to look for (from the
    # question alone), then grounds that phrase — no sidecar names involved
    if args.ground_query == "extract":
        q = sample["prompt"].strip().split("\n")[0]
        phrase = gen_(lr_img, EXTRACT_PROMPT.format(q=q), 16).split("\n")[0].strip().strip(".\"'")
        sample["_extracted"] = phrase or None
        if len(st["extracted"]) < 6:
            st["extracted"].append((q[:60], phrase[:40]))
    prompt_text = ground_prompt(sample, args.ground_query, args.ground_topk)
    # stage 1: boxes on the LR image
    g = gen_(lr_img, prompt_text, 48 if args.ground_topk <= 1 else 32 * args.ground_topk)
    if len(st["raw"]) < 4:
        st["raw"].append(g[:120])
    boxes, alt = parse_boxes(g, lr_img.size, args.ground_coords, args.ground_topk)
    if boxes and alt is not None:
        h = box_hits(alt, sample, hd_w, hd_h, args.oracle_min_cells)
        st["alt_in_win"] += bool(h and h[0])
    if boxes:
        h = box_hits(boxes[0], sample, hd_w, hd_h, args.oracle_min_cells)
        st["s1_in_win"] += bool(h and h[0])
    # stage 2 (--ground_stages 2): crop --ground_refine_pad x the first box
    # from the ORIGINAL image at <= --ground_refine_tok tokens (the target is
    # now several cells big) and ground again; the box maps back to the image
    s1_boxes, rw, g2 = list(boxes), None, ""
    if args.ground_stages >= 2 and boxes:
        rw = pad_window(boxes[0], args.ground_refine_pad, args.oracle_min_cells, hd_w, hd_h)
        rimg = make_crop_image(sample, rw, args.ground_refine_tok)
        g2 = gen_(rimg, ground_prompt(sample, args.ground_query, 1), 48)
        b2, _ = parse_boxes(g2, rimg.size, args.ground_coords, 1)
        if b2:
            x0, y0, x1, y1 = rw
            bx = b2[0]
            boxes = [(x0 + bx[0] * (x1 - x0), y0 + bx[1] * (y1 - y0),
                      x0 + bx[2] * (x1 - x0), y0 + bx[3] * (y1 - y0))] + boxes[1:]
    log = {"s1": s1_boxes, "final": list(boxes), "refine_win": rw, "raw": g[:80], "raw2": g2[:80],
           "phrase": sample.get("_extracted"), "hit": None, "iou": None, "wins": [], "crops": []}
    STATE.setdefault("ground_log", []).append(log)
    if boxes:
        st["parsed"] += 1
        st["n_boxes"].append(len(boxes))
        hits = [box_hits(b, sample, hd_w, hd_h, args.oracle_min_cells) for b in boxes]
        if hits[0] is not None:
            st["in_win"] += hits[0][0]
            st["in_box"] += hits[0][1]
            st["iou"].append(hits[0][2])
            st["any_in_win"] += any(h[0] for h in hits)
            log["hit"], log["iou"] = bool(hits[0][0]), float(hits[0][2])
        # one crop per box, the token budget split between them
        wins = [pad_window(b, args.ground_pad, args.oracle_min_cells, hd_w, hd_h) for b in boxes]
        imgs = [make_crop_image(sample, w, max(64, args.crop_tok // len(wins))) for w in wins]
        STATE["crop_stats_area"] = sum((w[2] - w[0]) * (w[3] - w[1]) for w in wins)
    else:
        wins = [(0.0, 0.0, 1.0, 1.0)]
        imgs = [make_crop_image(sample, wins[0], args.crop_tok)]
        STATE["crop_stats_area"] = 1.0
    log["wins"] = wins
    if args.dump_ground:
        log["crops"] = [im.copy() for im in imgs]
    return build_two_image_inputs(sample, lr_thw, imgs, processor, hr_processor, args, device, dtype)


# ── model-map -> second image (pred_window / pred_warp / warp_oracle) ──
PRED_SOURCES = ("pred_window", "pred_warp", "warp_oracle")


def _cell_grid(gs):
    """Cell-centre coordinates of the gs x gs relevance map in [-1, 1], row-major
    (y, x) like the model's p.view(R, gs, gs); same half-cell margin as
    _grid_generate / _quantile_grid."""
    m = 1.0 / max(gs - 1, 1)
    xs = np.linspace(-1 + m, 1 - m, gs)
    gx, gy = np.meshgrid(xs, xs)
    return xs, gx.ravel(), gy.ravel()


def model_map(sample_idx, layers):
    """Mean of the model's OWN relevance maps (question slot, pre-floor) at
    `layers` for the sample just processed; None if a layer did not record.
    Returns (p [N], gs)."""
    ms = []
    for l in layers:
        rows = PMAP.get(l)
        if not rows or len(rows) != sample_idx + 1:
            return None
        ms.append(np.asarray(rows[-1], dtype=np.float64))
    p = np.mean(ms, 0)
    p = p / max(p.sum(), 1e-12)
    gs = int(round(math.sqrt(p.size)))
    if gs * gs != p.size:
        return None
    return p, gs


def map_window(p, gs, floor_lam, min_cells, hd_w, hd_h):
    """The moments grid's extent (c +- s * ref half-extent, floor lam like the
    model's glob_floor) as (x0, y0, x1, y1) image fractions, min-cells / clamp
    like the oracle window."""
    N = gs * gs
    q = np.clip(p - floor_lam / N, 0.0, None)
    q = q / q.sum() if q.sum() > 1e-9 else p
    xs, gx, gy = _cell_grid(gs)
    cx, cy = float((q * gx).sum()), float((q * gy).sum())
    su = float(xs.std())
    sx = min(math.sqrt(float((q * (gx - cx) ** 2).sum())) / su, 1.0)
    sy = min(math.sqrt(float((q * (gy - cy) ** 2).sum())) / su, 1.0)
    half = 1.0 - 1.0 / max(gs - 1, 1)                       # ref grid half-extent
    fx0, fx1 = (cx - sx * half + 1) / 2, (cx + sx * half + 1) / 2
    fy0, fy1 = (cy - sy * half + 1) / 2, (cy + sy * half + 1) / 2
    ww = min(max(fx1 - fx0, min_cells * FACTOR / hd_w), 1.0)
    wh = min(max(fy1 - fy0, min_cells * FACTOR / hd_h), 1.0)
    mx_ = min(max((fx0 + fx1) / 2, ww / 2), 1 - ww / 2)
    my_ = min(max((fy0 + fy1) / 2, wh / 2), 1 - wh / 2)
    return mx_ - ww / 2, my_ - wh / 2, mx_ + ww / 2, my_ + wh / 2


def box_map(sample, gs, mass):
    """Synthetic relevance map: (1 - mass) uniform + mass on the cells whose
    centres fall in the GT box (nearest cell when the box is tiny)."""
    W, H = sample["image"].size
    bs = sample["bboxes"]
    bx = np.array([min(b[0] for b in bs) / W, min(b[1] for b in bs) / H,
                   max(b[0] + b[2] for b in bs) / W, max(b[1] + b[3] for b in bs) / H]) * 2 - 1
    _, gx, gy = _cell_grid(gs)
    inb = (gx >= bx[0]) & (gx <= bx[2]) & (gy >= bx[1]) & (gy <= bx[3])
    if not inb.any():
        c = np.array([(bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2])
        inb[np.argmin((gx - c[0]) ** 2 + (gy - c[1]) ** 2)] = True
    N = gs * gs
    return (1 - mass) / N + mass * inb / inb.sum()


def warp_image(img, p, gs, lam, out_tok):
    """Pixel-level rectilinear warp of the WHOLE image: per-axis marginals of
    (lam * p + (1 - lam) * uniform) -> CDF -> piecewise-linear inverse at the
    output pixel centres -> bilinear resample. Output area <= out_tok LLM
    tokens (image aspect, FACTOR-aligned). lam = 0 is a plain resize."""
    import torch.nn.functional as Fnn
    N = gs * gs
    pm = (lam * p + (1 - lam) / N).reshape(gs, gs)          # [y, x]
    mx, my = pm.sum(0), pm.sum(1)
    W, H = img.size
    sc = math.sqrt(min(out_tok * TOK_PX, W * H) / (W * H))
    ow = max(FACTOR, int(W * sc) // FACTOR * FACTOR)
    oh = max(FACTOR, int(H * sc) // FACTOR * FACTOR)
    # anti-alias: pre-shrink so that even the most compressed region is not
    # sampled far below the source density (4x the output area)
    pre = math.sqrt(min(1.0, 4.0 * ow * oh / (W * H)))
    if pre < 1.0:
        img = img.resize((max(1, int(W * pre)), max(1, int(H * pre))), Image.BICUBIC)
    edges = np.linspace(-1.0, 1.0, gs + 1)

    def inv_cdf(marg, n_out):
        cdf = np.concatenate([[0.0], np.cumsum(marg / marg.sum())])
        cdf[-1] = 1.0
        u = (np.arange(n_out) + 0.5) / n_out
        return np.interp(u, cdf, edges)

    xs = torch.tensor(inv_cdf(mx, ow), dtype=torch.float32)
    ys = torch.tensor(inv_cdf(my, oh), dtype=torch.float32)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([gx, gy], -1)[None]                   # [1, oh, ow, 2] (x, y)
    t = torch.from_numpy(np.asarray(img, dtype=np.float32)).permute(2, 0, 1)[None] / 255.0
    out = Fnn.grid_sample(t, grid, mode="bilinear", padding_mode="border", align_corners=False)
    arr = (out[0].permute(1, 2, 0).clamp(0, 1).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def build_pred_inputs(sample, sample_idx, lr_thw, hd_w, hd_h, processor, hr_processor, args,
                      device, dtype):
    """Second image from the model's OWN map (pred_window: moments window crop;
    pred_warp: whole-image rectilinear warp) or from a synthetic GT-box map
    (warp_oracle). None when no map / bbox is available."""
    if args.hd_source == "warp_oracle":
        if not sample.get("bboxes"):
            return None
        gs = args.grid_size
        img2 = warp_image(sample["image"], box_map(sample, gs, args.warp_box_mass), gs, 1.0,
                          args.crop_tok)
        STATE["crop_stats_area"] = float("nan")
    else:
        mp = model_map(sample_idx, args.map_layers)
        if mp is None:
            return None
        p, gs = mp
        if args.hd_source == "pred_window":
            win = map_window(p, gs, args.map_floor, args.oracle_min_cells, hd_w, hd_h)
            STATE["crop_stats_area"] = (win[2] - win[0]) * (win[3] - win[1])
            img2 = make_crop_image(sample, win, args.crop_tok)
        else:
            img2 = warp_image(sample["image"], p, gs, args.warp_lam, args.crop_tok)
            STATE["crop_stats_area"] = float("nan")
    return build_two_image_inputs(sample, lr_thw, img2, processor, hr_processor, args,
                                  device, dtype)


# ──────────────────────────────────────────────────────────────
# --hd_source sampled_tokens: DAT's OWN sampled HD features (bilinear
# grid_sample on the merger output, exactly what k_proj_hd / v_proj_hd see)
# placed IN THE SEQUENCE as ordinary image tokens instead of the K/V bypass.
# Separates "deformable sampling of features" from "the bypass readout":
#   samp_oracle  : the oracle window grid (= --hd_source oracle's forced grid)
#                  vs crop_tokens of the same window (re-encoded pixel crop)
#   samp_pred    : the model's own deformable grid (answer slot, --samp_layer)
#   samp_uniform : the identity grid over the whole HD map vs plain_tokens
# Positions (--samp_pos): square = the tokens pose as a gs x gs image (what a
# native crop gets); lr = the LR image's mRoPE frame at the sampled location
# (hd_position_mode='sampled'). Zero new parameters.
# ──────────────────────────────────────────────────────────────
PLOCS = defaultdict(list)        # layer_idx -> [answer-slot grid [G, gh, gw, 2] (x, y) in [-1, 1], ...]
SAMP_GRIDS = ("oracle", "pred", "uniform")


def samp_locs(kind, sample, idx, hd_w, hd_h, args):
    """Sampling grid for one sample as [G, Ns, 2] (x, y) in [-1, 1]; None when
    unavailable (no bbox for oracle, no recorded grid for pred)."""
    gs = args.grid_size
    if kind == "oracle":
        loc = oracle_locs(sample, idx, hd_w, hd_h, gs, args.oracle_min_cells, "oracle")
        return None if loc is None else loc[None]
    if kind == "uniform":
        m = 1.0 / max(gs - 1, 1)
        xs = torch.linspace(-1 + m, 1 - m, gs)
        gy, gx = torch.meshgrid(xs, xs, indexing="ij")
        return torch.stack([gx, gy], -1).reshape(1, -1, 2)
    rows = PLOCS.get(args.samp_layer)
    if not rows or len(rows) != idx + 1:
        return None
    return rows[-1].reshape(rows[-1].size(0), -1, 2)


def sample_hd_tokens(feat, locs_g, mode="bilinear"):
    """feat [H, W, C] merger output, locs_g [G, Ns, 2] -> [Ns, C]: group g's
    channel block sampled at group g's points, bilinear, align_corners=True —
    the model's _sample_hd_from_off_guide without the K/V projections.
    mode='nearest' snaps every point to its cell (no feature blending)."""
    H, W, C = feat.shape
    G = locs_g.size(0)
    chunks = feat.float().permute(2, 0, 1).reshape(G, C // G, H, W)       # 'h w (g c) -> g c h w'
    grid = locs_g.to(chunks.device).float().clamp(-1, 1)[:, None]          # [G, 1, Ns, 2]
    out = Fnn.grid_sample(chunks, grid, mode=mode, align_corners=True)     # [G, C/G, 1, Ns]
    return out[:, :, 0].permute(2, 0, 1).reshape(locs_g.size(1), C).to(feat.dtype)


def install_samp_hooks(model, image_token_id):
    """(1) visual forward hook: replace the LAST Ns merged tokens (the dummy
    2nd image) with STATE['samp_inject']; (2) get_rope_index wrapper: under
    STATE['samp_rope'] give those tokens the LR image's mRoPE frame."""
    def vis_hook(module, args_, output):
        inj = STATE.get("samp_inject")
        if inj is None:
            return None
        ns, tot = inj.size(0), STATE["samp_total"]

        def fix(t):
            # one ViT call for both images (rows = all merged tokens) or one
            # call per image (the dummy's own rows); never the LR image alone
            if torch.is_tensor(t) and t.dim() == 2 and (t.size(0) == tot or
                                                        (t.size(0) == ns and tot - ns != ns)):
                t = t.clone()
                t[-ns:] = inj.to(t.dtype)
                STATE["samp_inject_n"] = STATE.get("samp_inject_n", 0) + 1
            return t
        if torch.is_tensor(output):
            return fix(output)
        if isinstance(output, tuple):
            return tuple(fix(t) for t in output)
        for k_ in list(output.keys()):                      # ModelOutput
            v = output[k_]
            if torch.is_tensor(v):
                output[k_] = fix(v)
            elif isinstance(v, (list, tuple)) and v and torch.is_tensor(v[0]) \
                    and sum(x.size(0) for x in v) == tot:
                cat = fix(torch.cat(list(v), 0))
                output[k_] = list(torch.split(cat, [x.size(0) for x in v]))
        return output
    model.model.visual.register_forward_hook(vis_hook)
    # the LR-frame positions are applied in the rotary hook of
    # install_merge_hook (samp_rope_positions): the position ids reach the
    # rotary embedding whatever path computed them (get_rope_index is bypassed
    # by generate() on some transformers versions: 0 calls seen on the cluster)


def samp_rope_positions(position_ids):
    """Under STATE['samp_rope'] (prefill of a sampled_tokens @lr prompt): the
    dummy image's tokens get the LR image's mRoPE frame at their sampling
    location (t = LR t, h/w = fraction of the LR h/w range, rounded) — the
    formula of _hd_position_ids_from_locs. Handles [3, B, L] and [4, B, L]
    (text counter + mRoPE) position ids; decode steps (L = 1) untouched."""
    sr = STATE.get("samp_rope")
    if sr is None or position_ids.dim() != 3 or position_ids.size(-1) != sr["L"]:
        return position_ids
    o = position_ids.size(0) - 3
    if o not in (0, 1):
        return position_ids
    pid = position_ids.clone()
    dev = pid.device
    lr_i, hd_i = sr["lr_i"].to(dev), sr["hd_i"].to(dev)
    lr_pos = pid[o:, 0, lr_i].float()
    h_min, h_max = lr_pos[1].min(), lr_pos[1].max()
    w_min, w_max = lr_pos[2].min(), lr_pos[2].max()
    fx, fy = sr["fx"].to(dev), sr["fy"].to(dev)
    pid[o + 1, 0, hd_i] = (fy * (h_max - h_min) + h_min).round().to(pid.dtype)
    pid[o + 2, 0, hd_i] = (fx * (w_max - w_min) + w_min).round().to(pid.dtype)
    pid[o, 0, hd_i] = lr_pos[0, 0].to(pid.dtype)
    STATE["samp_rope_n"] = STATE.get("samp_rope_n", 0) + 1
    return pid


def tiled_hd_features(model, hd_img, hr_processor, device, dtype, tile_cells):
    """--samp_vit tiles: the HD feature map assembled from INDEPENDENT ViT
    runs on tiles of <= tile_cells x tile_cells merged cells (AnyRes-style:
    each tile is a small image to the ViT, no whole-image context / position
    interpolation). Same [H, W, C] layout as the model's single HD call."""
    W, H = hd_img.size
    cw, ch = W // FACTOR, H // FACTOR
    xs = np.array_split(np.arange(cw), max(1, math.ceil(cw / tile_cells)))
    ys = np.array_split(np.arange(ch), max(1, math.ceil(ch / tile_cells)))
    feat = None
    for ys_ in ys:
        for xs_ in xs:
            tile = hd_img.crop((int(xs_[0]) * FACTOR, int(ys_[0]) * FACTOR,
                                int(xs_[-1] + 1) * FACTOR, int(ys_[-1] + 1) * FACTOR))
            hr = hr_processor.image_processor(images=[tile], return_tensors="pt")
            f = model._generate_hd_features(hr["pixel_values"].to(device=device, dtype=dtype),
                                            hr["image_grid_thw"].to(device))[0]
            assert tuple(f.shape[:2]) == (len(ys_), len(xs_)), (f.shape, len(ys_), len(xs_), tile.size)
            if feat is None:
                feat = f.new_empty(ch, cw, f.size(-1))
            feat[int(ys_[0]):int(ys_[-1]) + 1, int(xs_[0]):int(xs_[-1]) + 1] = f
    return feat, len(xs) * len(ys)


def build_samp_inputs(sample, lr_thw, kind, locs_g, hd_feat, pos_mode, processor, hr_processor,
                      args, device, dtype, image_token_id):
    """Two-image prompt whose 2nd image is a gs x gs DUMMY (its ViT output is
    overwritten by the sampled HD tokens via the hooks)."""
    gs = args.grid_size
    dummy = Image.new("RGB", (gs * FACTOR, gs * FACTOR), (127, 127, 127))
    if kind == "oracle":
        lo, hi = locs_g[0].min(0).values, locs_g[0].max(0).values
        STATE["crop_stats_area"] = float(((hi[0] - lo[0]) / 2) * ((hi[1] - lo[1]) / 2))
    else:
        STATE["crop_stats_area"] = float("nan")
    inp = build_two_image_inputs(sample, lr_thw, dummy, processor, hr_processor, args, device, dtype)
    thw = inp["image_grid_thw"]
    assert int(thw[1][1] * thw[1][2]) // (MERGE * MERGE) == gs * gs, thw
    STATE["samp_inject"] = sample_hd_tokens(hd_feat, locs_g, args.samp_mode)
    STATE["samp_total"] = int((thw.prod(-1) // (MERGE * MERGE)).sum())
    fl = locs_g.float().mean(0).clamp(-1, 1)
    if pos_mode == "square":
        STATE["samp_rope"] = None
    else:
        ids = inp["input_ids"][0]
        img = (ids == image_token_id).nonzero().flatten()
        n_lr = int(thw[0][1] * thw[0][2]) // (MERGE * MERGE)
        assert img.numel() == n_lr + gs * gs, (img.numel(), n_lr, gs * gs)
        STATE["samp_rope"] = {"L": int(ids.numel()), "lr_i": img[:n_lr].cpu(), "hd_i": img[n_lr:].cpu(),
                              "fx": (fl[:, 0] + 1) * 0.5, "fy": (fl[:, 1] + 1) * 0.5}
    return inp


def clear_samp():
    STATE["samp_inject"] = None
    STATE["samp_rope"] = None


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

    moved = {}
    for k, v in inputs.items():
        if not isinstance(v, torch.Tensor):
            continue
        if k == "pixel_values":
            moved[k] = v.to(device=device, dtype=dtype)
        elif k in ("input_ids", "attention_mask", "image_grid_thw"):
            moved[k] = v.to(device)
    STATE["geom"] = (lr_thw, hd_w, hd_h)
    if args.hd_source == "ground_window":
        hd_extra = {}                        # built in the main loop (needs a generate call)
    elif args.hd_source in ("crop_tokens", "plain_tokens"):
        # no HD side at all; the "HD-on" config is the two-image prompt
        hd_extra = build_crop_token_inputs(sample, idx, lr_thw, hd_w, hd_h, processor,
                                           hr_processor, args, device, dtype) or {}
    else:
        # pred_* / warp_oracle: the HD pass runs on the real image (to record
        # the model's own map); the second-image prompt is built afterwards
        src = "real" if args.hd_source in PRED_SOURCES + ("sampled_tokens",) else args.hd_source
        hd_img = make_hd_image(sample, idx, samples, lr_thw, hd_w, hd_h, src)
        hr_inputs = hr_processor.image_processor(images=[hd_img], return_tensors="pt")
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
        STATE["pending_locs"] = []                    # diag 14: HD rotary calls of this layer are over
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
            STATE["orig_p"] = p.detach()            # the model's own map, for the map tables / hit split
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
            if kind == "gate":                      # gate:<k>:<thr>  unimodal -> win:k, else identity
                k, thr = rest.split(":")
                pf = floor(p, 1.0)
                m = argmax_window(p, int(k))
                conc = (pf * m).sum(-1, keepdim=True)              # floored mass around the peak
                on = conc >= float(thr)
                STATE.setdefault("gate_log", []).append(float(on.float().mean()))
                return torch.where(on, m / m.sum(-1, keepdim=True), torch.full_like(p, 1.0 / N))
            if kind == "warp":
                # the grid itself was replaced (see warp_grid); a flat map
                # makes the model's moments the identity (c = 0, s = 1)
                return torch.full_like(p, 1.0 / N)
            raise ValueError(f"unknown --glob_readout {mode}")
        return f

    # Diagnostic 13 (--glob_readout warp[:<lam>]): the QUANTILE grid of the
    # map instead of its moments. The model fits a Gaussian to p (c = mean,
    # s = std) and needed a floor because 'peak + flat background' is not
    # one; here p is a sampling density: per axis take the marginal, its
    # CDF, and put the i-th grid line at CDF^-1 of the i-th uniform-grid
    # quantile (piecewise-linear inverse, p constant inside a cell). Dense
    # cells get many lines, the flat background keeps its proportional share
    # (coverage for the misses), nothing is fitted, no constants. Monotone,
    # so the grid topology (and the dw-conv local offsets added on top) is
    # unchanged. Saliency-sampler style (Recasens 2018 / FOVEA 2021) with the
    # model's own attention as the saliency. lam < 1 mixes the map with the
    # uniform density (minimum coverage); lam = 1 = pure.
    def warp_grid(p, gs, lam):
        # p [R, N] -> warped reference grid [R, 2, gs, gs] in (x, y)
        R, N = p.shape
        if lam < 0:
            # 'warp:-1' = data-driven background: subtract each row's median
            # cell value (the flat level, estimated per sample instead of the
            # constant 1/N of glob_floor), relu, renormalise
            q = (p - p.median(-1, keepdim=True).values).clamp_min(0)
            z = q.sum(-1, keepdim=True)
            p = torch.where(z > 1e-9, q / z.clamp_min(1e-9), p)
        elif lam < 1.0:
            p = lam * p + (1.0 - lam) / N
        pm = p.view(R, gs, gs)                                       # [R, y, x]
        m = 1.0 / max(gs - 1, 1)
        ref = torch.linspace(-1.0 + m, 1.0 - m, gs, device=p.device, dtype=p.dtype)
        u = ((ref + 1.0) * 0.5).clamp(1e-6, 1 - 1e-6)                # [gs] uniform-grid quantiles
        edges = torch.linspace(-1.0, 1.0, gs + 1, device=p.device, dtype=p.dtype)

        def inv_cdf(marg):                                           # marg [R, gs] -> [R, gs] positions
            marg = marg / marg.sum(-1, keepdim=True).clamp_min(1e-9)
            F = torch.cat([marg.new_zeros(R, 1), marg.cumsum(-1)], -1)   # [R, gs+1], F[-1] = 1
            F[:, -1] = 1.0
            k = (torch.searchsorted(F, u.expand(R, gs).contiguous(), right=True) - 1).clamp(0, gs - 1)
            F0, F1 = F.gather(1, k), F.gather(1, k + 1)
            frac = (u - F0) / (F1 - F0).clamp_min(1e-9)
            return edges[k] + frac.clamp(0, 1) * (2.0 / gs)

        X = inv_cdf(pm.sum(1))                                       # marginal over x: sum over y
        Y = inv_cdf(pm.sum(2))
        gx = X[:, None, :].expand(R, gs, gs)
        gy = Y[:, :, None].expand(R, gs, gs)
        return torch.stack([gx, gy], 1)

    def patched_sample(self, *a, **kw):
        STATE["pending_locs"] = []                    # diag 14: never let stale locations leak
        mode = STATE.get("glob_readout") or ""
        if mode or STATE.get("glob_layers") is not None:
            STATE["orig_p"] = None
            torch.softmax = glob_softmax_for(self.layer_idx, self.grid_size)
            warped = None
            layers = STATE.get("glob_layers")
            if mode.startswith("warp") and (layers is None or self.layer_idx in layers) \
                    and kw.get("glob_logits") is not None:
                assert getattr(self, "conv_glob", None) is None, "warp probe needs a pure attn/qk map"
                lam = float(mode.partition(":")[2] or 1.0)
                with torch.no_grad():
                    gl = kw["glob_logits"].float().flatten(1)                     # [R, N]
                    warped = warp_grid(orig_softmax(gl, -1), self.grid_size, lam)  # [R, 2, gs, gs]
                    # the model reads the cell coordinates g from row 0 of the
                    # reference grid and adds c = E_uniform[g] = mean(row 0):
                    # pre-shift every row by -mean(row 0)/2 so x = warped + offsets exactly
                    c0 = warped[:1].flatten(2).mean(-1)                           # [1, 2]
                    ref_out = warped - 0.5 * c0[:, :, None, None]
                self._grid_generate = lambda h, w, n, device, _r=ref_out: _r
            try:
                key_hd, value_hd, locs = orig_sample(self, *a, **kw)
            finally:
                torch.softmax = orig_softmax
                if warped is not None:
                    del self._grid_generate                   # back to the class method
            # map-level tables and the hit/miss split always describe the
            # model's OWN map; only c / s / the grid follow the override
            op = STATE.get("orig_p")
            lp = getattr(self, "_dat_glob_last_p", None)
            if op is not None and lp is not None:
                self._dat_glob_last_p = (op, lp[1])
            if warped is not None:
                # effective centre / scale of the quantile grid for the
                # sample-dependence table: mean position, std / std(uniform)
                wf = warped.flatten(2)                                            # [R, 2, N]
                ref_u = self.__class__._grid_generate(self, self.grid_size, self.grid_size, 1, warped.device)[:1].flatten(2)
                c_eff = wf.mean(-1)
                s_eff = (wf.std(-1, unbiased=False) / ref_u.std(-1, unbiased=False)).clamp(0.0, 1.0)
                self._dat_glob_last_cs = (c_eff, s_eff)
                self._dat_glob_last = torch.stack([c_eff.norm(dim=1).mean(), s_eff.mean()])
        else:
            key_hd, value_hd, locs = orig_sample(self, *a, **kw)
        if STATE.get("hd_pos", "legacy") != "legacy" and getattr(self, "_dat_force_locs", None) is None:
            # Diagnostic 14: hand the ACTUAL sampling locations (one grid per
            # slot, groups averaged) to the rotary hook below, in slot order
            STATE["pending_locs"] = [locs[l].float().mean(0).reshape(-1, 2) for l in range(locs.size(0))]
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
                # the model's own map, recorded for every dataset (pred_* needs it
                # on bbox-free sets such as HR-Bench too)
                pg0 = getattr(self, "_dat_glob_last_p", None)
                if pg0 is not None:
                    pm0 = pg0[0]
                    if pm0.size(0) > self.off_grps:
                        pm0 = pm0[self.off_grps:]
                    PMAP[self.layer_idx].append(pm0.mean(0).float().cpu().numpy())
                # the model's own sampling grid (answer slot), for sampled_tokens
                PLOCS[self.layer_idx].append(locs[-1].float().cpu())
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
                    # Diagnostic 12 (coarse-to-fine c): hand the answer slot's
                    # sampling locations to the cross-attention wrapper below,
                    # which sees this layer's HD keys and the prompt-end query.
                    if locs.size(0) > 1:
                        aloc = locs[1].float().mean(0).reshape(-1, 2)      # [Ns, 2] (x, y), groups averaged
                        pib = ((aloc[:, 0] >= box[0]) & (aloc[:, 0] <= box[2]) &
                               (aloc[:, 1] >= box[1]) & (aloc[:, 1] <= box[3]))
                        s_lr = float(s.mean()) if cs is not None else 1.0    # this layer's coarse scale
                        STATE["hd_ctx"] = (self.layer_idx, aloc, tgt.reshape(-1, 2).mean(0), box, pib, s_lr)
        return key_hd, value_hd, locs

    Qwen3_5AttentionDAT._sample_hd_from_off_guide = patched_sample

    # Diagnostic 14 (--hd_pos sampled | sampled_frac): truthful HD positions.
    # _construct_hd_position_ids gives the model's OWN grid a fixed linspace
    # over the whole LR position range, whatever c / s / warp actually did
    # (only a teacher-FORCED grid gets positions from its real locations).
    # So at inference the mRoPE positions of the 400 HD keys claim 'uniform,
    # full image' while the keys come from a zoomed (and, under warp, non-
    # uniformly spaced) window; in training half the samples (forced) carry
    # truthful positions and half do not. Here the positions are rebuilt from
    # the actual sampling locations exactly as the forced path does
    # (sampled = rounded to integer LR units like the forced path;
    # sampled_frac = fractional). Inference-only, so read it as 'does the
    # readout already prefer truthful positions': hits up -> make the model
    # do this (hd_position_mode='sampled') and retrain.
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
    orig_rot_fwd = Qwen3_5TextRotaryEmbedding.forward

    def patched_rot_fwd(self, x, position_ids):
        position_ids = samp_rope_positions(position_ids)       # sampled_tokens @lr
        pend = STATE.get("pending_locs")
        if pend and position_ids.dim() == 3 and position_ids.size(1) == 1 \
                and position_ids.size(-1) == pend[0].size(0):
            loc = pend.pop(0).to(position_ids.device)
            pid = position_ids.float()
            h_min, h_max = pid[1].min(), pid[1].max()
            w_min, w_max = pid[2].min(), pid[2].max()
            fx = (loc[:, 0].clamp(-1, 1) + 1.0) * 0.5
            fy = (loc[:, 1].clamp(-1, 1) + 1.0) * 0.5
            h = fy * (h_max - h_min) + h_min
            w = fx * (w_max - w_min) + w_min
            new = torch.stack([pid[0, 0], h, w])[:, None, :]           # [3, 1, Ns]
            position_ids = new if STATE["hd_pos"] == "sampled_frac" else new.round().long()
            STATE["hd_pos_n"] = STATE.get("hd_pos_n", 0) + 1
        return orig_rot_fwd(self, x, position_ids)

    Qwen3_5TextRotaryEmbedding.forward = patched_rot_fwd

    # Diagnostic 12: could a SECOND localisation stage read c off the sampled
    # HD keys? At each DAT layer the answer segment's query (prompt end =
    # the model's ans_prev token) attends to the Ns HD keys sampled by this
    # layer's coarse grid (c +- s). Its attention is a relevance map over HD
    # POSITIONS (the sampling points, HD-cell precision) instead of LR cells:
    # a centroid over it is confined to the coarse window but can, inside it,
    # be as precise as the HD grid. Parameter-free, same q/k the model uses
    # (post-RoPE, k_norm'ed HD keys). Recorded per sample:
    #   c_hd  (head-mean attention centroid), c_hdF (same after the uniform
    #   floor), argmax point in the GT box, attention mass in the box, and the
    #   fraction of sampling points in the box (= what a flat attention gives).
    orig_xattn = _mod._dat_cross_attn_varlen

    def patched_xattn(seg_q_list, seg_k_list, seg_v_list, *a, **kw):
        ctx = STATE.pop("hd_ctx", None) if STATE["record"] else None
        if ctx is not None:
            lid, aloc, tc, box, pib, s_lr = ctx
            Ns = aloc.size(0)
            # the answer segment: HD keys of the answer slot (Ns rows) queried
            # by the few tokens after <|im_start|> ('assistant', '\n'); the
            # question segment has the same key count but a long query, the
            # LR-inject segment a long query too.
            cand = [(q, k) for q, k in zip(seg_q_list, seg_k_list)
                    if k.shape[0] == Ns and q.shape[0] <= 8]
            if cand:
                with torch.no_grad():
                    q, k = cand[-1]
                    ql = q[-1].float()                                   # [H, D]  prompt-end token
                    kk = k.float()                                       # [Ns, H, D]
                    att = torch.einsum('hd,nhd->hn', ql, kk) / math.sqrt(ql.size(-1))
                    att = att.softmax(-1)                                # [H, Ns]
                    w = att.mean(0)                                      # [Ns]
                    wf = (w - 1.0 / Ns).clamp_min(0)
                    wf = wf / wf.sum().clamp_min(1e-9) if wf.sum() > 1e-9 else w
                    al = aloc.to(w.device)
                    c_hd = (w[:, None] * al).sum(0)
                    c_hdf = (wf[:, None] * al).sum(0)
                    pk = int(w.argmax())
                    pibd = pib.to(w.device)
                    # spread of the HD attention in coarse-grid units: a fine
                    # scale candidate (std of the attended positions / std of
                    # the uniform reference grid)
                    var_w = (wf[:, None] * (al - c_hdf) ** 2).sum(0)      # [2]
                    var_u = (al - al.mean(0)).pow(2).mean(0).clamp_min(1e-9)
                    s_hd = float((var_w / var_u).sqrt().mean()) * s_lr    # in whole-image units
                    HDC[lid].append((c_hd[0].item(), c_hd[1].item(), c_hdf[0].item(), c_hdf[1].item(),
                                     float(pibd[pk]), float((w * pibd).sum()), float(pibd.float().mean()),
                                     tc[0].item(), tc[1].item(),
                                     box.float().cpu().tolist(), s_lr, s_hd))
        return orig_xattn(seg_q_list, seg_k_list, seg_v_list, *a, **kw)

    _mod._dat_cross_attn_varlen = patched_xattn


KHD = defaultdict(list)          # layer_idx -> [(shared/resid ratio, mean |offset|), ...]
KLR = defaultdict(list)          # layer_idx -> [shared/resid ratio of the LR image keys, ...]
LOC = defaultdict(list)          # layer_idx -> [(dist, dist_uniform, in_box, in_box_uniform), ...]
GLOB = defaultdict(list)         # layer_idx -> [(mean |centroid|, mean scale), ...]  (global offset term)
GLOBC = defaultdict(list)        # layer_idx -> [(cx, cy, sx, sy, tcx, tcy, tsx, tsy), ...] per sample
GLOBP = defaultdict(list)        # layer_idx -> [(relevance map [N], in-window mask [N], in-box mask [N]), ...]
PMAP = defaultdict(list)         # layer_idx -> [relevance map [N], ...] per sample, bbox or not
HDC = defaultdict(list)          # layer_idx -> [(c_hd x, y, c_hdF x, y, peak_in_box, mass_in_box, pts_in_box, tcx, tcy), ...]


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
                    choices=["real", "lr_up", "shuffle", "noise", "oracle", "oracle_rand",
                             "crop_tokens", "plain_tokens", "pred_window", "pred_warp",
                             "warp_oracle", "ground_window", "sampled_tokens"],
                    default="real",
                    help="oracle/oracle_rand need --dataset vstar (GT bboxes): HD is the real "
                         "image but the 400 sampling points are forced onto a window around the "
                         "target (oracle) or a same-size window elsewhere (oracle_rand). "
                         "crop_tokens = readout CEILING: no HD pass at all; the oracle window is "
                         "cropped from the original image and fed as a SECOND image of <= "
                         "--crop_tok LLM tokens (frozen trunk, native multi-image path). Same "
                         "region as 'oracle', but as ordinary tokens in the sequence instead of "
                         "400 cross-attended keys")
    ap.add_argument("--crop_tok", type=int, default=400,
                    help="crop_tokens / pred_* / warp_oracle: LLM-token budget of the 2nd image "
                         "(400 = the 20x20 sampling grid's key count)")
    ap.add_argument("--crop_pos", choices=["before", "after", "after_restate"], default="before",
                    help="2nd image before the question (native multi-image), AFTER it "
                         "(single-pass KV-cache continuation: only the answer sees it), or after "
                         "it with the question repeated behind the image (after_restate)")
    ap.add_argument("--map_layers", default="11,15,19,23",
                    help="pred_*: DAT layers whose (question-slot) relevance maps are averaged")
    ap.add_argument("--map_floor", type=float, default=1.0,
                    help="pred_window: glob_floor lambda applied to the averaged map before c/s")
    ap.add_argument("--warp_lam", type=float, default=0.5,
                    help="pred_warp: map mixed with uniform, lam * p + (1 - lam) / N (0 = plain resize)")
    ap.add_argument("--warp_box_mass", type=float, default=0.5,
                    help="warp_oracle: relevance mass placed on the GT-box cells (rest uniform)")
    ap.add_argument("--crop_pad", type=float, default=1.0,
                    help="crop_tokens: GT box scaled by this factor before the min-cells rule "
                         "(looser window = less zoom; how tight must the localiser be?)")
    ap.add_argument("--crop_hit_rate", type=float, default=1.0,
                    help="crop_tokens: fraction of samples that get the GT window; the rest get a "
                         "same-size window elsewhere (how often must the localiser be right?)")
    ap.add_argument("--crop_shift", type=float, default=0.0,
                    help="crop_tokens: move the window centre by this x window size in a random "
                         "direction (how precise must the localiser be?)")
    ap.add_argument("--samp_grids", default="oracle,pred,uniform",
                    help="sampled_tokens: which sampling grids to place in the sequence "
                         "(oracle = GT window grid, pred = the model's own deformable grid, "
                         "uniform = identity grid over the whole HD map)")
    ap.add_argument("--samp_pos", default="square,lr",
                    help="sampled_tokens: mRoPE positions of the sampled tokens: square = a gs x gs "
                         "image block (what a native crop gets), lr = the LR image's frame at the "
                         "sampled location (hd_position_mode='sampled')")
    ap.add_argument("--samp_vit", choices=["whole", "tiles"], default="whole",
                    help="sampled_tokens: HD feature map from the model's single whole-image ViT "
                         "call (as DAT does) or stitched from independent ViT runs on tiles "
                         "(is the whole-image ViT context what makes sampled features weak?)")
    ap.add_argument("--samp_mode", choices=["bilinear", "nearest"], default="bilinear",
                    help="sampled_tokens: bilinear = the model's sampling (points between cells blend "
                         "neighbouring features); nearest = snap to the cell")
    ap.add_argument("--samp_tile_cells", type=int, default=20,
                    help="sampled_tokens --samp_vit tiles: max merged cells per tile side")
    ap.add_argument("--samp_layer", type=int, default=15,
                    help="sampled_tokens pred: DAT layer whose answer-slot grid is used")
    ap.add_argument("--ground_coords", choices=["norm1000", "pixels"], default="norm1000",
                    help="ground_window: coordinate frame of the model's bbox output (Qwen3-VL "
                         "lineage = 0..1000 normalised; Qwen2.5-VL = pixels of the input image)")
    ap.add_argument("--ground_pad", type=float, default=2.0,
                    help="ground_window: window side = pad x the predicted box side (context)")
    ap.add_argument("--ground_query", choices=["target", "question", "extract"], default="target",
                    help="ground_window: what the model is asked to locate: the sidecar target names "
                         "(V* upper bound), the question's first line, or a phrase the model itself "
                         "extracts from the question first (realistic pipeline)")
    ap.add_argument("--ground_stages", type=int, default=1,
                    help="ground_window: 2 = ground again on a crop around the first box "
                         "(coarse-to-fine)")
    ap.add_argument("--ground_refine_pad", type=float, default=4.0,
                    help="ground_window stage 2: crop side = this x the stage-1 box side")
    ap.add_argument("--ground_refine_tok", type=int, default=640,
                    help="ground_window stage 2: token budget of the refinement crop")
    ap.add_argument("--dump_ground", default=None,
                    help="ground_window: directory for a ground.png contact sheet (--dump_n samples, "
                         "--dump_select first|miss|hit by the final box): full image with GT box "
                         "(green), stage-1 box (red), stage-2 box (orange), crop window (yellow); a "
                         "zoom around the GT; the crop the model answered from")
    ap.add_argument("--ground_topk", type=int, default=1,
                    help="ground_window: ask for up to k candidate boxes; one crop each, the "
                         "--crop_tok budget split between them")
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
                         "floor then moments inside the k x k window around the argmax; "
                         "'gate:<k>:<thr>' = win:k when the floored mass in that window >= thr "
                         "(unimodal map), identity grid otherwise; 'warp[:<lam>]' = the map's "
                         "QUANTILE grid (per-axis inverse-CDF of the marginals, no moments, no floor; "
                         "lam < 1 mixes in a uniform density for minimum coverage). "
                         "Default: model as trained")
    ap.add_argument("--glob_layers", default=None,
                    help="comma list of DAT layers that keep the global term at inference; the "
                         "others get a flat map (c=0, s=1). Default: all")
    ap.add_argument("--hd_pos", choices=["legacy", "sampled", "sampled_frac"], default="legacy",
                    help="mRoPE positions of the HD keys for the model's OWN grid: 'legacy' = fixed "
                         "linspace over the LR range (as trained), 'sampled' = from the actual sampling "
                         "locations rounded to LR units (what a teacher-forced grid gets), "
                         "'sampled_frac' = fractional. Inference-only (diag 14)")
    args = ap.parse_args()
    STATE["glob_readout"] = args.glob_readout
    STATE["hd_pos"] = args.hd_pos
    if args.hd_pos != "legacy":
        print(f"[probe] HD positions override: {args.hd_pos} (positions follow the sampled locations)")
    STATE["glob_layers"] = (None if args.glob_layers is None
                            else {int(x) for x in args.glob_layers.split(",") if x.strip()})
    if args.glob_readout or args.glob_layers:
        print(f"[probe] glob readout override: {args.glob_readout}  layers={args.glob_layers}  "
              f"(map tables / hit split still use the model's own map; c, s, grid follow the override)")

    from transformers import AutoProcessor
    from llava.model.language_model.modeling_qwen3_5_dat import (
        Qwen3_5DATForConditionalGeneration,
    )

    install_merge_hook()
    samples = load_samples(args)
    if args.hd_source == "shuffle":
        prepare_shuffle_sources(samples)
        print(f"[probe] shuffle: {len(samples)} content-checked donor pairs")
    has_gt = samples and samples[0]["gt"] is not None
    print(f"[probe] {len(samples)} samples  tok_budget={args.tok_budget}  "
          f"hd_source={args.hd_source}  biases={args.hd_bias}")

    model = Qwen3_5DATForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation=args.attn,
    ).eval()
    install_lr_key_hook(model)
    image_token_id = getattr(model.config, "image_token_id", None)
    args.grid_size = int((getattr(model.config, "dat_extra_args", None) or {}).get("grid_size", 20))
    samp_mode = args.hd_source == "sampled_tokens"
    args.samp_grids = [g for g in args.samp_grids.split(",") if g]
    args.samp_pos = [p for p in args.samp_pos.split(",") if p]
    if samp_mode:
        bad = [g for g in args.samp_grids if g not in SAMP_GRIDS] + \
              [p for p in args.samp_pos if p not in ("square", "lr")]
        if bad:
            raise SystemExit(f"sampled_tokens: unknown --samp_grids/--samp_pos {bad}")
        install_samp_hooks(model, image_token_id)
        k_exit = int((getattr(model.config, "dat_extra_args", None) or {}).get("hd_early_exit_k", 0) or 0)
        print(f"[probe] sampled_tokens: grids {args.samp_grids} x positions {args.samp_pos}; "
              f"pred grid from layer {args.samp_layer}; Ns = {args.grid_size ** 2}; "
              f"HD ViT early exit k = {k_exit or 'off (full ViT + merger)'}")
    if (args.hd_source.startswith("oracle") or args.hd_source in ("crop_tokens", "warp_oracle")
            or (samp_mode and "oracle" in args.samp_grids)) \
            and args.dataset not in ("vstar", "synth"):
        raise SystemExit("--hd_source oracle/oracle_rand/crop_tokens/warp_oracle needs "
                         "--dataset vstar or synth (GT bboxes); pred_window/pred_warp/plain_tokens "
                         "work on any dataset")
    args.map_layers = [int(x) for x in args.map_layers.split(",") if x.strip()]
    pred_mode = args.hd_source in PRED_SOURCES
    if pred_mode or samp_mode:
        args.hd_bias = [0.0]            # the real HD pass only serves to record the map / grid
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
    ground_mode = args.hd_source == "ground_window"
    samp_cfgs = [f"samp_{g}@{p}" for g in args.samp_grids for p in args.samp_pos] if samp_mode else []
    if pred_mode:
        configs = ["off", "real b=+0", args.hd_source]
    elif samp_mode:
        configs = ["off", "real b=+0"] + samp_cfgs
    elif ground_mode:
        configs = ["off", "ground_window"]
    else:
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
        if ground_mode:
            lr_thw_, hd_w_, hd_h_ = STATE["geom"]
            t = gen(ground_window_inputs(s, lr_thw_, hd_w_, hd_h_, model, processor, hr_processor,
                                         args, device, dtype))
            raw["ground_window"].append(t); preds["ground_window"].append(extract(t))
        else:
            for b, cname in zip(args.hd_bias, configs[1:]):
                STATE["bias"] = float(b)
                STATE["record"] = (b == record_bias)
                t = gen({**base_inputs, **hd_extra})
                STATE["record"] = False
                raw[cname].append(t); preds[cname].append(extract(t))
        set_force_locs(model, None)
        if pred_mode:
            # the real pass above recorded the model's own maps -> 2nd image
            lr_thw_, hd_w_, hd_h_ = STATE["geom"]
            inp2 = build_pred_inputs(s, i, lr_thw_, hd_w_, hd_h_, processor, hr_processor,
                                     args, device, dtype)
            if inp2 is None:
                STATE["pred_fallback"] = STATE.get("pred_fallback", 0) + 1
                t = raw["off"][-1]                               # no map -> LR answer
            else:
                t = gen(inp2)
            raw[args.hd_source].append(t); preds[args.hd_source].append(extract(t))
        if samp_mode:
            # the real pass recorded the model's own grid; the HD features are
            # recomputed once here (same ViT call / early exit as the model's)
            lr_thw_, hd_w_, hd_h_ = STATE["geom"]
            with torch.inference_mode():
                if args.samp_vit == "whole":
                    hd_feat = model._generate_hd_features(hd_extra["pixel_values_hd"],
                                                          hd_extra["image_grid_thw_hd"])[0]
                else:
                    hd_img = make_hd_image(s, i, samples, lr_thw_, hd_w_, hd_h_, "real")
                    hd_feat, n_tiles = tiled_hd_features(model, hd_img, hr_processor, device, dtype,
                                                         args.samp_tile_cells)
                    STATE.setdefault("samp_tiles", []).append(n_tiles)
            for g in args.samp_grids:
                locs_g = samp_locs(g, s, i, hd_w_, hd_h_, args)
                for p in args.samp_pos:
                    cname = f"samp_{g}@{p}"
                    if locs_g is None:
                        sf = STATE.setdefault("samp_fallback", defaultdict(int))
                        sf[cname] += 1
                        t = raw["off"][-1]
                    else:
                        inp2 = build_samp_inputs(s, lr_thw_, g, locs_g, hd_feat, p, processor,
                                                 hr_processor, args, device, dtype, image_token_id)
                        t = gen(inp2)
                        clear_samp()
                    raw[cname].append(t); preds[cname].append(extract(t))
            if "oracle" in args.samp_grids:
                win = crop_window(s, hd_w_, hd_h_, args.oracle_min_cells)
                if win is not None:
                    STATE.setdefault("samp_win_cells", []).append(
                        ((win[2] - win[0]) * hd_w_ / FACTOR, (win[3] - win[1]) * hd_h_ / FACTOR))
    STATE["bias"] = 0.0
    if samp_mode:
        wc = np.array(STATE.get("samp_win_cells") or [[float("nan")] * 2])
        tiles = STATE.get("samp_tiles")
        print(f"[probe] sampled_tokens: HD features from "
              + (f"{np.mean(tiles):.1f} independent ViT tiles per image (<= {args.samp_tile_cells} cells per side)"
                 if tiles else "the model's single whole-image ViT call")
              + f"; tokens injected in {STATE.get('samp_inject_n', 0)} ViT calls, "
              f"LR-frame positions applied in {STATE.get('samp_rope_n', 0)} rotary calls "
              f"(expect >= samples x #@lr configs); fallback to the LR answer: "
              f"{dict(STATE.get('samp_fallback', {})) or 'none'}; oracle window = "
              f"{np.nanmean(wc[:, 0]):.1f} x {np.nanmean(wc[:, 1]):.1f} HD cells on average "
              f"(grid {args.grid_size} x {args.grid_size}: > grid = the points sub-sample the window)")
    if STATE.get("gate_log"):
        print(f"[probe] gate readout: global term ON for {100 * float(np.mean(STATE['gate_log'])):.1f}% "
              f"of (sample, layer) calls")
    if STATE.get("hd_pos_n"):
        print(f"[probe] HD positions rebuilt from sampled locations in {STATE['hd_pos_n']} rotary calls "
              f"(expect ~ samples x DAT layers x slots)")
    if args.hd_source.startswith("oracle"):
        print(f"[probe] forced sampling window on {n_forced}/{len(samples)} samples "
              f"(min {args.oracle_min_cells} HD cells per side)")
    if STATE.get("crop_stats"):
        st = STATE["crop_stats"]
        area = np.nanmean(st["area"]) if st["area"] else float("nan")
        print(f"[probe] {args.hd_source}: 2nd image ({args.crop_pos} the question) on {st['n']}/{len(samples)} "
              f"samples; tokens mean {np.mean(st['tok']):.0f} (min {min(st['tok'])}, max {max(st['tok'])}, "
              f"budget {args.crop_tok}); window area {100 * area:.1f}% of image; LR grid identical to "
              f"the off config in {st['lr_same']}/{st['n']}"
              + (f"; map missing -> LR answer on {STATE['pred_fallback']}" if STATE.get("pred_fallback") else ""))
        if args.hd_source == "crop_tokens":
            cc = STATE.get("crop_corrupt", {"hit": st["n"], "n": st["n"]})
            print(f"[probe] crop_tokens window: pad x{args.crop_pad}, shift {args.crop_shift} x window, "
                  f"target hit-rate {args.crop_hit_rate} -> realised {cc['hit']}/{cc['n']} GT windows, "
                  f"the rest same-size windows away from the target")
    if STATE.get("ground_stats"):
        g = STATE["ground_stats"]
        iou = np.mean(g["iou"]) if g["iou"] else float("nan")
        print(f"[probe] ground_window: query={args.ground_query} stages={args.ground_stages} "
              f"topk={args.ground_topk}; bbox parsed on {g['parsed']}/{g['n']} (coords={args.ground_coords}, "
              f"pad x{args.ground_pad}); FINAL first-box centre in GT window {g['in_win']}/{g['n']}, "
              f"in GT box {g['in_box']}/{g['n']}; mean IoU with GT box {iou:.3f}; stage-1 (LR) box "
              f"in window {g['s1_in_win']}/{g['n']}; ANY of the {np.mean(g['n_boxes']) if g['n_boxes'] else 0:.2f} "
              f"boxes in window {g['any_in_win']}/{g['n']}; under the OTHER coordinate reading stage-1 "
              f"would be {g['alt_in_win']}/{g['n']}; unparsed -> whole image")
        for r in g["raw"]:
            print(f"[probe]   raw: {r!r}")
        for q_, p_ in g["extracted"]:
            print(f"[probe]   extracted: {q_!r} -> {p_!r}")

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

    # HD-key diagnostics also use correlation when the global offset is off.
    def _corr(a, b):
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        if len(a) < 3 or a.std() < 1e-9 or b.std() < 1e-9:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    if GLOBC:
        # Sample-dependence of the global term. r = Pearson correlation across
        # samples between the predicted centroid (scale) and the GT window's
        # centre (scale) per axis; |c-t| = mean centroid->window-centre distance
        # of the prediction vs. of the best CONSTANT centroid (sample mean, the
        # question-blind prior). std(c) = spread of the prediction across samples.
        # r ~ 0 and |c-t| ~ const -> the head learned a fixed prior, it does not
        # read the question/image; r >> 0 and |c-t| < const -> it localises.
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

    if HDC:
        # Diagnostic 12: a second, HD-resolution localisation stage read off
        # the sampled HD keys (see patched_xattn). Per layer:
        #   |c_hd-t|  centroid of the prompt-end token's attention over the Ns
        #             HD keys, at their sampling positions; F = after the
        #             uniform floor; c_lr = the model's own c at this layer
        #   pk_box    fraction of samples whose most-attended HD point is inside
        #             the GT box;  mass = attention mass on in-box points vs
        #             pts = fraction of sampling points in the box (flat attention)
        # 'covered' = only samples whose coarse grid put >= 1 point in the box
        # (a refinement stage can only find what the window contains).
        # Read: |c_hdF-t| << |c_lr-t| on covered samples and mass >> pts ->
        # the HD keys know where the answer is, a coarse-to-fine c is worth
        # building; mass ~ pts -> the answer token does not single out the
        # right HD points either, the bottleneck is not localisation precision.
        # in_box columns: fraction of a 20x20 grid's points inside the GT box
        # when the grid is centred at c_hdF with scale s_lr/2 (half the coarse
        # window) or s_hd (the HD attention's own spread), vs the model's grid
        # (LOC) -- the quantity that tracked the readout gain so far (model
        # 0.035-0.045 -> +2, oracle 0.10 -> +3.6).
        _gs = args.grid_size
        _m = 1.0 / max(_gs - 1, 1)
        _lin = np.linspace(-1 + _m, 1 - _m, _gs)
        _ref = np.stack(np.meshgrid(_lin, _lin, indexing="xy"), -1).reshape(-1, 2)   # [Ns, 2] (x, y)

        def _grid_in_box(cx, cy, sc, bx):
            p = np.array([cx, cy]) + sc * _ref
            p = np.clip(p, -1, 1)
            return float(((p[:, 0] >= bx[0]) & (p[:, 0] <= bx[2]) &
                          (p[:, 1] >= bx[1]) & (p[:, 1] <= bx[3])).mean())

        print(f"\n==== HD-key refinement (diag 12): prompt-end attention over the sampled HD points ====")
        print(f"{'layer':>6} {'set':>8} | {'n':>4} | {'|c_hd-t|':>8} {'|c_hdF-t|':>9} {'|c_lr-t|':>8} {'const':>6} | "
              f"{'r_cx':>6} {'r_cy':>6} | {'pk_box':>6} | {'mass':>6} {'pts':>6} | "
              f"{'ib_lr':>6} {'ib_s/2':>6} {'ib_shd':>6} {'s_hd':>5}")
        print("-" * 140)
        for lid in sorted(HDC):
            rows = HDC[lid]
            clr = GLOBC.get(lid)
            clr = clr if clr is not None and len(clr) == len(rows) else None
            loc_l = LOC.get(lid)
            loc_l = loc_l if loc_l is not None and len(loc_l) == len(rows) else None
            for set_name, keep in (("all", lambda r: True), ("covered", lambda r: r[6] > 0)):
                sel = [i for i, r in enumerate(rows) if keep(r)]
                if len(sel) < 5:
                    continue
                R = [rows[i] for i in sel]
                n_ = len(R)
                d_hd = sum(math.hypot(r[0] - r[7], r[1] - r[8]) for r in R) / n_
                d_hdf = sum(math.hypot(r[2] - r[7], r[3] - r[8]) for r in R) / n_
                mtx, mty = sum(r[7] for r in R) / n_, sum(r[8] for r in R) / n_
                d_const = sum(math.hypot(mtx - r[7], mty - r[8]) for r in R) / n_
                d_lr = (sum(math.hypot(clr[i][0] - clr[i][4], clr[i][1] - clr[i][5]) for i in sel) / n_
                        if clr is not None else float("nan"))
                r_cx = _corr([r[2] for r in R], [r[7] for r in R])
                r_cy = _corr([r[3] for r in R], [r[8] for r in R])
                pk = sum(r[4] for r in R) / n_
                mass = sum(r[5] for r in R) / n_
                pts = sum(r[6] for r in R) / n_
                ib_lr = (sum(loc_l[i][2] for i in sel) / n_) if loc_l is not None else float("nan")
                ib_half = sum(_grid_in_box(r[2], r[3], max(r[10] * 0.5, 0.1), r[9]) for r in R) / n_
                ib_shd = sum(_grid_in_box(r[2], r[3], max(r[11], 0.1), r[9]) for r in R) / n_
                s_hd = sum(r[11] for r in R) / n_
                layer_glob.setdefault(lid, {})[f"hdc_{set_name}"] = {
                    "n": n_, "d_hd": d_hd, "d_hdf": d_hdf, "d_lr": d_lr, "d_const": d_const,
                    "r_cx": r_cx, "r_cy": r_cy, "peak_in_box": pk, "mass_in_box": mass, "pts_in_box": pts,
                    "in_box_lr": ib_lr, "in_box_half": ib_half, "in_box_shd": ib_shd, "s_hd": s_hd}
                print(f"{lid:>6} {set_name:>8} | {n_:>4} | {d_hd:>8.3f} {d_hdf:>9.3f} {d_lr:>8.3f} {d_const:>6.3f} | "
                      f"{r_cx:>6.3f} {r_cy:>6.3f} | {pk:>6.3f} | {mass:>6.3f} {pts:>6.3f} | "
                      f"{ib_lr:>6.3f} {ib_half:>6.3f} {ib_shd:>6.3f} {s_hd:>5.2f}")

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
                on_cs = configs[1:]
                on_c = on_cs[0] if on_cs else None
                print(f"\n==== by localisation outcome at layer {ref_l}  (peak in GT window = hit)  "
                      f"and by GT box height at LR ====")
                if len(on_cs) > 1:
                    print("on/delta columns, in order: " + " | ".join(on_cs))
                on_hdr = " ".join(f"{'on':>6} {'delta':>6}" for _ in on_cs)
                print(f"{'subset':>12} | {'n':>4} | {'off':>6} {on_hdr} | "
                      f"{'s':>6} {'in_box':>6} {'ratio':>6}")
                print("-" * (56 + 14 * len(on_cs)))
                subset_out = {}
                for name, m in subsets:
                    idx = [k for k in range(n) if m[k]]
                    if not idx:
                        continue
                    a_off = acc(preds["off"], m)
                    a_on = acc(preds[on_c], m) if on_c else float("nan")
                    on_cells = " ".join(f"{acc(preds[c], m):>6.2f} {acc(preds[c], m) - a_off:>+6.2f}"
                                        for c in on_cs)
                    s_mean = float(np.mean([(GLOBC[ref_l][k][2] + GLOBC[ref_l][k][3]) / 2 for k in idx])) \
                        if ref_l in GLOBC and len(GLOBC[ref_l]) == n else float("nan")
                    if ref_l in LOC and len(LOC[ref_l]) == n:
                        ib = float(np.mean([LOC[ref_l][k][2] for k in idx]))
                        ratio = float(np.mean([LOC[ref_l][k][0] / max(LOC[ref_l][k][1], 1e-9) for k in idx]))
                    else:
                        ib = ratio = float("nan")
                    subset_out[name] = {"n": len(idx), "off": a_off, "on": a_on, "s": s_mean,
                                        "in_box": ib, "ratio": ratio}
                    print(f"{name:>12} | {len(idx):>4} | {a_off:>6.2f} {on_cells} | "
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

    if args.dump_ground and STATE.get("ground_log"):
        # Failure modes of the model's own grounding: per sample, [full image
        # with boxes] [zoom around the GT, same boxes] [the crop answered from].
        # Green = GT box, red = stage-1 (LR) box, orange = stage-2 box,
        # yellow = crop window, white = stage-2 search crop.
        from PIL import ImageDraw
        os.makedirs(args.dump_ground, exist_ok=True)
        glog = STATE["ground_log"]
        PW = 320

        def _draw_boxes(img, region, lg, gt):
            """img cropped to `region` (fractions) and scaled to PW wide, with
            every box (image fractions) drawn in region coordinates."""
            W, H = img.size
            rx0, ry0, rx1, ry1 = region
            px = (int(rx0 * W), int(ry0 * H), max(int(math.ceil(rx1 * W)), int(rx0 * W) + 1),
                  max(int(math.ceil(ry1 * H)), int(ry0 * H) + 1))
            crop = img.crop(px).convert("RGB")
            sc = PW / crop.width
            out = crop.resize((PW, max(1, int(crop.height * sc))))
            d = ImageDraw.Draw(out)

            def rect(b, color, w=2):
                if b is None:
                    return
                x0 = (b[0] * W - px[0]) * sc; y0 = (b[1] * H - px[1]) * sc
                x1 = (b[2] * W - px[0]) * sc; y1 = (b[3] * H - px[1]) * sc
                if x1 - x0 < 3:            # a tiny box: make it visible as a small square
                    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                    x0, x1, y0, y1 = cx - 3, cx + 3, cy - 3, cy + 3
                d.rectangle([x0, y0, x1, y1], outline=color, width=w)
            rect(lg.get("refine_win"), (255, 255, 255), 1)
            for wn in lg.get("wins", []):
                rect(wn, (255, 230, 0))
            for b in lg.get("s1", []):
                rect(b, (255, 40, 40))
            if lg.get("refine_win") is not None and lg.get("final"):
                rect(lg["final"][0], (255, 140, 0))
            rect(gt, (0, 255, 0))
            return out

        if args.dump_select == "first":
            sel = list(range(min(args.dump_n, len(glog))))
        else:
            want_hit = args.dump_select == "hit"
            sel = [i for i, lg in enumerate(glog) if lg["hit"] is not None and lg["hit"] == want_hit][:args.dump_n]
            print(f"[probe] dump_ground: {args.dump_select}: "
                  f"{sum(1 for lg in glog if lg['hit'] is not None and lg['hit'] == want_hit)}/{len(glog)} samples")
        panels = []
        on_c = "ground_window"
        for i in sel:
            s, lg = samples[i], glog[i]
            gt = gt_box_frac(s)
            zoom = pad_window(gt, 8.0, 8, *STATE["geom"][1:]) if gt else (0.0, 0.0, 1.0, 1.0)
            row = [_draw_boxes(s["image"], (0.0, 0.0, 1.0, 1.0), lg, gt),
                   _draw_boxes(s["image"], zoom, lg, gt)]
            if lg["crops"]:
                c = lg["crops"][0].convert("RGB")
                row.append(c.resize((PW, max(1, int(c.height * PW / c.width)))))
            rh = max(p.height for p in row) + 28
            strip = Image.new("RGB", (PW * 3 + 8, rh), (20, 20, 20))
            for j, p in enumerate(row):
                strip.paste(p, (j * (PW + 4), 28))
            ok_off = hit(preds["off"][i], gts[i]) if has_gt else None
            ok_on = hit(preds[on_c][i], gts[i]) if has_gt else None
            W_, H_ = s["image"].size
            lr_sc = math.sqrt(args.tok_budget * TOK_PX / max(W_ * H_, 1))
            bx = s["bboxes"][0] if s.get("bboxes") else (0, 0, 0, 0)
            l1 = (f"[{i}] {'HIT' if lg['hit'] else 'MISS'}  IoU={lg['iou'] if lg['iou'] is not None else float('nan'):.3f}  "
                  f"off={'ok' if ok_off else 'x'} crop={'ok' if ok_on else 'x'}  box@LR {bx[2] * lr_sc:.0f}x{bx[3] * lr_sc:.0f}px  "
                  f"img {W_}x{H_}  {s['prompt'][:70]}")
            l2 = (f"targets={s.get('targets')}  raw1={lg['raw']!r}" + (f"  raw2={lg['raw2']!r}" if lg["raw2"] else "")
                  + (f"  phrase={lg['phrase']!r}" if lg.get("phrase") else ""))
            d = ImageDraw.Draw(strip)
            d.text((4, 1), l1[:150], fill=(220, 220, 220))
            d.text((4, 14), l2[:150], fill=(170, 170, 170))
            panels.append(strip)
        if panels:
            sheet = Image.new("RGB", (panels[0].width, sum(p.height for p in panels)), (20, 20, 20))
            yy = 0
            for p in panels:
                sheet.paste(p, (0, yy)); yy += p.height
            fn = os.path.join(args.dump_ground, f"ground_{args.dump_select}.png")
            sheet.save(fn)
            print(f"[probe] dump_ground: wrote {fn}  ({len(panels)} samples; green GT, red stage-1, "
                  f"orange stage-2, yellow crop window, white stage-2 search crop)")

    if args.out:
        report = {
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
                {"sample_index": k, "gt": gts[k], "category": samples[k]["category"],
                 **{key: samples[k][key] for key in
                    ("image_sha256", "shuffle_source_index", "shuffle_source_sha256")
                    if key in samples[k]},
                 **{c: raw[c][k] for c in configs}}
                for k in range(n)
            ],
        }
        with open(args.out, "w") as output:
            json.dump(report, output, indent=2, ensure_ascii=False)
        print(f"[probe] saved -> {args.out}")


if __name__ == "__main__":
    main()
