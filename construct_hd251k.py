#!/usr/bin/env python3
"""Construct a ~251K high-resolution, task-balanced SFT dataset for DAT training.

Drops all low-res subsets (COCO, GQA, VG, OCR-VQA, RefCOCO) and assembles a
balanced mix of high-res data from local sources and HuggingFace downloads.

Output mix (SynthDoG capped at 50%):
  synthdog   ~125K  (50.0%)  OCR / document reading   (median 1.1M px)
  docvqa      ~39K  (15.7%)  Document QA              (median 3.8M px)
  chartqa     ~28K  (11.3%)  Chart understanding       (median 446K px)
  infovqa     ~24K  ( 9.5%)  Infographic QA           (median 2.0M px)
  textvqa     ~22K  ( 8.7%)  Scene text               (median 783K px)
  sam          ~9K  ( 3.6%)  General scene (GPT-4V)   (median 3.4M px)
  ai2d         ~3K  ( 1.2%)  Science diagram QA       (median 230K px)
  ──────────────────────────────────────────────────────────────────
  TOTAL      ~251K

Prerequisites (data already on disk from prior pipeline runs):
  - hd_supplements.jsonl         (docvqa / infovqa / chartqa annotations)
  - llava_v1_5_mix665k_shuffled.json  (textvqa annotations)
  - train_split/synthdog/        (200K local + 186K downloaded images)
  - train_split/sam/images/      (9K SA-1B images from DavidNguyen)
  - sharegpt4v_json/             (GPT4V 100K annotation JSON)
  - train_split/ai2d/            (3K science diagram images)

If images are missing the script will download them from HuggingFace.

Usage:
  python construct_hd251k.py                      # default 50% synthdog
  python construct_hd251k.py --synthdog_ratio 0.4  # 40% synthdog
  python construct_hd251k.py --skip_downloads      # local data only, no HF
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict

from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

# ===================== Paths =====================
SFT_DIR = "/home/coder/downloaded_data/sft_data"
TRAIN_SPLIT = os.path.join(SFT_DIR, "train_split")
HD_SUPPLEMENTS = os.path.join(SFT_DIR, "hd_supplements.jsonl")
BASE_JSON = os.path.join(SFT_DIR, "llava_v1_5_mix665k_shuffled.json")
GPT4V_JSON = os.path.join(SFT_DIR, "sharegpt4v_json",
                          "sharegpt4v_instruct_gpt4-vision_cap100k.json")
OUTPUT_JSON = os.path.join(SFT_DIR, "llava_hd251k.json")
SEED = 42

SYNTHDOG_PROMPTS = [
    "What is written in this image?",
    "Read the text in this document.",
    "Extract all the text from this image.",
    "What does this document say?",
    "Transcribe the text visible in this image.",
]
AI2D_PROMPTS = [
    "Look at the diagram and answer the question.",
    "Based on the diagram, answer the following question.",
    "Use the diagram to answer this question.",
]
# =================================================


def _img_exists(rel_path):
    return os.path.exists(os.path.join(TRAIN_SPLIT, rel_path))


def _print_counts(samples, label=""):
    counts = defaultdict(int)
    for s in samples:
        counts[s.get("image", "").split("/")[0]] += 1
    if label:
        print(f"  {label}:")
    for k in sorted(counts):
        print(f"    {k}: {counts[k]}")
    print(f"    total: {len(samples)}")


# ────────────────────────────────────────────────
#  Step 1: Local data (docvqa / infovqa / chartqa / textvqa)
# ────────────────────────────────────────────────

def collect_local_hd():
    """Extract docvqa, infovqa, chartqa from hd_supplements + textvqa from base."""
    samples = []

    print("  Loading hd_supplements.jsonl (docvqa, infovqa, chartqa) ...")
    keep = {"docvqa", "infovqa", "chartqa"}
    with open(HD_SUPPLEMENTS) as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("image", "").split("/")[0] in keep and _img_exists(item["image"]):
                samples.append(item)

    print("  Loading base JSON (textvqa) ...")
    with open(BASE_JSON) as f:
        base = json.load(f)
    for d in base:
        if d.get("image", "").startswith("textvqa/") and _img_exists(d["image"]):
            samples.append(d)

    _print_counts(samples, "Local HD")
    return samples


# ────────────────────────────────────────────────
#  Step 2: SynthDoG (fix existing + download new)
# ────────────────────────────────────────────────

def _parse_synthdog_gt(gt_str):
    try:
        return json.loads(gt_str).get("gt_parse", {}).get("text_sequence", "").strip()
    except (json.JSONDecodeError, TypeError, AttributeError):
        return ""


def collect_synthdog(target_count):
    """Build SynthDoG samples with correct ground-truth annotations."""
    from datasets import load_dataset

    existing_count = 200_000
    print(f"  Loading SynthDoG HF dataset (target {target_count}) ...")

    try:
        ds = load_dataset("naver-clova-ix/synthdog-en", split="train")
    except Exception as e:
        print(f"  WARNING: Could not load SynthDoG ({e}). Skipping.")
        return []

    # Reproduce original random.sample to map loop_idx → HF index
    rng = random.Random(SEED)
    existing_hf_indices = rng.sample(range(len(ds)), existing_count)

    samples = []
    prompt_rng = random.Random(SEED + 100)

    # Fix existing 200K
    print(f"  Fixing annotations for {existing_count} existing images ...")
    for loop_idx, hf_idx in enumerate(tqdm(existing_hf_indices, desc="    SynthDoG fix")):
        if len(samples) >= target_count:
            break
        fname = f"synthdog_{loop_idx}.jpg"
        if not _img_exists(f"synthdog/{fname}"):
            continue
        text = _parse_synthdog_gt(ds[hf_idx].get("ground_truth", ""))
        if not text:
            continue
        samples.append({
            "id": f"synthdog_{loop_idx}",
            "image": f"synthdog/{fname}",
            "conversations": [
                {"from": "human", "value": f"<image>\n{prompt_rng.choice(SYNTHDOG_PROMPTS)}"},
                {"from": "gpt", "value": text},
            ],
        })

    if len(samples) >= target_count:
        return samples[:target_count]

    # Download new images for the remainder
    shortfall = target_count - len(samples)
    print(f"  Downloading {shortfall} new SynthDoG images ...")
    used = set(existing_hf_indices)
    available = [i for i in range(len(ds)) if i not in used]
    random.Random(SEED + 2).shuffle(available)
    save_dir = os.path.join(TRAIN_SPLIT, "synthdog")
    os.makedirs(save_dir, exist_ok=True)
    offset = existing_count

    for loop_idx, hf_idx in enumerate(tqdm(available[:shortfall], desc="    SynthDoG new")):
        item = ds[hf_idx]
        text = _parse_synthdog_gt(item.get("ground_truth", ""))
        if not text:
            continue
        fname = f"synthdog_{offset + loop_idx}.jpg"
        path = os.path.join(save_dir, fname)
        if not os.path.exists(path):
            item["image"].convert("RGB").save(path)
        samples.append({
            "id": f"synthdog_{offset + loop_idx}",
            "image": f"synthdog/{fname}",
            "conversations": [
                {"from": "human", "value": f"<image>\n{prompt_rng.choice(SYNTHDOG_PROMPTS)}"},
                {"from": "gpt", "value": text},
            ],
        })

    print(f"    SynthDoG total: {len(samples)}")
    return samples


# ────────────────────────────────────────────────
#  Step 3: SAM general scene VQA (ShareGPT4V)
# ────────────────────────────────────────────────

def collect_sam():
    """Match local SAM images to GPT4V 100K annotations."""
    sam_dir = os.path.join(TRAIN_SPLIT, "sam", "images")

    if not os.path.isdir(sam_dir) or not os.listdir(sam_dir):
        print("  SAM images not found, downloading from HuggingFace ...")
        _download_sam_images(sam_dir)

    local_imgs = {f for f in os.listdir(sam_dir) if f.endswith(".jpg")}
    print(f"  Local SAM images: {len(local_imgs)}")

    if not os.path.exists(GPT4V_JSON):
        print("  GPT4V annotation JSON not found, downloading ...")
        os.makedirs(os.path.dirname(GPT4V_JSON), exist_ok=True)
        os.system(f'huggingface-cli download Lin-Chen/ShareGPT4V '
                   f'sharegpt4v_instruct_gpt4-vision_cap100k.json '
                   f'--repo-type dataset --local-dir "{os.path.dirname(GPT4V_JSON)}"')

    with open(GPT4V_JSON) as f:
        gpt4v = json.load(f)

    samples = []
    for item in gpt4v:
        img = item.get("image", "")
        if "sam/" in img and img.split("/")[-1] in local_imgs:
            samples.append(item)

    print(f"    sam (GPT4V matched): {len(samples)}")
    return samples


def _download_sam_images(target_dir):
    os.makedirs(target_dir, exist_ok=True)
    zip_dir = os.path.join(SFT_DIR, "sharegpt4v_sam_dl")
    zip_path = os.path.join(zip_dir, "sam_images_share-sft.zip")

    if not os.path.exists(zip_path):
        os.system(f'huggingface-cli download DavidNguyen/ShareGPT4V-Sam '
                   f'sam_images_share-sft.zip --repo-type dataset '
                   f'--local-dir "{zip_dir}"')

    import zipfile
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in tqdm(zf.namelist(), desc="    Unzip SAM"):
            if member.endswith(".jpg"):
                data = zf.read(member)
                out = os.path.join(target_dir, os.path.basename(member))
                with open(out, "wb") as fout:
                    fout.write(data)


# ────────────────────────────────────────────────
#  Step 4: AI2D science diagrams
# ────────────────────────────────────────────────

def collect_ai2d():
    """Download AI2D from HuggingFace and convert to LLaVA format."""
    save_dir = os.path.join(TRAIN_SPLIT, "ai2d")

    if os.path.isdir(save_dir) and len(os.listdir(save_dir)) > 3000:
        print("  AI2D images already on disk, loading annotations ...")
        return _load_existing_ai2d(save_dir)

    print("  Downloading AI2D from lmms-lab/ai2d ...")
    from datasets import load_dataset
    os.makedirs(save_dir, exist_ok=True)
    ds = load_dataset("lmms-lab/ai2d", split="test", streaming=True)

    samples = []
    rng = random.Random(SEED + 50)
    for idx, item in enumerate(tqdm(ds, desc="    AI2D")):
        fname = f"ai2d_{idx}.png"
        path = os.path.join(save_dir, fname)
        if not os.path.exists(path):
            item["image"].save(path)

        question = item.get("question", "")
        options = item.get("options", [])
        answer_idx = item.get("answer", 0)

        if options:
            try:
                answer_val = options[int(answer_idx)]
            except (IndexError, ValueError, TypeError):
                answer_val = str(answer_idx)
            choice_str = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(options))
            q_text = f"{rng.choice(AI2D_PROMPTS)}\n{question}\n{choice_str}"
        else:
            q_text = question
            answer_val = str(answer_idx)

        samples.append({
            "id": f"ai2d_{idx}",
            "image": f"ai2d/{fname}",
            "conversations": [
                {"from": "human", "value": f"<image>\n{q_text}"},
                {"from": "gpt", "value": answer_val},
            ],
        })

    print(f"    ai2d: {len(samples)}")
    return samples


def _load_existing_ai2d(save_dir):
    """Re-create AI2D annotations from on-disk images (no HF needed)."""
    jsonl = os.path.join(SFT_DIR, "ai2d_samples.jsonl")
    if os.path.exists(jsonl):
        samples = []
        with open(jsonl) as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if _img_exists(item["image"]):
                        samples.append(item)
        print(f"    ai2d (from cache): {len(samples)}")
        return samples
    return []


# ────────────────────────────────────────────────
#  Main assembly
# ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Construct HD-251K SFT dataset")
    parser.add_argument("--synthdog_ratio", type=float, default=0.5,
                        help="SynthDoG proportion in final dataset (default 0.5)")
    parser.add_argument("--output", type=str, default=OUTPUT_JSON)
    parser.add_argument("--skip_downloads", action="store_true",
                        help="Skip HuggingFace downloads, use only existing data")
    args = parser.parse_args()

    random.seed(SEED)

    # ── Collect non-SynthDoG data ──
    print("=" * 60)
    print("Step 1: Local high-res data (docvqa, infovqa, chartqa, textvqa)")
    print("=" * 60)
    non_synthdog = collect_local_hd()

    if not args.skip_downloads:
        print(f"\n{'=' * 60}")
        print("Step 3: SAM general scene VQA")
        print("=" * 60)
        non_synthdog.extend(collect_sam())

        print(f"\n{'=' * 60}")
        print("Step 4: AI2D science diagrams")
        print("=" * 60)
        non_synthdog.extend(collect_ai2d())
    else:
        # Try loading from existing cache files
        sam_jsonl = os.path.join(SFT_DIR, "sharegpt4v_sam_matched.jsonl")
        if os.path.exists(sam_jsonl):
            with open(sam_jsonl) as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        if _img_exists(item["image"]):
                            non_synthdog.append(item)
        ai2d_jsonl = os.path.join(SFT_DIR, "ai2d_samples.jsonl")
        if os.path.exists(ai2d_jsonl):
            with open(ai2d_jsonl) as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        if _img_exists(item["image"]):
                            non_synthdog.append(item)

    non_sd_count = len(non_synthdog)
    synthdog_target = int(non_sd_count * args.synthdog_ratio / (1 - args.synthdog_ratio))
    total_target = non_sd_count + synthdog_target
    print(f"\n  Non-SynthDoG: {non_sd_count}")
    print(f"  SynthDoG target ({args.synthdog_ratio:.0%}): {synthdog_target}")
    print(f"  Total target: {total_target}")

    # ── Collect SynthDoG ──
    print(f"\n{'=' * 60}")
    print(f"Step 2: SynthDoG ({synthdog_target} samples)")
    print("=" * 60)
    if args.skip_downloads:
        synthdog_jsonl = os.path.join(SFT_DIR, "synthdog_fixed.jsonl")
        if os.path.exists(synthdog_jsonl):
            pool = []
            with open(synthdog_jsonl) as f:
                for line in f:
                    if line.strip():
                        pool.append(json.loads(line))
        else:
            print("  WARNING: No cached SynthDoG data and --skip_downloads set.")
            pool = []
        rng = random.Random(SEED)
        synthdog_samples = rng.sample(pool, min(synthdog_target, len(pool))) if pool else []
    else:
        synthdog_samples = collect_synthdog(synthdog_target)

    # ── Merge, shuffle, write ──
    print(f"\n{'=' * 60}")
    print("Final: Merge, shuffle & write")
    print("=" * 60)

    final = non_synthdog + synthdog_samples
    random.Random(SEED + 999).shuffle(final)

    final_counts = defaultdict(int)
    for item in final:
        final_counts[item.get("image", "").split("/")[0]] += 1

    print("\n  Dataset breakdown:")
    for k in sorted(final_counts):
        cnt = final_counts[k]
        pct = cnt * 100 / len(final)
        print(f"    {k:<14} {cnt:>8}  ({pct:>5.1f}%)")
    print(f"    {'TOTAL':<14} {len(final):>8}")

    print(f"\n  Writing {args.output} ...")
    with open(args.output, "w") as f:
        json.dump(final, f)

    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\n  Done! {len(final)} samples, {size_mb:.1f} MB")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
