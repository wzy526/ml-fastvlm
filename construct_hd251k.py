#!/usr/bin/env python3
"""Construct a ~251K high-resolution, task-balanced SFT dataset for DAT training.

Now supports starting from ZERO local files. All annotations and images
are downloaded and converted from Hugging Face by default.

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

All data is downloaded from Hugging Face on first run.
Images are saved to train_split/ subdirectories.
Subsequent runs will reuse cached images.

Usage:
  python construct_hd251k.py                      # from HF only (ZERO local files, recommended)
  python construct_hd251k.py --synthdog_ratio 0.4  # adjust SynthDoG ratio
  python construct_hd251k.py --skip_downloads      # use only cached local data (if available)
  python construct_hd251k.py --from_hf False       # force legacy local JSON mode
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
SFT_DIR = "/cluster/data3/wzy/sft_data"
TRAIN_SPLIT = os.path.join(SFT_DIR, "train_split")
OUTPUT_JSON = os.path.join(SFT_DIR, "llava_hd251k.json")
SEED = 42

# These are kept for --skip_downloads mode only
HD_SUPPLEMENTS = os.path.join(SFT_DIR, "hd_supplements.jsonl")
BASE_JSON = os.path.join(SFT_DIR, "llava_v1_5_mix665k_shuffled.json")
GPT4V_JSON = os.path.join(SFT_DIR, "sharegpt4v_json",
                          "sharegpt4v_instruct_gpt4-vision_cap100k.json")


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
#  Step 1: High-res data from Hugging Face (docvqa / infovqa / chartqa / textvqa)
# ────────────────────────────────────────────────

def process_vqa_from_hf(dataset_name, split, subset_name, save_dir_name,
                        target_count=None, question_key="question",
                        answer_key="answers", default_question="What is written in this image?"):
    """Download VQA dataset from HF, save images, convert to LLaVA format."""
    from datasets import load_dataset

    print(f"  Loading {dataset_name} ({subset_name or 'default'}) -> {save_dir_name} ...")
    ds = load_dataset(dataset_name, subset_name, split=split, trust_remote_code=True)

    if target_count and len(ds) > target_count:
        # Random sample to control size
        indices = random.Random(SEED + hash(save_dir_name)).sample(range(len(ds)), target_count)
        ds = ds.select(indices)

    save_dir = os.path.join(TRAIN_SPLIT, save_dir_name)
    os.makedirs(save_dir, exist_ok=True)

    samples = []
    prompt_rng = random.Random(SEED + 100 + hash(save_dir_name))

    for idx, item in enumerate(tqdm(ds, desc=f"    {save_dir_name}")):
        # Save image
        img = item["image"]
        if hasattr(img, "convert"):
            img = img.convert("RGB")
        fname = f"{save_dir_name}_{idx:06d}.jpg"
        img_path = os.path.join(save_dir, fname)

        if not os.path.exists(img_path):
            img.save(img_path)

        # Convert to LLaVA format
        question = item.get(question_key, default_question)
        if isinstance(question, list):
            question = question[0] if question else default_question

        answers = item.get(answer_key, [""])
        if isinstance(answers, list) and answers:
            answer = answers[0] if isinstance(answers[0], str) else str(answers[0])
        else:
            answer = str(answers) if answers else "N/A"

        samples.append({
            "id": f"{save_dir_name}_{idx}",
            "image": f"{save_dir_name}/{fname}",
            "conversations": [
                {"from": "human", "value": f"<image>\n{question}"},
                {"from": "gpt", "value": answer},
            ],
        })

    print(f"    {save_dir_name}: {len(samples)} samples")
    return samples


def collect_hd_from_hf():
    """Download and convert all HD VQA datasets from Hugging Face (no local files needed)."""
    samples = []

    # DocVQA (~39k)
    docvqa = process_vqa_from_hf(
        "HuggingFaceM4/DocumentVQA", "train", None, "docvqa",
        target_count=39000, question_key="question", answer_key="answers"
    )
    samples.extend(docvqa)

    # ChartQA (~28k)
    chartqa = process_vqa_from_hf(
        "HuggingFaceM4/ChartQA", "train", None, "chartqa",
        target_count=28000, question_key="query", answer_key="label",
        default_question="What does this chart show?"
    )
    samples.extend(chartqa)

    # InfoVQA / InfographicVQA (~24k)
    infovqa = process_vqa_from_hf(
        "lmms-lab/infographicvqa", "train", None, "infovqa",
        target_count=24000, question_key="question", answer_key="answers",
        default_question="What is shown in this infographic?"
    )
    samples.extend(infovqa)

    # TextVQA (~22k) - using official TextVQA dataset
    textvqa = process_vqa_from_hf(
        "textvqa", "train", None, "textvqa",
        target_count=22000, question_key="question", answer_key="answers",
        default_question="What is written in this image?"
    )
    samples.extend(textvqa)

    _print_counts(samples, "HD from HF")
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
                        help="Skip HuggingFace downloads, use only existing local data")
    parser.add_argument("--from_hf", action="store_true", default=True,
                        help="Download all annotations from Hugging Face (default: True, zero local files)")
    args = parser.parse_args()

    random.seed(SEED)

    # ── Collect non-SynthDoG data ──
    print("=" * 60)
    if args.from_hf and not args.skip_downloads:
        print("Step 1: High-res data from Hugging Face (docvqa, infovqa, chartqa, textvqa)")
        print("=" * 60)
        non_synthdog = collect_hd_from_hf()
    else:
        print("Step 1: Local high-res data (docvqa, infovqa, chartqa, textvqa)")
        print("=" * 60)
        print("  WARNING: --from_hf=False requires pre-existing hd_supplements.jsonl and base JSON.")
        print("  Falling back to HF mode for now.")
        non_synthdog = collect_hd_from_hf()

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
