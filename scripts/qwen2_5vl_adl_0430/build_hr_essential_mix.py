#!/usr/bin/env python3
"""Build an HR-essential training mix from existing data.

Problem: Current hd251k has only 26% HR-essential samples (docvqa+infovqa).
74% of data gives DAT zero useful HR gradient (images too small or task
doesn't need HR). This wastes training compute and dilutes the HR signal.

Solution: Reconstruct the mix from llava_hd_merged_1m.json, keeping only
sources that genuinely need HR, and adding high-value sources (ocr_vqa, vg
grounding, coco grounding) that were available but unused.

Target mix (~350-400K, all HR-essential):
  - docvqa:     39,000  (all, median 3.8M px)
  - infovqa:    23,946  (all, median 1.8M px)
  - ocr_vqa:    80,000  (all, natural scene OCR)
  - vg:         47,130  (all, bbox grounding = small target)
  - coco_ground: 50,000 (subsample from 348K grounding, RefCOCO-style)
  - gqa:        39,343  (all, compositional reasoning about regions)
  - synthdog:   40,000  (subsample, only keep for OCR diversity)
  Total: ~319K

Usage:
    python scripts/qwen2_5vl_adl_0430/build_hr_essential_mix.py \
        --source_json /root/autodl-tmp/models_data/sft_data/llava_hd_merged_1m.json \
        --hd251k_json /root/autodl-tmp/models_data/sft_data/llava_hd251k.json \
        --output_json /root/autodl-tmp/models_data/sft_data/llava_hr_essential_350k.json \
        --image_folder /root/autodl-tmp/models_data/sft_data/train_split
"""

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path


def classify_sample(item: dict) -> str:
    """Classify a sample by its source dataset."""
    img = item.get("image", "")
    source = img.split("/")[0] if "/" in img else "unknown"
    return source


def is_grounding(item: dict) -> bool:
    """Check if a sample is a grounding/bbox task."""
    convs = item.get("conversations", [])
    if convs:
        q = convs[0].get("value", "").lower()
        return "bounding box" in q or "region" in q or "coordinate" in q
    return False


def build_mix(source_data: list, hd251k_data: list, image_folder: str,
              max_coco_ground: int = 50000,
              max_synthdog: int = 40000,
              seed: int = 42) -> list:
    """Build HR-essential mix."""
    random.seed(seed)

    by_source = defaultdict(list)
    for item in source_data:
        src = classify_sample(item)
        by_source[src].append(item)

    # hd251k has correct filenames for docvqa/infovqa/synthdog (zero-padded)
    # while merged_1m has non-padded names that don't match disk
    hd251k_by_source = defaultdict(list)
    for item in hd251k_data:
        src = classify_sample(item)
        hd251k_by_source[src].append(item)

    output = []

    # 1. docvqa - all from hd251k (correct filenames, median 3.8M px)
    docvqa = hd251k_by_source.get("docvqa", [])
    output.extend(docvqa)
    print(f"  docvqa:        {len(docvqa):>6d} (all, from hd251k)")

    # 2. infovqa - all from hd251k (correct filenames, median 1.8M px)
    infovqa = hd251k_by_source.get("infovqa", [])
    output.extend(infovqa)
    print(f"  infovqa:       {len(infovqa):>6d} (all, from hd251k)")

    # 3. ocr_vqa - all (natural scene OCR, HR important)
    ocr_vqa = by_source.get("ocr_vqa", [])
    output.extend(ocr_vqa)
    print(f"  ocr_vqa:       {len(ocr_vqa):>6d} (all)")

    # 4. vg - all (bbox grounding = small target detection)
    vg = by_source.get("vg", [])
    output.extend(vg)
    print(f"  vg:            {len(vg):>6d} (all)")

    # 5. coco grounding - subsample (RefCOCO-style bbox tasks)
    coco_all = by_source.get("coco", [])
    coco_ground = [item for item in coco_all if is_grounding(item)]
    if len(coco_ground) > max_coco_ground:
        random.shuffle(coco_ground)
        coco_ground = coco_ground[:max_coco_ground]
    output.extend(coco_ground)
    print(f"  coco_ground:   {len(coco_ground):>6d} (from {len([x for x in coco_all if is_grounding(x)])} total)")

    # 6. gqa - all (compositional reasoning about regions)
    gqa = by_source.get("gqa", [])
    output.extend(gqa)
    print(f"  gqa:           {len(gqa):>6d} (all)")

    # 7. synthdog - subsample (keep for OCR diversity, but reduce dominance)
    # Pull from hd251k since merged_1m may not have all synthdog
    synthdog = hd251k_by_source.get("synthdog", []) or by_source.get("synthdog", [])
    if len(synthdog) > max_synthdog:
        random.shuffle(synthdog)
        synthdog = synthdog[:max_synthdog]
    output.extend(synthdog)
    print(f"  synthdog:      {len(synthdog):>6d} (capped at {max_synthdog})")

    return output


def verify_images(samples: list, image_folder: str) -> tuple:
    """Check how many samples have their images on disk."""
    existing = []
    missing = []
    for item in samples:
        img_path = os.path.join(image_folder, item.get("image", ""))
        if os.path.exists(img_path):
            existing.append(item)
        else:
            missing.append(item)
    return existing, missing


def main():
    parser = argparse.ArgumentParser(description="Build HR-essential training mix")
    parser.add_argument("--source_json", type=str,
                        default="/root/autodl-tmp/models_data/sft_data/llava_hd_merged_1m.json")
    parser.add_argument("--hd251k_json", type=str,
                        default="/root/autodl-tmp/models_data/sft_data/llava_hd251k.json")
    parser.add_argument("--output_json", type=str,
                        default="/root/autodl-tmp/models_data/sft_data/llava_hr_essential_350k.json")
    parser.add_argument("--image_folder", type=str,
                        default="/root/autodl-tmp/models_data/sft_data/train_split")
    parser.add_argument("--max_coco_ground", type=int, default=50000)
    parser.add_argument("--max_synthdog", type=int, default=40000)
    parser.add_argument("--verify_images", action="store_true",
                        help="Filter out samples whose images don't exist on disk")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading source data from {args.source_json}...")
    with open(args.source_json) as f:
        source_data = json.load(f)
    print(f"  Loaded {len(source_data)} samples")

    print(f"Loading hd251k from {args.hd251k_json}...")
    with open(args.hd251k_json) as f:
        hd251k_data = json.load(f)
    print(f"  Loaded {len(hd251k_data)} samples")

    print("\nBuilding HR-essential mix:")
    output = build_mix(
        source_data, hd251k_data, args.image_folder,
        max_coco_ground=args.max_coco_ground,
        max_synthdog=args.max_synthdog,
        seed=args.seed,
    )

    print(f"\n  TOTAL:         {len(output):>6d}")

    if args.verify_images:
        print(f"\nVerifying images exist in {args.image_folder}...")
        existing, missing = verify_images(output, args.image_folder)
        print(f"  Found: {len(existing)}, Missing: {len(missing)}")
        if missing:
            missing_sources = Counter(classify_sample(x) for x in missing)
            print(f"  Missing by source: {dict(missing_sources)}")
        output = existing
        print(f"  Final count after filtering: {len(output)}")

    random.seed(args.seed)
    random.shuffle(output)

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(output, f)
    print(f"\nSaved {len(output)} samples to {args.output_json}")

    # Summary comparison
    print("\n" + "=" * 60)
    print("COMPARISON: old hd251k vs new HR-essential mix")
    print("=" * 60)
    print(f"{'Source':<15} {'hd251k':>8} {'new mix':>8} {'HR signal':>12}")
    print("-" * 60)

    old_counts = Counter(classify_sample(x) for x in hd251k_data)
    new_counts = Counter(classify_sample(x) for x in output)
    all_sources = sorted(set(list(old_counts.keys()) + list(new_counts.keys())))

    hr_signal = {
        "docvqa": "STRONG", "infovqa": "STRONG", "ocr_vqa": "STRONG",
        "vg": "STRONG", "coco": "STRONG", "gqa": "medium",
        "synthdog": "low", "chartqa": "NONE", "textvqa": "NONE",
        "ai2d": "NONE", "sam": "NONE",
    }

    for src in all_sources:
        old = old_counts.get(src, 0)
        new = new_counts.get(src, 0)
        sig = hr_signal.get(src, "?")
        marker = "  ★" if sig == "STRONG" else ""
        print(f"  {src:<13} {old:>8d} {new:>8d} {sig:>12}{marker}")

    print("-" * 60)
    print(f"  {'TOTAL':<13} {len(hd251k_data):>8d} {len(output):>8d}")

    # HR-essential ratio
    strong_sources = {"docvqa", "infovqa", "ocr_vqa", "vg", "coco", "gqa"}
    old_strong = sum(1 for x in hd251k_data if classify_sample(x) in strong_sources)
    new_strong = sum(1 for x in output if classify_sample(x) in strong_sources)
    print(f"\n  HR-essential ratio: {old_strong/len(hd251k_data)*100:.0f}% → {new_strong/len(output)*100:.0f}%")


if __name__ == "__main__":
    main()
