"""Merge LLaVA base dataset (downsampled) with HD supplements.

Strategy:
  1. From llava_v1_5_mix665k_shuffled.json (624k), keep OCR/text-heavy
     subsets in full (ocr_vqa, textvqa), proportionally downsample the
     rest (coco, gqa, vg) to hit BASE_TARGET_SIZE.
  2. Combine with hd_supplements.jsonl.
  3. Shuffle and write the final training JSON.
"""

import json
import random
import os
from collections import defaultdict

# ================= Config =================
SFT_DATA_DIR = "/mnt/ephemeral/sft_data"

BASE_JSON = os.path.join(SFT_DATA_DIR, "llava_v1_5_mix665k_shuffled.json")
HD_JSONL = os.path.join(SFT_DATA_DIR, "hd_supplements.jsonl")
OUTPUT_JSON = os.path.join(SFT_DATA_DIR, "llava_hd_merged_1m.json")

BASE_TARGET_SIZE = 387_000
SEED = 42

KEEP_FULL_PREFIXES = {"ocr_vqa", "textvqa"}
# ===========================================


def get_image_prefix(item):
    img = item.get("image", "")
    return img.split("/")[0] if "/" in img else "unknown"


def load_base_dataset(path):
    print(f"Loading base dataset: {path}")
    with open(path, "r") as f:
        data = json.load(f)
    print(f"  Total samples: {len(data)}")
    return data


def load_hd_supplements(path):
    print(f"Loading HD supplements: {path}")
    data = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    print(f"  Total samples: {len(data)}")
    return data


def downsample_base(data, target_size):
    """Keep OCR/text subsets in full; proportionally downsample the rest."""
    rng = random.Random(SEED)

    buckets = defaultdict(list)
    for item in data:
        buckets[get_image_prefix(item)].append(item)

    print("\n--- Base dataset breakdown ---")
    for prefix in sorted(buckets):
        print(f"  {prefix}: {len(buckets[prefix])}")

    keep_items = []
    downsample_pool = {}

    for prefix, items in buckets.items():
        if prefix in KEEP_FULL_PREFIXES:
            keep_items.extend(items)
        else:
            downsample_pool[prefix] = items

    kept_count = len(keep_items)
    pool_total = sum(len(v) for v in downsample_pool.values())
    budget = max(0, target_size - kept_count)

    if budget >= pool_total:
        print(f"\nBudget ({budget}) >= pool ({pool_total}), keeping all.")
        for items in downsample_pool.values():
            keep_items.extend(items)
    else:
        ratio = budget / pool_total
        print(f"\nDownsampling non-OCR subsets (ratio={ratio:.4f}):")
        for prefix in sorted(downsample_pool):
            items = downsample_pool[prefix]
            n = int(len(items) * ratio)
            sampled = rng.sample(items, n)
            print(f"  {prefix}: {len(items)} -> {n}")
            keep_items.extend(sampled)

    print(f"\nBase after downsampling: {len(keep_items)} "
          f"(target was {target_size})")
    return keep_items


def main():
    random.seed(SEED)

    base_data = load_base_dataset(BASE_JSON)
    base_sampled = downsample_base(base_data, BASE_TARGET_SIZE)

    hd_data = load_hd_supplements(HD_JSONL)

    merged = base_sampled + hd_data
    print(f"\n--- Merged dataset ---")
    print(f"  Base:    {len(base_sampled)}")
    print(f"  HD:      {len(hd_data)}")
    print(f"  Total:   {len(merged)}")

    print("\nShuffling...")
    random.shuffle(merged)

    # Final breakdown
    final_buckets = defaultdict(int)
    for item in merged:
        final_buckets[get_image_prefix(item)] += 1

    print("\n--- Final breakdown ---")
    for prefix in sorted(final_buckets):
        print(f"  {prefix}: {final_buckets[prefix]}")
    print(f"  TOTAL: {sum(final_buckets.values())}")

    print(f"\nWriting {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, "w") as f:
        json.dump(merged, f)

    size_mb = os.path.getsize(OUTPUT_JSON) / (1024 * 1024)
    print(f"Done! {len(merged)} samples, {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
