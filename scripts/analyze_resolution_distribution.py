#!/usr/bin/env python3
"""Analyze per-subset resolution (pixel count) distribution of the training dataset."""

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def get_image_size(path: str):
    """Return (width, height) by reading only the image header."""
    try:
        with Image.open(path) as img:
            return img.size  # (w, h)
    except Exception:
        return None


def process_batch(batch):
    """Process a batch of (subset, path) tuples; return list of (subset, w, h)."""
    results = []
    for subset, path in batch:
        sz = get_image_size(path)
        if sz is not None:
            results.append((subset, sz[0], sz[1]))
    return results


def humanize_pixels(n):
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    if n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(int(n))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str,
                        default="/home/coder/downloaded_data/sft_data/llava_hd_merged_1m.json")
    parser.add_argument("--image_folder", type=str,
                        default="/home/coder/downloaded_data/sft_data/train_split")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--sample_per_subset", type=int, default=0,
                        help="If >0, randomly sample this many per subset (faster)")
    args = parser.parse_args()

    print(f"Loading {args.data_path} ...")
    with open(args.data_path) as f:
        data = json.load(f)
    print(f"  Total samples: {len(data)}")

    subset_items = defaultdict(list)
    for item in data:
        img = item.get("image", "")
        if not img:
            continue
        subset = img.split("/")[0] if "/" in img else "(root)"
        full_path = os.path.join(args.image_folder, img)
        subset_items[subset].append(full_path)

    print(f"  Subsets: {sorted(subset_items.keys())}")
    for s in sorted(subset_items):
        print(f"    {s}: {len(subset_items[s])} images")

    import random
    if args.sample_per_subset > 0:
        for s in subset_items:
            if len(subset_items[s]) > args.sample_per_subset:
                subset_items[s] = random.sample(subset_items[s], args.sample_per_subset)
        print(f"  (Sampled {args.sample_per_subset} per subset)")

    all_tasks = []
    for subset, paths in subset_items.items():
        for p in paths:
            all_tasks.append((subset, p))
    total = len(all_tasks)
    print(f"\nScanning {total} images with {args.workers} workers ...")

    batches = [all_tasks[i:i + args.batch_size]
               for i in range(0, len(all_tasks), args.batch_size)]

    subset_data = defaultdict(list)  # subset -> list of (w, h, pixels)
    done = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_batch, b): len(b) for b in batches}
        for future in as_completed(futures):
            batch_size = futures[future]
            try:
                results = future.result()
                for subset, w, h in results:
                    subset_data[subset].append((w, h, w * h))
                failed += batch_size - len(results)
            except Exception as e:
                failed += batch_size
                print(f"  Batch error: {e}", file=sys.stderr)
            done += batch_size
            if done % 50000 < args.batch_size or done == total:
                print(f"  Progress: {done}/{total} ({done*100//total}%)", flush=True)

    if failed:
        print(f"\n  WARNING: {failed} images failed to read")

    import numpy as np

    print("\n" + "=" * 100)
    print(f"{'Subset':<14} {'Count':>8} {'MeanW':>7} {'MeanH':>7} "
          f"{'MeanPx':>10} {'MedianPx':>10} {'MinPx':>10} {'MaxPx':>10} "
          f"{'P10':>10} {'P25':>10} {'P75':>10} {'P90':>10}")
    print("=" * 100)

    all_pixels = []
    rows = []

    for subset in sorted(subset_data):
        entries = subset_data[subset]
        ws = np.array([e[0] for e in entries])
        hs = np.array([e[1] for e in entries])
        pxs = np.array([e[2] for e in entries], dtype=np.int64)
        all_pixels.extend(pxs.tolist())

        row = {
            "subset": subset,
            "count": len(entries),
            "mean_w": np.mean(ws),
            "mean_h": np.mean(hs),
            "mean_px": np.mean(pxs),
            "median_px": np.median(pxs),
            "min_px": np.min(pxs),
            "max_px": np.max(pxs),
            "p10": np.percentile(pxs, 10),
            "p25": np.percentile(pxs, 25),
            "p75": np.percentile(pxs, 75),
            "p90": np.percentile(pxs, 90),
        }
        rows.append(row)
        print(f"{row['subset']:<14} {row['count']:>8} "
              f"{row['mean_w']:>7.0f} {row['mean_h']:>7.0f} "
              f"{humanize_pixels(row['mean_px']):>10} {humanize_pixels(row['median_px']):>10} "
              f"{humanize_pixels(row['min_px']):>10} {humanize_pixels(row['max_px']):>10} "
              f"{humanize_pixels(row['p10']):>10} {humanize_pixels(row['p25']):>10} "
              f"{humanize_pixels(row['p75']):>10} {humanize_pixels(row['p90']):>10}")

    all_pixels = np.array(all_pixels, dtype=np.int64)
    print("-" * 100)
    print(f"{'TOTAL':<14} {len(all_pixels):>8} "
          f"{'':>7} {'':>7} "
          f"{humanize_pixels(np.mean(all_pixels)):>10} {humanize_pixels(np.median(all_pixels)):>10} "
          f"{humanize_pixels(np.min(all_pixels)):>10} {humanize_pixels(np.max(all_pixels)):>10} "
          f"{humanize_pixels(np.percentile(all_pixels, 10)):>10} "
          f"{humanize_pixels(np.percentile(all_pixels, 25)):>10} "
          f"{humanize_pixels(np.percentile(all_pixels, 75)):>10} "
          f"{humanize_pixels(np.percentile(all_pixels, 90)):>10}")
    print("=" * 100)

    # Pixel-count histogram buckets
    buckets = [
        (0, 100_000, "<100K"),
        (100_000, 250_000, "100K-250K"),
        (250_000, 500_000, "250K-500K"),
        (500_000, 1_000_000, "500K-1M"),
        (1_000_000, 2_000_000, "1M-2M"),
        (2_000_000, 4_000_000, "2M-4M"),
        (4_000_000, 8_000_000, "4M-8M"),
        (8_000_000, 16_000_000, "8M-16M"),
        (16_000_000, float("inf"), "16M+"),
    ]

    print(f"\n{'Pixel-count histogram per subset':}")
    header = f"{'Bucket':<14}"
    for subset in sorted(subset_data):
        header += f" {subset:>12}"
    header += f" {'TOTAL':>12}"
    print(header)
    print("-" * len(header))

    for lo, hi, label in buckets:
        line = f"{label:<14}"
        total_in_bucket = 0
        for subset in sorted(subset_data):
            entries = subset_data[subset]
            pxs = np.array([e[2] for e in entries], dtype=np.int64)
            cnt = int(np.sum((pxs >= lo) & (pxs < hi)))
            total_in_bucket += cnt
            pct = cnt * 100 / len(entries) if entries else 0
            line += f" {cnt:>6}({pct:>4.1f}%)"
        pct_total = total_in_bucket * 100 / len(all_pixels)
        line += f" {total_in_bucket:>6}({pct_total:>4.1f}%)"
        print(line)

    # Common resolutions per subset
    print(f"\nTop-5 resolutions (WxH) per subset:")
    print("-" * 80)
    for subset in sorted(subset_data):
        from collections import Counter
        entries = subset_data[subset]
        res_counter = Counter((e[0], e[1]) for e in entries)
        top5 = res_counter.most_common(5)
        print(f"  {subset}:")
        for (w, h), cnt in top5:
            pct = cnt * 100 / len(entries)
            print(f"    {w:>5}x{h:<5} = {humanize_pixels(w*h):>8} px  ({cnt:>6}, {pct:>5.1f}%)")


if __name__ == "__main__":
    main()
