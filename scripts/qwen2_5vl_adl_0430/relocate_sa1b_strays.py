#!/usr/bin/env python3
"""Relocate orphaned SA-1B jpgs that landed in the wrong directory.

Background
----------
``download_sa1b_shards.sh`` had a tar-extraction bug (``-C`` arg
ordering + bare-jpg tar structure) that silently dumped every jpg into
the script's CWD instead of ``$SA1B_DIR/sa_NNNNNN/``. This script does
the cleanup: scans the InternVL caption jsonl for the canonical
``sa_<filename>.jpg -> sa_NNNNNN`` mapping, then ``mv``s each stray jpg
to the correct shard directory (atomic, same-disk rename, no copy).

Usage
-----
    # Dry run (default): print stats only, no file moves
    python scripts/qwen2_5vl_adl_0430/relocate_sa1b_strays.py \
        --stray_dir /root/autodl-tmp/ml-fastvlm

    # Actually move
    python scripts/qwen2_5vl_adl_0430/relocate_sa1b_strays.py \
        --stray_dir /root/autodl-tmp/ml-fastvlm \
        --execute
"""

import argparse
import json
import os
import sys
import time
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stray_dir", type=str, required=True,
                    help="Directory containing the orphaned sa_*.jpg files.")
    ap.add_argument("--sa1b_image_dir", type=str,
                    default="/root/autodl-tmp/models_data/sa1b_images")
    ap.add_argument("--iv_jsonl", type=str,
                    default="/root/autodl-tmp/models_data/InternVL-SA-1B-Caption/"
                            "internvl_sa1b_caption_11m_single_image_en.jsonl")
    ap.add_argument("--shard_min", type=int, default=10,
                    help="Lower bound of shard IDs to relocate to (inclusive).")
    ap.add_argument("--shard_max", type=int, default=999,
                    help="Upper bound (inclusive). Only jpgs whose IV jsonl "
                         "entry maps to a shard in [shard_min, shard_max] "
                         "are moved. Helps avoid touching jpgs that belong "
                         "to already-populated shards (e.g. 0-9).")
    ap.add_argument("--execute", action="store_true",
                    help="Actually move files. Default is a dry-run print only.")
    args = ap.parse_args()

    if not os.path.isdir(args.stray_dir):
        sys.exit(f"stray_dir not found: {args.stray_dir}")
    if not os.path.isdir(args.sa1b_image_dir):
        sys.exit(f"sa1b_image_dir not found: {args.sa1b_image_dir}")
    if not os.path.isfile(args.iv_jsonl):
        sys.exit(f"IV jsonl not found: {args.iv_jsonl}")

    print(f"[1/4] Listing stray jpgs in {args.stray_dir}")
    t = time.time()
    stray_set = set()
    with os.scandir(args.stray_dir) as it:
        for entry in it:
            if (entry.is_file(follow_symlinks=False)
                    and entry.name.startswith("sa_")
                    and entry.name.endswith(".jpg")):
                stray_set.add(entry.name)
    print(f"      Found {len(stray_set):,} stray jpgs ({time.time() - t:.1f}s)")

    if not stray_set:
        print("Nothing to relocate. Exiting.")
        return

    print(f"\n[2/4] Building filename -> shard map from IV jsonl")
    print(f"      Scanning {args.iv_jsonl} (only need lines mapping to "
          f"shards [{args.shard_min}, {args.shard_max}])")
    t = time.time()
    fname_to_shard = {}
    needed = set(stray_set)
    found_lines = 0
    total_lines = 0
    with open(args.iv_jsonl) as f:
        for line in f:
            total_lines += 1
            if total_lines % 1_000_000 == 0:
                print(f"      ... {total_lines:,} lines scanned, "
                      f"{found_lines:,} stray jpgs located")
            if not needed:
                break
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            img_path = d.get("image", "")
            if "/" not in img_path:
                continue
            shard, fname = img_path.split("/", 1)
            if fname not in needed:
                continue
            try:
                shard_id = int(shard.replace("sa_", ""))
            except ValueError:
                continue
            if not (args.shard_min <= shard_id <= args.shard_max):
                continue
            fname_to_shard[fname] = shard
            needed.discard(fname)
            found_lines += 1
    print(f"      Built map for {len(fname_to_shard):,} files "
          f"(scanned {total_lines:,} jsonl lines, {time.time() - t:.1f}s)")

    unmapped = stray_set - set(fname_to_shard)
    if unmapped:
        print(f"      WARNING: {len(unmapped):,} stray jpgs have no "
              f"mapping (either belong to shards outside "
              f"[{args.shard_min}, {args.shard_max}], or not in IV jsonl). "
              f"These will be LEFT IN PLACE.")
        for example in list(unmapped)[:5]:
            print(f"        example unmapped: {example}")

    print(f"\n[3/4] Planning moves")
    by_shard = Counter(fname_to_shard.values())
    print(f"      Files to move: {sum(by_shard.values()):,}")
    print(f"      Shards touched: {len(by_shard)}")
    for shard, n in sorted(by_shard.items()):
        target = os.path.join(args.sa1b_image_dir, shard)
        already = 0
        if os.path.isdir(target):
            with os.scandir(target) as it:
                already = sum(1 for e in it if e.name.endswith(".jpg"))
        print(f"        {shard}: +{n:>6,}  (currently has {already:>6,} jpgs)")

    if not args.execute:
        print(f"\n[4/4] DRY RUN — no files moved. Re-run with --execute to apply.")
        return

    print(f"\n[4/4] Executing moves (atomic rename within same disk)")
    t = time.time()
    moved = 0
    skipped_exists = 0
    failed = []
    target_dirs = set()
    for fname, shard in fname_to_shard.items():
        dst_dir = os.path.join(args.sa1b_image_dir, shard)
        if dst_dir not in target_dirs:
            os.makedirs(dst_dir, exist_ok=True)
            target_dirs.add(dst_dir)
    for i, (fname, shard) in enumerate(fname_to_shard.items(), 1):
        src = os.path.join(args.stray_dir, fname)
        dst = os.path.join(args.sa1b_image_dir, shard, fname)
        if os.path.exists(dst):
            os.remove(src)
            skipped_exists += 1
            continue
        try:
            os.rename(src, dst)
            moved += 1
        except OSError as e:
            failed.append((src, dst, str(e)))
        if i % 50_000 == 0:
            print(f"      ... {i:,}/{len(fname_to_shard):,} processed "
                  f"({moved:,} moved, {skipped_exists:,} dup-removed, "
                  f"{len(failed)} failed, {time.time() - t:.1f}s)")

    print(f"\nDone in {time.time() - t:.1f}s")
    print(f"  Moved:               {moved:,}")
    print(f"  Skipped (dst exists, src removed): {skipped_exists:,}")
    print(f"  Failed:              {len(failed)}")
    if failed:
        print("  First 5 failures:")
        for src, dst, err in failed[:5]:
            print(f"    {src} -> {dst}: {err}")

    print(f"\nVerification — per-shard counts after relocation:")
    for shard in sorted(by_shard.keys()):
        d = os.path.join(args.sa1b_image_dir, shard)
        n = sum(1 for e in os.scandir(d) if e.name.endswith(".jpg"))
        print(f"  {shard}: {n:,} jpgs")


if __name__ == "__main__":
    main()
