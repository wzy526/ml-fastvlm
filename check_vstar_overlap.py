#!/usr/bin/env python3
"""Check whether DeepEyes-47k's 'vstar' subset shares images with the
lmms-lab/vstar-bench eval set. Run this on the TRAINING cluster where both
are already cached locally (pyarrow + datasets available).

Usage:
    python3 check_vstar_overlap.py
"""
import glob
import hashlib
import io
import os

os.environ.setdefault("HF_HOME", "/root/autodl-tmp/cache/hf")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import pyarrow.parquet as pq
from datasets import load_dataset


def hash_bytes(b):
    return hashlib.md5(b).hexdigest()


def deepeyes_vstar_hashes():
    cand = glob.glob(
        os.path.expanduser(
            "~/.cache/huggingface/hub/datasets--ChenShawn--DeepEyes-Datasets-47k/snapshots/*/*.parquet"
        )
    ) + glob.glob(
        "/root/autodl-tmp/cache/hf/hub/datasets--ChenShawn--DeepEyes-Datasets-47k/snapshots/*/*.parquet"
    )
    if not cand:
        raise FileNotFoundError("DeepEyes parquet not found in local HF cache; check HF_HOME")
    hashes = {}
    for f in cand:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=32, columns=["data_source", "images"]):
            for row in batch.to_pylist():
                if row.get("data_source") != "vstar":
                    continue
                for img in row.get("images") or []:
                    b = img.get("bytes")
                    if b:
                        hashes.setdefault(hash_bytes(b), []).append(f)
    return hashes


def vstar_bench_hashes():
    ds = load_dataset("lmms-lab/vstar_bench", split="test")
    hashes = {}
    for i, row in enumerate(ds):
        img = row.get("image")
        if img is None:
            continue
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        hashes.setdefault(hash_bytes(buf.getvalue()), []).append(i)
    return hashes


def main():
    print("Hashing DeepEyes 'vstar' subset images...")
    de_hashes = deepeyes_vstar_hashes()
    print(f"  -> {len(de_hashes)} unique images")

    print("Hashing lmms-lab/vstar_bench test images...")
    vb_hashes = vstar_bench_hashes()
    print(f"  -> {len(vb_hashes)} unique images")

    overlap = set(de_hashes) & set(vb_hashes)
    print(f"\nExact byte-level overlap: {len(overlap)} images")
    if overlap:
        print("!!! LEAKAGE DETECTED - remove these DeepEyes rows before training, "
              "or drop vstar_bench from the sweep for this ckpt.")
        for h in list(overlap)[:10]:
            print(f"  hash={h} deepeyes_file={de_hashes[h][0]} vstar_bench_idx={vb_hashes[h]}")
    else:
        print("No exact image overlap found. (Note: this only catches byte-identical "
              "images; resized/re-encoded duplicates would need perceptual hashing.)")


if __name__ == "__main__":
    main()
