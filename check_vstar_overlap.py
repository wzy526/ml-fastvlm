#!/usr/bin/env python3
"""Check whether the deepeyes / visualprobe images that were merged into
llava_hr_ocr_vs_0812.json overlap with the V*Bench (lmms-lab/vstar-bench)
eval images.

Run on the TRAINING cluster (needs: datasets, pillow; vstar-bench will be
pulled from local HF cache or the hf-mirror endpoint, ~270MB).

    SFT_DIR=/data/oss_bucket_0/wangziyi/models_data python3 check_vstar_overlap.py

Because the training copies were re-encoded to JPG by PIL, byte/pixel-exact
comparison cannot catch duplicates — we use a 64-bit dHash and report
near-duplicates by Hamming distance (<=4 is a near-certain duplicate,
5-8 is worth eyeballing).

Writes suspect training-image paths to vstar_overlap_suspects.txt.
"""
import hashlib
import io
import os
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from PIL import Image

SFT_DIR = os.environ.get("SFT_DIR", "/root/autodl-tmp/models_data/sft_data")
TRAIN_SPLIT = os.path.join(SFT_DIR, "train_split")
CHECK_DIRS = ["deepeyes", "visualprobe"]
REPORT = "vstar_overlap_suspects.txt"

EXACT_THRESH = 4    # hamming <= 4  -> treat as duplicate
REVIEW_THRESH = 8   # hamming 5..8  -> print for manual review


def dhash(img):
    """64-bit difference hash."""
    img = img.convert("L")
    img.thumbnail((256, 256))  # cheap pre-shrink keeps resize stable
    img = img.resize((9, 8), Image.LANCZOS)
    px = list(img.getdata())
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | (px[row * 9 + col] > px[row * 9 + col + 1])
    return bits


def hamming(a, b):
    return bin(a ^ b).count("1")


def bench_hashes():
    from datasets import load_dataset, Image as DsImage
    ds = load_dataset("lmms-lab/vstar-bench", split="test")
    ds = ds.cast_column("image", DsImage(decode=False))
    out = []  # (question_id, md5_of_raw_bytes, dhash)
    seen_md5 = set()
    for row in ds:
        b = row["image"]["bytes"]
        m = hashlib.md5(b).hexdigest()
        if m in seen_md5:  # same image reused by several questions
            continue
        seen_md5.add(m)
        h = dhash(Image.open(io.BytesIO(b)))
        out.append((row.get("question_id", "?"), m, h))
    return out


def train_image_paths():
    paths = []
    for d in CHECK_DIRS:
        root = os.path.join(TRAIN_SPLIT, d)
        if not os.path.isdir(root):
            print(f"  [WARN] missing dir: {root}")
            continue
        for cur, _dirs, files in os.walk(root):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    paths.append(os.path.join(cur, f))
    return paths


def hash_one(path):
    try:
        with open(path, "rb") as f:
            raw = f.read()
        return path, hashlib.md5(raw).hexdigest(), dhash(Image.open(io.BytesIO(raw)))
    except Exception as e:
        print(f"  ! {path}: {e}")
        return path, None, None


def main():
    print("Hashing V*Bench eval images...")
    bench = bench_hashes()
    print(f"  -> {len(bench)} unique bench images")
    bench_md5 = {m for _, m, _ in bench}

    paths = train_image_paths()
    print(f"Hashing {len(paths)} training images from {CHECK_DIRS} ...")
    with ThreadPoolExecutor(max_workers=16) as ex:
        results = [r for r in ex.map(hash_one, paths) if r[1] is not None]

    dup, review = [], []
    for path, m, h in results:
        if m in bench_md5:
            dup.append((path, "byte-exact", 0))
            continue
        best = min(bench, key=lambda t: hamming(h, t[2]))
        d = hamming(h, best[2])
        if d <= EXACT_THRESH:
            dup.append((path, best[0], d))
        elif d <= REVIEW_THRESH:
            review.append((path, best[0], d))

    print(f"\n=== duplicates (hamming <= {EXACT_THRESH}): {len(dup)} ===")
    for path, qid, d in dup[:20]:
        print(f"  d={d:2d}  {path}  ~  bench:{qid}")
    print(f"=== manual-review (hamming {EXACT_THRESH + 1}..{REVIEW_THRESH}): {len(review)} ===")
    for path, qid, d in review[:20]:
        print(f"  d={d:2d}  {path}  ~  bench:{qid}")

    with open(REPORT, "w") as f:
        for path, qid, d in dup + review:
            f.write(f"{d}\t{path}\t{qid}\n")
    print(f"\nfull list -> {REPORT}")

    if not dup:
        print("\nNo duplicates at the strict threshold: vstar_bench sweep numbers "
              "for the new ckpt can be trusted.")
    else:
        print("\nLEAKAGE: remove these images' entries from llava_hr_ocr_vs_0812.json "
              "before training, or ignore vstar_bench for this ckpt.")


if __name__ == "__main__":
    main()
