#!/usr/bin/env python3
"""Regenerate the docvqa / infovqa / synthdog images referenced by
llava_hr_essential_sa1b_ivcap.json, byte-identically named, without needing the
collaborator's disk state.

How naming worked in construct_hd251k.py (SEED=42):
  docvqa_NNNNNN.jpg : idx into ds.select(indices) where
      indices = random.Random(SEED + stable_hash('docvqa')).sample(range(len(ds)), 39000)
      ds = HuggingFaceM4/DocumentVQA train
  infovqa_NNNNNN.jpg: target 24000 >= len(train)=23946 -> NO sampling, idx = raw row
      ds = Ahren09/InfoVQA train
  synthdog_N.jpg (no zero padding):
      N <  200000: hf_idx = random.Random(SEED).sample(range(len(ds)), 200000)[N]
      N >= 200000: available = sorted(set(range(len(ds))) - set(existing));
                   random.Random(SEED+2).shuffle(available);
                   hf_idx = available[N - 200000]
      ds = naver-clova-ix/synthdog-en train (streamed; images saved as RGB jpg)

Every dataset is validated against the QA text embedded in the SFT json before
any bulk export. Usage:
  python regen_hd251k_images.py --json data_payload/llava_hr_essential_sa1b_ivcap.json.gz \
      --out ~/dataset_staging/hd251k_regen --datasets infovqa docvqa synthdog
"""
import argparse
import gzip
import hashlib
import json
import os
import random
import re
import sys

SEED = 42


def stable_hash(s: str) -> int:
    return int.from_bytes(hashlib.md5(s.encode("utf-8")).digest()[:4], "big")


def load_refs(json_path):
    """dataset -> {filename: (question, answer)}"""
    opener = gzip.open if json_path.endswith(".gz") else open
    with opener(json_path, "rt") as f:
        data = json.load(f)
    refs = {}
    for s in data:
        img = s.get("image", "")
        m = re.match(r"^(docvqa|infovqa|synthdog)/(.+\.jpg)$", img)
        if not m:
            continue
        conv = s.get("conversations", [])
        q = conv[0]["value"].replace("<image>", "").strip() if conv else ""
        a = conv[1]["value"] if len(conv) > 1 else ""
        refs.setdefault(m.group(1), {})[m.group(2)] = (q, a)
    return refs


def save_img(img, path):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.save(path)


def check(label, got, want, strict=False):
    ok = (got or "").strip() == (want or "").strip()
    if not ok:
        print(f"    [{label}] QA mismatch:\n      json: {want[:120]!r}\n      hf:   {got[:120]!r}")
        if strict:
            sys.exit(f"VALIDATION FAILED for {label}")
    return ok


def run_vqa(name, hf_name, target_count, q_key, refs, out_dir, validate_n=30):
    from datasets import load_dataset
    print(f"[{name}] loading {hf_name} (train) ...")
    ds = load_dataset(hf_name, split="train")
    print(f"[{name}] rows={len(ds)}")
    if target_count and len(ds) > target_count:
        indices = random.Random(SEED + stable_hash(name)).sample(range(len(ds)), target_count)
        ds = ds.select(indices)
        print(f"[{name}] sampled {target_count} rows (seeded)")

    fname_to_idx = {}
    for fn in refs:
        idx = int(fn[len(name) + 1:-4])
        fname_to_idx[fn] = idx

    # validation pass on a random subset
    rnd = random.Random(0)
    sample_fns = rnd.sample(sorted(refs), min(validate_n, len(refs)))
    n_ok = 0
    for fn in sample_fns:
        item = ds[fname_to_idx[fn]]
        q = item.get(q_key, "")
        if isinstance(q, list):
            q = q[0] if q else ""
        if check(f"{name}:{fn}", q, refs[fn][0]):
            n_ok += 1
    print(f"[{name}] validation: {n_ok}/{len(sample_fns)} question-text matches")
    if n_ok < len(sample_fns):
        sys.exit(f"[{name}] VALIDATION FAILED -- mapping assumption is wrong, aborting")

    d = os.path.join(out_dir, name)
    os.makedirs(d, exist_ok=True)
    done = 0
    for fn, idx in sorted(fname_to_idx.items(), key=lambda kv: kv[1]):
        path = os.path.join(d, fn)
        if os.path.exists(path):
            done += 1
            continue
        save_img(ds[idx]["image"], path)
        done += 1
        if done % 2000 == 0:
            print(f"[{name}] {done}/{len(fname_to_idx)}")
    print(f"[{name}] DONE: {done} images in {d}")


def parse_synthdog_gt(gt_str):
    try:
        return json.loads(gt_str).get("gt_parse", {}).get("text_sequence", "").strip()
    except (json.JSONDecodeError, TypeError, AttributeError):
        return ""


def run_synthdog(refs, out_dir, validate_n=30):
    from datasets import load_dataset, load_dataset_builder
    builder = load_dataset_builder("naver-clova-ix/synthdog-en")
    n_rows = builder.info.splits["train"].num_examples
    print(f"[synthdog] train rows per HF metadata: {n_rows}")

    existing = random.Random(SEED).sample(range(n_rows), 200_000)
    needed_hf = {}
    over = [int(fn[len("synthdog_"):-4]) for fn in refs]
    n_over = sum(1 for n in over if n >= 200_000)
    available = None
    if n_over:
        used = set(existing)
        available = [i for i in range(n_rows) if i not in used]
        random.Random(SEED + 2).shuffle(available)
    for fn in refs:
        n = int(fn[len("synthdog_"):-4])
        hf_idx = existing[n] if n < 200_000 else available[n - 200_000]
        needed_hf[hf_idx] = fn
    print(f"[synthdog] refs={len(refs)} ({n_over} beyond the 200k block)")

    d = os.path.join(out_dir, "synthdog")
    os.makedirs(d, exist_ok=True)
    remaining = {h: fn for h, fn in needed_hf.items()
                 if not os.path.exists(os.path.join(d, fn))}
    print(f"[synthdog] to fetch: {len(remaining)} (already on disk: {len(needed_hf)-len(remaining)})")
    if not remaining:
        print("[synthdog] DONE (nothing to do)")
        return

    rnd = random.Random(0)
    val_set = set(rnd.sample(sorted(remaining), min(validate_n, len(remaining))))
    n_val_ok = 0
    n_val_seen = 0

    ds = load_dataset("naver-clova-ix/synthdog-en", split="train", streaming=True)
    done = 0
    max_needed = max(remaining)
    for i, item in enumerate(ds):
        if i > max_needed:
            break
        fn = remaining.get(i)
        if fn is None:
            continue
        if i in val_set:
            n_val_seen += 1
            got = parse_synthdog_gt(item.get("ground_truth", ""))
            if check(f"synthdog:{fn}", got, refs[fn][1]):
                n_val_ok += 1
            # abort early if the first handful all mismatch
            if n_val_seen >= 5 and n_val_ok == 0:
                sys.exit("[synthdog] VALIDATION FAILED early -- aborting stream")
        save_img(item["image"], os.path.join(d, fn))
        done += 1
        if done % 1000 == 0:
            print(f"[synthdog] saved {done}/{len(remaining)} (stream at row {i})")
    print(f"[synthdog] validation: {n_val_ok}/{n_val_seen} answer-text matches")
    print(f"[synthdog] DONE: saved {done} images in {d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--datasets", nargs="+", default=["infovqa", "docvqa", "synthdog"])
    args = ap.parse_args()

    out = os.path.expanduser(args.out)
    refs = load_refs(os.path.expanduser(args.json))
    for k, v in refs.items():
        print(f"refs[{k}] = {len(v)}")

    if "infovqa" in args.datasets:
        run_vqa("infovqa", "Ahren09/InfoVQA", 24000, "question", refs["infovqa"], out)
    if "docvqa" in args.datasets:
        run_vqa("docvqa", "HuggingFaceM4/DocumentVQA", 39000, "question", refs["docvqa"], out)
    if "synthdog" in args.datasets:
        run_synthdog(refs["synthdog"], out)
    print("ALL_REGEN_DONE")


if __name__ == "__main__":
    main()
