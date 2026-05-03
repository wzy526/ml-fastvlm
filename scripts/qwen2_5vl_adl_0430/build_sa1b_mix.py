#!/usr/bin/env python3
"""Build SA-1B mixed training data from InternVL caption + AS-Core region data.

Converts both datasets to LLaVA conversation format and merges with the
existing HR-essential 350K mix. Also verifies which images exist on disk
and outputs a ready-to-train JSON.

Usage:
    python scripts/qwen2_5vl_adl_0430/build_sa1b_mix.py \
        --sa1b_image_dir /root/autodl-tmp/models_data/sa1b_images \
        --output_json /root/autodl-tmp/models_data/sft_data/llava_hr_essential_sa1b_mix.json

    # To also pack sa1b images for transfer to other machines:
    python scripts/qwen2_5vl_adl_0430/build_sa1b_mix.py \
        --sa1b_image_dir /root/autodl-tmp/models_data/sa1b_images \
        --output_json /root/autodl-tmp/models_data/sft_data/llava_hr_essential_sa1b_mix.json \
        --pack_dir /root/autodl-tmp/packed_sa1b
"""

import argparse
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


def load_jsonl(path, max_lines=None):
    data = []
    with open(path) as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            data.append(json.loads(line))
    return data


def convert_internvl_caption(samples, max_n=50000, seed=42):
    """Convert InternVL single-image captions to LLaVA format.
    Already in correct format, just subsample.
    """
    random.seed(seed)
    if len(samples) > max_n:
        samples = random.sample(samples, max_n)

    output = []
    for s in samples:
        output.append({
            "image": s["image"],
            "conversations": s["conversations"],
        })
    return output


def convert_as_core_region_caption(samples, max_n=50000, seed=42):
    """Convert AS-Core region_caption to LLaVA format with bbox."""
    random.seed(seed)
    if len(samples) > max_n:
        samples = random.sample(samples, max_n)

    prompts = [
        "What is in the region [{x1}, {y1}, {x2}, {y2}] of this image?",
        "Describe the object at [{x1}, {y1}, {x2}, {y2}].",
        "What can you see in the area [{x1}, {y1}, {x2}, {y2}]?",
        "Please describe the region [{x1}, {y1}, {x2}, {y2}] in detail.",
    ]

    output = []
    for s in samples:
        bbox = s["bbox"]
        w, h = s["width"], s["height"]
        x1 = round(bbox[0] / w, 3)
        y1 = round(bbox[1] / h, 3)
        x2 = round(bbox[2] / w, 3)
        y2 = round(bbox[3] / h, 3)

        prompt = random.choice(prompts).format(x1=x1, y1=y1, x2=x2, y2=y2)
        caption = s["caption"]

        output.append({
            "image": s["image"],
            "conversations": [
                {"from": "human", "value": f"<image>\n{prompt}"},
                {"from": "gpt", "value": caption},
            ],
        })
    return output


def convert_as_core_region_vqa(samples, max_n=50000, seed=42):
    """Convert AS-Core region_vqa to LLaVA format with bbox context."""
    random.seed(seed)
    if len(samples) > max_n:
        samples = random.sample(samples, max_n)

    output = []
    for s in samples:
        bbox = s["bbox"]
        w, h = s["width"], s["height"]
        x1 = round(bbox[0] / w, 3)
        y1 = round(bbox[1] / h, 3)
        x2 = round(bbox[2] / w, 3)
        y2 = round(bbox[3] / h, 3)

        question = s["question"]
        question_with_region = f"Regarding the region [{x1}, {y1}, {x2}, {y2}]: {question}"

        output.append({
            "image": s["image"],
            "conversations": [
                {"from": "human", "value": f"<image>\n{question_with_region}"},
                {"from": "gpt", "value": s["answer"]},
            ],
        })
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sa1b_image_dir", type=str,
                        default="/root/autodl-tmp/models_data/sa1b_images")
    parser.add_argument("--as_core_dir", type=str,
                        default="/root/autodl-tmp/models_data/AS-Core")
    parser.add_argument("--internvl_dir", type=str,
                        default="/root/autodl-tmp/models_data/InternVL-SA-1B-Caption")
    parser.add_argument("--hr_essential_json", type=str,
                        default="/root/autodl-tmp/models_data/sft_data/llava_hr_essential_350k.json")
    parser.add_argument("--output_json", type=str,
                        default="/root/autodl-tmp/models_data/sft_data/llava_hr_essential_sa1b_mix.json")
    parser.add_argument("--max_internvl_caption", type=int, default=50000)
    parser.add_argument("--max_region_caption", type=int, default=50000)
    parser.add_argument("--max_region_vqa", type=int, default=50000)
    parser.add_argument("--pack_dir", type=str, default=None,
                        help="If set, copy only used SA-1B images here for easy transfer")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # 1. Load existing HR-essential mix
    print(f"Loading HR-essential base: {args.hr_essential_json}")
    with open(args.hr_essential_json) as f:
        base_data = json.load(f)
    print(f"  Base samples: {len(base_data)}")

    # 2. Load and convert InternVL captions (only from downloaded shards)
    iv_path = os.path.join(args.internvl_dir, "internvl_sa1b_caption_11m_single_image_en.jsonl")
    print(f"\nLoading InternVL captions: {iv_path}")
    # Only load samples from shards that exist on disk
    available_shards = set()
    if os.path.isdir(args.sa1b_image_dir):
        for d in os.listdir(args.sa1b_image_dir):
            if d.startswith("sa_"):
                available_shards.add(d)
    print(f"  Available shards on disk: {sorted(available_shards)}")

    iv_samples = []
    with open(iv_path) as f:
        for line in f:
            d = json.loads(line)
            shard = d["image"].split("/")[0]
            if shard in available_shards:
                iv_samples.append(d)
    print(f"  InternVL samples in available shards: {len(iv_samples)}")

    iv_converted = convert_internvl_caption(iv_samples, args.max_internvl_caption, args.seed)
    print(f"  Selected: {len(iv_converted)}")

    # 3. Load and convert AS-Core region caption
    rc_path = os.path.join(args.as_core_dir, "region_caption_400k.jsonl")
    print(f"\nLoading AS-Core region captions: {rc_path}")
    rc_samples = [s for s in load_jsonl(rc_path) if s["image"].split("/")[0] in available_shards]
    print(f"  Region caption in available shards: {len(rc_samples)}")

    rc_converted = convert_as_core_region_caption(rc_samples, args.max_region_caption, args.seed)
    print(f"  Selected: {len(rc_converted)}")

    # 4. Load and convert AS-Core region VQA
    vqa_path = os.path.join(args.as_core_dir, "region_vqa_1m.jsonl")
    print(f"\nLoading AS-Core region VQA: {vqa_path}")
    vqa_samples = [s for s in load_jsonl(vqa_path) if s["image"].split("/")[0] in available_shards]
    print(f"  Region VQA in available shards: {len(vqa_samples)}")

    vqa_converted = convert_as_core_region_vqa(vqa_samples, args.max_region_vqa, args.seed)
    print(f"  Selected: {len(vqa_converted)}")

    # 5. Merge all
    # Tag SA-1B samples with sa1b_images prefix so image_folder logic works
    # Training script uses --image_folder which is prepended to each sample's "image" path
    # SA-1B images are at sa1b_images/sa_XXXXXX/sa_YYYY.jpg
    # We need to make sure the image path is relative to the image_folder
    sa1b_all = iv_converted + rc_converted + vqa_converted

    # Verify images exist on disk
    print(f"\nVerifying SA-1B images on disk...")
    verified = []
    missing = 0
    for s in sa1b_all:
        img_path = os.path.join(args.sa1b_image_dir, s["image"])
        if os.path.exists(img_path):
            verified.append(s)
        else:
            missing += 1
    print(f"  Verified: {len(verified)}, Missing: {missing}")

    # Prefix SA-1B image paths so they work with train_split as image_folder
    # We'll create a symlink: train_split/sa1b -> sa1b_images/
    # So image path becomes: sa1b/sa_000000/sa_1.jpg (relative to train_split)
    # BUT: actually simpler to just use the sa1b_images dir path
    # The training script can handle this if we create a symlink
    for s in verified:
        s["image"] = "sa1b/" + s["image"]

    # Merge with base
    merged = base_data + verified
    random.shuffle(merged)

    print(f"\n{'='*60}")
    print(f"Final mix composition:")
    print(f"  HR-essential base:     {len(base_data):>8,}")
    print(f"  InternVL caption:      {len(iv_converted):>8,}")
    print(f"  AS-Core region cap:    {len(rc_converted):>8,}")
    print(f"  AS-Core region VQA:    {len(vqa_converted):>8,}")
    print(f"  SA-1B verified:        {len(verified):>8,}")
    print(f"  TOTAL:                 {len(merged):>8,}")
    print(f"{'='*60}")

    # Save
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(merged, f)
    print(f"\nSaved to: {args.output_json}")

    # Create symlink for training
    train_split = os.path.dirname(args.hr_essential_json) + "/train_split"
    symlink_path = os.path.join(train_split, "sa1b")
    if not os.path.exists(symlink_path):
        os.symlink(args.sa1b_image_dir, symlink_path)
        print(f"Created symlink: {symlink_path} -> {args.sa1b_image_dir}")
    else:
        print(f"Symlink already exists: {symlink_path}")

    # Optional: pack only used SA-1B images
    if args.pack_dir:
        print(f"\nPacking used SA-1B images to {args.pack_dir}...")
        used_images = set()
        for s in verified:
            used_images.add(s["image"].replace("sa1b/", ""))

        os.makedirs(args.pack_dir, exist_ok=True)
        copied = 0
        for img_rel in sorted(used_images):
            src = os.path.join(args.sa1b_image_dir, img_rel)
            dst = os.path.join(args.pack_dir, img_rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                copied += 1

        print(f"  Packed {copied} images")
        du_result = os.popen(f"du -sh {args.pack_dir}").read().strip()
        print(f"  Pack size: {du_result}")
        print(f"\n  To transfer: rsync -avP {args.pack_dir}/ target:/root/autodl-tmp/models_data/sa1b_images/")


if __name__ == "__main__":
    main()
