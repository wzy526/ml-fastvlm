#!/usr/bin/env python3
"""Download and convert URECA dataset to LLaVA conversation format.

URECA (Unique Region Caption Anything) provides 138K mask-caption pairs
from SA-1B. Each sample has a region mask and a unique caption describing
what distinguishes that region from its surroundings — perfect supervision
for DAT's cross-attention to learn "look at THIS spot in HR".

Usage:
    python scripts/qwen2_5vl_adl_0430/prepare_ureca_data.py \
        --sa1b_dir /root/autodl-tmp/models_data/sft_data/sa1b \
        --output_json /root/autodl-tmp/models_data/sft_data/ureca_train_llava.json

Prerequisites:
    1. pip install huggingface_hub
    2. SA-1B images should be available at --sa1b_dir
       (If not available, the script will tell you which images are needed)
"""

import argparse
import json
import os
import random
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Please install huggingface_hub: pip install huggingface_hub")
    raise


def download_ureca_captions(cache_dir: str) -> str:
    """Download URECA caption file from HuggingFace."""
    print("Downloading URECA captions from HuggingFace...")
    path = hf_hub_download(
        repo_id="cvlab-kaist/URECA",
        filename="ureca_train.json",
        repo_type="dataset",
        cache_dir=cache_dir,
    )
    print(f"Downloaded to: {path}")
    return path


def build_region_prompt(caption: str, mask_bbox: dict) -> list:
    """Create a conversation asking about a specific region.

    We convert the mask bounding box to a region description so the model
    learns to attend to specific spatial locations via HR features.
    """
    prompts = [
        "Describe the highlighted region in detail.",
        "What is unique about the marked area in this image?",
        "Please provide a detailed description of the indicated region.",
        "What do you see in the specified area of this image?",
        "Describe what makes this particular region distinctive.",
    ]

    if mask_bbox:
        x1, y1, x2, y2 = mask_bbox["x1"], mask_bbox["y1"], mask_bbox["x2"], mask_bbox["y2"]
        region_hint = f" The region of interest is approximately at [{x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}]."
    else:
        region_hint = ""

    prompt = random.choice(prompts)

    return [
        {"from": "human", "value": f"<image>\n{prompt}{region_hint}"},
        {"from": "gpt", "value": caption},
    ]


def convert_ureca_to_llava(ureca_data: list, sa1b_dir: str) -> list:
    """Convert URECA format to LLaVA conversation format."""
    llava_samples = []
    missing_images = 0

    for item in ureca_data:
        image_id = item.get("image_id", "")
        caption = item.get("caption", "")
        mask_info = item.get("mask", {})

        if not caption:
            continue

        # SA-1B images are stored as sa_{image_id}.jpg
        # Handle various naming conventions
        if isinstance(image_id, int):
            image_file = f"sa_{image_id}.jpg"
        else:
            image_file = str(image_id)
            if not image_file.endswith((".jpg", ".png")):
                image_file = f"sa_{image_file}.jpg"

        # Extract normalized bbox if available
        mask_bbox = None
        if "bbox" in mask_info:
            bbox = mask_info["bbox"]
            if len(bbox) == 4:
                img_w = mask_info.get("width", 1)
                img_h = mask_info.get("height", 1)
                if img_w > 1 and img_h > 1:
                    mask_bbox = {
                        "x1": bbox[0] / img_w,
                        "y1": bbox[1] / img_h,
                        "x2": (bbox[0] + bbox[2]) / img_w,
                        "y2": (bbox[1] + bbox[3]) / img_h,
                    }

        conversations = build_region_prompt(caption, mask_bbox)

        llava_samples.append({
            "image": image_file,
            "conversations": conversations,
        })

    return llava_samples


def main():
    parser = argparse.ArgumentParser(description="Prepare URECA data for DAT training")
    parser.add_argument("--sa1b_dir", type=str, required=True,
                        help="Directory containing SA-1B images")
    parser.add_argument("--output_json", type=str, required=True,
                        help="Output path for LLaVA-format JSON")
    parser.add_argument("--cache_dir", type=str,
                        default="/root/autodl-tmp/cache/hf",
                        help="HuggingFace cache directory")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max number of samples to include (default: all)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    caption_path = download_ureca_captions(args.cache_dir)
    with open(caption_path, "r") as f:
        ureca_data = json.load(f)

    print(f"Loaded {len(ureca_data)} URECA samples")

    llava_samples = convert_ureca_to_llava(ureca_data, args.sa1b_dir)
    print(f"Converted {len(llava_samples)} samples to LLaVA format")

    # Filter to only include samples whose images exist on disk
    if os.path.isdir(args.sa1b_dir):
        existing = []
        for s in llava_samples:
            img_path = os.path.join(args.sa1b_dir, s["image"])
            if os.path.exists(img_path):
                existing.append(s)
        print(f"Found {len(existing)}/{len(llava_samples)} images on disk")
        llava_samples = existing

    if args.max_samples and len(llava_samples) > args.max_samples:
        random.shuffle(llava_samples)
        llava_samples = llava_samples[:args.max_samples]
        print(f"Subsampled to {len(llava_samples)}")

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(llava_samples, f, indent=2)

    print(f"Saved {len(llava_samples)} samples to {args.output_json}")

    # Print image requirements if sa1b_dir doesn't exist yet
    if not os.path.isdir(args.sa1b_dir):
        image_ids = set()
        for s in llava_samples:
            image_ids.add(s["image"])
        print(f"\nNeed {len(image_ids)} unique images from SA-1B")
        print(f"Download SA-1B and place images in: {args.sa1b_dir}")


if __name__ == "__main__":
    main()
