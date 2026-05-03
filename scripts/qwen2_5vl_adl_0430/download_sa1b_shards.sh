#!/usr/bin/env bash
set -euo pipefail

# Download SA-1B shards 0-9 (10 shards, ~110 GB)
# Each shard contains ~11K images at 1500px short side (3-6M px)
#
# Prerequisites: Get download links from https://ai.meta.com/datasets/segment-anything-downloads/
#   Save the links file as $LINKS_FILE (one "filename url" per line)
#
# Output: $SA1B_DIR/sa_000000/ ... sa_000009/ (images only, no masks)

SA1B_DIR="${SA1B_DIR:-/root/autodl-tmp/models_data/sa1b_images}"
LINKS_FILE="${LINKS_FILE:-/root/autodl-tmp/models_data/sa1b_links.txt}"
TMP_DIR="${TMP_DIR:-/root/autodl-tmp/sa1b_tmp}"
NUM_SHARDS="${NUM_SHARDS:-10}"

if [[ ! -f "$LINKS_FILE" ]]; then
    echo "[ERROR] SA-1B links file not found: $LINKS_FILE"
    echo "  1. Go to https://ai.meta.com/datasets/segment-anything-downloads/"
    echo "  2. Download the links file"
    echo "  3. Save as $LINKS_FILE"
    exit 1
fi

mkdir -p "$SA1B_DIR" "$TMP_DIR"

echo "=== Downloading SA-1B shards 0-$((NUM_SHARDS-1)) ==="
echo "  Links file: $LINKS_FILE"
echo "  Output dir: $SA1B_DIR"
echo "  Temp dir:   $TMP_DIR"
echo ""

# Extract lines for sa_000000.tar through sa_000009.tar
for i in $(seq 0 $((NUM_SHARDS - 1))); do
    shard_name=$(printf "sa_%06d" $i)
    tar_name="${shard_name}.tar"

    # Skip if already extracted
    if [[ -d "$SA1B_DIR/$shard_name" ]] && [[ $(ls "$SA1B_DIR/$shard_name"/*.jpg 2>/dev/null | head -1) ]]; then
        echo "[$shard_name] Already exists, skipping"
        continue
    fi

    # Find the download URL for this shard
    url=$(grep -E "^${tar_name}\s" "$LINKS_FILE" | awk '{print $2}' || true)
    if [[ -z "$url" ]]; then
        # Try alternate format: just the URL containing the shard name
        url=$(grep "$tar_name" "$LINKS_FILE" | head -1 | awk '{print $NF}')
    fi

    if [[ -z "$url" ]]; then
        echo "[$shard_name] WARNING: No URL found in links file, skipping"
        continue
    fi

    echo "[$shard_name] Downloading..."
    tar_path="$TMP_DIR/$tar_name"

    # Download with resume support
    if command -v aria2c &>/dev/null; then
        aria2c -x4 -c -o "$tar_path" "$url"
    else
        wget -c -O "$tar_path" "$url"
    fi

    echo "[$shard_name] Extracting images (skipping .json mask files)..."
    mkdir -p "$SA1B_DIR/$shard_name"
    # Only extract .jpg files, skip the large .json mask annotations
    tar xf "$tar_path" --wildcards '*.jpg' -C "$SA1B_DIR/" 2>/dev/null || \
    tar xf "$tar_path" -C "$SA1B_DIR/" --exclude='*.json' 2>/dev/null || \
    tar xf "$tar_path" -C "$SA1B_DIR/"

    # Clean up tar to save space
    rm -f "$tar_path"
    echo "[$shard_name] Done ($(ls "$SA1B_DIR/$shard_name"/*.jpg 2>/dev/null | wc -l) images)"
    echo ""
done

echo "=== Download complete ==="
echo "Images at: $SA1B_DIR"
du -sh "$SA1B_DIR"/sa_* 2>/dev/null
echo ""
total_images=$(find "$SA1B_DIR" -name "*.jpg" | wc -l)
echo "Total images: $total_images"
du -sh "$SA1B_DIR"
