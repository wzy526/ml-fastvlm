#!/usr/bin/env bash
set -euo pipefail

# Download SA-1B shards. Default: 0..9 (10 shards, ~110 GB).
# Each shard ~11k images at 1500px short side (3-6M px), ~11 GB on disk.
#
# Range selection (any of the two styles work):
#   START_SHARD / END_SHARD  (inclusive)
#       e.g. START_SHARD=10 END_SHARD=44  ->  sa_000010 .. sa_000044
#   NUM_SHARDS               (legacy, downloads 0..NUM_SHARDS-1)
#
# Source channel (auto-detected; explicit override via SOURCE=meta|hf):
#   - meta (default if $LINKS_FILE exists):
#       Use Meta presigned URLs from $LINKS_FILE (one "<filename> <url>"
#       per line, obtained at https://ai.meta.com/datasets/segment-anything-downloads/).
#   - hf:
#       Pull from the sailvideo/SA-1B mirror through hf-mirror.com. No token
#       needed for public mirrors, but the repo's availability has been
#       intermittent (see channel discussion in chat). Falls back to Meta
#       if HEAD on the mirror URL returns non-2xx.
#
# Output layout: $SA1B_DIR/sa_NNNNNN/sa_M.jpg (images only, .json masks dropped)

SA1B_DIR="${SA1B_DIR:-/root/autodl-tmp/models_data/sa1b_images}"
LINKS_FILE="${LINKS_FILE:-/root/autodl-tmp/models_data/sa1b_links.txt}"
TMP_DIR="${TMP_DIR:-/root/autodl-tmp/sa1b_tmp}"
SOURCE="${SOURCE:-auto}"
HF_BASE_URL="${HF_BASE_URL:-https://hf-mirror.com/datasets/sailvideo/SA-1B/resolve/main}"

if [[ -n "${START_SHARD:-}" ]] || [[ -n "${END_SHARD:-}" ]]; then
    START_SHARD="${START_SHARD:-0}"
    END_SHARD="${END_SHARD:?Set END_SHARD when START_SHARD is set}"
else
    NUM_SHARDS="${NUM_SHARDS:-10}"
    START_SHARD=0
    END_SHARD=$((NUM_SHARDS - 1))
fi

case "$SOURCE" in
    meta|hf|auto) ;;
    *) echo "[ERROR] SOURCE must be one of: meta, hf, auto (got '$SOURCE')"; exit 1 ;;
esac
if [[ "$SOURCE" == "auto" ]]; then
    if [[ -f "$LINKS_FILE" ]]; then
        SOURCE="meta"
    else
        SOURCE="hf"
    fi
fi
if [[ "$SOURCE" == "meta" && ! -f "$LINKS_FILE" ]]; then
    echo "[ERROR] SOURCE=meta but links file not found: $LINKS_FILE"
    echo "  1. Go to https://ai.meta.com/datasets/segment-anything-downloads/"
    echo "  2. Download the links file"
    echo "  3. Save as $LINKS_FILE"
    echo "  (or set SOURCE=hf to use the HF mirror instead.)"
    exit 1
fi

mkdir -p "$SA1B_DIR" "$TMP_DIR"

echo "=== Downloading SA-1B shards $(printf 'sa_%06d' $START_SHARD)..$(printf 'sa_%06d' $END_SHARD) ==="
echo "  Source channel: $SOURCE"
[[ "$SOURCE" == "meta" ]] && echo "  Links file:     $LINKS_FILE"
[[ "$SOURCE" == "hf"   ]] && echo "  HF base URL:    $HF_BASE_URL"
echo "  Output dir:     $SA1B_DIR"
echo "  Temp dir:       $TMP_DIR"
echo ""

for i in $(seq "$START_SHARD" "$END_SHARD"); do
    shard_name=$(printf "sa_%06d" $i)
    tar_name="${shard_name}.tar"

    # Skip if already extracted
    if [[ -d "$SA1B_DIR/$shard_name" ]] && [[ $(ls "$SA1B_DIR/$shard_name"/*.jpg 2>/dev/null | head -1) ]]; then
        echo "[$shard_name] Already exists, skipping"
        continue
    fi

    url=""
    if [[ "$SOURCE" == "meta" ]]; then
        url=$(grep -E "^${tar_name}\s" "$LINKS_FILE" | awk '{print $2}' || true)
        if [[ -z "$url" ]]; then
            url=$(grep "$tar_name" "$LINKS_FILE" | head -1 | awk '{print $NF}' || true)
        fi
    else
        url="${HF_BASE_URL}/${tar_name}?download=true"
        http_code=$(curl -sLI --max-time 30 -o /dev/null -w "%{http_code}" "$url" || echo "000")
        if [[ "$http_code" != "200" && "$http_code" != "302" && "$http_code" != "307" ]]; then
            if [[ -f "$LINKS_FILE" ]]; then
                echo "[$shard_name] HF mirror returned $http_code, falling back to Meta links."
                url=$(grep -E "^${tar_name}\s" "$LINKS_FILE" | awk '{print $2}' || true)
                if [[ -z "$url" ]]; then
                    url=$(grep "$tar_name" "$LINKS_FILE" | head -1 | awk '{print $NF}' || true)
                fi
            else
                echo "[$shard_name] HF mirror returned $http_code and no $LINKS_FILE for fallback, skipping"
                continue
            fi
        fi
    fi

    if [[ -z "$url" ]]; then
        echo "[$shard_name] WARNING: No URL found, skipping"
        continue
    fi

    echo "[$shard_name] Downloading..."
    tar_path="$TMP_DIR/$tar_name"

    if command -v aria2c &>/dev/null; then
        aria2c -x4 -s4 -c -d "$TMP_DIR" -o "$tar_name" "$url"
    else
        wget -c -O "$tar_path" "$url"
    fi

    echo "[$shard_name] Extracting images (skipping .json mask files)..."
    # CRITICAL: SA-1B tar files contain BARE jpg entries (no sa_NNNNNN/
    # prefix). Combined with -C path-ordering quirks in GNU tar, the
    # previous form silently dropped all jpgs into the script's cwd.
    # Force-cd into the shard subdir so jpgs land in $SA1B_DIR/$shard_name/
    # regardless of tar/-C parsing.
    mkdir -p "$SA1B_DIR/$shard_name"
    (
        cd "$SA1B_DIR/$shard_name" && \
        ( tar xf "$tar_path" --wildcards '*.jpg' 2>/dev/null || \
          tar xf "$tar_path" --exclude='*.json' 2>/dev/null || \
          tar xf "$tar_path" )
    )

    extracted=$(ls "$SA1B_DIR/$shard_name"/*.jpg 2>/dev/null | wc -l)
    if [[ "$extracted" -lt 100 ]]; then
        echo "[$shard_name] EXTRACTION FAILED ($extracted images). Tar kept at $tar_path for inspection." >&2
        continue
    fi

    rm -f "$tar_path"
    echo "[$shard_name] Done ($extracted images)"
    echo ""
done

echo "=== Download complete ==="
echo "Images at: $SA1B_DIR"
du -sh "$SA1B_DIR"/sa_* 2>/dev/null
echo ""
total_images=$(find "$SA1B_DIR" -name "*.jpg" | wc -l)
echo "Total images: $total_images"
du -sh "$SA1B_DIR"
