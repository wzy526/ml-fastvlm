#!/usr/bin/env bash
# Copy SA-1B shards sa_000001..sa_000011 (jpg only) from the LVLM collection on
# ea-cv-nlp-train-offline-2 into our xzf/sa1b_images. sa_000000 is already there.
# Run on the POD:
#   nohup bash /home/ea-cvfa-aigc-x2v-2/xzf/copy_sa1b_from_lvlm.sh \
#       > /home/ea-cvfa-aigc-x2v-2/xzf/scan_logs/copy_sa1b.out 2>&1 &
set -uo pipefail

SRC="${SRC:-/home/ea-cv-nlp-train-offline-2/multi_modal_datasets/LVLM开源数据集整理/InternVL-SA-1B-Caption/raw}"
DST="${DST:-/home/ea-cvfa-aigc-x2v-2/xzf/sa1b_images}"
SHARDS="${SHARDS:-$(seq 1 11)}"
PAR="${PAR:-8}"

for n in $SHARDS; do
    shard=$(printf "sa_%06d" "$n")
    src="$SRC/$shard"; dst="$DST/$shard"
    [[ -d "$src" ]] || { echo "[SKIP] $shard: source missing"; continue; }
    nsrc=$(ls "$src" | grep -c '\.jpg$')
    ndst=$(ls "$dst" 2>/dev/null | grep -c '\.jpg$' || true)
    if [[ "$ndst" == "$nsrc" && "$nsrc" -gt 0 ]]; then
        echo "[$(date +%H:%M:%S)] $shard already complete ($ndst jpgs), skip"; continue
    fi
    echo "[$(date +%H:%M:%S)] $shard: copying $nsrc jpgs (have $ndst)"
    mkdir -p "$dst"
    ls "$src" | grep '\.jpg$' | xargs -P "$PAR" -I{} cp -n "$src/{}" "$dst/" 2>/dev/null
    ndst=$(ls "$dst" | grep -c '\.jpg$')
    if [[ "$ndst" == "$nsrc" ]]; then
        echo "[$(date +%H:%M:%S)] $shard DONE ($ndst jpgs)"
    else
        echo "[$(date +%H:%M:%S)] $shard COUNT_MISMATCH src=$nsrc dst=$ndst (rerun to resume)"
    fi
done
echo "COPY_ALL_DONE"
