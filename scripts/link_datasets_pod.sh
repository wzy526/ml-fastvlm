#!/usr/bin/env bash
# Assemble the ONE canonical training image root via symlinks (run on the POD).
# Canonical root: /home/ea-cvfa-aigc-x2v-2/xzf/sft_data/train_split
# Physical data stays where it is:
#   gqa / ocr_vqa / vg           real dirs already inside train_split (untouched)
#   sa1b/sa_000001..11        -> LVLM collection (ea-cv-nlp-train-offline-2)
#   sa1b/sa_000000            -> xzf/sa1b_images/sa_000000 (staged earlier)
#   sa1b/sa_000012..44        -> extracted from Mac-uploaded tars on offline-2
#   coco/train2017            -> chubaofs-volume003/ocr/base/coco/train2017
#   docvqa/infovqa/synthdog   -> Mac-uploaded regen images on offline-2
# Idempotent: rerun any time (e.g. as more tars finish uploading).
# Run:  bash /home/ea-cvfa-aigc-x2v-2/xzf/link_datasets_pod.sh
#   DELETE_TAR=1  to remove each tar after verified extraction (default: keep)
set -uo pipefail

XZF=/home/ea-cvfa-aigc-x2v-2/xzf
TS=$XZF/sft_data/train_split
LVLM_RAW="/home/ea-cv-nlp-train-offline-2/multi_modal_datasets/LVLM开源数据集整理/InternVL-SA-1B-Caption/raw"
COCO_SRC=/home/chubaofs-volume003/ocr/base/coco/train2017
OFF2=/home/ea-cv-nlp-train-offline-2/xzf/sft_data     # Mac uploads land here
DELETE_TAR=${DELETE_TAR:-0}

link() {  # $1 = target, $2 = linkpath
    if [[ -L "$2" ]]; then
        [[ "$(readlink "$2")" == "$1" ]] && { echo "[ok]   $2 (already)"; return; }
        rm -f "$2"
    elif [[ -e "$2" ]]; then
        echo "[SKIP] $2 exists and is not a symlink (real data? not touching)"; return
    fi
    ln -s "$1" "$2" && echo "[link] $2 -> $1" || echo "[FAIL] $2"
}

mkdir -p "$TS/sa1b" "$TS/coco"

echo "=== 1) sa1b shards 1-11 from the LVLM collection ==="
for i in $(seq 1 11); do
    s=$(printf "sa_%06d" "$i")
    [[ -d "$LVLM_RAW/$s" ]] && link "$LVLM_RAW/$s" "$TS/sa1b/$s" || echo "[MISS] $LVLM_RAW/$s"
done

echo "=== 2) shards physically staged in xzf/sa1b_images (sa_000000) ==="
for d in "$XZF"/sa1b_images/sa_*/; do
    [[ -d "$d" ]] || continue
    link "${d%/}" "$TS/sa1b/$(basename "$d")"
done

echo "=== 3) coco train2017 ==="
link "$COCO_SRC" "$TS/coco/train2017"

echo "=== 4) docvqa / infovqa / synthdog (Mac-regenerated, on offline-2) ==="
for d in docvqa infovqa synthdog; do
    [[ -d "$OFF2/train_split/$d" ]] && link "$OFF2/train_split/$d" "$TS/$d" \
        || echo "[MISS] $OFF2/train_split/$d (upload not finished?)"
done

echo "=== 5) extract uploaded sa1b tars (12-44) and link shard dirs ==="
# preferred: extract next to the tars on offline-2; fall back to our own
# volume if that FUSE mount turns out to be read-only for the pod
EXTRACT=$OFF2/sa1b_extracted
if ! mkdir -p "$EXTRACT" 2>/dev/null || ! touch "$EXTRACT/.write_test" 2>/dev/null; then
    echo "[note] $EXTRACT not writable, falling back to $XZF/sa1b_images"
    EXTRACT=$XZF/sa1b_images
    mkdir -p "$EXTRACT"
else
    rm -f "$EXTRACT/.write_test"
fi
echo "[note] extract target: $EXTRACT"
shopt -s nullglob
for t in "$OFF2"/sa1b_tars/sa_*.tar; do
    s=$(basename "$t" .tar)
    dst=$EXTRACT/$s
    if [[ ! -f "$dst/.extract_done" ]]; then
        # skip tars still being uploaded: size must be stable for 30s
        sz1=$(stat -c%s "$t"); sleep 30; sz2=$(stat -c%s "$t")
        if [[ "$sz1" != "$sz2" ]]; then echo "[WAIT] $s.tar still growing, skip this run"; continue; fi
        echo "[extract] $s.tar ($((sz1/1024/1024/1024))GB, jpg only) ..."
        mkdir -p "$dst" || { echo "[FAIL] cannot mkdir $dst"; continue; }
        if tar -xf "$t" -C "$dst" --wildcards '*.jpg'; then
            n=$(ls "$dst" | wc -l)
            if [[ "$n" -gt 9000 ]]; then
                echo "$n" > "$dst/.extract_done"
                echo "[done] $s: $n jpgs"
                [[ "$DELETE_TAR" == "1" ]] && { rm -f "$t"; echo "[rm]   $s.tar"; }
            else
                echo "[FAIL] $s: only $n jpgs extracted, leaving for retry"
            fi
        else
            echo "[FAIL] tar extract $s (partial upload? retry next run)"
        fi
    fi
    [[ -f "$dst/.extract_done" ]] && link "$dst" "$TS/sa1b/$s"
done

echo ""
echo "=== verification through the links ==="
for p in "$TS/sa1b/sa_000001" "$TS/sa1b/sa_000000" "$TS/coco/train2017" \
         "$TS/docvqa" "$TS/infovqa" "$TS/synthdog"; do
    n=$(timeout 60 ls "$p" 2>/dev/null | wc -l | tr -d " ")
    echo "  $p : ${n} entries (first: $(ls "$p" 2>/dev/null | head -1))"
done
echo ""
echo "shard dirs now in train_split/sa1b: $(ls "$TS/sa1b" | wc -l)"
echo "LINK_TASKS_DONE"
