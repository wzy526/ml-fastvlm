#!/usr/bin/env bash
# Copy what ablation 5 needs from machine 1's home to a fresh machine 2, through OSS.
#   machine 1:  bash scripts/ablation/stage_machine2.sh pack     (~20-40 min)
#   machine 2:  bash scripts/ablation/stage_machine2.sh unpack   (then it checks the env)
# Both machines run as the same user with the same home, so the conda env and the
# absolute image paths in the box jsons stay valid. Only the images the three box
# jsons point to are copied; files already on OSS stay where they are, and a
# symlinked directory on an image path is copied as the link.
set -euo pipefail
S=${S:-/data/oss_bucket_0/wangziyi/machine2_stage}
OSS_DATA=${OSS_DATA:-/data/oss_bucket_0/wangziyi/models_data}
E=vldat_experiments/0924_miniE1_2k
cd "$HOME"

pack() {
    mkdir -p "$S"
    local hf
    hf=$(find .cache/huggingface -maxdepth 2 -iname "*hr*bench*" 2>/dev/null || true)
    [ -n "$hf" ] && echo "HR-Bench cache: $hf" || echo "no HR-Bench cache under ~/.cache/huggingface; unpack downloads it"
    local cfg=()
    [ -f .condarc ] && cfg=(.condarc)

    ~/miniconda3/bin/python - "$OSS_DATA" "$S/images.txt" <<'PY'
import json, os, sys
home = os.path.expanduser("~")
root = os.path.join(home, "sft_data/train_split")
jsons = [f"{sys.argv[1]}/extra_0916/viscot_bbox.train.json", f"{sys.argv[1]}/extra_1018/refcoco_bbox.json",
         f"{sys.argv[1]}/extra_1018/coco_small_bbox.json"]

def entry(p):
    rel = os.path.relpath(p, home)
    if rel.startswith(".."):
        return None
    cur = home
    for part in rel.split(os.sep):
        cur = os.path.join(cur, part)
        if os.path.islink(cur):
            return os.path.relpath(cur, home)
    return rel

keep, shared, missing = set(), 0, 0
for j in jsons:
    for s in json.load(open(j)):
        im = s.get("image")
        im = im[0] if isinstance(im, list) else im
        if not im:
            continue
        p = im if os.path.isabs(im) else os.path.join(root, im)
        if not os.path.exists(p):
            missing += 1
            continue
        e = entry(p)
        if e is None:
            shared += 1
        else:
            keep.add(e)
open(sys.argv[2], "w").write("".join(f"{k}\n" for k in sorted(keep)))
print(f"images: {len(keep)} in home to copy, {shared} already outside home, {missing} missing on machine 1")
PY

    echo "$(date '+%T') conda"
    tar -cf "$S/conda.tar.part" --exclude=miniconda3/pkgs --exclude=miniconda3/envs \
        miniconda3 .conda/envs/fastvlm "${cfg[@]}"
    mv "$S/conda.tar.part" "$S/conda.tar"
    echo "$(date '+%T') model, V*, lmms-eval-oldvstar, HR-Bench cache"
    # shellcheck disable=SC2086
    tar -cf "$S/data.tar.part" --exclude="$E/checkpoint-1000/global_step*" \
        --exclude="$E/checkpoint-1000/optimizer*" --exclude=lmms-eval-oldvstar/.git \
        "$E/checkpoint-1000" "$E/loc/where_lm8_ce20.pt" data/vstar_bench lmms-eval-oldvstar $hf
    mv "$S/data.tar.part" "$S/data.tar"
    echo "$(date '+%T') images"
    tar -cf "$S/images.tar.part" -T "$S/images.txt"
    mv "$S/images.tar.part" "$S/images.tar"
    ls -lh "$S"
    echo "$(date '+%T') pack done"
}

unpack() {
    local t
    for t in conda data images; do
        [ -f "$S/$t.tar" ] || { echo "missing $S/$t.tar (pack not finished?)"; exit 1; }
    done
    for t in conda data images; do
        echo "$(date '+%T') $t"
        tar -xf "$S/$t.tar"
    done
    # shellcheck disable=SC1091
    source ~/miniconda3/bin/activate fastvlm
    python - <<'PY'
import importlib, os
import torch, transformers
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "gpus", torch.cuda.device_count())
print("transformers", transformers.__version__)
for m in ("fla", "flash_attn", "lmms_eval", "llava"):
    try:
        print("ok", m, os.path.dirname(importlib.import_module(m).__file__))
    except Exception as e:
        print("FAILED", m, repr(e)[:200])
PY
    if ! HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python -c \
        "from datasets import load_dataset; load_dataset('DreamMr/HR-Bench', 'hrbench_version_split', split='hrbench_4k')" \
        > /dev/null 2>&1; then
        echo "HR-Bench not cached, downloading once"
        HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com} python -c \
            "from datasets import load_dataset; [load_dataset('DreamMr/HR-Bench', 'hrbench_version_split', split=s) for s in ('hrbench_4k', 'hrbench_8k')]"
    fi
    echo "$(date '+%T') unpack done"
}

case "${1:-}" in
    pack) pack ;;
    unpack) unpack ;;
    *) echo "usage: $0 pack|unpack"; exit 1 ;;
esac
