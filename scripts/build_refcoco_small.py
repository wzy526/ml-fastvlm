#!/usr/bin/env python3
"""Filter RefCOCO / + / g val into a small-object subset (relative bbox area).

RefCOCO referring boxes are almost never COCO-small (<32^2 px) or <1% of the
image; the small tail starts around 5–8% of image area. Default threshold is
8%. Writes DatasetDict folders under
$HF_HOME/extra/refcoco_small/{refcoco,refcocoplus,refcocog} plus stats.json.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("HF_HOME", "/Users/xiazhuofan.1/eval_hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

_FORK = Path("/Users/xiazhuofan.1/Desktop/lmms-eval-main")
if _FORK.is_dir():
    sys.path.insert(0, str(_FORK))

from datasets import DatasetDict  # noqa: E402

from lmms_eval.offline_hub import load_dataset_offline  # noqa: E402

# 8% of image area ≈ RefCOCO val p10–p20; 1% is empty on this split.
REL_THRESH = float(os.environ.get("REFCOCO_SMALL_REL", "0.08"))
COCO_SMALL_PX = 32 * 32
OUT_ROOT = Path(os.environ.get("HF_HOME", ".")) / "extra" / "refcoco_small"

SPECS = [
    ("refcoco", "lmms-lab/RefCOCO", None),
    ("refcocoplus", "lmms-lab/RefCOCOplus", None),
    ("refcocog", "lmms-lab/RefCOCOg", None),
]


def _rel_area(ex) -> float:
    im = ex["image"]
    w, h = im.size
    bb = ex["bbox"]  # xywh in pixels
    if w <= 0 or h <= 0:
        return 1.0
    return float(bb[2] * bb[3]) / float(w * h)


def _abs_area(ex) -> float:
    bb = ex["bbox"]
    return float(bb[2] * bb[3])


def _histogram(values, edges):
    counts = Counter()
    for v in values:
        placed = False
        for e in edges:
            if v < e:
                counts[f"<{e}"] += 1
                placed = True
                break
        if not placed:
            counts[f">={edges[-1]}"] += 1
    return dict(counts)


def _pct(sorted_vals, p):
    if not sorted_vals:
        return None
    k = min(len(sorted_vals) - 1, max(0, int(round((p / 100.0) * (len(sorted_vals) - 1)))))
    return sorted_vals[k]


def process_one(key: str, repo: str, config_name):
    print(f"\n==== {key}  {repo} val ====", flush=True)
    ds = load_dataset_offline(repo, config_name, split="val")
    n = len(ds)
    rels = []
    abs_areas = []
    for i, ex in enumerate(ds):
        rels.append(_rel_area(ex))
        abs_areas.append(_abs_area(ex))
        if (i + 1) % 2000 == 0:
            print(f"  scanned {i+1}/{n}", flush=True)

    keep_rel = [i for i, r in enumerate(rels) if r < REL_THRESH]
    keep_coco = [i for i, a in enumerate(abs_areas) if a < COCO_SMALL_PX]
    rel_sorted = sorted(rels)
    stats = {
        "repo": repo,
        "n_val": n,
        "rel_thresh": REL_THRESH,
        "n_rel_small": len(keep_rel),
        "n_coco_small_32": len(keep_coco),
        "rel_min": rel_sorted[0] if rel_sorted else None,
        "rel_p10": _pct(rel_sorted, 10),
        "rel_p25": _pct(rel_sorted, 25),
        "rel_p50": _pct(rel_sorted, 50),
        "rel_hist": _histogram(rels, [0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.4]),
        "frac_rel_small": len(keep_rel) / n if n else 0.0,
    }
    print(json.dumps(stats, indent=2), flush=True)

    if not keep_rel:
        print(f"SKIP save: 0 rows under rel<{REL_THRESH}", flush=True)
        return stats

    small = ds.select(keep_rel)
    out = OUT_ROOT / key
    out.mkdir(parents=True, exist_ok=True)
    DatasetDict({"val": small}).save_to_disk(str(out))
    print(f"saved {len(small)} rows -> {out}", flush=True)
    return stats


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_stats = {}
    for key, repo, cfg in SPECS:
        all_stats[key] = process_one(key, repo, cfg)
    stats_path = OUT_ROOT / "stats.json"
    stats_path.write_text(json.dumps(all_stats, indent=2), encoding="utf-8")
    print(f"\nWrote {stats_path}", flush=True)


if __name__ == "__main__":
    main()
