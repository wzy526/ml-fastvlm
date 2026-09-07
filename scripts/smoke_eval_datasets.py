#!/usr/bin/env python3
"""Offline smoke load for a few lmms-eval HF snapshots under $HF_HOME."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/Users/xiazhuofan.1/eval_hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_FORK = Path("/Users/xiazhuofan.1/Desktop/lmms-eval-main")
if _FORK.is_dir():
    sys.path.insert(0, str(_FORK))

from lmms_eval.offline_hub import load_dataset_offline, resolve_dataset_path  # noqa: E402

SPECS = [
    ("lmms-lab/ChartQA", None, "test"),
    ("lmms-lab/RefCOCO", None, "val"),
    ("lmms-lab/GQA", "testdev_balanced_instructions", "testdev"),
    ("lmms-lab/GQA", "testdev_balanced_images", "testdev"),
    ("BLINK-Benchmark/BLINK", "Art_Style", "val"),
    ("lmms-lab/OK-VQA", None, None),
]


def main():
    print("HF_HOME", os.environ.get("HF_HOME"))
    failed = []
    for repo, name, split in SPECS:
        label = f"{repo}" + (f":{name}" if name else "") + (f"[{split}]" if split else "")
        try:
            resolved = resolve_dataset_path(repo)
            print(f"\n-- {label}")
            print(f"   resolved={resolved}")
            kwargs = {}
            if split:
                kwargs["split"] = split
            ds = load_dataset_offline(repo, name, **kwargs)
            n = len(ds) if hasattr(ds, "__len__") else {k: len(v) for k, v in ds.items()}
            cols = list(ds.features) if hasattr(ds, "features") else list(next(iter(ds.values())).features)
            print(f"   OK n={n} cols={cols[:8]}")
        except Exception as e:
            print(f"   FAIL {type(e).__name__}: {e}")
            failed.append(label)
    print("\n==== summary ====")
    if failed:
        print("FAILED", failed)
        sys.exit(1)
    print("all ok")


if __name__ == "__main__":
    main()
