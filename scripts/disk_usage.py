#!/usr/bin/env python3
"""List what fills the home quota. Read-only: prints one table, deletes nothing.

Home is quota-capped (44G) while the OSS mount is not, so each row is tagged
HOME or OSS. Sizes come from `du`, which on the JuiceFS mount walks metadata
and can take a few minutes.

    python scripts/disk_usage.py --top 40
"""
import argparse
import os
import subprocess


def du(path, t=180):
    try:
        out = subprocess.run(["du", "-sk", path], capture_output=True, text=True, timeout=t).stdout
    except subprocess.TimeoutExpired:
        return None
    return int(out.split()[0]) * 1024 if out.strip() else None


def gb(n):
    return f"{n / 1024 ** 3:7.2f}G" if n is not None else "      ? "


def children(path):
    try:
        return sorted(os.path.join(path, e) for e in os.listdir(path))
    except OSError:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--min_mb", type=float, default=100, help="only list entries bigger than this")
    args = ap.parse_args()
    home = os.path.expanduser("~")
    min_b = args.min_mb * 1024 ** 2

    # (label, path, recurse into it one more level)
    roots = [
        ("~", home, True),
        ("~/.cache", os.path.join(home, ".cache"), True),
        ("~/vldat_experiments", os.path.join(home, "vldat_experiments"), True),
        ("/data/oss_bucket_0/wangziyi", "/data/oss_bucket_0/wangziyi", False),
    ]
    rows = []
    for label, path, recurse in roots:
        if not os.path.isdir(path):
            print(f"(skip, not found: {path})")
            continue
        rows.append((du(path), label, path))
        if not recurse:
            continue
        for c in children(path):
            if os.path.islink(c):
                continue
            n = du(c)
            if n is not None and n >= min_b:
                rows.append((n, "   " + os.path.basename(c), c))

    print(f"\n{'size':>8}  {'where':<6}  path")
    for n, label, path in sorted((r for r in rows), key=lambda r: -(r[0] or 0))[: args.top]:
        where = "OSS" if path.startswith("/data/oss_bucket_0") else "HOME"
        print(f"{gb(n):>8}  {where:<6}  {label}   {path}")


if __name__ == "__main__":
    main()
