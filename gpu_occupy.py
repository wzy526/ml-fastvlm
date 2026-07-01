#!/usr/bin/env python3
"""占用集群 GPU 的小脚本。

默认给每张可见的 GPU 起一个进程，各自分配一大块显存并持续做矩阵乘法，
从而同时占住显存和算力。用 Ctrl+C 即可全部退出。

用法示例:
    # 占用全部 8 张卡（假设机器有 8 张）
    python gpu_occupy.py --gpus 8

    # 只占用 0,1,2,3 号卡，每张卡吃约 20GB 显存
    python gpu_occupy.py --gpu-ids 0,1,2,3 --mem-gb 20

    # 只占显存、几乎不用算力（占卡但不烧电）
    python gpu_occupy.py --gpus 8 --idle
"""
import argparse
import os
import signal
import time

import torch
import torch.multiprocessing as mp


def parse_args():
    p = argparse.ArgumentParser(description="占用 GPU 显存与算力的小工具")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--gpus", type=int, default=None,
                   help="占用前 N 张卡；默认占用所有可见卡")
    g.add_argument("--gpu-ids", type=str, default=None,
                   help="指定卡号，逗号分隔，如 0,1,2,3")
    p.add_argument("--mem-gb", type=float, default=None,
                   help="每张卡占用的显存(GB)；不填则尽量吃满可用显存的 90%%")
    p.add_argument("--matrix", type=int, default=8192,
                   help="计算用的方阵边长，越大算力占用越高")
    p.add_argument("--idle", action="store_true",
                   help="只占显存，几乎不做计算(不烧算力)")
    p.add_argument("--interval", type=float, default=0.0,
                   help="每轮计算之间的休眠秒数，越大算力占用越低")
    return p.parse_args()


def worker(gpu_id: int, mem_gb, matrix: int, idle: bool, interval: float):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    name = torch.cuda.get_device_name(gpu_id)
    total = torch.cuda.get_device_properties(gpu_id).total_memory

    # 计算需要 hold 住的显存字节数
    if mem_gb is not None:
        target_bytes = int(mem_gb * 1024 ** 3)
    else:
        free, _ = torch.cuda.mem_get_info(gpu_id)
        target_bytes = int(free * 0.9)

    # 预留一部分给计算用的临时张量
    reserve_for_compute = 3 * matrix * matrix * 4  # 三个 float32 方阵
    hold_bytes = max(target_bytes - reserve_for_compute, 0)

    # 用一大块 float32 张量把显存占住
    n_float = hold_bytes // 4
    blobs = []
    if n_float > 0:
        # 分块申请，避免单次超大申请失败
        chunk = 256 * 1024 * 1024  # 每块 ~1GB (256M float32)
        remaining = n_float
        while remaining > 0:
            cur = min(chunk, remaining)
            blobs.append(torch.empty(int(cur), dtype=torch.float32, device=device))
            remaining -= cur

    held_gb = sum(b.numel() for b in blobs) * 4 / 1024 ** 3
    print(f"[GPU {gpu_id}] {name} | 总显存 {total/1024**3:.1f}GB | 已占用约 {held_gb:.1f}GB",
          flush=True)

    if idle:
        # 只占显存，睡觉即可
        while True:
            time.sleep(3600)

    # 持续做矩阵乘法占用算力
    a = torch.randn(matrix, matrix, device=device)
    b = torch.randn(matrix, matrix, device=device)
    while True:
        c = a @ b
        a = c / (c.abs().max() + 1e-6)  # 归一化防止数值爆炸
        torch.cuda.synchronize(gpu_id)
        if interval > 0:
            time.sleep(interval)


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("未检测到可用的 CUDA GPU")

    n_visible = torch.cuda.device_count()

    if args.gpu_ids:
        gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip() != ""]
    elif args.gpus:
        gpu_ids = list(range(min(args.gpus, n_visible)))
    else:
        gpu_ids = list(range(n_visible))

    print(f"可见 GPU 数量: {n_visible} | 本次占用: {gpu_ids}", flush=True)

    mp.set_start_method("spawn", force=True)
    procs = []
    for gid in gpu_ids:
        p = mp.Process(target=worker,
                       args=(gid, args.mem_gb, args.matrix, args.idle, args.interval))
        p.start()
        procs.append(p)

    def shutdown(signum, frame):
        print("\n收到退出信号，正在终止所有进程...", flush=True)
        for p in procs:
            p.terminate()
        for p in procs:
            p.join()
        os._exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    for p in procs:
        p.join()


if __name__ == "__main__":
    main()
