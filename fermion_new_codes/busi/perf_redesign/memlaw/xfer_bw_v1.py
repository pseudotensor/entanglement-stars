#!/usr/bin/env python
"""xfer_bw_v1.py (busi, inbox #384): the host <-> device copy rate of the channel-streamed cache's unit (one 2n x 2n complex128 U,
4.29 GB at 8192 sites), pinned and pageable, synchronous and on a side stream under a concurrent FP64 matmul (the overlap the
streamed build relies on). Torch only; for the 2-record card probe (seconds; about 30 GB of pinned host memory touched at 8192).
    python xfer_bw_v1.py [--nsites 8192] [--gpu 0] [--reps 3]"""
import argparse
import time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--nsites", type=int, default=8192); ap.add_argument("--gpu", type=int, default=0); ap.add_argument("--reps", type=int, default=3)
a = ap.parse_args()
dev = torch.device("cuda:%d" % a.gpu); torch.cuda.set_device(dev)
n2 = 2 * a.nsites
U = torch.randn(n2, n2, dtype=torch.complex128, device=dev)
nb = U.numel() * U.element_size()
hp = torch.empty(U.shape, dtype=U.dtype, pin_memory=True)
hq = torch.empty(U.shape, dtype=U.dtype)
out = {}


def tmin(f):
    ts = []
    for _ in range(a.reps):
        torch.cuda.synchronize(dev); t_ = time.time(); f(); torch.cuda.synchronize(dev); ts.append(time.time() - t_)
    return min(ts)


out["d2h_pinned"] = tmin(lambda: hp.copy_(U))
D = torch.empty_like(U)
out["h2d_pinned"] = tmin(lambda: D.copy_(hp, non_blocking=True))
out["d2h_pageable"] = tmin(lambda: hq.copy_(U))
out["h2d_pageable"] = tmin(lambda: D.copy_(hq))
# overlap: an H2D on a side stream while the default stream runs a 2n x n x n complex matmul (the J.v's product class)
A = U[:, : n2 // 2].contiguous(); B = torch.randn(n2 // 2, n2 // 2, dtype=torch.complex128, device=dev)
mm = tmin(lambda: A @ B)
s = torch.cuda.Stream(device=dev)


def both():
    with torch.cuda.stream(s):
        D.copy_(hp, non_blocking=True)
    C = A @ B
    s.synchronize()
    return C
ov = tmin(both)
print("GPU %s, unit %.3f GB" % (torch.cuda.get_device_name(dev), nb / 1e9))
for k_, v_ in out.items():
    print("  %-13s %.3f s  %.1f GB/s" % (k_, v_, nb / 1e9 / v_))
print("  matmul 2n x n x n alone %.3f s; with a concurrent pinned H2D of the unit %.3f s (serial sum %.3f s)" % (mm, ov, mm + out["h2d_pinned"]))
