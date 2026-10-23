#!/usr/bin/env python
"""xfer_bw_v2.py (busi, MERGE-1; reviewer A ReviewA_v16e_v1 sec. 4 (a)): the host <-> device copy rates on the card, the GPU idle otherwise.
Torch only. For each unit size (default 0.25, 1.07 and 4.29 GB: a 2n x 2n complex128 U at 2048, 4096 and 8192 sites):
  pinned D2H and H2D, blocking and non_blocking (then synchronized), the minimum and the median of --reps;
  pageable D2H and H2D;
  device-to-device copy (the HBM reference);
  host memcpy of the pinned buffer into a second pinned buffer (the host memory bandwidth);
  overlap: a pinned H2D (and a pinned D2H) of the unit on a side stream while the default stream runs a dense complex128 eigh of
  2n x 2n at 4096 sites (the engine's dense eigh) and, separately, a 2n x n x n matmul; the wall of the pair against each alone.
Prints one line per measurement and a json summary (--out).
    python xfer_bw_v2.py [--sizes-gb 0.25,1.07,4.29] [--reps 5] [--gpu 0] [--out xfer_bw_v2.json]"""
import argparse
import json
import statistics
import time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--sizes-gb", default="0.25,1.07,4.29"); ap.add_argument("--reps", type=int, default=5); ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--out", default=None)
a = ap.parse_args()
dev = torch.device("cuda:%d" % a.gpu); torch.cuda.set_device(dev)
res = {"device": torch.cuda.get_device_name(dev), "torch": torch.__version__, "units": []}
print("device %s, torch %s" % (res["device"], res["torch"]), flush=True)


def timed(f):
    ts = []
    for _ in range(a.reps):
        torch.cuda.synchronize(dev); t_ = time.perf_counter(); f(); torch.cuda.synchronize(dev); ts.append(time.perf_counter() - t_)
    return min(ts), statistics.median(ts)


for gb in [float(x) for x in a.sizes_gb.split(",")]:
    n = int(gb * 1e9 / 16)
    U = torch.randn(n, dtype=torch.complex128, device=dev)
    nb = U.numel() * U.element_size()
    hp = torch.empty(U.shape, dtype=U.dtype, pin_memory=True); hp2 = torch.empty(U.shape, dtype=U.dtype, pin_memory=True)
    hq = torch.empty(U.shape, dtype=U.dtype)
    D = torch.empty_like(U)
    u = {"GB": nb / 1e9, "pinned_is_pinned": bool(hp.is_pinned())}
    for name, f in [("d2h_pinned_blocking", lambda: hp.copy_(U)), ("d2h_pinned_nonblocking", lambda: hp.copy_(U, non_blocking=True)),
                    ("h2d_pinned_blocking", lambda: D.copy_(hp)), ("h2d_pinned_nonblocking", lambda: D.copy_(hp, non_blocking=True)),
                    ("d2h_pageable", lambda: hq.copy_(U)), ("h2d_pageable", lambda: D.copy_(hq)),
                    ("d2d", lambda: D.copy_(U)), ("host_memcpy_pinned", lambda: hp2.copy_(hp))]:
        tmin, tmed = timed(f)
        u[name] = {"min_s": tmin, "median_s": tmed, "GBps_best": nb / 1e9 / tmin, "GBps_median": nb / 1e9 / tmed}
        print("  %.2f GB  %-24s best %6.1f GB/s  median %6.1f GB/s" % (nb / 1e9, name, nb / 1e9 / tmin, nb / 1e9 / tmed), flush=True)
    res["units"].append(u)
    del U, hp, hp2, hq, D
    torch.cuda.empty_cache()

# overlap under a running dense eigh (2n = 8192, complex128, the 4096-site engine's dense call) and a 2n x n x n matmul
n2 = 8192
H = torch.randn(n2, n2, dtype=torch.complex128, device=dev); H = H + H.conj().T
X = torch.randn(n2, n2 // 2, dtype=torch.complex128, device=dev); Y = torch.randn(n2 // 2, n2 // 2, dtype=torch.complex128, device=dev)
Uc = torch.randn(n2 * n2, dtype=torch.complex128, device=dev); hpc = torch.empty(Uc.shape, dtype=Uc.dtype, pin_memory=True); Dc = torch.empty_like(Uc)
s = torch.cuda.Stream(device=dev)
ov = {"unit_GB": Uc.numel() * 16 / 1e9}
for cname, comp in [("eigh", lambda: torch.linalg.eigh(H)), ("matmul", lambda: X @ Y)]:
    c_alone = timed(comp)[0]
    for dname, cp in [("h2d", lambda: Dc.copy_(hpc, non_blocking=True)), ("d2h", lambda: hpc.copy_(Uc, non_blocking=True))]:
        x_alone = timed(cp)[0]

        def pair():
            with torch.cuda.stream(s):
                cp()
            comp()
        both = timed(pair)[0]
        ov["%s+%s" % (cname, dname)] = {"compute_alone_s": c_alone, "copy_alone_s": x_alone, "both_s": both, "serial_sum_s": c_alone + x_alone}
        print("  overlap %-6s + %s: compute %.3f s, copy %.3f s, both %.3f s (serial %.3f s)" % (cname, dname, c_alone, x_alone, both, c_alone + x_alone), flush=True)
res["overlap"] = ov
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
