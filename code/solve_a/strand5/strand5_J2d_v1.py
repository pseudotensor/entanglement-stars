#!/usr/bin/env python
"""Strand 5 J2d (spec solve/specs/SPEC_strand5_kernel_star.md sec. 10(i); builder analyst): the physical row's Newtonian coefficient with the
light packet taken out of the q-fit.
(A) The light packet's q -> 0 constant c_0(L) = (3/L^3) sum_k cos^2 k_1/E(k), E(k) = sqrt(m^2 + sum_i sin^2 k_i), k = (2 pi j + theta)/L per
axis (the first-variation form at q = 0), for the runs 'massless' (m = 0, theta = 0.371) and 'm1e-4' (m = 1e-4, theta = 0), at L = 2048,
4096 and 8192: the transverse sum over the distinct values of sin^2 k_2 + sin^2 k_3 with their counts, the classes split into chunks over a
pool of single-thread workers, each chunk's float64 sum accumulated by math.fsum.
(B) The heavy packets' symbols k_s(q) at L = 2048, q = 2 pi n/L, n = 1..4, each packet alone (strand5_J2c_v1.symbol with one mass), at the
run's twist (0.371 for 'massless', 0 for 'm1e-4'); masses 0.0130903 and 0.0185125.
The report (strand5_J2d_report_v1.py) forms the Richardson extrapolation in 1/L^2, the apparent order, the per-packet fits k_s0 + k_s2 q^2 +
k_s4 q^4, and k_a(0; phys) = sum_heavy eta_s k_s0 + c_0(infinity) with its budget.  Writes <out>/J2d.json."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import math
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import strand5_J2c_v1 as J2C  # noqa: E402

RUNS = {"massless": {"m": 0.0, "theta": 0.371}, "m1e-4": {"m": 1e-4, "theta": 0.0}}
HEAVY = [0.013090347820517666, 0.01851254742395717]


def chunk_sum(args):
    L, theta, m, sp, cnt = args
    k1 = (2 * np.pi * np.arange(L) + theta) / L
    c2 = np.cos(k1)[:, None] ** 2
    a = np.sin(k1)[:, None] ** 2 + m * m
    parts = []
    for c0 in range(0, sp.size, 1024):
        s = sp[None, c0:c0 + 1024]
        parts.append(float(np.sum((c2 / np.sqrt(a + s)) * cnt[None, c0:c0 + 1024])))
    return math.fsum(parts)


def heavy_task(args):
    run, m, n, L = args
    R = RUNS[run]
    t0 = time.time()
    k, q, ncls = J2C.symbol(L, n, R["theta"], [m], [1])
    return {"run": run, "m": m, "n": n, "L": L, "q": q, "k": k, "classes": ncls, "seconds": time.time() - t0}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--Ls", default="2048,4096,8192")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    res = {"meta": {"runs": RUNS, "heavy": HEAVY, "Ls": [int(x) for x in a.Ls.split(",")], "workers": a.workers}, "c0": [], "heavy": []}
    with Pool(a.workers) as pool:
        # (B) first: small
        tasks = [(run, m, n, 2048) for run in RUNS for m in HEAVY for n in (1, 2, 3, 4)]
        for r in pool.imap_unordered(heavy_task, tasks):
            res["heavy"].append(r)
            print("heavy %s m %.7g n %d: k %.15e (%.0f s)" % (r["run"], r["m"], r["n"], r["k"], time.time() - t0), flush=True)
        with open(os.path.join(a.out, "J2d.json"), "w") as fh:
            json.dump(res, fh, indent=1)
        # (A)
        for L in res["meta"]["Ls"]:
            for run, R in RUNS.items():
                t1 = time.time()
                sp, cnt = J2C.classes(L, R["theta"])
                nch = max(a.workers * 8, 1)
                bounds = np.linspace(0, sp.size, nch + 1).astype(int)
                args = [(L, R["theta"], R["m"], sp[b0:b1], cnt[b0:b1]) for b0, b1 in zip(bounds[:-1], bounds[1:]) if b1 > b0]
                tot = math.fsum(pool.map(chunk_sum, args, chunksize=1))
                c0 = 3.0 * tot / L ** 3
                res["c0"].append({"run": run, "L": L, "c0": c0, "classes": int(sp.size), "seconds": time.time() - t1})
                print("c0 %s L %d: %.16e (%d classes, %.0f s)" % (run, L, c0, sp.size, time.time() - t1), flush=True)
                with open(os.path.join(a.out, "J2d.json"), "w") as fh:
                    json.dump(res, fh, indent=1)
    res["meta"]["seconds"] = time.time() - t0
    with open(os.path.join(a.out, "J2d.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
