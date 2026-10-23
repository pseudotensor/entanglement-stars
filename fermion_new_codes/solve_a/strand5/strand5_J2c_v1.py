#!/usr/bin/env python
"""Strand 5 J2c (spec solve/specs/SPEC_strand5_kernel_star.md sec. 10(f); builder analyst): the physical row's Newtonian coefficient at
the needed precision.  The eta-summed kernel of the lambda = 1 rectangle (masses 0, 0.0130903, 0.0130903, 0.0185125; orientations
(+, -, -, +)) by the first-variation form
    k_a(q) = (6/L^3) sum_k [sin(k_1 + q) - sin k_1]^2 sum_s eta_s / (q_lat^2 [E_s(k) + E_s(k + q)]),   E_s(k) = sqrt(m_s^2 + sum_i sin^2 k_i),
with the four packets summed inside the momentum sum at every k-point (the lattice constant c_0 cancels there), k = (2 pi j + theta)/L per
axis, q = 2 pi n/L along axis 1, q_lat^2 = 2 (1 - cos q); the transverse sum over the distinct values of sin^2 k_2 + sin^2 k_3 with their
counts; the k_1 x class array in chunks, each chunk's float64 partial sum accumulated by math.fsum (compensated).  Runs: 'massless' (the
light packet massless, twist 0.371 on every axis, as the 2c physical row) and 'm1e-4' (m_light a = 1e-4, twist 0); L in {1024, 2048},
n in {1, 2, 3, 4}; the single massless (or 1e-4) packet alone at the same points as the cancellation's control (its symbol c_0(q)).
Control (--control): this form against strand2c_kernel_v1's second-order kernel (packet_terms, eta-summed) at L = 64 and 128, n = 1, 2, 5,
both runs.  Writes <out>/J2c_<run>_L<L>_n<n>.json per point (or <out>/control.json)."""
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
for _p in (os.path.join(HERE, "..", "strand2"), HERE):
    sys.path.insert(0, _p)

HEAVY = [0.013090347820517666, 0.013090347820517666, 0.01851254742395717]
ETAS = [1, -1, -1, 1]
RUNS = {"massless": {"m_light": 0.0, "theta": 0.371}, "m1e-4": {"m_light": 1e-4, "theta": 0.0}}


def classes(L, theta):
    k = (2 * np.pi * np.arange(L) + theta) / L
    s2 = np.sin(k) ** 2
    sp = (s2[:, None] + s2[None, :]).ravel()
    u, inv, cnt = np.unique(np.round(sp, 13), return_inverse=True, return_counts=True)
    vals = np.bincount(np.asarray(inv).ravel(), weights=sp) / cnt
    return vals, cnt.astype(float)


def symbol(L, n, theta, masses, etas, chunk=8192):
    """(6/L^3) sum_k [sin(k_1 + q) - sin k_1]^2 sum_s eta_s/(E_s(k) + E_s(k + q)) / q_lat^2, fsum over chunk partial sums."""
    q = 2 * np.pi * n / L
    ql2 = 2 * (1 - np.cos(q))
    k1 = (2 * np.pi * np.arange(L) + theta) / L
    s1 = np.sin(k1)[:, None]
    s1q = np.sin(k1 + q)[:, None]
    num = (s1q - s1) ** 2
    a1 = s1 * s1
    a1q = s1q * s1q
    sp, cnt = classes(L, theta)
    parts = []
    for c0 in range(0, sp.size, chunk):
        spc = sp[None, c0:c0 + chunk]
        acc = np.zeros((L, spc.shape[1]))
        for m, e in zip(masses, etas):
            acc += e / (np.sqrt(m * m + a1 + spc) + np.sqrt(m * m + a1q + spc))
        parts.append(float(np.sum((num * acc) * cnt[None, c0:c0 + chunk])))
    return 6.0 * math.fsum(parts) / (L ** 3 * ql2), q, int(sp.size)


def task(args):
    run, L, n, out = args
    R = RUNS[run]
    t0 = time.time()
    masses = [R["m_light"]] + HEAVY
    ka, q, ncls = symbol(L, n, R["theta"], masses, ETAS)
    c0, _, _ = symbol(L, n, R["theta"], [R["m_light"]], [1])
    rec = {"run": run, "L": L, "n": n, "q": q, "theta": R["theta"], "masses": masses, "etas": ETAS, "k_a": ka, "light_alone_c0": c0,
           "transverse_classes": ncls, "seconds": time.time() - t0}
    with open(os.path.join(out, "J2c_%s_L%d_n%d.json" % (run, L, n)), "w") as fh:
        json.dump(rec, fh, indent=1)
    print("%s L %d n %d: k_a %.15e light alone %.15e (%d classes, %.0f s)" % (run, L, n, ka, c0, ncls, time.time() - t0), flush=True)
    return rec


def control(out):
    import strand2c_kernel_v1 as KC
    rows = []
    for run, R in RUNS.items():
        masses = [R["m_light"]] + HEAVY
        for L in (64, 128):
            th = R["theta"]
            sp, cnt = KC.perp_classes(L, th, th)
            for n in (1, 2, 5):
                kr = KC.kernel(L, masses, ETAS, n, (th, th, th))
                ka, q, _ = symbol(L, n, th, masses, ETAS)
                t2, u2, _ = KC.packet_terms(masses[0], L, n, (th, th, th), sp, cnt)
                c0_ref = -(u2 - t2) / (2 * (1 - np.cos(2 * np.pi * n / L)) * L ** 3)
                c0, _, _ = symbol(L, n, th, [masses[0]], [1])
                rows.append({"run": run, "L": L, "n": n, "k_a_first_variation": ka, "k_a_second_order": kr["k_a"], "rel": ka / kr["k_a"] - 1,
                             "light_first_variation": c0, "light_second_order": c0_ref, "light_rel": c0 / c0_ref - 1})
                print(json.dumps(rows[-1]), flush=True)
    with open(os.path.join(out, "control.json"), "w") as fh:
        json.dump(rows, fh, indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--Ls", default="1024,2048")
    ap.add_argument("--ns", default="1,2,3,4")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.control:
        control(a.out)
        return
    tasks = [(run, L, n, a.out) for L in sorted(int(x) for x in a.Ls.split(",")) for run in RUNS for n in [int(x) for x in a.ns.split(",")]]
    tasks.sort(key=lambda t: -t[1])
    with Pool(a.workers) as pool:
        pool.map(task, tasks, chunksize=1)


if __name__ == "__main__":
    main()
