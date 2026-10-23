#!/usr/bin/env python
"""Strand 5 J2b (spec solve/specs/SPEC_strand5_kernel_star.md sec. 10(c); builder analyst): single-packet vacuum symbols and their q -> 0
coefficient k(m).

The 2c table (kernel_L256.json) stores the kernel summed over the four packets only, so the single-packet symbols are recomputed here with
the 2c packet routine (strand2c_kernel_v1.packet_terms, unmodified, imported): for one packet of mass m (orientation +), on the periodic
L^3 box with twist theta, at q = 2 pi n/L (n = 1..L/2) along [100],
    k(m; q) = -(U'' - T'')/(q_lat^2 L^3),  C(m; q) = -k(m; q) q_lat^2,
so that a packet of orientation eta contributes eta k(m; q) and the 2c table's k_a = sum_s eta_s k(m_s; q) (checked here against the table
for every equal-rectangle aH and for the physical row).  The zone-edge entry (n = L/2) is halved as in strand5_kernel_v2 (the norm of the
(-1)^x mode).  Per mass: the line kernel K_n (the inverse transform of C on the L-ring), its decay |K_n|/|K_0| at n = L/2 and L/4, the
moment k(m) = (1/2) sum n^2 K_n, and k(m; q_1) at the smallest q.  Mass sets: (A) the distinct masses of 2c's equal rectangles,
0.1 and sqrt(0.01 + aH^2), sqrt(0.01 + 2 aH^2) for aH in {1, 0.5, 0.25, 0.1, 0.05, 0.02, 0.013} (twist 0, as the table); (B) the
physical row's heavy masses 0.0130903, 0.0185125 (twist 0.371, as the physical row) at L = 256 and 512; (C) further masses
0.02, 0.03, 0.05, 0.07, 0.15, 0.2, 0.3 (twist 0) for the fit k(m) = m^2 (alpha log m^2 + beta) over m <= 0.3; (D) the massless packet
(twist 0.371) at L = 128 and 256, its symbol and the log-log slope of |C| against q over q in [0.05, 0.4].  Parallel over q (workers,
one thread each).  Writes <out>/packets_<tag>.json per (set, mass, L)."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(HERE, "..", "strand2"), HERE):
    sys.path.insert(0, _p)
import strand2c_kernel_v1 as KC  # noqa: E402

_G = {}


def init(L, theta):
    sp, cnt = KC.perp_classes(L, theta[1], theta[2])
    _G.update(L=L, theta=theta, sp=sp, cnt=cnt)


def one(args):
    m, n = args
    L = _G["L"]
    t2, u2, w11 = KC.packet_terms(m, L, n, _G["theta"], _G["sp"], _G["cnt"])
    q = 2 * np.pi * n / L
    ql2 = 2 * (1 - np.cos(q))
    return {"n": n, "q": q, "k": float(-(u2 - t2) / (ql2 * L ** 3)), "mixed": float(-w11 / (ql2 * L ** 3)), "T2": t2, "U2": u2, "W11": w11}


def analyse(rows, L):
    rows = sorted(rows, key=lambda r: r["n"])
    q = np.array([r["q"] for r in rows])
    k = np.array([r["k"] for r in rows])
    kk = k.copy()
    kk[-1] = kk[-1] / 2          # zone edge
    Cq = np.zeros(L)
    ql2 = 2 * (1 - np.cos(q))
    for j, r in enumerate(rows):
        n = r["n"]
        Cq[n % L] = -kk[j] * ql2[j]
        Cq[(-n) % L] = -kk[j] * ql2[j]
    K = np.real(np.fft.ifft(Cq))
    nn = np.arange(L)
    nn = np.where(nn > L // 2, nn - L, nn)
    out = {"k_moment": float(0.5 * np.sum(nn ** 2 * K)), "c4_moment": float(-np.sum(nn.astype(float) ** 4 * K) / 24),
           "k_smallest_q": float(k[0]), "q_smallest": float(q[0]), "K0": float(K[0]),
           "decay_half": float(abs(K[L // 2]) / abs(K[0])), "decay_quarter": float(abs(K[L // 4]) / abs(K[0])),
           "k_zone_edge_tabulated": float(k[-1]), "k_next_to_edge": float(k[-2])}
    sel = (q >= 0.05) & (q <= 0.4)
    if np.sum(sel) >= 3:
        C = np.abs(kk * ql2)
        out["loglog_slope_C_q_0.05_0.4"] = float(np.polyfit(np.log(q[sel]), np.log(C[sel]), 1)[0])
    return out, K


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", required=True, choices=["A", "B", "C", "D"])
    ap.add_argument("--L", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--masses", default=None, help="override: comma-separated masses")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    if a.set == "A":
        ms = sorted({0.1} | {float(np.sqrt(0.01 + h * h)) for h in (1, .5, .25, .1, .05, .02, .013)} |
                    {float(np.sqrt(0.01 + 2 * h * h)) for h in (1, .5, .25, .1, .05, .02, .013)})
        theta = (0.0, 0.0, 0.0)
    elif a.set == "B":
        ms = [0.013090347820517666, 0.01851254742395717]
        theta = (0.371, 0.371, 0.371)
    elif a.set == "C":
        ms = [0.02, 0.03, 0.05, 0.07, 0.15, 0.2, 0.3]
        theta = (0.0, 0.0, 0.0)
    else:
        ms = [0.0]
        theta = (0.371, 0.371, 0.371)
    if a.masses:
        ms = [float(x) for x in a.masses.split(",")]
    L = a.L
    ns = list(range(1, L // 2 + 1))
    Ks = {}
    with Pool(a.workers, initializer=init, initargs=(L, theta)) as pool:
        for m in ms:
            rows = pool.map(one, [(m, n) for n in ns], chunksize=4)
            an, K = analyse(rows, L)
            rec = {"set": a.set, "m": m, "L": L, "theta": list(theta), "rows": rows, "analysis": an, "seconds": time.time() - t0}
            tag = "%s_m%.8g_L%d" % (a.set, m, L)
            with open(os.path.join(a.out, "packets_%s.json" % tag), "w") as fh:
                json.dump(rec, fh, indent=1)
            Ks[tag] = K
            print("set %s m %.8g L %d: k_moment %.10e k(q1) %.10e decay half %.2e quarter %.2e slope %s (%.0f s)" % (
                a.set, m, L, an["k_moment"], an["k_smallest_q"], an["decay_half"], an["decay_quarter"],
                an.get("loglog_slope_C_q_0.05_0.4"), time.time() - t0), flush=True)
    np.savez_compressed(os.path.join(a.out, "packets_%s_L%d_K.npz" % (a.set, L)), **Ks)


if __name__ == "__main__":
    main()
