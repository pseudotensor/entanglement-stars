#!/usr/bin/env python
"""Strand 3 addendum part F'' (2) (spec solve/specs/SPEC_strand3_addendum.md, part F''; builder analyst): the cut-entropy coefficients
eta_100 of the seas of the equal aH = 1 rectangle's masses m in {0.1, sqrt(1.01), sqrt(2.01)} (multiplicities 1, 2, 1):
  modified: the Wilson-type mass M_kappa(k) = m + kappa^-1 sum_i (1 - cos k_i), kappa = 0.0023 (lattice_entropy_wilson.interval_covariance_wilson
            with mass m), on part A's nested transverse quadrature for this kappa (strand3_addAB_v1.quadrature: the inner square of one
            cell pi/72 on 48 x 48 points, the middle 24 x 24 cells of [-pi/6, pi/6]^2 without it, the outer 12 x 12 points);
  unmodified: the constant mass m (kappa = infinity) on part C's massive-sea route (the 24 x 24 midpoint grid of the whole zone), with
            the paper's eta.value (12 x 12) beside it as the control;
L in {128, 256}, nfft in {16 L, 32 L}; and the transverse-refinement checks at L = 128, nfft = 16 L: the modified seas with the inner
square of four cells (4 pi/72) on 48 x 48 points, the unmodified with a 36 x 36 grid.  eta = sum_w w S_channel/(2 |p|), |p| = 1, the
channel entropy of the interval covariance on L sites (lattice_entropy.entropy/2), single-thread workers.  Writes <out>/partFpp_eta.json
(rewritten after every row)."""
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
sys.path.insert(0, os.path.join(HERE, "orient"))
from lattice_entropy import entropy  # noqa: E402
from lattice_entropy_wilson import interval_covariance_wilson  # noqa: E402
import eta as E  # noqa: E402
import strand3_addAB_v1 as AB  # noqa: E402

KAPPA = 0.0023
MASSES = {"0.1": 0.1, "sqrt(1.01)": math.sqrt(1.01), "sqrt(2.01)": math.sqrt(2.01)}
MULT = {"0.1": 1, "sqrt(1.01)": 2, "sqrt(2.01)": 1}
P100 = (1, 0, 0)


def quad_modified(nin):
    """Part A's nested quadrature with the inner square of nin cells (part A's rule gives nin = 1 at kappa = 0.0023)."""
    cell = np.pi / 72
    K1 = nin * cell
    inner = [((a, b), (2 * K1) ** 2 / (48 ** 2 * (2 * np.pi) ** 2))
             for a in -K1 + 2 * K1 * (np.arange(48) + 0.5) / 48 for b in -K1 + 2 * K1 * (np.arange(48) + 0.5) / 48]
    mid = []
    cen = -np.pi / 6 + cell * (np.arange(24) + 0.5)
    for a in cen:
        for b in cen:
            if abs(a) < K1 and abs(b) < K1:
                continue
            mid.append(((a, b), cell ** 2 / (2 * np.pi) ** 2))
    outer = []
    for a in 2 * np.pi * (np.arange(12) + 0.5) / 12:
        for b in 2 * np.pi * (np.arange(12) + 0.5) / 12:
            if abs(AB.wrap(a)) > np.pi / 6 + 1e-12 or abs(AB.wrap(b)) > np.pi / 6 + 1e-12:
                outer.append(((a, b), 1.0 / 144))
    return inner + mid + outer


def quad_uniform(n):
    g = -np.pi + 2 * np.pi * (np.arange(n) + 0.5) / n
    w = 1.0 / n ** 2
    return [((a, b), w) for a in g for b in g]


def chan(args):
    qq, L, nfft, mass, kappa = args
    return entropy(interval_covariance_wilson(P100, qq, L, nfft, mass, kappa)) / 2.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    nin_rule = AB.quadrature(KAPPA)["n_cells"]
    res = {"meta": {"kappa": KAPPA, "masses": MASSES, "multiplicities": MULT, "part_A_inner_cells_at_kappa": nin_rule}, "rows": []}
    jobs = []
    for L in (128, 256):
        for mult in (16, 32):
            for name in MASSES:
                jobs.append(("modified", name, L, mult * L, "part A rule (inner %d cell)" % nin_rule, quad_modified(nin_rule)))
                jobs.append(("unmodified", name, L, mult * L, "24 x 24 whole zone", quad_uniform(24)))
    for name in MASSES:
        jobs.append(("modified", name, 128, 16 * 128, "inner 4 cells (refinement check)", quad_modified(4)))
        jobs.append(("unmodified", name, 128, 16 * 128, "36 x 36 whole zone (refinement check)", quad_uniform(36)))
    with Pool(a.workers) as pool:
        for sea, name, L, nfft, qname, Q in jobs:
            kap = KAPPA if sea == "modified" else np.inf
            S = pool.map(chan, [(qq, L, nfft, MASSES[name], kap) for qq, w in Q], chunksize=4)
            eta = float(math.fsum(w * s for (qq, w), s in zip(Q, S)))
            row = {"sea": sea, "mass": name, "m": MASSES[name], "L": L, "nfft": nfft, "quadrature": qname, "channels": len(Q),
                   "weight_sum_minus_1": float(sum(w for _, w in Q) - 1), "eta": eta}
            if sea == "unmodified" and qname.startswith("24"):
                row["eta_value_12x12"] = float(E.value(P100, L, 12, nfft, MASSES[name]))
            res["rows"].append(row)
            print("%s %s L %d nfft %d (%s, %d channels): eta %.10e%s (%.0f s)" % (sea, name, L, nfft, qname, len(Q), eta,
                  (" ; eta.value 12x12 %.10e" % row["eta_value_12x12"]) if "eta_value_12x12" in row else "", time.time() - t0), flush=True)
            with open(os.path.join(a.out, "partFpp_eta.json"), "w") as fh:
                json.dump(res, fh, indent=1)
    res["meta"]["seconds"] = time.time() - t0
    with open(os.path.join(a.out, "partFpp_eta.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
