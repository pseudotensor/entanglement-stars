#!/usr/bin/env python
"""Strand 3 addendum part F''(b) (spec solve/specs/SPEC_strand3_addendum.md, "Part F''(b) and F''' additions"; builder analyst): at kappa =
0.0023, the equal aH = 1 rectangle's modified seas (masses 0.1, sqrt(1.01), sqrt(2.01); M_kappa(k) = m + kappa^-1 sum (1 - cos k_i)), L = 128,
nfft 16 L (part F'''s refinement setting), the transverse regions of part A's rule and their refinements at fixed inner resolution:
  inner     the square of one cell pi/72 (part A's rule at this kappa) on 48 x 48 midpoints (unchanged);
  middle24  the 24 x 24 cells (pi/72) of [-pi/6, pi/6]^2 without the inner square's cells (part F'''s rule); middle48 the 48 x 48 cells (pi/144)
            without the cells inside the inner square (aligned: the inner square is 4 x 4 of them);
  outer12   the 12 x 12 points outside the pi/6 square (part F'''s rule); outer24 the 24 x 24 points outside it (aligned).
Each transverse D4 orbit evaluated once with its weight (as stage 2 of part F'''; checked there).  eta = sum_w w S/2.  Writes <out>/partFpp_refine.json."""
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
import strand3_partFppp_eta_v1 as S2  # noqa: E402

KAPPA = 0.0023
MASSES = {"0.1": 0.1, "sqrt(1.01)": math.sqrt(1.01), "sqrt(2.01)": math.sqrt(2.01)}
L = 128
NFFT = 16 * L
P100 = (1, 0, 0)
K1 = math.pi / 72


def regions():
    return {"inner": S2.region_inner(K1, 48), "middle24": S2.region_middle(K1, 24), "middle48": S2.region_middle(K1, 48),
            "outer12": S2.region_outer(12), "outer24": S2.region_outer(24)}


def chan(args):
    q, m = args
    return entropy(interval_covariance_wilson(P100, q, L, NFFT, m, KAPPA)) / 2.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    R = regions()
    res = {"meta": {"kappa": KAPPA, "masses": MASSES, "L": L, "nfft": NFFT, "K1": K1, "version": 1,
                    "points": {k: len(v) for k, v in R.items()}, "orbits": {k: len(S2.orbits(v)) for k, v in R.items()},
                    "weights": {k: math.fsum(w for _, w in v) for k, v in R.items()}}, "rows": []}
    out = os.path.join(a.out, "partFpp_refine.json")
    with Pool(a.workers) as pool:
        for name, m in MASSES.items():
            for reg, pts in R.items():
                orb = S2.orbits(pts)
                S = pool.map(chan, [(q, m) for q, w in orb], chunksize=2)
                eta = math.fsum(w * s for (q, w), s in zip(orb, S))
                res["rows"].append({"mass": name, "m": m, "region": reg, "orbits": len(orb), "eta": eta})
                print("%s %s (%d orbits): eta %.12e (%.0f s)" % (name, reg, len(orb), eta, time.time() - t0), flush=True)
                json.dump(res, open(out, "w"), indent=1)
    res["meta"]["seconds"] = time.time() - t0
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
