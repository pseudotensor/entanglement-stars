#!/usr/bin/env python
"""Strand 3 addendum part B' step 1 (spec solve/specs/SPEC_strand3_addendum.md, parts B' and F; builder analyst): the four transverse
valley centres of the [100] cut for part B's 8-valley family M'(k) = kappa'^-1 sum_i sin^2 k_i (q = 0).
The channel entropy is unchanged under k_perp,i -> pi - k_perp,i (sin and sin^2 are), so the neighbourhoods of (pi, 0), (0, pi), (pi, pi)
are copies of the origin's.  Part B's quadrature refined only the origin's pi/6 square (inner + middle) and kept the coarse 12 x 12 outer
rule elsewhere, whose four points in the pi/6 square around each of the other three centres (2 pi (n + 1/2)/12 at 165 and 195 degrees,
or 345 and 15) all equal the channel at (pi/12, pi/12) by that symmetry.  Corrected eta_total = eta_total(printed) - 12 (1/144) S(pi/12, pi/12)
+ 3 (eta_inner + eta_middle), at kappa' = 0.01 and 0.03, L = 128, nfft = 16 L (part B's settings; S from the addendum's
interval_covariance_sin2, the same chan).  Writes <out>/Bprime1.json."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import json
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "orient"))
import strand3_addAB_v1 as AB  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--addB", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = json.load(open(a.addB))["rows"]
    res = []
    q15 = (np.pi / 12, np.pi / 12)
    for kp in (0.01, 0.03):
        r = [x for x in rows if x["kappa"] == kp and x["L"] == 128][0]
        S15 = AB.chan(("B", q15, 128, 16 * 128, kp))
        S165 = AB.chan(("B", (np.pi - np.pi / 12, np.pi / 12), 128, 16 * 128, kp))
        corr = r["eta_total"] - 12 * S15 / 144.0 + 3 * (r["eta_inner"] + r["eta_middle"])
        res.append({"kappa_prime": kp, "L": 128, "nfft": 2048, "eta_total_printed": r["eta_total"], "eta_inner": r["eta_inner"],
                    "eta_middle": r["eta_middle"], "eta_outer": r["eta_outer"], "S_at_15_15": S15, "S_at_165_15": S165,
                    "removed_outer_12_points": 12 * S15 / 144.0, "added_three_origin_patches": 3 * (r["eta_inner"] + r["eta_middle"]),
                    "eta_total_corrected": corr, "corrected_over_printed": corr / r["eta_total"]})
        print(json.dumps(res[-1]), flush=True)
    with open(os.path.join(a.out, "Bprime1.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
