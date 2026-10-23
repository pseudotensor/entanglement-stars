#!/usr/bin/env python
"""Strand 3 addendum part B' step 2 and the part A refinement (spec solve/specs/SPEC_strand3_addendum.md, parts B' and F; builder analyst).

B' step 2: part B's 8-valley family M'(k) = kappa'^-1 sum_i sin^2 k_i (q = 0) with the transverse quadrature replicated at the four valley
centres k_perp in {0, pi}^2 of the [100] cut: the origin's nesting (inner square, middle 24 x 24 grid of [-pi/6, pi/6]^2) at each centre and
the 12 x 12 outer rule without its points in the four pi/6 squares.  The channel entropy is unchanged under k_perp,i -> pi - k_perp,i, so the
three other centres' nestings are exact copies of the origin's: eta = 4 (eta_inner + eta_middle) + eta_outer(without the 12 points of the
other three squares), with the outer rule's points evaluated as in part B; the printed part B total is reported beside.  Settings
kappa' in {0.1, 0.03, 0.01}, L in {64, 128, 256}, nfft 16 L and 32 L; the secant powers d log eta/d log kappa' recomputed per (L, nfft).
Part A refinement (the Wilson family M_kappa(k) = kappa^-1 sum_i (1 - cos k_i), q = 0; doubler-free, one valley): at kappa = 0.002 and
0.003, L = 128, nfft 16 L: the middle rule 24 x 24 -> 48 x 48 on [-pi/6, pi/6]^2 outside the inner square, and the outer rule 12 x 12 ->
24 x 24 outside the pi/6 square, each changed separately; each region's value at both resolutions.
Channels in a pool of single-thread workers (strand3_addAB_v1's chan).  Writes <out>/Bprime2.json and <out>/Arefine.json."""
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
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "orient"))
import strand3_addAB_v1 as AB  # noqa: E402


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def outer_points(n_per_axis, exclude_centres):
    """Midpoints 2 pi (j + 1/2)/n of the full zone outside the pi/6 square around each listed centre; weight 1/n^2."""
    pts = []
    g = 2 * np.pi * (np.arange(n_per_axis) + 0.5) / n_per_axis
    for a in g:
        for b in g:
            inside = False
            for (ca, cb) in exclude_centres:
                if abs(wrap(a - ca)) < np.pi / 6 + 1e-12 and abs(wrap(b - cb)) < np.pi / 6 + 1e-12:
                    inside = True
            if not inside:
                pts.append(((a, b), 1.0 / n_per_axis ** 2))
    return pts


def middle_points(K1, n_cells_per_side):
    cell = (np.pi / 3) / n_cells_per_side
    cen = -np.pi / 6 + cell * (np.arange(n_cells_per_side) + 0.5)
    pts = []
    for a in cen:
        for b in cen:
            if abs(a) < K1 and abs(b) < K1:
                continue
            pts.append(((a, b), cell ** 2 / (2 * np.pi) ** 2))
    return pts


def run_parts(pool, fam, kappa, L, nfft, parts):
    out = {}
    for name, pts in parts.items():
        S = pool.map(AB.chan, [(fam, qq, L, nfft, kappa) for qq, w in pts], chunksize=8)
        out[name] = float(sum(w * s for (qq, w), s in zip(pts, S)))
        out["n_" + name] = len(pts)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--only", default="both", choices=["both", "B", "A"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    centres = [(0.0, 0.0), (np.pi, 0.0), (0.0, np.pi), (np.pi, np.pi)]
    with Pool(a.workers) as pool:
        if a.only in ("both", "B"):
            rows = []
            for kp in (0.1, 0.03, 0.01):
                Q = AB.quadrature(kp)
                for L in (64, 128, 256):
                    for mult in (16, 32):
                        nfft = mult * L
                        parts = {"inner": Q["inner"], "middle": Q["middle"], "outer_origin_only": Q["outer"],
                                 "outer_four_centres": outer_points(12, centres)}
                        r = run_parts(pool, "B", kp, L, nfft, parts)
                        r.update({"kappa_prime": kp, "L": L, "nfft": nfft, "K1": Q["K1"]})
                        r["eta_part_B_rule"] = r["inner"] + r["middle"] + r["outer_origin_only"]
                        r["eta_four_centres"] = 4 * (r["inner"] + r["middle"]) + r["outer_four_centres"]
                        r["ratio"] = r["eta_four_centres"] / r["eta_part_B_rule"]
                        rows.append(r)
                        print("B' kappa' %g L %d nfft %d: part B rule %.6e four centres %.6e ratio %.4f (%.0f s)" % (
                            kp, L, nfft, r["eta_part_B_rule"], r["eta_four_centres"], r["ratio"], time.time() - t0), flush=True)
            powers = []
            for L in (64, 128, 256):
                for mult in (16, 32):
                    rr = sorted([r for r in rows if r["L"] == L and r["nfft"] == mult * L], key=lambda r: -r["kappa_prime"])
                    for x, y in zip(rr[:-1], rr[1:]):
                        powers.append({"L": L, "nfft": mult * L, "kappa_pair": [x["kappa_prime"], y["kappa_prime"]],
                                       "p_four_centres": float(np.log(y["eta_four_centres"] / x["eta_four_centres"]) / np.log(y["kappa_prime"] / x["kappa_prime"])),
                                       "p_part_B_rule": float(np.log(y["eta_part_B_rule"] / x["eta_part_B_rule"]) / np.log(y["kappa_prime"] / x["kappa_prime"]))})
            with open(os.path.join(a.out, "Bprime2.json"), "w") as fh:
                json.dump({"rows": rows, "secant_powers": powers, "seconds": time.time() - t0}, fh, indent=1)
        if a.only in ("both", "A"):
            rows = []
            for kappa in (0.002, 0.003):
                Q = AB.quadrature(kappa)
                L, nfft = 128, 16 * 128
                parts = {"inner": Q["inner"], "middle_24": Q["middle"], "middle_48": middle_points(Q["K1"], 48),
                         "outer_12": Q["outer"], "outer_24": outer_points(24, [(0.0, 0.0)])}
                r = run_parts(pool, "A", kappa, L, nfft, parts)
                r.update({"kappa": kappa, "L": L, "nfft": nfft, "K1": Q["K1"]})
                r["eta_printed_rule"] = r["inner"] + r["middle_24"] + r["outer_12"]
                r["eta_middle_refined"] = r["inner"] + r["middle_48"] + r["outer_12"]
                r["eta_outer_refined"] = r["inner"] + r["middle_24"] + r["outer_24"]
                r["eta_both_refined"] = r["inner"] + r["middle_48"] + r["outer_24"]
                rows.append(r)
                print("A kappa %g: printed %.6e middle refined %.6e outer refined %.6e both %.6e (%.0f s)" % (
                    kappa, r["eta_printed_rule"], r["eta_middle_refined"], r["eta_outer_refined"], r["eta_both_refined"], time.time() - t0), flush=True)
            with open(os.path.join(a.out, "Arefine.json"), "w") as fh:
                json.dump({"rows": rows, "seconds": time.time() - t0}, fh, indent=1)


if __name__ == "__main__":
    main()
