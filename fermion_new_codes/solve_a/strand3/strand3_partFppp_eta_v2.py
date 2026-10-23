#!/usr/bin/env python
"""Strand 3 addendum part F''' stage 2 (spec solve/specs/SPEC_strand3_addendum.md, part F'''; the lead's ruling of 15:34 UTC, JUDGMENT_strand3_
addendum.md sec. 12; builder analyst): the cut-entropy coefficients eta_100 of the modified seas of the PHYSICAL inventory, the massless light
sea (m = 0) and the heavy seas m = mu and sqrt2 mu (mu = 0.0130903478; multiplicities 1, 2, 1), each with the Wilson-type mass
M_kappa(k) = m + kappa^-1 sum_i (1 - cos k_i) (lattice_entropy_wilson.interval_covariance_wilson: strand 3's interval covariance with the
longitudinal twist 0.371), at kappa = 0.0023 and 0.001.  eta = sum_w w S_channel/(2 |p|), |p| = 1, S_channel the entropy of the interval
covariance on L sites (lattice_entropy.entropy), single-thread workers.

Transverse quadrature (nested, every region a union of cells, the weights summing to 1):
  inner    the square [-a, a]^2, a = 4 kappa (the specification's side 8 kappa), on n_in x n_in midpoints (64 x 64);
  annulus  [-b, b]^2 without the inner square, b = pi/72 (the central 4 x 4 cells of the 48 x 48 middle grid; b > a at both kappa): per
           axis the cells [-a, a] in n_s = 16 uniform cells and [a, b] (mirrored) in n_g = 12 geometric cells (ratio (b/a)^(1/12)), the
           tensor product without the central n_s x n_s block, each cell at its midpoint (builder's addition, stated: the specification's
           inner square does not align with the middle grid, and this makes the nesting exact);
  middle   the n_mid x n_mid midpoint grid of [-pi/6, pi/6]^2 (48 x 48, cell pi/144) without the cells inside [-b, b]^2;
  outer    the n_out x n_out grid points 2 pi (j + 1/2)/n_out (24 x 24) outside the pi/6 square (cells aligned with it).
The channels are symmetric under the transverse D4 (q_i -> -q_i, q_1 <-> q_2; a reflection or rotation of the [100] cut implemented by an
on-site unitary): every rule here is D4-symmetric, so each orbit is evaluated once at its representative (|q|_max, |q|_min) with the orbit's
total weight (builder's choice, stated); the symmetry is checked at sample points in every region (all eight images, per sea and kappa).

Runs (per sea and kappa): production L = 256 and 512 at nfft = 32 L; at L = 256 the checks, each changing one region only (the other regions
are the production's): nfft = 16 L (all regions); the inner square doubled (a = 8 kappa) at doubled point count (128 x 128) with its
annulus (n_s = 32); the middle rule refined to 96 x 96; the outer rule refined to 48 x 48.  Writes <out>/partFppp_eta.json (rewritten after
every region).
(version 2, part F'''(d): --kappas selects the kappa (default 1e-4); the annulus's geometric cell count scales with log(b/a) so that the cell
ratio stays version 1's at kappa = 0.0023, (b/a)^(1/12) with a = 4 kappa there: n_g = ceil(log(b/a)/log(that ratio)) (36 at kappa = 1e-4), stated;
--Ls and --checks select the production and check runs; --seas the seas (default all three).)"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import math
import socket
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "orient"))
from lattice_entropy import entropy  # noqa: E402
from lattice_entropy_wilson import interval_covariance_wilson  # noqa: E402

MU = 0.013090347820517666
SEAS = {"light": 0.0, "mu": MU, "sqrt2mu": math.sqrt(2.0) * MU}
MULT = {"light": 1, "mu": 2, "sqrt2mu": 1}
KAPPAS = (0.0023, 0.001)
P100 = (1, 0, 0)
B_CENTRAL = math.pi / 72
TWO_PI_SQ = (2 * math.pi) ** 2


def wrap(x):
    return (x + math.pi) % (2 * math.pi) - math.pi


def cells_1d(edges):
    e = np.asarray(edges, float)
    return 0.5 * (e[:-1] + e[1:]), np.diff(e)


def region_inner(a, n_in):
    x, h = cells_1d(np.linspace(-a, a, n_in + 1))
    return [((p, q), hp * hq / TWO_PI_SQ) for p, hp in zip(x, h) for q, hq in zip(x, h)]


def region_annulus(a, b, n_s, n_g):
    r = (b / a) ** (1.0 / n_g)
    pos = [a * r ** j for j in range(n_g + 1)]
    pos[-1] = b
    edges = [-x for x in reversed(pos)] + list(np.linspace(-a, a, n_s + 1)[1:-1]) + pos
    x, h = cells_1d(edges)
    centre = [abs(v) < a for v in x]
    pts = []
    for i in range(len(x)):
        for j in range(len(x)):
            if centre[i] and centre[j]:
                continue
            pts.append(((x[i], x[j]), h[i] * h[j] / TWO_PI_SQ))
    return pts


def region_middle(b, n_mid):
    x, h = cells_1d(np.linspace(-math.pi / 6, math.pi / 6, n_mid + 1))
    return [((p, q), hp * hq / TWO_PI_SQ) for p, hp in zip(x, h) for q, hq in zip(x, h) if not (abs(p) < b and abs(q) < b)]


def region_outer(n_out):
    g = 2 * math.pi * (np.arange(n_out) + 0.5) / n_out
    return [((p, q), 1.0 / n_out ** 2) for p in g for q in g if abs(wrap(p)) > math.pi / 6 + 1e-12 or abs(wrap(q)) > math.pi / 6 + 1e-12]


def orbits(points):
    """Group D4-equivalent points: representative (max |q_i|, min |q_i|) (wrapped), total weight."""
    acc = {}
    for (p, q), w in points:
        u, v = abs(wrap(p)), abs(wrap(q))
        u, v = max(u, v), min(u, v)
        key = (round(u, 13), round(v, 13))
        if key in acc:
            acc[key][1] += w
        else:
            acc[key] = [(u, v), w]
    return list(acc.values())


def chan(args):
    qq, L, nfft, mass, kappa = args
    return entropy(interval_covariance_wilson(P100, qq, L, nfft, mass, kappa)) / 2.0


NG_RATIO = (B_CENTRAL / (4 * 0.0023)) ** (1.0 / 12)


def n_geo(a):
    return max(12, int(math.ceil(math.log(B_CENTRAL / a) / math.log(NG_RATIO))))


def rule(kappa, which):
    a = 4 * kappa
    if which == "inner":
        return region_inner(a, 64)
    if which == "annulus":
        return region_annulus(a, B_CENTRAL, 16, n_geo(a))
    if which == "middle":
        return region_middle(B_CENTRAL, 48)
    if which == "outer":
        return region_outer(24)
    if which == "inner_doubled":
        return region_inner(2 * a, 128)
    if which == "annulus_doubled":
        return region_annulus(2 * a, B_CENTRAL, 32, n_geo(2 * a))
    if which == "middle_96":
        return region_middle(B_CENTRAL, 96)
    if which == "outer_48":
        return region_outer(48)
    raise ValueError(which)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--Ls", default="256,512")
    ap.add_argument("--kappas", default="1e-4")
    ap.add_argument("--checks", default="yes", choices=["yes", "no"])
    ap.add_argument("--seas", default="light,mu,sqrt2mu")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, "partFppp_eta.json")
    t0 = time.time()
    KAPPAS_ = [float(x) for x in a.kappas.split(",")]
    SEAS_ = {k_: SEAS[k_] for k_ in a.seas.split(",")}
    res = {"meta": {"host": socket.gethostname(), "seas": SEAS_, "multiplicities": MULT, "kappas": KAPPAS_, "b_central": B_CENTRAL,
                    "twist": 0.371, "version": 2, "ng_ratio": NG_RATIO, "n_geo": {str(k_): n_geo(4 * k_) for k_ in KAPPAS_}}, "weights": {}, "symmetry_check": [], "regions": []}
    for kappa in KAPPAS_:
        tot = {}
        for which in ("inner", "annulus", "middle", "outer", "inner_doubled", "annulus_doubled", "middle_96", "outer_48"):
            pts = rule(kappa, which)
            tot[which] = {"points": len(pts), "orbits": len(orbits(pts)), "weight": math.fsum(w for _, w in pts)}
        tot["production_sum_minus_1"] = math.fsum(tot[w]["weight"] for w in ("inner", "annulus", "middle", "outer")) - 1.0
        tot["doubled_sum_minus_1"] = math.fsum(tot[w]["weight"] for w in ("inner_doubled", "annulus_doubled", "middle", "outer")) - 1.0
        tot["middle_96_sum_minus_1"] = math.fsum(tot[w]["weight"] for w in ("inner", "annulus", "middle_96", "outer")) - 1.0
        tot["outer_48_sum_minus_1"] = math.fsum(tot[w]["weight"] for w in ("inner", "annulus", "middle", "outer_48")) - 1.0
        res["weights"][str(kappa)] = tot
        print("kappa %g weights %s" % (kappa, json.dumps(tot)), flush=True)
    Ls = [int(x) for x in a.Ls.split(",")]
    jobs = []
    for L in sorted(Ls):
        for kappa in KAPPAS_:
            for sea in SEAS_:
                for which in ("inner", "annulus", "middle", "outer"):
                    jobs.append((sea, kappa, L, 32 * L, which))
                if L == 256 and a.checks == "yes":
                    for which in ("inner_doubled", "annulus_doubled", "middle_96", "outer_48"):
                        jobs.append((sea, kappa, L, 32 * L, which))
                    for which in ("inner", "annulus", "middle", "outer"):
                        jobs.append((sea, kappa, L, 16 * L, which))
    with Pool(a.workers) as pool:
        samples = {"inner": lambda k: (2.7 * k, 1.1 * k), "annulus": lambda k: (6.3 * k, 1.7 * k), "middle": lambda k: (0.11, 0.047),
                   "outer": lambda k: (1.3, 0.41)}
        for kappa in KAPPAS_:
            for sea, m in SEAS_.items():
                for reg, fn in samples.items():
                    u, v = fn(kappa)
                    imgs = [(u, v), (-u, v), (u, -v), (-u, -v), (v, u), (-v, u), (v, -u), (-v, -u)]
                    S = pool.map(chan, [(q, 256, 32 * 256, m, kappa) for q in imgs], chunksize=1)
                    rec = {"kappa": kappa, "sea": sea, "region": reg, "point": [u, v], "S": S[0],
                           "max_abs_dev": max(abs(x - S[0]) for x in S), "max_rel_dev": max(abs(x - S[0]) for x in S) / abs(S[0])}
                    res["symmetry_check"].append(rec)
                    print("symmetry kappa %g %s %s: S %.12e max rel dev %.2e (%.0f s)" % (kappa, sea, reg, S[0], rec["max_rel_dev"], time.time() - t0), flush=True)
        json.dump(res, open(out, "w"), indent=1)
        for sea, kappa, L, nfft, which in jobs:
            t1 = time.time()
            orb = orbits(rule(kappa, which))
            S = pool.map(chan, [(q, L, nfft, SEAS[sea], kappa) for q, w in orb], chunksize=1)
            eta = math.fsum(w * s for (q, w), s in zip(orb, S))
            res["regions"].append({"sea": sea, "m": SEAS[sea], "kappa": kappa, "L": L, "nfft": nfft, "region": which, "orbits": len(orb),
                                   "weight": math.fsum(w for _, w in orb), "eta": eta, "seconds": time.time() - t1})
            print("%s kappa %g L %d nfft %d %s (%d orbits): eta %.10e (%.0f s; total %.0f s)" % (sea, kappa, L, nfft, which, len(orb), eta,
                                                                                            time.time() - t1, time.time() - t0), flush=True)
            json.dump(res, open(out, "w"), indent=1)
    res["meta"]["seconds"] = time.time() - t0
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
