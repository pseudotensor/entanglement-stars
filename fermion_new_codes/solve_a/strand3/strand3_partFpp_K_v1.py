#!/usr/bin/env python
"""Strand 3 addendum part F'' (1) (spec solve/specs/SPEC_strand3_addendum.md, part F''; builder analyst): the q = 0 coefficients of the equal
aH = 1 rectangle (masses 0.1, sqrt(1.01) twice, sqrt(2.01); orientations (+, -, -, +)) with the Wilson-type momentum-dependent mass
M_kappa,s(k) = m_s + kappa^-1 sum_i (1 - cos k_i), kappa = 0.0023, E_s(k) = (M_kappa,s(k)^2 + sum_i sin^2 k_i)^(1/2):
  K_N = k_sym(0) = sum_s eta_s 3 <cos^2 k_1/E_s(k)>   (the symmetric placement, all four packets modified),
  k_left(0) = sum_s eta_s <[3 cos^2 k_1 - kappa^-2 sin^2 k_1]/E_s(k)>   (the left placement),
  k_a(0) = sum_s eta_s 3 <cos^2 k_1/E_s(k)> with M = m_s (unmodified; control: J2's moment 0.31338 and J2f's integral),
<.> the Brillouin-zone average, the packets summed inside the integrand.  Two quadratures on [0, pi]^3 (the integrands are even in each
k_i): (a) a composite Gauss-Legendre product rule, panels [0, 0.02, 0.06, 0.2, 0.6, pi] in each direction with n nodes per panel, n = 16
and 32, the transverse pair reduced to k_2 <= k_3 (weight 2 off the diagonal); (b) the uniform midpoint grid k = pi (j + 1/2)/(L/2), L =
1024 and 2048 (the full-zone midpoint grid of L points per direction folded by the evenness), the same pair reduction, chunks over a pool,
math.fsum.  The cancellation check: the summed integrands at 2000 random points in float64 against mpmath at 40 digits.  Writes
<out>/partFpp_K.json."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import math
import time
from multiprocessing import Pool

import mpmath as mp
import numpy as np

KAPPA = 0.0023
MASSES = [0.1, math.sqrt(1.01), math.sqrt(1.01), math.sqrt(2.01)]
ETAS = [1, -1, -1, 1]
PANELS = [0.0, 0.02, 0.06, 0.2, 0.6, math.pi]


def integrands(k1, k2, k3, kappa=KAPPA):
    """Arrays (broadcast) of the three summed integrands: sym (3 cos^2 k1/E), left, unmodified."""
    c1 = np.cos(k1)
    s1 = np.sin(k1)
    t = (1 - c1) + (1 - np.cos(k2)) + (1 - np.cos(k3))
    s2 = s1 * s1 + np.sin(k2) ** 2 + np.sin(k3) ** 2
    sym = 0.0
    left = 0.0
    un = 0.0
    for m, e in zip(MASSES, ETAS):
        M = m + t / kappa
        E = np.sqrt(M * M + s2)
        Eu = np.sqrt(m * m + s2)
        sym = sym + e * 3 * c1 * c1 / E
        left = left + e * (3 * c1 * c1 - s1 * s1 / kappa ** 2) / E
        un = un + e * 3 * c1 * c1 / Eu
    return sym, left, un


def gl_nodes(n):
    x, w = np.polynomial.legendre.leggauss(n)
    xs, ws = [], []
    for a, b in zip(PANELS[:-1], PANELS[1:]):
        xs.append(0.5 * (b - a) * x + 0.5 * (b + a))
        ws.append(0.5 * (b - a) * w)
    return np.concatenate(xs), np.concatenate(ws)


def pair_task(args):
    k1, w1, k2, k3, wp = args
    sym, left, un = integrands(k1[:, None], k2[None, :], k3[None, :])
    W = w1[:, None] * wp[None, :]
    return [math.fsum((W * a).ravel()) for a in (sym, left, un)]


def run_grid(pool, k, w, nch):
    """sum over k1 in k (weights w) and pairs k2 <= k3 (weight 2 off the diagonal); normalized by pi^3."""
    i, j = np.triu_indices(len(k))
    wp = w[i] * w[j] * np.where(i == j, 1.0, 2.0)
    bounds = np.linspace(0, len(i), nch + 1).astype(int)
    tasks = [(k, w, k[i[a:b]], k[j[a:b]], wp[a:b]) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
    parts = pool.map(pair_task, tasks, chunksize=1)
    return [math.fsum(p[c] for p in parts) / math.pi ** 3 for c in range(3)]


def mp_check(npts=2000, seed=7):
    mp.mp.dps = 40
    rng = np.random.default_rng(seed)
    K = rng.uniform(0, math.pi, (npts, 3))
    worst = [0.0, 0.0, 0.0]
    for k in K:
        f64 = integrands(*[np.float64(x) for x in k])
        k1, k2, k3 = [mp.mpf(float(x)) for x in k]
        c1, s1 = mp.cos(k1), mp.sin(k1)
        t = (1 - c1) + (1 - mp.cos(k2)) + (1 - mp.cos(k3))
        s2 = s1 ** 2 + mp.sin(k2) ** 2 + mp.sin(k3) ** 2
        vals = [mp.mpf(0)] * 3
        for m, e in zip(MASSES, ETAS):
            mm = mp.mpf(m)
            M = mm + t / mp.mpf(KAPPA)
            E = mp.sqrt(M * M + s2)
            Eu = mp.sqrt(mm * mm + s2)
            vals[0] += e * 3 * c1 ** 2 / E
            vals[1] += e * (3 * c1 ** 2 - s1 ** 2 / mp.mpf(KAPPA) ** 2) / E
            vals[2] += e * 3 * c1 ** 2 / Eu
        scale = [max(abs(float(v)), 1e-300) for v in vals]
        for c in range(3):
            worst[c] = max(worst[c], abs(float(f64[c]) - float(vals[c])) / scale[c])
    return {"points": npts, "max_rel_diff_sym": worst[0], "max_rel_diff_left": worst[1], "max_rel_diff_unmodified": worst[2]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    res = {"meta": {"kappa": KAPPA, "masses": MASSES, "etas": ETAS, "panels": PANELS}, "gl": [], "midpoint": []}
    names = ("K_N_sym", "k_left", "k_a_unmodified")
    with Pool(a.workers) as pool:
        for n in (16, 32):
            k, w = gl_nodes(n)
            v = run_grid(pool, k, w, a.workers * 16)
            res["gl"].append({"n_per_panel": n, "nodes_per_direction": len(k), **dict(zip(names, v))})
            print("GL n %d (%d nodes per direction): K_N %.15e k_left %.15e k_a %.15e (%.0f s)" % (n, len(k), *v, time.time() - t0), flush=True)
        for L in (1024, 2048):
            h = L // 2
            k = math.pi * (np.arange(h) + 0.5) / h
            w = np.full(h, math.pi / h)
            v = run_grid(pool, k, w, a.workers * 64)
            res["midpoint"].append({"L": L, **dict(zip(names, v))})
            print("midpoint L %d: K_N %.15e k_left %.15e k_a %.15e (%.0f s)" % (L, *v, time.time() - t0), flush=True)
    res["mp_check"] = mp_check()
    print("mp check: %s (%.0f s)" % (res["mp_check"], time.time() - t0), flush=True)
    res["meta"]["seconds"] = time.time() - t0
    with open(os.path.join(a.out, "partFpp_K.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
