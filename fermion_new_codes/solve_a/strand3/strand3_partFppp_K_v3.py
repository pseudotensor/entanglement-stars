#!/usr/bin/env python
"""Strand 3 addendum part F''' stage 1 (spec solve/specs/SPEC_strand3_addendum.md, part F'''; builder analyst): the q = 0 coefficients of the
PHYSICAL inventory (the massless light packet and the heavies mu, mu, sqrt2 mu, mu = 0.0130903478 (J2e's m_1); orientations (+, -, -, +)) with
the Wilson-type mass M_kappa,s(k) = m_s + kappa^-1 sum_i (1 - cos k_i), E_s = (M^2 + sum_i sin^2 k_i)^(1/2), at kappa = 0.0023 and 0.001, and the
unmodified inventory (kappa = infinity) as the control:
  K_N = 3 sum_s eta_s <cos^2 k_1/E_s>,  k_left(0) = sum_s eta_s <[3 cos^2 k_1 - kappa^-2 sin^2 k_1]/E_s>,  and the massless packet alone
  c_0(kappa) = 3 <cos^2 k_1/E_0>,  <.> the zone average, the packets summed inside the integrand.
Quadratures on [0, pi]^3 (the integrands are even in each k_i; the massless packet's 1/E is singular at the origin, and at kappa = infinity
at every zone corner), the pair k_2 <= k_3 (weight 2 off the diagonal):
  (a) composite Gauss-Legendre: the specification's panels [0, 0.002, 0.006, 0.02, 0.06, 0.2, 0.6, pi] with 32 nodes per panel (primary)
      and 16 (the resolution check), and the graded variant [0, 1e-5, 1e-4, 1e-3, 0.002, 0.006, 0.02, 0.06, 0.2, 0.6, pi] with 32 (the
      origin-refinement check);
  (b) the midpoint grid pi (j + 1/2)/(L/2) (the full-zone midpoint grid of L points per direction, folded by evenness; no node at a zone
      corner, so the origin and corner cells enter only through their midpoints), L = 2048 and 4096, and the one-step Richardson in 1/L^2,
      (4 V(4096) - V(2048))/3 (the singular corners' leading error is O(1/L^2), as J2d found for the twisted grid).
The 40-digit check of the summed integrands (float64 against mpmath) at 2000 random points per finite kappa.  Controls (kappa = infinity):
k_a(0; phys) = 7.2222462613476e-5 and c_0 = 1.5382631884368 (J2e).  Writes <out>/partFppp_K.json.
(version 2: the Gauss-Legendre panels also refined toward pi (the specification's panels mirrored about pi/2), 32 and 16 nodes per panel:
at kappa = infinity the massless packet's 1/E is singular at every zone corner (the doublers at pi), which version 1's panels did not
resolve (its kappa = infinity control missed J2e's c_0 by 1.3e-6, while its midpoint Richardson reproduced it to 3.5e-13); at finite kappa
the corners at pi are gapped (the mass 2/kappa).  --only-gl runs only the Gauss-Legendre rules.)
(version 3, the lead's ruling of 15:34 UTC (JUDGMENT_strand3_addendum.md sec. 12): stage 1 extended to the kappa given by --kappas (default
3e-4 and 1e-4) with the same rules and controls, plus the builder's additions, stated: (i) the Gauss-Legendre rule on version 2's panels
with the breakpoints 0.2 kappa, 2 kappa and 10 kappa inserted (the specification's innermost breakpoint 0.002 lies above 2 kappa at these
kappa), 32 nodes per panel; (ii) the 40-digit check also at up to 2000 node triples of that rule with every coordinate below 10 kappa (the
near-origin nodes; the red team of the part F'' judgment, sec. 4).  The midpoint grids' spacing 2 pi/L exceeds 2 kappa at these kappa;
they are reported as they come.)"""
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

MU = 0.013090347820517666
MASSES = [0.0, MU, MU, math.sqrt(2.0) * MU]
ETAS = [1, -1, -1, 1]
KAPPAS = [0.0003, 0.0001]
PANELS_SPEC = [0.0, 0.002, 0.006, 0.02, 0.06, 0.2, 0.6, math.pi]
PANELS_GRADED = [0.0, 1e-5, 1e-4, 1e-3, 0.002, 0.006, 0.02, 0.06, 0.2, 0.6, math.pi]
PANELS_MIRRORED = [0.0, 0.002, 0.006, 0.02, 0.06, 0.2, 0.6, math.pi / 2] + [math.pi - x for x in (0.6, 0.2, 0.06, 0.02, 0.006, 0.002, 0.0)]
J2E = {"k_a_phys": 7.2222462613476e-5, "c0": 1.5382631884368}


def panels_kappa(kappa):
    left = sorted(set([x for x in PANELS_MIRRORED if x <= math.pi / 2] + [0.2 * kappa, 2 * kappa, 10 * kappa]))
    return left + [math.pi - x for x in reversed(left) if x < math.pi / 2]


def integrands(k1, k2, k3, kappa):
    c1 = np.cos(k1)
    s1 = np.sin(k1)
    t = (1 - c1) + (1 - np.cos(k2)) + (1 - np.cos(k3))
    s2 = s1 * s1 + np.sin(k2) ** 2 + np.sin(k3) ** 2
    inv = 0.0 if math.isinf(kappa) else 1.0 / kappa
    sym = 0.0
    left = 0.0
    c0 = None
    for m, e in zip(MASSES, ETAS):
        M = m + t * inv
        E = np.sqrt(M * M + s2)
        sym = sym + e * 3 * c1 * c1 / E
        left = left + e * (3 * c1 * c1 - s1 * s1 * inv * inv) / E
        if m == 0.0:
            c0 = 3 * c1 * c1 / E
    return sym, left, c0


def gl_nodes(panels, n):
    x, w = np.polynomial.legendre.leggauss(n)
    xs, ws = [], []
    for a, b in zip(panels[:-1], panels[1:]):
        xs.append(0.5 * (b - a) * x + 0.5 * (b + a))
        ws.append(0.5 * (b - a) * w)
    return np.concatenate(xs), np.concatenate(ws)


def pair_task(args):
    k1, w1, k2, k3, wp, kappa = args
    vals = integrands(k1[:, None], k2[None, :], k3[None, :], kappa)
    W = w1[:, None] * wp[None, :]
    return [math.fsum((W * v).ravel()) for v in vals]


def run_grid(pool, k, w, kappa, nch):
    i, j = np.triu_indices(len(k))
    wp = w[i] * w[j] * np.where(i == j, 1.0, 2.0)
    bounds = np.linspace(0, len(i), nch + 1).astype(int)
    tasks = [(k, w, k[i[a:b]], k[j[a:b]], wp[a:b], kappa) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
    parts = pool.map(pair_task, tasks, chunksize=1)
    return [math.fsum(p[c] for p in parts) / math.pi ** 3 for c in range(3)]


def mp_check(kappa, npts=2000, seed=7, near=False):
    mp.mp.dps = 40
    rng = np.random.default_rng(seed)
    if near:
        k, _ = gl_nodes(panels_kappa(kappa), 32)
        kn = k[k < 10 * kappa]
        K = kn[rng.integers(0, len(kn), (npts, 3))]
    else:
        K = rng.uniform(0, math.pi, (npts, 3))
    worst = [0.0, 0.0]
    kap = mp.mpf(kappa)
    for k in K:
        f64 = integrands(*[np.float64(x) for x in k], kappa)
        k1, k2, k3 = [mp.mpf(float(x)) for x in k]
        c1, s1 = mp.cos(k1), mp.sin(k1)
        t = (1 - c1) + (1 - mp.cos(k2)) + (1 - mp.cos(k3))
        s2 = s1 ** 2 + mp.sin(k2) ** 2 + mp.sin(k3) ** 2
        vals = [mp.mpf(0), mp.mpf(0)]
        for m, e in zip(MASSES, ETAS):
            M = mp.mpf(m) + t / kap
            E = mp.sqrt(M * M + s2)
            vals[0] += e * 3 * c1 ** 2 / E
            vals[1] += e * (3 * c1 ** 2 - s1 ** 2 / kap ** 2) / E
        for c in range(2):
            worst[c] = max(worst[c], abs(float(f64[c]) - float(vals[c])) / max(abs(float(vals[c])), 1e-300))
    return {"points": npts, "max_rel_diff_sym": worst[0], "max_rel_diff_left": worst[1]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only-gl", action="store_true")
    ap.add_argument("--kappas", default="0.0003,0.0001")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    names = ("K_N_sym", "k_left", "c0_massless")
    res = {"meta": {"masses": MASSES, "etas": ETAS, "kappas": ["inf" if math.isinf(k) else k for k in KAPPAS], "panels_spec": PANELS_SPEC,
                    "panels_graded": PANELS_GRADED, "panels_mirrored": PANELS_MIRRORED, "J2e": J2E, "version": 3, "only_gl": a.only_gl}, "rows": []}
    kappas = [float(x) for x in a.kappas.split(",")]
    res["meta"]["kappas"] = kappas
    res["meta"]["panels_kappa"] = {str(k): panels_kappa(k) for k in kappas}
    out = os.path.join(a.out, "partFppp_K.json")
    with Pool(a.workers) as pool:
        for kappa in kappas:
            kl = "inf" if math.isinf(kappa) else kappa
            for qname, panels, n in (("GL spec panels, 32", PANELS_SPEC, 32), ("GL spec panels, 16", PANELS_SPEC, 16), ("GL graded panels, 32", PANELS_GRADED, 32),
                                     ("GL mirrored panels, 32", PANELS_MIRRORED, 32), ("GL mirrored panels, 16", PANELS_MIRRORED, 16),
                                     ("GL kappa-adapted mirrored panels, 32", panels_kappa(kappa), 32)):
                k, w = gl_nodes(panels, n)
                v = run_grid(pool, k, w, kappa, a.workers * 16)
                res["rows"].append({"kappa": kl, "quadrature": qname, "nodes_per_direction": len(k), **dict(zip(names, v))})
                print("kappa %s %s (%d nodes): K_N %.15e k_left %.15e c0 %.15e (%.0f s)" % (kl, qname, len(k), *v, time.time() - t0), flush=True)
                json.dump(res, open(out, "w"), indent=1)
            if a.only_gl:
                continue
            mids = {}
            for L in (2048, 4096):
                h = L // 2
                k = math.pi * (np.arange(h) + 0.5) / h
                w = np.full(h, math.pi / h)
                v = run_grid(pool, k, w, kappa, a.workers * 64)
                mids[L] = v
                res["rows"].append({"kappa": kl, "quadrature": "midpoint L = %d" % L, **dict(zip(names, v))})
                print("kappa %s midpoint L %d: K_N %.15e k_left %.15e c0 %.15e (%.0f s)" % (kl, L, *v, time.time() - t0), flush=True)
                json.dump(res, open(out, "w"), indent=1)
            rich = [(4 * mids[4096][c] - mids[2048][c]) / 3 for c in range(3)]
            res["rows"].append({"kappa": kl, "quadrature": "midpoint Richardson (4 V(4096) - V(2048))/3", **dict(zip(names, rich))})
            print("kappa %s midpoint Richardson: K_N %.15e k_left %.15e c0 %.15e (%.0f s)" % (kl, *rich, time.time() - t0), flush=True)
            if not math.isinf(kappa):
                res.setdefault("mp_check", {})[str(kl)] = mp_check(kappa)
                print("kappa %s mp check %s" % (kl, res["mp_check"][str(kl)]), flush=True)
                res.setdefault("mp_check_near_origin", {})[str(kl)] = mp_check(kappa, near=True)
                print("kappa %s mp check near origin %s" % (kl, res["mp_check_near_origin"][str(kl)]), flush=True)
            json.dump(res, open(out, "w"), indent=1)
    res["meta"]["seconds"] = time.time() - t0
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
