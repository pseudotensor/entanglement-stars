#!/usr/bin/env python
"""Strand S, job S1 (SPEC_strandS_graded_entropy.md sec. 2 items 1 and 2, sec. 5 C3 and C4; builder analyst): the chain table S_ch(mu).

The [100] channel of mass mu (the 4-component sine chain gamma^1 sin k + beta mu) is the direct sum of 2 copies of the 2-component chain
sigma_1 sin k + sigma_2 mu, so the 4-component chain's entropy per boundary, S_ch(mu) = entropy(C_4)/2, equals entropy(C_2) of the
2-component interval.  cov2() is the paper's lattice_entropy.interval_covariance with the 2-component symbol (the same twist 0.371, the same
Fourier route, nfft >= 4L); the entropy is the paper's h2 summed over the eigenvalues of the Hermitian part.

Table: mu on a logarithmic grid from --mu-min to --mu-max with --per-decade points per decade; L = min(--L-cap, max(64, ceil(--muL/mu))),
even, nfft = 4L.  The error estimate: the ladder (L, nfft) -> (2L, 2 nfft) at the --ladder masses (both rungs on --gpu if given, one
process, torch eigvalsh in complex128).  The eigh rounding floor n eps |log eps| (n = 2L) is reported.

Fit (SPEC sec. 2.2): with alpha FIXED at 2/3, S_ch + alpha log mu = beta + gamma mu^2 log mu + delta mu^2 + eps mu^4 log mu + zeta mu^4 by least
squares on [mu-min, --fit-hi]; the largest residual (target 1e-9) and the free-alpha refit (control C4: 2/3 within 1e-4).
C3: S_4 = 2 S_2 (entropy of the paper's 4-component interval against twice the 2-component one) at the --c3 masses, to 1e-12.

Writes --out-json (mu, S_ch, L, nfft, the ladder, the fits, C3, C4) and --md.  CPU workers (--workers), one thread each."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.linalg import eigvalsh

ORIENT = os.environ.get("ORIENT_DIR", "<internal>/orient")
sys.path.insert(0, ORIENT)
import lattice_entropy as LE  # noqa: E402

SX = np.array([[0, 1], [1, 0]], complex)
SY = np.array([[0, -1j], [1j, 0]], complex)
I2 = np.eye(2, dtype=complex)
ALPHA = 2.0 / 3.0


def cov2(L, nfft, mu, twist=0.371):
    """the 2-component chain sigma_1 sin k + sigma_2 mu: the interval covariance of length L (the paper's interval_covariance with this symbol)."""
    if nfft < 4 * L:
        raise ValueError("Use nfft >= 4*length")
    th = 2 * np.pi * (np.arange(nfft) + twist) / nfft
    s = np.sin(th)
    E = np.sqrt(mu * mu + s * s)
    H = s[:, None, None] * SX[None] + mu * SY[None]
    P = 0.5 * (I2[None] - H / E[:, None, None])
    K = np.fft.ifft(P, axis=0)
    d = np.arange(L)[:, None] - np.arange(L)[None, :]
    blocks = K[d % nfft] * np.exp(2j * np.pi * twist * d / nfft)[..., None, None]
    return blocks.transpose(0, 2, 1, 3).reshape(2 * L, 2 * L)


def spec(C, gpu=None):
    C = (C + C.conj().T) / 2
    if gpu is None:
        return eigvalsh(C, check_finite=False, overwrite_a=True)
    import torch
    a = torch.as_tensor(C, device="cuda:%d" % gpu)
    x = torch.linalg.eigvalsh(a).cpu().numpy()
    del a
    torch.cuda.empty_cache()
    return x


def S2(L, mu, gpu=None, nfft_mult=4):
    return float(LE.h2(spec(cov2(L, nfft_mult * L, mu), gpu)).sum())


def Lof(mu, muL, Lcap):
    L = min(Lcap, max(64, int(math.ceil(muL / mu))))
    return L + (L % 2)


def job(args):
    mu, muL, Lcap = args
    t = time.time()
    L = Lof(mu, muL, Lcap)
    return mu, L, S2(L, mu), time.time() - t


def ladder_job(args):
    mu, muL, Lcap, gpu = args
    L = Lof(mu, muL, Lcap)
    a = S2(L, mu, gpu)
    b = S2(2 * L, mu, gpu, nfft_mult=4)  # (2L, 2 nfft) with nfft = 4L
    return mu, L, a, b


def c3_job(args):
    mu, muL, Lcap = args
    L = Lof(mu, muL, Lcap)
    s4 = LE.entropy(LE.interval_covariance((1, 0, 0), (0.0, 0.0), L, 4 * L, mass=mu), -1)
    s2 = S2(L, mu)
    return mu, L, s4, s2


def fit(mus, vals, hi, free_alpha=False):
    sel = mus <= hi
    m = mus[sel]
    lm = np.log(m)
    cols = [np.ones_like(m), m * m * lm, m * m, m ** 4 * lm, m ** 4]
    if free_alpha:
        X = np.stack([-lm] + cols, 1)
        y = vals[sel]
    else:
        X = np.stack(cols, 1)
        y = vals[sel] + ALPHA * lm
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = X @ coef - y
    return [float(c) for c in coef], float(np.max(np.abs(res))), int(sel.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mu-min", type=float, default=0.006)
    ap.add_argument("--mu-max", type=float, default=3.0)
    ap.add_argument("--per-decade", type=int, default=200)
    ap.add_argument("--muL", type=float, default=24.0)
    ap.add_argument("--L-cap", type=int, default=4096)
    ap.add_argument("--fit-hi", type=float, default=0.05)
    ap.add_argument("--ladder", default="0.006,0.01,0.03,0.1,0.3,1.0")
    ap.add_argument("--c3", default="0.05,0.1,0.3,1.0,2.0")
    ap.add_argument("--gpu", type=int, default=None, help="the GPU for the ladder's doubled L (one process)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    t0 = time.time()
    n = int(round(math.log10(a.mu_max / a.mu_min) * a.per_decade)) + 1
    mus = np.geomspace(a.mu_min, a.mu_max, n)
    lad_mus = [float(v) for v in a.ladder.split(",")]
    c3_mus = [float(v) for v in a.c3.split(",")]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        fut_c3 = [ex.submit(c3_job, (m, a.muL, a.L_cap)) for m in c3_mus]
        res = list(ex.map(job, [(m, a.muL, a.L_cap) for m in mus], chunksize=1))
        c3 = [f.result() for f in fut_c3]
    print("table: %d points in %.0f s" % (len(res), time.time() - t0), flush=True)
    lad = []
    for m in lad_mus:  # serial: the doubled L on the GPU (one process), the base L on the CPU
        lad.append(ladder_job((m, a.muL, a.L_cap, a.gpu)))
        print("ladder mu %g: L %d, S(L) %.15f, S(2L) %.15f, diff %.3e" % (lad[-1][0], lad[-1][1], lad[-1][2], lad[-1][3], lad[-1][3] - lad[-1][2]), flush=True)
    mus = np.array([r[0] for r in res])
    Ls = np.array([r[1] for r in res])
    vals = np.array([r[2] for r in res])
    secs = np.array([r[3] for r in res])
    coef, resid, nfit = fit(mus, vals, a.fit_hi)
    coef_free, resid_free, _ = fit(mus, vals, a.fit_hi, free_alpha=True)
    eps = np.finfo(float).eps
    floor = 2 * Ls.max() * eps * abs(math.log(eps))
    out = {"meta": vars(a), "mu": mus.tolist(), "S_ch": vals.tolist(), "L": Ls.tolist(), "seconds_per_point": secs.tolist(),
           "ladder": [{"mu": m, "L": L, "S_L": s1, "S_2L": s2, "diff": s2 - s1} for m, L, s1, s2 in lad],
           "fit_fixed_alpha": {"alpha": ALPHA, "beta": coef[0], "gamma": coef[1], "delta": coef[2], "eps": coef[3], "zeta": coef[4],
                               "max_resid": resid, "points": nfit, "range": [a.mu_min, a.fit_hi]},
           "fit_free_alpha": {"alpha": coef_free[0], "beta": coef_free[1], "gamma": coef_free[2], "delta": coef_free[3], "eps": coef_free[4],
                              "zeta": coef_free[5], "max_resid": resid_free},
           "C3": [{"mu": m, "L": L, "S4": s4, "S2": s2, "S4_minus_2S2": s4 - 2 * s2} for m, L, s4, s2 in c3],
           "eigh_floor_estimate": floor, "seconds": time.time() - t0}
    json.dump(out, open(a.out_json, "w"), indent=1)

    def f(v, d=6):
        return "-" if v is None else ("%.*g" % (d, v) if (v == 0 or 1e-3 <= abs(v) < 1e5) else "%.4e" % v)
    L = ["# Strand S job S1: the chain table (strandS_table_v1.py; SPEC_strandS_graded_entropy.md sec. 2)", "",
         "- Grid: %d points of mu from %s to %s (%d per decade), L = min(%d, max(64, ceil(%s/mu))), nfft = 4L; the largest L %d; the eigh rounding "
         "floor estimate n eps |log eps| at n = 2L = %d: %s; %.0f s in all." % (len(mus), f(a.mu_min), f(a.mu_max), a.per_decade, a.L_cap, f(a.muL),
                                                                              int(Ls.max()), 2 * int(Ls.max()), f(floor, 3), out["seconds"]),
         "- The ladder (L, nfft) -> (2L, 2 nfft): " + "; ".join("mu %s: L %d, S(2L) - S(L) = %s" % (f(r["mu"]), r["L"], f(r["diff"], 3)) for r in out["ladder"]) + ".",
         "- The fit with alpha fixed at 2/3 on [%s, %s] (%d points): beta %s, gamma %s, delta %s, eps %s, zeta %s; the largest residual %s (target 1e-9)." % (
             f(a.mu_min), f(a.fit_hi), nfit, f(coef[0], 12), f(coef[1], 10), f(coef[2], 10), f(coef[3], 8), f(coef[4], 8), f(resid, 3)),
         "- C4, the free-alpha refit: alpha %s (2/3 = %s, difference %s), the largest residual %s." % (
             f(coef_free[0], 10), f(ALPHA, 10), f(coef_free[0] - ALPHA, 3), f(resid_free, 3)),
         "- C3, S_4 - 2 S_2 (the paper's 4-component interval against twice the 2-component one): " + "; ".join(
             "mu %s (L %d): %s" % (f(r["mu"]), r["L"], f(r["S4_minus_2S2"], 3)) for r in out["C3"]) + "."]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
