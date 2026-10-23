#!/usr/bin/env python
"""Strand 3 addendum part F''' addition (a) (spec solve/specs/SPEC_strand3_addendum.md, "Part F''(b) and F''' additions"; builder analyst):
the lower bound of the modified massless light sea's cut-entropy coefficient from h(t) >= 4 log 2 t (1 - t), i.e. per channel
S >= 4 log 2 Tr[C_R (1 - C_R)], with C_R the interval covariance on L sites (strand3_partFppp_eta_v1's channels: lattice_entropy_wilson's
covariance, the longitudinal twist 0.371, mass 0, M_kappa = kappa^-1 sum (1 - cos k_i)).  No eigendecomposition: C_R is block Toeplitz with
4 x 4 blocks K(d) times unit phases, so Tr C_R = L Tr K(0) and Tr C_R^2 = sum_{|d| < L} (L - |d|) ||K(d)||_F^2, K = the inverse FFT of the
projector over nfft longitudinal points.  eta_bound = sum_w w 4 log 2 Tr[C_R (1 - C_R)]/2 (the same 1/(2 |p|) as eta) on stage 2's
production quadrature (inner square of side 8 kappa on 64 x 64, the graded annulus to pi/72, the 48 x 48 middle, the 24 x 24 outer; D4
orbits), at kappa = 0.0023 and 0.001, L = 256 and 512 (nfft 32 L) and, since it is cheap, L = 1024 and 2048; beside it pi K_N/2 (K_N from
stage 1's primary rule, read from the record given) and, where stage 2's eta_light exists at the same (kappa, L, nfft), the ratio
eta_bound/eta_light.  Per kappa: the check at 64 random channels of the bound against the eigenvalue form 4 log 2 sum t (1 - t) (same
covariance; the identity, to rounding).  Writes <out>/partFppp_bound.json.
(version 2, part F'''(d): the quadrature of strand3_partFppp_eta_v2 (the annulus's geometric cells scaled with log(b/a)); --kappas; K_N for
each kappa from the stage-1 records given (comma-separated), the primary rule the kappa-adapted panels where the record has them, else
the panels refined toward 0 and pi.)"""
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
from lattice_entropy import basis, GAMMA, BETA, I4  # noqa: E402
from lattice_entropy_wilson import interval_covariance_wilson  # noqa: E402
import strand3_partFppp_eta_v2 as S2  # noqa: E402

P100 = (1, 0, 0)
FOUR_LOG2 = 4 * math.log(2.0)


def blocks_K(q, nfft, kappa, twist=0.371):
    """The 4 x 4 blocks K(d), d = 0..nfft-1, of the massless modified sea's covariance (lattice_entropy_wilson's construction)."""
    p, _, A = basis(P100)
    th = 2 * np.pi * (np.arange(nfft) + twist) / nfft
    k = np.asarray(q) @ A[:2] + th[:, None] * p
    s = np.sin(k)
    M = np.sum(1 - np.cos(k), axis=1) / kappa
    E = np.sqrt(M * M + np.sum(s * s, axis=1))
    H = np.einsum("ki,iab->kab", s, GAMMA) + M[:, None, None] * BETA
    P = .5 * (I4[None] - H / E[:, None, None])
    return np.fft.ifft(P, axis=0)


def bound_channel(args):
    q, L, nfft, kappa = args
    K = blocks_K(q, nfft, kappa)
    f2 = np.sum(np.abs(K) ** 2, axis=(1, 2))
    d = np.arange(1, L)
    trC = L * float(np.real(np.trace(K[0])))
    trC2 = L * float(f2[0]) + float(np.sum((L - d) * (f2[d] + f2[(-d) % nfft])))
    return FOUR_LOG2 * (trC - trC2) / 2.0


def eig_channel(args):
    q, L, nfft, kappa = args
    C = interval_covariance_wilson(P100, q, L, nfft, 0.0, kappa)
    t = np.linalg.eigvalsh((C + C.conj().T) / 2)
    return FOUR_LOG2 * float(np.sum(t * (1 - t))) / 2.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--K", required=True, help="stage 1's record with the primary rule at kappa = 0.0023 and 0.001")
    ap.add_argument("--eta", default=None, help="stage 2's partFppp_eta.json (for the ratio)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--Ls", default="256,512,1024,2048")
    ap.add_argument("--kappas", default="1e-4,3e-5,1e-5")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    KN, KQ = {}, {}
    for p in a.K.split(","):
        for r in json.load(open(p))["rows"]:
            if r["kappa"] == "inf":
                continue
            kq = float(r["kappa"])
            if r["quadrature"] == "GL kappa-adapted mirrored panels, 32" or (r["quadrature"] == "GL mirrored panels, 32" and KQ.get(kq) != "kappa-adapted"):
                KN[kq] = r["K_N_sym"]
                KQ[kq] = "kappa-adapted" if "adapted" in r["quadrature"] else "mirrored"
    E = json.load(open(a.eta)) if a.eta else None
    kappas = [float(x) for x in a.kappas.split(",")]
    res = {"meta": {"K_N": {str(k): v for k, v in KN.items()}, "K_N_rule": {str(k): v for k, v in KQ.items()}, "Ls": a.Ls, "version": 2,
                    "n_geo": {str(k): S2.n_geo(4 * k) for k in kappas}}, "rows": [], "eig_check": []}
    out = os.path.join(a.out, "partFppp_bound.json")
    rng = np.random.default_rng(11)
    with Pool(a.workers) as pool:
        for kappa in kappas:
            qs = [tuple(x) for x in rng.uniform(-0.5, 0.5, (64, 2)) * np.array([1, 1]) * rng.choice([1e-2, 1e-1, 1.0], (64, 1))]
            b = pool.map(bound_channel, [(q, 256, 32 * 256, kappa) for q in qs], chunksize=1)
            e = pool.map(eig_channel, [(q, 256, 32 * 256, kappa) for q in qs], chunksize=1)
            dev = max(abs(x - y) / max(abs(y), 1e-300) for x, y in zip(b, e))
            res["eig_check"].append({"kappa": kappa, "channels": len(qs), "L": 256, "max_rel_diff": dev})
            print("kappa %g: Fourier-sum bound against the eigenvalue form at 64 channels: max rel diff %.2e (%.0f s)" % (kappa, dev, time.time() - t0), flush=True)
            for L in [int(x) for x in a.Ls.split(",")]:
                regs = {}
                for which in ("inner", "annulus", "middle", "outer"):
                    orb = S2.orbits(S2.rule(kappa, which))
                    v = pool.map(bound_channel, [(q, L, 32 * L, kappa) for q, w in orb], chunksize=4)
                    regs[which] = math.fsum(w * x for (q, w), x in zip(orb, v))
                tot = math.fsum(regs.values())
                row = {"kappa": kappa, "L": L, "nfft": 32 * L, "regions": regs, "eta_bound": tot, "pi_K_N_over_2": math.pi * KN[kappa] / 2,
                       "bound_over_pi_K_N_over_2": tot / (math.pi * KN[kappa] / 2)}
                if E:
                    el = [x for x in E["regions"] if x["sea"] == "light" and x["kappa"] == kappa and x["L"] == L and x["nfft"] == 32 * L
                          and x["region"] in ("inner", "annulus", "middle", "outer")]
                    if len(el) == 4:
                        row["eta_light"] = math.fsum(x["eta"] for x in el)
                        row["bound_over_eta_light"] = tot / row["eta_light"]
                res["rows"].append(row)
                print("kappa %g L %d: eta_bound %.10e, pi K_N/2 %.10e, ratio %.6g%s (%.0f s)" % (kappa, L, tot, row["pi_K_N_over_2"], row["bound_over_pi_K_N_over_2"],
                      (", bound/eta_light %.6f" % row["bound_over_eta_light"]) if "bound_over_eta_light" in row else "", time.time() - t0), flush=True)
                json.dump(res, open(out, "w"), indent=1)
    res["meta"]["seconds"] = time.time() - t0
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
