#!/usr/bin/env python
"""Strand 3 part C (spec sec. 5; scouting the high-momentum dispersion door, not the model; builder analyst).

eta_100 for M_kappa(k) = q + kappa^-1 sum_i (1 - cos k_i), kappa in {1e6, 1, 0.3, 0.1, 0.03, 0.01}, q in {0, 0.02}, at
chain length L = 64 (the finest length of the depth-16 ladder; 4L = 256) and nfft = 16 L, with the transverse integral
split: inner square |k_perp,i| <= K = pi j/6 (j = 6 for kappa >= 1, 3 for 0.3, 2 for 0.1, 1 for kappa <= 0.03) on a 24 x 24
midpoint grid, weight (2K)^2/((2 pi)^2 24^2) per channel; outer = the standard 12 x 12 channels (eta.value's
q = 2 pi (n + 1/2)/12, wrapped to (-pi, pi]) outside the square, weight 1/144.  eta = sum_w w S_channel/(2|p|).
Controls: at kappa = 1e6 the split total against eta.value((1,0,0), L, 12, 16 L, q); the weights sum to 1.
C.2 local power d log eta/d log kappa for kappa <= 0.1 and the kappa with 4 eta = 1e-4 (log-log interpolation);
C.3 the cone check E_kappa(k)/|k| - 1 at |k| = kappa/4, kappa/2 along [100], q = 0.  Writes <out>/partC.json.
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import socket
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "orient"))
from lattice_entropy import entropy  # noqa: E402
import eta as E  # noqa: E402
from lattice_entropy_wilson import interval_covariance_wilson  # noqa: E402

KAPPAS = (1e6, 1.0, 0.3, 0.1, 0.03, 0.01)
QS = (0.0, 0.02)
L = 64
NFFT = 16 * L
P100 = (1, 0, 0)


def jval(kappa):
    if kappa >= 1:
        return 6
    if kappa >= 0.3:
        return 3
    if kappa >= 0.1:
        return 2
    return 1


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def channels(kappa):
    K = np.pi * jval(kappa) / 6
    inner, outer = [], []
    fine = -K + 2 * K * (np.arange(24) + 0.5) / 24
    w_in = (2 * K) ** 2 / ((2 * np.pi) ** 2 * 24 ** 2)
    for a in fine:
        for b in fine:
            inner.append(((a, b), w_in))
    coarse = 2 * np.pi * (np.arange(12) + 0.5) / 12
    for a in coarse:
        for b in coarse:
            if abs(wrap(a)) > K + 1e-12 or abs(wrap(b)) > K + 1e-12:
                outer.append(((a, b), 1.0 / 144))
    return inner, outer, K


def chan(args):
    q, kappa, mass = args
    C = interval_covariance_wilson(P100, q, L, NFFT, mass, kappa)
    return entropy(C) / 2.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    res = []
    with Pool(args.workers) as pool:
        for kappa in KAPPAS:
            inner, outer, K = channels(kappa)
            for mass in QS:
                si = pool.map(chan, [(qq, kappa, mass) for qq, w in inner], chunksize=4)
                so = pool.map(chan, [(qq, kappa, mass) for qq, w in outer], chunksize=4) if outer else []
                ei = float(sum(w * s for (qq, w), s in zip(inner, si)))
                eo = float(sum(w * s for (qq, w), s in zip(outer, so)))
                wsum = sum(w for _, w in inner) + sum(w for _, w in outer)
                r = {"kappa": kappa, "q": mass, "j": jval(kappa), "K": K, "n_inner": len(inner), "n_outer": len(outer),
                     "eta_inner": ei, "eta_outer": eo, "eta_total": ei + eo, "weight_sum_minus_1": wsum - 1.0,
                     "M_pi00": mass + 2.0 / kappa}
                if kappa == 1e6:
                    ref = E.value(P100, L, 12, NFFT, mass)
                    r["eta_value_uniform_12x12"] = float(ref)
                    r["split_minus_uniform"] = float(ei + eo - ref)
                res.append(r)
                print("kappa %g q %g: inner %.6e outer %.6e total %.6e (%.0f s)" % (kappa, mass, ei, eo, ei + eo,
                                                                                   time.time() - t0), flush=True)
    power, target = {}, {}
    for mass in QS:
        rows = sorted([r for r in res if r["q"] == mass and r["kappa"] <= 0.1], key=lambda r: -r["kappa"])
        power[str(mass)] = [{"kappa_pair": [a["kappa"], b["kappa"]],
                             "p": float(np.log(b["eta_total"] / a["eta_total"]) / np.log(b["kappa"] / a["kappa"]))}
                            for a, b in zip(rows[:-1], rows[1:])]
        allr = sorted([r for r in res if r["q"] == mass], key=lambda r: r["kappa"])
        y = np.log(np.array([4 * r["eta_total"] for r in allr]))
        x = np.log(np.array([r["kappa"] for r in allr]))
        tgt = np.log(1e-4)
        hit = None
        for i in range(len(allr) - 1):
            if (y[i] - tgt) * (y[i + 1] - tgt) <= 0:
                hit = float(np.exp(x[i] + (tgt - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i])))
                break
        if hit is None:
            i = 0
            hit_ex = float(np.exp(x[i] + (tgt - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i])))
            target[str(mass)] = {"bracketed": False, "kappa_extrapolated_from_two_smallest": hit_ex,
                                 "min_4eta": float(np.exp(y.min()))}
        else:
            target[str(mass)] = {"bracketed": True, "kappa": hit}
    cone = []
    for kappa in KAPPAS:
        for frac in (0.25, 0.5):
            k = kappa * frac
            M = (1 - np.cos(k)) / kappa
            Ek = np.sqrt(M * M + np.sin(k) ** 2)
            cone.append({"kappa": kappa, "k": k, "E_over_k_minus_1": float(Ek / k - 1)})
    with open(os.path.join(args.out, "partC.json"), "w") as fh:
        json.dump({"meta": {"host": socket.gethostname(), "seconds": time.time() - t0, "L": L, "nfft": NFFT},
                   "rows": res, "local_power": power, "target_kappa": target, "cone": cone}, fh, indent=1)


if __name__ == "__main__":
    main()
