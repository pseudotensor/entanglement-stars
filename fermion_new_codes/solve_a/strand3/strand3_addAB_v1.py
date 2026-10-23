#!/usr/bin/env python
"""Strand 3 addendum, parts A and B (spec solve/specs/SPEC_strand3_addendum.md; builder analyst; scouting, not the model).

A: eta_100 for M_kappa(k) = q + kappa^-1 sum_i (1 - cos k_i) (lattice_entropy_wilson), q = 0, kappa in {0.01, 0.005, 0.003, 0.002},
   L in {64, 128, 256}, nfft in {16 L, 32 L}.
B: eta_100 for M'_kappa'(k) = q + kappa'^-1 sum_i sin^2 k_i (lattice_entropy_sin2), q = 0, kappa' in {1, 0.3, 0.1, 0.03, 0.01},
   L in {64, 128}, nfft = 16 L; the cone check E/|k| - 1 at |k| = kappa'/4, kappa'/2 along [100] and the mass at the 8 valley centres.
Transverse quadrature (both parts): inner square |k_perp,i| <= K_1 = (pi/72) min(12, max(1, round(12 kappa 72/pi))) on 48 x 48 midpoint
points; middle: the 24 x 24 midpoint grid of [-pi/6, pi/6]^2 (cell pi/72) without the inner square's cells; outer: the standard 12 x 12
grid points (eta.value's 2 pi (n + 1/2)/12) outside the pi/6 square (140).  Weights by area ((2 K_1)^2/(48^2 (2 pi)^2), (pi/72)^2/(2 pi)^2,
1/144); their sum is checked against 1.  eta = sum_w w S_channel/(2 |p|), |p| = 1.  Writes <out>/addA.json or addB.json."""
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
from lattice_entropy import entropy, basis, GAMMA, BETA, I4  # noqa: E402
from lattice_entropy_wilson import interval_covariance_wilson  # noqa: E402

P100 = (1, 0, 0)


def interval_covariance_sin2(p, q, length, nfft, mass=0., kappa=np.inf, twist=.371):
    """lattice_entropy.interval_covariance with the mass replaced by M'(k) = q + kappa^-1 sum_i sin^2 k_i (same k array)."""
    p, _, A = basis(p)
    th = 2 * np.pi * (np.arange(nfft) + twist) / nfft
    k = np.asarray(q) @ A[:2] + th[:, None] * p
    s = np.sin(k)
    M = mass + np.sum(s * s, axis=1) / kappa
    E = np.sqrt(M * M + np.sum(s * s, axis=1))
    if E.min() < 1e-14:
        raise ValueError("Quadrature hit a node; change twist/grid")
    H = np.einsum("ki,iab->kab", s, GAMMA) + M[:, None, None] * BETA
    P = .5 * (I4[None] - H / E[:, None, None])
    K = np.fft.ifft(P, axis=0)
    d = np.arange(length)[:, None] - np.arange(length)[None, :]
    blocks = K[d % nfft] * np.exp(2j * np.pi * twist * d / nfft)[..., None, None]
    return blocks.transpose(0, 2, 1, 3).reshape(4 * length, 4 * length)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def quadrature(kappa):
    cell = np.pi / 72
    nin = int(min(12, max(1, round(12 * kappa / cell))))
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
            if abs(wrap(a)) > np.pi / 6 + 1e-12 or abs(wrap(b)) > np.pi / 6 + 1e-12:
                outer.append(((a, b), 1.0 / 144))
    return {"K1": K1, "n_cells": nin, "inner": inner, "middle": mid, "outer": outer}


def chan(args):
    fam, qq, L, nfft, kappa = args
    if fam == "A":
        C = interval_covariance_wilson(P100, qq, L, nfft, 0.0, kappa)
    else:
        C = interval_covariance_sin2(P100, qq, L, nfft, 0.0, kappa)
    return entropy(C) / 2.0


def run(fam, kappas, Ls, nfft_mult, workers, t0):
    rows = []
    with Pool(workers) as pool:
        for kappa in kappas:
            Q = quadrature(kappa)
            wsum = sum(w for part in ("inner", "middle", "outer") for _, w in Q[part])
            for L in Ls:
                for mult in nfft_mult:
                    nfft = mult * L
                    r = {"kappa": kappa, "L": L, "nfft": nfft, "K1": Q["K1"], "n_cells_inner_side": 2 * Q["n_cells"],
                         "n_inner": len(Q["inner"]), "n_middle": len(Q["middle"]), "n_outer": len(Q["outer"]),
                         "weight_sum_minus_1": wsum - 1.0}
                    for part in ("inner", "middle", "outer"):
                        S = pool.map(chan, [(fam, qq, L, nfft, kappa) for qq, w in Q[part]], chunksize=8)
                        r["eta_" + part] = float(sum(w * s for (qq, w), s in zip(Q[part], S)))
                    r["eta_total"] = r["eta_inner"] + r["eta_middle"] + r["eta_outer"]
                    rows.append(r)
                    print("%s kappa %g L %d nfft %d: inner %.6e middle %.6e outer %.6e total %.6e (%.0f s)" % (
                        fam, kappa, L, nfft, r["eta_inner"], r["eta_middle"], r["eta_outer"], r["eta_total"], time.time() - t0), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", choices=["A", "B"], required=True)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    if a.part == "A":
        rows = run("A", (0.01, 0.005, 0.003, 0.002), (64, 128, 256), (16, 32), a.workers, t0)
        extra = {"targets": {"flat_per_sea": 2.5e-5, "static_per_sea": 4e-4}}
    else:
        kps = (1.0, 0.3, 0.1, 0.03, 0.01)
        rows = run("B", kps, (64, 128), (16,), a.workers, t0)
        cone = []
        for kp in kps:
            for frac in (0.25, 0.5):
                k = kp * frac
                M = np.sin(k) ** 2 / kp
                cone.append({"kappa": kp, "k": k, "E_over_k_minus_1": float(np.sqrt(M * M + np.sin(k) ** 2) / k - 1)})
        valley = [{"kappa": kp, "valley": v, "M_at_centre": float(sum(np.sin(c) ** 2 for c in v) / kp)}
                  for kp in kps for v in [(x, y, z) for x in (0, np.pi) for y in (0, np.pi) for z in (0, np.pi)]]
        power = []
        for L in (64, 128):
            rr = sorted([r for r in rows if r["L"] == L], key=lambda r: -r["kappa"])
            power += [{"L": L, "kappa_pair": [x["kappa"], y["kappa"]],
                       "p": float(np.log(y["eta_total"] / x["eta_total"]) / np.log(y["kappa"] / x["kappa"]))} for x, y in zip(rr[:-1], rr[1:])]
        extra = {"cone": cone, "valley_mass_max": max(abs(v["M_at_centre"]) for v in valley), "secant_powers": power}
    with open(os.path.join(a.out, "add%s.json" % a.part), "w") as fh:
        json.dump({"meta": {"host": socket.gethostname(), "seconds": time.time() - t0, "part": a.part}, "rows": rows, **extra}, fh, indent=1)
    print("part %s done in %.0f s" % (a.part, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
