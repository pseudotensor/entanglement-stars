#!/usr/bin/env python
"""Strand 6 (b) (spec solve/specs/SPEC_strand6_validation.md (b); builder analyst): the Omega-law cut entropy on the finite sine lattice,
the axis-normal cut [100].

Law (lane T10 (T25)) on the uniform ruler P = 1, lattice units a = 1: per sea s the Dirac projector of H(k) = sum_i sin k_i gamma_i + R_s(k) beta
(the strand 3 addendum's lattice_entropy_wilson route with the mass M(k) -> R_s(k)), R_s(k) = sqrt(m_s^2 + T(k)^4/(a Omega)^6) (order four) or
sqrt(m_s^2 + T(k)^6/(a Omega)^10) (order six), T(k) = sum_i sin^2 k_i; the pilot (lane, fixed before the count): a Omega = 1/250,
m_s^2/Omega^2 = (1e-6, 0.5000005, 0.5000005, 1); the entropies of the four seas add; S/(A Omega^2) = eta_total/(a Omega)^2.
Per transverse channel q the interval of length L along [100] (the orient module's basis and twist 0.371, nfft = mult L) gives the 4L x 4L
covariance of the addendum; at fixed q the three matrices gamma_1, the unit transverse gamma (s_2 gamma_2 + s_3 gamma_3)/|s_perp| and beta
anticommute, so the constant Hermitian involution G = i gamma_1 gamma_perp beta commutes with H(k) for every k_1 and splits the covariance
into two 2L x 2L blocks (G = +1, -1); the channel entropy entropy(C_4L)/2 (the addendum's, two cuts) equals the mean of the two blocks'
entropies (checked here at L = 32 against the 4L route and between the blocks).  The channel entropy depends on q only through
sigma^2 = sin^2 k_2 + sin^2 k_3 (a rotation about axis 1 is a symmetry of gamma_1, beta and T), so channels are evaluated once per distinct
sigma^2 (13 digits) with their summed weights.
Transverse quadrature (the addendum's nesting): inner square |k_perp,i| <= K_1 on 48 x 48 midpoint points, K_1 = (pi/72) max(1,
round(max(12 m_4 a, a Omega)/(pi/72))) (the spec's max(12 x 4e-3, a Omega) rounded to the 24-grid at the pilot); middle: the 24 x 24
midpoint grid of [-pi/6, pi/6]^2 without the inner cells; outer: the 12 x 12 points 2 pi (n + 1/2)/12 outside the pi/6 square; weights by
area; eta = sum_w w S_channel with S_channel = entropy(C_4L)/2 (the addendum's chan value; |p| = 1).  Writes <out>/omega_<tag>.json per setting, or <out>/check.json in --check mode."""
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
from scipy.linalg import eigvalsh

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(HERE, "..", "strand3"), os.path.join(HERE, "..", "strand3", "orient"), os.path.join(HERE, "orient"), HERE):
    sys.path.insert(0, _p)
from lattice_entropy import basis, GAMMA, BETA, I4, h2  # noqa: E402

P100 = (1, 0, 0)


def masses(aO, ratios=(1e-6, 0.5000005, 0.5000005, 1.0)):
    return [aO * np.sqrt(r) for r in ratios]


def k_array(q, nfft, twist):
    p, _, A = basis(P100)
    th = 2 * np.pi * (np.arange(nfft) + twist) / nfft
    return np.asarray(q) @ A[:2] + th[:, None] * p, p


def projector(k, m, aO, order):
    s = np.sin(k)
    T = np.sum(s * s, axis=1)
    if order == 4:
        R = np.sqrt(m * m + T ** 4 / aO ** 6)
    elif order == 6:
        R = np.sqrt(m * m + T ** 6 / aO ** 10)
    else:
        raise ValueError(order)
    E = np.sqrt(R * R + T)
    if E.min() < 1e-14:
        raise ValueError("Quadrature hit a node; change twist/grid")
    H = np.einsum("ki,iab->kab", s, GAMMA) + R[:, None, None] * BETA
    return 0.5 * (I4[None] - H / E[:, None, None]), s


def blocks_of(P, L, nfft, twist):
    K = np.fft.ifft(P, axis=0)
    d = np.arange(L)[:, None] - np.arange(L)[None, :]
    return K[d % nfft] * np.exp(2j * np.pi * twist * d / nfft)[..., None, None]


def cov4(q, L, nfft, m, aO, order, twist=.371):
    k, _ = k_array(q, nfft, twist)
    P, _ = projector(k, m, aO, order)
    B = blocks_of(P, L, nfft, twist)
    return B.transpose(0, 2, 1, 3).reshape(4 * L, 4 * L)


def split_unitary(q, twist=.371):
    k, p = k_array(q, 4, twist)
    s = np.sin(k[0])
    # transverse components: the directions other than p's
    perp = [i for i in range(3) if p[i] == 0]
    s2, s3 = s[perp[0]], s[perp[1]]
    sp = np.hypot(s2, s3)
    n2, n3 = (s2 / sp, s3 / sp) if sp > 1e-14 else (1.0, 0.0)
    gam1 = GAMMA[[i for i in range(3) if p[i] != 0][0]]
    gperp = n2 * GAMMA[perp[0]] + n3 * GAMMA[perp[1]]
    G = 1j * gam1 @ gperp @ BETA
    G = 0.5 * (G + G.conj().T)
    w, U = np.linalg.eigh(G)
    order_ = np.argsort(-w)                  # +1 first
    return U[:, order_], w[order_]


def cov2_pair(q, L, nfft, m, aO, order, twist=.371):
    k, _ = k_array(q, nfft, twist)
    P, _ = projector(k, m, aO, order)
    U, w = split_unitary(q, twist)
    Pp = np.einsum("ai,kab,bj->kij", U.conj(), P, U)
    off = float(np.max(np.abs(Pp[:, :2, 2:])))
    out = []
    for sl in (slice(0, 2), slice(2, 4)):
        B = blocks_of(Pp[:, sl, sl], L, nfft, twist)
        out.append(B.transpose(0, 2, 1, 3).reshape(2 * L, 2 * L))
    return out, off


def ent(C):
    ev = eigvalsh(0.5 * (C + C.conj().T))
    return float(h2(np.clip(ev, 0, 1)).sum())


def quadrature(aO):
    cell = np.pi / 72
    m4 = masses(aO)[3]
    nin = int(max(1, round(max(12 * m4, aO) / cell)))
    K1 = nin * cell
    pts = []
    for a in -K1 + 2 * K1 * (np.arange(48) + 0.5) / 48:
        for b in -K1 + 2 * K1 * (np.arange(48) + 0.5) / 48:
            pts.append((a, b, (2 * K1) ** 2 / (48 ** 2 * (2 * np.pi) ** 2), "inner"))
    cen = -np.pi / 6 + cell * (np.arange(24) + 0.5)
    for a in cen:
        for b in cen:
            if abs(a) < K1 and abs(b) < K1:
                continue
            pts.append((a, b, cell ** 2 / (2 * np.pi) ** 2, "middle"))
    wrap = lambda x: (x + np.pi) % (2 * np.pi) - np.pi
    for a in 2 * np.pi * (np.arange(12) + 0.5) / 12:
        for b in 2 * np.pi * (np.arange(12) + 0.5) / 12:
            if abs(wrap(a)) > np.pi / 6 + 1e-12 or abs(wrap(b)) > np.pi / 6 + 1e-12:
                pts.append((a, b, 1.0 / 144, "outer"))
    return K1, nin, pts


def sigma2_key(q, twist=.371):
    k, p = k_array(q, 4, twist)
    s = np.sin(k[0])
    perp = [i for i in range(3) if p[i] == 0]
    return round(float(s[perp[0]] ** 2 + s[perp[1]] ** 2), 13)


def chan(args):
    q, L, nfft, m, aO, order = args
    (C1, C2), off = cov2_pair(q, L, nfft, m, aO, order)
    e1 = ent(C1)
    e2 = ent(C2)
    return 0.5 * (e1 + e2), e1, e2, off


def check(out):
    rows = []
    for order in (4, 6):
        for m in masses(1 / 250.0)[:1] + masses(1 / 250.0)[3:]:
            for q in ((0.0123, 0.0456), (0.3, -0.7), (2.1, 1.3)):
                C4 = cov4(q, 32, 512, m, 1 / 250.0, order)
                s4 = ent(C4) / 2
                s, e1, e2, off = chan((q, 32, 512, m, 1 / 250.0, order))
                rows.append({"order": order, "m": m, "q": q, "S4_over_2": s4, "S_blocks_mean": s, "block_plus": e1, "block_minus": e2,
                             "rel_diff": s / s4 - 1 if s4 else None, "offdiag_max": off})
                print(rows[-1], flush=True)
    with open(os.path.join(out, "check.json"), "w") as fh:
        json.dump(rows, fh, indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--aOmega", type=float, default=1 / 250.0)
    ap.add_argument("--L", type=int, default=1024)
    ap.add_argument("--mult", type=int, default=16)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.check:
        check(a.out)
        return
    t0 = time.time()
    aO = a.aOmega
    K1, nin, pts = quadrature(aO)
    wsum = sum(p_[2] for p_ in pts)
    groups = {}
    for (qa, qb, w, part) in pts:
        key = sigma2_key((qa, qb))
        g = groups.setdefault(key, {"q": (qa, qb), "w": 0.0, "parts": {}})
        g["w"] += w
        g["parts"][part] = g["parts"].get(part, 0.0) + w
    keys = sorted(groups)
    ms = masses(aO)
    distinct = [0, 1, 3]          # seas 2 and 3 coincide
    nfft = a.mult * a.L
    res = {"meta": {"host": socket.gethostname(), "order": a.order, "aOmega": aO, "L": a.L, "nfft": nfft, "K1": K1, "inner_cells": nin,
                    "n_points": len(pts), "n_distinct_sigma2": len(keys), "weight_sum_minus_1": wsum - 1.0, "masses": ms,
                    "mass_ratios": [1e-6, 0.5000005, 0.5000005, 1.0]}, "seas": {}}
    with Pool(a.workers) as pool:
        for si in distinct:
            S = pool.map(chan, [(groups[kk]["q"], a.L, nfft, ms[si], aO, a.order) for kk in keys], chunksize=1)
            eta = {"total": 0.0, "inner": 0.0, "middle": 0.0, "outer": 0.0}
            maxblock = 0.0
            maxoff = 0.0
            for kk, (s, e1, e2, off) in zip(keys, S):
                g = groups[kk]
                eta["total"] += g["w"] * s
                for part, w in g["parts"].items():
                    eta[part] += w * s
                maxblock = max(maxblock, abs(e1 - e2))
                maxoff = max(maxoff, off)
            res["seas"][str(si)] = {"m": ms[si], "eta": eta, "max_block_difference": maxblock, "max_offdiag": maxoff,
                                    "channels": [{"sigma2": kk, "w": groups[kk]["w"], "S": s} for kk, (s, *_r) in zip(keys, S)]}
            print("order %d aO %g L %d nfft %d sea %d (m %.3e): eta %.10e (inner %.4e middle %.4e outer %.4e) (%.0f s)" % (
                a.order, aO, a.L, nfft, si, ms[si], eta["total"], eta["inner"], eta["middle"], eta["outer"], time.time() - t0), flush=True)
    eta_tot = res["seas"]["0"]["eta"]["total"] + 2 * res["seas"]["1"]["eta"]["total"] + res["seas"]["3"]["eta"]["total"]
    res["eta_total"] = eta_tot
    res["S_over_A_Omega2"] = eta_tot / aO ** 2
    res["meta"]["seconds"] = time.time() - t0
    tag = "o%d_aO%g_L%d_m%d" % (a.order, 1 / aO, a.L, a.mult)
    with open(os.path.join(a.out, "omega_%s.json" % tag), "w") as fh:
        json.dump(res, fh, indent=1)
    print("S/(A Omega^2) = %.10f (%.0f s)" % (res["S_over_A_Omega2"], time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
