#!/usr/bin/env python
"""Strand 3 part F, high-precision control (builder analyst): the float64 dense control of strand3_partF_v1 reaches an absolute rounding
floor at small kappa (the traces Tr Psi E are of order 2 L^3 <M> while W^(2) eps^2 is small), so the symmetric placement's small W^(2) is
checked here without that floor: the dense real-space W(eps) of one Wilson-mass packet on the periodic L x Lp x Lp box (L = 8, Lp = 3,
twist 0, m = 0.1, q = 2 pi/L along axis 1) in mpmath at 40 digits (A built from M = m + (1/(2 kappa))(-Delta_lattice) with the placement
M Psi^-1 or Psi^-1/2 M Psi^-1/2, and i Psi^-3 D; W(eps) = Tr Psi (E_1 - E_0) by mpmath.eigh), eps in {1e-3, 5e-4, 2.5e-4} with both signs,
D(eps) = (W(eps) + W(-eps) - 2 W(0))/eps^2 extrapolated twice (R1(e) = (4 D(e/2) - D(e))/3, R2 = (16 R1(e/2) - R1(e))/15), against
strand3_partF_v1.packet_terms_M on the same box (its k_1 grid of L points and the Lp x Lp transverse classes); kappa in {0.3, 0.01, 0.002},
both placements, one process per case.  Writes <out>/control_mp.json."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
from multiprocessing import Pool

import mpmath as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import strand3_partF_v1 as PF  # noqa: E402

EPS_MP = ("1e-3", "5e-4", "2.5e-4")


def dense_mp(L, Lp, m, kappa, n, placement, dps=40):
    mp.mp.dps = dps
    N3 = L * Lp * Lp
    ar = np.arange(N3)
    pos = np.stack([ar % L, (ar // L) % Lp, ar // (L * Lp)], 1)
    dims = (L, Lp, Lp)
    S = [[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]]
    dim = 2 * N3
    Dw = mp.matrix(dim, dim)
    Lap = mp.matrix(N3, N3)
    half = mp.mpf(1) / 2
    for x in range(N3):
        for j in range(3):
            for sg in (1, -1):
                p = pos[x].copy()
                p[j] = (p[j] + sg) % dims[j]
                y = int(p[0] + L * p[1] + L * Lp * p[2])
                for a in range(2):
                    for b in range(2):
                        if S[j][a][b] != 0:
                            Dw[2 * x + a, 2 * y + b] += sg * mp.mpc(S[j][a][b]) / mp.mpc(0, 2)
                Lap[x, y] += -half
        Lap[x, x] += 3
    kap = mp.mpf(kappa)
    mm = mp.mpf(m)
    Mop = mp.matrix(N3, N3)
    for x in range(N3):
        for y in range(N3):
            Mop[x, y] = Lap[x, y] / kap
        Mop[x, x] += mm
    q = 2 * mp.pi * n / L
    W = {}
    for es in ["0"] + [s + e for e in EPS_MP for s in ("", "-")]:
        e = mp.mpf(es)
        psi = [1 + e * mp.cos(q * int(pos[x, 0])) for x in range(N3)]
        A = mp.matrix(dim, dim)
        for x in range(N3):
            for y in range(N3):
                if Mop[x, y] != 0:
                    if placement == "left":
                        v = Mop[x, y] / psi[y]
                    else:
                        v = Mop[x, y] / mp.sqrt(psi[x] * psi[y])
                    A[2 * x, 2 * y] += v
                    A[2 * x + 1, 2 * y + 1] += v
        for i in range(dim):
            f = psi[i // 2] ** -3
            for jj in range(dim):
                if Dw[i, jj] != 0:
                    A[i, jj] += mp.mpc(0, 1) * f * Dw[i, jj]
        AH = A.H
        tr = []
        for X in (AH * A, A * AH):
            E, Q = mp.eighe(X) if hasattr(mp, "eighe") else mp.eigh(X)
            t = mp.mpf(0)
            for jj in range(dim):
                sj = mp.sqrt(max(E[jj], mp.mpf(0)))
                for i in range(dim):
                    t += psi[i // 2] * abs(Q[i, jj]) ** 2 * sj
            tr.append(t)
        W[es] = tr[1] - tr[0]
    D = {e: (W[e] + W["-" + e] - 2 * W["0"]) / mp.mpf(e) ** 2 for e in EPS_MP}
    R1 = [(4 * D[EPS_MP[1]] - D[EPS_MP[0]]) / 3, (4 * D[EPS_MP[2]] - D[EPS_MP[1]]) / 3]
    R2 = (16 * R1[1] - R1[0]) / 15
    return {"W2_R2": float(R2 / 2), "W2_R1": float(R1[1] / 2), "W_at_0": float(W["0"]), "R2_digits": mp.nstr(R2 / 2, 25)}


def case(args):
    L, Lp, m, kappa, n, pl = args
    t0 = time.time()
    d = dense_mp(L, Lp, m, kappa, n, pl)
    cls = PF.classes(Lp, 0.0, True)
    r = PF.packet_terms_M(m, kappa, L, n, 0.0, cls, pl)
    W2 = 0.5 * (r[1] - r[0])
    out = {"L": L, "Lp": Lp, "m": m, "kappa": kappa, "n": n, "placement": pl, "W2_route": W2, "rel_R2": abs(W2 - d["W2_R2"]) / abs(d["W2_R2"]),
           "rel_R1": abs(W2 - d["W2_R1"]) / abs(d["W2_R1"]), "seconds": time.time() - t0}
    out.update(d)
    print(json.dumps(out), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cases = [(8, 3, 0.1, k, 1, pl) for k in (0.3, 0.01, 0.002) for pl in ("left", "sym")]
    with Pool(a.workers) as pool:
        rows = pool.map(case, cases)
    with open(os.path.join(a.out, "control_mp.json"), "w") as fh:
        json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
