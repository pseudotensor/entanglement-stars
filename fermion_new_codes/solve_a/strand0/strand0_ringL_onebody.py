#!/usr/bin/env python
"""Strand 0 T8: ring-L one-body scan at psi = 1, N = 1 (spec sec. 4, T8 and control C12; builder analyst).

L x L sine stencil D; A = m + iD; E = (A^dag A)^(1/2) and U = A E^-1 built numerically (eigh), not from the closed forms;
h_* = [[0, E U^dag], [U E, 0]] (block layout); X1_0 = [[0, X_0 U^dag], [U X_0, 0]], X_0 = E^(1/2) P_0 E^(1/2);
M1_0 = (1/2){P_0 (x) 1_2, h_*}.  Momentum basis |p>(x) = e^{ipx}/sqrt L, p = 2 pi n/L; v^+-_p = (|p>, +-U|p>)/sqrt2.
Reports the pp and anomalous matrix elements in the two valleys, the closed forms of C12, the coefficients c_1, c_2, c_3,
|L Delta_pp| for the listed pairs, and the extreme eigenvalues of the one-particle blocks of X1_0 and M1_0.
Writes <out>/ringL.json.
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import json
import socket
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import strand0_lib as Lb  # noqa: E402

LS = (4, 8, 16, 32, 64, 128, 256, 512, 1024)
MS = (0.1, 0.2, 0.5)


def wrap(d):
    return (d + np.pi) % (2 * np.pi) - np.pi


def one(L, m):
    D = Lb.ring_D(L)
    A = m * np.eye(L) + 1j * D
    w, V = np.linalg.eigh(Lb.herm(A.conj().T @ A))
    E = (V * np.sqrt(w)) @ V.conj().T
    Eh = (V * w ** 0.25) @ V.conj().T
    U = A @ ((V / np.sqrt(w)) @ V.conj().T)
    Z = np.zeros((L, L), dtype=complex)
    h = np.block([[Z, E @ U.conj().T], [U @ E, Z]])
    X0 = np.outer(Eh[:, 0], Eh[0, :])
    X1 = np.block([[Z, X0 @ U.conj().T], [U @ X0, Z]])
    P0 = np.zeros((L, L))
    P0[0, 0] = 1.0
    P02 = np.block([[P0, Z], [Z, P0]])
    M1 = 0.5 * (P02 @ h + h @ P02)
    xs = np.arange(L)

    def ket(n):
        return np.exp(2j * np.pi * n * xs / L) / np.sqrt(L)

    def vpm(n, sign):
        k = ket(n)
        return np.concatenate([k, sign * (U @ k)]) / np.sqrt(2.0)

    def Ep(p):
        return np.sqrt(m * m + np.sin(p) ** 2)

    def phi(p):
        return np.angle(m + 1j * np.sin(p))

    rec = {"L": L, "m": m, "valleys": {}}
    # control: U|p> = e^{i phi_p}|p>
    worstU = 0.0
    for n in range(L):
        p = 2 * np.pi * n / L
        worstU = max(worstU, float(np.linalg.norm(U @ ket(n) - np.exp(1j * phi(p)) * ket(n))))
    rec["C12_U_eigen"] = worstU
    worst = {"X_pp": 0.0, "X_anom": 0.0, "M_pp": 0.0, "M_anom": 0.0, "LDelta_pm": 0.0}
    for vname, n0 in (("0", 0), ("pi", L // 2)):
        ns_ = [n0 + j for j in range(-3, 4)] if L >= 8 else [n0 + j for j in range(-(L // 4), L // 4 + 1)]
        ns_ = sorted({n % L for n in ns_})
        vals = {}
        for n in ns_:
            for npr in ns_:
                p, pp = 2 * np.pi * n / L, 2 * np.pi * npr / L
                a, b = vpm(n, +1), vpm(npr, +1)
                bm = vpm(npr, -1)
                Xpp = a.conj() @ X1 @ b
                Xan = a.conj() @ X1 @ bm
                Mpp = a.conj() @ M1 @ b
                Man = a.conj() @ M1 @ bm
                ph = np.exp(1j * (phi(pp) - phi(p)))
                worst["X_pp"] = max(worst["X_pp"], abs(Xpp - np.sqrt(Ep(p) * Ep(pp)) / L))
                worst["X_anom"] = max(worst["X_anom"], abs(Xan))
                worst["M_pp"] = max(worst["M_pp"], abs(Mpp - (Ep(p) + Ep(pp)) * (1 + ph) / (4 * L)))
                worst["M_anom"] = max(worst["M_anom"], abs(Man - (Ep(p) - Ep(pp)) * (1 - ph) / (4 * L)))
                vals[(n, npr)] = (Xpp, Mpp, Man)
        # the mirror pair within the valley: p = valley + 2 pi/L, p' = -p (mod 2 pi)
        n1 = (n0 + 1) % L
        n2 = (-n1) % L
        p1, p2 = 2 * np.pi * n1 / L, 2 * np.pi * n2 / L
        a, b = vpm(n1, +1), vpm(n2, +1)
        LD = L * ((a.conj() @ X1 @ b) - (a.conj() @ M1 @ b))
        LDcf = -1j * m * np.sin(p1) / Ep(p1) - np.sin(p1) ** 2 / Ep(p1)
        worst["LDelta_pm"] = max(worst["LDelta_pm"], abs(LD - LDcf))
        dp = wrap(p2 - p1)
        c1 = float(np.imag(LD) / dp)
        c2 = float(np.real(LD) / dp ** 2)
        # third order: M_anom(p, valley bottom)
        nb = n0 % L
        pa = 2 * np.pi * n1 / L
        Man = vpm(n1, +1).conj() @ M1 @ vpm(nb, -1)
        dpa = wrap(pa - 2 * np.pi * nb / L)
        c3 = float(L * abs(Man) * 8 * m * m / abs(dpa) ** 3)
        absLD = {"%d,%d" % (n - n0, npr - n0): float(L * abs(vals[(n, npr)][0] - vals[(n, npr)][1]))
                 for (n, npr) in vals if (n, npr) in vals}
        rec["valleys"][vname] = {"c1": c1, "c2": c2, "4m_c2": 4 * m * c2, "c3": c3, "LDelta_mirror": [float(LD.real), float(LD.imag)],
                                 "abs_L_Delta_pp": absLD}
    rec["C12"] = {k: float(v) for k, v in worst.items()}
    # one-particle blocks on the positive-energy states of h_*
    wh, Vh = np.linalg.eigh(Lb.herm(h))
    Vp = Vh[:, wh > 0]
    bx = np.linalg.eigvalsh(Lb.herm(Vp.conj().T @ X1 @ Vp))
    bm = np.linalg.eigvalsh(Lb.herm(Vp.conj().T @ M1 @ Vp))
    wm, vm = np.linalg.eigh(Lb.herm(Vp.conj().T @ M1 @ Vp))
    rec["onepart_X1"] = {"min": float(bx[0]), "max": float(bx[-1]), "E00": float(np.real(E[0, 0])),
                         "n_nonzero": int(np.sum(np.abs(bx) > 1e-10))}
    rec["onepart_M1"] = {"min": float(bm[0]), "max": float(bm[-1]),
                         "witness_energy": float(np.real(vm[:, 0].conj() @ np.diag(wh[wh > 0]) @ vm[:, 0]))}
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--Lmax", type=int, default=1024)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    recs = []
    for L in LS:
        if L > args.Lmax:
            continue
        for m in MS:
            r = one(L, m)
            recs.append(r)
            print("L=%d m=%g c1(0)=%.6f c1(pi)=%.6f 4mc2(0)=%.6f c3(0)=%.6f C12max=%.2e (%.0f s)" % (
                L, m, r["valleys"]["0"]["c1"], r["valleys"]["pi"]["c1"], r["valleys"]["0"]["4m_c2"],
                r["valleys"]["0"]["c3"], max(r["C12"].values()), time.time() - t0), flush=True)
    with open(os.path.join(args.out, "ringL.json"), "w") as fh:
        json.dump({"meta": {"host": socket.gethostname(), "seconds": time.time() - t0, "numpy": np.__version__},
                   "records": recs}, fh, indent=1)


if __name__ == "__main__":
    main()
