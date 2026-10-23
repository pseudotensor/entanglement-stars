#!/usr/bin/env python
"""Strand 6 (a), version 3 (builder analyst): the spin splitting at fixed position, by wavepackets.

Version 2 found that the chain's two sigma_1 sectors have identical spectra with the x_1 valley labels exchanged (sin(k + pi) = -sin k
maps s D_1 at valley 0 to -s D_1 at valley pi), and that in the linear potential a spin-dependent energy shift moves the eigenstates'
centroids, so the eigenvalue difference between the sectors is not a splitting at fixed position.  This version measures it at fixed
position: the same normalized packet phi(x_1) = exp(-(x_1 - x_0)^2/(2 w^2)) exp(i k_nu x_1), k_nu in {0, pi} (the x_1 valley), centred at
x_0 = L_1/2 = 64 (where psi = 1), in the two sigma_1 sectors: E_s = <phi|B_s|phi> with B_s = E_s^(1/2) psi E_s^(1/2), E_s = |A_s|,
A_s = m Psi^-1 + i Psi^-3 (s D_1) (polar) or the symmetric placement; the splitting Delta(nu) = E_+ - E_- and Delta/g_u, for widths
w in {8, 16, 24} and g_u in {0, 5e-4, 1e-3, 2e-3}, m in {0.1, 0.2}; beside it the packet's spin-averaged energy against m(1 - u(x_0))
= m and the linearity ratio Delta(1e-3)/Delta(5e-4).  Writes <out>/spingrad_v3.json."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import time

import numpy as np


def chain_D(L1):
    D = np.zeros((L1, L1), dtype=complex)
    for x in range(L1 - 1):
        D[x, x + 1] += 1 / 2j
        D[x + 1, x] += -1 / 2j
    return D


def herm(M):
    return 0.5 * (M + M.conj().T)


def sandwich(A, psi):
    w, V = np.linalg.eigh(herm(A.conj().T @ A))
    Eh = (V * w ** 0.25) @ V.conj().T
    return herm(Eh @ np.diag(psi) @ Eh)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--L1", type=int, default=128)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    L1 = a.L1
    D1 = chain_D(L1)
    x1 = np.arange(L1, dtype=float)
    x0 = L1 / 2
    res = {"meta": {"L1": L1, "x0": x0}, "rows": []}
    for m in (0.1, 0.2):
        for g in (0.0, 5e-4, 1e-3, 2e-3):
            psi = 1 + g * (x1 - x0)
            Pinv = np.diag(1 / psi)
            P3 = Pinv @ Pinv @ Pinv
            B = {}
            for place in ("pol", "sym"):
                for s in (+1, -1):
                    A = m * Pinv + (1j * P3 @ (s * D1) if place == "pol" else 0.5j * (P3 @ (s * D1) + (s * D1) @ P3))
                    B[(place, s)] = sandwich(A, psi)
            for wd in (8.0, 16.0, 24.0):
                for nu, kn in (("0", 0.0), ("pi", np.pi)):
                    phi = np.exp(-(x1 - x0) ** 2 / (2 * wd * wd)) * np.exp(1j * kn * x1)
                    phi = phi / np.linalg.norm(phi)
                    rec = {"m": m, "g_u": g, "width": wd, "valley": nu}
                    for place in ("pol", "sym"):
                        Ep = float(np.real(phi.conj() @ B[(place, 1)] @ phi))
                        Em = float(np.real(phi.conj() @ B[(place, -1)] @ phi))
                        rec[place] = {"E_plus": Ep, "E_minus": Em, "split": Ep - Em, "split_over_g": (Ep - Em) / g if g else None,
                                      "mean_minus_m": 0.5 * (Ep + Em) - m}
                    res["rows"].append(rec)
    for r in res["rows"]:
        if r["g_u"] == 1e-3:
            r0 = [t for t in res["rows"] if t["g_u"] == 5e-4 and t["m"] == r["m"] and t["width"] == r["width"] and t["valley"] == r["valley"]][0]
            for place in ("pol", "sym"):
                r[place]["linearity_ratio_1e-3_over_5e-4"] = r[place]["split"] / r0[place]["split"] if r0[place]["split"] else None
    res["meta"]["seconds"] = time.time() - t0
    with open(os.path.join(a.out, "spingrad_v3.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    for r in res["rows"]:
        if r["g_u"] in (1e-3,):
            print("m %g w %g valley %s: pol split/g %.6f (lin %.4f) sym split/g %.6f mean-m pol %.3e" % (
                r["m"], r["width"], r["valley"], r["pol"]["split_over_g"], r["pol"]["linearity_ratio_1e-3_over_5e-4"], r["sym"]["split_over_g"],
                r["pol"]["mean_minus_m"]), flush=True)


if __name__ == "__main__":
    main()
