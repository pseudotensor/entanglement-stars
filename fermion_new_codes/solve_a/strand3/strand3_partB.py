#!/usr/bin/env python
"""Strand 3 part B (spec sec. 4; builder analyst): the transported frame's exact zero on a 4 x 4 x 4 periodic lattice.

D_W = sum_i sigma_i (x) (T_i^+ - T_i^-)/(2i) (128 x 128, site-major, spinor-minor; site index x1 + 4 x2 + 16 x3), m = 0.1.
A = m psi^-1 + i psi^-3 D_W, E0 = (A^dag A)^(1/2), U = A E0^-1, B0 = E0^(1/2)(N psi)E0^(1/2), B1 = E1^(1/2)(N psi)E1^(1/2),
h_* = [[0, B0 U^dag], [U B0, 0]] (256 x 256, block layout: u then d).  Region R: x1 in {0, 1} (32 sites, 128 modes).
B.1 S_R in the c-frame; B.2 C' = V P_- V^dag, V = diag(1, U^dag), against (1 - sigma_1)/2 (x) 1 and its S_R;
B.3 V' = diag(U, 1); B.4 min eig B1; C4 two independent copies.  Writes <out>/partB.json.
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import json
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "orient"))
from lattice_entropy import h2  # noqa: E402  (the paper's module, unmodified)

SX = np.array([[0, 1], [1, 0]], complex)
SY = np.array([[0, -1j], [1j, 0]], complex)
SZ = np.array([[1, 0], [0, -1]], complex)
NL = 4
NSITE = NL ** 3


def site(x1, x2, x3):
    return (x1 % NL) + NL * (x2 % NL) + NL * NL * (x3 % NL)


def coords():
    return np.array([(i % NL, (i // NL) % NL, i // (NL * NL)) for i in range(NSITE)])


def weyl():
    D = np.zeros((2 * NSITE, 2 * NSITE), complex)
    for i, sig in enumerate((SX, SY, SZ)):
        Ti = np.zeros((NSITE, NSITE))
        for x in range(NSITE):
            c = list(coords()[x])
            c[i] += 1
            Ti[x, site(*c)] = 1.0
        Di = (Ti - Ti.T) / 2j
        D += np.kron(Di, sig)
    return D


def herm(M):
    return 0.5 * (M + M.conj().T)


def roots(M):
    w, V = np.linalg.eigh(herm(M))
    return w, V


def law(m, psi, N, D):
    P = np.kron(np.diag(psi), np.eye(2))
    Pinv = np.kron(np.diag(1.0 / psi), np.eye(2))
    A = m * Pinv + 1j * Pinv @ Pinv @ Pinv @ D
    w0, V0 = roots(A.conj().T @ A)
    w1, V1 = roots(A @ A.conj().T)
    E0h = (V0 * w0 ** 0.25) @ V0.conj().T
    E0inv = (V0 / np.sqrt(w0)) @ V0.conj().T
    E1h = (V1 * np.clip(w1, 0, None) ** 0.25) @ V1.conj().T
    U = A @ E0inv
    L = np.kron(np.diag(N * psi), np.eye(2))
    B0 = herm(E0h @ L @ E0h)
    B1 = herm(E1h @ L @ E1h)
    n = 2 * NSITE
    Z = np.zeros((n, n), complex)
    h = np.block([[Z, B0 @ U.conj().T], [U @ B0, Z]])
    return h, U, B0, B1, float(w0.min())


def ent(C):
    return float(h2(np.linalg.eigvalsh(herm(C))).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    D = weyl()
    xyz = coords()
    m = 0.1
    bgs = {"flat": (np.ones(NSITE), np.ones(NSITE)),
           "nonuniform": (1 + 0.2 * np.cos(2 * np.pi * xyz[:, 0] / 4) * np.cos(2 * np.pi * xyz[:, 1] / 4),
                          1 + 0.3 * np.cos(2 * np.pi * xyz[:, 2] / 4))}
    inR = np.repeat(xyz[:, 0] <= 1, 2)  # (site, spinor) indices of R in one chiral block
    Ridx = np.concatenate([np.where(inR)[0], 2 * NSITE + np.where(inR)[0]])
    n = 2 * NSITE
    target = np.block([[0.5 * np.eye(n), -0.5 * np.eye(n)], [-0.5 * np.eye(n), 0.5 * np.eye(n)]])
    out = {"checks": {"D_hermitian": float(np.linalg.norm(D - D.conj().T)), "R_modes": int(len(Ridx))}, "backgrounds": {}}
    for name, (psi, N) in bgs.items():
        h, U, B0, B1, minAA = law(m, psi, N, D)
        w, V = np.linalg.eigh(herm(h))
        Vm = V[:, w < 0]
        Pm = Vm @ Vm.conj().T
        rec = {"n_negative": int(Vm.shape[1]), "min_abs_energy": float(np.min(np.abs(w))), "min_eig_AdagA": minAA}
        CR = Pm[np.ix_(Ridx, Ridx)]
        rec["S_R_A1"] = ent(CR)
        Vt = np.block([[np.eye(n), np.zeros((n, n))], [np.zeros((n, n)), U.conj().T]])
        Cp = Vt @ Pm @ Vt.conj().T
        rec["A2_max_dev_from_product_projector"] = float(np.max(np.abs(Cp - target)))
        rec["S_R_A2"] = ent(Cp[np.ix_(Ridx, Ridx)])
        Vr = np.block([[U, np.zeros((n, n))], [np.zeros((n, n)), np.eye(n)]])
        Cr = Vr @ Pm @ Vr.conj().T
        er = np.linalg.eigvalsh(herm(Cr[np.ix_(Ridx, Ridx)]))
        rec["B3_S_R"] = float(h2(er).sum())
        rec["B3_max_dev_from_product_projector"] = float(np.max(np.abs(Cr - target)))
        rec["B3_max_restricted_eig_distance_from_0_1"] = float(np.max(np.minimum(np.abs(er), np.abs(1 - er))))
        rec["B3_max_entry_distance_from_0_1"] = float(np.max(np.minimum(np.abs(Cr), np.abs(1 - Cr))))
        rec["B4_min_eig_B1"] = float(np.linalg.eigvalsh(B1)[0])
        rec["min_eig_B0"] = float(np.linalg.eigvalsh(B0)[0])
        C2 = np.block([[CR, np.zeros_like(CR)], [np.zeros_like(CR), CR]])
        rec["C4_two_copies_minus_2S"] = abs(ent(C2) - 2 * rec["S_R_A1"])
        out["backgrounds"][name] = rec
        print(name, rec, flush=True)
    with open(os.path.join(args.out, "partB.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
