#!/usr/bin/env python
"""Strand 0 part B: the open 2 x 2 x 2 cube at the one-body level (spec sec. 7; builder analyst).

D_W = sum_j sigma_j (x) D_j from existing neighbours only (native_core.lattice("cube")["D"], 16 x 16, site-major,
spinor-minor).  Law: A = m psi^-1 + i psi^-3 D_W (psi_x (x) 1_2), E0, E1, U, B0 (16 x 16); h_* = [[0, B0 U^dag],
[U B0, 0]] (32 x 32, block layout: chiral u block of 16 (site, spinor) indices, then d).  For x = site 0 and the
opposite corner: X1_x = [[0, X_x U^dag], [U X_x, 0]], X_x = psi_x E0^(1/2) (P_x (x) 1_2) E0^(1/2) (rank 2);
M1_x = (1/2){P_x (x) 1_4, h_*}.  One-particle blocks on the 16 positive eigenvectors of h_*.  Writes <out>/cube.json.
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
sys.path.insert(0, os.path.join(HERE, "frozen"))
import strand0_lib as Lb  # noqa: E402
import native_core as nc  # noqa: E402

NS = 8


def polar16(m, psi, DW):
    P = np.kron(np.diag(np.asarray(psi, dtype=float)), np.eye(2))
    Pinv = np.linalg.inv(P)
    A = m * Pinv + 1j * np.linalg.matrix_power(Pinv, 3) @ DW
    w0, V0 = np.linalg.eigh(Lb.herm(A.conj().T @ A))
    w1, V1 = np.linalg.eigh(Lb.herm(A @ A.conj().T))
    E0 = (V0 * np.sqrt(w0)) @ V0.conj().T
    E0h = (V0 * w0 ** 0.25) @ V0.conj().T
    U = A @ ((V0 / np.sqrt(w0)) @ V0.conj().T)
    E1 = (V1 * np.sqrt(np.clip(w1, 0, None))) @ V1.conj().T
    return A, E0, E0h, E1, U, float(w0.min()), P


def case(m, psi, DW, positions):
    A, E0, E0h, E1, U, minAA, P = polar16(m, psi, DW)
    B0 = Lb.herm(E0h @ P @ E0h)
    Z = np.zeros((16, 16), dtype=complex)
    h = np.block([[Z, B0 @ U.conj().T], [U @ B0, Z]])
    wh, Vh = np.linalg.eigh(Lb.herm(h))
    Vp = Vh[:, wh > 0]
    Epos = wh[wh > 0]
    far = [i for i, p in enumerate(positions) if p == (1, 1, 1)][0]
    rec = {"m": m, "psi": list(map(float, psi)), "min_eig_AdagA": minAA,
           "C2_A_minus_UE0": float(np.linalg.norm(A - U @ E0)), "sites": {}}
    X1s, M1s = {}, {}
    for x in range(NS):
        Px2 = np.zeros((16, 16))
        Px2[2 * x:2 * x + 2, 2 * x:2 * x + 2] = np.eye(2)
        Xx = Lb.herm(psi[x] * E0h @ Px2 @ E0h)
        X1s[x] = np.block([[Z, Xx @ U.conj().T], [U @ Xx, Z]])
        Px4 = np.block([[Px2, Z], [Z, Px2]])
        M1s[x] = 0.5 * (Px4 @ h + h @ Px4)
    rec["sum_X1_minus_h"] = float(np.linalg.norm(sum(X1s.values()) - h))
    rec["sum_M1_minus_h"] = float(np.linalg.norm(sum(M1s.values()) - h))
    for x in range(NS):
        bx = Lb.herm(Vp.conj().T @ X1s[x] @ Vp)
        bm = Lb.herm(Vp.conj().T @ M1s[x] @ Vp)
        ex = np.linalg.eigvalsh(bx)
        wm, vm = np.linalg.eigh(bm)
        rec["sites"][str(x)] = {
            "position": positions[x],
            "X1_onepart_eigs": [float(v) for v in ex],
            "M1_onepart_eigs": [float(v) for v in wm],
            "M1_min": float(wm[0]),
            "witness_energy": float(np.real(vm[:, 0].conj() @ np.diag(Epos) @ vm[:, 0])),
            "witness_X1": float(np.real(vm[:, 0].conj() @ bx @ vm[:, 0])),
            "rel_fro_X1_minus_M1": float(np.linalg.norm(X1s[x] - M1s[x]) / np.linalg.norm(X1s[x])),
            "rel_spec_X1_minus_M1": float(np.linalg.norm(X1s[x] - M1s[x], 2) / np.linalg.norm(X1s[x], 2)),
            "trace_onepart_X1": float(np.trace(bx).real),
            "trace_onepart_M1": float(np.trace(bm).real),
        }
    rec["report_sites"] = [0, far]
    rec["E_pos_min"] = float(Epos.min())
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    lat = nc.lattice("cube")
    DW = lat["D"]
    positions = [tuple(map(int, p)) for p in lat["positions"]]
    recs = []
    checks = {"DW_hermitian": float(np.linalg.norm(DW - DW.conj().T)),
              "DW_squared_minus_3_4": float(np.linalg.norm(DW @ DW - 0.75 * np.eye(16)))}
    for m in (0.1, 0.0):
        for name, psi in (("flat", np.ones(NS)), ("bump0", 1 + 0.2 * (np.arange(NS) == 0))):
            r = case(m, psi, DW, positions)
            r["bg"] = name
            recs.append(r)
            print("cube m=%g %s: min eig A^dag A %.3e, corner M1 min %.6f (site 0), sum rules %.1e %.1e" % (
                m, name, r["min_eig_AdagA"], r["sites"]["0"]["M1_min"], r["sum_X1_minus_h"], r["sum_M1_minus_h"]),
                flush=True)
    with open(os.path.join(args.out, "cube.json"), "w") as fh:
        json.dump({"meta": {"host": socket.gethostname(), "seconds": time.time() - t0}, "checks": checks,
                   "records": recs}, fh, indent=1)


if __name__ == "__main__":
    main()
