#!/usr/bin/env python
"""Strand 2a addendum (spec solve/specs/SPEC_strand2a_addendum_transport.md; builder analyst): one-body only.

A: rings L in {4, ..., 256}, m in {0.1, 0.2}: the law's positive spectrum = eig(B0), B0 = E0^(1/2) (N psi) E0^(1/2) with
A = m psi^-1 + i psi^-3 D (1-component Weyl D, strand 0); the native: positive eigenvalues of sum_x N_x (1/2){P_x (x) 1_2,
h_nat(s = psi)}, h_nat(s) = m (1 (x) sigma_3) + s^-1 (D (x) sigma_1) s^-1.  A.1 ruler, A.2 clock, A.3 N psi = 1 and N = psi.
B: B.1 open 3 x 3 x 3 cube and B.2 periodic 4^3 box at psi = 1: one-particle blocks of X1_x and M1_x = (1/2){P_x (x) 1_4, h_*}.
Writes <out>/addendum.json.
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import itertools
import json
import socket
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "frozen"))
import native_core as nc  # noqa: E402

SX, SY, SZ = nc.SX, nc.SY, nc.SZ


def herm(M):
    return 0.5 * (M + M.conj().T)


def ring_D(L):
    D = np.zeros((L, L), dtype=complex)
    for x in range(L):
        D[x, (x + 1) % L] += -0.5j
        D[x, (x - 1) % L] += 0.5j
    return D


def law_spectrum(m, psi, N, D, ncomp=1):
    P = np.kron(np.diag(1.0 / psi), np.eye(ncomp))
    A = m * P + 1j * P @ P @ P @ D
    w, V = np.linalg.eigh(herm(A.conj().T @ A))
    E0h = (V * w ** 0.25) @ V.conj().T
    B0 = herm(E0h @ np.kron(np.diag(N * psi), np.eye(ncomp)) @ E0h)
    return np.sort(np.linalg.eigvalsh(B0)), float(w.min())


def native_spectrum(m, s, N, D):
    L = len(s)
    Sinv = np.kron(np.diag(1.0 / s), np.eye(2))
    h = m * np.kron(np.eye(L), SZ) + Sinv @ np.kron(D, SX) @ Sinv
    H = np.zeros_like(h)
    for x in range(L):
        Px = np.zeros((2 * L, 2 * L))
        Px[2 * x, 2 * x] = Px[2 * x + 1, 2 * x + 1] = 1.0
        H += N[x] * 0.5 * (Px @ h + h @ Px)
    w = np.sort(np.linalg.eigvalsh(herm(H)))
    return w[w > 0]


def partA():
    rows = []
    for L in (4, 8, 16, 32, 64, 128, 256):
        D = ring_D(L)
        x = np.arange(L)
        c = np.cos(2 * np.pi * x / L)
        for m in (0.1, 0.2):
            for eps in (0.05, 0.2):
                cases = {"A1_ruler": (1 + eps * c, np.ones(L)), "A2_clock": (np.ones(L), 1 + eps * c),
                         "A3_Npsi1": (1 + eps * c, 1.0 / (1 + eps * c)), "A3_N_eq_psi": (1 + eps * c, 1 + eps * c)}
                for name, (psi, N) in cases.items():
                    wl, minAA = law_spectrum(m, psi, N, D)
                    wn = native_spectrum(m, psi, N, D)
                    rows.append({"item": name, "L": L, "m": m, "eps": eps, "law_lowest4": wl[:4].tolist(),
                                 "native_lowest4": wn[:4].tolist(), "law_shift_below_m": float(m - wl[0]),
                                 "native_shift_below_m": float(m - wn[0]), "law_bottom_split": float(wl[1] - wl[0]),
                                 "native_bottom_split": float(wn[1] - wn[0]), "m_Nmin": float(m * N.min()),
                                 "min_eig_AdagA": minAA})
    # fitted powers of the law's ruler shift (A.1): in eps at fixed (L, m); in 1/L at fixed (eps, m)
    fits = []
    for m in (0.1, 0.2):
        for L in (4, 8, 16, 32, 64, 128, 256):
            s = {r["eps"]: r["law_shift_below_m"] for r in rows if r["item"] == "A1_ruler" and r["L"] == L and r["m"] == m}
            if s.get(0.05, 0) > 0 and s.get(0.2, 0) > 0:
                fits.append({"m": m, "L": L, "power_in_eps": float(np.log(s[0.2] / s[0.05]) / np.log(4.0))})
        for eps in (0.05, 0.2):
            pts = sorted((r["L"], r["law_shift_below_m"]) for r in rows if r["item"] == "A1_ruler" and r["m"] == m and r["eps"] == eps)
            for (L1, a1), (L2, a2) in zip(pts[:-1], pts[1:]):
                if a1 > 0 and a2 > 0:
                    fits.append({"m": m, "eps": eps, "L_pair": [L1, L2], "power_in_1_over_L": float(np.log(a1 / a2) / np.log(L2 / L1))})
    return rows, fits


def open_cube_D(n):
    pos = [(x0, x1, x2) for x2 in range(n) for x1 in range(n) for x0 in range(n)]
    idx = {p: i for i, p in enumerate(pos)}
    D = np.zeros((2 * len(pos), 2 * len(pos)), dtype=complex)
    for p in pos:
        for j, s in enumerate((SX, SY, SZ)):
            q = list(p)
            q[j] += 1
            q = tuple(q)
            if q in idx:
                a, b = idx[p], idx[q]
                D[2 * a:2 * a + 2, 2 * b:2 * b + 2] += s / 2j
                D[2 * b:2 * b + 2, 2 * a:2 * a + 2] += -s / 2j
    return D, pos


def periodic_D(n):
    pos = [(x0, x1, x2) for x2 in range(n) for x1 in range(n) for x0 in range(n)]
    idx = {p: i for i, p in enumerate(pos)}
    D = np.zeros((2 * len(pos), 2 * len(pos)), dtype=complex)
    for p in pos:
        for j, s in enumerate((SX, SY, SZ)):
            q = list(p)
            q[j] = (q[j] + 1) % n
            a, b = idx[p], idx[tuple(q)]
            D[2 * a:2 * a + 2, 2 * b:2 * b + 2] += s / 2j
            D[2 * b:2 * b + 2, 2 * a:2 * a + 2] += -s / 2j
    return D, pos


def chiral_objects(m, D):
    nW = D.shape[0]
    A = m * np.eye(nW) + 1j * D
    w, V = np.linalg.eigh(herm(A.conj().T @ A))
    if w.min() <= 0:
        return None, float(w.min())
    E = (V * np.sqrt(w)) @ V.conj().T
    Eh = (V * w ** 0.25) @ V.conj().T
    U = A @ ((V / np.sqrt(w)) @ V.conj().T)
    Z = np.zeros((nW, nW), dtype=complex)
    h = np.block([[Z, E @ U.conj().T], [U @ E, Z]])
    return {"E": E, "Eh": Eh, "U": U, "h": h, "nW": nW}, float(w.min())


def site_blocks(o, x):
    nW = o["nW"]
    Z = np.zeros((nW, nW), dtype=complex)
    P2 = np.zeros((nW, nW))
    P2[2 * x:2 * x + 2, 2 * x:2 * x + 2] = np.eye(2)
    X = herm(o["Eh"] @ P2 @ o["Eh"])
    X1 = np.block([[Z, X @ o["U"].conj().T], [o["U"] @ X, Z]])
    P4 = np.block([[P2, Z], [Z, P2]])
    M1 = 0.5 * (P4 @ o["h"] + o["h"] @ P4)
    return X1, M1


def partB():
    out = {}
    for m in (0.0, 0.1):
        D, pos = open_cube_D(3)
        o, minAA = chiral_objects(m, D)
        used = m
        if o is None:
            used = 1e-3
            o, minAA2 = chiral_objects(used, D)
        wh, Vh = np.linalg.eigh(herm(o["h"]))
        Vp, Ep = Vh[:, wh > 0], wh[wh > 0]
        sites = {"corner": (0, 0, 0), "edge": (1, 0, 0), "face_centre": (1, 1, 0), "body_centre": (1, 1, 1)}
        rec = {"m_requested": m, "m_used": used, "min_eig_AdagA_at_requested_m": minAA,
               "one_particle_spectrum": np.unique(np.round(Ep, 10)).tolist(), "sites": {}}
        for name, p in sites.items():
            x = pos.index(p)
            X1, M1 = site_blocks(o, x)
            bx = np.linalg.eigvalsh(herm(Vp.conj().T @ X1 @ Vp))
            wm, vm = np.linalg.eigh(herm(Vp.conj().T @ M1 @ Vp))
            rec["sites"][name] = {"X1_nonzero": [float(v) for v in bx if abs(v) > 1e-12], "X1_min": float(bx[0]),
                                  "M1_min": float(wm[0]), "M1_max": float(wm[-1]),
                                  "witness_energy": float(np.real(vm[:, 0].conj() @ (Ep * vm[:, 0])))}
        out["B1_open3_m%g" % m] = rec
    # B.2 periodic 4^3
    for m in (0.1,):
        D, pos = periodic_D(4)
        o, minAA = chiral_objects(m, D)
        wh, Vh = np.linalg.eigh(herm(o["h"]))
        Vp, Ep = Vh[:, wh > 0], wh[wh > 0]
        X1, M1 = site_blocks(o, 0)
        bx = np.linalg.eigvalsh(herm(Vp.conj().T @ X1 @ Vp))
        wm, vm = np.linalg.eigh(herm(Vp.conj().T @ M1 @ Vp))
        rec = {"m": m, "X1_nonzero": [float(v) for v in bx if abs(v) > 1e-12], "M1_min": float(wm[0]),
               "witness_energy": float(np.real(vm[:, 0].conj() @ (Ep * vm[:, 0])))}
        # momentum states of the valley at k = 0: shells k_j in {0, +-pi/2}, lowest shells with >= 8 states (spin up/down)
        nW = o["nW"]
        X = np.array(pos)
        ks = [k for k in itertools.product((0, 1, -1), repeat=3)]
        ks.sort(key=lambda k: sum(abs(v) for v in k))
        shells = {}
        for k in ks:
            shells.setdefault(sum(abs(v) for v in k), []).append(k)
        chosen = []
        for sh in sorted(shells):
            chosen += shells[sh]
            if 2 * len(chosen) >= 8:
                break
        states_p, states_m, labels = [], [], []
        Z = np.zeros(nW, dtype=complex)
        for k in chosen:
            kv = np.array(k) * np.pi / 2
            pw = np.exp(1j * X @ kv) / np.sqrt(len(pos))
            for sp in (0, 1):
                v = np.zeros(nW, dtype=complex)
                v[sp::2] = pw
                Uv = o["U"] @ v
                states_p.append(np.concatenate([v, Uv]) / np.sqrt(2))
                states_m.append(np.concatenate([v, -Uv]) / np.sqrt(2))
                labels.append([list(map(int, k)), sp])
        Sp, Sm = np.array(states_p).T, np.array(states_m).T
        Xpp = Sp.conj().T @ X1 @ Sp
        Mpp = Sp.conj().T @ M1 @ Sp
        Man = Sp.conj().T @ M1 @ Sm
        Xan = Sp.conj().T @ X1 @ Sm
        rec.update({"valley_states": labels, "n_states": len(labels), "norm_X_pp": float(np.linalg.norm(Xpp)),
                    "norm_M_pp": float(np.linalg.norm(Mpp)), "norm_pp_diff": float(np.linalg.norm(Mpp - Xpp)),
                    "norm_M_anom": float(np.linalg.norm(Man)), "norm_X_anom": float(np.linalg.norm(Xan)),
                    "check_states_are_eigenstates": float(max(np.linalg.norm(o["h"] @ Sp[:, i] - (Sp[:, i].conj() @ o["h"] @ Sp[:, i]) * Sp[:, i])
                                                             for i in range(Sp.shape[1])))})
        out["B2_periodic4_m%g" % m] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    rows, fits = partA()
    B = partB()
    with open(os.path.join(a.out, "addendum.json"), "w") as fh:
        json.dump({"meta": {"host": socket.gethostname(), "seconds": time.time() - t0}, "A": rows, "A_fits": fits, "B": B}, fh, indent=1)
    print("addendum done in %.0f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
