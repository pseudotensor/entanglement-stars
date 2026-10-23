#!/usr/bin/env python
"""Strand 2a addendum, items A.1c and A.4 (added to SPEC_strand2a_addendum_transport.md by the lead after the strand 0 red
team; builder analyst).  One-body, rings of L sites, 1-component Weyl D (strand 0 sec. 1), flat positive basis
v_p^+ = (|p>, U|p>)/sqrt2 (block layout; U = A E^-1 at psi = 1), mode layout j = 2x + chiral.

A.1c (L = 256, m = 0.1, psi = N = 1): <rho^M_0> - <rho^P_0> = <psi| M1_0 - X1_0 |psi> in (i) a one-valley packet
c_p ~ exp(-p^2/(2 sigma^2)), sigma = 4 x 2 pi/L, p in the valley at 0 (n in [-L/4, L/4)), real phases; (ii) its two-valley
counterpart (equal weights at 0 and pi); (iii) (|v_p^+> + e^{i alpha}|v_{-p}^+>)/sqrt2 at p = 2 pi/L, alpha in {0, pi/2, pi}.
A.4 (L in {64, 256}, m in {0.1, 0.2}, psi = 1 + u, u_x = eps cos(2 pi j x/L), j in {1, 2}, eps in {1e-3, 1e-2}):
V_law(p,p') = <v_p^+| h_*[psi] - h_*[1] |v_p'^+>/eps and V_nat(p,p') = <v_p^+| W (h_nat[psi] - h_nat[1]) W^dag |v_p'^+>/eps for
intravalley p, p' (p - p' = +-2 pi j/L) in both valleys; their difference Delta = V_law - V_nat against
u_pp' [-2 i nu (p' - p) + (p' - p)^2/m] (u_pp' = <p|u|p'>/eps, nu = +1 at 0, -1 at pi): least-squares fits of Delta/u_pp'
= b (p' - p) + c (p' - p)^2 per (L, m, eps, valley), coefficients and residual.  Hellmann-Feynman check: the lowest
positive level shift of h_*[psi] (unit lapse) against the diagonal first-order vertex V_law(0, 0) (zero for a cosine u).
Writes <out>/addendum2.json.
Version 3: A.4 also reports every raw element (p, p', V_law, V_nat, Delta/u_pp') and a combined fit over j in {1, 2} (four
distinct transfers) per (L, m, eps, valley); writes <out>/addendum3.json.
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
import native_core as nc  # noqa: E402

SX, SY, SZ, I2 = nc.SX, nc.SY, nc.SZ, nc.I2


def herm(M):
    return 0.5 * (M + M.conj().T)


def ring_D(L):
    D = np.zeros((L, L), dtype=complex)
    for x in range(L):
        D[x, (x + 1) % L] += -0.5j
        D[x, (x - 1) % L] += 0.5j
    return D


def perm(n):
    P = np.zeros((2 * n, 2 * n))
    for x in range(n):
        for s in range(2):
            P[2 * x + s, n * s + x] = 1.0
    return P


def law_h(m, psi, D, P):
    p = np.asarray(psi, float)
    A = m * np.diag(1 / p) + 1j * np.diag(p ** -3) @ D
    w, V = np.linalg.eigh(herm(A.conj().T @ A))
    E = (V * np.sqrt(w)) @ V.conj().T
    Eh = (V * w ** 0.25) @ V.conj().T
    U = A @ ((V / np.sqrt(w)) @ V.conj().T)
    B0 = herm(Eh @ np.diag(p) @ Eh)
    n = len(p)
    Z = np.zeros((n, n), dtype=complex)
    return P @ np.block([[Z, B0 @ U.conj().T], [U @ B0, Z]]) @ P.T, U, E


def nat_h(m, s, D):
    L = len(s)
    Sinv = np.kron(np.diag(1.0 / np.asarray(s, float)), I2)
    return m * np.kron(np.eye(L), SZ) + Sinv @ np.kron(D, SX) @ Sinv


def flat_basis(m, L, D, P):
    h1, U, E = law_h(m, np.ones(L), D, P)
    xs = np.arange(L)
    vp = {}
    for n in range(L):
        k = np.exp(2j * np.pi * n * xs / L) / np.sqrt(L)
        vp[n] = P @ np.concatenate([k, U @ k]) / np.sqrt(2)
    return h1, U, vp


def A1c(L=256, m=0.1):
    D = ring_D(L)
    P = perm(L)
    h1, U, vp = flat_basis(m, L, D, P)
    P0 = np.zeros((2 * L, 2 * L))
    P0[0, 0] = P0[1, 1] = 1.0
    M1 = 0.5 * (P0 @ h1 + h1 @ P0)
    w, V = np.linalg.eigh(herm(U.conj().T @ U))
    A = m * np.eye(L) + 1j * D
    wA, VA = np.linalg.eigh(herm(A.conj().T @ A))
    Eh = (VA * wA ** 0.25) @ VA.conj().T
    X0 = np.outer(Eh[:, 0], Eh[0, :])
    Zl = np.zeros((L, L), dtype=complex)
    X1 = P @ np.block([[Zl, X0 @ U.conj().T], [U @ X0, Zl]]) @ P.T
    Dm = M1 - X1
    sig = 4 * 2 * np.pi / L

    def packet(center):
        c = np.zeros(2 * L, dtype=complex)
        for n in range(-L // 4, L // 4):
            nn = (n + (L // 2 if center == "pi" else 0)) % L
            p = 2 * np.pi * n / L
            c += np.exp(-p * p / (2 * sig * sig)) * vp[nn]
        return c / np.linalg.norm(c)
    out = {}
    one = packet("0")
    two = (packet("0") + packet("pi")) / np.sqrt(2)
    out["one_valley_packet"] = float(np.real(one.conj() @ Dm @ one))
    out["two_valley_packet"] = float(np.real(two.conj() @ Dm @ two))
    out["pi_valley_packet"] = float(np.real(packet("pi").conj() @ Dm @ packet("pi")))
    n1 = 1
    for alpha in (0.0, np.pi / 2, np.pi):
        st = (vp[n1] + np.exp(1j * alpha) * vp[(-n1) % L]) / np.sqrt(2)
        out["two_mode_alpha_%.4f" % alpha] = float(np.real(st.conj() @ Dm @ st))
    return out


def A4():
    rows = []
    for L in (64, 256):
        D = ring_D(L)
        P = perm(L)
        Wf = np.kron(np.eye(L), 0.5 * (I2 - 1j * (SX + SY + SZ)))
        xs = np.arange(L)
        for m in (0.1, 0.2):
            h1, U, vp = flat_basis(m, L, D, P)
            hn1 = Wf @ nat_h(m, np.ones(L), D) @ Wf.conj().T
            for j in (1, 2):
                for eps in (1e-3, 1e-2):
                    u = eps * np.cos(2 * np.pi * j * xs / L)
                    hl, _, _ = law_h(m, 1 + u, D, P)
                    hn = Wf @ nat_h(m, 1 + u, D) @ Wf.conj().T
                    dl, dn = (hl - h1) / eps, (hn - hn1) / eps
                    w = np.linalg.eigvalsh(herm(hl))
                    w1 = np.linalg.eigvalsh(herm(h1))
                    for valley, n0, nu in (("0", 0, +1), ("pi", L // 2, -1)):
                        X, Y = [], []
                        raw = []
                        for a in range(-3, 4):
                            for sgn in (+1, -1):
                                nb = a + sgn * j
                                p, pp = 2 * np.pi * a / L, 2 * np.pi * nb / L
                                na, nbb = (n0 + a) % L, (n0 + nb) % L
                                upp = np.exp(-1j * 0) * (1 / L) * np.sum(np.exp(-1j * 2 * np.pi * (na - nbb) * xs / L) * u) / eps
                                Vl = vp[na].conj() @ dl @ vp[nbb]
                                Vn = vp[na].conj() @ dn @ vp[nbb]
                                X.append(pp - p)
                                Y.append((Vl - Vn) / upp)
                                raw.append({"p": p, "pp": pp, "V_law": [float(Vl.real), float(Vl.imag)], "V_nat": [float(Vn.real), float(Vn.imag)],
                                            "u_pp": [float(upp.real), float(upp.imag)],
                                            "Delta_over_u": [float(((Vl - Vn) / upp).real), float(((Vl - Vn) / upp).imag)]})
                        X, Y = np.array(X), np.array(Y)
                        Amat = np.column_stack([X, X ** 2]).astype(complex)
                        coef, res, rk, sv = np.linalg.lstsq(Amat, Y, rcond=None)
                        resid = float(np.linalg.norm(Amat @ coef - Y) / np.linalg.norm(Y))
                        rows.append({"L": L, "m": m, "j": j, "eps": eps, "valley": valley, "nu": nu,
                                     "b": [float(coef[0].real), float(coef[0].imag)], "c": [float(coef[1].real), float(coef[1].imag)],
                                     "prediction_b": [0.0, -2.0 * nu], "prediction_c": [1.0 / m, 0.0], "rel_residual": resid,
                                     "HF_lowest_shift": float(w[w > 0].min() - w1[w1 > 0].min()),
                                     "HF_vertex_00": float(np.real(vp[n0].conj() @ dl @ vp[n0]) * eps), "raw": raw})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    rows = A4()
    comb = []
    for L in (64, 256):
        for m in (0.1, 0.2):
            for eps in (1e-3, 1e-2):
                for valley in ("0", "pi"):
                    sel = [r for r in rows if r["L"] == L and r["m"] == m and r["eps"] == eps and r["valley"] == valley]
                    X = np.array([e["pp"] - e["p"] for r in sel for e in r["raw"]])
                    Y = np.array([complex(*e["Delta_over_u"]) for r in sel for e in r["raw"]])
                    Am = np.column_stack([X, X ** 2]).astype(complex)
                    coef = np.linalg.lstsq(Am, Y, rcond=None)[0]
                    comb.append({"L": L, "m": m, "eps": eps, "valley": valley, "b": [float(coef[0].real), float(coef[0].imag)],
                                 "c": [float(coef[1].real), float(coef[1].imag)],
                                 "rel_residual": float(np.linalg.norm(Am @ coef - Y) / np.linalg.norm(Y)), "n_points": int(len(X))})
    out = {"meta": {"host": socket.gethostname()}, "A1c": A1c(), "A4": rows, "A4_combined_fit": comb}
    out["meta"]["seconds"] = time.time() - t0
    with open(os.path.join(a.out, "addendum3.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("addendum v3 done in %.0f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
