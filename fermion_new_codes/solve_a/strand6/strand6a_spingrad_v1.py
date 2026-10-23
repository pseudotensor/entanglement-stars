#!/usr/bin/env python
"""Strand 6 (a) (spec solve/specs/SPEC_strand6_validation.md (a); builder analyst): the light packet's one-particle spectrum on a linear
ruler gradient at unit clock, polar placement against the symmetric placement.

Lattice: an L_1 x L_t x L_t slab (default 128 x 2 x 2), the cube's Weyl sine stencil (D[x, x + e_j] += sigma_j/(2i), Hermitian), open
along x_1 (the links that would wrap in x_1 are dropped), periodic in x_2, x_3.  On a periodic direction of length 2 the stencil's
forward and backward terms land on the same site with opposite signs and cancel, so the transverse operator is zero there and the
transverse momenta are k_2, k_3 in {0, pi} with sin k = 0: the four transverse sectors carry identical operators (stated in the result).
N = 1, psi(x_1) = 1 + g_u (x_1 - L_1/2).  h_pol = B_0 = E_0^(1/2) psi E_0^(1/2), E_0 = |A|, A = m Psi^-1 + i Psi^-3 D (strand2a_lib.Packet,
the particle block); h_sym the same sandwich with A_sym = m Psi^-1 + (i/2)(Psi^-3 D + D Psi^-3).  For each operator the lowest n_lev levels
with <sigma_1>, the centroid <x_1>, the x_1 valley content (the weight of the Fourier transform of the x_1 profile within pi/4 of k_1 = 0
against within pi/4 of pi), and the spin pairs: the two states of opposite <sigma_1> sign whose valley labels agree and whose centroids
are nearest, their energy difference E(sigma_1 > 0) - E(sigma_1 < 0) divided by g_u.  g_u = 0 is the control (h_pol and h_sym coincide).
Writes <out>/spingrad.json."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import json
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(HERE, "..", "strand2", "frozen"), os.path.join(HERE, "..", "strand2"), os.path.join(HERE, "frozen"), HERE):
    sys.path.insert(0, _p)   # the box keeps strand2a_lib.py and frozen/ beside this file; the repository keeps them in ../strand2
import strand2a_lib as S2  # noqa: E402

SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
SIG = (SX, SY, SZ)


def slab(L1, Lt):
    n = L1 * Lt * Lt
    pos = np.array([[x % L1, (x // L1) % Lt, x // (L1 * Lt)] for x in range(n)], dtype=int)
    idx = {tuple(p): i for i, p in enumerate(pos)}
    D = np.zeros((2 * n, 2 * n), dtype=complex)
    for x in range(n):
        for j in range(3):
            for sgn in (1, -1):
                c = pos[x].copy()
                c[j] += sgn
                if j == 0:
                    if c[0] < 0 or c[0] >= L1:
                        continue
                else:
                    c[j] %= Lt
                y = idx[tuple(c)]
                D[2 * x:2 * x + 2, 2 * y:2 * y + 2] += sgn * SIG[j] / 2j
    return {"name": "slab", "nsite": n, "ncomp": 2, "DW": S2.herm(D), "coord": pos, "L1": L1, "Lt": Lt}


def herm(M):
    return 0.5 * (M + M.conj().T)


def sandwich(A, psi_modes):
    w, V = np.linalg.eigh(herm(A.conj().T @ A))
    Eh = (V * w ** 0.25) @ V.conj().T
    return herm(Eh @ np.diag(psi_modes) @ Eh), float(w.min())


def analyse(lat, H, n_lev):
    n = lat["nsite"]
    L1 = lat["L1"]
    x1 = lat["coord"][:, 0].astype(float)
    w, V = np.linalg.eigh(H)
    S1 = np.kron(np.eye(n), SX)
    k1 = 2 * np.pi * np.fft.fftfreq(L1)
    near0 = np.abs(np.angle(np.exp(1j * k1))) < np.pi / 4
    nearpi = np.abs(np.angle(np.exp(1j * (k1 - np.pi)))) < np.pi / 4
    rows = []
    for i in range(n_lev):
        v = V[:, i]
        s1 = float(np.real(v.conj() @ S1 @ v))
        dens = np.abs(v.reshape(n, 2)) ** 2
        site_w = dens.sum(1)
        xc = float(site_w @ x1)
        # x_1 profile per (transverse site, component), Fourier along x_1
        prof = v.reshape(lat["Lt"], lat["Lt"], L1, 2)                 # site index x = x1 + L1 (x2 + Lt x3)
        F = np.fft.fft(prof, axis=2)
        pw = (np.abs(F) ** 2).sum(axis=(0, 1, 3))
        pw = pw / pw.sum()
        rows.append({"E": float(w[i]), "sigma1": s1, "x1_centroid": xc, "w_k1_near_0": float(pw[near0].sum()), "w_k1_near_pi": float(pw[nearpi].sum())})
    return rows


def pairs(rows, g_u):
    out = []
    used = set()
    for i, r in enumerate(rows):
        if i in used or r["sigma1"] <= 0:
            continue
        val = "0" if r["w_k1_near_0"] >= r["w_k1_near_pi"] else "pi"
        best, bd = None, None
        for j, t in enumerate(rows):
            if j in used or j == i or t["sigma1"] >= 0:
                continue
            valj = "0" if t["w_k1_near_0"] >= t["w_k1_near_pi"] else "pi"
            if valj != val:
                continue
            d = abs(t["x1_centroid"] - r["x1_centroid"])
            if bd is None or d < bd:
                best, bd = j, d
        if best is not None:
            used.update({i, best})
            out.append({"valley_x1": val, "E_plus": r["E"], "E_minus": rows[best]["E"], "split": r["E"] - rows[best]["E"],
                        "split_over_g": (r["E"] - rows[best]["E"]) / g_u if g_u else None, "centroids": [r["x1_centroid"], rows[best]["x1_centroid"]],
                        "sigma1": [r["sigma1"], rows[best]["sigma1"]]})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--L1", type=int, default=128)
    ap.add_argument("--Lt", type=int, default=2)
    ap.add_argument("--nlev", type=int, default=16)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    lat = slab(a.L1, a.Lt)
    n = lat["nsite"]
    x1 = lat["coord"][:, 0].astype(float)
    D = lat["DW"]
    res = {"meta": {"L1": a.L1, "Lt": a.Lt, "note": "on a periodic direction of length 2 the sine stencil's transverse terms cancel"}, "rows": []}
    # the transverse part of D, measured: D with the x_1 links removed
    Dt = D.copy()
    for x in range(n):
        for sgn in (1, -1):
            c = lat["coord"][x].copy()
            c[0] += sgn
            if 0 <= c[0] < a.L1:
                y = x + sgn
                Dt[2 * x:2 * x + 2, 2 * y:2 * y + 2] = 0
    res["meta"]["transverse_operator_max_abs"] = float(np.max(np.abs(Dt)))
    for m in (0.1, 0.2):
        for g in (0.0, 5e-4, 1e-3, 2e-3):
            psi = 1 + g * (x1 - a.L1 / 2)
            psim = np.repeat(psi, 2)
            Pinv = np.diag(1 / psim)
            A_pol = m * Pinv + 1j * Pinv @ Pinv @ Pinv @ D
            P3 = Pinv @ Pinv @ Pinv
            A_sym = m * Pinv + 0.5j * (P3 @ D + D @ P3)
            # the library's construction for the polar placement, as a check
            pk = S2.Packet(lat, m, +1, psi, np.ones(n))
            Hpol_lib = pk.B(0)
            Hpol, mn1 = sandwich(A_pol, psim)
            Hsym, mn2 = sandwich(A_sym, psim)
            rp = analyse(lat, Hpol, a.nlev)
            rs = analyse(lat, Hsym, a.nlev)
            rec = {"m": m, "g_u": g, "lib_check": float(np.max(np.abs(Hpol - Hpol_lib))), "min_AdagA": [mn1, mn2],
                   "pol_levels": rp, "sym_levels": rs, "pol_pairs": pairs(rp, g), "sym_pairs": pairs(rs, g)}
            if g == 0:
                rec["control_pol_minus_sym_spectrum"] = float(np.max(np.abs(np.linalg.eigvalsh(Hpol) - np.linalg.eigvalsh(Hsym))))
            res["rows"].append(rec)
            print("m %g g %g: pol pairs %s | sym pairs %s (%.0f s)" % (m, g, json.dumps([round(p["split_over_g"], 4) if p["split_over_g"] else None for p in rec["pol_pairs"]]),
                                                                       json.dumps([round(p["split_over_g"], 4) if p["split_over_g"] else None for p in rec["sym_pairs"]]),
                                                                       time.time() - t0), flush=True)
    res["meta"]["seconds"] = time.time() - t0
    with open(os.path.join(a.out, "spingrad.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
