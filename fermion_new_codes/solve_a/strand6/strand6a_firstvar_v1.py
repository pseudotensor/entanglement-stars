#!/usr/bin/env python
"""Strand 6 (a'), the first-variation driver (spec solve/specs/SPEC_strand6_validation.md (a'); builder analyst).

On the open chain of L sites (D = the sine derivative of strand6a_spingrad_v3, chain_D), at g_u = 0: E = (m^2 + D^2)^(1/2) (A = m + i D, D
Hermitian), X = diag(x - x_0), C = -i[X, D]; Y solves E Y + Y E = C, Y_ij = C_ij/(e_i + e_j) in E's eigenbasis; the first-order splittings
(Delta/g_u)_pol^lin = 8 m <phi|Y|phi> and (Delta/g_u)_sym^lin = 2 m <phi|Y|phi>.  Beside them: (i) the central derivative of the version-3
Rayleigh-quotient difference Delta(g) = <phi|B_+(g)|phi> - <phi|B_-(g)|phi> (B_s = E_s^(1/2) psi E_s^(1/2), E_s = |A_s|, A_s = m Psi^-1 +
i Psi^-3 (s D) for the polar placement or the symmetric one, psi = 1 + g (x - x_0)) at g = +-2.5e-4 and +-1.25e-4, Richardson (4 D(h/2) -
D(h))/3; (ii) version 3's values Delta(g)/g at g = 5e-4, 1e-3, 2e-3 (recomputed here with the same code); (iii) the bulk packet average
4 m <cos k/(m^2 + sin^2 k)^(1/2)> over the packet's momentum distribution |phi^(k)|^2 on the infinite chain (phi^(k) = sum_x phi(x) e^(-i k x),
2^16 points).  Per packet also the B_0-space centroid <phi|X|phi>, the fermion-space centroid (1/2) <phi|(X + U^dag X U)|phi> with U = A E^-1
the polar unitary at g = 0, and the valley leakage: the weight of |phi^(k)|^2 in the half of the zone nearer the other valley (|k| > pi/2 for
valley 0, |k| < pi/2 for valley pi).  Packets: version 3's envelopes exp(-(x - x_0)^2/(2 w^2)) exp(i k_nu x), normalized after truncation, x_0
= L/2; m in {0.1, 0.2}, valley 0 and pi; L = 128 with w in {8, 16, 24}, and L = 256 with w in {24, 48}.  Writes <out>/firstvar.json."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import json
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from strand6a_spingrad_v3 import chain_D, sandwich  # noqa: E402

HS = (2.5e-4, 1.25e-4)
GS_V3 = (5e-4, 1e-3, 2e-3)


def B_ops(m, D1, x1, x0, g):
    psi = 1 + g * (x1 - x0)
    Pinv = np.diag(1 / psi)
    P3 = Pinv @ Pinv @ Pinv
    out = {}
    for place in ("pol", "sym"):
        for s in (+1, -1):
            A = m * Pinv + (1j * P3 @ (s * D1) if place == "pol" else 0.5j * (P3 @ (s * D1) + (s * D1) @ P3))
            out[(place, s)] = sandwich(A, psi)
    return out


def split(Bs, phi, place):
    return float(np.real(phi.conj() @ Bs[(place, 1)] @ phi) - np.real(phi.conj() @ Bs[(place, -1)] @ phi))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    res = {"meta": {"h": list(HS), "g_v3": list(GS_V3)}, "rows": []}
    for L, widths in ((128, (8.0, 16.0, 24.0)), (256, (24.0, 48.0))):
        D1 = chain_D(L)
        x1 = np.arange(L, dtype=float)
        x0 = L / 2
        X = np.diag(x1 - x0)
        C = -1j * (X @ D1 - D1 @ X)
        nk = 2 ** 16
        kk = 2 * np.pi * np.arange(nk) / nk
        kk = np.where(kk > np.pi, kk - 2 * np.pi, kk)
        for m in (0.1, 0.2):
            ev, V = np.linalg.eigh(m * m * np.eye(L) + D1 @ D1)
            e = np.sqrt(ev)
            Ct = V.conj().T @ C @ V
            Yt = Ct / (e[:, None] + e[None, :])
            Y = V @ Yt @ V.conj().T
            Einv = (V / e) @ V.conj().T
            U = (m * np.eye(L) + 1j * D1) @ Einv
            UXU = U.conj().T @ X @ U
            Bg = {g: B_ops(m, D1, x1, x0, g) for g in [s * h for h in HS for s in (1, -1)] + list(GS_V3)}
            for wd in widths:
                for nu, kn in (("0", 0.0), ("pi", np.pi)):
                    phi = np.exp(-(x1 - x0) ** 2 / (2 * wd * wd)) * np.exp(1j * kn * x1)
                    phi = phi / np.linalg.norm(phi)
                    yv = float(np.real(phi.conj() @ Y @ phi))
                    rec = {"L": L, "m": m, "width": wd, "valley": nu, "Y": yv, "pol_lin": 8 * m * yv, "sym_lin": 2 * m * yv}
                    for place in ("pol", "sym"):
                        Dh = {h: (split(Bg[h], phi, place) - split(Bg[-h], phi, place)) / (2 * h) for h in HS}
                        rec[place + "_derivative"] = (4 * Dh[HS[1]] - Dh[HS[0]]) / 3
                        rec[place + "_derivative_by_h"] = {str(h): Dh[h] for h in HS}
                        rec[place + "_v3_over_g"] = {str(g): split(Bg[g], phi, place) / g for g in GS_V3}
                    ph = np.fft.fft(np.concatenate([phi, np.zeros(nk - L)]))
                    w = np.abs(ph) ** 2
                    w = w / w.sum()
                    rec["bulk_average"] = float(np.sum(w * 4 * m * np.cos(kk) / np.sqrt(m * m + np.sin(kk) ** 2)))
                    rec["valley_leakage"] = float(np.sum(w[np.abs(kk) > np.pi / 2]) if nu == "0" else np.sum(w[np.abs(kk) < np.pi / 2]))
                    rec["centroid_B0"] = float(np.real(phi.conj() @ X @ phi))
                    rec["centroid_fermion"] = float(np.real(0.5 * (phi.conj() @ X @ phi + phi.conj() @ UXU @ phi)))
                    res["rows"].append(rec)
                    print("L %d m %g w %g valley %s: 8m<Y> %.8f pol derivative %.8f (v3 at 1e-3 %.8f) bulk %.8f | 2m<Y> %.8f sym derivative %.8f | "
                          "leak %.2e centroids %.4f %.4f (%.0f s)" % (L, m, wd, nu, rec["pol_lin"], rec["pol_derivative"], rec["pol_v3_over_g"]["0.001"],
                                                                     rec["bulk_average"], rec["sym_lin"], rec["sym_derivative"], rec["valley_leakage"],
                                                                     rec["centroid_B0"], rec["centroid_fermion"], time.time() - t0), flush=True)
    res["meta"]["seconds"] = time.time() - t0
    with open(os.path.join(a.out, "firstvar.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
