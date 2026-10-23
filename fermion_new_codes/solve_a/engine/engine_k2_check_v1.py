#!/usr/bin/env python
"""engine_k2_check_v1.py (builder analyst; red team R27a item 1: "the trace identity cancels K_2 identically ... check K_2 against
independent angular quadrature"): a shell of radius R carrying the tensor S_ij = S_t delta_ij + (S_r - S_t) n_i n_j (unit shell totals),
smoothed by the normalized 3D Gaussian of width ell; the smoothed radial-radial and tangential components at the point r z_hat by direct
quadrature over the shell's directions (Gauss-Legendre in cos(theta) times a uniform azimuth grid), against engine_stress_v1.smooth_stress's
kernel form [S_t]_0 + (1/3)[S_r - S_t]_0 + (2/3)[S_r - S_t]_2 and [S_t]_0 + (1/3)[S_r - S_t]_0 - (1/3)[S_r - S_t]_2, at several (r, R, ell),
including z = rR/ell^2 near the series switch 1e-2. Args: --md."""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_stress_v1 as STR  # noqa: E402


def direct(r, R, ell, Sr, St, nth=400, nph=64):
    x, w = np.polynomial.legendre.leggauss(nth)        # cos(theta)
    ph = np.linspace(0.0, 2.0 * math.pi, nph, endpoint=False)
    ct = x[:, None]; st = np.sqrt(1.0 - ct ** 2)
    nx = st * np.cos(ph)[None, :]; ny = st * np.sin(ph)[None, :]; nz = ct * np.ones_like(nx)
    d2 = (r * 0.0 - R * nx) ** 2 + (0.0 - R * ny) ** 2 + (r - R * nz) ** 2
    G = np.exp(-d2 / (2.0 * ell ** 2)) / ((2.0 * math.pi * ell ** 2) ** 1.5)
    dA = (w[:, None] * (2.0 * math.pi / nph))            # the solid-angle weights; unit shell total = 1 over 4 pi
    norm = 1.0 / (4.0 * math.pi)
    # at the point r z_hat the radial direction is z_hat: S_zz = S_t + (S_r - S_t) nz^2; the tangential S_xx = S_t + (S_r - S_t) nx^2
    Szz = St + (Sr - St) * nz ** 2; Sxx = St + (Sr - St) * nx ** 2
    return float(np.sum(G * Szz * dA) * norm), float(np.sum(G * Sxx * dA) * norm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    L = ["## K_2 against direct angular quadrature (engine_k2_check_v1.py)", "",
         "| r | R | ell | z = rR/ell^2 | S_r,ell direct | S_r,ell kernels | relative difference | S_t,ell direct | S_t,ell kernels | relative difference |", "|---|---|---|---|---|---|---|---|---|---|"]
    worst = 0.0
    for (r, R, ell) in [(8.0, 40.0, 8.0), (40.0, 40.0, 8.0), (30.0, 34.0, 4.0), (100.0, 104.0, 8.0), (0.05, 2.0, 4.0), (0.2, 2.0, 4.0), (1.0, 5.0, 16.0), (200.0, 210.0, 16.0)]:
        Sr, St = 1.0, 0.3
        Vshell = 4.0 * math.pi * R ** 2                   # smooth_stress takes shell values X with totals 4 pi R^2 X
        dr, dt = direct(r, R, ell, Sr, St)
        kr, kt, _ = STR.smooth_stress(np.array([Sr / Vshell]), np.array([St / Vshell]), np.array([R]), ell, np.array([r]))
        kr, kt = float(kr[0]), float(kt[0])
        e1 = abs(kr - dr) / max(abs(dr), 1e-300); e2 = abs(kt - dt) / max(abs(dt), 1e-300)
        worst = max(worst, e1, e2)
        L.append("| %g | %g | %g | %.3g | %.10e | %.10e | %.2e | %.10e | %.10e | %.2e |" % (r, R, ell, r * R / ell ** 2, dr, kr, e1, dt, kt, e2))
    L += ["", "- the largest relative difference %.2e; every number above is printed by the script." % worst]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
