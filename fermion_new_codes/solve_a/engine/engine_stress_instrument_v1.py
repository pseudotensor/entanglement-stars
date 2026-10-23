#!/usr/bin/env python
"""engine_stress_instrument_v1.py (builder analyst; engine spec 24.2: "the instrument's own control: one bound positive-energy state of one
channel at N = 1, v = 0 in a wide box, where S_r + 2 S_t = rho - m b must hold to (pa)^2 and the Noether S_t must match its continuum form;
a member-independent 0.25 is a factor in a definition before it is physics"): the flat radial chain of channel kappa (engine_collapse_v5.Chan,
2 components per site, m a given) at N = 1, v = 0; its positive-energy eigenstates (the box's standing waves) of increasing lattice momentum;
per state the stresses of engine_stress_v2 (the Noether flux of Q: S_r at the shells, S_t = f/(8 pi R)), the virial check S_r + 2 S_t against
rho - m b and the Noether S_t against its continuum form (kappa/R) <sigma_y>/(8 pi R^2), each (i) per shell (the largest difference over the
shells away from the centre and the box's end, relative to the largest rhs), (ii) for the shell totals summed (4 pi R^2 times each side,
summed over the same shells: the integrated identity), and (iii) after the Gaussian smoothing at ell (the hydrodynamic window); beside them
the state's (p a)^2 from its energy (E^2 = m^2 + sin^2 p). Args: --kappa --m --nsites --states --ell --md."""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_collapse_v5 as C5  # noqa: E402
import engine_stress_v2 as STR  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kappa", type=int, default=1)
    ap.add_argument("--m", type=float, default=0.2)
    ap.add_argument("--nsites", type=int, default=2048)
    ap.add_argument("--states", default="0,4,16,64,256")
    ap.add_argument("--ell", type=float, default=8.0)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    C5.backend(a.gpu)
    torch = C5.BK["torch"]
    ch = C5.Chan(a.kappa, a.m, a.nsites, False)
    R = np.asarray(ch.R, float)
    N = np.ones(a.nsites); v = np.zeros(a.nsites)
    e, U = C5.eigT(ch.clocked_dense(N, v))
    e_np = e.real.cpu().numpy()
    posi = np.where(e_np > 0)[0]
    cs = STR.ChanStress(ch, torch, C5.BK["dev"])
    sel = (R > 4.0 * a.ell) & (R < R[-1] - 4.0 * a.ell)
    rr = np.arange(4.0 * a.ell, R[-1] - 4.0 * a.ell, a.ell)
    L = ["## The stress instrument's control: single states of one channel at N = 1, v = 0 (engine_stress_instrument_v1.py)", "",
         "Channel kappa %d, m a %g, %d sites (a box of radius %d); the shells from 4 ell to the box's end less 4 ell; ell %g for the smoothed check." % (a.kappa, a.m, a.nsites, a.nsites, a.ell), "",
         "| state | E | (p a)^2 | virial per shell | virial summed | virial smoothed | S_t against its continuum form per shell | summed | smoothed |",
         "|---|---|---|---|---|---|---|---|---|"]
    for k in [int(x) for x in a.states.split(",")]:
        if k >= len(posi):
            continue
        j = posi[k]
        col = U[:, j:j + 1]
        P = STR.stress_parts(cs, col, np.ones(1))
        P = {kk: np.asarray(vv, float) for kk, vv in P.items()}
        P["sy_k"] = a.kappa * P["sy"]
        SV = STR.shell_values(P, R, a.m)
        lhs = SV["virial_lhs"]; rhs = SV["virial_rhs"]
        V = 4.0 * math.pi * R ** 2
        per = float(np.max(np.abs(lhs - rhs)[sel])) / max(float(np.max(np.abs(rhs[sel]))), 1e-300)
        summ = abs(float(np.sum((V * (lhs - rhs))[sel]))) / max(abs(float(np.sum((V * rhs)[sel]))), 1e-300)
        Srl, Stl, _ = STR.smooth_stress(SV["S_r"], SV["S_t"], R, a.ell, rr)
        K0 = STR.kernels_l(rr, R, a.ell, 0)
        rho_l = K0 @ (V * SV["rho"]); b_l = K0 @ (V * SV["b"])
        rhs_l = rho_l - a.m * b_l; lhs_l = Srl + 2.0 * Stl
        smo = float(np.max(np.abs(lhs_l - rhs_l))) / max(float(np.max(np.abs(rhs_l))), 1e-300)
        stc = SV["S_t_continuum"]
        per_t = float(np.max(np.abs(SV["S_t"] - stc)[sel])) / max(float(np.max(np.abs(SV["S_t"][sel]))), 1e-300)
        sum_t = abs(float(np.sum((V * (SV["S_t"] - stc))[sel]))) / max(abs(float(np.sum((V * SV["S_t"])[sel]))), 1e-300)
        stc_l = K0 @ (V * stc); st0_l = K0 @ (V * SV["S_t"])
        smo_t = float(np.max(np.abs(st0_l - stc_l))) / max(float(np.max(np.abs(st0_l))), 1e-300)
        Ej = float(e_np[j]); pa2 = max(Ej * Ej - a.m * a.m, 0.0)
        L.append("| %d | %.6f | %.3e | %.2e | %.2e | %.2e | %.2e | %.2e | %.2e |" % (k, Ej, pa2, per, summ, smo, per_t, sum_t, smo_t))
        print(L[-1], flush=True)
    L += ["", "- (p a)^2 is sin^2 p from the state's energy (E^2 = m^2 + sin^2 p at R much larger than kappa); every number above is printed by the script."]
    open(a.md, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
