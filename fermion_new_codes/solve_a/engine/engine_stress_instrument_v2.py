#!/usr/bin/env python
"""engine_stress_instrument_v2.py (builder analyst; engine spec 25.3, BUILDER: "the ladder (i) the regime member's channels filled with the
weights 16 kappa to N_B at N = 1, v = 0, (ii) the member's own fields, the virial and tangential checks at each step naming the filling or
the fields"): for a converged static member (engine_star_v7's record): step (i) every channel's positive levels at N = 1, v = 0 on the
member's chain (engine_star_v6.positive_levels, shift-invert Lanczos) pooled with the weights 16 kappa and filled from the bottom to the
member's N_B by the member's rule (engine_rows_static_v2's filling), step (ii) the member's own occupied columns at its stored site fields
(engine_rows_static_v2.orbitals); at each step the Noether stresses (engine_stress_v2, dust only: the reference is the ground state at the
step's fields) and the two checks as the static check printed them: the virial S_r + 2 S_t against rho - m b (per shell: the largest
difference over the shells relative to the largest right side; summed: the 4 pi R^2-weighted sums over the shells; smoothed at ell: the
K_0 projections at nodes ell apart) and the Noether S_t against its continuum form (kappa/R) <sigma_y>/(8 pi R^2) (per shell inside the
step's 0.9 radius of the energy, relative to the largest S_t there; summed); beside them the step's occupied channels, its highest filled
level, (p_F a)^2 = E_top^2 - m^2 at N = 1 (the filling's lattice momentum), and the radius holding 0.9 of the energy. Every number is
printed by the script. CPU. Args: --family json --NB value --md."""
import argparse
import json
import math
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_rows_static_v2 as RS2  # noqa: E402
import engine_star_v6 as S6  # noqa: E402
import engine_vacflow_v1 as VF  # noqa: E402
import engine_stress_v2 as STR2  # noqa: E402
from engine_ops_v1 import sources_static  # noqa: E402


def checks(chans, R, m, ell):
    import torch
    P = None; syk = np.zeros(len(R)); eps = np.zeros(len(R))
    for kappa, h, Q, U, f in chans:
        w = 16.0 * kappa
        e1, _ = sources_static(h, Q, U, f, w); eps += e1
        ch = SimpleNamespace(hs=h.tocsr(), Qs=Q.tocsr(), kappa=kappa, w=w, R=R)
        cs = STR2.ChanStress(ch, torch, torch.device("cpu"))
        p = STR2.stress_parts(cs, torch.as_tensor(np.asarray(U), dtype=torch.complex128), f)
        P = {k: w * np.asarray(x, float) for k, x in p.items()} if P is None else {k: P[k] + w * np.asarray(p[k], float) for k in P}
        syk += w * kappa * np.asarray(p["sy"], float)
    P["sy_k"] = syk
    SV = STR2.shell_values(P, R, m)
    V = 4.0 * math.pi * R ** 2
    lhs, rhs = SV["virial_lhs"], SV["virial_rhs"]
    cum = np.cumsum(eps) / max(float(np.sum(eps)), 1e-300)
    R90 = float(R[min(int(np.searchsorted(cum, 0.9)), len(R) - 1)])
    inside = R < R90
    per = float(np.max(np.abs(lhs - rhs))) / max(float(np.max(np.abs(rhs))), 1e-300)
    summ = abs(float(np.sum(V * (lhs - rhs)))) / max(abs(float(np.sum(V * rhs))), 1e-300)
    rr = np.arange(ell, R90, ell) if R90 > 2 * ell else np.array([ell])
    Srl, Stl, _ = STR2.smooth_stress(SV["S_r"], SV["S_t"], R, ell, rr)
    K0 = STR2.kernels_l(rr, R, ell, 0)
    rhs_l = K0 @ (V * rhs)
    smo = float(np.max(np.abs(Srl + 2.0 * Stl - rhs_l))) / max(float(np.max(np.abs(rhs_l))), 1e-300)
    stc = SV["S_t_continuum"]
    per_t = float(np.max(np.abs(SV["S_t"] - stc)[inside])) / max(float(np.max(np.abs(SV["S_t"][inside]))), 1e-300)
    sum_t = abs(float(np.sum((V * (SV["S_t"] - stc))[inside]))) / max(abs(float(np.sum((V * SV["S_t"])[inside]))), 1e-300)
    return {"virial_per_shell": per, "virial_summed": summ, "virial_smoothed": smo, "St_per_shell": per_t, "St_summed": sum_t, "R90": R90, "E": float(eps.sum())}


def flat_filling(m, nsites, NB, half_offset, kappa_cap=400):
    """step (i): every channel's positive levels at N = 1, v = 0 up to an energy E_up raised (its excess over m doubled) until the pooled
    weight 16 kappa below it reaches N_B (a channel whose lowest level lies above E_up ends the channel loop: the centrifugal barrier
    raises it with kappa), then the member's filling rule (engine_rows_static_v2.orbitals': ascending, degenerate levels together, at most
    one multiplet fractional). Returns (R, [(kappa, h, Q, U_occ, f_occ)], E_top)."""
    R = np.arange(nsites, dtype=float) + (0.5 if half_offset else 1.0)
    N1 = np.ones(nsites); v0 = np.zeros(nsites)
    E_up = m * 1.001
    while True:
        ep_all, store = [], {}
        for kappa in range(1, kappa_cap + 1):
            h, Q = VF.chain_sparse(R, m, float(kappa))
            H = S6.clocked_sparse(h, Q, N1, v0)
            ep, Up, _ = S6.positive_levels(H, m, upto=E_up)
            sel = ep <= E_up
            if not sel.any():
                break
            ep_all.append((kappa, ep[sel])); store[kappa] = (h, Q, Up[:, sel])
        if sum(16.0 * k * len(e) for k, e in ep_all) >= NB:
            break
        E_up = m + 2.0 * (E_up - m)
    lev = sorted([(float(x), 16.0 * k, ic, j) for ic, (k, ep) in enumerate(ep_all) for j, x in enumerate(ep)])
    occ = [np.zeros(len(ep)) for _, ep in ep_all]
    left = float(NB); i = 0; E_top = None
    while i < len(lev) and left > 1e-12:
        j = i
        while j < len(lev) and abs(lev[j][0] - lev[i][0]) < 1e-9:
            j += 1
        w = sum(x[1] for x in lev[i:j]); frac = min(1.0, left / w)
        for x in lev[i:j]:
            occ[x[2]][x[3]] = frac
        left -= frac * w; E_top = lev[i][0]; i = j
    out = []
    for (kappa, ep), f in zip(ep_all, occ):
        o_ = f > 1e-15
        if o_.any():
            h, Q, Up = store[kappa]
            out.append((kappa, h, Q, Up[:, o_], f[o_]))
    return R, out, E_top


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--NB", type=float, required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    Jf = json.load(open(a.family))
    mem = [q for q in Jf["members"] if not q.get("failed") and abs(q["N_B"] - a.NB) < 1e-3][0]
    m = float(mem["m"]); ell = float(mem["ell"]); nsites = int(mem["nsites"]); half = bool(Jf["args"].get("half_offset", 0))
    rows = []
    Rf, chf, Etop_f = flat_filling(m, nsites, mem["N_B"], half)
    Rm, chm = RS2.orbitals(np.asarray(mem["N"], float), np.asarray(mem["v"], float), m, nsites, mem["N_B"], half)
    for lab, R, chans, etop in (("(i) the filling at N = 1, v = 0", Rf, chf, Etop_f), ("(ii) the member's own fields", Rm, chm, None)):
        c = checks(chans, R, m, ell)
        c["E_top"] = etop
        rows.append((lab, chans, c))
    L = ["## The stress instrument's ladder: the filling against the fields (engine_stress_instrument_v2.py)", "",
         "Member: %s, N_B %.3f, E %.6f, R_surf %.2f (m a %g, ell %g, %d sites)." % (os.path.basename(a.family), mem["N_B"], mem["E"], mem["R_surf"], m, ell, nsites), "",
         "| step | channels (kappa: occupied levels) | highest filled level, (p_F a)^2 at N = 1 | energy | radius of 0.9 of the energy | virial per shell | virial summed | virial smoothed at ell | S_t against its continuum form per shell (inside) | summed (inside) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for lab, chans, c in rows:
        chs = "; ".join("%d: %d" % (kappa, U.shape[1]) for kappa, h, Q, U, f in chans)
        et = ("%.6f, %.3e" % (c["E_top"], c["E_top"] ** 2 - m * m)) if c.get("E_top") is not None else "-"
        L.append("| %s | %s | %s | %.6f | %.1f | %.3e | %.3e | %.3e | %.3e | %.3e |" % (lab, chs, et, c["E"], c["R90"], c["virial_per_shell"], c["virial_summed"], c["virial_smoothed"], c["St_per_shell"], c["St_summed"]))
    L += ["", "- the reference at each step is the ground state at that step's fields, so the stresses are the occupied columns' (dust) alone; every number above is printed by the script."]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
