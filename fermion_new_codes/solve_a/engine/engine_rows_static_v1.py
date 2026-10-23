#!/usr/bin/env python
"""engine_rows_static_v1.py (builder analyst; engine spec 20.1's first check of the rows: "on every static member they vanish to the
(pa)^2 and (ell/R_*)^2 floors with d_t k = 0"): a converged member of a static star family (engine_star_v6's record: its site fields and
shell totals) rebuilt: per channel the positive levels of the clocked chain at the member's fields (engine_star_v6.positive_levels,
shift-invert Lanczos), the zero-temperature filling at the member's N_B (engine_star_v6's rule), the occupied columns; the rebuilt shell
totals against the stored ones; the Noether stresses of these orbitals (engine_stress_v1 per channel with the weight 16 kappa, the dust
alone: the member's reference is the ground state at its fields, so the pairs vanish), the balance identity and the virial check; the
dense fields re-solved from the shell totals (engine_fields_v14); the stresses smoothed at the coarse nodes; the two rows' brackets
(engine_rows_v2.brackets; d_t k = 0) relative to their largest term, inside the star (r below 0.9 R_surf), across its surface and outside,
at h = ell/32 and ell/64 (the derivative's control); beside them the continuum form of S_t ((kappa/R) <sigma_y>/(8 pi R^2)) against the
Noether S_t. Args: --family json --NB value --md."""
import argparse
import json
import math
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_star_v6 as S6  # noqa: E402
import engine_vacflow_v1 as VF  # noqa: E402
import engine_fields_v14 as F14  # noqa: E402
import engine_stress_v1 as STR  # noqa: E402
import engine_rows_v2 as RW  # noqa: E402


def orbitals(N, v, m, nsites, NB, half_offset, kappa_cap=600):
    """engine_star_v6.channel_pass's levels and filling, returning the occupied columns per channel."""
    R = np.arange(nsites, dtype=float) + (0.5 if half_offset else 1.0)
    ep_all, store = [], {}
    above = 0
    for kappa in range(1, kappa_cap + 1):
        h, Q = VF.chain_sparse(R, m, float(kappa))
        H = S6.clocked_sparse(h, Q, N, v)
        ep, Up, kk = S6.positive_levels(H, m)
        ep_all.append((kappa, ep)); store[kappa] = (h, Q, Up, H)
        lv = np.concatenate([x[1] for x in ep_all]); wv = np.concatenate([np.full(len(x[1]), 16.0 * x[0]) for x in ep_all])
        o = np.argsort(lv); cw = np.cumsum(wv[o])
        E_top_est = float(lv[o][min(int(np.searchsorted(cw, NB - 1e-9)), len(lv) - 1)]) if cw[-1] >= NB else np.inf
        above = above + 1 if ep.min() > E_top_est else 0
        if above >= 2:
            break
    lev = sorted([(float(x), 16.0 * k, ic, j) for ic, (k, ep) in enumerate(ep_all) for j, x in enumerate(ep)])
    occ = [np.zeros(len(ep)) for _, ep in ep_all]
    left = float(NB); i = 0
    while i < len(lev):
        j = i
        while j < len(lev) and abs(lev[j][0] - lev[i][0]) < 1e-9:
            j += 1
        if left <= 1e-12:
            break
        w = sum(x[1] for x in lev[i:j]); frac = min(1.0, left / w)
        for x in lev[i:j]:
            occ[x[2]][x[3]] = frac
        left -= frac * w
        i = j
    out = []
    for (kappa, ep), f in zip(ep_all, occ):
        o_ = f > 1e-15
        if not o_.any():
            continue
        h, Q, Up, _ = store[kappa]
        out.append((kappa, h, Q, Up[:, o_], f[o_]))
    return R, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--NB", type=float, required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    import torch
    Jf = json.load(open(a.family))
    mem = [q for q in Jf["members"] if not q.get("failed") and abs(q["N_B"] - a.NB) < 1e-3][0]
    m = float(mem["m"]); g = float(mem["g"]); ell = float(mem["ell"]); nsites = int(mem["nsites"])
    N = np.asarray(mem["N"], float); v = np.asarray(mem["v"], float)
    R, chans = orbitals(N, v, m, nsites, mem["N_B"], bool(Jf["args"].get("half_offset", 0)))
    from engine_ops_v1 import sources_static
    eps = np.zeros(nsites); J = np.zeros(nsites)
    P = None
    syk = np.zeros(nsites)
    for kappa, h, Q, U, f in chans:
        w = 16.0 * kappa
        e1, J1 = sources_static(h, Q, U, f, w)
        eps += e1; J += J1
        ch = SimpleNamespace(hs=h.tocsr(), Qs=Q.tocsr(), kappa=kappa, w=w, R=R)
        cs = STR.ChanStress(ch, torch, torch.device("cpu"))
        p = STR.stress_parts(cs, torch.as_tensor(np.asarray(U), dtype=torch.complex128), f)
        P = {k: w * np.asarray(x, float) for k, x in p.items()} if P is None else {k: P[k] + w * np.asarray(p[k], float) for k in P}
        syk += w * kappa * np.asarray(p["sy"], float)
    P["sy_k"] = syk
    es = np.asarray(mem["eps_tot"], float); Js = np.asarray(mem["J_tot"], float)
    d_e = float(np.max(np.abs(eps - es))) / float(np.max(np.abs(es))); d_j = float(np.max(np.abs(J - Js))) / max(float(np.max(np.abs(Js))), 1e-300)
    SV = STR.shell_values(P, R, m)
    _, _, _, info = F14.solve_constraints_ivp(eps, J, g, R, ell, collar_policy="flag")
    Rs = float(mem["R_surf"])
    L = ["## The rows on a static lattice member (engine_rows_static_v1.py)", "",
         "Member: %s, N_B %.3f, E %.6f, R_surf %.2f, 2gE/R_surf %.4f (m a %g, g %g, ell %g, %d sites, %d channels occupied); the rebuilt shell totals against the stored ones: eps %.2e, J %.2e (relative); "
         "the balance identity's largest difference %.2e; the virial residual %.2e relative; E of the v14 solve %.6f." % (
             os.path.basename(a.family), mem["N_B"], mem["E"], Rs, 2 * g * mem["E"] / Rs, m, g, ell, nsites, len(chans), d_e, d_j, SV["identity_max_abs"],
             float(np.max(np.abs(SV["virial_lhs"] - SV["virial_rhs"]))) / float(np.max(np.abs(SV["virial_rhs"]))), F14.charge(info, g)), ""]
    inside = R < 0.9 * Rs
    stc = SV.get("S_t_continuum")
    if stc is not None:
        L.append("- S_t: the Noether shell value against its continuum form (kappa/R) <sigma_y>/(8 pi R^2), largest difference inside 0.9 R_surf relative to the largest S_t there: %.2e." % (
            float(np.max(np.abs(SV["S_t"][inside] - stc[inside]))) / float(np.max(np.abs(SV["S_t"][inside])))))
    L += ["", "| region | h | max abs rr bracket / max term | max abs tt bracket / max term |", "|---|---|---|---|"]
    for lab, lo, hi in (("inside, ell to 0.9 R_surf", ell, 0.9 * Rs), ("across the surface, 0.9 to 1.1 R_surf", 0.9 * Rs, 1.1 * Rs), ("outside, 1.1 R_surf to the collar - 2 ell", 1.1 * Rs, float(info["r_collar"]) - 2 * ell)):
        rr = np.arange(math.ceil(lo / ell) * ell, hi, ell)
        if not len(rr):
            continue
        Sr_l, St_l, tr = STR.smooth_stress(SV["S_r"], SV["S_t"], R, ell, rr)
        rho = np.asarray(info["rho_ell"](rr), float)
        for hf in (1.0 / 32.0, 1.0 / 64.0):
            b = RW.brackets(info["N_dense"], info["v_dense"], rho, Sr_l, St_l, g, rr, ell * hf)
            L.append("| %s | ell/%d | %.2e | %.2e |" % (lab, round(1 / hf), float(np.max(np.abs(b["rhs_rr"]) / b["terms_rr"])), float(np.max(np.abs(b["rhs_tt"]) / b["terms_tt"]))))
    L += ["", "- d_t k = 0 for a static member, so the brackets are the residuals; the floors expected: the stress dictionary's (p a)^2 and the smoothing's (ell/R_*)^2.",
          "- every number above is printed by the script."]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
