#!/usr/bin/env python
"""engine_rows_static_diag_v1.py (builder analyst; a diagnostic of engine_rows_static_v2's finding): on the same static member, the
boosted-fluid relation S_r - S_t = J w (w = -v/N, the fluid's speed relative to the normal observers; exact for a perfect fluid at rest in
the static frame, since S_r - S_t = (e + P) w^2/F and J = (e + P) w/F) printed at the coarse nodes from the smoothed Noether stresses and
the smoothed momentum density J_ell of the same orbitals, at ell 4 and 8. Args: --family --NB --md."""
import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_rows_static_v2 as RS  # noqa: E402
import engine_fields_v14 as F14  # noqa: E402
import engine_stress_v1 as STR  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--NB", type=float, required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    import torch
    from engine_ops_v1 import sources_static
    Jf = json.load(open(a.family))
    mem = [q for q in Jf["members"] if not q.get("failed") and abs(q["N_B"] - a.NB) < 1e-3][0]
    m = float(mem["m"]); g = float(mem["g"]); nsites = int(mem["nsites"])
    N = np.asarray(mem["N"], float); v = np.asarray(mem["v"], float)
    R, chans = RS.orbitals(N, v, m, nsites, mem["N_B"], bool(Jf["args"].get("half_offset", 0)))
    eps = np.zeros(nsites); J = np.zeros(nsites); P = None
    for kappa, h, Q, U, f in chans:
        w = 16.0 * kappa
        e1, J1 = sources_static(h, Q, U, f, w)
        eps += e1; J += J1
        cs = STR.ChanStress(SimpleNamespace(hs=h.tocsr(), Qs=Q.tocsr(), kappa=kappa, w=w, R=R), torch, torch.device("cpu"))
        p = STR.stress_parts(cs, torch.as_tensor(np.asarray(U), dtype=torch.complex128), f)
        P = {k: w * np.asarray(x, float) for k, x in p.items()} if P is None else {k: P[k] + w * np.asarray(p[k], float) for k in P}
    SV = STR.shell_values(P, R, m)
    Rs = float(mem["R_surf"])
    L = ["## The boosted-fluid relation S_r - S_t = J w on the static member (engine_rows_static_diag_v1.py; N_B %.1f, R_surf %.1f)" % (mem["N_B"], Rs), ""]
    for ell in (4.0, 8.0):
        _, _, _, info = F14.solve_constraints_ivp(eps, J, g, R, ell, collar_policy="flag")
        rr = np.arange(ell, 1.1 * Rs, 2 * ell)
        Sr_l, St_l, _ = STR.smooth_stress(SV["S_r"], SV["S_t"], R, ell, rr)
        Jl = np.asarray(info["J_ell"](rr), float); rho = np.asarray(info["rho_ell"](rr), float)
        wv = -np.asarray(info["v_dense"](rr), float) / np.asarray(info["N_dense"](rr), float)
        L += ["### ell %g" % ell, "", "| r | rho_ell | J_ell | w | S_r,ell | S_t,ell | (S_r - S_t)/(J w) | S_t/rho |", "|---|---|---|---|---|---|---|---|"]
        for i in range(len(rr)):
            L.append("| %.1f | %.4e | %.4e | %.4f | %.4e | %.4e | %.4f | %.4f |" % (rr[i], rho[i], Jl[i], wv[i], Sr_l[i], St_l[i], (Sr_l[i] - St_l[i]) / (Jl[i] * wv[i]) if Jl[i] * wv[i] != 0 else float("nan"), St_l[i] / rho[i]))
        L.append("")
    L.append("- every number above is printed by the script.")
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
