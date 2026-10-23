#!/usr/bin/env python
"""engine_diag_ivp_dump_v1.py (builder analyst; spec 19.3's diagnosis, step 2): a failed solve's shell totals (engine_diag_turn27_dump_v1's
dump) re-solved by engine_fields_v13.solve_constraints_ivp without the floor rule (version 12's equations: the failure reproduced, with the
integrator's last r, U, J_ell, rho_ell, q and the ln N integrand) and with it (the region, the omitted term's scale, U's minimum); and the
profiles of rho_ell, J_ell and the enclosed-energy scale over the interior. Args: --dump --md."""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_fields_v13 as F13  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    Z = np.load(a.dump, allow_pickle=True)
    eps = np.asarray(Z["eps"], float); J = np.asarray(Z["J"], float); g = float(Z["g"]); R = np.asarray(Z["R"], float); ell = float(Z["ell"])
    L = ["## The failed solve re-solved (engine_diag_ivp_dump_v1.py over %s; the run's message: %s)" % (os.path.basename(a.dump), str(Z["failed"])), ""]
    for fr in (False, True):
        N, v, W, info = F13.solve_constraints_ivp(eps, J, g, R, ell, collar_policy="flag", support_rel=1e-14, floor_rule=fr)
        L.append("- floor rule %s: %s" % ("on" if fr else "off (version 12's equations)", ("FAILED: " + info["failed"]) if info.get("failed") else "solved, E %.10f" % info["E_collar"]))
        if info.get("failure_detail"):
            L.append("  - failure detail: %s" % {k: (float("%.4g" % v_) if isinstance(v_, float) else v_) for k, v_ in info["failure_detail"].items()})
        if info.get("floor_region"):
            fr_ = info["floor_region"]
            L.append("  - floor region: empty %s, %d intervals %s, %d points, omitted-term scale %.3e (%s), U min / U_ref %.3e" % (
                fr_["empty"], fr_["n_intervals"], fr_["intervals"], fr_["n_points"], fr_["omitted_scale"], "lapse unresolved" if fr_["lapse_unresolved"] else "resolved", fr_["U_min_over_Uref"]))
    src = F13.SmoothedSources5(eps, J, R, ell, float(R[-1]), 1e-14)
    rr = np.arange(0.0, float(R[-1]) + 1e-9, ell / 4.0)
    rho = np.asarray(src.rho(rr), float); Jl = np.asarray(src.J(rr), float)
    Wn = 8.0 * np.pi * g * np.concatenate([[0.0], np.cumsum(0.5 * (rr[1:] ** 2 * rho[1:] + rr[:-1] ** 2 * rho[:-1]) * np.diff(rr))])
    L += ["", "The interior profiles (every 2 ell up to 40 ell, then every 8 ell): r, rho_ell, J_ell, the enclosed-energy scale 8 pi g int r^2 rho_ell dr (U's value without the flow term), over its final value:", "",
          "| r | rho_ell | J_ell | 8 pi g int r^2 rho dr | over the final |", "|---|---|---|---|---|"]
    for i in list(range(0, min(len(rr), 160), 8)) + list(range(160, len(rr), 32)):
        L.append("| %.2f | %.3e | %.3e | %.3e | %.2e |" % (rr[i], rho[i], Jl[i], Wn[i], Wn[i] / max(abs(Wn[-1]), 1e-300)))
    neg = Wn < 0
    L += ["", "- the enclosed-energy scale is negative on %d of %d points (first at r = %s, smallest %.3e = %.2e of the final)." % (
        int(neg.sum()), len(rr), ("%.2f" % rr[np.argmax(neg)]) if neg.any() else "-", float(Wn.min()), float(Wn.min()) / max(abs(Wn[-1]), 1e-300)),
          "- every number above is printed by the script."]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L[:12]))


if __name__ == "__main__":
    main()
