#!/usr/bin/env python
"""engine_fields_v14_test_v1.py (builder analyst): engine_fields_v14.solve_constraints_ivp against version 12's on stored shell totals
(the production's fields npz): per record the charge E, the dense lapse and the site fields' differences, the gauge and regime labels,
the centre series' check and N_centre (both versions), the r_0 -> r_0/2 check on the first and last records; then the gauge continuation
on a constructed empty interior (the record's dust removed inside --hollow-R, a negative density of --neg-frac of the largest shell total
on the first --neg-sites sites): version 12's verdict beside version 14's labels. Args: --fields npz --records list --md."""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_fields_v12 as F12  # noqa: E402
import engine_fields_v14 as F14  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", required=True)
    ap.add_argument("--records", default="0,1,2,40,88,97,101")
    ap.add_argument("--hollow-R", type=float, default=60.0)
    ap.add_argument("--neg-frac", type=float, default=1e-6)
    ap.add_argument("--neg-sites", type=int, default=20)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    Z = np.load(a.fields, allow_pickle=True)
    R = np.asarray(Z["R"], float); g = float(Z["g"]); ell = float(Z["ell"])
    L = ["## engine_fields_v14 against engine_fields_v12 on stored shell totals (engine_fields_v14_test_v1.py)", "", "Source: %s; g %g, ell %g." % (os.path.basename(a.fields), g, ell), "",
         "| record | T | E (v12) | E (v14) - E (v12), relative | max abs dN (sites) | max abs dv (sites) | N_centre v12 / v14 | gauge (v14) | regime max ratio | series check rho, J | seconds v12 / v14 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    ks = [int(x) for x in a.records.split(",")]
    T = np.asarray(Z["T"], float)
    for k in ks:
        if k >= len(T):
            continue
        e = np.asarray(Z["eps_tot"][k], float); J = np.asarray(Z["J_tot"][k], float)
        t0 = time.time(); N12, v12, W12, i12 = F12.solve_constraints_ivp(e, J, g, R, ell, collar_policy="flag"); s12 = time.time() - t0
        t0 = time.time(); N14, v14, W14, i14 = F14.solve_constraints_ivp(e, J, g, R, ell, collar_policy="flag"); s14 = time.time() - t0
        if i12.get("failed") or i14.get("failed"):
            L.append("| %d | %.1f | failed: v12 %s; v14 %s |" % (k, T[k], i12.get("failed"), i14.get("failed")))
            continue
        E12 = F12.charge(i12, g); E14 = F14.charge(i14, g)
        csc = i14["centre_series_check"]
        L.append("| %d | %.1f | %.12f | %+.3e | %.3e | %.3e | %.10f / %.10f | %s | %.3e | %.1e, %.1e | %.1f / %.1f |" % (
            k, T[k], E12, (E14 - E12) / E12, float(np.max(np.abs(N14 - N12))), float(np.max(np.abs(v14 - v12))), i12["ivp"]["N_centre"], i14["ivp"]["N_centre"],
            i14["gauge"]["label"], i14["regime"]["max_ratio"], csc["rho_spline_minus_series_rel"], csc["J_spline_minus_series_rel"], s12, s14))
        print(L[-1], flush=True)
        if k in (ks[0], ks[-1]):
            rc = F14.r0_check(e, J, g, R, ell, i14)
            L.append("| | r_0 -> r_0/2 (v14): %s |" % ", ".join("%s %s" % (k_, ("%.3e" % v_) if isinstance(v_, float) else v_) for k_, v_ in rc.items()))
            print(L[-1], flush=True)
    # the constructed empty interior
    k = ks[-1] if ks[-1] < len(T) else len(T) - 1
    e = np.asarray(Z["eps_tot"][k], float).copy(); J = np.asarray(Z["J_tot"][k], float).copy()
    inner = R < a.hollow_R
    e[inner] = 0.0; J[inner] = 0.0
    e[: a.neg_sites] = -a.neg_frac * float(np.max(np.abs(e)))
    _, _, _, i12 = F12.solve_constraints_ivp(e, J, g, R, ell, collar_policy="flag")
    _, _, _, i14 = F14.solve_constraints_ivp(e, J, g, R, ell, collar_policy="flag")
    L += ["", "### A constructed empty interior (record %d with its shell totals removed inside R %.0f and %.1e of the largest total negative on the first %d sites)" % (
        k, a.hollow_R, a.neg_frac, a.neg_sites), "",
          "- version 12: %s" % (i12.get("failed") or ("E %.12f" % F12.charge(i12, g))),
          "- version 14: %s; E %s; gauge: %s (delta L_bound %.3e, segments %d); regime: %s (max ratio %.3e); N_centre %s; W negative: %s" % (
              i14.get("failed") or "solved", ("%.12f" % F14.charge(i14, g)) if not i14.get("failed") else "-", (i14.get("gauge") or {}).get("label"),
              (i14.get("gauge") or {}).get("delta_L_bound", float("nan")), (i14.get("gauge") or {}).get("segments", 0), (i14.get("regime") or {}).get("label"),
              (i14.get("regime") or {}).get("max_ratio", float("nan")), (i14.get("ivp") or {}).get("N_centre"), i14.get("W_negative"))]
    print("\n".join(L[-3:]), flush=True)
    L += ["", "- every number above is printed by the script."]
    open(a.md, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
