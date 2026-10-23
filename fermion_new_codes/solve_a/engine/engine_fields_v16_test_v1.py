#!/usr/bin/env python
"""engine_fields_v16_test_v1.py (builder analyst; engine spec 24.1 (1)'s gate and the zero handling): (a) the gate: engine_fields_v16
against v14 on the production's first three stored records' shell totals, bitwise (E, the site fields, the dense lapse on the fine grid);
(b) the empty interiors: v14, v15 and v16 on the constructed empty interior of engine_fields_v14_test_v1 and on the massless shell's
first re-solves stored by the diagnostic engine_collapse_v23 (fields v15): per solve the verdict, E, the zeros (kind, r, U'), the stretches
and near-tangencies, the continued intervals with min U/U_ref and delta L_bound at two resolutions, the centre flag. Args: --prod-fields
--shell-fields --md."""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_fields_v14 as F14  # noqa: E402
import engine_fields_v15 as F15  # noqa: E402
import engine_fields_v16 as F16  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prod-fields", required=True)
    ap.add_argument("--shell-fields", default=None)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    Z = np.load(a.prod_fields, allow_pickle=True)
    R = np.asarray(Z["R"], float); g = float(Z["g"]); ell = float(Z["ell"])
    L = ["## engine_fields_v16: the gate against v14 and the empty interiors (engine_fields_v16_test_v1.py)", "", "### (a) The gate (the production's first three records)", "",
         "| record | E v14 | E v16 | bitwise E | max abs dN (sites) | max abs dv (sites) | max abs dN (dense, fine grid) |", "|---|---|---|---|---|---|---|"]
    for k in range(3):
        e = np.asarray(Z["eps_tot"][k], float); J = np.asarray(Z["J_tot"][k], float)
        N4, v4, W4, i4 = F14.solve_constraints_ivp(e, J, g, R, ell, collar_policy="flag")
        N6, v6, W6, i6 = F16.solve_constraints_ivp(e, J, g, R, ell, collar_policy="flag")
        rf = i4["rfine"]
        L.append("| %d | %.15f | %.15f | %s | %.2e | %.2e | %.2e |" % (k, F14.charge(i4, g), F16.charge(i6, g), F14.charge(i4, g) == F16.charge(i6, g),
                                                                    float(np.max(np.abs(N4 - N6))), float(np.max(np.abs(v4 - v6))), float(np.max(np.abs(i4["N_dense"](rf) - i6["N_dense"](rf))))))
    L += ["", "### (b) The empty interiors", "", "| case | module | verdict | E | zeros (kind r U') | stretches, near-tangencies | continued intervals | min U/U_ref | delta L_bound (ell/16, ell/32) | centre flag |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    cases = []
    e = np.asarray(Z["eps_tot"][2], float).copy(); J = np.asarray(Z["J_tot"][2], float).copy()
    inner = R < 60.0
    e[inner] = 0.0; J[inner] = 0.0; e[:20] = -1e-6 * float(np.max(np.abs(e)))
    cases.append(("constructed empty interior (record 2, R < 60 emptied, -1e-6 on 20 sites)", e, J, R, g, ell))
    if a.shell_fields and os.path.exists(a.shell_fields):
        S = np.load(a.shell_fields, allow_pickle=True)
        for k in range(1, min(3, len(S["T"]))):
            cases.append(("massless shell's re-solve at T %.3f (v23's stored sources)" % float(S["T"][k]), np.asarray(S["eps_tot"][k], float), np.asarray(S["J_tot"][k], float),
                          np.asarray(S["R"], float), float(S["g"]), float(S["ell"])))
    for lab, e_, J_, R_, g_, ell_ in cases:
        for name, M in (("v14", F14), ("v15", F15), ("v16", F16)):
            _, _, _, inf = M.solve_constraints_ivp(e_, J_, g_, R_, ell_, collar_policy="flag")
            if inf.get("failed"):
                L.append("| %s | %s | failed: %s | - | - | - | - | - | - | - |" % (lab, name, str(inf["failed"])[:120]))
                continue
            ga = inf.get("gauge") or {}
            zs = "; ".join("%s %.4f %.2e" % (z["kind"], z["r"], z["U_prime"]) for z in (ga.get("zeros") or []))[:200] or "-"
            L.append("| %s | %s | solved | %.10f | %s | %s, %s | %s | %.3e | %s | %s |" % (
                lab, name, M.charge(inf, g_), zs, ga.get("stretches", ga.get("t_stretches", "-")), ga.get("near_tangencies", ga.get("step_failure_zeros", "-")),
                ", ".join("[%.3f, %.3f]" % tuple(x) for x in (ga.get("intervals") or [])) or "-", ga.get("min_U_over_U_ref", float("nan")),
                ("%.2e, %.2e" % tuple(ga["delta_L_bound_two_resolutions"])) if ga.get("delta_L_bound_two_resolutions") else ("%.2e" % ga.get("delta_L_bound", 0.0)),
                inf.get("centre_flag") or "-"))
    L += ["", "- every number above is printed by the script."]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
