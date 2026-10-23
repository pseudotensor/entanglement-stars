#!/usr/bin/env python
"""engine_fields_v17_gate_v2.py (builder analyst): version 1 with Part 1 run at --gate-empty-rel (default 1e-6, the tolerance
engine_collapse_v26 passes; version 1 ran Part 1 at the module default 1e-14) and the tolerance printed in Part 1's rows. Version 1's
docstring: engine_fields_v17_gate_v1.py (builder analyst; the brief BRIEF_v26_builder_S4_S9, S4 (a): "engine_fields_v16 plus this rule;
bitwise v16 wherever the interval is absent, which you test on a stored v25 record"). Part 1 (the gate): per listed record of stored runs (the fields
npz's eps_tot, J_tot, R, g, ell), engine_fields_v16.solve_constraints_ivp and engine_fields_v17.solve_constraints_ivp with the engine's
options (collar policy flag, support_rel 1e-14, rtol 1e-12, atol 1e-16); bitwise when the site fields N, v, the coarse W, the charge E and
the dense N, W on the fine grid are equal to the last bit (max |difference| exactly 0), with v17's statement of the empty interior.
Part 2 (where the rule acts): the same on the listed massless-shell records at empty_rel 1e-14 (the default) and 1e-6: the empty interior's
end r_e, v16's N(0) and N(r_e) against v17's interior N, the omitted momentum-row increment, the enclosed charge at r_e against E, the
largest change of N outside the interval and of E. Every number printed by the script. CPU.
Args: --gate run_dir/tag:k1,k2,... (repeatable) --massless run_dir/tag:k1,... (repeatable) --md --json."""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_fields_v16 as F16  # noqa: E402
import engine_fields_v17 as F17  # noqa: E402

KW = dict(collar_policy="flag", support_rel=1e-14, rtol=1e-12, atol=1e-16)


def load(spec):
    path, ks = spec.split(":")
    d, tag = os.path.split(path)
    Z = np.load(os.path.join(d, "collapse_%s_fields.npz" % tag), allow_pickle=True)
    ks = [int(x) for x in ks.split(",")]
    T = np.asarray(Z["T"], float)
    ks = [k if k >= 0 else len(T) + k for k in ks]
    return tag, Z, ks, T


def maxdiff(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="append", default=[]); ap.add_argument("--massless", action="append", default=[])
    ap.add_argument("--gate-empty-rel", type=float, default=1e-6)
    ap.add_argument("--md", required=True); ap.add_argument("--json", required=True)
    a = ap.parse_args()
    out = {"script": "engine_fields_v17_gate_v2.py", "gate_empty_rel": a.gate_empty_rel, "gate": [], "massless": []}
    L = ["## engine_fields_v17 against engine_fields_v16 (engine_fields_v17_gate_v2.py; the brief's S4 (a))", "",
         "### Part 1: the gate on stored records at empty_rel %g (bitwise where the empty interior is absent)" % a.gate_empty_rel, "",
         "| run | record (T) | v17's empty interior | max dN sites | max dv sites | max dW nodes | dE | max dN dense | max dW dense | bitwise |", "|---|---|---|---|---|---|---|---|---|---|"]
    allbit = True
    for spec in a.gate:
        tag, Z, ks, T = load(spec)
        g = float(Z["g"]); ell = float(Z["ell"]); R = np.asarray(Z["R"], float); rs_ = None
        for k in ks:
            eps = np.asarray(Z["eps_tot"][k], float); J = np.asarray(Z["J_tot"][k], float)
            t0 = time.time()
            n6, v6, w6, i6 = F16.solve_constraints_ivp(eps, J, g, R, int(ell), **KW)
            n7, v7, w7, i7 = F17.solve_constraints_ivp(eps, J, g, R, int(ell), empty_rel=a.gate_empty_rel, **KW)
            rf = i6["rfine"]
            d = {"dN": maxdiff(n6, n7), "dv": maxdiff(v6, v7), "dW": maxdiff(w6, w7), "dE": abs(float(i6["E_collar"]) - float(i7["E_collar"])),
                 "dNd": maxdiff(i6["N_dense"](rf), i7["N_dense"](rf)), "dWd": maxdiff(i6["W_field"](rf), i7["W_field"](rf))}
            bit = all(v_ == 0.0 for v_ in d.values())
            allbit = allbit and bit
            ei = i7.get("empty_interior") or {}
            row = {"run": tag, "k": k, "T": float(T[k]), "empty_interior": ei.get("label") or ei.get("rule"), "r_e": ei.get("r_e"), **d, "bitwise": bit,
                   "module_v17": i7.get("module"), "seconds": time.time() - t0}
            out["gate"].append(row)
            L.append("| %s | %d (%.4f) | %s | %.3e | %.3e | %.3e | %.3e | %.3e | %.3e | %s |" % (tag, k, float(T[k]),
                     row["empty_interior"], d["dN"], d["dv"], d["dW"], d["dE"], d["dNd"], d["dWd"], "yes" if bit else "NO"))
            print("gate %s k %d: bitwise %s (%.1f s)" % (tag, k, bit, row["seconds"]), flush=True)
    L += ["", "- the record column prints the record index and its time T (lattice units); bitwise on every gate record: %s." % ("yes" if allbit else "NO"), ""]
    L += ["### Part 2: the massless-shell records (where the rule acts)", "",
          "| run | record (T) | empty_rel | r_e | v16 N(0) | v16 N(r_e) | v17 N interior | omitted dL (capped; uncapped) | enclosed charge at r_e over E | max dN outside r_e | dE/E | rho, J on [0, r_e] over their maxima | v16 gauge | v17 gauge |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for spec in a.massless:
        tag, Z, ks, T = load(spec)
        g = float(Z["g"]); ell = float(Z["ell"]); R = np.asarray(Z["R"], float)
        for k in ks:
            eps = np.asarray(Z["eps_tot"][k], float); J = np.asarray(Z["J_tot"][k], float)
            n6, v6, w6, i6 = F16.solve_constraints_ivp(eps, J, g, R, int(ell), **KW)
            for er in (1e-14, 1e-6):
                n7, v7, w7, i7 = F17.solve_constraints_ivp(eps, J, g, R, int(ell), empty_rel=er, **KW)
                if i6.get("failed") or i7.get("failed"):
                    row = {"run": tag, "k": k, "T": float(T[k]), "empty_rel": er, "failed_v16": i6.get("failed"), "failed_v17": i7.get("failed")}
                    out["massless"].append(row)
                    L.append("| %s | %d (%.3f) | %.0e | failed: v16 %s; v17 %s |  |  |  |  |  |  |  |  |  |  |" % (tag, k, float(T[k]), er, i6.get("failed"), i7.get("failed")))
                    continue
                ei = i7.get("empty_interior") or {}
                r_e = ei.get("r_e")
                rf = i6["rfine"]
                if r_e is not None:
                    outm = rf > r_e
                    dNo = maxdiff(i6["N_dense"](rf[outm]), i7["N_dense"](rf[outm]))
                    N6e = float(i6["N_dense"](np.array([r_e]))[0])
                else:
                    dNo = maxdiff(i6["N_dense"](rf), i7["N_dense"](rf)); N6e = None
                dE = abs(float(i7["E_collar"]) - float(i6["E_collar"])) / abs(float(i6["E_collar"]))
                g6 = (i6.get("gauge") or {}).get("label"); g7 = (i7.get("gauge") or {}).get("label")
                row = {"run": tag, "k": k, "T": float(T[k]), "empty_rel": er, "r_e": r_e, "N0_v16": float(i6["N_dense"](0.0)[0]), "N_re_v16": N6e,
                       "N_interior_v17": float(i7["N_dense"](0.0)[0]), "omitted_dL_capped": ei.get("omitted_dL_capped"), "omitted_dL_uncapped": ei.get("omitted_dL_uncapped"),
                       "enclosed_over_E": ei.get("enclosed_over_E"), "max_dN_outside": dNo, "dE_rel": dE, "rho_rel": ei.get("rho_max_rel_on_interval"),
                       "J_rel": ei.get("J_max_rel_on_interval"), "gauge_v16": g6, "gauge_v17": g7, "label": ei.get("label")}
                out["massless"].append(row)
                f = lambda x, fmt="%.6e": (fmt % x) if x is not None else "-"
                L.append("| %s | %d (%.3f) | %.0e | %s | %.10f | %s | %.10f | %s; %s | %s | %.3e | %.3e | %s; %s | %s | %s |" % (
                    tag, k, float(T[k]), er, f(r_e, "%.4f"), row["N0_v16"], f(N6e, "%.10f"), row["N_interior_v17"], f(ei.get("omitted_dL_capped"), "%.3e"),
                    f(ei.get("omitted_dL_uncapped"), "%.3e"), f(ei.get("enclosed_over_E"), "%.3e"), dNo, dE, f(ei.get("rho_max_rel_on_interval"), "%.1e"),
                    f(ei.get("J_max_rel_on_interval"), "%.1e"), g6, g7))
                print("massless %s k %d empty_rel %.0e: r_e %s" % (tag, k, er, r_e), flush=True)
    L += ["", "- N(0) is the lapse at the centre (the interior clock), normalized at the collar by both modules; the omitted dL is the momentum row's integral on [0, r_e] that v16 integrates and v17 replaces by N' = 0 (capped at 1/ell per unit length as v16's delta L bound; uncapped beside); every number above is printed by the script."]
    open(a.md, "w").write("\n".join(L) + "\n")
    json.dump(out, open(a.json, "w"), indent=1, default=float)
    print("\n".join(L))


if __name__ == "__main__":
    main()
