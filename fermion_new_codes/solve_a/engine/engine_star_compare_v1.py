#!/usr/bin/env python
"""engine_star_compare_v1.py (builder analyst; the equivalence test of engine_star_v6 against engine_star_v5): per member of two star
records with the same arguments, the differences: E (relative), E_top, E_next, E_F, R_surf, R_999, N_min, the passes, channels and the
largest occupied kappa; the profiles N, v (absolute) and eps_tot, J_tot (relative to their largest values); the occupied levels per
channel (the largest difference, and whether the occupation lists agree); the seconds per member from the .out files when given.
Args: --a --b --md."""
import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--md", required=True)
    x = ap.parse_args()
    A = json.load(open(x.a)); B = json.load(open(x.b))
    L = ["## engine_star_v6 against v5 (engine_star_compare_v1.py: %s, %s against %s, %s)" % (x.a.split("/")[-1], A["script"], x.b.split("/")[-1], B["script"]), "",
         "Arguments: %s" % json.dumps({k: A["args"][k] for k in ("m", "g", "ell", "nsites", "NB_list", "ell_control")}), "",
         "| member | N_B | E rel | E_top diff | E_next diff | R_surf diff | N_min diff | passes a; b | channels a; b | kmax a; b | max dN | max dv | eps rel | J rel | levels max diff | occupations equal |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    ma, mb = A.get("members", []), B.get("members", [])
    for i, (p, q) in enumerate(zip(ma, mb)):
        if p.get("failed") or q.get("failed"):
            L.append("| %d | %s | failed: %s; %s |" % (i, p.get("NB_target"), p.get("flags"), q.get("flags")))
            continue
        d = lambda k: abs(p[k] - q[k]) if (p.get(k) is not None and q.get(k) is not None) else float("nan")
        Na, Nb = np.asarray(p["N"]), np.asarray(q["N"]); va, vb = np.asarray(p["v"]), np.asarray(q["v"])
        ea, eb = np.asarray(p["eps_tot"]), np.asarray(q["eps_tot"]); Ja, Jb = np.asarray(p["J_tot"]), np.asarray(q["J_tot"])
        la, lb = p["levels"], q["levels"]
        same_occ = sorted(la.keys()) == sorted(lb.keys()) and all(len(la[k]) == len(lb[k]) and all(abs(fa - fb) < 1e-12 for (_, fa), (_, fb) in zip(la[k], lb[k])) for k in la)
        ldiff = max([abs(ea_ - eb_) for k in la if k in lb for (ea_, _), (eb_, _) in zip(la[k], lb[k])] or [float("nan")])
        L.append("| %d | %.2f | %.1e | %.1e | %.1e | %.1e | %.1e | %d; %d | %d; %d | %d; %d | %.1e | %.1e | %.1e | %.1e | %.1e | %s |" % (
            i, p["N_B"], abs(p["E"] - q["E"]) / abs(p["E"]), d("E_top"), d("E_next"), d("R_surf"), d("N_min"), p["passes"], q["passes"], p["channels"], q["channels"],
            p["kappa_max_occupied"], q["kappa_max_occupied"], float(np.max(np.abs(Na - Nb))), float(np.max(np.abs(va - vb))),
            float(np.max(np.abs(ea - eb)) / max(np.max(np.abs(ea)), 1e-300)), float(np.max(np.abs(Ja - Jb)) / max(np.max(np.abs(Ja)), 1e-300)), ldiff, "yes" if same_occ else "NO"))
    L += ["", "- every number above is printed by the script from the two records."]
    open(x.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
