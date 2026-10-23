#!/usr/bin/env python
"""engine_certify_v1.py (builder analyst; engine 17.3 (1) and (2)): the certificates of a collapse history's stored slices by post-processing.
Per slice of a fields npz (the shell totals eps_tot and J_tot, the run's ell and collar policy): the constraints re-solved by engine_fields_v8
(the same solver as the run's version 7, with its iteration history; starting from the stored lapse), then engine_fields_v8.certify: q_c and
the pre-floor q_c, eta, eps_map (rtol 1e-10 against 1e-12), B = (eta + eps_map)/(1 - q_c) or "uncertified", the U = W/N^2 and ln N routes'
defects, the residual ladder and the matching region's residuals, the collar's tail sums (delta E and delta(1/N), signed and absolute) and
W_tail; the re-solve's largest difference from the stored site fields is printed as the replay check. --tol below the run's 1e-12 forces the
floor path (the first test of that path's certificate, 17.3 (1)). Args: --fields --g --slices --every --tol --collar-policy --md --out-json."""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_fields_v8 as F8  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", required=True)
    ap.add_argument("--g", type=float, default=0.25)
    ap.add_argument("--slices", type=int, nargs="*", default=None)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--tol", type=float, default=1e-12)
    ap.add_argument("--collar-policy", default="flag")
    ap.add_argument("--md", required=True)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    Z = np.load(a.fields, allow_pickle=True)
    R = np.asarray(Z["R"], float); ell = int(Z["ell"])
    T = np.asarray(Z["T"], float)
    idx = a.slices if a.slices else list(range(0, len(T), a.every))
    rows = []
    L = ["## Certificates of the stored slices (engine_certify_v1.py on %s; ell %d; tol %g; collar policy %s)" % (os.path.basename(a.fields), ell, a.tol, a.collar_policy), "",
         "| slice | T | iterations (floor) | q_c; pre-floor | eta | eps_map | B; pre-floor | certified | U defect | ln N defect | residuals W, 1/N (ladder ell/32) | matching region W, 1/N | collar: dE/E abs; d(1/N) abs; certified | replay max dN, dv |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for i in idx:
        t0 = time.time()
        eps = np.asarray(Z["eps_tot"][i], float); J = np.asarray(Z["J_tot"][i], float); N0 = np.asarray(Z["N"][i], float)
        N, v, W, info = F8.solve_constraints(eps, J, a.g, R, ell, N0=N0, tol=a.tol, collar_policy=a.collar_policy)
        if info.get("failed"):
            rows.append({"slice": i, "T": float(T[i]), "failed": info["failed"]})
            L.append("| %d | %.1f | failed: %s |" % (i, T[i], info["failed"]) + " - |" * 12)
            print(L[-1], flush=True)
            continue
        c = F8.certify(eps, J, a.g, R, ell, N, v, info, collar_policy=a.collar_policy)
        replay = (float(np.max(np.abs(N - np.asarray(Z["N"][i], float)))), float(np.max(np.abs(v - np.asarray(Z["v"][i], float)))))
        rl = c.get("residual_ladder") or {}
        rm = c.get("residual_matching_region") or {}
        fl = info.get("converged_at_floor")
        f = lambda x, fmt="%.2e": (fmt % x) if x is not None else "-"
        row = {"slice": i, "T": float(T[i]), "it": info["it"], "floor": fl, "cert": c, "replay": replay, "seconds": time.time() - t0}
        rows.append(row)
        L.append("| %d | %.1f | %d (%s) | %s; %s | %.2e | %s | %s; %s | %s | %.1e | %.1e | %s (%s) | %s, %s | %.1e; %.1e; %s | %.1e, %.1e |" % (
            i, T[i], info["it"], "floor" if fl else "tol", f(c["q_c"], "%.3f"), f(c["q_c_prefloor"], "%.3f"), c["eta"], f(c["eps_map"]), f(c["B"]), f(c["B_prefloor"]),
            "yes" if c["certified"] else "uncertified", c["U_defect"], c["lnN_defect"], ", ".join("%.1e" % x for x in rl.get("h_ell/64_and_/128", [])),
            ", ".join("%.1e" % x for x in rl.get("h_ell/32_and_/64", [])), f(rm.get("W")), f(rm.get("invN")), c["collar"]["delta_E_rel_abs"], c["collar"]["delta_invN_abs"],
            "yes" if c["collar"]["certified"] else "no", replay[0], replay[1]))
        print(L[-1], flush=True)
        json.dump({"script": "engine_certify_v1.py", "args": vars(a), "rows": rows}, open(a.out_json, "w"), indent=1, default=lambda o: None)
    L += ["", "- every number above is printed by the script; B = (eta + eps_map)/(1 - q_c) when q_c < 0.9 (engine 17.3 (1)); the pre-floor q_c (the last three",
          "  ratios among changes above 1e-10) is printed beside, labelled; the collar's sums run over the chain's last 8 ell (17.3 (2))."]
    open(a.md, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
