#!/usr/bin/env python
"""engine_fields_v9_test_v1.py (builder analyst): engine_fields_v9's on_maxit = "return" path on a stored slice. Per chosen record of a
collapse fields npz (the sources eps_tot, J_tot and the stored lapse as the start): (1) the default solve (version 8's rule) with its
iteration history's tail, the iteration count, the floor entry; (2) the same solve cut at --maxit-short iterations with on_maxit = "return":
the returned pair against (1)'s (max |N - N_1| on the fine grid, max |W - W_1|/max|W_1|), not_converged; (3) engine 17.3 (1)'s certificate on
both pairs (B, q_c, B_prefloor, q_c_prefloor, eta, eps_map). Args: --fields --records (indices; -1 the last) --maxit-short --md."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import argparse
import json
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_fields_v9 as FLD  # noqa: E402

OPT = {"collar_policy": "flag", "support_rel": 1e-14}


def fmt(x):
    return ("%.2e" % x) if isinstance(x, float) else str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", required=True)
    ap.add_argument("--records", type=int, nargs="+", default=[-1])
    ap.add_argument("--maxit-short", type=int, default=6)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    Z = np.load(a.fields, allow_pickle=True)
    R = np.asarray(Z["R"], float); g = float(Z["g"]); ell = float(Z["ell"])
    T = np.asarray(Z["T"], float)
    L = ["## engine_fields_v9's on_maxit = \"return\" on stored slices (engine_fields_v9_test_v1.py over %s)" % os.path.basename(a.fields), "",
         "| record | T | solve | it | converged | floor / not_converged | last changes (dN, dW) | max dN vs (1) | max dW/W vs (1) | B | q_c | B_prefloor | q_c_prefloor | eta | eps_map | seconds |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in a.records:
        eps = np.asarray(Z["eps_tot"][k], float); J = np.asarray(Z["J_tot"][k], float); N0 = np.asarray(Z["N"][k], float)
        t0 = time.time()
        N1, v1, W1, i1 = FLD.solve_constraints(eps, J, g, R, ell, N0=N0, collar_policy=OPT["collar_policy"], support_rel=OPT["support_rel"])
        s1 = time.time() - t0
        rows = []
        if i1.get("failed"):
            rows.append(("(1) default", i1.get("it"), False, i1["failed"], i1["hist"][-1] if i1["hist"] else None, None, None, None, s1))
        else:
            c1 = FLD.certify(eps, J, g, R, ell, N1, v1, i1, collar_policy=OPT["collar_policy"], support_rel=OPT["support_rel"], with_eps_map=True)
            rows.append(("(1) default", i1.get("it"), i1.get("converged"), i1.get("converged_at_floor"), i1["hist"][-1], c1, 0.0, 0.0, s1))
        t0 = time.time()
        N2, v2, W2, i2 = FLD.solve_constraints(eps, J, g, R, ell, N0=N0, maxit=a.maxit_short, collar_policy=OPT["collar_policy"], support_rel=OPT["support_rel"],
                                               on_maxit="return", stall_cap=1e-8)
        s2 = time.time() - t0
        if i2.get("failed"):
            rows.append(("(2) maxit %d, return" % a.maxit_short, i2.get("it"), False, i2["failed"], i2["hist"][-1] if i2["hist"] else None, None, None, None, s2))
        else:
            c2 = FLD.certify(eps, J, g, R, ell, N2, v2, i2, collar_policy=OPT["collar_policy"], support_rel=OPT["support_rel"], with_eps_map=True)
            dN = dW = None
            if not i1.get("failed"):
                rf = i1["rfine"]; rf = rf[rf <= min(i1["r_collar"], i2["r_collar"])]
                dN = float(np.max(np.abs(i2["N_dense"](rf) - i1["N_dense"](rf))))
                Wa = i1["W_field"](rf); dW = float(np.max(np.abs(i2["W_field"](rf) - Wa))) / max(float(np.max(np.abs(Wa))), 1e-300)
            rows.append(("(2) maxit %d, return" % a.maxit_short, i2.get("it"), i2.get("converged"), i2.get("not_converged"), i2["hist"][-1], c2, dN, dW, s2))
        for lab, it, conv, fl, last, c, dN, dW, s in rows:
            L.append("| %d | %.4f | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %.0f |" % (
                k, T[k], lab, it, conv, json.dumps(fl, default=float) if fl is not None else "-", (", ".join("%.2e" % x for x in last)) if last else "-",
                fmt(dN) if dN is not None else "-", fmt(dW) if dW is not None else "-",
                fmt(c["B"]) if c and c["B"] is not None else ("uncertified" if c else "-"), fmt(c["q_c"]) if c and c["q_c"] is not None else "-",
                fmt(c["B_prefloor"]) if c and c["B_prefloor"] is not None else "-", fmt(c["q_c_prefloor"]) if c and c["q_c_prefloor"] is not None else "-",
                fmt(c["eta"]) if c else "-", fmt(c["eps_map"]) if c and c["eps_map"] is not None else "-", s))
        if not i1.get("failed"):
            L.append("| | | (1)'s history (max(dN, dW) per iteration) | %s |" % ", ".join("%.1e" % max(x) for x in i1["hist"]))
        print("\n".join(L[-3:]), flush=True)
    L += ["", "- every number above is printed by the script; (1) is version 8's rule (on_maxit = fail), (2) the return path cut at --maxit-short."]
    open(a.md, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
