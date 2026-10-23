#!/usr/bin/env python
"""s24_audit_v1.py (busi, inbox #437, reviewer B's check of NOTCOMP 2.4 v2): statistics from the stored s24_analysis_v1.json only.
Per kappa, over the fixed pair set P = {T in 50, 60, ..., 100} x {cuts 2, 16, 24, 30, 48 r_s} (30 pairs): the signed weighted terms
a = 16 kappa x for production linear (lin), C2 spline and C1 Catmull-Rom; the norm ratio sum|a_m| / sum|a_lin|; the difference norm
sum|a_m - a_lin| / sum|a_lin|; the six largest |a_lin| pairs with their relative changes; C1 against C2 pointwise; the largest pair;
the front-crossing pairs with absolute terms; the decay fit of the 50 r_s / 16 r_s pair over kappa."""
import json, sys, math
import numpy as np
d = json.load(open(sys.argv[1]))
out = {}
P = [(T, c) for T in (50.0, 60.0, 70.0, 80.0, 90.0, 100.0) for c in ("2", "16", "24", "30", "48")]
FRONT = [(40.0, "16"), (50.0, "24"), (60.0, "30"), (80.0, "48")]
for K in ("12", "16", "24", "48"):
    rows = {(r["T"], r["cut"]): r for r in d["kappas"][K]["rows"]}
    o = {}
    lin = np.array([rows[p]["w_prod"] for p in P])
    for m in ("spline", "cr"):
        if rows[P[0]].get("w_" + m) is None:
            continue
        a = np.array([rows[p]["w_" + m] for p in P])
        rel = (a - lin) / np.abs(lin)
        big = np.argsort(-np.abs(lin))[:6]
        o[m] = {"norm_ratio": float(np.abs(a).sum() / np.abs(lin).sum()), "diff_norm_ratio": float(np.abs(a - lin).sum() / np.abs(lin).sum()),
                "max_abs_rel_change_all": float(np.max(np.abs(rel))), "six_largest": [{"T": P[i][0], "cut": P[i][1], "lin": float(lin[i]), m: float(a[i]), "rel_change": float(rel[i])} for i in big],
                "six_largest_max_abs_rel_change": float(np.max(np.abs(rel[big]))), "largest_pair_rel_change": float(rel[big[0]])}
    if "cr" in o:
        a2 = np.array([rows[p]["w_spline"] for p in P]); a1 = np.array([rows[p]["w_cr"] for p in P])
        o["cr_vs_spline"] = {"diff_norm_over_spline_norm": float(np.abs(a1 - a2).sum() / np.abs(a2).sum()), "max_pointwise_rel": float(np.max(np.abs(a1 - a2) / np.abs(a2))),
                             "median_pointwise_rel": float(np.median(np.abs(a1 - a2) / np.abs(a2)))}
    if rows[P[0]].get("w_linear") is not None:
        ll = np.array([rows[p]["linear"] for p in P]); pp = np.array([rows[p]["prod"] for p in P])
        o["lean_linear_minus_prod_per_chain_max"] = float(np.max(np.abs(ll - pp)))
        o["lean_linear_minus_prod_weighted_max"] = float(16 * int(K) * np.max(np.abs(ll - pp)))
    o["front"] = [{"T": T, "cut": c, "lin": rows[(T, c)]["w_prod"], "spline": rows[(T, c)]["w_spline"], "ratio": rows[(T, c)]["w_spline"] / rows[(T, c)]["w_prod"]} for (T, c) in FRONT]
    out[K] = o
ks = np.array([12, 16, 24, 48.0]); y = np.array([abs(d["kappas"][str(int(k))]["rows"][[i for i, r in enumerate(d["kappas"][str(int(k))]["rows"]) if r["T"] == 50.0 and r["cut"] == "16"][0]]["w_prod"]) for k in ks])
slope = np.polyfit(np.log(ks), np.log(y), 1)[0]
out["decay_50_16"] = {"kappa": ks.tolist(), "abs_w_prod": y.tolist(), "endpoint_exponent": float(math.log(y[0] / y[-1]) / math.log(48 / 12)), "lsq_log_log_slope": float(slope)}
json.dump(out, open(sys.argv[2], "w"), indent=1)
for K, o in out.items():
    if K == "decay_50_16":
        print("decay at 50/16:", o); continue
    print("kappa", K)
    for m in ("spline", "cr"):
        if m in o:
            x = o[m]
            print("  %s: norm ratio %.5f  diff-norm ratio %.5f  max|rel| all %.3f  six-largest max|rel| %.4f  largest pair rel %+.5f" % (m, x["norm_ratio"], x["diff_norm_ratio"], x["max_abs_rel_change_all"], x["six_largest_max_abs_rel_change"], x["largest_pair_rel_change"]))
            print("     six largest:", [(s["T"], s["cut"], "%.4e" % s["lin"], "%+.4f" % s["rel_change"]) for s in x["six_largest"]])
    if "cr_vs_spline" in o: print("  C1 vs C2:", o["cr_vs_spline"])
    if "lean_linear_minus_prod_per_chain_max" in o: print("  lean-prod max per chain %.3e, weighted %.3e" % (o["lean_linear_minus_prod_per_chain_max"], o["lean_linear_minus_prod_weighted_max"]))
    print("  front:", [(f["T"], f["cut"], "%.3e" % f["lin"], "%.3e" % f["spline"], "%.3f" % f["ratio"]) for f in o["front"]])
