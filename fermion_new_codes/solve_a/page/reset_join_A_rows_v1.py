#!/usr/bin/env python3
"""reset_join_A_rows_v1.py (solve builder 10, analyst; engine spec 49.2 (1), 52.5 item 4): light parsing only. Joins the reset readout's
per-interval sums (reset_terms_<tag>.json, its "checkpoints") with A's own stored rows (A's summary json, its "rows") over the same
interval (lo, hi]: A's W_acc_interval_weighted and W_hdot_trapezoid summed over A's rows in the interval, W_hdot_weighted and the energy
and collar identity residuals (the largest in the interval) printed beside Sum Q and Sum D. Every number is one of the two files'.
Usage: reset_join_A_rows_v1.py RESET.json A_SUMMARY.json OUT_PREFIX"""
import json
import sys

R = json.load(open(sys.argv[1])); A = json.load(open(sys.argv[2])); pre = sys.argv[3]
rows = A.get("rows") or []
out = []
lines = ["# The reset terms beside A's stored rows (reset_join_A_rows_v1.py)", "",
         "Reset readout: %s. A's rows: %s. Weighted (16 kappa) sums per interval; A's W_acc and trapezoid summed over A's rows in (lo, hi]." % (sys.argv[1], sys.argv[2]), "",
         "| interval T/r_s | flagged records | Sum Q | Sum Q on flagged | Sum D | A rows | A W_acc (sum) | A W_hdot trapezoid (sum) | A W_hdot (sum) | A max energy identity residual | A max collar identity residual | Sum Q / A W_acc |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
g = lambda r, k: r.get(k) if isinstance(r.get(k), (int, float)) else None
for ck in R["checkpoints"]:
    lo, hi = ck["T_lo_over_rs"], ck["T_hi_over_rs"]
    m = [r for r in rows if r.get("t_over_rs") is not None and lo + 1e-9 < float(r["t_over_rs"]) <= hi + 1e-9]
    s = lambda k: sum(g(r, k) for r in m if g(r, k) is not None) if any(g(r, k) is not None for r in m) else None
    mx = lambda k: max(abs(g(r, k)) for r in m if g(r, k) is not None) if any(g(r, k) is not None for r in m) else None
    j = {"T_lo_over_rs": lo, "T_hi_over_rs": hi, "flagged_records": ck["flagged_records"], "sum_Q_weighted": ck["sum_Q_weighted"],
         "sum_Q_flagged_weighted": ck["sum_Q_flagged_weighted"], "sum_D_weighted": ck["sum_D_weighted"], "A_rows": [r["t_over_rs"] for r in m],
         "A_W_acc_interval_weighted": s("W_acc_interval_weighted"), "A_W_hdot_trapezoid": s("W_hdot_trapezoid"), "A_W_hdot_weighted": s("W_hdot_weighted"),
         "A_energy_identity_residual_weighted_max": mx("energy_identity_residual_weighted"), "A_collar_identity_residual_weighted_max": mx("collar_identity_residual_weighted")}
    out.append(j)
    f = lambda x: "-" if x is None else "%.6e" % x
    ratio = (ck["sum_Q_weighted"] / j["A_W_acc_interval_weighted"]) if j["A_W_acc_interval_weighted"] else None
    lines.append("| %.1f to %.1f | %s | %.6e | %.6e | %.6e | %s | %s | %s | %s | %s | %s | %s |" % (
        lo, hi, ", ".join("%.3f" % x for x in ck["flagged_records"]) or "none", ck["sum_Q_weighted"], ck["sum_Q_flagged_weighted"], ck["sum_D_weighted"],
        ", ".join("%g" % x for x in j["A_rows"]) or "none", f(j["A_W_acc_interval_weighted"]), f(j["A_W_hdot_trapezoid"]), f(j["A_W_hdot_weighted"]),
        f(j["A_energy_identity_residual_weighted_max"]), f(j["A_collar_identity_residual_weighted_max"]), f(ratio)))
open(pre + ".md", "w").write("\n".join(lines) + "\n")
json.dump(out, open(pre + ".json", "w"), indent=1)
print("\n".join(lines))
