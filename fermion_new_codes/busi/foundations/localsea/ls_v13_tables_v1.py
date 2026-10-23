#!/usr/bin/env python
"""ls_v13_tables_v1.py (busi, inbox #468): prints the v1.3 tables (markdown rows) from ls_sizectl_read_v1.json, so that no number in
FOUNDATIONS_localsea_v1.3.md is retyped. One thread, seconds."""
import json
J = json.load(open("<internal>/ls_sizectl_read_v1.json"))
print("T1 | form | c(128, j1) | c(256, j2) | shift q_min | shift q_max (64 j2 vs 32 j1) | linear dc0 from q_min | from q_max |")
for k, o in J["coef"].items():
    print("| %s | %.10e | %.10e | %+.1e | %+.3f%% | %+.1e | %+.2e |" % (k, o["c_L128_j1"], o["c_L256_j2"], o["shift_qmin"], 100 * o["shift_qmax"],
          o["c0_change_linear_qmin"], o["c0_change_linear_qmax"]))
print("T2 | form | free c0 v1.2 (c0/c) | q_min doubled | q_min and q_max doubled | falls with q_min box | falls with both |")
for k, o in J["coef"].items():
    f = lambda s: "%+.3e (%+.4f)" % (o["sets"][s]["free_1_q2_q4_4pt"]["c0"], o["sets"][s]["free_1_q2_q4_4pt"]["c0_over_c_qmin"])   # noqa: E731
    print("| %s | %s | %s | %s | %s | %s |" % (k, f("v12"), f("qmin_doubled"), f("qmin_and_qmax_doubled"),
          "yes" if o["free_c0_falls_with_box"] else "no", "yes" if o["free_c0_falls_with_both"] else "no"))
for s in ("qmin_and_qmax_doubled", "v12"):
    print("T3[%s] | form | free 1,q2,q4 (4 q) | exact 1,q2,q4,q6 (4 q) | exact 1,q2,q4 (3 smallest) | exact 1,q2 (2 smallest) | spread |" % s)
    for k, o in J["coef"].items():
        d = o["sets"][s]
        g = lambda m: "%+.2e (%+.4f)" % (d[m]["c0"], d[m]["c0_over_c_qmin"])   # noqa: E731
        print("| %s | %s | %s | %s | %s | %.2e |" % (k, g("free_1_q2_q4_4pt"), g("exact_1_q2_q4_q6_4pt"), g("exact_1_q2_q4_3smallest"),
              g("exact_1_q2_2smallest"), d["model_spread_c0"]))
