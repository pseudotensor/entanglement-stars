#!/usr/bin/env python
"""s46_branch_v1.py (busi, inbox #420, NOTCOMP item 4.6: the branch value <B> on the Hartree state; ReviewA_notcomp_v1 sec. 4.6 with
ReviewB_notcomp_v1 sec. 7.2's condition). Reads table T1 of CORE_HF_full_v1.2.md (two routes: CORE_HF route 1 and busi ROUTE2_core_B29 route 2,
178 of 180 comparisons within budget), the Wick variance Var_C of the linearized, reference-dressed mass observable at R_in, R_50, R_t on every
stored slice, and evaluates on each slice
   <B>_lin = 4 sum_j Var_C(j) / M_H(j)^2      (j over the slice's represented radii)
which is <B> = 4 sum_j <(M_j_hat/M_j - 1)^2> with the mean offset and the second-order operator's terms omitted (they are bounded separately).
Second-order margin (reviewer B's condition): with M_hat - M = L + Q (L linear, Q the second-order operator, offset included),
<(M_hat/M - 1)^2> <= (rel_L + rel_Q)^2 by Cauchy-Schwarz, so <B> < 1 holds for every rel_Q below x*, the root of 4 sum_j (rel_L_j + x)^2 = 1.
Also: the number n_max = 1/(4 max_j Var/M^2) of represented masses at the slice's worst relative fluctuation that keeps <B> < 1.
"""
import json
import math
import re
import sys

src = sys.argv[1]; out = sys.argv[2]
rows = []
t = open(src).read()
i = t.find("### T1."); j = t.find("###", i + 5)
for line in t[i:j].splitlines():
    if not line.startswith("| prod") and not line.startswith("| s8"):
        continue
    c = [x.strip() for x in line.strip().strip("|").split("|")]
    rows.append({"history": c[0], "T": float(c[1]), "radius": c[3], "r": float(c[4]), "M_H": float(c[6]), "Var_C": float(c[8]), "Var_C0": float(c[9]),
                 "rel": float(c[11]), "Nm12": float(c[12])})
sl = {}
for r in rows:
    sl.setdefault((r["history"], r["T"]), []).append(r)
res = []
for (h, T), rr in sorted(sl.items()):
    q = [r["Var_C"] / r["M_H"] ** 2 for r in rr]
    q0 = [r["Var_C0"] / r["M_H"] ** 2 for r in rr]
    B = 4.0 * sum(q); B0 = 4.0 * sum(q0)
    Bcount = 4.0 * sum(r["Nm12"] ** 2 for r in rr)
    aa = [math.sqrt(x) for x in q]; nn = len(aa)
    A2 = nn; A1 = 2.0 * sum(aa); A0 = sum(x * x for x in aa) - 0.25
    xstar = (-A1 + math.sqrt(A1 * A1 - 4 * A2 * A0)) / (2 * A2)
    res.append({"history": h, "T_rs": T, "radii": [r["radius"] for r in rr], "rel_rms": [r["rel"] for r in rr], "B_lin": B, "B_sea_part": B0,
                "B_count_bound": Bcount, "n_max_at_worst": 1.0 / (4.0 * max(q)), "x_star_rel_Q": xstar, "worst": rr[q.index(max(q))]["radius"]})
    print("%-4s T %5.1f  radii %-18s rel rms %-24s  <B>_lin %.4f  (sea part %.4f; count bound 4 sum 1/N_< %.4f)  n_max %.1f at %s  x* %.4f" % (
        h, T, ",".join(r["radius"] for r in rr), ",".join("%.4f" % r["rel"] for r in rr), B, B0, Bcount, 1.0 / (4.0 * max(q)), res[-1]["worst"], xstar))
json.dump({"source": src, "rows": rows, "slices": res}, open(out, "w"), indent=1)
