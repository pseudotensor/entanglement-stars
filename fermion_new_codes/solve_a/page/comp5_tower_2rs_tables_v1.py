#!/usr/bin/env python
"""comp5_tower_2rs_tables_v1.py (solve lane, composition 5): the tables from comp5_tower_2rs_v1.json; computes nothing new.
Usage: python comp5_tower_2rs_tables_v1.py DATA.json TABLES.md PERCHANNEL.tsv SUMS.tsv"""
import json
import sys

d = json.load(open(sys.argv[1]))
cl = d["clocks"]
T = {int(k): v for k, v in d["table"].items()}
B = d["brackets"]
SHOW = [40.0, 60.0, 70.0, 80.0, 100.0]
idx = [cl.index(t) for t in SHOW]
grp = {}
for g, ks in d["missing_groups"].items():
    for k in ks:
        grp[k] = g
side_by_k = {}
for n, s in d["side"].items():
    side_by_k.setdefault(s["kappa"], []).append((n, s))


def e(x):
    return "-" if x is None else "%+.3e" % x


L = []
L.append("## Table 1. The 2 r_s cut, per channel: 16 kappa x_kappa (nats) at five clocks, with 16 kappa F_meas")
L.append("")
L.append("A star marks a term not resolved against its own floor (|x| <= F_meas). kappa/2c with 2c = 64.25 (C11). MISSING rows carry no value; the bracket column is bracket C at 100 r_s (a bracket, not a value); the side column lists data that exist outside the 54-channel set and are NOT in any sum.")
L.append("")
L.append("| kappa | kappa/2c | status | T/r_s 40 | 60 | 70 | 80 | 100 | 16 kappa F_meas at 100 | bracket C at 100 (missing only) | side data (not summed) |")
L.append("|---|---|---|---|---|---|---|---|---|---|---|")
for k in range(1, d["kmax"] + 1):
    t = T[k]
    sd = []
    for n, s in side_by_k.get(k, []):
        r = {q["t"]: q for q in s["rows"]}
        vals = ", ".join("%g: %s" % (tt, e(r[tt]["wx"])) for tt in SHOW if tt in r)
        tag = "k64 R2 final" if n.startswith("k64") else ("probe v6o lean" if n.startswith("probe") else "mirror partial to %g, not final" % s["last_clock"])
        if n.startswith("partial_k64"):
            continue
        sd.append("%s (%s)" % (tag, vals if vals else "no clock at or after 40"))
    if t["status"] == "present":
        cells = ["%s%s" % (e(t["wx"][i]), "" if t["resolved"][i] else "*") for i in idx]
        L.append("| %d | %.3f | present | %s | %s | | %s |" % (k, k / d["two_c"], " | ".join(cells), "%.2e" % t["wF"][cl.index(100.0)], "; ".join(sd)))
    else:
        c = B["C"]["per_channel"][str(k)]
        i = cl.index(100.0)
        L.append("| %d | %.3f | MISSING (%s) | - | - | - | - | - | - | [%s, %s] from %s | %s |" % (k, k / d["two_c"], grp.get(k, "?"), e(c["lo"][i]), e(c["hi"][i]),
                                                                               ",".join(str(q) for q in c["neighbours"]), "; ".join(sd)))
L.append("")
L.append("## Table 2. The sums at the 2 r_s cut (nats), every 10 r_s from 40 (all 31 readout clocks in the SUMS tsv)")
L.append("")
L.append("| T/r_s | S over 1 to 18 | present tower 19 to 80 (36 ch) | of which 19 to 63 (20 ch) | of which 64 to 80 (65 to 80, 16 ch) | S over the 54 | 16 kappa F_meas summed over the 54 | bracket A on the 26 missing | bracket B | bracket C |")
L.append("|---|---|---|---|---|---|---|---|---|---|")
S = d["sums"]
for t in (40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0):
    i = cl.index(t)
    L.append("| %g | %.6f | %s | %s | %s | %.6f | %.3e | [%s, %s] | [%s, %s] | [%s, %s] |" % (
        t, S["S_1to18"][i], e(S["S_tower_present"][i]), e(S["S_19to63_present"][i]), e(S["S_64to80_present"][i]), S["S_54"][i], S["F_54"][i],
        e(B["A"]["sum_lo"][i]), e(B["A"]["sum_hi"][i]), e(B["B"]["sum_lo"][i]), e(B["B"]["sum_hi"][i]), e(B["C"]["sum_lo"][i]), e(B["C"]["sum_hi"][i])))
L.append("")
L.append("Leave-one-out test of each bracket rule on the present channels with present neighbours on both sides, all clocks from 40 r_s: " +
         "; ".join("%s %d of %d inside" % (lab, B[lab]["leave_one_out_from40"]["inside"], B[lab]["leave_one_out_from40"]["total"]) for lab in "ABC") + ".")
open(sys.argv[2], "w").write("\n".join(L) + "\n")
with open(sys.argv[3], "w") as f:
    f.write("kappa\tstatus\t" + "\t".join("wx_T%g" % t for t in cl) + "\t" + "\t".join("wF_T%g" % t for t in cl) + "\n")
    for k in range(1, d["kmax"] + 1):
        t = T[k]
        if t["status"] == "present":
            f.write("%d\tpresent\t%s\t%s\n" % (k, "\t".join(repr(v) for v in t["wx"]), "\t".join(repr(v) for v in t["wF"])))
        else:
            f.write("%d\tMISSING\t%s\n" % (k, "\t".join(["NA"] * (2 * len(cl)))))
with open(sys.argv[4], "w") as f:
    keys = ["S_1to18", "S_tower_present", "S_19to63_present", "S_64to80_present", "S_54", "F_54"]
    f.write("T_over_rs\t" + "\t".join(keys) + "\tbrA_lo\tbrA_hi\tbrB_lo\tbrB_hi\tbrC_lo\tbrC_hi\n")
    for i, t in enumerate(cl):
        f.write("%g\t%s\t%r\t%r\t%r\t%r\t%r\t%r\n" % (t, "\t".join(repr(S[q][i]) for q in keys), B["A"]["sum_lo"][i], B["A"]["sum_hi"][i],
                                                    B["B"]["sum_lo"][i], B["B"]["sum_hi"][i], B["C"]["sum_lo"][i], B["C"]["sum_hi"][i]))
print("ok")
