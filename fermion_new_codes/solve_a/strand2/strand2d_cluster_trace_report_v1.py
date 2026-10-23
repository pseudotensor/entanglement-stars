#!/usr/bin/env python
"""Strand 2d sec. 11 report (builder analyst): tables from strand2d_cluster_trace_v1.py (cluster_trace.json).  Numbers emitted, not retyped."""
import argparse
import json


def f(v, sig=4):
    if v is None:
        return "-"
    v = float(v)
    if v == 0:
        return "0"
    return ("%.*e" % (sig - 1, v)) if (abs(v) < 1e-3 or abs(v) >= 1e5) else ("%.*g" % (sig, v))


def table(head, rows):
    esc = lambda c: str(c).replace("|", "\\|")
    return "\n".join(["| " + " | ".join(esc(c) for c in head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    R = json.load(open(a.json))
    M = R["meta"]
    out = ["### Sec. 11 T11a The dressed B4 (2,0) block at (0.2, 1): dimension %d, E_1 - E_0 = %s, the cluster's width E_119 - E_0 = %s, the gap "
           "E_120 - E_119 = %s; per q the exact cluster-trace susceptibility over the lowest 120 levels and over the lowest 8, the ground level's "
           "sum (the check against sec. 10) split into the transitions inside the cluster (r = 1..119) and outside, the first-order trace/120 "
           "and /8, the largest off-diagonal |<a|H_N|b>| inside the cluster, the affinity check" % (M["dim"], f(M["E1_minus_E0"], 4),
                                                                                                   f(M["cluster_width_E119_minus_E0"], 4),
                                                                                                   f(M["gap_E120_minus_E119"], 4)), ""]
    rows = []
    for r in R["rows"]:
        rows.append([f(r["q"], 6), f(r["J_NN_trace_lowest_120"], 10), f(r["J_NN_trace_lowest_8"], 10), f(r["J_NN_ground"], 10),
                     f(r["J_NN_ground_within_cluster_r_1_to_119"], 10), f(r["J_NN_ground_outside_cluster"], 6),
                     f(r["first_derivative_trace_lowest_120"], 2), f(r["first_derivative_trace_lowest_8"], 2), f(r["max_offdiag_within_cluster"], 6),
                     f(r["affinity_check"], 2)])
    out += [table(["q", "J_NN^cl (120)", "J_NN (lowest 8)", "J_NN ground", "ground: r = 1..119", "ground: r >= 120", "first-order trace/120",
                   "first-order trace/8", "max |<a|H_N|b>| in cluster", "affinity"], rows), ""]
    for r in R["rows"]:
        rows = [[t["r"], f(t["E_r_minus_E0"], 6), f(t["term"], 8), f(t["abs_amp"], 6)] for t in r["ground_top10"]]
        out += ["### Sec. 11 T11b q = %s: the ten largest terms |<r|H_N|0>|^2/(E_r - E_0) of the ground level's sum (J_NN = -2 sum)" % f(r["q"], 6), "",
                table(["level r", "E_r - E_0", "term", "|<r|H_N|0>|"], rows), ""]
    open(a.md, "w").write("\n".join(out) + "\n")
    print("wrote", a.md)


if __name__ == "__main__":
    main()
