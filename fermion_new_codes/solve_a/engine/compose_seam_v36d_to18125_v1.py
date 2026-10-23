#!/usr/bin/env python
"""compose_seam_v36d_to18125_v1.py (solve builder 12; engine spec 54.1 (1), (2)): the sigma/a 2 point's seam table to T/r_s 18.125 from
results/engine/V36D/cmp_v36d_vs_member.md (the point's run, v36d from 17.625, bitwise with the v36c run to 17.5, against the v32 member at
twice the floor) and the flip's record-step term from results/engine/V36C/seam_v36c_vs_member_to18125_v1.md (the stopped run, the
start's-set branch from 17.625). Light parsing; every number copied from those files. Writes results/engine/V36D/seam_v36d_vs_member_to18125_v1.md."""
import os
H = os.path.dirname(os.path.abspath(__file__)); E = os.path.join(H, "..", "..", "results", "engine")


def rows(f, t0):
    out = []
    for l in open(f):
        if l.startswith("| ") and l[2].isdigit():
            c = [x.strip() for x in l.split("|")[1:-1]]
            if float(c[0]) >= t0:
                out.append(c)
    return out


P = rows(os.path.join(E, "V36D", "cmp_v36d_vs_member.md"), 16.0)
S = {c[0]: c for c in rows(os.path.join(E, "V36C", "seam_v36c_vs_member_to18125_v1.md"), 17.625)}
L = ["# The sigma/a 2 point's seam table to T/r_s 18.125 (engine spec 54.1 (1), (2))", "",
     "The point's run: v36d from 17.625 (pod 4, tag c32_rs64_A_R2_tau8_ell2_prep4_avg_nk40r2b_v36cc_v36d; bitwise with the v36c run at 17.125 to 17.5), "
     "against the v32 member at twice the floor (E/E0 3.0e-07, roots 3.4e-06, min(N - |v|) 4.6e-08). Every number is copied from "
     "results/engine/V36D/cmp_v36d_vs_member.md (compare_member_floor_v2, 01:48 UTC) and, for the flip's record-step term, from "
     "results/engine/V36C/seam_v36c_vs_member_to18125_v1.md (the stopped run's records). Composed by code/engine/compose_seam_v36d_to18125_v1.py.", "",
     "Label from 17.625 (54.1 (2), verbatim): \"v36d from 17.625; two self-consistent sets at 17.625, the record on the previous record's set, the start's-set "
     "solve beside it; the stopped run's records the start's-set branch, their departure from the member (9.6e-05 in E/E0, 1.6e-02 in the outer root at 18.125) "
     "the flip's record-step term\".", "",
     "Reading (54.1 (1)): seam terms, nothing more, the size of 17.125 to 17.5's (9.05e-08, 3.72e-06); the terms end at 17.875; within at 18.0 and 18.125; "
     "no half-step record, no re-solve. Past 18.125, 52.1 (2) governs.", "",
     "| T/r_s | E/E0 difference | min(N - abs(v)) difference | roots (inner, outer) run | root differences | masks equal | within | kept residual run; member | reading |",
     "|---|---|---|---|---|---|---|---|---|"]
for c in P:
    t = float(c[0]); w = c[10]
    rd = "within" if w == "yes" else ("seam term (52.1 (3))" if t <= 17.5 else "seam term (54.1 (1))")
    L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (c[0], c[2], c[5], c[7], c[8], c[9], w, c[12], rd))
L += ["", "## The flip's record-step term: the stopped run (the start's-set branch from 17.625) against the member", "",
      "| T/r_s | E/E0 difference | min(N - abs(v)) difference | roots (inner, outer) stopped run | root differences | masks equal | kept residual stopped run; member |",
      "|---|---|---|---|---|---|---|"]
for t in sorted(S, key=float):
    c = S[t]
    L.append("| %s | %s | %s | %s | %s | %s | %s |" % (c[0], c[2], c[5], c[7], c[8], c[9], c[12]))
L += ["", "Note for the lead (the label's digits against this table): the largest E/E0 difference of the stopped run over 17.625 to 18.125 is "
      "-9.58e-05, at 17.75; at 18.125 it is -7.98e-05; the outer root's is -1.58e-02 at 18.125. The label is printed verbatim as ruled."]
open(os.path.join(E, "V36D", "seam_v36d_vs_member_to18125_v1.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
