#!/usr/bin/env python
"""compose_relabel_v1_18.py (builder analyst; inbox #34, the icloud lane's bounded check of the temperature v1.4 amendments,
review_TEMPERATURE_solved_exterior_v1_4_amendments.tex, items 1 to 3): version 1.17 with the title's source list and the sources line
extended and three rows added before the sources line, the review's three precisions applied to TEMPERATURE_solved_exterior_v1.4.md
(composed by compose_temperature_v1_4) until the lead's version 1.5 (turn 46); the review's words quoted, its TeX quotes and math written
plain (the absolute value written abs(), pipes breaking the table). Args: --v117 --out."""
import argparse
import re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v117", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rv = "review_TEMPERATURE_solved_exterior_v1_4_amendments.tex, inbox #34"
    rows = ["| TEMPERATURE_solved_exterior_v1.4.md (amendment (a), the lead's turn 45): \"the two phases' readings follow the record's own dense surface slope to within about 12 percent\" | read with the icloud lane's qualification (%s, item 1): \"'Within about 12 percent' holds for these selected readings, not all packets or the other three frozen records in the table\"; until the lead's version 1.5 (turn 46) |" % rv,
            "| TEMPERATURE_solved_exterior_v1.4.md (amendment (a)): \"so the order-one test the solved exterior supplies is T against the local conversion scale kappa_T/(2 pi)\" | \"the local conversion scale kappa_T/(2 pi)\" read \"the dense-slope comparison scale kappa_T/(2 pi)\" (%s, item 2: \"The probe uses applied fields; kappa_T is the dense derivative at its dense root\"; \"the ratio table supplies only their product (1 + D_1)(1 + delta_conn), not the separate factors or a measured connection error\"); until the lead's version 1.5 |" % rv,
            "| TEMPERATURE_solved_exterior_v1.4.md (amendment (b)): \"the difference between the dense surface slope kappa_T and the applied-field root slope kappa_m at the record, 9.2 percent at the maximum and 26.6 percent at the minimum\" | read as abs(kappa_T - kappa_m)/kappa_T at the respective dense and applied roots, \"a supplied field-comparison diagnostic, not a certified temperature error or the unevaluated source-based collar prediction kappa_0\" (%s, item 3); compose_temperature_v1_4 used that form for the recomputed sums and counts (solve/results/obs/J3v7/compose_temperature_v1_4.md); until the lead's version 1.5 |" % rv]
    t = open(a.v117).read()
    tm = re.search(r"^# RELABEL_table_v1\.17: (.*?)\(engine spec (.*?); inbox #41\)$", t, re.M)
    assert tm
    t = t.replace(tm.group(0), "# RELABEL_table_v1.18: %s(engine spec %s; inbox #41; inbox #34)" % (tm.group(1), tm.group(2)), 1)
    pm = re.search(r"Version 1\.17 = version 1\.16 with ", t)
    assert pm
    t = t.replace(pm.group(0), "Version 1.18 = version 1.17 with 3 rows added (inbox #34: the icloud lane's check of the temperature v1.4 amendments, its three precisions applied to TEMPERATURE_solved_exterior_v1.4 until the lead's version 1.5). Version 1.17 = version 1.16 with ", 1)
    si = t.index("\n- Sources: ")
    t = t[:si].rstrip("\n") + "\n" + "\n".join(rows) + "\n" + t[si:]
    sm = re.search(r"\n- Sources: ([^\n]*)", t)
    assert sm
    t = t.replace(sm.group(0), "\n- Sources: " + sm.group(1).rstrip(".") + "; deliveries/solve_b/review_TEMPERATURE_solved_exterior_v1_4_amendments.tex (inbox #34); deliveries/solve_a/TEMPERATURE_solved_exterior_v1.4_amendments.md, TEMPERATURE_solved_exterior_v1.4.md; solve/results/obs/J3v7/compose_temperature_v1_4.md.", 1)
    open(a.out, "w").write(t)
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
