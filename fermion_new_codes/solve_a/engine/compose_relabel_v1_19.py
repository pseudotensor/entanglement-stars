#!/usr/bin/env python
"""compose_relabel_v1_19.py (builder analyst; inbox #44, the icloud lane's composition check of TEMPERATURE_solved_exterior_v1.4,
review_TEMPERATURE_solved_exterior_v1_4.tex, its disposition and items 1 to 3): version 1.18 with the title's source list and the sources
line extended and four rows added before the sources line, the review's precisions on version 1.4 not already carried by version 1.18's
three rows, until the lead's version 1.5 (turn 46); the review's words quoted, its TeX quotes and math written plain (no pipes inside a
cell). Args: --v118 --out."""
import argparse
import re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v118", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rv = "review_TEMPERATURE_solved_exterior_v1_4.tex, inbox #44"
    rows = ["| TEMPERATURE_solved_exterior_v1.4.md (composed): \"16 of 18 and 12 of 12 readings per record lie inside them (diagnostic counts, not coverage probabilities)\" | read with the icloud lane's disposition (%s): \"These are diagnostic counts, including eligible hold-16 readings, not a count exclusively of frozen-stage readings or certified coverage\"; until the lead's version 1.5 (turn 46) |" % rv,
            "| TEMPERATURE_solved_exterior_v1.4.md (version 1.3's tau 4 clauses, kept verbatim): \"8 of 9 readings inside their budgets under its own step partner\" and \"3 of 9 under its partner\" | these use the old budget, the vacuum shift term (%s, item 2: \"The retained tau=4 counts 8/9 at 39.5 and 3/9 at 39.75 use the old budget. Label that convention explicitly, or use the same report's replacement counts, 9/9 at each record. Do not present the two conventions as one revised budget\"); the replacement counts are in solve/results/obs/J3v7/compose_temperature_v1_4.md; until the lead's version 1.5 |" % rv,
            "| TEMPERATURE_solved_exterior_v1.4.md (amendment (a)): \"in the factorization T/T_H = (1 + D_1)(1 + D_2)(1 + delta_conn) with D_2 = 4 G E kappa_T - 1 read from the stored fields (+0.26 and -0.04 at the two records), D_1 the ray-map readout against that slope and delta_conn the conversion fit; the probes propagate on the applied fields and yet track the dense slope better, which is left open\" | read with D_1 = kappa_read/kappa_T - 1 and delta_conn = 2 pi T_fit/kappa_read - 1 retained explicitly (%s, item 1: \"Both production wave-readout tables still mark kappa_read unresolved: the ratio supplies only (1 + D_1)(1 + delta_conn), not either factor separately or a measured connection error. 'Within about 12 percent' is restricted to the selected long-hold fits. Better dense-comparator agreement does not explain the propagation or phase spread\"); until the lead's version 1.5 |" % rv,
            "| TEMPERATURE_solved_exterior_v1.4.md (the amendments' condition): \"the collar prediction kappa_0 enters version 1.5 when obs_p26_slope_ratios_v2 lands\" | the table is delivered (TEMPERATURE_kappa0_table_v1.md, printed by obs_p26_slope_ratios_v3; version 2 had the same values at lower displayed precision); read with %s, item 3: \"Its five passing reproduction checks and kappa_0 ~ kappa_T concern re-solved dense constraints at the dense root, with the declared current sign. They neither identify the applied kappa_m or measured kappa_read, nor measure a connection or thermal temperature. Publication in v1.5 remains a separate composition decision\"; until the lead's version 1.5 |" % rv]
    t = open(a.v118).read()
    tm = re.search(r"^# RELABEL_table_v1\.18: (.*?)\(engine spec (.*?); inbox #41; inbox #34\)$", t, re.M)
    assert tm
    t = t.replace(tm.group(0), "# RELABEL_table_v1.19: %s(engine spec %s; inbox #41; inbox #34; inbox #44)" % (tm.group(1), tm.group(2)), 1)
    pm = re.search(r"Version 1\.18 = version 1\.17 with ", t)
    assert pm
    t = t.replace(pm.group(0), "Version 1.19 = version 1.18 with 4 rows added (inbox #44: the icloud lane's composition check of TEMPERATURE_solved_exterior_v1.4, its precisions not already carried, until the lead's version 1.5). Version 1.18 = version 1.17 with ", 1)
    si = t.index("\n- Sources: ")
    t = t[:si].rstrip("\n") + "\n" + "\n".join(rows) + "\n" + t[si:]
    sm = re.search(r"\n- Sources: ([^\n]*)", t)
    assert sm
    t = t.replace(sm.group(0), "\n- Sources: " + sm.group(1).rstrip(".") + "; deliveries/solve_b/review_TEMPERATURE_solved_exterior_v1_4.tex (inbox #44); deliveries/solve_a/TEMPERATURE_kappa0_table_v1.md.", 1)
    for r in rows:
        assert r.count("|") == 3, r[:60]
    open(a.out, "w").write(t)
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
