#!/usr/bin/env python
"""compose_relabel_v1_5.py (builder analyst; engine spec 28.9: "RELABEL_table_v1.5 (a supra-maximal row; the record line "rows: raw, floor
unattributed; calibrated against the step at the innermost node, first order (28.7)")"): version 1.4 with the title's spec list, the
provenance line, the rows line's new wording and the sources line changed, and one row added: any stored or delivered text calling the r_s 64
primary near-maximal (or by its static family's maximum) reads as the lead's D1 sentence of turn 36 (verbatim from
COLLAPSE_records_v1.1_labels_turn36.md). Each replaced text asserted unique; the quoted wordings read from the files, not retyped.
Args: --v14 --spec --t36 --out."""
import argparse
import re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v14", required=True); ap.add_argument("--spec", required=True); ap.add_argument("--t36", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t = open(a.v14).read(); spec = open(a.spec).read(); t36 = open(a.t36).read()
    m = re.search(r'the record line "(rows: raw, floor unattributed; calibrated against the step at the innermost node, first order \(28\.7\))"', spec)
    assert m, "28.9's record line not found"
    rec = m.group(1)
    d1 = re.search(r"^## D1 [^\n]*\n\n(.*?)(?=\n## |\Z)", t36, re.S | re.M).group(1).strip()
    sm = re.search(r"(The primary's label \(owner item D7\): supra-maximal.*?never near-maximal\.)", d1, re.S)
    assert sm, "D1's supra-maximal sentence not found"
    old_rows = re.search(r'\| "rows: flat-slice ADM, K\^ADM = -K" \(engine_collapse_v19, engine_rows_v1/v2 tables\) and "rows: \.\.\. raw, floor in the stress\'s trace \(25\.3\)" \(engine_collapse_v25 records\) \| "rows: flat-slice ADM, K\^ADM = -K; (rows: raw, floor unattributed; survives the density, differentiation and map tests \(27\.2\))" \(engine 27\.2; replaces version 1\.3\'s "rows: raw, floor unattributed \(26\.1\)" and version 1\.2\'s "floor in the stress\'s trace", withdrawn\) \|', t)
    assert old_rows, "version 1.4's rows line not found"
    new_rows = ('| "rows: flat-slice ADM, K^ADM = -K" (engine_collapse_v19, engine_rows_v1/v2 tables) and "rows: ... raw, floor in the stress\'s trace (25.3)" (engine_collapse_v25 records) | '
                '"rows: flat-slice ADM, K^ADM = -K; %s" (engine 28.7 and 28.9; replaces version 1.4\'s "%s") |' % (rec, old_rows.group(1)))
    t = t.replace(old_rows.group(0), new_rows)
    reps = [("# RELABEL_table_v1.4: the stored record lines and the wording that replaces them in every handoff and report (engine spec 22.0, 23.6, 24.1, 25.1 to 25.3, 26.1, 27.2, 27.5)",
             "# RELABEL_table_v1.5: the stored record lines and the wording that replaces them in every handoff and report (engine spec 22.0, 23.6, 24.1, 25.1 to 25.3, 26.1, 27.2, 27.5, 28.7, 28.9)"),
            ("Version 1.4 = version 1.3 with the rows line read by engine 27.2",
             "Version 1.5 = version 1.4 with the rows line read by engine 28.9 (\"%s\") and the primary's supra-maximal row (owner item D7; the lead's turn 36). Version 1.4 = version 1.3 with the rows line read by engine 27.2" % rec),
            ("- Sources: solve/specs/SPEC_final4b_engine.md secs. 12.2, 13.1, 18.1, 20.1, 21.1, 21.2, 22.0, 22.4, 23.2, 23.6, 24.1, 25.1 to 25.3, 26.1, 27.2, 27.5; ",
             "- Sources: solve/specs/SPEC_final4b_engine.md secs. 12.2, 13.1, 18.1, 20.1, 21.1, 21.2, 22.0, 22.4, 23.2, 23.6, 24.1, 25.1 to 25.3, 26.1, 27.2, 27.5, 28.7, 28.9; "),
            ("and COLLAPSE_records_v1.1_labels_turn35.md (B, C).", "COLLAPSE_records_v1.1_labels_turn35.md (B, C) and COLLAPSE_records_v1.1_labels_turn36.md (A, B, C, D).")]
    for o, n in reps:
        assert t.count(o) == 1, o[:60]
        t = t.replace(o, n)
    row = ('| any stored or delivered text calling the r_s 64 primary (N_B 640 at (m a, G/a^2) = (0.2, 1/4)) near its maximum mass or near-maximal (requirements_new.txt\'s S1 framing, '
           'as the review\'s section 4 lists it; the review\'s 1.16) | "%s" (engine 28.4, the lead\'s D1 of turn 36; owner item D7) |' % sm.group(1))
    anchor = '| "LZ convention (c_LZ = ..., the lane\'s declaration)" in the record lines of engine_collapse_v24 |'
    i = t.index(anchor); j = t.index("\n", i)
    t = t[:j + 1] + row + "\n" + t[j + 1:]
    open(a.out, "w").write(t)
    print("%d chars" % len(t))


if __name__ == "__main__":
    main()
