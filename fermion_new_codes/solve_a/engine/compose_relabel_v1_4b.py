#!/usr/bin/env python
"""compose_relabel_v1_4b.py (builder analyst; version 1.4 of the table without a builder paraphrase of the reading; engine spec 27.2: "The record line reads "rows: raw, floor unattributed; survives the density,
differentiation and map tests (27.2)" (RELABEL_table_v1.4, BUILDER)" and 27.5 (4)): RELABEL_table_v1.4 = version 1.3 with the title's spec list,
the provenance line, the rows line and the sources line changed; each replaced text asserted unique in version 1.3; the record line's wording
is read from the spec, not retyped. Args: --v13 --spec --out."""
import argparse
import re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v13", required=True); ap.add_argument("--spec", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t = open(a.v13).read(); spec = open(a.spec).read()
    sec = spec[spec.index("## 27."):]
    m = re.search(r'The record line reads "(rows: raw, floor unattributed; survives the density, differentiation and map tests \(27\.2\))" \(RELABEL_table_v1\.4, BUILDER\)', sec)
    assert m, "27.2's record line not found"
    rec = m.group(1)
    reps = [
        ("# RELABEL_table_v1.3: the stored record lines and the wording that replaces them in every handoff and report (engine spec 22.0, 23.6, 24.1, 25.1 to 25.3, 26.1)",
         "# RELABEL_table_v1.4: the stored record lines and the wording that replaces them in every handoff and report (engine spec 22.0, 23.6, 24.1, 25.1 to 25.3, 26.1, 27.2, 27.5)"),
        ("Version 1.3 = version 1.2 with the rows line read by engine 26.1",
         "Version 1.4 = version 1.3 with the rows line read by engine 27.2 (\"%s\"). Version 1.3 = version 1.2 with the rows line read by engine 26.1" % rec),
        ("| \"rows: flat-slice ADM, K^ADM = -K\" (engine_collapse_v19, engine_rows_v1/v2 tables) and \"rows: ... raw, floor in the stress's trace (25.3)\" (engine_collapse_v25 records before 26.1) | \"rows: flat-slice ADM, K^ADM = -K; rows: raw, floor unattributed (26.1)\" (engine 26.1; replaces version 1.2's \"floor in the stress's trace\", withdrawn) |",
         "| \"rows: flat-slice ADM, K^ADM = -K\" (engine_collapse_v19, engine_rows_v1/v2 tables) and \"rows: ... raw, floor in the stress's trace (25.3)\" (engine_collapse_v25 records) | \"rows: flat-slice ADM, K^ADM = -K; %s\" (engine 27.2; replaces version 1.3's \"rows: raw, floor unattributed (26.1)\" and version 1.2's \"floor in the stress's trace\", withdrawn) |" % rec),
        ("25.1 to 25.3, 26.1; deliveries/solve_a/COLLAPSE_records_v1.1_labels_turn33.md (A, C) and COLLAPSE_records_v1.1_labels_turn34.md (A, B, C).",
         "25.1 to 25.3, 26.1, 27.2, 27.5; deliveries/solve_a/COLLAPSE_records_v1.1_labels_turn33.md (A, C), COLLAPSE_records_v1.1_labels_turn34.md (A, B, C) and COLLAPSE_records_v1.1_labels_turn35.md (B, C)."),
    ]
    for old, new in reps:
        assert t.count(old) == 1, ("not unique", old[:80])
        t = t.replace(old, new)
    open(a.out, "w").write(t)
    print("%d chars; %d replacements" % (len(t), len(reps)))


if __name__ == "__main__":
    main()
