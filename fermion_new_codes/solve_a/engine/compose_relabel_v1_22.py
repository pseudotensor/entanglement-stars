#!/usr/bin/env python
"""compose_relabel_v1_22.py (builder analyst; the lead's turn 47 FILE 3, filled as ITEM7_single_object_numbers_v2.md by
engine_single_object_numbers_v2): version 1.21 with the title's source list and the sources line extended and one row added before the
sources line: the printable paragraph of ITEM7_single_object_numbers_v2 ends with the lead's words "correlations across the root, not
escape and not emitted particles", while the same file's Not printed list names "escape" and engine spec 39.5 (1) (x) bars the word in
HORIZON version 2; the row holds that paragraph from print until the lead's turn 48 rules on its closing words; no replacement wording is
supplied (the lead's). Args: --v121 --numbers --out."""
import argparse
import re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v121", required=True); ap.add_argument("--numbers", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    num = open(a.numbers).read()
    q = "correlations across the root, not escape and not emitted particles"
    assert num.count(q) == 1, num.count(q)
    assert num.count('"escape"') == 1
    row = ("| ITEM7_single_object_numbers_v2.md (the lead's turn 47 FILE 3, filled): the printable paragraph's closing words \"%s\" | the paragraph is held from print until the lead's turn 48 rules on these words: the same file's Not printed list names \"escape\" and engine spec 39.5 (1) (x) bars the word in HORIZON version 2; the paragraph's numbers stand (the second route, inbox #51, agreeing to its printed rounding); no replacement wording is supplied here |" % q)
    assert row.count("|") == 3
    t = open(a.v121).read()
    tm = re.search(r"^# RELABEL_table_v1\.21: (.*?)\(engine spec (.*?)\)$", t, re.M)
    assert tm
    t = t.replace(tm.group(0), "# RELABEL_table_v1.22: %s(engine spec %s; turn 47 FILE 3)" % (tm.group(1), tm.group(2)), 1)
    pm = re.search(r"Version 1\.21 = version 1\.20 with ", t)
    assert pm
    t = t.replace(pm.group(0), "Version 1.22 = version 1.21 with 1 row added (ITEM7_single_object_numbers_v2's printable paragraph held from print until the lead's turn 48 rules on its closing words). Version 1.21 = version 1.20 with ", 1)
    si = t.index("\n- Sources: ")
    t = t[:si].rstrip("\n") + "\n" + row + "\n" + t[si:]
    sm = re.search(r"\n- Sources: ([^\n]*)", t)
    assert sm
    t = t.replace(sm.group(0), "\n- Sources: " + sm.group(1).rstrip(".") + "; deliveries/solve_a/ITEM7_single_object_numbers_v2.md and ITEM7_single_object_numbers_v2_template.md.", 1)
    open(a.out, "x").write(t)
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
