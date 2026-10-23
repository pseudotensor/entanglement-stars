#!/usr/bin/env python
"""compose_relabel_v1_7.py (builder analyst; engine spec 30.6: "RELABEL_table_v1.7 (C5, D4, E2's line; the ITEM5 version 3 relabel)"):
version 1.6 with the title's spec list and the sources line extended and four rows added before the sources line, the read-as texts quoted
from the lead's files, not retyped: (1) any text quoting the charge deficit's dependence on the step (label C) reads with the lead's C5;
(2) any text quoting the stepping ladder's reproduction of the first surface (label D) reads with D4; (3) any text on the ell 16 record's
collar budget reads as E2's budget line; (4) ITEM5_collapse_sentences_v1.md and _v2.md read as ITEM5_collapse_sentences_v3.md (turn 38).
Args: --v16 --t38 --out."""
import argparse
import re


def block(txt, head_re):
    m = re.search(r"^## %s[^\n]*\n(.*?)(?=^## |\Z)" % head_re, txt, re.S | re.M)
    assert m, head_re
    return m.group(1).strip().replace("|", "/")


def main():
    ap = argparse.ArgumentParser()
    for k in ("v16", "t38", "out"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    t = open(a.v16).read(); t38 = open(a.t38).read()
    c5 = block(t38, r"C5 "); d4 = block(t38, r"D4 "); e2 = block(t38, r"E2 \(the budget line")
    old_title = "# RELABEL_table_v1.6: the stored record lines and the wording that replaces them in every handoff and report (engine spec 22.0, 23.6, 24.1, 25.1 to 25.3, 26.1, 27.2, 27.5, 28.7, 28.9, 29.2, 29.4, 29.11)"
    assert t.count(old_title) == 1
    t = t.replace(old_title, old_title.replace("RELABEL_table_v1.6", "RELABEL_table_v1.7").replace("29.11)", "29.11, 30.4, 30.6)"))
    pm = re.search(r"Version 1\.6 = version 1\.5 with ", t)
    assert pm
    t = t.replace(pm.group(0), "Version 1.7 = version 1.6 with four rows added (engine 30.6: labels C5, D4, E2's budget line, the item 5 version 3 relabel). Version 1.6 = version 1.5 with ", 1)
    rows = ["| any text quoting the charge deficit's dependence on the step (the lead's label C) | read with \"%s\" (the lead's label C5, COLLAPSE_records_v1.1_labels_turn38.md) |" % c5,
            "| any text quoting the stepping ladder's reproduction of the first trapping surface (the lead's label D) | read with \"%s\" (the lead's label D4, COLLAPSE_records_v1.1_labels_turn38.md) |" % d4,
            "| any text on the ell 16 record's collar budget | \"%s\" (E2's budget line, COLLAPSE_records_v1.1_labels_turn38.md) |" % e2,
            "| ITEM5_collapse_sentences_v1.md and ITEM5_collapse_sentences_v2.md | read as ITEM5_collapse_sentences_v3.md (engine 30.4, turn 38) |"]
    si = t.index("\n- Sources: ")
    t = t[:si].rstrip("\n") + "\n" + "\n".join(rows) + "\n" + t[si:]
    sm = re.search(r"\n- Sources: solve/specs/SPEC_final4b_engine\.md secs\. ([^;]*); ", t)
    assert sm
    t = t.replace(sm.group(0), "\n- Sources: solve/specs/SPEC_final4b_engine.md secs. %s, 30.4, 30.6; deliveries/solve_a/COLLAPSE_records_v1.1_labels_turn38.md (C5, D4, E2), ITEM5_collapse_sentences_v3.md; " % sm.group(1), 1)
    open(a.out, "w").write(t)
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
