#!/usr/bin/env python
"""compose_relabel_v1_8.py (builder analyst; engine spec 31.6: "the labels C6 and E3 with compose_coll11_labels_v6 and RELABEL_table_v1.8"):
version 1.7 with the title's spec list and the sources line extended and four rows added before the sources line, the read-as texts
quoted from the lead's files, not retyped: (1) any text quoting the charge deficit's dependence on the averaging scale reads with the lead's
C6; (2) any text quoting the cut current's share of the deficit reads with E3; (3) ITEM5_collapse_sentences_v1.md to _v3.md read as
ITEM5_collapse_sentences_v4.md (turn 39); (4) MASSLESS_shell_v1.1.md to _v1.3.md read as MASSLESS_shell_v1.4.md (turn 39). Args: --v17
--t39 --out."""
import argparse
import re


def block(txt, head_re):
    m = re.search(r"^## %s[^\n]*\n(.*?)(?=^## |\Z)" % head_re, txt, re.S | re.M)
    assert m, head_re
    return m.group(1).strip().replace("|", "/")


def main():
    ap = argparse.ArgumentParser()
    for k in ("v17", "t39", "out"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    t = open(a.v17).read(); t39 = open(a.t39).read()
    c6 = block(t39, r"C6 "); e3 = block(t39, r"E3 ")
    tm = re.search(r"^# RELABEL_table_v1\.7: (.*?)\(engine spec (.*?)\)$", t, re.M)
    assert tm
    old_title = tm.group(0)
    assert t.count(old_title) == 1
    t = t.replace(old_title, "# RELABEL_table_v1.8: %s(engine spec %s, 31.1, 31.2, 31.6)" % (tm.group(1), tm.group(2)))
    pm = re.search(r"Version 1\.7 = version 1\.6 with ", t)
    assert pm
    t = t.replace(pm.group(0), "Version 1.8 = version 1.7 with four rows added (engine 31.6: labels C6 and E3, the item 5 version 4 and the massless version 1.4 relabels). Version 1.7 = version 1.6 with ", 1)
    rows = ["| any text quoting the charge deficit's dependence on the averaging scale | read with \"%s\" (the lead's label C6, COLLAPSE_records_v1.1_labels_turn39.md) |" % c6,
            "| any text quoting the collar's cut current as a share of the charge deficit | read with \"%s\" (the lead's label E3, COLLAPSE_records_v1.1_labels_turn39.md) |" % e3,
            "| ITEM5_collapse_sentences_v1.md, _v2.md and _v3.md | read as ITEM5_collapse_sentences_v4.md (engine 31.1, turn 39) |",
            "| MASSLESS_shell_v1.1.md, _v1.2.md and _v1.3.md | read as MASSLESS_shell_v1.4.md (engine 31.2, turn 39) |"]
    si = t.index("\n- Sources: ")
    t = t[:si].rstrip("\n") + "\n" + "\n".join(rows) + "\n" + t[si:]
    sm = re.search(r"\n- Sources: solve/specs/SPEC_final4b_engine\.md secs\. ([^;]*); ", t)
    assert sm
    t = t.replace(sm.group(0), "\n- Sources: solve/specs/SPEC_final4b_engine.md secs. %s, 31.1, 31.2, 31.6; deliveries/solve_a/COLLAPSE_records_v1.1_labels_turn39.md (C6, E3), ITEM5_collapse_sentences_v4.md, MASSLESS_shell_v1.4.md; " % sm.group(1), 1)
    open(a.out, "w").write(t)
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
