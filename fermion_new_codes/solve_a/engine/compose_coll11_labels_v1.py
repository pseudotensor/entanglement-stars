#!/usr/bin/env python
"""compose_coll11_labels_v1.py (builder analyst; engine spec 25.7 and 26.8: "FILE 1 applied (A, B, C in COLLAPSE_records_v1.1's header ...)"
and 24.1 (11)'s status line): composes, mechanically and verbatim, the label and the status texts of COLLAPSE_records_v1.1 from the lead's
labels files: A = turn 33's section A with turn 34's replacement made and turn 34's appended text appended; B = turn 34's section B; C =
turn 33's section C with turn 34's appended text appended; D = turn 33's section D; the status = engine 24.1 (11)'s status line (quoted from
the spec) and D, with the record line of engine 26.1. Args: --t33 --t34 --spec --label-out --status-out."""
import argparse
import re


def section(text, letter):
    m = re.search(r"^## %s\. [^\n]*\n\n(.*?)(?=\n## |\Z)" % letter, text, re.S | re.M)
    assert m, letter
    return m.group(1).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t33", required=True); ap.add_argument("--t34", required=True); ap.add_argument("--spec", required=True)
    ap.add_argument("--label-out", required=True); ap.add_argument("--status-out", required=True)
    a = ap.parse_args()
    t33 = open(a.t33).read(); t34 = open(a.t34).read(); spec = open(a.spec).read()
    A33 = section(t33, "A"); A34 = section(t34, "A")
    m = re.search(r'Replace "(.*?)" by "(.*?)"\. Append: (.*)$', A34, re.S)
    assert m, "turn 34's A has no replace/append form"
    old, new, app = m.group(1), m.group(2), m.group(3).strip()
    assert A33.count(old) == 1, "the replaced text is not unique in turn 33's A"
    A = A33.replace(old, new) + " " + app
    B = section(t34, "B")
    C33 = section(t33, "C"); C34 = section(t34, "C")
    mc = re.search(r'Append after "(.*?)": (.*)$', C34, re.S)
    assert mc, "turn 34's C has no append form"
    anchor, capp = mc.group(1), mc.group(2).strip()
    assert C33.count(anchor) == 1, "the anchor is not unique in turn 33's C"
    i = C33.index(anchor) + len(anchor)
    C = C33[:i] + (C33[i] if i < len(C33) and C33[i] in ".;" else "") + " " + capp + C33[i + (1 if i < len(C33) and C33[i] in ".;" else 0):]
    D = section(t33, "D")
    ms = re.search(r"COLLAPSE_records' status line: \"(.*?)\"", spec, re.S)
    assert ms, "24.1 (11)'s status line not found"
    status_line = ms.group(1)
    rec_line = re.search(r'the record line "(rows: raw, floor unattributed \(26\.1\))"', spec)
    assert rec_line, "26.1's record line not found"
    label = ("Continuation (the lead's label A, turns 33 and 34): %s\n\nResidual columns (the lead's label B, turn 34): %s\n\n"
             "The balance table and the r_s 32 set (the lead's label C, turns 33 and 34): %s\n\nRecord line: %s." % (A, B, C, rec_line.group(1)))
    status = "%s\n\nThe production's record-level statement (the lead's label D, turn 33): %s" % (status_line, D)
    open(a.label_out, "w").write(label + "\n"); open(a.status_out, "w").write(status + "\n")
    print("label %d chars, status %d chars" % (len(label), len(status)))


if __name__ == "__main__":
    main()
