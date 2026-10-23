#!/usr/bin/env python
"""compose_coll11_labels_v3.py (builder analyst; engine spec 28.10: "compose_coll11_labels_v3 (FILE 1 applied as it says) and
COLLAPSE_records_v1.1 re-headed"; FILE 1 = COLLAPSE_records_v1.1_labels_turn36.md, whose header says: "A is turns 33 and 34's text with its
envelope sentence replaced by A1 and A2 appended; B is turn 35's text with B1 appended; C is turn 35's text with its attribution sentence
("The attribution is by construction ... strand R's question.") replaced by C1 and C2 appended; D is turn 33's text with D1 appended and its
"Not said" sentence replaced by D2; E is new"). Mechanical and verbatim, on version 2's composition (label_v2.txt, status_v2.txt): A's
sentence from "Measured sensitivity over T/rs 11 to 14" to "when it lands." replaced by A1, A2 appended; B1 appended; C's sentence from "The
attribution is by construction" to "strand R's question." replaced by C1, C2 appended; D1 appended to D and D's sentence from "Not said:" to
its first full stop replaced by D2; the record line of RELABEL_table_v1.5 (engine 28.9); each replaced text asserted unique. E belongs to the
ell 8 control's record, not this header. Args: --label-v2 --status-v2 --t36 --spec --label-out --status-out."""
import argparse
import re


def sec36(text, key):
    m = re.search(r"^## %s [^\n]*\n\n(.*?)(?=\n## |\Z)" % key, text, re.S | re.M)
    assert m, key
    return m.group(1).strip()


def replace_span(text, start, end_incl, new):
    i = text.find(start)
    assert i >= 0 and text.count(start) == 1, start[:40]
    j = text.find(end_incl, i)
    assert j >= 0, end_incl
    return text[:i] + new + text[j + len(end_incl):]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-v2", required=True); ap.add_argument("--status-v2", required=True); ap.add_argument("--t36", required=True); ap.add_argument("--spec", required=True)
    ap.add_argument("--label-out", required=True); ap.add_argument("--status-out", required=True)
    a = ap.parse_args()
    lab = open(a.label_v2).read().rstrip("\n"); st = open(a.status_v2).read().rstrip("\n"); t36 = open(a.t36).read(); spec = open(a.spec).read()
    pars = lab.split("\n\n")
    assert len(pars) == 4 and pars[0].startswith("Continuation (the lead's label A, turns 33 and 34): ") and pars[1].startswith("Residual columns (the lead's label B, turn 35): ") \
        and pars[2].startswith("The balance table and the r_s 32 set (the lead's label C, turn 35): ") and pars[3].startswith("Record line: "), [p[:50] for p in pars]
    A = pars[0][len("Continuation (the lead's label A, turns 33 and 34): "):]
    B = pars[1][len("Residual columns (the lead's label B, turn 35): "):]
    C = pars[2][len("The balance table and the r_s 32 set (the lead's label C, turn 35): "):]
    A = replace_span(A, "Measured sensitivity over T/rs 11 to 14", "when it lands.", sec36(t36, "A1")) + " " + sec36(t36, "A2")
    B = B + " " + sec36(t36, "B1")
    C = replace_span(C, "The attribution is by construction", "strand R's question.", sec36(t36, "C1")) + " " + sec36(t36, "C2")
    m = re.search(r'the record line "(rows: raw, floor unattributed; calibrated against the step at the innermost node, first order \(28\.7\))"', spec)
    assert m, "28.9's record line not found"
    sp = st.split("\n\n")
    assert len(sp) == 2 and sp[1].startswith("The production's record-level statement (the lead's label D, turn 33): "), [x[:60] for x in sp]
    D = sp[1][len("The production's record-level statement (the lead's label D, turn 33): "):]
    D = D + " " + sec36(t36, "D1")
    i = D.find("Not said:"); assert i >= 0 and D.count("Not said:") == 1
    j = D.find(".", i)
    D = D[:i] + sec36(t36, "D2") + D[j + 1:]
    label = ("Continuation (the lead's label A, turns 33, 34 and 36): %s\n\nResidual columns (the lead's label B, turns 35 and 36): %s\n\n"
             "The balance table and the r_s 32 set (the lead's label C, turns 35 and 36): %s\n\nRecord line: %s." % (A, B, C, m.group(1)))
    status = "%s\n\nThe production's record-level statement (the lead's label D, turns 33 and 36): %s" % (sp[0], D)
    open(a.label_out, "w").write(label + "\n"); open(a.status_out, "w").write(status + "\n")
    print("label %d chars, status %d chars" % (len(label), len(status)))


if __name__ == "__main__":
    main()
