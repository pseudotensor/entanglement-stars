#!/usr/bin/env python
"""compose_coll11_labels_v7.py (builder analyst; engine spec 32.6: "FILE 4 (C7 and F1; compose_coll11_labels_v7; the production's records
re-headed)"; FILE 4 = COLLAPSE_records_v1.1_labels_turn40.md, whose header says: "turn 39's labels stand; C7 appended to C; F1 opens label
F (the controls of the coupling)"). Mechanical and verbatim, on version 6's label file (label_v6.txt): C7 appended to the paragraph of label
C (its turn list gains "and 40"); a new paragraph "The controls of the coupling (the lead's label F, turn 40): " followed by F1 verbatim,
placed after label C's paragraph. Args: --label-v6 --t40 --label-out."""
import argparse
import re


def block(txt, key):
    m = re.search(r"^## %s(?: \([^\n]*)?\n(.*?)(?=^## |\Z)" % re.escape(key), txt, re.S | re.M)
    assert m, key
    return m.group(1).strip()


def main():
    ap = argparse.ArgumentParser()
    for k in ("label-v6", "t40", "label-out"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    t40 = open(a.t40).read()
    c7 = block(t40, "C7"); f1 = block(t40, "F1")
    paras = open(a.label_v6).read().split("\n\n")
    iC = [k for k, p in enumerate(paras) if p.startswith("The balance table and the r_s 32 set (the lead's label C")]
    assert len(iC) == 1
    iC = iC[0]
    mm = re.match(r"^([^(]*\(the lead's label C, turns )([^)]*)\)", paras[iC]); assert mm
    paras[iC] = paras[iC].replace(mm.group(0), mm.group(1) + mm.group(2).replace(" and ", ", ") + " and 40)", 1)
    paras[iC] = paras[iC].rstrip() + " " + c7
    assert not any(p.startswith("The controls of the coupling (the lead's label F") for p in paras)
    paras.insert(iC + 1, "The controls of the coupling (the lead's label F, turn 40): " + f1)
    open(a.label_out, "w").write("\n\n".join(paras))
    print("C +%d, F +%d" % (len(c7), len(f1)))


if __name__ == "__main__":
    main()
