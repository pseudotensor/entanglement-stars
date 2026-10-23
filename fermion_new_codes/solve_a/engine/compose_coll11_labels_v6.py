#!/usr/bin/env python
"""compose_coll11_labels_v6.py (builder analyst; engine spec 31.6: "FILE 1 and FILE 3 applied (... compose_coll11_labels_v6 for C6 and
E3 ...)"; FILE 3 = COLLAPSE_records_v1.1_labels_turn39.md, whose header says: "turn 38's labels stand; C6 appended to C; E3 appended to
E"). Mechanical and verbatim: C6 appended to the production's label line C (version 5's label_v5.txt; the heading's turn list gains "and
39"), the status file unchanged (D gains nothing this turn); E3 appended to the ell 8 record's label E (label_c24_rs64_ell8_v2.txt: E with
E1 and the builder's note on the replay; E3 goes after E1, the builder's note kept last). Args: --label-v5 --e-v2 --t39 --label-out --e-out."""
import argparse
import re


def block(txt, key):
    m = re.search(r"^## %s \([^\n]*\n(.*?)(?=^## |\Z)" % re.escape(key), txt, re.S | re.M)
    assert m, key
    return m.group(1).strip()


def main():
    ap = argparse.ArgumentParser()
    for k in ("label-v5", "e-v2", "t39", "label-out", "e-out"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    t39 = open(a.t39).read()
    c6 = block(t39, "C6"); e3 = block(t39, "E3")
    lab = open(a.label_v5).read().split("\n")
    iC = [k for k, l in enumerate(lab) if l.startswith("The balance table and the r_s 32 set (the lead's label C")]
    assert len(iC) == 1
    iC = iC[0]
    mm = re.match(r"^([^(]*\(the lead's label C, turns )([^)]*)\)", lab[iC]); assert mm
    lab[iC] = lab[iC].replace(mm.group(0), mm.group(1) + mm.group(2).replace(" and ", ", ") + " and 39)", 1)
    lab[iC] = lab[iC].rstrip() + " " + c6
    open(a.label_out, "w").write("\n".join(lab))
    e = open(a.e_v2).read().strip()
    k = e.rfind(" (Builder: ")
    assert k > 0 and e.endswith(")"), "the builder's note is not at the end of the E label"
    out = e[:k].rstrip() + " " + e3 + e[k:].rstrip()
    open(a.e_out, "w").write(out + "\n")
    print("C +%d chars, E +%d chars" % (len(c6), len(e3)))


if __name__ == "__main__":
    main()
