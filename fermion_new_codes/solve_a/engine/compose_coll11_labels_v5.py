#!/usr/bin/env python
"""compose_coll11_labels_v5.py (builder analyst; engine spec 30.6: "FILE 1 and FILE 5 applied with compose_coll11_labels_v5"; FILE 5 =
COLLAPSE_records_v1.1_labels_turn38.md, whose header says: "turn 37's labels stand; C5 appended to C; D4 appended to D; E2 gains its budget
line"). Mechanical and verbatim, on version 4's composition (label_v4.txt, status_v4.txt) and the ell 16 record's filled E2
(label_c25_rs64_ell16_v1.txt): C5 appended to the label line C, D4 appended to the statement D, the headings' turn lists gain "and 38" (C
and D), and E2's budget line appended to the ell 16 record's label. Args: --label-v4 --status-v4 --e2-v1 --t38 --label-out --status-out
--e2-out."""
import argparse
import re


def block(txt, key):
    m = re.search(r"^## %s \([^\n]*\n(.*?)(?=^## |\Z)" % re.escape(key), txt, re.S | re.M)
    assert m, key
    return m.group(1).strip()


def main():
    ap = argparse.ArgumentParser()
    for k in ("label-v4", "status-v4", "e2-v1", "t38", "label-out", "status-out", "e2-out"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    t38 = open(a.t38).read()
    c5 = block(t38, "C5"); d4 = block(t38, "D4")
    m = re.search(r"^## E2 \(the budget line appended\)\n(.*?)(?=^## |\Z)", t38, re.S | re.M)
    assert m
    e2b = m.group(1).strip()
    lab = open(a.label_v4).read().split("\n")
    iC = [k for k, l in enumerate(lab) if l.startswith("The balance table and the r_s 32 set (the lead's label C")]
    assert len(iC) == 1
    iC = iC[0]
    mm = re.match(r"^([^(]*\(the lead's label C, turns )([^)]*)\)", lab[iC]); assert mm
    lab[iC] = lab[iC].replace(mm.group(0), mm.group(1) + mm.group(2).replace(" and ", ", ") + " and 38)", 1)
    lab[iC] = lab[iC].rstrip() + " " + c5
    st = open(a.status_v4).read()
    dh = re.search(r"The production's record-level statement \(the lead's label D, turns ([^)]*)\): ", st); assert dh
    st = st.replace(dh.group(0), "The production's record-level statement (the lead's label D, turns %s and 38): " % dh.group(1).replace(" and ", ", "), 1)
    paras = st.split("\n\n")
    iD = [k for k, p in enumerate(paras) if p.startswith("The production's record-level statement (the lead's label D")]
    assert len(iD) == 1
    paras[iD[0]] = paras[iD[0]].rstrip() + " " + d4
    open(a.label_out, "w").write("\n".join(lab))
    open(a.status_out, "w").write("\n\n".join(paras))
    e2 = open(a.e2_v1).read().strip()
    open(a.e2_out, "w").write(e2 + " " + e2b + "\n")
    print("C +%d, D +%d, E2 +%d" % (len(c5), len(d4), len(e2b)))


if __name__ == "__main__":
    main()
