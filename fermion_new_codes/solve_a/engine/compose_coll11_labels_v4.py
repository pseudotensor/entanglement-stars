#!/usr/bin/env python
"""compose_coll11_labels_v4.py (builder analyst; engine spec 29.11: "FILE 1 to FILE 5 applied ... (compose_coll11_labels_v4 ...)"; FILE 1 =
COLLAPSE_records_v1.1_labels_turn37.md, whose header says: "turn 36's A, B, C, D, E stand; A3, B2, D3 appended to A, B, D; C3 replaces
C1's clause "and the physical sea's first adiabatic correction carries the collar current F_R[C^(1)] = -D_R, cancelling it to that
order"; C4 appended to C; E1 appended to E; E2 is new"). Mechanical and verbatim, on version 3's composition (label_v3.txt, status_v3.txt):
A3 appended to the label line A, B2 to B, C's clause replaced by C3 (its final full stop dropped so the sentence's own punctuation
continues) and C4 appended to C, D3 appended to the statement D in the status text; each heading's turn list gains "and 37"; each
replaced or extended text asserted unique. E1 and E2 belong to the ell 8 and ell 16 records' own files (written to --e1-out and --e2-out).
Args: --label-v3 --status-v3 --t37 --label-out --status-out --e1-out --e2-out."""
import argparse
import re


def blocks(path):
    txt = open(path).read()
    out = {}
    for m in re.finditer(r"^## (\S+)[^\n]*\n(.*?)(?=^## |\Z)", txt, re.S | re.M):
        out[m.group(1).rstrip(".")] = m.group(2).strip()
    return out


def main():
    ap = argparse.ArgumentParser()
    for k in ("label-v3", "status-v3", "t37", "label-out", "status-out", "e1-out", "e2-out"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    B = blocks(a.t37)
    for k in ("A3", "B2", "C3", "C4", "D3", "E1", "E2"):
        assert k in B and B[k], "FILE 1 lacks %s" % k
    lab = open(a.label_v3).read().split("\n")
    st = open(a.status_v3).read()

    def line_of(prefix):
        i = [k for k, l in enumerate(lab) if l.startswith(prefix)]
        assert len(i) == 1, (prefix, len(i))
        return i[0]
    iA = line_of("Continuation (the lead's label A"); iB = line_of("Residual columns (the lead's label B"); iC = line_of("The balance table and the r_s 32 set (the lead's label C")
    clause = "and the physical sea's first adiabatic correction carries the collar current F_R[C^(1)] = -D_R, cancelling it to that order"
    assert lab[iC].count(clause) == 1, "the C1 clause occurs %d times" % lab[iC].count(clause)
    c3 = B["C3"].rstrip()
    c3 = c3[:-1] if c3.endswith(".") else c3
    lab[iC] = lab[iC].replace(clause, c3)
    lab[iA] = lab[iA].rstrip() + " " + B["A3"]
    lab[iB] = lab[iB].rstrip() + " " + B["B2"]
    lab[iC] = lab[iC].rstrip() + " " + B["C4"]
    for i in (iA, iB, iC):
        mm = re.match(r"^([^(]*\(the lead's label [A-Z], turns? )([^)]*)\)", lab[i])
        assert mm, lab[i][:80]
        lab[i] = lab[i].replace(mm.group(0), mm.group(1) + mm.group(2).replace(" and ", ", ") + " and 37)", 1)
    dh = re.search(r"The production's record-level statement \(the lead's label D, turns? ([^)]*)\): ", st)
    assert dh, "label D heading not found"
    st = st.replace(dh.group(0), "The production's record-level statement (the lead's label D, turns %s and 37): " % dh.group(1).replace(" and ", ", "), 1)
    paras = st.split("\n\n")
    iD = [k for k, p in enumerate(paras) if p.startswith("The production's record-level statement (the lead's label D")]
    assert len(iD) == 1
    paras[iD[0]] = paras[iD[0]].rstrip() + " " + B["D3"]
    st = "\n\n".join(paras)
    open(a.label_out, "w").write("\n".join(lab))
    open(a.status_out, "w").write(st)
    open(a.e1_out, "w").write(B["E1"] + "\n")
    open(a.e2_out, "w").write(B["E2"] + "\n")
    print("label: A +%d, B +%d, C (clause replaced, +%d); status: D +%d; E1 %d chars, E2 %d chars" % (len(B["A3"]), len(B["B2"]), len(B["C4"]), len(B["D3"]), len(B["E1"]), len(B["E2"])))


if __name__ == "__main__":
    main()
