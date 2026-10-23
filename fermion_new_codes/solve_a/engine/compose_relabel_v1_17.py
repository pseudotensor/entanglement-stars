#!/usr/bin/env python
"""compose_relabel_v1_17.py (builder analyst; inbox #41, the owner's approval of reviewer A's one-object / no-mutual-information ruling, its
S4 C: "The delivered ITEM7_s7_part3_sentence_v2.md is withdrawn from the paper in its 2-preparation form"): version 1.16 with the title's
source list and the sources line extended and one row added before the sources line. No delivered file of this lane prints mutual
information (checked by grep over deliveries/solve_a/*.md before this version), so D15 adds no row. Args: --v116 --out."""
import argparse
import re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v116", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = ["| ITEM7_s7_part3_sentence_v2.md (and version 1) | withdrawn from the paper in its 2-preparation form (inbox #41, S4 C: the owner's one-object ruling; REVIEWA_RULING_oneobject_MI.md): the windowed rotation's D_det prints, if at all, only as \"a perturbation of the correlations across shells n_L to n_R inside the root reached the collar with covariance witness D_det >= 5e-4 to 1.5e-3, a fraction [value] of its size\" and only if those shell bands lie inside the outer root, which S4 C (revised) decides from the lane's own files; otherwise nothing from part 3 prints |"]
    t = open(a.v116).read()
    tm = re.search(r"^# RELABEL_table_v1\.16: (.*?)\(engine spec (.*?); 35\.1\)$", t, re.M)
    assert tm
    t = t.replace(tm.group(0), "# RELABEL_table_v1.17: %s(engine spec %s; 35.1; inbox #41)" % (tm.group(1), tm.group(2)), 1)
    pm = re.search(r"Version 1\.16 = version 1\.15 with ", t)
    assert pm
    t = t.replace(pm.group(0), "Version 1.17 = version 1.16 with 1 row added (inbox #41: the S7 part 3 sentence withdrawn from the paper in its 2-preparation form). Version 1.16 = version 1.15 with ", 1)
    si = t.index("\n- Sources: ")
    t = t[:si].rstrip("\n") + "\n" + "\n".join(rows) + "\n" + t[si:]
    sm = re.search(r"\n- Sources: ([^\n]*)", t)
    assert sm
    t = t.replace(sm.group(0), "\n- Sources: " + sm.group(1).rstrip(".") + "; inbox #41 and <internal>/REVIEWA_RULING_oneobject_MI.md.", 1)
    open(a.out, "w").write(t)
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
