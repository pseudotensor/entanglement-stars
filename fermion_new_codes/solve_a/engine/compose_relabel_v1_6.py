#!/usr/bin/env python
"""compose_relabel_v1_6.py (builder analyst; engine spec 29.11: "RELABEL_table_v1.6 (B2's innermost-node row; C3; the J3P v5 relabel)"):
version 1.5 with the title's spec list and the sources line extended and three rows added before the sources line, each read-as text quoted
from the files, not retyped: (1) any text reading the rows' innermost-node maxima (label B's calibration) as a certified residual reads as
the lead's B2 (COLLAPSE_records_v1.1_labels_turn37.md); (2) any text identifying D_R with Tr(H_R P_dot) or stating the first-adiabatic
current's cancellation F_R[C^(1)] = -D_R (C1's clause) reads as C3; (3) the three J3P_raymap_v5 deliveries read as engine 29.4's relabel
("the stored history's ray-map diagnostics, packets re-tuned per history; no temperature"). Each replaced text asserted unique.
Args: --v15 --spec --t37 --out."""
import argparse
import re


def block(txt, key):
    m = re.search(r"^## %s [^\n]*\n(.*?)(?=^## |\Z)" % re.escape(key), txt, re.S | re.M)
    assert m, key
    return m.group(1).strip()


def main():
    ap = argparse.ArgumentParser()
    for k in ("v15", "spec", "t37", "out"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    t = open(a.v15).read(); spec = open(a.spec).read(); t37 = open(a.t37).read()
    b2 = block(t37, "B2"); c3 = block(t37, "C3")
    jm = re.search(r'relabelled "(the stored history\'s ray-map diagnostics, packets re-tuned per history; no temperature)"', spec)
    assert jm
    old_title = "# RELABEL_table_v1.5: the stored record lines and the wording that replaces them in every handoff and report (engine spec 22.0, 23.6, 24.1, 25.1 to 25.3, 26.1, 27.2, 27.5, 28.7, 28.9)"
    assert t.count(old_title) == 1
    t = t.replace(old_title, old_title.replace("RELABEL_table_v1.5", "RELABEL_table_v1.6").replace("28.7, 28.9)", "28.7, 28.9, 29.2, 29.4, 29.11)"))
    pm = re.search(r"Version 1\.5 = version 1\.4 with ", t)
    assert pm
    t = t.replace(pm.group(0), "Version 1.6 = version 1.5 with three rows added (engine 29.11: label B2's innermost-node row, label C3, the J3P version 5 relabel of engine 29.4). Version 1.5 = version 1.4 with ", 1)
    rows = ["| any text reading the evolution rows' innermost-node maxima (label B's calibration against the step at r = 4) as a certified residual | \"%s\" (the lead's label B2, COLLAPSE_records_v1.1_labels_turn37.md) |" % b2.replace("|", "/"),
            "| any text identifying D_R with Tr(H_R P_dot) or stating that the first adiabatic correction's collar current F_R[C^(1)] = -D_R cancels it (label C1's clause, COLLAPSE_records_v1.1_labels_turn36.md) | \"%s\" (the lead's label C3, COLLAPSE_records_v1.1_labels_turn37.md) |" % c3.replace("|", "/"),
            "| J3P_raymap_v5_c19_rs64_main.md, J3P_raymap_v5_c19_rs64_j1.5.md, J3P_raymap_v5_c19_rs64_tau4_main.md (and their json) read as a temperature or a test of T against T_H | \"%s\" (engine 29.4); the controlled read is J3' version 6 on the frozen late exteriors with fixed packets |" % jm.group(1)]
    si = t.index("\n- Sources: ")
    t = t[:si].rstrip("\n") + "\n" + "\n".join(rows) + "\n" + t[si:]
    sm = re.search(r"\n- Sources: solve/specs/SPEC_final4b_engine\.md secs\. ([^;]*); ", t)
    assert sm
    t = t.replace(sm.group(0), "\n- Sources: solve/specs/SPEC_final4b_engine.md secs. %s, 29.2, 29.4, 29.11; deliveries/solve_a/COLLAPSE_records_v1.1_labels_turn37.md (B2, C3); " % sm.group(1), 1)
    open(a.out, "w").write(t)
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
