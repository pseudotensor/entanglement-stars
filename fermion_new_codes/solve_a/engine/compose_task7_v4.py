#!/usr/bin/env python
"""compose_task7_v4.py (builder analyst; engine spec 29.7 and 29.11: "the sense-odd term is reported to the icloud lane as a measured term
(TO_ICLOUD_task7_v4.md, the table copied, varpi = theta0 stated as the lane's definition on the stored orbits) ... the kernel discrepancy
goes to the icloud lane in the same note"): assembles the note mechanically: the lead's two passages of 29.7 verbatim (from "S11
(s11_advance_v1):" to the end of its ruling, and from "GAP13 (ps_defect_v1):" to the end of its ruling), then engine_s11_advance_v1's
table verbatim, then the kernel-moment line of engine_ps_defect_v1's output verbatim. Args: --spec --s11 --psd --out."""
import argparse
import re


def main():
    ap = argparse.ArgumentParser()
    for k in ("spec", "s11", "psd", "out"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    spec = open(a.spec).read()
    sec = re.search(r"^### 29\.7 .*?(?=^### 29\.8 )", spec, re.S | re.M).group(0)
    p1 = re.search(r"S11 \(s11_advance_v1\):.*?varpi = theta0 stated as the lane's definition on the stored orbits\)\.", sec, re.S)
    p2 = re.search(r"GAP13 \(ps_defect_v1\):.*?the kernel discrepancy goes to the icloud lane in the same note\.", sec, re.S)
    assert p1 and p2, "29.7's passages not found"
    s11 = open(a.s11).read().rstrip("\n")
    kl = [l for l in open(a.psd).read().split("\n") if l.startswith("- s_ell^2 on the module's own kernel")]
    assert len(kl) == 1
    L = ["# TO_ICLOUD_task7_v4: the periastron laws' sense-odd term and this lane's averaging-kernel moment (for the icloud lane; the goal's task 7)", "",
         "Assembled by the builder (analyst) on the lead's instruction of turn 37 (engine spec 29.7); the lead's rulings verbatim, then the tables printed by the scripts. The orbits are the stored ones of engine_ppn_v10 ([011]) and engine_ppn_v8 ([111]); no new run.", "",
         "## The lead's rulings (engine spec 29.7, verbatim)", "", p1.group(0), "", p2.group(0), "",
         "## The periastron table (engine_s11_advance_v1.py, verbatim; varpi = theta0 is this lane's definition on the stored orbits)", "", s11, "",
         "## This lane's averaging kernel (engine_ps_defect_v1.py's line, verbatim)", "", kl[0], ""]
    open(a.out, "w").write("\n".join(L) + "\n")
    print("wrote %s (%d lines)" % (a.out, len(L)))


if __name__ == "__main__":
    main()
