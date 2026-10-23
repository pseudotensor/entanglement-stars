#!/usr/bin/env python3
"""build_to_icloud_task7_v3.py (builder analyst): assembles deliveries/TO_ICLOUD_task7_v3.md on the lead's instruction of turn 25
(SPEC_final4b_engine.md sec. 17.2: "What E1, .lane and .alt may be told now: TO_ICLOUD_task7_v3.md (v2 superseded) with version 8's
tables, version 9's [111] ladder and round-trip rows and the ray fits as finite-boundary values with their convention, the settled formulas
above ..., the [011] closed form with the 640 r_g comparison and the window caveat, the relabellings; the request to E1 unchanged"). Nothing
is retyped: the lead's section is copied from the spec, the request to E1 from version 2 of the hand-off, the tables from the result files
(engine_ppn_v8.py, engine_ppn_v9.py), each unedited. Args: --spec --v2 --ppn8 --ppn9 --out --stamp."""
import argparse


def section(text, head, stop_prefix):
    i = text.index(head)
    j = text.index(stop_prefix, i + len(head))
    return text[i:j].rstrip() + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--v2", required=True)
    ap.add_argument("--ppn8", required=True)
    ap.add_argument("--ppn9", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stamp", required=True, help="date -u at assembly, e.g. '2026-10-15 19:50 UTC'")
    a = ap.parse_args()
    spec = open(a.spec).read()
    lead = section(spec, "### 17.2 (e) T7.1", "### 17.3")
    lead_body = lead.split("\n", 1)[1].strip() + "\n"
    v2 = open(a.v2).read()
    req = section(v2, "## The request to E1", "## Table:")
    req_body = req.split("\n", 1)[1].strip() + "\n"
    t8 = open(a.ppn8).read().strip() + "\n"
    t9 = open(a.ppn9).read().strip() + "\n"
    L = ["# To the icloud lane (.lane, strand E1), to .lane and to .alt from solve_a: task 7's lattice corrections, version 3 (supersedes TO_ICLOUD_task7_v2.md)", "",
         "Assembled by the builder (analyst, %s) on the lead's instruction of turn 25 (SPEC_final4b_engine.md sec. 17.2, \"What E1, .lane and .alt may be told now\"). "
         "The section below is the lead's text verbatim: it states what is settled (the leading ray coefficients, the half-integer term, the [111] periastron with its first "
         "post-Newtonian correction, the radial round trip) and what is not (the generic-direction r_g/b coefficients, the [011] periastron, whose closed form and 640 r_g "
         "comparison it gives with the window caveat), and the relabellings (\"The R_far ladder passes to four digits\" of version 2's text withdrawn; the r_E = 20 r_g round "
         "trip inside the collar, exact-PG continuation; the exit-sphere columns boundary-dependent diagnostics). The request to E1 is version 2's, unchanged. The tables "
         "are script-made (engine_ppn_v8.py, engine_ppn_v9.py) and appended unedited; version 9's ray fits are finite-boundary values at the stated R_far/b. Version 10 "
         "(rays extrapolated to lambda = 0, the [011] orbits on the m a and A/r_g ladder) is not part of this hand-off: its red team (T7.1 part 2) comes first." % a.stamp, "",
         "## The lead's text (engine spec sec. 17.2, turn 25, verbatim)", "", lead_body,
         "## The request to E1 (unchanged from version 2)", "", req_body,
         "## Table: T7.1 version 8 (results/engine/T7/ppn8_t7m0.md, unedited)", "", t8,
         "## Table: T7.1 version 9 (results/engine/T7/ppn9_t7m0.md, unedited)", "", t9]
    open(a.out, "w").write("\n".join(L))
    print("written", a.out, sum(len(x) + 1 for x in L), "characters")


if __name__ == "__main__":
    main()
