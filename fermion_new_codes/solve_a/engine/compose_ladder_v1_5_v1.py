#!/usr/bin/env python
"""compose_ladder_v1_5_v1.py (solve builder 8, analyst; engine spec 49.4 (1) and 48.1 (3)): COLLAPSE_ladder_v1.5 assembled mechanically,
nothing retyped from a run: (1) a title line; (2) the head: the sigma/a 2 member's formation read on two solves side by side (the v32 member
and the tightened v36 run), the roots table of results/engine/V36tol/readings_48_1_3_v1.md verbatim (its section from 11.5 to 12.5) and its
first-record line, with turn 57's reading (both at T/r_s 12.0; outer roots 38.0982 and 38.0931, 5.1e-03 a apart, 1.3e-04 of the root, the
lead's arithmetic: the solver spread of the formation radius; the ladder's sigma/a 2 point stays the v32 member); (3) engine_ladder_compare_v5's
tables on the v32 member's final json (its run stopped by the lead on the owner's word) with the sigma/a 4 and 8 members, verbatim; (4)
version 1.4 verbatim below as the record. Args: --v14 --readings --tables --tables-note --out."""
import argparse


def section(lines, start_prefix):
    i = next(k for k, l in enumerate(lines) if l.startswith(start_prefix))
    j = next((k for k in range(i + 1, len(lines)) if lines[k].startswith("## ")), len(lines))
    return lines[i:j]


def main():
    ap = argparse.ArgumentParser()
    for k in ("v14", "readings", "tables", "tables-note", "out"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    V14 = open(a.v14).read().rstrip("\n").split("\n")
    assert V14[0].startswith("# COLLAPSE_ladder_v1.4"), V14[0][:60]
    RD = open(a.readings).read().rstrip("\n").split("\n")
    roots = section(RD, "## Roots (trapping zeros of N - |v|)")
    T = open(a.tables).read().rstrip("\n").split("\n")
    out = ["# COLLAPSE_ladder_v1.5: the sigma/a 2 member's formation on two solves side by side (engine spec 49.4 (1), 48.1 (3)); the ladder's tables re-run on the sigma/a 2 member's final records; version 1.4 below as the record",
           "",
           "The sigma/a 2 point of the ladder is the v32 member (engine_collapse_v32, the Newton-Krylov solve with 2 Krylov restarts under the stored stop rule). "
           "A second solve of the same member from the same initial data, engine_collapse_v36 at tenfold-tightened tolerances, ran to T/r_s 12.75. "
           "Both form their first trapping surface at T/r_s 12.0, with outer roots 38.0982 (the v32 member) and 38.0931 (the tightened run): "
           "5.1e-03 lattice units apart, 1.3e-04 of the root (lead's arithmetic, turn 57). "
           "That difference is the solver spread of the formation radius. "
           "The tightened run's record at 12.75 found no solution within its tightened rule and is not part of either history.",
           "",
           "## The formation on the two solves (" + a.readings.split("/")[-1] + ", verbatim)", ""]
    out += roots[1:]
    out += [""]
    out += ["## The ladder's tables on the sigma/a 2 member's final records (" + a.tables_note + ")", ""]
    out += [l for l in T if not l.startswith("# ")]
    out += ["", "## Version 1.4 (the record, verbatim)", ""]
    out += V14
    open(a.out, "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
