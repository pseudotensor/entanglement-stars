#!/usr/bin/env python
"""compose_relabel_v1_12.py (builder analyst; engine spec 33.2 (1): "the builder composes RELABEL_table_v1.12 with: version 1 read as 1.1,
the ten rows closed; J3P_raymap_v8_tau4_swingmax read as v9 (version 8's oscillation term on that run the wiggle's, withdrawn);
J3P_raymap_v8_prod19_max read as _r2 (equal to v9_prod19_max; the 39.75-partner budget the record of version 8's letter); _min_r2 a record
equal to v8_prod19_min (the partner label only); COLLAPSE_ladder_v1.2's rows naming the v31 solver read 'LGMRES, scipy's default 30 Krylov
vectors per Newton step (--nk-inner inert)' (33.5)"): version 1.11 with the title's source list and the sources line extended and six rows
added before the sources line. The numbers are read from the J3' mds by this script (the oscillation column's distinct values; the budget
tables' rows compared): nothing retyped. Args: --v111 --j3dir --ladder --out."""
import argparse
import os
import re


def cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def budget(md):
    lines = open(md).read().splitlines()
    hi = [i for i, l in enumerate(lines) if l.startswith("| omega r_s | m/omega | hold/r_s | frozen stage | T/T_H |")]
    assert len(hi) == 1, (md, len(hi))
    head = cells(lines[hi[0]])
    rows = []
    for l in lines[hi[0] + 2:]:
        if not l.startswith("|"):
            break
        rows.append(dict(zip(head, cells(l))))
    return head, rows


def main():
    ap = argparse.ArgumentParser()
    for k in ("v111", "j3dir", "ladder", "out"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    J = lambda n: os.path.join(a.j3dir, n)
    _, s8 = budget(J("J3P_raymap_v8_tau4_swingmax.md"))
    _, s9 = budget(J("J3P_raymap_v9_tau4_swingmax.md"))
    o8 = sorted(set(r["oscillation"] for r in s8 if r["oscillation"] not in ("-", "")))
    o9 = sorted(set(r["oscillation"] for r in s9 if r["oscillation"] not in ("-", "")))
    print("v8 swingmax oscillation values", o8, "v9", o9)
    _, m8 = budget(J("J3P_raymap_v8_prod19_min.md"))
    _, m8r2 = budget(J("J3P_raymap_v8_prod19_min_r2.md"))
    same_min = sum(1 for x, y in zip(m8, m8r2) if x == y)
    print("prod19_min vs _r2: %d of %d budget rows identical (%d and %d rows)" % (same_min, max(len(m8), len(m8r2)), len(m8), len(m8r2)))
    _, x9 = budget(J("J3P_raymap_v9_prod19_max.md"))
    _, x8r2 = budget(J("J3P_raymap_v8_prod19_max_r2.md"))
    same_max = sum(1 for x, y in zip(x9, x8r2) if x == y)
    print("prod19_max v9 vs v8_r2: %d of %d budget rows identical (%d and %d rows)" % (same_max, max(len(x9), len(x8r2)), len(x9), len(x8r2)))
    lad = open(a.ladder).read()
    stored = "15 Krylov vectors"
    nlad = lad.count(stored)
    print("COLLAPSE_ladder_v1.2: %d occurrence(s) of %r" % (nlad, stored))
    assert nlad >= 1

    rows = [
        "| TEMPERATURE_solved_exterior_v1.md | read as TEMPERATURE_solved_exterior_v1.1.md (engine 33.2 (1), turn 41: the icloud lane's review applied in full, items 1 to 5; version 1 stays the record) |",
        "| version 1.11's ten rows on TEMPERATURE_solved_exterior_v1.md (the umbrella row and the nine clause rows) | closed: TEMPERATURE_solved_exterior_v1.1.md replaces those clauses (engine 33.2 (1)) |",
        "| J3P_raymap_v8_tau4_swingmax.md and .json (the 36.25 run's version 8 rebudget; oscillation column %s) | read as J3P_raymap_v9_tau4_swingmax (oscillation column %s, the pair actually run, 36.25 and 39.5): version 8's oscillation term on that run is the late-phase table's wiggle (39.75 and 39.5), withdrawn (engine 33.2 (1), reading (ii)) |" % (", ".join(o8), ", ".join(o9)),
        "| J3P_raymap_v8_prod19_max.md and .json (the production's maximum with the 39.75 run as its tau 4 partner) | read as J3P_raymap_v8_prod19_max_r2 (the 36.25 run as partner; equal to J3P_raymap_v9_prod19_max: %d of %d budget rows identical); the 39.75-partner budget is the record of version 8's letter (engine 33.2 (1)) |" % (same_max, max(len(x9), len(x8r2))),
        "| J3P_raymap_v8_prod19_min_r2.md and .json | a record equal to J3P_raymap_v8_prod19_min (%d of %d budget rows identical; the partner label only): the minimum's matched phase is the tau 4 trough 39.5 (engine 33.2 (1), 33.0 (ii)) |" % (same_min, max(len(m8), len(m8r2))),
        "| COLLAPSE_ladder_v1.2.md, the members table's sigma/a 2 solver \"newton (at most 8 Newton steps, 15 Krylov vectors)\" | reads \"15 Krylov vectors\" as \"LGMRES, scipy's default 30 Krylov vectors per Newton step (--nk-inner inert)\" (engine 33.5; the v32 builder's finding: v31's --nk-inner never reached the solver) |",
    ]
    t = open(a.v111).read()
    tm = re.search(r"^# RELABEL_table_v1\.11: (.*?)\(engine spec (.*?); the icloud lane's review of the solved-exterior temperature, inbox #14\)$", t, re.M)
    assert tm
    t = t.replace(tm.group(0), "# RELABEL_table_v1.12: %s(engine spec %s; the icloud lane's review of the solved-exterior temperature, inbox #14; 33.2, 33.5)" % (tm.group(1), tm.group(2)), 1)
    pm = re.search(r"Version 1\.11 = version 1\.10 with ", t)
    assert pm
    t = t.replace(pm.group(0), "Version 1.12 = version 1.11 with %d rows added (engine 33.2 (1), turn 41: the temperature sentence read as version 1.1 and version 1.11's ten rows closed; the J3' rebudgets' readings; the v31 solver's Krylov label, 33.5), the numbers read by compose_relabel_v1_12. Version 1.11 = version 1.10 with " % len(rows), 1)
    si = t.index("\n- Sources: ")
    t = t[:si].rstrip("\n") + "\n" + "\n".join(rows) + "\n" + t[si:]
    sm = re.search(r"\n- Sources: ([^\n]*)", t)
    assert sm
    t = t.replace(sm.group(0), "\n- Sources: " + sm.group(1).rstrip(".") + "; solve/specs/SPEC_final4b_engine.md secs. 33.2, 33.5; deliveries/solve_a/TEMPERATURE_solved_exterior_v1.1.md; solve/results/obs/J3v7/J3P_raymap_v8_tau4_swingmax, _v9_tau4_swingmax, _v8_prod19_max_r2, _v9_prod19_max, _v8_prod19_min, _v8_prod19_min_r2 (.md); solve/lead/api/turn41/v32_builder_report.md.", 1)
    open(a.out, "w").write(t)
    print("wrote %s (%d rows added)" % (a.out, len(rows)))


if __name__ == "__main__":
    main()
