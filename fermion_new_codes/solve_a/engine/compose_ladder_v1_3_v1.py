#!/usr/bin/env python
"""compose_ladder_v1_3_v1.py (builder analyst; engine spec 33.5 (4): "COLLAPSE_ladder_v1.3 when the v32 records change, its v31 solver
rows relabelled (33.2 (1))", and COLLAPSE_ladder_v1.2's head: "a last solve (engine_collapse_v32, ...) runs from T = 0 and this file's
version 1.3 reads it"): assembles the file mechanically, nothing retyped: (1) a title line naming the parts; (2) COLLAPSE_ladder_v1.2's
head (its lines after the title up to its first table heading), the lead's text of turn 40, verbatim; (3) engine_ladder_compare_v5's
tables on version 1.2's own inputs (the v31 sigma/a 2 member, 12:12 UTC: numbers identical to version 1.2's engine_ladder_compare_v4
tables, the solver label relabelled), the record the head reads; (4) engine_ladder_compare_v5's tables with the v32 member as sigma/a 2,
read at the stated time. Args: --v12 --v31-tables --v32-tables --v32-time --out."""
import argparse
import os


def main():
    ap = argparse.ArgumentParser()
    for k in ("v12", "v31-tables", "v32-tables", "v32-time", "out"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    L = open(a.v12).read().split("\n")
    assert L[0].startswith("# COLLAPSE_ladder_v1.2:"), L[0][:60]
    hi = [i for i, l in enumerate(L) if l.startswith("## ")]
    assert hi, "no table heading in version 1.2"
    head = L[1:hi[0]]
    t31 = open(a.v31_tables).read().rstrip("\n").split("\n")
    t32 = open(a.v32_tables).read().rstrip("\n").split("\n")
    assert t31[0].startswith("## ") and t32[0].startswith("## ")
    out = ["# COLLAPSE_ladder_v1.3: version 1.2's head (the lead's text of turn 40, verbatim), version 1.2's tables re-run by engine_ladder_compare_v5 on its own inputs (the numbers identical; the v31 solver relabelled as RELABEL_table_v1.12 states: LGMRES, scipy's default 30 Krylov vectors per Newton step, --nk-inner inert), and the sigma/a 2 member's last bounded solve on engine_collapse_v32 (--nk-restarts 0, 40 Krylov vectors, one cycle; engine spec 33.5 (1)) as engine_ladder_compare_v5 reads it at %s, re-run as its records change (engine spec 33.5 (4); composed by compose_ladder_v1_3_v1)" % a.v32_time]
    out += head
    out += ["## Version 1.2's tables on its own inputs (the v31 sigma/a 2 member at 12:12 UTC), re-run by engine_ladder_compare_v5 (%s)" % os.path.basename(a.v31_tables), ""] + t31[1:] + [""]
    out += ["## The sigma/a 2 member on engine_collapse_v32 at %s (%s)" % (a.v32_time, os.path.basename(a.v32_tables)), ""] + t32[1:] + [""]
    open(a.out, "w").write("\n".join(out) + "\n")
    print("wrote %s: %d lines (head %d lines)" % (a.out, len(out), len(head)))


if __name__ == "__main__":
    main()
