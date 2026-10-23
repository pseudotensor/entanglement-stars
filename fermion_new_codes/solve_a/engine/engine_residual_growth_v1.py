#!/usr/bin/env python
"""engine_residual_growth_v1.py (builder analyst; for the lead's reading of a solver rule shared by the ladder's members): for each given
run json (label=path), over its records' self-consistent iteration histories: the records, how many meet the stop rule, the largest
iterate count, and the largest ratio of an iterate's residual max(dN, dv) to the smallest residual before it within the same record (the
growth a switch rule keyed to the running minimum would see), with the record where it occurs; records whose running minimum reaches an
exact zero are counted and skipped from that iterate on. Every number is printed by the script. Args: --run label=json (repeated) --md."""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--md", default=None)
    a = ap.parse_args()
    out = ["## The self-consistent iteration's growth after its running minimum, per member (engine_residual_growth_v1.py)", "",
           "| member | run | records | stop rule met | largest iterate count | largest ratio to the running minimum | at T/rs | records with an exact zero minimum |",
           "|---|---|---|---|---|---|---|---|"]
    for s in a.run:
        lab, p = s.split("=", 1)
        J = json.load(open(p))
        R = [r for r in J["records"] if r.get("iteration_history")]
        worst, worst_T, zeros, maxit = 1.0, None, 0, 0
        for r in R:
            m = [max(h[1], h[2]) for h in r["iteration_history"]]
            maxit = max(maxit, len(m))
            best = m[0]
            for x in m[1:]:
                if best == 0:
                    zeros += 1
                    break
                if x / best > worst:
                    worst, worst_T = x / best, r["T_over_rs"]
                best = min(best, x)
        out.append("| %s | %s | %d | %d | %d | %.3f | %s | %d |" % (lab, p.split("/")[-1], len(R), sum(1 for r in R if r.get("converged")), maxit, worst,
                                                                  "%.3f" % worst_T if worst_T is not None else "none", zeros))
    out += ["", "- every number above is printed by the script; a member still running is read to its last record."]
    txt = "\n".join(out)
    print(txt)
    if a.md:
        open(a.md, "w").write(txt + "\n")


if __name__ == "__main__":
    main()
