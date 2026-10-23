#!/usr/bin/env python
"""Strand 7 S7a part 2 so far (builder analyst): the throat-crossing records (r_sh = A, interpolated by the run) and the end records of the
part 2 runs that have finished, read from their JSONs and printed as markdown with no reading (the lead's part 2 judgment needs the full part 2
report, strand7_part2_report_v1.py, once the dt/4 rung and the R_max = 128 r_s run land).  Args: --runs label=path,... --md out."""
import argparse
import json


def f(x, d=6):
    if x is None:
        return "-"
    x = float(x)
    return ("%." + str(d) + "g") % x if (x == 0 or 1e-3 <= abs(x) < 1e5) else "%.4e" % x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    L = ["# Strand 7 S7a part 2 so far: the throat-crossing records of the finished part 2 runs (strand7_part2_facts_v1.py)", "",
         "| run | dt (lattice units) | T/r_s0 at r_sh = A | c_+^(r)(r_sh) | N(r_sh) | B/A | D/E_bdy | R_sh/r_s0 (spline) | stop | end T/r_s0 | end A/A(0) - 1 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for item in a.runs.split(","):
        lab, p = item.split("=", 1)
        R = json.load(open(p))["run"]
        t = R.get("throat_crossing") or {}
        recs = R.get("records") or []
        last = recs[-1] if recs else {}
        Tend = last.get("T_over_rs", last.get("T"))
        Arel = last.get("A_rel")
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            lab, f(R.get("dt")), f(t.get("T_over_rs")), f(t.get("c_plus_r")), f(t.get("N_sh")), f(t.get("B_over_A")), f(t.get("D_over_Ebdy")),
            f(t.get("R_sh_over_rs")), str(R.get("stop")), f(Tend), f(Arel)))
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
