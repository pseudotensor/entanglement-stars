#!/usr/bin/env python
"""engine_rows_floor_ratio_v1.py (builder analyst; engine spec 27.2: "the two ratios of the innermost-node maxima (tau 8 to 4, ell 4 to 8)
are the calibration label B waits for, read in turn 36"): reads engine_rows_floor_v1's records tables for the production (tau 8, ell 4), the
tau 4 control (tau 4, ell 4) and the ell 8 control (tau 8, ell 8) and prints, per T/rs and row (rr, tt), the three runs' support maxima of
the full (centred) residual with their nodes, whether each maximum sits at the innermost node r = ell, and the ratios production / tau 4
control and production / ell 8 control; the same for the residual with d_t k set to zero. The fields versions are read from each table's
heading. Args: --prod --tau4 --ell8 (the .md tables) --md."""
import argparse
import re


def read(path):
    t = open(path).read()
    head = t.split("\n")[0]
    fv = re.search(r"fields (engine_fields_v\d+)", head); tau = re.search(r"tau (\d+(?:\.\d+)?)\)", head); tag = re.search(r"records of (\S+) \(", head)
    rows = {}
    for m in re.finditer(r"^\| (\d+\.\d+) \| ([^|]+?) \| ([0-9.e+-]+) \(([0-9.]+)\) \| ([0-9.e+-]+) \(([0-9.]+)\) \|$", t, re.M):
        kind = "full" if m.group(2).startswith("full") else "zero"
        rows[(float(m.group(1)), kind)] = (float(m.group(3)), float(m.group(4)), float(m.group(5)), float(m.group(6)))
    return {"fields": fv.group(1) if fv else "-", "tau": tau.group(1) if tau else "-", "tag": tag.group(1) if tag else path, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prod", required=True); ap.add_argument("--tau4", required=True); ap.add_argument("--ell8", required=True); ap.add_argument("--md", required=True)
    a = ap.parse_args()
    P, T4, E8 = read(a.prod), read(a.tau4), read(a.ell8)
    ell = {"prod": 4.0, "tau4": 4.0, "ell8": 8.0}
    L = ["## The rows' innermost-node maxima: the production against the tau 4 and ell 8 controls (engine_rows_floor_ratio_v1.py)", "",
         "- production: %s (tau %s, ell 4; fields %s); tau 4 control: %s (tau %s, ell 4; fields %s); ell 8 control: %s (tau %s, ell 8; fields %s)." % (
             P["tag"], P["tau"], P["fields"], T4["tag"], T4["tau"], T4["fields"], E8["tag"], E8["tau"], E8["fields"]),
         "- each entry: the support maximum of the residual over the largest term (its node r; * marks a maximum at the innermost node r = ell).", "",
         "| T/rs | d_t k | row | production (tau 8, ell 4) | tau 4 control | ell 8 control | production / tau 4 | production / ell 8 |",
         "|---|---|---|---|---|---|---|---|"]
    for t in sorted({k[0] for k in P["rows"]} & {k[0] for k in T4["rows"]} & {k[0] for k in E8["rows"]}):
        for kind, lab in (("full", "full (centred)"), ("zero", "set to zero")):
            if not all((t, kind) in X["rows"] for X in (P, T4, E8)):
                continue
            p, q, e = P["rows"][(t, kind)], T4["rows"][(t, kind)], E8["rows"][(t, kind)]
            for row, i in (("rr", 0), ("tt", 2)):
                cell = lambda v, r, el: "%.4e (%.0f%s)" % (v, r, "*" if abs(r - el) < 1e-9 else "")
                L.append("| %.3f | %s | %s | %s | %s | %s | %.3f | %.3f |" % (t, lab, row, cell(p[i], p[i + 1], ell["prod"]), cell(q[i], q[i + 1], ell["tau4"]),
                                                                    cell(e[i], e[i + 1], ell["ell8"]), p[i] / q[i], p[i] / e[i]))
    L += ["", "- every number above is read from the three tables or printed by the script (the ratios); the three runs' fields versions differ as listed."]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
