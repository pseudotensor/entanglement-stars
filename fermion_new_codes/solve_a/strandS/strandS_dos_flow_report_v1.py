#!/usr/bin/env python
"""Strand S diagnostic D-a2's report (SPEC sec. 9.1; builder analyst): from strandS_dos_flow_v1.py's JSONs (the flow chart's PG background at
r_s = 16, 32, 64), per size and packet mass: N (Killing frequencies below T_H = 1/(4 pi r_s)), S_th, E_th and their parts inside R < 2 r_s and
R < r_s, and the local powers p = d log X/d log r_s between neighbouring sizes (matched T r_s).  Args: --jsons --json --md."""
import argparse
import json
import math


def f(v, d=5):
    if v is None:
        return "-"
    v = float(v)
    return "%.*g" % (d, v) if (v == 0 or 1e-3 <= abs(v) < 1e6) else "%.3e" % v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsons", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    runs = sorted([json.load(open(p)) for p in a.jsons.split(",")], key=lambda d: d["meta"]["rs"])
    L = ["# Strand S diagnostic D-a2: the flow chart's density of states (strandS_dos_flow_report_v1.py; prescribed flow (simpler version); PG radial; v declared)", "",
         "| r_s | j_max | m | N | S_th | S_th (R < 2 r_s) | S_th (R < r_s) | E_th | N (R < 2 r_s) |", "|---|---|---|---|---|---|---|---|---|"]
    tab = {}
    out = {"rows": [], "powers": []}
    for d in runs:
        rs = d["meta"]["rs"]
        for mk, row in d["rows"].items():
            t = row["total"]
            tab[(mk, rs)] = t
            out["rows"].append({"rs": rs, "mass": mk, **t})
            L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (f(rs), f(d["meta"]["j_max"]), mk.replace("m=", ""), f(t["N"]), f(t["S"]), f(t["S_ns_2rs"]),
                                                                    f(t["S_ns_rs"]), f(t["E"], 4), f(t["N_ns_2rs"])))
    L += ["", "| m | sizes | p(S_th) | p(S_th, R < 2 r_s) | p(S_th, R < r_s) | p(N) |", "|---|---|---|---|---|---|"]
    rss = sorted({d["meta"]["rs"] for d in runs})
    for mk in sorted({k[0] for k in tab}):
        for r1, r2 in zip(rss[:-1], rss[1:]):
            x1, x2 = tab.get((mk, r1)), tab.get((mk, r2))
            if not x1 or not x2:
                continue
            pw = {}
            for k in ("S", "S_ns_2rs", "S_ns_rs", "N"):
                try:
                    pw[k] = math.log(x2[k] / x1[k]) / math.log(r2 / r1)
                except (ValueError, ZeroDivisionError):
                    pw[k] = None
            out["powers"].append({"mass": mk, "rs": [r1, r2], **pw})
            L.append("| %s | %s to %s | %s | %s | %s | %s |" % (mk.replace("m=", ""), f(r1), f(r2), f(pw["S"], 3), f(pw["S_ns_2rs"], 3), f(pw["S_ns_rs"], 3), f(pw["N"], 3)))
    L += ["", "- Every number above is emitted by the script."]
    json.dump(out, open(a.json, "w"), indent=1, default=float)
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
