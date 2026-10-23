#!/usr/bin/env python
"""s24_analyze_v1.py (busi, inbox #420, NOTCOMP item 2.4): the channel tail's common mode against the records' time interpolation.
Reads the lean runs of s24_lean_queue_v1.sh (kappa 12, 16, 24, 48; linear = the production's ramp, cr = Catmull-Rom C1, spline = natural
cubic spline C2) and the production's own linear runs (v6o, full mode, 8192 sites): the per-chain exterior-cut excess x_kappa at the cuts
2, 16, 24, 30, 48 r_s at every 10 r_s from 40 to 100, and the weighted terms 16 kappa x_kappa (the paper's sum weights).
Prints per kappa and pair: production, lean linear, cr, spline; the ratios to linear; the code-path check (lean linear against production).
Usage: python s24_analyze_v1.py S24DIR OUT.json
"""
import json
import os
import sys

PROD = {12: "<internal>/v6o_B2_k12_summary.json", 16: "<internal>/v6o_B2_k16_summary.json",
        24: "<internal>/v6o_B2r2b_k24_k24_summary.json", 48: "<internal>/v6o_B2r2b_k48_k48_summary.json"}
CUTS = ["2", "16", "24", "30", "48"]
CLOCKS = [40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]


def prod_x(K):
    p = PROD[K]
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    out = {}
    for r in d["rows"]:
        t = round(float(r["t_over_rs"]), 6)
        pc = r["per_channel"].get(str(K))
        if pc is None or "d13" not in pc:
            continue
        cu = pc["d13"].get("cuts", {})
        out[t] = {c: (cu[c]["excess"] if c in cu and cu[c].get("excess") is not None else None) for c in CUTS}
    return out


def lean_x(path, K):
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    out = {}
    for r in d["rows"]:
        t = round(float(r["t_over_rs"]), 6)
        pc = r["per_channel"].get(str(K), {})
        cu = pc.get("interior", {}).get("cuts", {})
        out[t] = {c: (cu[c]["excess"] if c in cu else None) for c in CUTS}
    return {"x": out, "final": d.get("final")}


def main():
    sd, outp = sys.argv[1], sys.argv[2]
    res = {"kappas": {}}
    for K in (12, 16, 24, 48):
        P = prod_x(K)
        runs = {m: lean_x(os.path.join(sd, "runs", "k%d_%s" % (K, m), "k%d_%s_k%d_summary.json" % (K, m, K)), K) for m in ("linear", "cr", "spline")}
        rows = []
        for t in CLOCKS:
            for c in CUTS:
                row = {"T": t, "cut": c, "prod": (P or {}).get(round(t, 6), {}).get(c)}
                for m, R in runs.items():
                    row[m] = None if R is None else R["x"].get(round(t, 6), {}).get(c)
                if all(row[k] is None for k in ("prod", "linear", "cr", "spline")):
                    continue
                ref = row["linear"] if row["linear"] is not None else row["prod"]
                for m in ("cr", "spline"):
                    row[m + "_over_linear"] = (row[m] / ref) if (row[m] is not None and ref not in (None, 0.0)) else None
                row["lean_minus_prod"] = (row["linear"] - row["prod"]) if (row["linear"] is not None and row["prod"] is not None) else None
                for m in ("prod", "linear", "cr", "spline"):
                    row["w_" + m] = None if row[m] is None else 16.0 * K * row[m]
                rows.append(row)
        res["kappas"][str(K)] = {"rows": rows, "final": {m: (None if R is None else R["final"]) for m, R in runs.items()}}
        print("kappa %d (weight %d): finals %s" % (K, 16 * K, res["kappas"][str(K)]["final"]))
        print("   T    cut   16k x prod    16k x linear   16k x cr      16k x spline   cr/lin   spl/lin  lean-prod")
        f = lambda x: ("%+.4e" % x) if x is not None else "     -     "
        g = lambda x: ("%+.3f" % x) if x is not None else "  -  "
        for r in rows:
            print("  %5.1f %4s  %s  %s  %s  %s  %s  %s  %s" % (r["T"], r["cut"], f(r["w_prod"]), f(r["w_linear"]), f(r["w_cr"]), f(r["w_spline"]), g(r["cr_over_linear"]),
                  g(r["spline_over_linear"]), f(r["lean_minus_prod"])))
    json.dump(res, open(outp, "w"), indent=1)


if __name__ == "__main__":
    main()
