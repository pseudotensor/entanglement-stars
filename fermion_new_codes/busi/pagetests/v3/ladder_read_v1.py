#!/usr/bin/env python
"""ladder_read_v1.py (busi; reads the B' dt ladder: runs ladder_dt<D> for D in --dts, same prep (prep_v5, orthonormal), same columns and
clocks). Per D: max |G_D - G_Dmin| and max |p_D - p_Dmin| over the clocks (Dmin the smallest step, the reference), the successive
differences |G_D - G_(D/2)|, the observed order p from the successive differences, and the Richardson estimate of each D's own error
|G_D - G_exact| ~ |G_D - G_(D/2)| / (1 - 2^-p). Decision: the largest D whose Richardson error estimate (and whose difference to Dmin)
is <= --target (1e-4, ReviewA_bpchk2_v1 sec. 4 criterion 2). Writes <dir>/ladder_read_v1.json and prints a table."""
import argparse
import json
import math
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--dts", default="4,2,1,0.5")
    ap.add_argument("--target", type=float, default=1e-4)
    a = ap.parse_args()
    dts = [float(x) for x in a.dts.split(",")]
    Z = {D: np.load(os.path.join(a.dir, "ladder_dt%g.npz" % D)) for D in dts}
    dmin = min(dts)
    rows = []
    for i, D in enumerate(sorted(dts, reverse=True)):
        G = Z[D]["G"]; Gr = Z[dmin]["G"]
        p_ = np.array([np.linalg.eigvalsh(g) for g in G]); pr = np.array([np.linalg.eigvalsh(g) for g in Gr])
        row = {"dt": D, "dG_vs_min": float(np.max(np.abs(G - Gr))), "dp_vs_min": float(np.max(np.abs(p_ - pr)))}
        if D / 2 in Z:
            row["dG_vs_half"] = float(np.max(np.abs(G - Z[D / 2]["G"])))
        rows.append(row)
    for r_ in rows:
        D = r_["dt"]
        if "dG_vs_half" in r_ and D / 2 in Z and D / 4 in Z:
            nxt = [x for x in rows if x["dt"] == D / 2][0]
            if "dG_vs_half" in nxt and nxt["dG_vs_half"] > 0:
                r_["order"] = math.log2(r_["dG_vs_half"] / nxt["dG_vs_half"])
        order = r_.get("order") or 2.0
        if "dG_vs_half" in r_:
            r_["richardson_error"] = r_["dG_vs_half"] / (1 - 2 ** (-order))
    ok = [r_["dt"] for r_ in rows if r_.get("richardson_error", r_["dG_vs_min"]) <= a.target and r_["dG_vs_min"] <= a.target]
    out = {"target": a.target, "rows": rows, "largest_dt_meeting_target": max(ok) if ok else None,
           "note": "the smallest dt is the reference; its own error is not bounded by this ladder"}
    json.dump(out, open(os.path.join(a.dir, "ladder_read_v1.json"), "w"), indent=1)
    print("| dt (a) | max dG vs dt_min | max dp vs dt_min | max dG vs dt/2 | order | Richardson error |")
    print("|---|---|---|---|---|---|")
    for r_ in rows:
        f = lambda k: ("%.3g" % r_[k]) if k in r_ else "-"
        print("| %g | %s | %s | %s | %s | %s |" % (r_["dt"], f("dG_vs_min"), f("dp_vs_min"), f("dG_vs_half"), f("order"), f("richardson_error")))
    print("largest dt meeting |dG| <= %g: %s" % (a.target, out["largest_dt_meeting_target"]))


if __name__ == "__main__":
    main()
