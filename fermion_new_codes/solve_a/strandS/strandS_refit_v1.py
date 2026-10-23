#!/usr/bin/env python
"""Strand S revision 1a (the lead's judgment sec. 1 and hand-back job (1); builder analyst): the refits of c_S, R_gr and hat k_0/((aM)^2 j)
from S2's JSON on aM <= 0.1, 0.05 and 0.025 for both rectangles, with the same form X(aM) = X_0 + c (aM)^2 log(1/aM) + c' (aM)^2, the spread
of the three limits per quantity (the extrapolation's error, which replaces the 1e-5 budget if larger), and the limits against 1/(3 pi), 2/9
and 3/pi^2.  Appends a section "Revision 1a" to --result.  Args: --s2-json --result --stamp."""
import argparse
import json
import math

import numpy as np


def fit(aMs, vals, fmax):
    a = np.array(aMs)
    v = np.array(vals)
    sel = a <= fmax * (1 + 1e-12)
    X = np.stack([np.ones(sel.sum()), a[sel] ** 2 * np.log(1 / a[sel]), a[sel] ** 2], 1)
    c, *_ = np.linalg.lstsq(X, v[sel], rcond=None)
    r = X @ c - v[sel]
    return {"limit": float(c[0]), "c2_log": float(c[1]), "c2": float(c[2]), "max_resid": float(np.max(np.abs(r))), "points": int(sel.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2-json", required=True)
    ap.add_argument("--result", required=True)
    ap.add_argument("--stamp", required=True)
    a = ap.parse_args()
    S2 = json.load(open(a.s2_json))
    tg = {"c_S": 1 / (3 * math.pi), "R_gr": 2 / 9, "k0_over_aM2j": 3 / math.pi ** 2}
    out = {}
    L = ["", "## Revision 1a (%s): the refits of the extrapolation (strandS_refit_v1.py on S2's JSON; the lead's judgment sec. 1)" % a.stamp, "",
         "| rectangle | quantity | aM <= | points | limit | fit residual | limit / target - 1 |", "|---|---|---|---|---|---|---|"]
    for rect in ("light", "r1223"):
        rr = [r for r in S2["rows"] if r["rect"] == rect]
        out[rect] = {}
        for q in ("c_S", "R_gr", "k0_over_aM2j"):
            fs = {fm: fit([r["aM"] for r in rr], [r[q] for r in rr], fm) for fm in (0.1, 0.05, 0.025)}
            lims = [v["limit"] for v in fs.values()]
            out[rect][q] = {"fits": fs, "spread": float(max(lims) - min(lims)), "spread_rel": float((max(lims) - min(lims)) / tg[q])}
            for fm, v in fs.items():
                L.append("| %s | %s | %g | %d | %.10f | %.2e | %.2e |" % (rect, q, fm, v["points"], v["limit"], v["max_resid"], v["limit"] / tg[q] - 1))
    L += ["", "- The spread of the three limits (the extrapolation's error): " + "; ".join(
        "%s %s %.2e (%.2e relative)" % (rect, q, d["spread"], d["spread_rel"]) for rect, v in out.items() for q, d in v.items()) + ".",
          "- Targets: 1/(3 pi) = %.10f, 2/9 = %.10f, 3/pi^2 = %.10f. Every number above is emitted by the script." % (tg["c_S"], tg["R_gr"], tg["k0_over_aM2j"])]
    with open(a.result, "a") as fh:
        fh.write("\n".join(L) + "\n")
    json.dump(out, open(a.s2_json.replace(".json", "_refits_v1.json"), "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
