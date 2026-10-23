#!/usr/bin/env python
"""engine_tau_table_v2.py (builder analyst; version 2 = version 1 for any number of members halving tau: the observed order from every
consecutive triple (16, 8, 4 and 8, 4, 2 with tau 2 added; R27b: "tau 2 tests the competing scaling patterns") and each triple's
extrapolations, the trapping episodes per member (formation and closing). Version 1's docstring: engine spec 22.4, BUILDER: "E/E0 at matched T/rs (18, 30, 40) for tau 16, 8, 4 with the Richardson
order"): the r_s 32 members at three steps read at the same times: E/E0, the deficit D = 1 - E/E0, the observed order p = log2((D_16 - D_8)/
(D_8 - D_4)) where the differences share a sign, and the extrapolated deficit D_4 - (D_8 - D_4)/(2^p - 1) (and the first-order one, D_4 - (D_8
- D_4), beside); also R_t where a record has one. Args: --runs dir/tag (largest tau first) ... --times --md."""
import argparse
import json
import math
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--times", default="16,18,20,30,40")
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    runs = []
    for run in a.runs:
        d, tag = os.path.split(run)
        J = json.load(open(os.path.join(d, "collapse_%s.json" % tag)))
        rs = float(J["args"]["rs"]); E0 = float(J["head"]["E0"])
        recs = {round(r["T"] / rs, 6): r for r in J["records"] if r.get("E") is not None}
        runs.append((tag, float(J["args"]["tau"]), E0, recs))
    ntr = max(0, len(runs) - 2)
    trip = ["%g/%g/%g" % (runs[i][1], runs[i + 1][1], runs[i + 2][1]) for i in range(ntr)]
    L = ["## The r_s 32 tau ladder at matched times (engine_tau_table_v2.py)", "",
         "| T/rs | " + " | ".join("E/E0 (tau %g)" % t for _, t, _, _ in runs) + " | D differences | " + " | ".join("p (%s) | D extrapolated, order p (%s) | D extrapolated, first order (%s)" % (x, x, x) for x in trip) + " | " + " | ".join("R_t (tau %g)" % t for _, t, _, _ in runs) + " |",
         "|---|" + "---|" * (len(runs) + 1 + 3 * ntr + len(runs))]
    for T in [float(x) for x in a.times.split(",")]:
        vals = []; rts = []
        for tag, tau, E0, recs in runs:
            r = recs.get(round(T, 6))
            vals.append(r["E"] / E0 if r else None); rts.append(r.get("R_t") if r else None)
        D = [1.0 - v if v is not None else None for v in vals]
        diffs = ", ".join(("%+.3e" % (D[i] - D[i + 1])) if (D[i] is not None and D[i + 1] is not None) else "-" for i in range(len(D) - 1)) or "-"
        cells = []
        for i in range(ntr):
            p = Dx = D1 = None
            if all(x is not None for x in D[i:i + 3]):
                d1 = D[i] - D[i + 1]; d2 = D[i + 1] - D[i + 2]
                if d1 * d2 > 0:
                    p = math.log2(d1 / d2)
                    Dx = D[i + 2] - d2 / (2 ** p - 1) if p > 0 else None
                D1 = D[i + 2] - d2
            cells += [("%.2f" % p) if p is not None else "-", ("%.4e" % Dx) if Dx is not None else "-", ("%.4e" % D1) if D1 is not None else "-"]
        L.append("| %g | %s | %s | %s | %s |" % (T, " | ".join(("%.7f" % v) if v is not None else "-" for v in vals), diffs, " | ".join(cells),
                                               " | ".join(("%.3f" % x) if x else "-" for x in rts)))
    L += ["", "### The trapping episodes per member", ""]
    for tag, tau, E0, recs in runs:
        eps_ = []; cur = None
        for t in sorted(recs):
            if recs[t].get("R_t"):
                if cur is None:
                    cur = [t, None]
            elif cur is not None:
                cur[1] = t; eps_.append(cur); cur = None
        if cur is not None:
            eps_.append(cur)
        last = max(recs) if recs else float("nan")
        L.append("- tau %g (%s, records to T/rs %.3f): %s." % (tau, tag, last, "; ".join("formed at %.3f, %s" % (e[0], ("closed by %.3f" % e[1]) if e[1] is not None else "open at the last record") for e in eps_) or "no trapping zero"))
    L += ["", "- D = 1 - E/E0; the differences D(tau) - D(tau/2) in the order of the runs; every number above is read from the runs' json by the script."]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
