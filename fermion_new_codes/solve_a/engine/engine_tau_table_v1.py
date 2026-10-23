#!/usr/bin/env python
"""engine_tau_table_v1.py (builder analyst; engine spec 22.4, BUILDER: "E/E0 at matched T/rs (18, 30, 40) for tau 16, 8, 4 with the Richardson
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
    L = ["## The r_s 32 tau ladder at matched times (engine_tau_table_v1.py)", "",
         "| T/rs | " + " | ".join("E/E0 (tau %g)" % t for _, t, _, _ in runs) + " | D differences | observed order p | D extrapolated (order p) | D extrapolated (first order) | " + " | ".join("R_t (tau %g)" % t for _, t, _, _ in runs) + " |",
         "|---|" + "---|" * (2 * len(runs) + 4)]
    for T in [float(x) for x in a.times.split(",")]:
        vals = []; rts = []
        for tag, tau, E0, recs in runs:
            r = recs.get(round(T, 6))
            vals.append(r["E"] / E0 if r else None); rts.append(r.get("R_t") if r else None)
        D = [1.0 - v if v is not None else None for v in vals]
        p = Dx = D1 = None; diffs = "-"
        if len(D) >= 3 and all(x is not None for x in D[:3]):
            d1 = D[0] - D[1]; d2 = D[1] - D[2]
            diffs = "%+.3e, %+.3e" % (d1, d2)
            if d1 * d2 > 0:
                p = math.log2(d1 / d2)
                Dx = D[2] - d2 / (2 ** p - 1) if p > 0 else None
            D1 = D[2] - d2
        L.append("| %g | %s | %s | %s | %s | %s | %s |" % (T, " | ".join(("%.7f" % v) if v is not None else "-" for v in vals), diffs, ("%.2f" % p) if p is not None else "-",
                                                     ("%.4e" % Dx) if Dx is not None else "-", ("%.4e" % D1) if D1 is not None else "-",
                                                     " | ".join(("%.3f" % x) if x else "-" for x in rts)))
    L += ["", "- D = 1 - E/E0; the differences D(tau) - D(tau/2) in the order of the runs; every number above is read from the runs' json by the script."]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
