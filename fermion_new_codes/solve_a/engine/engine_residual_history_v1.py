#!/usr/bin/env python
"""engine_residual_history_v1.py (builder analyst; engine spec 30.3: "the builder prints the residual's history over the first 20 records
(stationary, oscillating or growing) into the status file"): reads an engine_collapse_v25/v27 run json and prints, for its first n
records, the self-consistent iteration's unmixed residual max(max |N_out - N_in|, max |v_out - v_in|) of the returned iterate (the
smallest over the record's iterations, the engine's own rule), its iteration index, |dE/E| there, the iteration count and whether the
stop rule was met; then the history's reading by a rule fixed here (builder's choice, labelled): over the n records, s = the
least-squares slope of log10(residual) against the record index times (n - 1) (the fitted change in decades), c = the number of sign
changes of successive differences over the n - 2 interior records, q = max/min of the residuals; "growing" if s > log10(2), "falling" if
s < -log10(2), else "oscillating" if c > (n - 2)/2 and q > 2, else "stationary". Every number is printed by the script. Args: --run-json
--n (20) --label --md (optional)."""
import argparse
import json
import math


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-json", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--label", required=True)
    ap.add_argument("--md", default=None)
    a = ap.parse_args()
    J = json.load(open(a.run_json))
    R = [r for r in J["records"] if "iteration_history" in r and "E" in r][:a.n]
    rows, res = [], []
    for r in R:
        ih = r["iteration_history"]
        vals = [max(h[1], h[2]) for h in ih]
        if r.get("converged"):
            k = len(vals) - 1
            rr, dE = vals[k], ih[k][0]
        else:
            u = r["unconverged"]
            k, rr, dE = u["iteration"], u["residual"], u["dE"]
        res.append(rr)
        rows.append("| %.3f | %s | %d | %d | %.2e | %s | %.2e |" % (r["T_over_rs"], "met" if r.get("converged") else "not met", r["iterations"], k, rr,
                                                                 "%.1e" % dE if dE is not None else "none", vals[-1]))
    n = len(res)
    L = [math.log10(x) for x in res]
    xm = (n - 1) / 2.0
    ym = sum(L) / n
    sl = sum((i - xm) * (L[i] - ym) for i in range(n)) / sum((i - xm) ** 2 for i in range(n))
    s = sl * (n - 1)
    d = [res[i + 1] - res[i] for i in range(n - 1)]
    c = sum(1 for i in range(len(d) - 1) if d[i] * d[i + 1] < 0)
    q = max(res) / min(res)
    if s > math.log10(2):
        verdict = "growing"
    elif s < -math.log10(2):
        verdict = "falling"
    elif c > (n - 2) / 2.0 and q > 2:
        verdict = "oscillating"
    else:
        verdict = "stationary"
    srt = sorted(res)
    med = srt[n // 2] if n % 2 else 0.5 * (srt[n // 2 - 1] + srt[n // 2])
    out = ["## The self-consistent iteration's residual over the first %d records: %s (engine_residual_history_v1.py)" % (n, a.label), "",
           "- run: %s (script %s; stop |dE/E| < %s and fields < %s)" % (a.run_json, J.get("script"), J["args"].get("stop_E"), J["args"].get("stop_fields")),
           "", "| T/rs | stop rule | iterations | returned iterate | its residual | its abs dE/E | the last iterate's residual |", "|---|---|---|---|---|---|---|"] + rows + [
           "", "- the returned residuals: min %.2e, median %.2e, max %.2e (max/min %.2f); fitted change over the %d records %+.3f decades; %d sign changes of successive differences over %d interior records" % (
               min(res), med, max(res), q, n, s, c, n - 2),
           "- reading (the rule in the script's docstring, builder's choice): %s" % verdict, "- every number above is printed by the script."]
    txt = "\n".join(out)
    print(txt)
    if a.md:
        open(a.md, "w").write(txt + "\n")


if __name__ == "__main__":
    main()
