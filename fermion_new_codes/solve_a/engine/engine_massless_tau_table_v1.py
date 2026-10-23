#!/usr/bin/env python
"""engine_massless_tau_table_v1.py (builder analyst; engine spec 24.4: "One control, the finding's label: the clean member's first
re-solve on v23 at tau 2, 1, 0.5; the pairs' energy and |min U|/U_ref falling as tau (a rate: the sea's own response, the model's) or as
tau^2 (the sudden switch of the quasi-static step)"): reads the three records of the massless shell's tau control (engine_collapse_v23 on
the diagnostic fields v15) and prints, per tau, the first re-solve's time, the pairs' shell energy, E/E0 - 1, min U/U_ref on the continued
interval, the interval, the iteration's residual, the step-failure zeros and stretches of the diagnostic fields, and between neighbouring
taus the observed exponent log2 of each ratio. Args: --runs (comma list of tau:json) --md."""
import argparse
import json
import math


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    rows = []
    for item in a.runs.split(","):
        t, path = item.split(":", 1)
        d = json.load(open(path))
        r = d["records"][-1]
        g = r["gauge"]
        res = None
        if r.get("iteration_history"):
            res = r["iteration_history"][-1][1]
        rows.append({"tau": float(t), "T_over_rs": r["T_over_rs"], "pairs": r["E_pairs_shells"], "dE": r["E_over_E0"] - 1.0,
                     "minU": g.get("min_U_over_U_ref"), "intervals": g.get("intervals"), "res": res, "unconv": r.get("unconverged"),
                     "zeros": g.get("step_failure_zeros"), "stretches": g.get("t_stretches"), "dL": g.get("delta_L_bound"),
                     "it": r.get("iterations"), "H_in": r.get("H_in")})
    rows.sort(key=lambda x: -x["tau"])
    L = ["## The massless shell's tau control: the clean member's first re-solve (engine_massless_tau_table_v1.py)", "",
         "engine_collapse_v23 on the diagnostic fields v15; r_s 16, R_0 6 r_s, (k_0, w) (0.45, 18) as 1.125 r_s; every number is read from the records.", "",
         "| tau | T/rs | pairs' shell energy | E/E0 - 1 | min U/U_ref | continued interval | residual (last iteration) | iterations | step-failure zeros | stretches | delta L bound |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in rows:
        L.append("| %g | %.5f | %.4e | %.4e | %.4e | %s | %.3e | %s | %s | %s | %.3e |" % (
            x["tau"], x["T_over_rs"], x["pairs"], x["dE"], x["minU"], x["intervals"], x["res"] if x["res"] is not None else float("nan"),
            x["it"], x["zeros"], x["stretches"], x["dL"] if x["dL"] is not None else float("nan")))
    L += ["", "Observed exponents p (quantity proportional to tau^p) between neighbouring taus, p = log2 of the ratio:", "",
          "| taus | pairs' shell energy | E/E0 - 1 | min U/U_ref |", "|---|---|---|---|"]
    for i in range(len(rows) - 1):
        u, w = rows[i], rows[i + 1]
        lr = math.log(u["tau"] / w["tau"])

        def p(key):
            if u[key] is None or w[key] is None or u[key] == 0.0 or w[key] == 0.0 or (u[key] > 0) != (w[key] > 0):
                return "n/a"
            return "%.3f" % (math.log(u[key] / w[key]) / lr)
        L.append("| %g / %g | %s | %s | %s |" % (u["tau"], w["tau"], p("pairs"), p("dE"), p("minU")))
    L += ["", "- The first re-solve is at T = tau, so a quantity produced at a fixed rate reads p = 1 and one produced by the step's sudden switch reads p = 2.",
          "- Each record's fields carry the diagnostic fields v15's step-failure zeros and stretches as listed (the rule engine_fields_v16 removed, engine 24.1 (1))."]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
