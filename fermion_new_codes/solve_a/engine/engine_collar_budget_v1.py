#!/usr/bin/env python
"""engine_collar_budget_v1.py (builder analyst; engine spec 28.2, S1's collar budget, the part that stored data decide: "Collar budget per
interval: M_R = r w^2/(2G) at the stored Killing cut's radius R, N_R Delta M_R, the cut current times Delta T as F_R, the remainder printed as
the measured D_R (1.26's missing term) per interval and per tau beside the reset term"). The review's collar identity
N_R M_dot_R = -F_R - Tr(H_R P_dot) gives, per interval, D_R Delta T := -N_R Delta M_R - F_R Delta T. Here: R the radius of the record's
stored cut site (M1_parts["cut"], the engine's cut index round(r_collar) - 1; R = R_sites[cut]); M_R = R w^2/(2 g), w = v/N, from the
record's solved site fields (the data npz's N and v at that site); N_R the interval's mean of the two records' N at R; F_R Delta T the
engine's stored trapezoid of the collar's Killing cut current over the interval (M1_balance["evolution_trapezoid"], the record's M1);
beside them: dE (the stored charge E), the Gauss balance's reset term -sum_n (<N^2>_n mean) Delta eps_R2,n (label C's reset, from the data
npz's N2avg and eps_R2), the M1 balance's reference_reset (the inside Killing energy of R2, stored), and the cut-change flag. The terms that
need the reference's eigenvectors (Tr(h_k Delta P_k), Tr(h_R Delta P_k), the first-adiabatic residual) are engine_replay_v4's. Per interval
rows every --every records; cumulative sums at --cum-times (T/rs); every number printed by the script. CPU, seconds. Args: --data --json
(the run json(s), comma list, records looked up by T) --md --out-json --every --cum-times."""
import argparse
import json
import math

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--json", required=True); ap.add_argument("--md", required=True); ap.add_argument("--out-json", required=True)
    ap.add_argument("--every", type=int, default=8); ap.add_argument("--cum-times", default="5,10,12,14,16,18,20,22,25,30,35,40")
    a = ap.parse_args()
    Z = np.load(a.data, allow_pickle=True)
    T = np.asarray(Z["T"], float); R = np.asarray(Z["R"], float); g = float(Z["g"]); rs = float(Z["rs"])
    recs = {}
    for f in a.json.split(","):
        for r in json.load(open(f))["records"]:
            if r.get("E") is not None:
                recs.setdefault(round(float(r["T"]), 6), r)
    rows = []
    for k in range(1, len(T)):
        r0 = recs.get(round(T[k - 1], 6)); r1 = recs.get(round(T[k], 6))
        if r0 is None or r1 is None:
            rows.append({"T_over_rs": T[k] / rs, "missing": True}); continue
        cut0 = int(r0["M1_parts"]["cut"]); cut1 = int(r1["M1_parts"]["cut"])
        Mof = lambda kk, c: R[c] * (float(Z["v"][kk][c]) / float(Z["N"][kk][c])) ** 2 / (2.0 * g)
        M0, M1 = Mof(k - 1, cut1), Mof(k, cut1)
        NR = 0.5 * (float(Z["N"][k - 1][cut1]) + float(Z["N"][k][cut1]))
        bal = r1.get("M1_balance") or {}
        FR = float(bal.get("evolution_trapezoid", float("nan")))
        DR = -NR * (M1 - M0) - FR
        N2m = 0.5 * (np.asarray(Z["N2avg"][k - 1], float) + np.asarray(Z["N2avg"][k], float))
        reset_gauss = -float(np.sum(N2m * (np.asarray(Z["eps_R2"][k], float) - np.asarray(Z["eps_R2"][k - 1], float))))
        rows.append({"T_over_rs": T[k] / rs, "dT": T[k] - T[k - 1], "cut": cut1, "R": float(R[cut1]), "cut_changed": cut0 != cut1, "N_R": NR, "dM_R": M1 - M0,
                     "N_R_dM_R": NR * (M1 - M0), "F_R_dT": FR, "D_R_dT": DR, "dE": float(r1["E"] - r0["E"]), "reset_gauss": reset_gauss,
                     "reset_M1": float(bal.get("reference_reset", float("nan"))), "M_R": M1, "E": float(r1["E"])})
    ok = [r for r in rows if not r.get("missing")]
    cum_t = [float(x) for x in a.cum_times.split(",")]
    L = ["## The collar budget per interval from the stored records (engine_collar_budget_v1.py; engine spec 28.2, S1): D_R Delta T = -N_R Delta M_R - F_R Delta T", "",
         "- data %s (r_s %g, g %g; %d records); json %s; R the stored cut site's radius (the engine's cut index; %s); M_R = R w^2/(2g) from the record's site fields; N_R the interval mean; F_R Delta T the stored trapezoid of the collar's Killing cut current (M1); the Gauss reset -sum (<N^2> mean) Delta eps_R2 (label C's); the M1 reference reset (stored)." % (
             a.data, rs, g, len(T), a.json, "cut changes on %d intervals" % sum(1 for r in ok if r["cut_changed"])), "",
         "### Cumulative sums from T = 0", "",
         "| T/rs | sum dE | sum N_R dM_R | sum F_R dT | sum D_R dT | sum Gauss reset | sum M1 reset | E | M_R at the cut |", "|---|---|---|---|---|---|---|---|---|"]
    acc = {q: 0.0 for q in ("dE", "N_R_dM_R", "F_R_dT", "D_R_dT", "reset_gauss", "reset_M1")}
    ci = 0; cum_rows = []
    for r in ok:
        for q in acc:
            acc[q] += r[q] if r[q] == r[q] else 0.0
        while ci < len(cum_t) and r["T_over_rs"] >= cum_t[ci] - 1e-9:
            if abs(r["T_over_rs"] - cum_t[ci]) < 1e-6:
                cum_rows.append(dict(acc, T_over_rs=r["T_over_rs"], E=r["E"], M_R=r["M_R"]))
                L.append("| %.3f | %+.6e | %+.6e | %+.6e | %+.6e | %+.6e | %+.6e | %.6f | %.6f |" % (r["T_over_rs"], acc["dE"], acc["N_R_dM_R"], acc["F_R_dT"], acc["D_R_dT"],
                                                                                   acc["reset_gauss"], acc["reset_M1"], r["E"], r["M_R"]))
            ci += 1
    L += ["", "### Per interval (every %d-th interval)" % a.every, "",
          "| T/rs | R | N_R | dM_R | N_R dM_R | F_R dT | D_R dT | dE | Gauss reset | M1 reset |", "|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(ok):
        if i % a.every == a.every - 1 or i == len(ok) - 1:
            L.append("| %.3f | %.1f | %.6f | %+.4e | %+.4e | %+.4e | %+.4e | %+.4e | %+.4e | %+.4e |" % (r["T_over_rs"], r["R"], r["N_R"], r["dM_R"], r["N_R_dM_R"], r["F_R_dT"],
                                                                                    r["D_R_dT"], r["dE"], r["reset_gauss"], r["reset_M1"]))
    L += ["", "- every number above is printed by the script; D_R is the remainder of the stored terms, its identification with Tr(H_R P_dot) the review's 1.26 (engine_replay_v4 computes that trace and the first-adiabatic residual)."]
    open(a.md, "w").write("\n".join(L) + "\n")
    json.dump({"script": "engine_collar_budget_v1.py", "data": a.data, "rows": rows, "cumulative": cum_rows}, open(a.out_json, "w"), indent=1, default=float)
    print("\n".join(L[:20]))


if __name__ == "__main__":
    main()
