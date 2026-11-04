#!/usr/bin/env python3
"""obs_plateau_report_v3.py (builder analyst; version 3: table (4), the greybody Richardson column the lead asked for (observables sec. 10.1): per
continuation and R_lead, the (kappa a)^2 extrapolation of Gamma from the two smallest kappa a, against P25. Version 2: table (2) takes, per (r, mu), the largest R_lead at which that point exists; version 1
took the class maximum, which only the greybody points reach, so most of its rows were empty).  Version 1: the J-PL tables from obs_plateau_v2's records, printed by the script (no hand-copied numbers).

Per geometry class of profile pgs: (1) per size kappa a, the least-squares fit of c = (1 - theta)/(kappa a)^2 over the (r, mu) grid to
c_0 + c_2 r^2 + c_m mu^2 with its errors, against P25's Painleve-Gullstrand (-5/8, -1, 3/4); (2) per (r, mu), c at every size beside P25's
-(5/8 + r^2 - 3 mu^2/4); (3) the greybody Gamma at (mu, r) = (0.2, 0.25) and (0.4, 0.6) against R_lead at each size, beside P25's 0.88523533
and 0.98975910, with the last-two-entries difference (the spec's 1e-4 rule).  Args: json records...; --md out."""
import argparse
import json
import math

import numpy as np

P25_GB = {(0.25, 0.2): 0.88523533, (0.6, 0.4): 0.98975910}


def p25_c(r, mu):
    return -(5.0 / 8.0 + r * r - 0.75 * mu * mu)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records", nargs="+")
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    rows = []
    for f in a.records:
        rows += json.load(open(f))["rows"]
    rows = [rd for rd in rows if rd.get("geometry", {}).get("R1") is not None]
    cls = lambda rd: (rd["geometry"]["R1"] / rd["geometry"]["r_s"], rd["geometry"]["w"], rd["geometry"].get("outer_w"))
    L = ["## J-PL tables (obs_plateau_report_v3.py over %s)" % ", ".join(a.records), ""]
    L += ["### (1) Per size: the fit c = c_0 + c_2 r^2 + c_m mu^2 (P25, Painleve-Gullstrand: c_0 = -0.625, c_2 = -1, c_m = 0.75)", "",
          "| continuation R_1/r_s, w | R_lead/r_s | kappa a | kappa 2 r_s | c_0 | c_2 | c_m | rms residual | points |", "|---|---|---|---|---|---|---|---|---|"]
    for c in sorted({cls(rd) for rd in rows}, key=lambda t: (t[0], t[1], str(t[2]))):
        for RL in sorted({rd["geometry"]["R_lead_nominal"] for rd in rows if cls(rd) == c}):
            for ka in sorted({rd["kappa_a"] for rd in rows if cls(rd) == c}, reverse=True):
                sel = [rd for rd in rows if cls(rd) == c and rd["geometry"]["R_lead_nominal"] == RL and rd["kappa_a"] == ka and rd.get("theta") is not None]
                if len(sel) < 4:
                    continue
                A = np.array([[1.0, rd["r"] ** 2, rd["mu"] ** 2] for rd in sel])
                b = np.array([(1.0 - rd["theta"]) / ka ** 2 for rd in sel])
                coef = np.linalg.lstsq(A, b, rcond=None)[0]
                res = b - A @ coef
                dof = max(len(sel) - 3, 1)
                s2 = float(res @ res) / dof
                err = np.sqrt(np.maximum(np.diag(s2 * np.linalg.pinv(A.T @ A)), 0))
                L.append("| %g, %g%s | %g | %g | %.9f | %.4f +- %.4f | %.4f +- %.4f | %.4f +- %.4f | %.1e | %d |" % (
                    c[0], c[1], (", outer %g" % c[2]) if c[2] else "", RL, ka, sel[0]["geometry"]["kappa_2rs"], coef[0], err[0], coef[1], err[1], coef[2], err[2],
                    math.sqrt(s2), len(sel)))
    L += ["", "### (2) c per (r, mu) and size, beside P25's -(5/8 + r^2 - 3 mu^2/4) (R_lead = the largest at which the point exists)", "",
          "| continuation | r | mu | R_lead/r_s | P25 | c per kappa a |", "|---|---|---|---|---|---|"]
    for c in sorted({cls(rd) for rd in rows}, key=lambda t: (t[0], t[1], str(t[2]))):
        for (r, mu) in sorted({(rd["r"], rd["mu"]) for rd in rows if cls(rd) == c}):
            RL = max(rd["geometry"]["R_lead_nominal"] for rd in rows if cls(rd) == c and rd["r"] == r and rd["mu"] == mu)
            cs = []
            for ka in sorted({rd["kappa_a"] for rd in rows if cls(rd) == c}, reverse=True):
                th = [rd["theta"] for rd in rows if cls(rd) == c and rd["geometry"]["R_lead_nominal"] == RL and rd["kappa_a"] == ka and rd["r"] == r
                      and rd["mu"] == mu and rd.get("theta") is not None]
                if th:
                    cs.append("%g: %.5f" % (ka, (1.0 - th[0]) / ka ** 2))
            L.append("| %g, %g%s | %g | %g | %g | %.5f | %s |" % (c[0], c[1], (", outer %g" % c[2]) if c[2] else "", r, mu, RL, p25_c(r, mu), ", ".join(cs)))
    L += ["", "### (3) Greybody Gamma against R_lead (P25's continuum values: 0.88523533 at (mu, r) = (0.2, 0.25), 0.98975910 at (0.4, 0.6))", "",
          "| continuation | kappa a | (r, mu) | Gamma per R_lead/r_s | last two differ by | last minus P25 |", "|---|---|---|---|---|---|"]
    for c in sorted({cls(rd) for rd in rows}, key=lambda t: (t[0], t[1], str(t[2]))):
        for ka in sorted({rd["kappa_a"] for rd in rows if cls(rd) == c}, reverse=True):
            for key, ref in P25_GB.items():
                G = sorted((rd["geometry"]["R_lead_nominal"], rd["Gamma"]) for rd in rows if cls(rd) == c and rd["kappa_a"] == ka
                           and (rd["r"], rd["mu"]) == key and rd.get("Gamma") is not None)
                if len(G) < 2:
                    continue
                L.append("| %g, %g%s | %g | (%g, %g) | %s | %.2e | %.2e |" % (c[0], c[1], (", outer %g" % c[2]) if c[2] else "", ka, key[0], key[1],
                         ", ".join("%g: %.9f" % (RL, g) for RL, g in G), G[-1][1] - G[-2][1], G[-1][1] - ref))
    L += ["", "### (4) Greybody: the (kappa a)^2 Richardson value per R_lead from the two smallest kappa a, against P25", "",
          "| continuation | (r, mu) | R_lead/r_s | kappa a pair | Gamma at the pair | Richardson | minus P25 |", "|---|---|---|---|---|---|---|"]
    for c in sorted({cls(rd) for rd in rows}, key=lambda t: (t[0], t[1], str(t[2]))):
        for key, ref in P25_GB.items():
            for RL in sorted({rd["geometry"]["R_lead_nominal"] for rd in rows if cls(rd) == c and (rd["r"], rd["mu"]) == key}):
                G = {rd["kappa_a"]: rd["Gamma"] for rd in rows if cls(rd) == c and (rd["r"], rd["mu"]) == key and rd["geometry"]["R_lead_nominal"] == RL
                     and rd.get("Gamma") is not None}
                if len(G) < 2:
                    continue
                k1, k2 = sorted(G)[:2]
                ext = (G[k1] * k2 ** 2 - G[k2] * k1 ** 2) / (k2 ** 2 - k1 ** 2)
                L.append("| %g, %g%s | (%g, %g) | %g | %g, %g | %.9f, %.9f | %.9f | %.2e |" % (c[0], c[1], (", outer %g" % c[2]) if c[2] else "", key[0], key[1], RL,
                         k1, k2, G[k1], G[k2], ext, ext - ref))
    L += ["", "- every number above is printed by this script from the records named in the title."]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
