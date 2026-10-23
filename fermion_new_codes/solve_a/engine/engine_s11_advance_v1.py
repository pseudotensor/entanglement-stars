#!/usr/bin/env python
"""engine_s11_advance_v1.py (builder analyst; S11 of REVIEWA_REVIEW_web40e, engine spec 28.5: "Confirm the [001] advance coefficient -(am)^2
and the [011] law -3/4 + cos 2 varpi/16 against -(4/3) B_L (am)^2, B_L = (3/8)(1 + sum_i Lhat_i^4); retire the record's -1/2 and -1/3. No
new runs beyond the existing v9 orbits"): on the stored task-7 orbits only. (1) The review's formula per orbit-plane normal Lhat: sum_i
Lhat_i^4, B_L and -(4/3) B_L for [001], [011], [111]. (2) The [011] periastron (engine_ppn_v10's stored orbits: m a 0.1, 0.05, 0.025; A/r_g
160, 640, 2560; theta0 -pi/4, 0, pi/8, pi/4; both senses): per (m a, theta0) the 7-period coefficient (advance lattice/geodesic - 1)/(m a)^2
per sense, their mean and half-difference, per sense the extrapolation in r_g/A (c0 + c1 r_g/A through the two largest A, and c0 + c1 r_g/A +
c2 (r_g/A)^2 through all three, both printed), and beside them the review's law -3/4 + cos(2 varpi)/16 at varpi = theta0 (the stored orbit's
periastron angle in the plane from e1 = (1, 0, 0); the identification of the review's varpi with theta0 is the builder's and is labelled)
and the record's own T7g column. (3) The [111] periastron (engine_ppn_v8's stored ladder): (ratio - 1)/(m a)^2 per (A, e, m a) against
-(4/3) B_L at [111] and the red team's first post-Newtonian value stored beside it. (4) [001]: the stored orbits hold none; the formula's
value is printed as the comparison for a run the review does not ask for. Every number printed by the script from the stored jsons.
Args: --v10 --v8 --md --out-json."""
import argparse
import json
import math

import numpy as np


def B_L(n):
    n = np.asarray(n, float); n = n / np.linalg.norm(n)
    s4 = float(np.sum(n ** 4))
    return s4, 0.375 * (1.0 + s4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v10", required=True); ap.add_argument("--v8", required=True); ap.add_argument("--md", required=True); ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    out = {"script": "engine_s11_advance_v1.py", "formula": {}, "orbits_011": [], "precession_111": []}
    L = ["## S11: the periastron advance coefficients against -(4/3) B_L (engine_s11_advance_v1.py; the stored task-7 orbits only)", "",
         "### (1) The review's formula per orbit-plane normal", "", "| normal | sum Lhat_i^4 | B_L = (3/8)(1 + sum Lhat_i^4) | -(4/3) B_L |", "|---|---|---|---|"]
    for lab, n in (("[001]", (0, 0, 1)), ("[011]", (0, 1, 1)), ("[111]", (1, 1, 1))):
        s4, b = B_L(n)
        out["formula"][lab] = {"sum_L4": s4, "B_L": b, "coefficient": -4.0 * b / 3.0}
        L.append("| %s | %.6f | %.6f | %.6f |" % (lab, s4, b, -4.0 * b / 3.0))
    J = json.load(open(a.v10)); orb = J["orbits_011"]
    L += ["", "### (2) The [011] periastron (engine_ppn_v10's stored orbits; e = 0.3; the coefficient over 7 periods)", "",
          "| m a | theta0 | sense | 160 r_g | 640 r_g | 2560 r_g | c0 (two largest A) | c0 (three A, quadratic) | T7g (sense +1) | -3/4 + cos(2 theta0)/16 |", "|---|---|---|---|---|---|---|---|---|---|"]
    key = {}
    for o in orb:
        w = o["coefficient_windows"]
        c7 = w.get("7", w.get(7)) if isinstance(w, dict) else None
        if c7 is None:
            c7 = list(w.values())[-1]
        key.setdefault((o["ma"], round(o["theta0"], 6), o["sense"]), {})[o["A_over_rg"]] = (float(c7), o.get("C011_T7g_sense_plus"))
    rows = []
    for (ma, th, se), d in sorted(key.items()):
        As = sorted(d)
        c = [d[x][0] for x in As]; xs = [1.0 / x for x in As]
        c0_2 = c0_3 = None
        if len(As) >= 2:
            x1, x2 = xs[-2], xs[-1]; y1, y2 = c[-2], c[-1]
            c0_2 = y2 - (y2 - y1) / (x2 - x1) * x2
        if len(As) >= 3:
            c0_3 = float(np.polyfit(xs, c, 2)[-1])
        law = -0.75 + math.cos(2.0 * th) / 16.0
        r = {"ma": ma, "theta0": th, "sense": se, "A": As, "c7": c, "c0_two": c0_2, "c0_three": c0_3, "T7g": d[As[0]][1], "law": law}
        rows.append(r)
        L.append("| %g | %+.4f | %+d | %s | %s | %s | %s | %s | %s | %.6f |" % (ma, th, se, *[("%.5f" % d[x][0]) if x in d else "-" for x in (160.0, 640.0, 2560.0)],
                 ("%.5f" % c0_2) if c0_2 is not None else "-", ("%.5f" % c0_3) if c0_3 is not None else "-", ("%.6f" % d[As[0]][1]) if d[As[0]][1] is not None else "-", law))
    out["orbits_011"] = rows
    L += ["", "| m a | theta0 | sense mean of c0 (three A) | half the sense difference | the law at varpi = theta0 | mean minus law |", "|---|---|---|---|---|---|"]
    byp = {}
    for r in rows:
        byp.setdefault((r["ma"], r["theta0"]), {})[r["sense"]] = r
    for (ma, th), d in sorted(byp.items()):
        if 1 in d and -1 in d and d[1]["c0_three"] is not None and d[-1]["c0_three"] is not None:
            m_ = 0.5 * (d[1]["c0_three"] + d[-1]["c0_three"]); h_ = 0.5 * (d[1]["c0_three"] - d[-1]["c0_three"])
            L.append("| %g | %+.4f | %.5f | %+.5f | %.6f | %+.5f |" % (ma, th, m_, h_, d[1]["law"], m_ - d[1]["law"]))
    K = json.load(open(a.v8))["precession"]
    c111 = out["formula"]["[111]"]["coefficient"]
    L += ["", "### (3) The [111] periastron (engine_ppn_v8's stored ladder)", "", "| A/r_g | e | m a | (ratio - 1)/(m a)^2 | -(4/3) B_L at [111] | the red team's first post-Newtonian value (stored) |", "|---|---|---|---|---|---|"]
    for p in K:
        co = p["ratio_minus_1"] / p["m"] ** 2
        out["precession_111"].append({"A": p["a_over_rg"], "e": p["e"], "ma": p["m"], "coefficient": co, "redteam_first_PN": p.get("redteam_first_PN")})
        L.append("| %g | %g | %g | %.5f | %.6f | %s |" % (p["a_over_rg"], p["e"], p["m"], co, c111, ("%.5f" % p["redteam_first_PN"]) if p.get("redteam_first_PN") is not None else "-"))
    L += ["", "### (4) [001]", "", "- the stored orbits hold no [001] periastron; the formula's value there is %.6f (the review's -(am)^2); no run is made (S11: no new runs)." % out["formula"]["[001]"]["coefficient"], "",
          "- varpi = theta0 is the builder's identification (the stored orbit's periastron angle in the plane from e1 = (1, 0, 0), e2 = (0, 1, -1)/sqrt 2); the extrapolations in r_g/A are the builder's (two- and three-point); every number above is printed by the script from engine_ppn_v10's and engine_ppn_v8's stored jsons; the retirement of the record's coefficients is the lead's."]
    open(a.md, "w").write("\n".join(L) + "\n")
    json.dump(out, open(a.out_json, "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
