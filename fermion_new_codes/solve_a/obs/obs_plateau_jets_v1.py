#!/usr/bin/env python3
"""obs_plateau_jets_v1.py (builder analyst; after the J-PL red team A1, reviewb redteam_JPL3a of 2026-10-15, defects 1, 5 and 6): for every
analytic-continuation geometry (profile pgs) in obs_plateau_v2 records: (a) the surface x_h (rho(x_h) = r_s) and kappa = v'(x_h) analytically
(rho by the stable form of the outer saturation, rho_out = rho_in - w_L log(1 + e^((rho_in - R_lead)/w_L)), its derivative a logistic factor;
the records' kappa came from a central difference of the unstable form with h = 1e-4, red team A1's defect 5); (b) the profile's actual
Taylor coefficients at the surface in P25's normalization v = -1 + z + A z^2 + B z^3 + C z^4, z = kappa (x - x_h) (mpmath, 40 digits), and
P25's coefficients at those jets, c_0 = A^2 - (3/2) B - (7/2) A^3 + 7 A B - (7/2) C, c_2 = 4 A^2 - 2 B + 10 A^3 - 10 A B + 2 C,
c_m = (3/2)(2 A^2 + A - B) (the Painleve-Gullstrand values (-3/2, 5/2, -35/8) give (-5/8, -1, 3/4)); (c) every row's theta renormalized by the
analytic kappa (theta' = theta kappa_record/kappa_analytic: T_conv comes from the scattering and does not depend on kappa), and
c' = (1 - theta')/(kappa_analytic a)^2 with the measured kappa a (defect 5's denominator); (d) the rows deduplicated by the physical key
(R_1/r_s, w, outer_w, R_lead/r_s, kappa a nominal, r, mu), the duplicates' theta spread printed (defect 6); (e) per (continuation, R_lead,
size): c' against P25 at the actual jets and at the Painleve-Gullstrand jets, and per continuation the per-size fit of c' - c_P25(actual jets)
and its two-size Richardson value. Args: records... --md --out-json."""
import argparse
import json
import math

import mpmath as mp
import numpy as np

mp.mp.dps = 40


def p25_coeffs(A, B, C):
    c0 = A * A - 1.5 * B - 3.5 * A ** 3 + 7 * A * B - 3.5 * C
    c2 = 4 * A * A - 2 * B + 10 * A ** 3 - 10 * A * B + 2 * C
    cm = 1.5 * (2 * A * A + A - B)
    return c0, c2, cm


def geometry_jets(rs, R1, w, outer_w, R_lead):
    rs, R1, w = mp.mpf(rs), mp.mpf(R1), mp.mpf(w)
    wL = mp.mpf(outer_w) * mp.mpf(R_lead) if outer_w else None
    RL = mp.mpf(R_lead)

    def rho(x):
        r_ = R1 + w * mp.log1p(mp.exp((x - R1) / w))
        if wL:
            r_ = r_ - wL * mp.log1p(mp.exp((r_ - RL) / wL))
        return r_
    v = lambda x: -mp.sqrt(rs / rho(x))
    xh = mp.findroot(lambda x: rho(x) - rs, rs)
    d = [mp.diff(v, xh, n) for n in range(1, 5)]
    kap = d[0]
    A = d[1] / (2 * kap ** 2); B = d[2] / (6 * kap ** 3); C = d[3] / (24 * kap ** 4)
    return float(xh), float(kap), float(A), float(B), float(C)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records", nargs="+")
    ap.add_argument("--md", required=True)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    rows = []
    for f in a.records:
        for rd in json.load(open(f))["rows"]:
            if rd.get("geometry", {}).get("R1") is not None and rd.get("theta") is not None:
                rd["_file"] = f.split("/")[-1]
                rows.append(rd)
    key = lambda rd: (round(rd["geometry"]["R1"] / rd["geometry"]["r_s"], 9), rd["geometry"]["w"], rd["geometry"].get("outer_w"),
                      rd["geometry"]["R_lead_nominal"], rd["kappa_a"], rd["r"], rd["mu"])
    groups = {}
    for rd in rows:
        groups.setdefault(key(rd), []).append(rd)
    dups = [(k, [g["theta"] for g in v]) for k, v in groups.items() if len(v) > 1]
    uniq = [v[0] for v in groups.values()]
    gcache = {}
    out = []
    for rd in uniq:
        g = rd["geometry"]
        gk = (g["r_s"], g["R1"], g["w"], g.get("outer_w"), g["R_lead"])
        if gk not in gcache:
            gcache[gk] = geometry_jets(*gk)
        xh, kap, A, B, C = gcache[gk]
        c0, c2, cm = p25_coeffs(A, B, C)
        th2 = rd["theta"] * g["kappa"] / kap
        cmeas = (1.0 - th2) / kap ** 2
        cjet = c0 + c2 * rd["r"] ** 2 + cm * rd["mu"] ** 2
        cpg = -(5.0 / 8 + rd["r"] ** 2 - 0.75 * rd["mu"] ** 2)
        out.append({"cont": [g["R1"] / g["r_s"], g["w"], g.get("outer_w")], "R_lead": g["R_lead_nominal"], "kappa_a_nominal": rd["kappa_a"], "kappa_analytic": kap,
                    "kappa_record_over_analytic_minus_1": g["kappa"] / kap - 1, "x_h": xh, "jets": [A, B, C], "c_p25_jets": [c0, c2, cm], "r": rd["r"], "mu": rd["mu"],
                    "theta_record": rd["theta"], "theta_renorm": th2, "c_renorm": cmeas, "c_p25_actual": cjet, "c_p25_pg": cpg, "c_record_nominal": (1 - rd["theta"]) / rd["kappa_a"] ** 2})
    L = ["## J-PL against P25 at the actual horizon jets (obs_plateau_jets_v1.py over %s)" % ", ".join(f.split("/")[-1] for f in a.records), "",
         "### (a, b) Per geometry: the analytic surface, kappa against the record's, the jets and P25's coefficients at them (Painleve-Gullstrand: A, B, C = -1.5, 2.5, -4.375; c = -0.625, -1, 0.75)", "",
         "| R_1/r_s, w, outer_w | R_lead/r_s | r_s | kappa 2 r_s (analytic) | kappa_record/kappa_analytic - 1 | A | B | C | c_0 | c_2 | c_m |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for gk, (xh, kap, A, B, C) in sorted(gcache.items(), key=lambda t: (t[0][1] / t[0][0], t[0][2], str(t[0][3]), t[0][4] / t[0][0], t[0][0])):
        rs, R1, w, ow, RL = gk
        rec = [o for o in out if abs(o["kappa_analytic"] - kap) < 1e-15 and abs(o["x_h"] - xh) < 1e-9]
        kr = rec[0]["kappa_record_over_analytic_minus_1"] if rec else float("nan")
        c0, c2, cm = p25_coeffs(A, B, C)
        L.append("| %g, %g, %s | %g | %g | %.12f | %.2e | %.9f | %.9f | %.9f | %.6f | %.6f | %.6f |" % (R1 / rs, w, ow, RL / rs, rs, kap * 2 * rs, kr, A, B, C, c0, c2, cm))
    L += ["", "### (c) Per row: c from the record (nominal kappa a), renormalized c' (analytic kappa), P25 at the actual jets and at Painleve-Gullstrand", "",
          "| R_1/r_s, w, outer_w | R_lead/r_s | kappa a | r | mu | c (record) | c' (renormalized) | P25 actual jets | P25 PG | c' - P25 actual |", "|---|---|---|---|---|---|---|---|---|---|"]
    for o in sorted(out, key=lambda o: (o["cont"][0], o["cont"][1], str(o["cont"][2]), o["R_lead"], -o["kappa_a_nominal"], o["mu"], o["r"])):
        L.append("| %g, %g, %s | %g | %g | %g | %g | %.6f | %.6f | %.6f | %.6f | %+.6f |" % (o["cont"][0], o["cont"][1], o["cont"][2], o["R_lead"], o["kappa_a_nominal"], o["r"], o["mu"],
                 o["c_record_nominal"], o["c_renorm"], o["c_p25_actual"], o["c_p25_pg"], o["c_renorm"] - o["c_p25_actual"]))
    L += ["", "### (e) Per continuation and R_lead: the two-size Richardson value of c' - c_P25(actual jets) per (r, mu) (the lattice's (kappa a)^4 term removed)", "",
          "| R_1/r_s, w, outer_w | R_lead/r_s | r | mu | sizes (kappa a) | c' - P25 actual per size | Richardson |", "|---|---|---|---|---|---|---|"]
    rich_rows = []
    cl = sorted({(tuple(o["cont"]), o["R_lead"]) for o in out}, key=lambda t: (t[0][0], t[0][1], str(t[0][2]), t[1]))
    for (cont, RL) in cl:
        for (r, mu) in sorted({(o["r"], o["mu"]) for o in out if tuple(o["cont"]) == cont and o["R_lead"] == RL}):
            sel = sorted([o for o in out if tuple(o["cont"]) == cont and o["R_lead"] == RL and o["r"] == r and o["mu"] == mu], key=lambda o: -o["kappa_analytic"])
            if not sel:
                continue
            dv = [(o["kappa_analytic"], o["c_renorm"] - o["c_p25_actual"]) for o in sel]
            rich = None
            if len(dv) >= 2:
                (k1, d1), (k2, d2) = dv[-2], dv[-1]
                rich = (d2 * k1 ** 2 - d1 * k2 ** 2) / (k1 ** 2 - k2 ** 2)
            rich_rows.append({"cont": cont, "R_lead": RL, "r": r, "mu": mu, "per_size": dv, "richardson": rich})
            L.append("| %g, %g, %s | %g | %g | %g | %s | %s | %s |" % (cont[0], cont[1], cont[2], RL, r, mu, ", ".join("%.6g" % k for k, _ in dv),
                     ", ".join("%+.6f" % d for _, d in dv), ("%+.6f" % rich) if rich is not None else "-"))
    L += ["", "### (d) Duplicated physical keys (the report's concatenation; the theta values of each duplicate)", ""]
    if dups:
        for k, ths in dups:
            L.append("- %s: %s (spread %.2e)" % (k, ", ".join("%.12f" % t for t in ths), max(ths) - min(ths)))
    else:
        L.append("- none")
    L += ["", "- every number above is printed by the script; the renormalization changes only kappa (T_conv is the scattering's), so no scattering is re-run."]
    json.dump({"script": "obs_plateau_jets_v1.py", "records": a.records, "rows": out, "richardson": rich_rows, "duplicates": [[list(k), t] for k, t in dups]},
              open(a.out_json, "w"), indent=1, default=float)
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L[:40]))


if __name__ == "__main__":
    main()
