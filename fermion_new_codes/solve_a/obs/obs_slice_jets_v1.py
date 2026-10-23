#!/usr/bin/env python3
"""obs_slice_jets_v1.py (builder analyst; after the J-PL red team part B round 4, reviewb redteam_JPL5 of 2026-10-15, defect 2: "B1 stops at the
third derivative, whereas P25 requires the fourth"): the surface's four derivatives on a stored slice of the collapse (engine_collapse_v5's
fields npz, the dense site values) and P25's coefficients at them, the measured Taylor data of the fallback statement.

Per slice: the trapping surface R_t (the outermost zero of N + v, by the degree-5 parity spline through the site values, obs_plateau_v5's
route); least-squares polynomials of degree --deg in x = (R - R_t)/W through the site values of v and of N within |R - R_t| <= W
(W = --window), their derivatives at R_t to the fourth order; kappa = d(N + v)/dR at R_t; P25's normalization z = kappa (R - R_t),
v = -1 + z + A z^2 + B z^3 + C z^4 (A = v''/(2 kappa^2), B = v'''/(6 kappa^3), C = v''''/(24 kappa^4)), computed (a) for v (P25's unit lapse)
and (b) for v/N (the lapse folded in), with max |N - 1| over the window printed; P25's coefficients c_0 = A^2 - (3/2) B - (7/2) A^3 + 7 A B -
(7/2) C, c_2 = 4 A^2 - 2 B + 10 A^3 - 10 A B + 2 C, c_m = (3/2)(2 A^2 + A - B) at each; against the exact profile's at the solved E
(A, B, C) = (-3/2, 5/2, -35/8), kappa = 1/(4 g E). The error: the spread over the windows (W, 1.5 W, 2 W) and degrees (deg - 1, deg, deg + 1),
printed. --control-core R_c compares the jets with the smooth-core function -sqrt(2 g E) r (r^2 + R_c^2)^(-3/4) evaluated analytically
(mpmath, 40 digits; the fixture's profile). Args: --fields --records --slices (indices; default the last) --g --window --deg --control-core
--md --out-json."""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import obs_plateau_jets_v1 as JT  # noqa: E402
import obs_plateau_v5 as P5  # noqa: E402


def derivs(R, y, R0, W, deg):
    sel = np.abs(R - R0) <= W
    x = (R[sel] - R0) / W
    c = np.polynomial.polynomial.polyfit(x, y[sel], deg)
    return [math.factorial(n) * c[n] / W ** n for n in range(5)], int(sel.sum())


def jets_from(d, kap):
    return d[2] / (2 * kap ** 2), d[3] / (6 * kap ** 3), d[4] / (24 * kap ** 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--slices", type=int, nargs="+", default=[-1])
    ap.add_argument("--g", type=float, default=0.25)
    ap.add_argument("--window", type=float, default=16.0)
    ap.add_argument("--deg", type=int, default=8)
    ap.add_argument("--control-core", type=float, default=None)
    ap.add_argument("--md", required=True)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    Z = np.load(a.fields, allow_pickle=True)
    recs = [r for r in json.load(open(a.records))["records"] if "E" in r]
    Rs = np.asarray(Z["R"], float)
    out = []
    L = ["## The surface's four derivatives on stored slices and P25's coefficients at them (obs_slice_jets_v1.py; %s)" % os.path.basename(a.fields), "",
         "| slice | T/r_s | E | R_t | kappa 4 g E | A, B, C of v (spread) | A, B, C of v/N | max |N - 1| in the window | c_0, c_2, c_m at v's jets | at v/N's jets | Painleve-Gullstrand |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for si in a.slices:
        idx = si if si >= 0 else len(Z["T"]) + si
        T = float(Z["T"][idx])
        rec = min(recs, key=lambda r: abs(r["T"] - T))
        E = float(rec["E"])
        N = np.asarray(Z["N"][idx], float); v = np.asarray(Z["v"][idx], float)
        sN = P5.site_spline(Rs, N, +1, 5); sv = P5.site_spline(Rs, v, -1, 5)
        f = N + v
        z = np.nonzero(f[:-1] * f[1:] <= 0)[0]
        if not len(z):
            L.append("| %d | %.3f | %.6f | no zero of N + v | | | | | | | |" % (idx, T / (2 * a.g * E), E))
            continue
        i = z[-1]
        from scipy.optimize import brentq
        Rt = brentq(lambda x: float(sN(x) + sv(x)), Rs[i], Rs[i + 1], xtol=1e-13)
        res = []
        for W in (a.window, 1.5 * a.window, 2.0 * a.window):
            for deg in (a.deg - 1, a.deg, a.deg + 1):
                dv, n_ = derivs(Rs, v, Rt, W, deg); dN, _ = derivs(Rs, N, Rt, W, deg)
                kap = dv[1] + dN[1]
                Av = jets_from(dv, kap)
                # v/N by the product rule on the fitted polynomials
                q = v / N
                dq, _ = derivs(Rs, q, Rt, W, deg)
                Aq = jets_from(dq, kap)
                res.append({"W": W, "deg": deg, "n": n_, "kappa": kap, "jets_v": Av, "jets_vN": Aq})
        base = [r_ for r_ in res if r_["W"] == a.window and r_["deg"] == a.deg][0]
        spread = [float(np.ptp([r_["jets_v"][j] for r_ in res])) for j in range(3)]
        spread_q = [float(np.ptp([r_["jets_vN"][j] for r_ in res])) for j in range(3)]
        selw = np.abs(Rs - Rt) <= a.window
        maxN1 = float(np.max(np.abs(N[selw] - 1.0)))
        cv = JT.p25_coeffs(*base["jets_v"]); cq = JT.p25_coeffs(*base["jets_vN"])
        row = {"slice": idx, "T": T, "T_over_rs_geo": T / (2 * a.g * E), "E": E, "R_t": Rt, "R_t_record": rec.get("R_t"), "kappa": base["kappa"], "kappa_4gE": base["kappa"] * 4 * a.g * E,
               "kappa_record": rec.get("kappa_T"), "jets_v": base["jets_v"], "jets_v_spread": spread, "jets_vN": base["jets_vN"], "jets_vN_spread": spread_q,
               "max_N_minus_1": maxN1, "c_v": cv, "c_vN": cq, "fits": res}
        if a.control_core:
            import mpmath as mp
            mp.mp.dps = 40
            rsg, Rc = mp.mpf(2 * a.g * E), mp.mpf(a.control_core)
            vf = lambda r: -mp.sqrt(rsg) * r * (r * r + Rc * Rc) ** mp.mpf(-0.75)
            xh = mp.findroot(lambda r: 1 + vf(r), Rt)
            dd = [mp.diff(vf, xh, n) for n in range(1, 5)]
            k_ = dd[0]
            row["control"] = {"R_t": float(xh), "kappa_4gE": float(k_ * 4 * a.g * E), "jets": [float(dd[1] / (2 * k_ ** 2)), float(dd[2] / (6 * k_ ** 3)), float(dd[3] / (24 * k_ ** 4))]}
        out.append(row)
        L.append("| %d | %.3f | %.6f | %.4f | %.6f | %.6f, %.6f, %.6f (%.1e, %.1e, %.1e) | %.6f, %.6f, %.6f | %.2e | %.5f, %.5f, %.5f | %.5f, %.5f, %.5f | -0.625, -1, 0.75 |" % (
            idx, row["T_over_rs_geo"], E, Rt, row["kappa_4gE"], *base["jets_v"], *spread, *base["jets_vN"], maxN1, *cv, *cq))
        if "control" in row:
            c = row["control"]
            L.append("| control | | | %.4f | %.6f | %.6f, %.6f, %.6f (the smooth-core function, R_c = %g, analytic) | | | %.5f, %.5f, %.5f | | |" % (
                c["R_t"], c["kappa_4gE"], *c["jets"], a.control_core, *JT.p25_coeffs(*c["jets"])))
        json.dump({"script": "obs_slice_jets_v1.py", "args": vars(a), "rows": out}, open(a.out_json, "w"), indent=1, default=float)
    L += ["", "- every number above is printed by the script; the spread is the range over windows (W, 1.5 W, 2 W) and degrees (deg - 1 to deg + 1);",
          "  the exact profile at the solved E has A, B, C = -1.5, 2.5, -4.375 and kappa 4 g E = 1."]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
