#!/usr/bin/env python3
"""obs_slice_jets_v3.py (builder analyst; B13 of SPEC_final4b_observables.md sec. 12.6: version 3 of obs_slice_jets_v2.py).

Version 3: beside version 2's polynomial fits through the site values, the constraint-differentiated route: on a slice that carries the shell
totals (engine_collapse_v6's export) the dense functions are re-solved (engine_fields_v5.resolve_dense, the run's collar policy), and the
Taylor series of W and u = 1/N about the surface R_t are integrated from the constraints themselves,
    W' = 8 pi g r^2 rho_ell(r)/u^2,      u' = 4 pi g r J_ell(r)/v,      v = -sqrt(W/r)      (the engine's dictionary, engine 12.2),
with W(R_t) and u(R_t) from the re-solved dense functions and the sources' Taylor coefficients from the direct kernel sums (exact values on a
stencil of 17 points at spacing ell/8 about R_t, a degree-8 polynomial fit); the series arithmetic (products, quotients, square roots of
truncated series) gives the coefficients of v, N, v/N and N + v to the fourth order, hence kappa = (N + v)' and P25's jets A, B, C of v and of
v/N, and P25's coefficients at them. Printed per slice: both routes side by side, their differences, the stencil fit's spread (degrees 7 and 9)
as the source jets' error, and max |N - 1| over the window. On a slice without totals (an engine_collapse_v5 record) only version 2's route
runs. Args: --fields --records --slices --g --window --deg --fits --collar-policy --control-core --md --out-json."""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "engine", "code"))
sys.path.insert(0, os.path.join(HERE, "..", "engine"))
import obs_plateau_jets_v1 as JT  # noqa: E402
import obs_slice_jets_v2 as SJ2  # noqa: E402

K = 6                                                                      # series length: orders 0 .. 5


def smul(a, b):
    return np.array([sum(a[i] * b[k - i] for i in range(k + 1)) for k in range(K)])


def sdiv(a, b):
    c = np.zeros(K)
    for k in range(K):
        c[k] = (a[k] - sum(c[i] * b[k - i] for i in range(k))) / b[0]
    return c


def ssqrt(a):
    c = np.zeros(K)
    c[0] = math.sqrt(a[0])
    for k in range(1, K):
        c[k] = (a[k] - sum(c[i] * c[k - i] for i in range(1, k))) / (2 * c[0])
    return c


def series_from_values(F, R0, h, deg):
    """the Taylor coefficients (orders 0 .. 5) of F about R0 from a polynomial fit of degree deg through F at R0 + j h, j = -8 .. 8."""
    x = np.arange(-8, 9) * h
    y = np.array([F(R0 + xi) for xi in x])
    c = np.polynomial.polynomial.polyfit(x, y, deg)
    out = np.zeros(K)
    out[:min(K, len(c))] = c[:K]
    return out


def constraint_jets(info, g, Rt, ell):
    """the constraint-differentiated Taylor series about R_t; returns the series of v, N, v/N, N + v and the source jets' spread."""
    rho_dir = lambda r: float(info["src_direct"](np.array([r]))[0][0])
    J_dir = lambda r: float(info["src_direct"](np.array([r]))[1][0])
    h = ell / 8.0
    rho_s = series_from_values(rho_dir, Rt, h, 8); J_s = series_from_values(J_dir, Rt, h, 8)
    rho_s7 = series_from_values(rho_dir, Rt, h, 7); rho_s9 = series_from_values(rho_dir, Rt, h, 9)
    spread = float(np.max(np.abs(rho_s9 - rho_s7)[:5]) / max(np.max(np.abs(rho_s[:5])), 1e-300))
    r_s = np.zeros(K); r_s[0] = Rt; r_s[1] = 1.0
    W = np.zeros(K); u = np.zeros(K)
    W[0] = float(info["W_field"](Rt)[0]); u[0] = 1.0 / float(info["N_dense"](Rt)[0])
    for k in range(K - 1):
        v = -ssqrt(sdiv(W, r_s))
        rhsW = 8.0 * math.pi * g * sdiv(smul(smul(r_s, r_s), rho_s), smul(u, u))
        rhsu = 4.0 * math.pi * g * sdiv(smul(r_s, J_s), v)
        W[k + 1] = rhsW[k] / (k + 1)
        u[k + 1] = rhsu[k] / (k + 1)
    v = -ssqrt(sdiv(W, r_s))
    one = np.zeros(K); one[0] = 1.0
    N = sdiv(one, u)
    return {"v": v, "N": N, "v_over_N": smul(v, u), "f": N + v, "source_spread": spread}


def deriv_from_series(c):
    return [math.factorial(n) * c[n] for n in range(1, 5)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--slices", type=int, nargs="+", default=[-1])
    ap.add_argument("--g", type=float, default=0.25)
    ap.add_argument("--window", type=float, default=16.0)
    ap.add_argument("--deg", type=int, default=8)
    ap.add_argument("--fits", default="16:8,16:9,24:9,24:10,32:10")
    ap.add_argument("--collar-policy", default="flag")
    ap.add_argument("--md", required=True)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    import engine_fields_v5 as F5
    Z = np.load(a.fields, allow_pickle=True)
    D = json.load(open(a.records))
    recs = [r for r in D["records"] if "E" in r]
    Rs = np.asarray(Z["R"], float)
    has_tot = "eps_tot" in Z.files
    ell = float(Z["ell"])
    rows = []
    L = ["## The surface's four derivatives by two routes: polynomial fits through the site values (v2) and the constraints differentiated (v3) (obs_slice_jets_v3.py; %s)" % os.path.basename(a.fields), "",
         "| slice | T/r_s | R_t | kappa 4 g E (fits; constraints) | A, B, C of v by fits (spread) | A, B, C of v by the constraints (source spread) | A, B, C of v/N by the constraints | max |N - 1| | c_0, c_2, c_m at v's constraint jets | at v/N's |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for si in a.slices:
        idx = si if si >= 0 else len(Z["T"]) + si
        T = float(Z["T"][idx])
        rec = min(recs, key=lambda r: abs(r["T"] - T))
        E = float(rec["E"])
        N = np.asarray(Z["N"][idx], float); v = np.asarray(Z["v"][idx], float)
        # route 1: version 2's fits through the site values
        import obs_plateau_v5 as P5
        sN = P5.site_spline(Rs, N, +1, 5); sv = P5.site_spline(Rs, v, -1, 5)
        f = N + v
        z = np.nonzero(f[:-1] * f[1:] <= 0)[0]
        if not len(z):
            L.append("| %d | %.3f | no zero of N + v | | | | | | | |" % (idx, T / (2 * a.g * E)))
            continue
        from scipy.optimize import brentq
        i = z[-1]
        Rt = brentq(lambda x: float(sN(x) + sv(x)), Rs[i], Rs[i + 1], xtol=1e-13)
        fits = []
        for W_, deg in [(float(p.split(":")[0]), int(p.split(":")[1])) for p in a.fits.split(",")]:
            dv, _ = SJ2.derivs(Rs, v, Rt, W_, deg); dN, _ = SJ2.derivs(Rs, N, Rt, W_, deg)
            kap = dv[1] + dN[1]
            fits.append({"W": W_, "deg": deg, "kappa": kap, "jets_v": SJ2.jets_from(dv, kap)})
        base = [q for q in fits if q["W"] == a.window and q["deg"] == a.deg][0]
        spread = [float(np.ptp([q["jets_v"][j] for q in fits])) for j in range(3)]
        selw = np.abs(Rs - Rt) <= a.window
        row = {"slice": idx, "T": T, "E": E, "R_t": Rt, "kappa_fits": base["kappa"], "jets_v_fits": base["jets_v"], "jets_v_fits_spread": spread,
               "max_N_minus_1": float(np.max(np.abs(N[selw] - 1.0)))}
        # route 2: the constraints differentiated, on the re-solved dense functions
        if has_tot:
            eps_t = np.asarray(Z["eps_tot"][idx], float); J_t = np.asarray(Z["J_tot"][idx], float)
            Nr, vr, Wr, info = F5.resolve_dense(eps_t, J_t, a.g, Rs, int(ell), N0=N, collar_policy=a.collar_policy)
            if info.get("failed"):
                row["constraint_route"] = {"failed": info["failed"]}
            else:
                src = F5.SmoothedSources5(eps_t, J_t, Rs, ell, info["r_nodes"][-1])
                info["src_direct"] = src.direct
                cj = constraint_jets(info, a.g, Rt, ell)
                kap = float(cj["f"][1])
                dvs = deriv_from_series(cj["v"]); dqs = deriv_from_series(cj["v_over_N"])
                jv = [dvs[1] / (2 * kap ** 2), dvs[2] / (6 * kap ** 3), dvs[3] / (24 * kap ** 4)]
                jq = [dqs[1] / (2 * kap ** 2), dqs[2] / (6 * kap ** 3), dqs[3] / (24 * kap ** 4)]
                row["constraint_route"] = {"kappa": kap, "jets_v": jv, "jets_v_over_N": jq, "source_spread": cj["source_spread"],
                                           "c_v": JT.p25_coeffs(*jv), "c_v_over_N": JT.p25_coeffs(*jq), "resolve_vs_stored_max_dv": float(np.max(np.abs(vr - v)))}
        rows.append(row)
        cr = row.get("constraint_route", {})
        fmt3 = lambda x: ", ".join("%.6f" % q for q in x) if x else "-"
        L.append("| %d | %.3f | %.4f | %.6f; %s | %s (%s) | %s (%s) | %s | %.2e | %s | %s |" % (
            idx, T / (2 * a.g * E), Rt, base["kappa"] * 4 * a.g * E, ("%.6f" % (cr["kappa"] * 4 * a.g * E)) if cr.get("kappa") else "-",
            fmt3(base["jets_v"]), ", ".join("%.1e" % q for q in spread), fmt3(cr.get("jets_v")), ("%.1e" % cr["source_spread"]) if cr.get("source_spread") is not None else "-",
            fmt3(cr.get("jets_v_over_N")), row["max_N_minus_1"], fmt3(cr.get("c_v")), fmt3(cr.get("c_v_over_N"))))
        json.dump({"script": "obs_slice_jets_v3.py", "args": vars(a), "rows": rows}, open(a.out_json, "w"), indent=1, default=float)
    L += ["", "- every number above is printed by the script; the constraint route needs the slice's shell totals (engine_collapse_v6's export);",
          "  the exact profile at the solved E has A, B, C = -1.5, 2.5, -4.375, P25's c = -0.625, -1, 0.75."]
    json.dump({"script": "obs_slice_jets_v3.py", "args": vars(a), "rows": rows}, open(a.out_json, "w"), indent=1, default=float)
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
