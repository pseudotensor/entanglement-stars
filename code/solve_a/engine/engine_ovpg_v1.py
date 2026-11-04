#!/usr/bin/env python
"""engine_ovpg_v1.py (lead's draft; SPEC_final4b_engine.md 4.5 C1.2): the continuum Oppenheimer-Volkoff family with the lattice equation
of state of the sine fermion (strand5_eos_v4 tables at the engine's m a), in flat-slice form: N = N_s/sqrt(F), v = -N sqrt(2 g m(r)/r),
E_F = mu_inf.  Areal coordinates: m' = 4 pi r^2 e, P' = -(e + P) g (m + 4 pi r^3 P)/(r (r - 2 g m)), Phi' = g (m + 4 pi r^3 P)/(r(r - 2gm)),
with the EOS by interpolation in mu (the Tolman condition mu N_s = E_F).  Lattice units; g = G/a^2.  Output: M(R_*), the maximum, the
compactness, N(r), v(r) sampled at R_n = n + 1 for the lattice solver's start and comparison."""
import argparse
import json
import os

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


def load_eos(path, mhat):
    t = np.load(path)
    sel = np.abs(t["m_hat"] - mhat) < 1e-12
    if not sel.any():
        raise ValueError("BUILDER: run strand5_eos_v4.py at m_hat = %g (minutes)" % mhat)
    mu, e, P, rho = (t[k][sel] for k in ("mu_hat", "e_hat", "P_hat", "rho"))
    o = np.argsort(mu)
    f = lambda y: interp1d(mu[o], y[o], kind="cubic", bounds_error=False, fill_value=(0.0, y[o][-1]))
    return f(e), f(P), f(rho)


def shoot(EF, m, g, e_of_mu, P_of_mu, n_of_mu, mu_c, rmax=1e6):
    """integrate from the centre at central chemical potential mu_c; the surface where mu reaches m; returns the profile and the charges."""
    def rhs(r, y):
        mm, Phi = y
        mu = EF / np.exp(Phi)
        e, P = (float(e_of_mu(mu)), float(P_of_mu(mu))) if mu > m else (0.0, 0.0)
        return [4 * np.pi * r * r * e, g * (mm + 4 * np.pi * r ** 3 * P) / (r * (r - 2 * g * mm))]

    def surface(r, y):
        return EF / np.exp(y[1]) - m
    surface.terminal = True
    Phi0 = np.log(EF / mu_c)
    r0 = 1e-6
    sol = solve_ivp(rhs, (r0, rmax), [0.0, Phi0], events=surface, rtol=1e-11, atol=1e-14, dense_output=True)
    R_star = float(sol.t_events[0][0]); M = float(sol.y_events[0][0][0]); Phi_s = float(sol.y_events[0][0][1])
    # the Schwarzschild lapse normalization: N_s(R_*) = sqrt(1 - 2gM/R_*) = exp(Phi_s + c)
    c = 0.5 * np.log(1 - 2 * g * M / R_star) - Phi_s
    return {"R_star": R_star, "M": M, "EF_consistent": EF * np.exp(c), "sol": sol, "c": c}


def profile_on_lattice(res, g, R):
    sol = res["sol"]
    mm = np.array([sol.sol(min(r, res["R_star"]))[0] if r <= res["R_star"] else res["M"] for r in R])
    Phi = np.array([sol.sol(min(r, res["R_star"]))[1] + res["c"] if r <= res["R_star"] else 0.5 * np.log(1 - 2 * g * res["M"] / r) for r in R])
    F = 1 - 2 * g * mm / R
    N = np.exp(Phi) / np.sqrt(F)
    v = -N * np.sqrt(2 * g * mm / R)
    return N, v, mm


# BUILDER (completed): the family driver.  For mu_c from m sqrt(1 + z_c^2) (z_c = k_F/m at the centre) one shot with EF = m fixes the
# profile (the equation is homogeneous in the normalization of exp(Phi)); EF_consistent = m e^c is the Fermi level at infinity.  N_B =
# int 4 pi r^2 n(mu)/sqrt(F) dr (the proper volume).  R_999: the radius where m(r) N(r)^2 = 0.999 M, the lattice code's r_star definition
# (its cumulative sum_{k<n} N_k^2 eps_k equals r v^2/(2g) = m(r) N(r)^2); R_surf: mu = m.  The maximum of M by a parabola in z_c.
def member(m, g, e_of, P_of, n_of, z_c, nsamp=4001):
    mu_c = m * np.sqrt(1 + z_c * z_c)
    res = shoot(m, m, g, e_of, P_of, n_of, mu_c)
    R = res["R_star"]
    rr = np.linspace(1e-6, R, nsamp)
    Y = res["sol"].sol(rr)
    mm, Phi = Y[0], Y[1] + res["c"]
    F = 1 - 2 * g * mm / rr
    mu = res["EF_consistent"] / np.exp(Phi)
    n = np.array([float(n_of(x)) if x > m else 0.0 for x in mu])
    NB = float(np.trapz(4 * np.pi * rr ** 2 * n / np.sqrt(F), rr))
    N = np.exp(Phi) / np.sqrt(F)
    cum = mm * N ** 2
    R999 = float(rr[np.searchsorted(cum, 0.999 * res["M"])]) if cum[-1] >= 0.999 * res["M"] else R
    return {"z_c": z_c, "mu_c": float(mu_c), "M": res["M"], "R_surf": R, "R_999": R999, "compactness_surf": 2 * g * res["M"] / R,
            "compactness_999": 2 * g * res["M"] / R999, "EF": float(res["EF_consistent"]), "EF_over_m": float(res["EF_consistent"] / m),
            "N_B": NB, "binding": res["M"] - NB * m, "N_centre": float(N[0]), "min_N_minus_absv": float(np.min(N - N * np.sqrt(2 * g * mm / rr)))}


def family(eos, m, g, zcs):
    e_of, P_of, n_of = load_eos(eos, m)
    rows = []
    for z in zcs:
        try:
            rows.append(member(m, g, e_of, P_of, n_of, float(z)))
        except Exception as ex:                                            # a shot that fails is recorded, not skipped silently
            rows.append({"z_c": float(z), "error": str(ex)})
    ok = [r for r in rows if "M" in r]
    i = int(np.argmax([r["M"] for r in ok]))
    mx = None
    if 0 < i < len(ok) - 1:
        x = np.array([ok[i - 1]["z_c"], ok[i]["z_c"], ok[i + 1]["z_c"]]); y = np.array([ok[i - 1]["M"], ok[i]["M"], ok[i + 1]["M"]])
        c2, c1, c0 = np.polyfit(x, y, 2)
        zm = -c1 / (2 * c2)
        mx = {"z_c": float(zm), "M_max": float(c0 + c1 * zm + c2 * zm * zm), "member_nearest": ok[i]}
    return {"m": m, "g": g, "rows": rows, "maximum": mx, "M_max_scaled": (mx["M_max"] * g ** 1.5 * m ** 2) if mx else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eos", required=True, help="strand5_eos_v4's eos_v4_tables.npz")
    ap.add_argument("--grid", default="0.2:0.25,0.2:0.0625,0.1:0.25,0.1:0.0625,0.05:0.25,0.1:0.015625")
    ap.add_argument("--zcs", default=",".join("%.4g" % x for x in np.geomspace(0.05, 3.0, 48)))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    zcs = [float(x) for x in a.zcs.split(",")]
    res = {"script": "engine_ovpg_v1.py", "args": vars(a), "families": []}
    for spec in a.grid.split(","):
        m, g = (float(x) for x in spec.split(":"))
        fam = family(a.eos, m, g, zcs)
        res["families"].append(fam)
        mx = fam["maximum"]
        print("m %g g %g: M_max %s at z_c %s (EF/m %s, R_999 %s, C %s); M_max g^1.5 m^2 %s" % (
            m, g, mx and "%.6g" % mx["M_max"], mx and "%.4g" % mx["z_c"], mx and "%.5f" % mx["member_nearest"]["EF_over_m"],
            mx and "%.4g" % mx["member_nearest"]["R_999"], mx and "%.4f" % mx["member_nearest"]["compactness_999"], fam["M_max_scaled"]), flush=True)
    json.dump(res, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
