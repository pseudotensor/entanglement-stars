#!/usr/bin/env python
"""engine_ovpg_v2.py (builder analyst; engine spec 22.3 (1)): the continuum Oppenheimer-Volkoff-PG family of engine_ovpg_v1 with a
thermodynamically consistent equation of state: version 1 interpolated e(mu), P(mu) and n(mu) from the strand5_eos_v4 tables by three
independent cubic interpolants, whose mu dP/dmu/(e + P) - 1 reaches 1e-3 to 8e-2 near mu = m (engine_rows_v2's validation), so the
profiles do not satisfy hydrostatic equilibrium there; here P(mu) is the table's cubic spline (scipy CubicSpline on the tabulated mu, P at
the run's m a), n = dP/dmu its derivative and e = mu n - P, every quantity zero at and below mu = m. Everything else is version 1's
(shoot, member, family; imported). continuum_member_at_NB mirrors engine_star_v6's with this equation of state; family_maximum scans z_c
and fits the parabola as version 1's family does. Functions: load_eos_consistent, member, family, continuum_member_at_NB."""
import numpy as np
from scipy.interpolate import CubicSpline

import engine_ovpg_v1 as OV

shoot = OV.shoot


def load_eos_consistent(path, mhat):
    t = np.load(path)
    sel = np.abs(t["m_hat"] - mhat) < 1e-12
    if not sel.any():
        raise ValueError("no table at m_hat = %g" % mhat)
    mu, P = t["mu_hat"][sel], t["P_hat"][sel]
    o = np.argsort(mu)
    Ps = CubicSpline(mu[o], P[o]); dPs = Ps.derivative()
    m = float(mhat)
    P_of = lambda u: float(Ps(u)) if u > m else 0.0
    n_of = lambda u: float(dPs(u)) if u > m else 0.0
    e_of = lambda u: (u * float(dPs(u)) - float(Ps(u))) if u > m else 0.0
    return e_of, P_of, n_of


def member(m, g, e_of, P_of, n_of, z_c):
    return OV.member(m, g, e_of, P_of, n_of, z_c)


def family(eos, m, g, zcs):
    e_of, P_of, n_of = load_eos_consistent(eos, m)
    rows = []
    for z in zcs:
        try:
            rows.append(member(m, g, e_of, P_of, n_of, float(z)))
        except Exception as ex:
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


def continuum_member_at_NB(eos, m, g, NB, z_lo=0.05, z_hi=None):
    """engine_star_v6.continuum_member_at_NB with the consistent equation of state: the stable-branch member with the given N_B by bisection on z_c."""
    e_of, P_of, n_of = load_eos_consistent(eos, m)

    def nb(z):
        try:
            return member(m, g, e_of, P_of, n_of, z)["N_B"]
        except (IndexError, ValueError):
            return np.nan
    lo, hi = z_lo, z_hi
    while not np.isfinite(nb(lo)) and lo < hi:
        lo *= 1.5
    if not (nb(lo) <= NB <= nb(hi)):
        return None, {"note": "N_B outside the continuum stable branch", "NB_range": [nb(lo), nb(hi)]}
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if nb(mid) < NB else (lo, mid)
    z = 0.5 * (lo + hi)
    return z, member(m, g, e_of, P_of, n_of, z)
