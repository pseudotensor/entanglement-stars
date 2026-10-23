#!/usr/bin/env python3
"""obs_interior_monitors_v1.py (lead's skeleton; SPEC_final4b_observables.md sec. 5; J10 on task 2's continued collapse, both versions).

Per step of the engine's record (BUILDER: the engine's file format; the arrays r, rho, N, v, J_r, m_enc on the radial grid, the version
label): the central density against rho_max = 2/a^3, the lapse's minimum and positivity, v against -sqrt(2 G m(r)/r) (the core's uniform
expansion rate: v/r constant inside the saturated region), the bond flux against 1/a^4, the core radius and charge and their growth, and the
trust map eps_lat = max(a|d_r v|/N, a|d_r N|/N, a|d_r rho|/rho) with the untrusted band eps_lat > 0.3.  Units a = 1; G/a^2 as the run's."""
import numpy as np

RHO_MAX = 2.0   # the positive band full: 2 particles per site above the sea
J_MAX = 1.0     # 2 particles per site at the two-step current's unit slope 1/(2a)


def two_point_derivative(r, y):
    d = np.gradient(y, r)
    return d


def monitors(step):
    """step: dict with r, rho, N, v, J, m_enc, G, t, version."""
    r, rho, N, v, J, m = (np.asarray(step[k], float) for k in ("r", "rho", "N", "v", "J", "m_enc"))
    G = float(step["G"])
    with np.errstate(divide="ignore", invalid="ignore"):
        v_pred = -np.sqrt(2 * G * m / np.maximum(r, 1e-300))
        eps = np.maximum.reduce([np.abs(two_point_derivative(r, v)) / np.maximum(N, 1e-300), np.abs(two_point_derivative(r, N)) / np.maximum(N, 1e-300),
                                 np.abs(two_point_derivative(r, rho)) / np.maximum(np.abs(rho), 1e-300)])
    core = rho >= 0.99 * RHO_MAX
    out = {"t": step["t"], "version": step["version"], "rho_centre_over_max": float(rho[0] / RHO_MAX), "N_min": float(N.min()),
           "N_positive": bool((N > 0).all()), "v_over_pred_max_dev": float(np.nanmax(np.abs(v / np.where(v_pred != 0, v_pred, np.nan) - 1))),
           "v_over_r_core_spread": float(np.ptp((v / r)[core])) if core.sum() > 2 else None, "J_over_max": float(np.max(np.abs(J)) / J_MAX),
           "R_core": float(r[core].max()) if core.any() else 0.0, "M_core": float(m[core].max()) if core.any() else 0.0,
           "untrusted_band": [float(r[eps > 0.3].min()), float(r[eps > 0.3].max())] if (eps > 0.3).any() else None,
           "eps_lat_max": float(np.nanmax(eps))}
    return out


def growth(records):
    """R_core(t), M_core(t) and the accretion rate dM_core/dt against the matter flux at R_core (BUILDER: the engine's flux)."""
    t = np.array([x["t"] for x in records]); R = np.array([x["R_core"] for x in records]); M = np.array([x["M_core"] for x in records])
    return {"t": t.tolist(), "R_core": R.tolist(), "M_core": M.tolist(), "dM_dt": np.gradient(M, t).tolist() if len(t) > 2 else None}
