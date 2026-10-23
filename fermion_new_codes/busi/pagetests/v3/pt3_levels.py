#!/usr/bin/env python
"""pt3_levels.py (busi, PAGE design v3 secs. 3 and 4; A5, A6, F4): the supplying levels tracked from the start as coupled resonances, the H_2
crossing parameters, and the three endpoints.

At each clock the window eigenpairs of the actual operator h(t) (pt3_core.window_spectrum_banded on long chains, dense on fixtures) and their
interior weights w_in(j) = <j|Q|j> (Q the sites inside a radius R_in, between the inner turning point and the root) give the resonance by a
Lorentzian fit of the weight distribution on the dilute ladder: w_in(e_j) = A (Gamma/2)/pi / ((e_j - e_c)^2 + (Gamma/2)^2) * Delta_box (sum rule
sum_j w_in = A), Delta_box the local spacing; V = (Gamma Delta_box/2 pi)^(1/2) (the golden rule). Where Delta_box > ~ Gamma the fit degenerates
to the single most-interior eigenvector (Gamma then from the hybridization of its neighbours) and is labelled so. The port weight is the
resonance's weight beyond the root (the kernel-weighted outside weight of the top interior eigenvector), its occupation <c|C|c> from the shared C.
Trajectory e_c(t); s_j = |de_c/dt - de_j/dt| per crossed exterior level, lambda_j = 2 pi |V_j|^2/s_j; the declared tolerances on Delta_box/Gamma,
Delta_box T_obs and Gamma Delta_box/s_j are checked by regime_checks (C2, C5 options decide which side is required).
Endpoints (A6): E1 the first clock the level is inside the declared thermal band (0, T_H) (or (-T_H, 0) for a pocket), E2 the clocks it traverses
an exterior level (an avoided crossing: the ladder level's energy passed), E3 the first zero crossing of e_c; each None when not reached, and a
null before E2 or E3 is inconclusive (A6).
wkb_level_count: the static exterior's level count n(E) on the chain symbol (N = 1, v = 0) by the standing-wave rule
n(E) = (1/pi) sum_valleys int k(E, R) dR, the known answer of the level-spacing unit test."""
import math

import numpy as np
from scipy.optimize import least_squares


def lorentz_fit(e, w, guess_c=None):
    e = np.asarray(e, float); w = np.asarray(w, float)
    if len(e) < 3:
        return None
    o = np.argsort(e); e, w = e[o], w[o]
    dloc = np.gradient(e)
    dens = w / dloc                                    # weight per unit energy
    c0 = float(e[np.argmax(w)]) if guess_c is None else guess_c
    A0 = float(w.sum())
    half = dens.max() / 2
    above = e[dens > half]
    G0 = max(float(above.max() - above.min()) if len(above) > 1 else float(np.median(dloc)), 1e-12)

    def res(p):
        A, c, G = p
        return A * (G / 2) / math.pi / ((e - c) ** 2 + (G / 2) ** 2) * dloc - w
    r = least_squares(res, [A0, c0, G0], bounds=([0, e.min(), 1e-14], [10 * A0 + 1, e.max(), 10 * (e.max() - e.min())]))
    A, c, G = r.x
    D = float(np.median(dloc))
    return {"A": float(A), "e_c": float(c), "Gamma": float(G), "Delta_box": D, "V": math.sqrt(G * D / (2 * math.pi)), "rms": float(np.sqrt(np.mean(r.fun ** 2))),
            "Delta_over_Gamma": D / G, "regime": "dilute (Delta_box > Gamma)" if D > G else "dense (Delta_box < Gamma)"}


def resonance_at(e, U, qmask, lo, hi, C=None, outmask=None):
    """the resonance in (lo, hi) at one clock: the fit, the top interior eigenvector, its occupation and port weight."""
    m = (e > lo) & (e < hi)
    if not m.any():
        return None
    ee, UU = e[m], U[:, m]
    w = np.real((np.abs(UU[qmask, :]) ** 2).sum(axis=0))
    fit = lorentz_fit(ee, w) if m.sum() >= 3 else None
    j = int(np.argmax(w))
    out = {"e_top": float(ee[j]), "w_top": float(w[j]), "n_in_window": int(m.sum()), "fit": fit, "w_sum": float(w.sum())}
    if C is not None:
        u = UU[:, j]
        out["occ_top"] = float(np.real(np.vdot(u, C @ u)))
    if outmask is not None:
        out["port_weight_top"] = float(np.real((np.abs(UU[outmask, j]) ** 2).sum()))
    return out


def endpoints(t, e_c, T_H, band_sign=+1, ladder=None):
    """E1, E2 (list), E3 from a trajectory e_c(t); T_H scalar or per clock; ladder: list per clock of exterior level energies (for E2)."""
    t = np.asarray(t, float); e = np.asarray(e_c, float)
    TH = np.broadcast_to(np.asarray(T_H, float), t.shape)
    inband = (e > 0) & (e < TH) if band_sign > 0 else (e < 0) & (e > -TH)
    E1 = float(t[np.argmax(inband)]) if inband.any() else None
    sg = np.sign(e)
    zc = np.where(sg[1:] != sg[:-1])[0]
    E3 = None
    if len(zc):
        i = zc[0]
        E3 = float(t[i] - e[i] * (t[i + 1] - t[i]) / (e[i + 1] - e[i]))
    E2 = []
    if ladder is not None:
        for i in range(len(t) - 1):
            a, b = e[i], e[i + 1]
            lo, hi = min(a, b), max(a, b)
            L = np.asarray(ladder[i], float)
            for x in L[(L > lo) & (L <= hi)]:
                E2.append(float(t[i] + (x - a) * (t[i + 1] - t[i]) / (b - a)))
    return {"E1_entry_thermal_band": E1, "E2_crossings": E2, "n_E2": len(E2), "E3_zero_crossing": E3}


def crossing_params(t, e_c, e_j, V, i_cross):
    """s_j = |de_c/dt - de_j/dt| at the crossing interval i_cross (finite differences), lambda_j = 2 pi V^2/s_j."""
    dt = t[i_cross + 1] - t[i_cross]
    s = abs((e_c[i_cross + 1] - e_c[i_cross]) / dt - (e_j[i_cross + 1] - e_j[i_cross]) / dt)
    return {"s_j": float(s), "V_j": float(V), "lambda_j": float(2 * math.pi * V * V / s) if s > 0 else math.inf}


def regime_checks(Delta_box, Gamma, T_obs, s_j, tol=0.1):
    """the declared tolerances (A5): Delta_box/Gamma, Delta_box T_obs, Gamma Delta_box/s_j for the open-continuum swept arm (all << 1, i.e. below
    tol) and their dilute complements for the eigenstate reference (Gamma/Delta_box below tol, F/pc9 1.3 (vi)); both printed, the C2 option picks."""
    a = Delta_box / Gamma; b = Delta_box * T_obs; c = Gamma * Delta_box / s_j if s_j else math.inf
    return {"Delta_over_Gamma": a, "Delta_T_obs": b, "Gamma_Delta_over_s": c, "open_continuum_ok": bool(a < tol and b < tol and c < tol),
            "Gamma_over_Delta": 1.0 / a, "dilute_reference_ok": bool(1.0 / a < tol)}


def wkb_level_count(E, R, kappa, N=1.0):
    """n(E): the number of positive levels below E on a static chain of sites R (N constant, v = 0); two valleys (k ~ 0 and k ~ pi), each with
    int k dR / pi standing waves, k(E, R) = arcsin(sqrt(E^2/N^2 - y^2)), y = kappa/R."""
    R = np.asarray(R, float)
    y = kappa / R
    q = (E / N) ** 2 - y ** 2
    k = np.where(q > 0, np.arcsin(np.sqrt(np.clip(q, 0, 1))), 0.0)
    return float(2.0 * np.sum(k) / math.pi)
