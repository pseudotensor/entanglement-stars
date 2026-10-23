#!/usr/bin/env python
"""pt3_calib.py (busi, PAGE design v3 sec. 6; A3, F6): the packet-charge calibration on the pure preparation.

Packet: a particle psi -> (1 - C) psi/||.|| so that C' = C + |psi'><psi'| is a covariance (F6), or a hole phi -> C phi/||.||, C' = C - |phi'><phi'|,
excitation energy -<phi'|h|phi'> (A3). C is pure (C = Phi Phi^dag, orthonormal columns; the full-rank f_ws state has no exact unit addition, A3).
The availability of the empty (filled) mode is the kept norm ||(1 - C) psi|| (||C phi||), printed.
Reads of the added charge, one physical copy (target <psi'|h|psi'> = omega) and the weighted copies (the chain's w = 16 kappa, target w omega):
  * direct: Tr(dC h) and on the enclosing surface Tr(dC h_{R_enc}) (h_R = sum_{x <= R} eps_x), the collar's Tr(dC h_{R_col}) separately;
  * Gauss rows: the engine's sources eps_x = sum_ch w dens(h_0, .), J_x = sum_ch w dens(Q, .) (engine_collapse_v19 sources(), engine_collapse_v5.dens)
    with the packet's dens added, re-solved by the field module of record (engine_fields_v16.solve_constraints_ivp, collar flag, support 1e-14, the
    inventory's dense route), M(R) = W(R)/(2g) read on a surface R_enc enclosing the packet (E_collar) and at the collar R_col; the packet's
    sources enter at scale +s and -s and the central difference cancels the second order: dE_one_copy = (E(+s) - E(-s))/(2 s).
Pass (F6): |dE - <psi'|h|psi'>| <= max(1e-3 omega, the Gauss-row residual). Label (A3, F6): with the background and P fixed, dE largely cancels
the subtraction by construction; passing it does not vindicate a reference's treatment of internal emission."""
import math

import numpy as np

import pt3_core as K


def project_particle(psi, Phi):
    """(1 - Phi Phi^dag) psi normalized; returns (psi', kept norm)."""
    r = psi - Phi @ (Phi.conj().T @ psi)
    nr = float(np.linalg.norm(r))
    return r / nr, nr


def project_hole(phi, Phi):
    r = Phi @ (Phi.conj().T @ phi)
    nr = float(np.linalg.norm(r))
    return r / nr, nr


def direct_reads(ops, N, v, x, sign, R_enc, R_col):
    """Tr(dC O) for dC = sign |x><x| and O = h, h_{R_enc}, h_{R_col}."""
    h = ops.h(N, v)
    e = lambda O: sign * float(np.real(np.vdot(x, O @ x)))
    return {"Tr_dC_h": e(h), "Tr_dC_h_Renc": e(ops.h_R(N, v, R_enc)), "Tr_dC_h_Rcol": e(ops.h_R(N, v, R_col))}


def gauss_read(FM, eps0, J0, g, R_sites, ell, d_eps, d_J, s, R_reads):
    """M(R) = W(R)/(2g) and E_collar at sources eps0 + s d_eps (and J); returns a dict per scale."""
    out = {}
    for sc in (+s, -s, 0.0):
        _, _, _, info = FM.solve_constraints_ivp(eps0 + sc * d_eps, J0 + sc * d_J, g, R_sites, ell, collar_policy="flag", support_rel=1e-14)
        if info.get("failed"):
            raise RuntimeError("Gauss solve failed at scale %g: %s" % (sc, info["failed"]))
        W = info["W_dense"]
        out[sc] = {"E_collar": float(info["E_collar"]), "r_collar": float(info["r_collar"]),
                   "M": {str(R): float(W(np.array([R]))[0] / (2.0 * g)) for R in R_reads},
                   "residual_W": float(info["residual_W"]), "residual_invN": float(info["residual_invN"])}
    return out


def calibrate(ops, N_avg, v_avg, Phi, psi, kind, omega, R_enc, R_col, gauss=None, weight=1.0, s_list=(1.0,)):
    """one calibration packet; gauss = dict(FM, eps0, J0, g, R_sites, ell) for the rows' read (None: direct reads only)."""
    if kind == "particle":
        x, kept = project_particle(psi, Phi); sign = +1.0
    else:
        x, kept = project_hole(psi, Phi); sign = -1.0
    target = sign * float(np.real(np.vdot(x, ops.h(N_avg, v_avg) @ x)))       # the excitation's charge, one copy
    res = {"kind": kind, "omega": omega, "kept_norm": kept, "target_one_copy": target, "target_over_omega": target / abs(omega),
           "weight": weight, "target_weighted": weight * target}
    res.update(direct_reads(ops, N_avg, v_avg, x, sign, R_enc, R_col))
    res["direct_pass"] = bool(abs(res["Tr_dC_h_Renc"] - target) <= 1e-3 * abs(omega))
    res["direct_collar_unchanged"] = bool(abs(res["Tr_dC_h_Rcol"]) <= 1e-3 * abs(omega))
    if gauss is not None:
        d_eps = sign * ops.dens(ops.h0, x[:, None])
        d_J = sign * ops.dens(ops.Q, x[:, None])
        rows = []
        for s in s_list:
            g_ = gauss_read(gauss["FM"], gauss["eps0"], gauss["J0"], gauss["g"], gauss["R_sites"], gauss["ell"], d_eps, d_J, s, (R_col,))
            dE = (g_[+s]["E_collar"] - g_[-s]["E_collar"]) / (2.0 * s)
            dMcol = (g_[+s]["M"][str(R_col)] - g_[-s]["M"][str(R_col)]) / (2.0 * s)
            nl = (g_[+s]["E_collar"] + g_[-s]["E_collar"] - 2.0 * g_[0.0]["E_collar"]) / (2.0 * s * s)
            resid = max(g_[0.0]["residual_W"], g_[0.0]["residual_invN"]) * abs(g_[0.0]["E_collar"]) / max(s, 1e-300)
            tol = max(1e-3 * abs(omega), resid)
            rows.append({"scale_copies": s, "dE_one_copy": dE, "dE_minus_target": dE - target, "dE_over_target": dE / target if target else None,
                         "dM_collar_one_copy": dMcol, "second_order_per_copy": nl, "E_collar_0": g_[0.0]["E_collar"], "r_collar": g_[0.0]["r_collar"],
                         "gauss_residual_scaled": resid, "tol": tol, "pass": bool(abs(dE - target) <= tol),
                         "collar_unchanged": bool(abs(dMcol) <= tol)})
        res["gauss"] = rows
        res["pass"] = bool(all(r["pass"] for r in rows))
    else:
        res["pass"] = res["direct_pass"]
    return res
