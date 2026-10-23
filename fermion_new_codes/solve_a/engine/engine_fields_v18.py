#!/usr/bin/env python
"""engine_fields_v18.py (builder analyst; engine spec 30.2's decisive test, the withdrawn relaunch's preparation on pod 1: "the upward
stretch from the zero at r = 74.6119 failed: Required step size is less than spacing between numbers", the same preparation passing on
box GPU 2): version 17 with one fallback. Version 17's empty-interior rule sets N' = 0 on [0, r_e] but integrates U through it with
version 16's zero machinery; there U is the integral of sources at roundoff (the r_s 32 member's T = 0 preparation reads "flat-slice gauge
continued on [0.0000, 59.3234]: min U/U_ref = -1.633e-19" on the box), so its zeros and stretches are roundoff events and one can fail
on another machine. Version 18: when version 17's integration fails at a radius inside the empty interior (r_fail <= r_e, the radius read
from the failure's message), the solve is repeated with the interval flat, U = 0 and ln N = 0 on [0, r_e] (M = rho = J = v = 0 there, as
the rule declares), the integration starting at r_e from U = 0 (upward by version 16's stretch where U'(r_e) > 0, else on the continued
branch from a down zero at r_e); info["empty_interior"]["flat_fallback"] states version 17's failure. Every other call is version 17's
(bitwise version 17 wherever it succeeds, and its failures outside the empty interior stand: 24.1, 24.4 (4)). Version 17's docstring
follows."""
import re

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq  # noqa: F401

import engine_fields_v16 as F16
import engine_fields_v17 as F17
from engine_fields_v17 import (kernels, coarse_grid, parity_spline, SmoothedSources5, charge, gauss_sum, trapping_radius, mass_function,  # noqa: F401
                               misner_sharp, exterior_check, residuals, site_fields_kernel_average, lattice_scale_power, centre_coefficients,
                               Piecewise, _intervals, padded_collar_cost, DICTIONARY, OPT17, empty_interior, _empty_report)
__doc__ = __doc__ + "\n" + (F17.__doc__ or "")


def integrate_UL_v18(src, g, r0, r_collar, cc, rtol, atol, ell, r_empty, eps_switch=1e-6, max_segments=2000):
    """docstring (1): the outward integration of (U, ln N) from r0; returns (segments, failure, counts, zeros). A segment is
    (r_start, r_end, solution(r) -> (U, L), sign, nfev); the positive segments of a stretch carry the solution in r by inverting r(s)."""
    rho_ = src.rho; J_ = src.J
    fpi = 4.0 * np.pi * g
    rg = np.linspace(0.0, r_collar, 4001)
    U_scale = float(np.sum(0.5 * (np.abs(rho_(rg[1:])) * rg[1:] ** 2 + np.abs(rho_(rg[:-1])) * rg[:-1] ** 2) * np.diff(rg))) * 2.0 * fpi
    U_sw = eps_switch * U_scale
    s_sw = float(np.sqrt(U_sw))

    def rhs_pos(r, y):
        U = y[0]
        rho = float(rho_(r)); J = float(J_(r))
        if U > 0.0:
            q = -np.sqrt(U / r)
            return [2.0 * fpi * r * r * (rho + q * J), (0.0 if r < r_empty else -fpi * r * J / q)]
        return [2.0 * fpi * r * r * rho, 0.0]

    def rhs_neg(r, y):
        return [2.0 * fpi * r * r * float(rho_(r)), 0.0]

    def Uprime(r, s):
        return 2.0 * fpi * r * r * (float(rho_(r)) - (s / np.sqrt(r)) * float(J_(r)))

    def rhs_s(s, y):
        r = y[0]
        up = Uprime(r, s)
        return [2.0 * s / up, (0.0 if r < r_empty else 2.0 * fpi * r ** 1.5 * float(J_(r)) / up)]

    def ev_flat(s, y):
        return Uprime(y[0], s)
    ev_flat.terminal = True; ev_flat.direction = 0

    def ev_sw(r, y):
        return y[0] - U_sw
    ev_sw.terminal = True; ev_sw.direction = -1

    def ev_up(r, y):
        return y[0]
    ev_up.terminal = True; ev_up.direction = 1

    def s_stretch(r_a, L_a, s_a, s_b):
        return solve_ivp(rhs_s, (s_a, s_b), [r_a, L_a], method="DOP853", dense_output=True, rtol=rtol, atol=atol, events=ev_flat)

    def seg_from_s(sol):
        """the stretch as a function of r: s(r) by inverting the monotone r(s) (interpolation, then two Newton steps), U = s^2."""
        ss = np.asarray(sol.t, float); rr = np.asarray(sol.y[0], float)
        o = np.argsort(rr)
        rr_o, ss_o = rr[o], ss[o]

        def f(x, _r=rr_o, _s=ss_o, _sol=sol):
            x = np.atleast_1d(np.asarray(x, float))
            sv = np.interp(x, _r, _s)
            for _ in range(2):
                yv = _sol.sol(sv)
                up = np.array([Uprime(b_, a_) for a_, b_ in zip(sv, yv[0])])
                drds = np.where(up != 0.0, 2.0 * sv / np.where(up != 0.0, up, 1.0), 0.0)
                step = np.where(np.abs(drds) > 0.0, (x - yv[0]) / np.where(np.abs(drds) > 0.0, drds, 1.0), 0.0)
                sv = np.clip(sv + step, min(_s.min(), 0.0), _s.max())
            yv = _sol.sol(sv)
            return np.vstack([sv ** 2, yv[1]])
        return float(rr_o[0]), float(rr_o[-1]), f
    segs = []
    zeros = []
    counts = {"stretches": 0, "near_tangencies": 0, "zeros_down": 0, "zeros_up": 0, "U_scale": U_scale, "U_switch": U_sw}
    # version 18: the empty interior flat, U = 0 and ln N = 0 on [0, r_empty] (M = rho = J = v = 0 there by the rule), the integration from r_empty
    zero_seg = lambda x: np.vstack([np.zeros_like(np.atleast_1d(np.asarray(x, float))), np.zeros_like(np.atleast_1d(np.asarray(x, float)))])
    segs.append((r0, r_empty, zero_seg, 1, 0))
    r = r_empty
    y = np.array([0.0, 0.0], float)
    up0 = Uprime(r_empty, 0.0)
    counts["flat_empty_start"] = {"r": r_empty, "U_prime": up0}
    if up0 > 0.0:
        sign = 1; fresh_zero = True
    else:
        sign = -1; fresh_zero = False
        zeros.append({"r": r_empty, "kind": "down", "U_prime": up0}); counts["zeros_down"] += 1
    while r < r_collar:
        if len(segs) >= max_segments:
            return segs, "more than %d segments of the gauge split" % max_segments, counts, zeros
        if sign > 0 and fresh_zero:
            st = s_stretch(r, float(y[1]), 0.0, s_sw)
            if not st.success or len(st.t) < 2:
                return segs, "the upward stretch from the zero at r = %.6g failed: %s" % (r, st.message), counts, zeros
            a_, b_, f_ = seg_from_s(st)
            segs.append((a_, b_, f_, 1, int(st.nfev))); counts["stretches"] += 1
            if st.status == 1:
                counts["near_tangencies"] += 1
            fresh_zero = False
            r = float(st.y[0, -1]); y = np.array([float(st.t[-1]) ** 2, float(st.y[1, -1])])
            continue
        if sign > 0:
            r_seg0 = r
            sol = solve_ivp(rhs_pos, (r, (min(r_empty, r_collar) if r < r_empty else r_collar)), y, method="DOP853", dense_output=True, rtol=rtol, atol=atol, events=ev_sw)
            if not sol.success:
                Uf_ = float(sol.y[0, -1]) if len(sol.t) else float("nan")
                return segs, "integration failed at r = %.6g, U/U_ref = %.3e, no zero bracketed: %s" % (
                    float(sol.t[-1]), Uf_ / max(U_scale, 1e-300), sol.message), counts, zeros
            r_end = float(sol.t[-1])
            segs.append((r, r_end, sol.sol, 1, int(sol.nfev)))
            if sol.status == 1 and len(sol.t_events[0]) and r_end < r_collar:
                ye = sol.y_events[0][0]
                r_e = float(sol.t_events[0][0])
                if Uprime(r_e, float(np.sqrt(max(float(ye[0]), 0.0)))) >= 0.0:
                    # the threshold touched with U turning up (no approach): continue in r below the threshold's event
                    sol2 = solve_ivp(rhs_pos, (r_e, r_collar), np.array(ye, float), method="DOP853", dense_output=True, rtol=rtol, atol=atol)
                    if not sol2.success:
                        return segs, "integration failed at r = %.6g, U/U_ref = %.3e, no zero bracketed: %s" % (
                            float(sol2.t[-1]), float(sol2.y[0, -1]) / max(U_scale, 1e-300), sol2.message), counts, zeros
                    segs.append((r_e, float(sol2.t[-1]), sol2.sol, 1, int(sol2.nfev)))
                    break
                st = s_stretch(r_e, float(ye[1]), float(np.sqrt(max(float(ye[0]), 0.0))), 0.0)
                if not st.success or len(st.t) < 2:
                    return segs, "the downward stretch at r = %.6g failed: %s" % (r_e, st.message), counts, zeros
                a_, b_, f_ = seg_from_s(st)
                segs.append((a_, b_, f_, 1, int(st.nfev))); counts["stretches"] += 1
                if st.status == 1:
                    counts["near_tangencies"] += 1
                    r = float(st.y[0, -1]); y = np.array([float(st.t[-1]) ** 2, float(st.y[1, -1])])
                    continue
                r = float(st.y[0, -1]); y = np.array([0.0, float(st.y[1, -1])])
                zeros.append({"r": r, "kind": "down", "U_prime": Uprime(r, 0.0)}); counts["zeros_down"] += 1
                sign = -1
                continue
            if r_seg0 < r_empty and r_end < r_collar:
                r = r_end; y = np.array(sol.y[:, -1], float); counts["empty_breaks"] = counts.get("empty_breaks", 0) + 1
                continue
            break
        r_seg0 = r
        sol = solve_ivp(rhs_neg, (r, (min(r_empty, r_collar) if r < r_empty else r_collar)), y, method="DOP853", dense_output=True, rtol=rtol, atol=atol, events=ev_up)
        if not sol.success:
            return segs, "integration failed at r = %.6g on the continued branch: %s" % (float(sol.t[-1]), sol.message), counts, zeros
        r_end = float(sol.t[-1])
        segs.append((r, r_end, sol.sol, -1, int(sol.nfev)))
        if sol.status == 1 and len(sol.t_events[0]) and r_end < r_collar:
            y = np.array(sol.y_events[0][0], float); y[0] = 0.0
            r = float(sol.t_events[0][0])
            zeros.append({"r": r, "kind": "up", "U_prime": Uprime(r, 0.0)}); counts["zeros_up"] += 1
            sign = 1
            fresh_zero = True
            continue
        if r_seg0 < r_empty and r_end < r_collar:
            r = r_end; y = np.array(sol.y[:, -1], float); counts["empty_breaks"] = counts.get("empty_breaks", 0) + 1
            continue
        break
    return segs, None, counts, zeros


def _solve_v18_flat(eps_tot, J_tot, g, R, ell, N0=None, tol=1e-12, maxit=1, rtol=1e-12, atol=1e-16, collar_policy="fail", support_rel=1e-14,
                     site_fields="dense", N0_func=None, on_maxit="fail", stall_cap=1e-8, r0_frac=1.0 / 64.0, r_empty=None, empty_rel=None, src_pre=None):
    """version 14 (docstring): the same interface and returned diagnostics as version 12's, plus info["regime"], info["gauge"],
    info["centre_series_check"] and the series coefficients in info["ivp"]."""
    R = np.asarray(R, float)
    r_nodes = coarse_grid(R, ell)
    src = src_pre if src_pre is not None else SmoothedSources5(eps_tot, J_tot, R, ell, r_nodes[-1], support_rel)
    info = {"it": 0, "converged": False, "ell": ell, "r_nodes": r_nodes, "R_last": src.R_last, "failed": None, "dictionary": DICTIONARY,
            "site_fields": "dense", "source_spline_error": src.spline_error, "collar_policy": collar_policy, "support_rel": support_rel,
            "charge_conservation_quadrature": (src.total_eps_quadrature / src.total_eps - 1.0) if src.total_eps else 0.0,
            "hist": [], "src": src, "rtol": rtol, "atol": atol, "module": "engine_fields_v18 (the empty interior [0, %.4f] flat after version 17's integration failed inside it)" % r_empty}
    i_need = int(np.ceil((src.R_last + 8.0 * ell) / ell))
    if i_need > len(r_nodes) - 1:
        last8 = R > r_nodes[-1] - 8.0 * ell
        tot_e = max(float(np.abs(np.asarray(eps_tot, float)).sum()), 1e-300)
        info["collar_margin"] = {"needed_node": i_need, "last_node": len(r_nodes) - 1, "deficit_ell": i_need - (len(r_nodes) - 1),
                                 "eps_share_last_8ell": float(np.abs(np.asarray(eps_tot, float)[last8]).sum()) / tot_e,
                                 "J_abs_last_8ell": float(np.abs(np.asarray(J_tot, float)[last8]).sum())}
        if collar_policy == "fail":
            info["failed"] = "the collar margin unavailable: R_last %.1f + 8 ell beyond the chain's last node %.1f" % (src.R_last, r_nodes[-1])
            return None, None, None, info
        i_collar = len(r_nodes) - 1
    else:
        i_collar = i_need
        info["collar_margin"] = {"needed_node": i_need, "last_node": len(r_nodes) - 1, "deficit_ell": 0}
    r_collar = float(r_nodes[i_collar])
    info.update({"i_collar": i_collar, "r_collar": r_collar})
    rfine = np.arange(0.0, r_collar + 1e-9, ell / 16.0)
    r0 = ell * r0_frac
    cc = centre_coefficients(src, g, ell)
    cc = dict(cc); cc["L2_v16"], cc["L4_v16"] = cc["L2"], cc["L4"]; cc["L2"] = 0.0; cc["L4"] = 0.0   # N' = 0 on the empty interior
    cc["a_v17"], cc["b_v17"] = cc["a"], cc["b"]; cc["a"] = 0.0; cc["b"] = 0.0; cc["regular"] = True   # version 18: U = 0 there too
    rho_r0 = float(src.rho(r0)); J_r0 = float(src.J(r0))
    info["centre_series_check"] = {"rho_spline_minus_series_rel": abs(rho_r0 - (cc["rho0"] + cc["rho2"] * r0 ** 2)) / max(abs(rho_r0), 1e-300),
                                   "J_spline_minus_series_rel": abs(J_r0 - (cc["J1"] * r0 + cc["J3"] * r0 ** 3)) / max(abs(J_r0), 1e-300),
                                   "regular_start": cc["regular"]}
    segs, why, counts, zeros = integrate_UL_v18(src, g, r0, r_collar, cc, rtol, atol, ell, r_empty)
    info["centre_flag"] = None if cc["regular"] else "centre outside the regular series (rho_0 <= 0)"
    if why:
        info["failed"] = why
        return None, None, None, info
    pw = Piecewise(cc, r0, segs)
    _Uc, _Lc = pw(np.array([r_collar]))
    lnN_c = float(_Lc[0])
    Uf, Lf = pw(rfine)
    Umax = max(float(np.max(Uf)), 1e-300)
    neg = Uf < -1e-12 * Umax
    gauge = {"label": "intact", "intervals": [], "min_U_over_U_ref": float(Uf.min() / Umax), "delta_L_bound": 0.0, "lapse_unresolved": False,
             "segments": len(segs), "zeros": zeros, "stretches": counts["stretches"], "near_tangencies": counts["near_tangencies"],
             "U_scale": counts["U_scale"], "U_switch": counts["U_switch"]}
    downs = [z["r"] for z in zeros if z["kind"] == "down"]; ups = [z["r"] for z in zeros if z["kind"] == "up"]
    ivs = []
    if not cc["regular"]:
        ivs.append([0.0, ups[0] if ups else r_collar]); ups = ups[1:]
    for k_, a_ in enumerate(downs):
        b_ = ups[k_] if k_ < len(ups) else r_collar
        ivs.append([a_, b_])
    if ivs:
        gauge["intervals"] = ivs

        def dLb(h):
            tot = 0.0
            for (r1, r2) in ivs:
                rm = np.arange(r1, r2 + 1e-12, h)
                if len(rm) < 2:
                    continue
                Um, _ = pw(rm)
                qm = np.sqrt(np.abs(Um) / np.maximum(rm, 1e-300))
                fm = np.minimum(4.0 * np.pi * g * np.maximum(rm, 1e-300) * np.abs(src.J(rm)) / np.maximum(qm, 1e-300), 1.0 / ell)
                tot += float(np.sum(0.5 * (fm[1:] + fm[:-1]) * np.diff(rm)))
            return tot
        dL1 = dLb(ell / 16.0); dL2 = dLb(ell / 32.0)
        gauge["delta_L_bound"] = dL2; gauge["delta_L_bound_two_resolutions"] = [dL1, dL2]
        gauge["lapse_unresolved"] = bool(abs(gauge["min_U_over_U_ref"]) > 1e-5 or dL2 > 1e-6)
        gauge["label"] = "flat-slice gauge continued on %s: min U/U_ref = %.3e (auxiliary signed U, the energy deficit's bookkeeping)%s" % (
            ", ".join("[%.4f, %.4f]" % (a_, b_) for a_, b_ in ivs), gauge["min_U_over_U_ref"], "; lapse unresolved" if gauge["lapse_unresolved"] else "")
    info["gauge"] = gauge
    pos = (Uf > 0.0) & (rfine > 0.0) & (rfine >= r_empty)
    regime = {"max_ratio": 0.0, "r_at_max": None, "intervals_above_1": [], "crossings": "located by brentq on the dense output"}

    def ratio_f(x):
        Ux, _ = pw(np.array([x]))
        if Ux[0] <= 0.0:
            return -1.0
        return 4.0 * np.pi * g * x * abs(float(src.J(x))) * ell / np.sqrt(Ux[0] / x) - 1.0
    if pos.any():
        rp = rfine[pos]; qp = np.sqrt(Uf[pos] / rp)
        ratio = 4.0 * np.pi * g * rp * np.abs(src.J(rp)) * ell / qp
        k = int(np.argmax(ratio))
        regime["max_ratio"] = float(ratio[k]); regime["r_at_max"] = float(rp[k])
        above = np.zeros(len(rfine), bool); above[np.where(pos)[0][ratio > 1.0]] = True
        ivr = []
        for (a_, b_) in _intervals(above, rfine):
            lo = a_ - ell / 16.0; hi = b_ + ell / 16.0
            try:
                a2 = brentq(ratio_f, max(lo, 1e-9), a_) if (lo > 0 and ratio_f(max(lo, 1e-9)) < 0 < ratio_f(a_)) else a_
            except ValueError:
                a2 = a_
            try:
                b2 = brentq(ratio_f, b_, min(hi, r_collar)) if (hi <= r_collar and ratio_f(b_) > 0 > ratio_f(min(hi, r_collar))) else b_
            except ValueError:
                b2 = b_
            ivr.append([float(a2), float(b2)])
        regime["intervals_above_1"] = ivr
    regime["label"] = ("lapse outside the regime on %s" % ", ".join("[%.3f, %.3f]" % (a_, b_) for a_, b_ in regime["intervals_above_1"])) if regime["intervals_above_1"] else "in the regime"
    info["regime"] = regime
    roundoff_negW = int(((Uf < 0) & ~neg).sum())

    def Nfin(r, _c=lnN_c):
        x = np.abs(np.atleast_1d(np.asarray(r, float)))
        inside = x <= r_collar
        _u, lnN = pw(np.minimum(x, r_collar))
        return np.where(inside, np.exp(lnN - _c), 1.0)
    info["it"] = 1
    info["converged"] = True
    info["hist"] = []
    info["ivp"] = {"r0": r0, "U0": float(cc["a"] * r0 ** 3 + cc["b"] * r0 ** 5), "lnN0": float(cc["L2"] * r0 ** 2 + cc["L4"] * r0 ** 4), "lnN_collar": lnN_c,
                   "N_centre": float(np.exp(-lnN_c)), "nfev": int(sum(s[4] for s in segs)), "segments": len(segs), "series": {k: cc[k] for k in cc}}
    info["last_dN"], info["last_dW"] = 0.0, 0.0
    rhoW = src.rho
    solW = solve_ivp(lambda r, W: [8.0 * np.pi * g * r * r * float(Nfin(r)[0]) ** 2 * float(rhoW(r))], (0.0, r_collar), [0.0], method="DOP853",
                     dense_output=True, rtol=rtol, atol=atol)
    if not solW.success:
        info["failed"] = "the final W integration failed: %s" % solW.message
        return None, None, None, info
    Wf = solW.sol(rfine)[0]
    Wmax = max(float(np.max(np.abs(Wf))), 1e-300)
    if Wf.min() < -1e-12 * Wmax:
        info["W_negative"] = {"min_W_over_max": float(Wf.min() / Wmax), "r": float(rfine[int(np.argmin(Wf))]), "inside_gauge_intervals": bool(neg.any())}
    def W_geom_f(r, _ivs=gauge["intervals"]):
        r = np.abs(np.atleast_1d(np.asarray(r, float)))
        w = np.asarray(solW.sol(np.minimum(r, r_collar))[0], float)
        for (a_, b_) in _ivs:
            w = np.where((r >= a_) & (r <= b_), 0.0, w)
        return np.maximum(w, 0.0)
    info["W_geom"] = W_geom_f
    info["W_aux_signed_label"] = "auxiliary signed U (the energy deficit's bookkeeping)"
    E = float(solW.sol(r_collar)[0] / (2.0 * g))
    _sw, _gE = solW, g * E

    def W_field(r):
        r = np.abs(np.atleast_1d(np.asarray(r, float)))
        out = np.full(r.shape, 2.0 * _gE)
        m_ = r < r_collar
        if m_.any():
            out[m_] = _sw.sol(r[m_])[0]
        return out

    def N_dense_f(r):
        return Nfin(r)

    def v_dense_f(r):
        r = np.atleast_1d(np.asarray(r, float))
        a_ = np.abs(r)
        W = W_field(a_)
        out = -np.sqrt(np.where(W > 0.0, W, 0.0) / np.maximum(a_, 1e-300))
        out = np.where(a_ > 0.0, out, 0.0)
        return np.where(r < 0, -out, out)
    cd = lambda F, x, hh: (F(x + hh) - F(x - hh)) / (2 * hh)
    invNf = lambda x: 1.0 / N_dense_f(x)

    def residuals_at(h):
        ri = rfine[(rfine > 2 * h) & (rfine <= r_collar)]
        rho_i = src.rho(ri); J_i = src.J(ri)
        Nr = N_dense_f(ri); vr = v_dense_f(ri)
        dWdr = (4.0 * cd(W_field, ri, h / 2) - cd(W_field, ri, h)) / 3.0
        termW = 8.0 * np.pi * g * ri ** 2 * Nr ** 2 * rho_i
        dinvN = (4.0 * cd(invNf, ri, h / 2) - cd(invNf, ri, h)) / 3.0
        termN = np.where(np.abs(vr) > 1e-12 * max(float(np.abs(vr).max()), 1e-300), 4.0 * np.pi * g * ri * J_i / np.where(vr != 0, vr, 1.0), 0.0)
        inside_e = ri < r_empty
        termN_omitted = np.where(inside_e, termN, 0.0); termN = np.where(inside_e, 0.0, termN)
        sW = max(float(np.max(np.abs(termW))), 1e-300); sN = max(float(np.max(np.abs(termN))), 1e-300)
        omitted_.append(float(np.max(np.abs(termN_omitted))) / sN)
        m8 = ri >= r_collar - 8.0 * ell
        return (float(np.max(np.abs(dWdr - termW))) / sW, float(np.max(np.abs(dinvN - termN))) / sN,
                (float(np.max(np.abs(dWdr - termW)[m8])) / sW) if m8.any() else None, (float(np.max(np.abs(dinvN - termN)[m8])) / sN) if m8.any() else None)
    omitted_ = []
    rW, rN, rW8, rN8 = residuals_at(ell / 64.0)
    info["residual_W"], info["residual_invN"] = rW, rN
    info["residual_matching_region"] = {"W": rW8, "invN": rN8}
    rW2, rN2, _, _ = residuals_at(ell / 32.0)
    info["residual_ladder"] = {"h_ell/64_and_/128": [rW, rN], "h_ell/32_and_/64": [rW2, rN2]}
    rc = ell / 16.0
    W0 = float(W_field(rc)[0]); ser = 8.0 * np.pi * g / 3.0 * float(N_dense_f(0.0)[0]) ** 2 * float(src.rho(0.0)) * rc ** 3
    info["centre_series_ratio"] = (W0 / ser) if ser != 0 else None
    info["W_roundoff_negative_points"] = roundoff_negW
    if r_collar < r_nodes[-1]:
        rhsT = lambda r, W: [8.0 * np.pi * g * r * r * float(src.rho(r))]
        solT = solve_ivp(rhsT, (r_collar, r_nodes[-1]), [2.0 * _gE], method="DOP853", rtol=rtol, atol=atol)
        info["W_tail"] = {"r_end": float(r_nodes[-1]), "E_tail_end": float(solT.y[0][-1] / (2.0 * g)), "E_collar": E, "flagged": False}
    else:
        rt = np.arange(r_collar, r_collar + 8.0 * ell + 1e-9, ell / 16.0)
        rho_t, _ = src.direct(rt)
        f_t = 8.0 * np.pi * g * rt ** 2 * rho_t
        Wt = 2.0 * _gE + float(np.sum(0.5 * (f_t[1:] + f_t[:-1]) * np.diff(rt)))
        info["W_tail"] = {"r_end": float(rt[-1]), "E_tail_end": float(Wt / (2.0 * g)), "E_collar": E, "flagged": True}
    N_sites = np.array(N_dense_f(R), float); v_sites = np.array(v_dense_f(R), float)
    ext = R >= r_collar
    N_sites[ext] = 1.0; v_sites[ext] = -np.sqrt(2.0 * g * E / R[ext])
    W_nodes = W_field(r_nodes)
    N_nodes = np.array(N_dense_f(r_nodes), float)
    v_nodes = np.zeros(len(r_nodes)); v_nodes[1:] = -np.sqrt(np.where(W_nodes[1:] > 0, W_nodes[1:], 0.0) / r_nodes[1:])
    inner = ~ext
    if inner.any():
        N_spl = parity_spline(r_nodes, N_nodes, +1); v_spl = parity_spline(r_nodes, v_nodes, -1)
        info["interp_floor_N"] = float(np.max(np.abs(N_spl(R[inner]) - N_sites[inner])))
        info["interp_floor_v"] = float(np.max(np.abs(v_spl(R[inner]) - v_sites[inner])))
    info["rfine"] = rfine
    info["U_dense"] = lambda r: pw(r)[0]
    info.update({"N_dense": N_dense_f, "v_dense": v_dense_f, "W_dense": W_field, "W_field": W_field, "rho_ell": src.rho, "J_ell": src.J,
                 "W_nodes": W_nodes, "N_nodes": N_nodes, "v_nodes": v_nodes, "E_collar": E, "N_min": float(N_sites.min()),
                 "spectrum": lattice_scale_power(N_sites - 1.0, v_sites, R, ell, float(N_dense_f(0.0)[0]) - 1.0)})
    info["empty_interior"] = _empty_report(src, g, ell, r_empty, empty_rel, pw, N_dense_f, E, omitted_, counts)
    return N_sites, v_sites, W_nodes, info


def _failure_radius(why):
    m = re.search(r"r = ([-+0-9.eE]+)", why or "")
    try:
        return float(m.group(1)) if m else None
    except ValueError:
        return None


def solve_constraints_ivp(eps_tot, J_tot, g, R, ell, N0=None, tol=1e-12, maxit=1, rtol=1e-12, atol=1e-16, collar_policy="fail", support_rel=1e-14,
                          site_fields="dense", N0_func=None, on_maxit="fail", stall_cap=1e-8, r0_frac=1.0 / 64.0, empty_rel="default"):
    """version 18 (docstring): version 17's call; on its failure inside the empty interior, the flat interval's solve."""
    out = F17.solve_constraints_ivp(eps_tot, J_tot, g, R, ell, N0=N0, tol=tol, maxit=maxit, rtol=rtol, atol=atol, collar_policy=collar_policy,
                                    support_rel=support_rel, site_fields=site_fields, N0_func=N0_func, on_maxit=on_maxit, stall_cap=stall_cap, r0_frac=r0_frac,
                                    empty_rel=empty_rel)
    info = out[3]
    if not info.get("failed"):
        return out
    er = OPT17["empty_rel"] if (isinstance(empty_rel, str) and empty_rel == "default") else empty_rel
    if er is None:
        return out
    R_ = np.asarray(R, float)
    r_nodes = coarse_grid(R_, ell)
    src = SmoothedSources5(eps_tot, J_tot, R_, ell, r_nodes[-1], support_rel)
    r_e = empty_interior(src, ell, er)
    r_f = _failure_radius(info.get("failed"))
    if r_e is None or r_f is None or r_f > r_e:
        return out
    kw = dict(N0=N0, tol=tol, maxit=maxit, rtol=rtol, atol=atol, collar_policy=collar_policy, support_rel=support_rel, site_fields=site_fields,
              N0_func=N0_func, on_maxit=on_maxit, stall_cap=stall_cap, r0_frac=r0_frac)
    out2 = _solve_v18_flat(eps_tot, J_tot, g, R, ell, r_empty=r_e, empty_rel=er, src_pre=src, **kw)
    i2 = out2[3]
    fb = {"v17_failure": info.get("failed"), "failure_radius": r_f, "r_e": r_e}
    if i2.get("failed"):
        i2["failed"] = "%s (version 18's flat empty interior [0, %.4f] after version 17's failure: %s)" % (i2["failed"], r_e, info.get("failed"))
    if isinstance(i2.get("empty_interior"), dict):
        i2["empty_interior"]["flat_fallback"] = fb
        i2["empty_interior"]["label"] = (i2["empty_interior"].get("label") or "") + "; flat (U = 0) after version 17's failure at r = %.4f inside it" % r_f
    else:
        i2["empty_interior"] = {"rule": "on", "r_e": r_e, "flat_fallback": fb}
    return out2


def r0_check(eps_tot, J_tot, g, R, ell, info, rtol=1e-12, atol=1e-16, collar_policy="flag", support_rel=1e-14, empty_rel="default"):
    """the solve repeated from r_0/2 (engine 20.2 (1)) on this version's solve."""
    r0f = float(info["ivp"]["r0"]) / float(info["ell"])
    _, _, _, ib = solve_constraints_ivp(eps_tot, J_tot, g, R, ell, rtol=rtol, atol=atol, collar_policy=collar_policy, support_rel=support_rel, r0_frac=0.5 * r0f, empty_rel=empty_rel)
    if ib.get("failed"):
        return {"failed": ib["failed"]}
    rf = info["rfine"]
    Wa = info["W_field"](rf); Wmax = max(float(np.max(np.abs(Wa))), 1e-300)
    return {"r0": float(info["ivp"]["r0"]), "r0_half": float(ib["ivp"]["r0"]), "dN_max": float(np.max(np.abs(ib["N_dense"](rf) - info["N_dense"](rf)))),
            "dW_rel_max": float(np.max(np.abs(ib["W_field"](rf) - Wa))) / Wmax, "dE_rel": abs(float(ib["E_collar"]) - float(info["E_collar"])) / abs(float(info["E_collar"]))}
