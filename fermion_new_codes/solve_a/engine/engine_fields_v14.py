#!/usr/bin/env python
"""engine_fields_v14.py (builder analyst; engine spec 20.2 (1) to (3) and R26a's items 2 to 5 and 9; version 13's floor rule is withdrawn
by the lead (20.2) and its code stays as record). Version 14 = version 12 with solve_constraints_ivp replaced:
(1) the momentum equation kept everywhere in the undivided form q dL = -4 pi g r J_ell dr: no running qmax (version 12 set L' = 0 where -q
fell below 1e-12 of the running maximum of -q over the right-hand side's evaluations, R26a item 1); the centre by the series to r^5 with the
kernels' analytic coefficients (R26a item 3): with c_n = (2 pi ell^2)^(-3/2) exp(-R_n^2/(2 ell^2)),
  rho_ell = rho_0 + rho_2 r^2 + ..., rho_0 = sum c_n e_n, rho_2 = sum c_n e_n (R_n^2/(6 ell^4) - 1/(2 ell^2)),
  J_ell = J_1 r + J_3 r^3 + ..., J_1 = sum c_n J_n R_n/(3 ell^2), J_3 = sum c_n J_n (R_n^3/(30 ell^6) - R_n/(6 ell^4)),
  U = a r^3 + b r^5, a = (8 pi g/3) rho_0, b = (8 pi g/5)(rho_2 + q_1 J_1), q_1 = -sqrt(a), q_3 = q_1 b/(2a),
  ln N - ln N(0) = L_2 r^2 + L_4 r^4, L_2 = -2 pi g J_1/q_1, L_4 = -pi g (J_3/q_1 - J_1 q_3/q_1^2)
(version 12: the leading terms with J_1 = J_ell(r_0)/r_0); below r_0 = ell/64 the dense functions are the series; the coefficients are
checked against the source splines at r_0 (info["centre_series_check"]); N_centre = exp(-ln N(r_collar)) (R26a item 4); r0_check repeats
the solve from r_0/2 (the largest change of the dense N, of W/max W and of E);
(2) the regime diagnostic: the lapse's gradient has no structure below ell, so the constraint is in the regime only where
4 pi g r |J_ell| ell <= |q|; info["regime"]: the largest ratio over the fine grid where U > 0 and the intervals where it exceeds 1
("lapse outside the regime on [r1, r2]"); the run continues with the exact equation while U > 0;
(3) the flat-slice gauge continued where U < -1e-12 max U (version 12 failed the solve there): the integration is split at the zeros of U
(terminal events), the momentum equation has no real q where U <= 0, and there the geometry is the Minkowski patch (L' = 0, U' = 8 pi g r^2
rho_ell, the enclosed-energy scale integrated through; a positive segment starting at a zero of U begins with an ell/4 stretch in
t = sqrt(r - r_*), where the momentum equation's integrable 1/sqrt(U) is regular, and a step-size failure of the positive branch below
1e-6 of the largest U is taken as the zero there, counted; the first test of this module (fields_v14_test_v1) failed on a constructed
empty interior before these two were added, and no run used the module before them); the dense W and v take U := 0 on the intervals (v = -sqrt(max(W, 0)/r)); E is the
collar charge as before; info["gauge"]: the intervals, min U/U_ref (U_ref = max U on the fine grid), delta L_bound = int min(4 pi g r
|J_ell|/|q|, 1/ell) dr over the intervals (|q| = sqrt(|U|/r), the lapse change the omitted momentum would produce under the regime's
cap), "lapse unresolved" when |min U|/U_ref > 1e-5 or delta L_bound > 1e-6 (never a failure: the handoff wants flagged histories), and
the label "flat-slice gauge continued on [r1, r2]: min U/U_ref = ..." or "intact"; a negative final W on the intervals is recorded, not
rejected;
padded_collar_cost checks the integration's success before its acceptance (R26a item 5). The sign convention (R26a item 2, engine 20.3):
J_ell = <P_r> = -J_r of the goal's display, documented in the dictionary line; no number changes. Everything else is version 12's."""
import numpy as np
from scipy.integrate import solve_ivp

import engine_fields_v12 as F12

kernels = F12.kernels
coarse_grid = F12.coarse_grid
parity_spline = F12.parity_spline
SmoothedSources5 = F12.SmoothedSources5
charge = F12.charge
gauss_sum = F12.gauss_sum
trapping_radius = F12.trapping_radius
mass_function = F12.mass_function
misner_sharp = F12.misner_sharp
exterior_check = F12.exterior_check
residuals = F12.residuals
site_fields_kernel_average = F12.site_fields_kernel_average
lattice_scale_power = F12.lattice_scale_power
DICTIONARY = F12.DICTIONARY + "; J_ell = <P_r> = -J_r of the goal's display (engine 20.3)"


def centre_coefficients(src, g, ell):
    """the analytic Taylor coefficients of the smoothed sources at the centre and the series of U and ln N (docstring (1))."""
    sel = src.sel
    Rs = src.R[sel]; es = src.eps_tot[sel]; Js = src.J_tot[sel]
    c = np.exp(-Rs ** 2 / (2.0 * ell ** 2)) / ((2.0 * np.pi) ** 1.5 * ell ** 3)
    rho0 = float(c @ es); rho2 = float(c @ (es * (Rs ** 2 / (6.0 * ell ** 4) - 1.0 / (2.0 * ell ** 2))))
    J1 = float(c @ (Js * Rs / (3.0 * ell ** 2))); J3 = float(c @ (Js * (Rs ** 3 / (30.0 * ell ** 6) - Rs / (6.0 * ell ** 4))))
    a = 8.0 * np.pi * g * rho0 / 3.0
    if a > 0.0:
        q1 = -np.sqrt(a); b = 8.0 * np.pi * g / 5.0 * (rho2 + q1 * J1); q3 = q1 * b / (2.0 * a)
        L2 = -2.0 * np.pi * g * J1 / q1; L4 = -np.pi * g * (J3 / q1 - J1 * q3 / q1 ** 2)
        regular = True
    else:
        q1 = 0.0; b = 8.0 * np.pi * g / 5.0 * rho2; q3 = 0.0; L2 = 0.0; L4 = 0.0
        regular = False
    return {"rho0": rho0, "rho2": rho2, "J1": J1, "J3": J3, "a": a, "b": b, "q1": q1, "q3": q3, "L2": L2, "L4": L4, "regular": regular}


class Piecewise:
    """U(r) and ln N(r) (ln N(0) = 0) from the series below r0 and the integration's segments above it."""

    def __init__(self, cc, r0, segs):
        self.cc, self.r0, self.segs = cc, r0, segs
        self.starts = np.array([s[0] for s in segs])

    def __call__(self, r):
        r = np.atleast_1d(np.asarray(r, float))
        x = np.abs(r)
        U = np.empty_like(x); L = np.empty_like(x)
        lo = x < self.r0
        if lo.any():
            xl = x[lo]
            U[lo] = self.cc["a"] * xl ** 3 + self.cc["b"] * xl ** 5
            L[lo] = self.cc["L2"] * xl ** 2 + self.cc["L4"] * xl ** 4
        hi = ~lo
        if hi.any():
            xh = x[hi]
            k = np.clip(np.searchsorted(self.starts, xh, side="right") - 1, 0, len(self.segs) - 1)
            Uh = np.empty_like(xh); Lh = np.empty_like(xh)
            for j in np.unique(k):
                m_ = k == j
                s0, s1, sol = self.segs[j][:3]
                y = sol(np.clip(xh[m_], s0, s1))
                Uh[m_] = y[0]; Lh[m_] = y[1]
            U[hi] = Uh; L[hi] = Lh
        return U, L


def _intervals(mask, r):
    out = []
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append([float(r[i]), float(r[j])])
            i = j + 1
        else:
            i += 1
    return out


def integrate_UL(src, g, r0, r_collar, cc, rtol, atol, ell, max_segments=200):
    """the outward integration of (U, ln N) from r0, split at the zeros of U (docstring (3)); a positive segment that starts at U = 0 (an
    upward zero) is integrated over its first ell/4 in t = sqrt(r - r_*), where dL/dt = 8 pi g r J t sqrt(r/U) is regular (its limit
    8 pi g r J sqrt(r/U'(r_*)) at t = 0), then in r; a step-size failure of the positive branch at U below 1e-6 of the largest U met is the
    approach to a zero of U (the momentum equation's integrable 1/sqrt(U)) and is taken as the zero there (counted). Returns
    (segments, failure, counts)."""
    rho_ = src.rho; J_ = src.J
    fpi = 4.0 * np.pi * g

    def rhs_pos(r, y):
        U = y[0]
        rho = float(rho_(r)); J = float(J_(r))
        if U > 0.0:
            q = -np.sqrt(U / r)
            return [2.0 * fpi * r * r * (rho + q * J), -fpi * r * J / q]
        return [2.0 * fpi * r * r * rho, 0.0]

    def rhs_neg(r, y):
        return [2.0 * fpi * r * r * float(rho_(r)), 0.0]

    def rhs_t(rs_):
        def f(t, y):
            r = rs_ + t * t
            U = y[0]; rho = float(rho_(r)); J = float(J_(r))
            if U > 0.0:
                q = -np.sqrt(U / r)
                return [2.0 * t * 2.0 * fpi * r * r * (rho + q * J), 2.0 * fpi * r * J * t * np.sqrt(r / U)]
            Up = 2.0 * fpi * r * r * rho
            return [2.0 * t * Up, (2.0 * fpi * r * J * np.sqrt(r / Up)) if Up > 0.0 else 0.0]
        return f

    def ev_down(r, y):
        return y[0]
    ev_down.terminal = True; ev_down.direction = -1

    def ev_up(r, y):
        return y[0]
    ev_up.terminal = True; ev_up.direction = 1
    U0 = cc["a"] * r0 ** 3 + cc["b"] * r0 ** 5
    L0 = cc["L2"] * r0 ** 2 + cc["L4"] * r0 ** 4
    y = np.array([U0, L0], float)
    sign = 1 if U0 > 0.0 else -1
    r = r0
    segs = []
    counts = {"step_failure_zeros": 0, "t_stretches": 0}
    Useen = max(U0, 0.0)
    fresh_zero = False
    while r < r_collar:
        if len(segs) >= max_segments:
            return segs, "more than %d segments of the gauge split" % max_segments, counts
        if sign > 0 and fresh_zero:
            d = min(0.25 * ell, r_collar - r)
            rs_ = r
            sol = solve_ivp(rhs_t(rs_), (0.0, np.sqrt(d)), y, method="DOP853", dense_output=True, rtol=rtol, atol=atol, events=ev_down)
            if not sol.success or len(sol.t) < 2:
                return segs, "the t-stretch from the zero of U at r = %.4g failed: %s" % (rs_, sol.message), counts
            counts["t_stretches"] += 1
            r_end = rs_ + float(sol.t[-1]) ** 2
            segs.append((r, r_end, (lambda rr, s_=sol.sol, a_=rs_: s_(np.sqrt(np.maximum(np.asarray(rr, float) - a_, 0.0)))), sign, int(sol.nfev)))
            Useen = max(Useen, float(np.max(sol.y[0])))
            fresh_zero = False
            if sol.status == 1 and len(sol.t_events[0]) and r_end < r_collar:
                y = np.array(sol.y_events[0][0], float); y[0] = 0.0
                r = rs_ + float(sol.t_events[0][0]) ** 2
                sign = -1
                continue
            y = np.array(sol.y[:, -1], float); r = r_end
            continue
        f, ev = (rhs_pos, ev_down) if sign > 0 else (rhs_neg, ev_up)
        sol = solve_ivp(f, (r, r_collar), y, method="DOP853", dense_output=True, rtol=rtol, atol=atol, events=ev)
        if not sol.success:
            if sign > 0 and len(sol.t) >= 2 and float(sol.y[0, -1]) < 1e-6 * max(Useen, 1e-300):
                segs.append((r, float(sol.t[-1]), sol.sol, sign, int(sol.nfev)))
                r = float(sol.t[-1]); y = np.array(sol.y[:, -1], float); y[0] = 0.0
                sign = -1 if float(rho_(r)) < 0.0 else 1
                fresh_zero = sign > 0
                counts["step_failure_zeros"] += 1
                continue
            return segs, "the initial-value integration failed at r = %.4g: %s" % (float(sol.t[-1]), sol.message), counts
        r_end = float(sol.t[-1])
        segs.append((r, r_end, sol.sol, sign, int(sol.nfev)))
        Useen = max(Useen, float(np.max(sol.y[0])))
        if sol.status == 1 and len(sol.t_events[0]) and r_end < r_collar:
            y = np.array(sol.y_events[0][0], float); y[0] = 0.0
            r = float(sol.t_events[0][0])
            sign = -sign
            fresh_zero = sign > 0
            continue
        break
    return segs, None, counts


def solve_constraints_ivp(eps_tot, J_tot, g, R, ell, N0=None, tol=1e-12, maxit=1, rtol=1e-12, atol=1e-16, collar_policy="fail", support_rel=1e-14,
                          site_fields="dense", N0_func=None, on_maxit="fail", stall_cap=1e-8, r0_frac=1.0 / 64.0):
    """version 14 (docstring): the same interface and returned diagnostics as version 12's, plus info["regime"], info["gauge"],
    info["centre_series_check"] and the series coefficients in info["ivp"]."""
    R = np.asarray(R, float)
    r_nodes = coarse_grid(R, ell)
    src = SmoothedSources5(eps_tot, J_tot, R, ell, r_nodes[-1], support_rel)
    info = {"it": 0, "converged": False, "ell": ell, "r_nodes": r_nodes, "R_last": src.R_last, "failed": None, "dictionary": DICTIONARY,
            "site_fields": "dense", "source_spline_error": src.spline_error, "collar_policy": collar_policy, "support_rel": support_rel,
            "charge_conservation_quadrature": (src.total_eps_quadrature / src.total_eps - 1.0) if src.total_eps else 0.0,
            "hist": [], "src": src, "rtol": rtol, "atol": atol, "module": "engine_fields_v14"}
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
    rho_r0 = float(src.rho(r0)); J_r0 = float(src.J(r0))
    info["centre_series_check"] = {"rho_spline_minus_series_rel": abs(rho_r0 - (cc["rho0"] + cc["rho2"] * r0 ** 2)) / max(abs(rho_r0), 1e-300),
                                   "J_spline_minus_series_rel": abs(J_r0 - (cc["J1"] * r0 + cc["J3"] * r0 ** 3)) / max(abs(J_r0), 1e-300),
                                   "regular_start": cc["regular"]}
    segs, why, counts = integrate_UL(src, g, r0, r_collar, cc, rtol, atol, ell)
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
             "segments": len(segs), "step_failure_zeros": counts["step_failure_zeros"], "t_stretches": counts["t_stretches"]}
    if neg.any():
        gauge["intervals"] = _intervals(neg, rfine)
        Jn = np.abs(src.J(rfine[neg])); rn = np.maximum(rfine[neg], 1e-300); qn = np.sqrt(np.abs(Uf[neg]) / rn)
        integrand = np.minimum(4.0 * np.pi * g * rn * Jn / np.maximum(qn, 1e-300), 1.0 / ell)
        dL = 0.0
        for (r1, r2) in gauge["intervals"]:
            m_ = (rfine >= r1) & (rfine <= r2)
            Jm = np.abs(src.J(rfine[m_])); rm = np.maximum(rfine[m_], 1e-300); qm = np.sqrt(np.abs(Uf[m_]) / rm)
            fm = np.minimum(4.0 * np.pi * g * rm * Jm / np.maximum(qm, 1e-300), 1.0 / ell)
            dL += float(np.sum(0.5 * (fm[1:] + fm[:-1]) * np.diff(rfine[m_]))) if m_.sum() > 1 else float(fm.sum() * ell / 16.0)
        gauge["delta_L_bound"] = dL
        gauge["integrand_max"] = float(integrand.max()) if len(integrand) else 0.0
        gauge["lapse_unresolved"] = bool(abs(gauge["min_U_over_U_ref"]) > 1e-5 or dL > 1e-6)
        gauge["label"] = "flat-slice gauge continued on %s: min U/U_ref = %.3e%s" % (
            ", ".join("[%.2f, %.2f]" % (a_, b_) for a_, b_ in gauge["intervals"]), gauge["min_U_over_U_ref"], "; lapse unresolved" if gauge["lapse_unresolved"] else "")
    info["gauge"] = gauge
    pos = (Uf > 0.0) & (rfine > 0.0)
    regime = {"max_ratio": 0.0, "r_at_max": None, "intervals_above_1": []}
    if pos.any():
        rp = rfine[pos]; qp = np.sqrt(Uf[pos] / rp)
        ratio = 4.0 * np.pi * g * rp * np.abs(src.J(rp)) * ell / qp
        k = int(np.argmax(ratio))
        regime["max_ratio"] = float(ratio[k]); regime["r_at_max"] = float(rp[k])
        above = np.zeros(len(rfine), bool); above[np.where(pos)[0][ratio > 1.0]] = True
        regime["intervals_above_1"] = _intervals(above, rfine)
    regime["label"] = ("lapse outside the regime on %s" % ", ".join("[%.2f, %.2f]" % (a_, b_) for a_, b_ in regime["intervals_above_1"])) if regime["intervals_above_1"] else "in the regime"
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
        sW = max(float(np.max(np.abs(termW))), 1e-300); sN = max(float(np.max(np.abs(termN))), 1e-300)
        m8 = ri >= r_collar - 8.0 * ell
        return (float(np.max(np.abs(dWdr - termW))) / sW, float(np.max(np.abs(dinvN - termN))) / sN,
                (float(np.max(np.abs(dWdr - termW)[m8])) / sW) if m8.any() else None, (float(np.max(np.abs(dinvN - termN)[m8])) / sN) if m8.any() else None)
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
    return N_sites, v_sites, W_nodes, info


def r0_check(eps_tot, J_tot, g, R, ell, info, rtol=1e-12, atol=1e-16, collar_policy="flag", support_rel=1e-14):
    """the solve repeated from r_0/2 (engine 20.2 (1)): the largest change of the dense N and of W/max W on the fine grid, and of E."""
    r0f = float(info["ivp"]["r0"]) / float(info["ell"])
    _, _, _, ib = solve_constraints_ivp(eps_tot, J_tot, g, R, ell, rtol=rtol, atol=atol, collar_policy=collar_policy, support_rel=support_rel, r0_frac=0.5 * r0f)
    if ib.get("failed"):
        return {"failed": ib["failed"]}
    rf = info["rfine"]
    Wa = info["W_field"](rf); Wmax = max(float(np.max(np.abs(Wa))), 1e-300)
    return {"r0": float(info["ivp"]["r0"]), "r0_half": float(ib["ivp"]["r0"]), "dN_max": float(np.max(np.abs(ib["N_dense"](rf) - info["N_dense"](rf)))),
            "dW_rel_max": float(np.max(np.abs(ib["W_field"](rf) - Wa))) / Wmax, "dE_rel": abs(float(ib["E_collar"]) - float(info["E_collar"])) / abs(float(info["E_collar"]))}


def padded_collar_cost(eps_tot, J_tot, g, R, ell, info, pad_ell=8.0, h_frac=1.0 / 16.0):
    """version 12's padded collar cost (engine 18.1) with the integration's success checked before the acceptance (R26a item 5): a failed or
    incomplete integration returns passes False with the solver's message."""
    R = np.asarray(R, float); eps_tot = np.asarray(eps_tot, float); J_tot = np.asarray(J_tot, float)
    a = float(info["r_collar"]); b = a + pad_ell * ell
    rr = np.arange(a, b + 1e-9, ell * h_frac)
    sel = (np.abs(eps_tot) > 0.0) | (np.abs(J_tot) > 0.0)
    K0, K1 = kernels(rr, R[sel], ell)
    rho = K0 @ eps_tot[sel]; Jl = K1 @ J_tot[sel]
    from scipy.interpolate import make_interp_spline
    srho = make_interp_spline(rr, rho, k=5); sJ = make_interp_spline(rr, Jl, k=5)
    Ua = float(info["W_field"](np.array([a]))[0])

    def rhs(r, y):
        U = y[0]
        q = -np.sqrt(U / r) if U > 0.0 else 0.0
        J = float(sJ(r)); rh = float(srho(r))
        if q == 0.0:
            return [8.0 * np.pi * g * r * r * rh, 0.0]
        return [8.0 * np.pi * g * r * r * (rh + q * J), -4.0 * np.pi * g * r * J / q]
    sol = solve_ivp(rhs, (a, b), [Ua, 0.0], method="DOP853", rtol=1e-12, atol=1e-16)
    E = Ua / (2.0 * g)
    if not sol.success or abs(float(sol.t[-1]) - b) > 1e-9 * max(b, 1.0):
        return {"a": a, "b": b, "delta_E": None, "delta_E_rel": None, "lnN_b_at_a": None, "E_at_a": E, "passes": False,
                "failed": "the padded integration failed: %s" % sol.message, "evaluated_at": "the old collar a = %.1f" % a}
    dE = (float(sol.y[0, -1]) - Ua) / (2.0 * g)
    lnNba = -float(sol.y[1, -1])
    return {"a": a, "b": b, "delta_E": dE, "delta_E_rel": dE / E, "lnN_b_at_a": lnNba, "E_at_a": E,
            "passes": bool(abs(dE) < 1e-9 * abs(E) and abs(lnNba) < 1e-9), "evaluated_at": "the old collar a = %.1f" % a}
