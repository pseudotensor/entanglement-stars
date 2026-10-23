#!/usr/bin/env python
"""build_engine_fields_v16e.py (busi, analyst; PERF row RD-2b; REVIEWA_algorithm_redesign_v1 sec. 2.4, the flat-slice port): engine_fields_v16e.py
from engine_fields_v16d.py (md5 7a69faad...) by three marked insertion blocks:
  RD2b-settings  after the RD-2 settings (_RD2): the switch QI2_FIELDS_FLAT (on | off, default on), the grading and chunk parameters, the
                 counter "negative_centre";
  RD2b-entry     in _rd2_collocation, before the line that refuses a centre outside the regular series: with the switch on, that solve goes
                 to _rd2e_negative_centre instead of the ivp fallback;
  RD2b-negcentre before "def _rd2_solve(": the negative-centre collocation (the flat-slice gauge from the centre to the first upward zero r_*
                 of U by the cumulative Gauss quadrature of U' = 8 pi g r^2 rho_ell, the zero by brentq on the panel's collocation polynomial,
                 then the positive branch in t = sqrt(r - r_*) on panels graded toward r_*, Newton on the same banded system) with every info
                 key of v16's solve built by v16's own formulas, the gauge intervals, zeros and labels included.
The route is only reached under QI2_FIELDS_ROUTE=collocation, so with the route unset (or "ivp") no new line runs inside a solve: the default
is v16d's, hence v16c's, arithmetic and control flow exactly. With QI2_FIELDS_FLAT=off the collocation route is v16d's exactly.
Every new line lies between a "# >>> v16e RD2b-<X>" and a "# <<< v16e RD2b-<X>" line (checked); the reverse pass (NEW -> OLD) restores v16d
byte for byte (in memory and on the written file; --reverse --expect); it compiles. The file is never overwritten. Deployment as v16d's: the
content goes into a NEW code dir under the name engine_fields_v16.py (the engines, engine_fields_v17, the kernel cache and the pair workers
import the module by that name); the four rd2 engines (v36nq_rd2, v36r_rd2, v37a_rd2, v37n_rd2) run on it unchanged.
    python build_engine_fields_v16e.py --src .../engine_fields_v16d.py [--dst .../engine_fields_v16e.py] [--check-only]
    python build_engine_fields_v16e.py --reverse --src .../engine_fields_v16e.py --dst .../restored.py --expect .../engine_fields_v16d.py
"""
import argparse
import hashlib
import os

MARK = "v16e RD2b"
SRC_MD5_PREFIX = "7a69faad"

ANCHOR_SETTINGS = ('if _RD2["warm"] and _RD2["route"] != "collocation":\n'
                   '    raise ValueError("engine_fields_v16d: QI2_FIELDS_WARM=on needs QI2_FIELDS_ROUTE=collocation")\n')

SETTINGS = ANCHOR_SETTINGS + r'''# >>> v16e RD2b-settings: engine_fields_v16e = engine_fields_v16d + the negative-centre collocation (build_engine_fields_v16e.py; PERF row
# RD-2b; REVIEWA_algorithm_redesign_v1 sec. 2.4, the flat-slice port). Under QI2_FIELDS_ROUTE=collocation a solve whose centre is outside the
# regular series (rho_0 <= 0, v16's docstring (8): the flat-slice gauge from the centre) is no longer an automatic ivp fallback: the negative
# branch U = U(r_0) + int 8 pi g r^2 rho_ell dr (L constant) is the cumulative Gauss quadrature to its first upward zero r_* (brentq on the
# panel's collocation polynomial; v16's event U = 0 crossed upward), and the positive branch from r_* is the collocation in t = sqrt(r - r_*)
# (U, ln N and W analytic in t at a simple zero, where they carry sqrt(r - r_*) terms in r: the role of v16's s = sqrt(U) stretch), on panels
# graded toward r_* (in r - r_*: doubling from h_u 2^-K up to h_u = ell/2, then uniform), Newton on the same banded system. Still on the ivp
# route (logged): a second zero or a dip below the switch level after r_* (a downward approach), a near-tangency before the switch level
# (U not increasing from r_* to 1e-6 U_scale), U(r_0) >= 0 with rho_0 <= 0, U'(r_*) <= 0, the switch level not reached before the collar,
# and every v16d rule. QI2_FIELDS_FLAT=off: v16d's route exactly (the negative centre falls back). No warm start on this branch (r_* moves the mesh).
_RD2["flat"] = _rd2_env("QI2_FIELDS_FLAT", "on", ("on", "off")) == "on"
_RD2["flat_K"] = 6
_RD2["flat_chunk"] = 32
_RD2["stats"]["negative_centre"] = 0
RD2B_VERSION = "engine_fields_v16e (RD-2b: negative-centre collocation)"
# <<< v16e RD2b-settings
'''

ANCHOR_ENTRY = ('    info["centre_flag"] = None if cc["regular"] else "centre outside the regular series (rho_0 <= 0)"\n'
                '    if not cc["regular"]:\n'
                '        return None, "the centre is outside the regular series (rho_0 <= 0)"\n')

ENTRY = ('    info["centre_flag"] = None if cc["regular"] else "centre outside the regular series (rho_0 <= 0)"\n'
         '    # >>> v16e RD2b-entry: the negative centre by collocation (QI2_FIELDS_FLAT=on, the default under the collocation route)\n'
         '    if not cc["regular"] and _RD2["flat"]:\n'
         '        return _rd2e_negative_centre(src, info, cc, g, R, ell, r_nodes, r_collar, rfine, r0, t_all, tm)\n'
         '    # <<< v16e RD2b-entry\n'
         '    if not cc["regular"]:\n'
         '        return None, "the centre is outside the regular series (rho_0 <= 0)"\n')

ANCHOR_NEG = "\n\ndef _rd2_solve(eps_tot, J_tot, g, R, ell, kw):\n"

NEGCENTRE = r'''# >>> v16e RD2b-negcentre: the negative-centre collocation (see RD2b-settings)
def _rd2e_bary(xs, wb, vals, tau):
    """one value of a panel's collocation polynomial through {0, c_1..c_p, 1} (barycentric Lagrange, as _RD2Panels.evaluate)."""
    d = tau - xs
    hit = np.where(np.abs(d) < 1e-15)[0]
    if len(hit):
        return float(vals[hit[0]])
    t_ = wb / d
    return float(np.dot(t_, vals) / np.sum(t_))


def _rd2e_tmesh(D, ell, ppe, K):
    """the positive branch's panel ends in t = sqrt(r - r_*) on [0, sqrt(D)]: in d = r - r_*, doubling from h_u 2^-K up to h_u = ell/ppe,
    then uniform panels of at most h_u up to D."""
    h_u = ell / float(ppe)
    d = [h_u * 2.0 ** (-k) for k in range(K, 0, -1)]
    d = [x for x in d if x < D]
    if D <= h_u:
        db = np.array([0.0] + d + [D])
    else:
        n_u = max(1, int(np.ceil((D - h_u) / h_u)))
        db = np.concatenate([np.array([0.0] + d), np.linspace(h_u, D, n_u + 1)])
    return np.sqrt(db)


def _rd2e_negative_centre(src, info, cc, g, R, ell, r_nodes, r_collar, rfine, r0, t_all, tm):
    """the collocation solve of a centre outside the regular series (v16's docstring (8)); returns ("ok", (N_sites, v_sites, W_nodes, info))
    or (None, the reason), as _rd2_collocation. v16's construction: on [0, r_*] the flat-slice gauge (L' = 0, U' = 8 pi g r^2 rho_ell), r_*
    the first upward zero of U; beyond it the positive branch U' = f + g1 sqrt(U), L' = 4 pi g r^(3/2) J_ell / sqrt(U), here in t."""
    S = _RD2["stats"]
    fpi = 4.0 * np.pi * g
    rg = np.linspace(0.0, r_collar, 4001)
    U_scale = float(np.sum(0.5 * (np.abs(src.rho(rg[1:])) * rg[1:] ** 2 + np.abs(src.rho(rg[:-1])) * rg[:-1] ** 2) * np.diff(rg))) * 2.0 * fpi
    U_sw = _RD2["eps_switch"] * U_scale
    U0 = cc["a"] * r0 ** 3 + cc["b"] * r0 ** 5
    L0 = cc["L2"] * r0 ** 2 + cc["L4"] * r0 ** 4
    if not U0 < 0.0:
        return None, "negative centre with U(r_0) >= 0 (U(r_0) %.3e; v16 starts on the positive branch there): ivp route" % U0
    p, ppe, K = _RD2["p"], _RD2["ppe"], _RD2["flat_K"]
    q = p
    c, A_, b_g, xs, wb = _rd2_gauss(p)
    # 1. the negative branch to its first upward zero (the quadrature on the standard mesh, chunk by chunk)
    bnd = _rd2_mesh(r0, r_collar, ell, ppe)
    nb = len(bnd) - 1
    r_star = None
    Ua = U0
    k0 = 0
    while k0 < nb and r_star is None:
        k1 = min(nb, k0 + _RD2["flat_chunk"])
        Pc = _RD2Panels(bnd[k0:k1 + 1], p)
        fc = 2.0 * fpi * Pc.r ** 2 * np.asarray(src.rho(Pc.r.ravel()), float).reshape(Pc.r.shape)
        un, ue, us = Pc.cumulative(fc, Ua)
        seq = np.concatenate([un, ue[:, None]], axis=1)
        posk = np.where((seq > 0.0).any(axis=1))[0]
        if len(posk):
            j = int(posk[0])
            vals = np.concatenate([[us[j]], un[j], [ue[j]]])
            m = int(np.where(vals > 0.0)[0][0])
            if m == 0:
                return None, "negative centre: U > 0 at a panel start before any upward crossing (ivp route)"
            fun = lambda tau, _v=vals: _rd2e_bary(xs, wb, _v, tau)
            tau = brentq(fun, float(xs[m - 1]), float(xs[m]), xtol=1e-16, rtol=8.9e-16, maxiter=200)
            r_star = float(Pc.bnd[j] + Pc.h[j] * tau)
        Ua = float(ue[-1])
        k0 = k1
    tm["zero_s"] = _rd2_time.time() - t_all; t0 = _rd2_time.time()
    if r_star is not None and not (r0 < r_star < r_collar):
        return None, "negative centre: the upward zero %.6g outside (r_0, r_collar) (ivp route)" % r_star
    up_star = None
    if r_star is not None:
        up_star = 2.0 * fpi * r_star * r_star * float(src.rho(r_star))
        if not up_star > 0.0:
            return None, "negative centre: U'(r_*) = %.3e <= 0 at the zero r_* = %.6g (not a simple upward crossing; ivp route)" % (up_star, r_star)
    # 2. the negative branch on its own mesh [r_0, r_*] (or to the collar when U < 0 throughout)
    r_end_N = r_star if r_star is not None else r_collar
    bndN = _rd2_mesh(r0, r_end_N, ell, ppe)
    PN = _RD2Panels(bndN, p)
    rhoN = np.asarray(src.rho(PN.r.ravel()), float).reshape(PN.r.shape)
    fN = 2.0 * fpi * PN.r ** 2 * rhoN
    UNn, UNe, UNs = PN.cumulative(fN, U0)
    iters, hist, PP = 0, [], None
    if r_star is not None:
        # 3. the positive branch in t = sqrt(r - r_*): dU/dt = 2t (f + g1 sqrt(U)), U(t = 0) = 0
        tb = _rd2e_tmesh(r_collar - r_star, ell, ppe, K)
        PP = _RD2Panels(tb, p)
        tP = PP.r
        rP = r_star + tP ** 2
        jac = 2.0 * tP
        rhoP = np.asarray(src.rho(rP.ravel()), float).reshape(rP.shape)
        JP = np.asarray(src.J(rP.ravel()), float).reshape(rP.shape)
        fP = jac * 2.0 * fpi * rP ** 2 * rhoP
        g1P = -jac * 2.0 * fpi * rP ** 1.5 * JP
        tm["sources_s"] = _rd2_time.time() - t0; t0 = _rd2_time.time()
        n = PP.n
        Un0, Ue0, _ = PP.cumulative(fP, 0.0)
        y = np.empty(n * (q + 1))
        y.reshape(n, q + 1)[:, :q] = Un0; y.reshape(n, q + 1)[:, q] = Ue0
        y, conv, hist = _rd2_newton(PP, fP, g1P, 0.0, y)
        iters = len(hist)
        S["newton_iterations"] += iters
        tm["newton_s"] = _rd2_time.time() - t0; t0 = _rd2_time.time()
        if not conv:
            return None, "negative centre: Newton did not converge on the positive branch (last steps %s)" % ["%.1e" % s_ for s_ in hist[-3:]]
        if not np.all(np.isfinite(y)):
            return None, "negative centre: a non-finite Newton solution"
        Y = y.reshape(n, q + 1)
        UPn = Y[:, :q].copy(); UPe = Y[:, q].copy(); UPs = np.concatenate([[0.0], UPe[:-1]])
        UPseq = np.concatenate([UPn, UPe[:, None]], axis=1).ravel()
        if float(UPseq.min()) <= 0.0:
            return None, "negative centre: U <= 0 on (r_*, r_collar] after the upward zero (min U/U_scale %.3e): a second zero (ivp route)" % (float(UPseq.min()) / max(U_scale, 1e-300))
        above = np.where(UPseq >= U_sw)[0]
        if len(above) == 0:
            return None, "negative centre: U stays below the switch level 1e-6 U_scale up to the collar (ivp route)"
        if not np.all(np.diff(np.concatenate([[0.0], UPseq[:above[0] + 1]])) > 0.0):
            return None, "negative centre: U not increasing from r_* to the switch level (a near-tangency of the upward stretch; ivp route)"
        if float(UPseq[above[0]:].min()) < U_sw:
            return None, "negative centre: U falls below the switch level 1e-6 U_scale after the upward zero (min U/U_scale %.3e there): a downward approach (ivp route)" % (
                float(UPseq[above[0]:].min()) / max(U_scale, 1e-300))
        GlP = jac * fpi * rP ** 1.5 * JP / np.sqrt(UPn)
        LPn, LPe, LPs = PP.cumulative(GlP, L0)
        lnN_c = float(LPe[-1])
    else:
        tm["sources_s"] = _rd2_time.time() - t0; t0 = _rd2_time.time()
        lnN_c = float(L0)
    xg, wg = _rd2_leg.leggauss(_RD2["centre_quad"])
    rq = 0.5 * r0 * (xg + 1.0); wq = 0.5 * r0 * wg
    Nq = np.exp(cc["L2"] * rq ** 2 + cc["L4"] * rq ** 4 - lnN_c)
    rhoq = np.asarray(src.rho(rq), float)
    W0 = float(np.sum(wq * 2.0 * fpi * rq ** 2 * Nq ** 2 * rhoq))
    NNeg = float(np.exp(L0 - lnN_c))
    WNn, WNe, WNs = PN.cumulative(fN * NNeg ** 2, W0)
    if PP is not None:
        NnP = np.exp(LPn - lnN_c)
        WPn, WPe, WPs = PP.cumulative(fP * NnP ** 2, float(WNe[-1]))
        if not (np.all(np.isfinite(LPn)) and np.all(np.isfinite(WPn))):
            return None, "negative centre: a non-finite lapse or W quadrature"
        W_collar = float(WPe[-1])
    else:
        W_collar = float(WNe[-1])
    if not np.all(np.isfinite(WNn)):
        return None, "negative centre: a non-finite W quadrature"
    tm["quadratures_s"] = _rd2_time.time() - t0; t0 = _rd2_time.time()
    rS = r_star if r_star is not None else np.inf

    def _x(r):
        return np.sqrt(np.maximum(np.minimum(r, r_collar) - r_star, 0.0))

    def U_dense(r):
        r = np.abs(np.atleast_1d(np.asarray(r, float)))
        out = np.empty_like(r)
        lo = r < r0
        out[lo] = cc["a"] * r[lo] ** 3 + cc["b"] * r[lo] ** 5
        mid = (~lo) & (r < rS)
        if mid.any():
            out[mid] = PN.evaluate(UNn, UNs, UNe, np.minimum(r[mid], bndN[-1]))
        hi = (~lo) & (r >= rS)
        if hi.any():
            out[hi] = PP.evaluate(UPn, UPs, UPe, _x(r[hi]))
        return out

    def lnN_dense(r):
        r = np.abs(np.atleast_1d(np.asarray(r, float)))
        out = np.empty_like(r)
        lo = r < r0
        out[lo] = cc["L2"] * r[lo] ** 2 + cc["L4"] * r[lo] ** 4
        mid = (~lo) & (r < rS)
        out[mid] = L0
        hi = (~lo) & (r >= rS)
        if hi.any():
            out[hi] = PP.evaluate(LPn, LPs, LPe, _x(r[hi]))
        return out

    def W_inner(r):
        """W on [0, r_collar] (the argument clipped there): the series integrand's Gauss quadrature below r_0, the collocation above."""
        r = np.minimum(np.abs(np.atleast_1d(np.asarray(r, float))), r_collar)
        w = np.empty_like(r)
        lo = r < r0
        if lo.any():
            rl = r[lo]
            xx = 0.5 * rl[:, None] * (xg[None, :] + 1.0); ww = 0.5 * rl[:, None] * wg[None, :]
            Nx = np.exp(cc["L2"] * xx ** 2 + cc["L4"] * xx ** 4 - lnN_c)
            w[lo] = np.sum(ww * 2.0 * fpi * xx ** 2 * Nx ** 2 * np.asarray(src.rho(xx.ravel()), float).reshape(xx.shape), axis=1)
        mid = (~lo) & (r < rS)
        if mid.any():
            w[mid] = PN.evaluate(WNn, WNs, WNe, np.minimum(r[mid], bndN[-1]))
        hi = (~lo) & (r >= rS)
        if hi.any():
            w[hi] = PP.evaluate(WPn, WPs, WPe, _x(r[hi]))
        return w

    Uf = U_dense(rfine)
    Umax = max(float(np.max(Uf)), 1e-300)
    neg = Uf < -1e-12 * Umax
    zeros = [{"r": r_star, "kind": "up", "U_prime": up_star}] if r_star is not None else []
    nseg = 3 if r_star is not None else 1
    gauge = {"label": "intact", "intervals": [], "min_U_over_U_ref": float(Uf.min() / Umax), "delta_L_bound": 0.0, "lapse_unresolved": False,
             "segments": nseg, "zeros": zeros, "stretches": 1 if r_star is not None else 0, "near_tangencies": 0, "U_scale": U_scale, "U_switch": U_sw}
    ivs = [[0.0, r_star if r_star is not None else r_collar]]
    gauge["intervals"] = ivs

    def dLb(h):
        tot = 0.0
        for (r1, r2) in ivs:
            rm = np.arange(r1, r2 + 1e-12, h)
            if len(rm) < 2:
                continue
            Um = U_dense(rm)
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
    pos = (Uf > 0.0) & (rfine > 0.0)
    regime = {"max_ratio": 0.0, "r_at_max": None, "intervals_above_1": [], "crossings": "located by brentq on the dense output"}

    def ratio_f(x):
        Ux = U_dense(np.array([x]))
        if Ux[0] <= 0.0:
            return -1.0
        return 4.0 * np.pi * g * x * abs(float(src.J(x))) * ell / np.sqrt(Ux[0] / x) - 1.0
    if pos.any():
        rp = rfine[pos]; qp = np.sqrt(Uf[pos] / rp)
        ratio = 4.0 * np.pi * g * rp * np.abs(src.J(rp)) * ell / qp
        k = int(np.argmax(ratio))
        regime["max_ratio"] = float(ratio[k]); regime["r_at_max"] = float(rp[k])
        above1 = np.zeros(len(rfine), bool); above1[np.where(pos)[0][ratio > 1.0]] = True
        ivr = []
        for (a_, b_) in _intervals(above1, rfine):
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
        return np.where(inside, np.exp(lnN_dense(np.minimum(x, r_collar)) - _c), 1.0)
    info["it"] = 1
    info["converged"] = True
    info["hist"] = []
    info["ivp"] = {"r0": r0, "U0": float(U0), "lnN0": float(L0), "lnN_collar": lnN_c, "N_centre": float(np.exp(-lnN_c)), "nfev": 0, "segments": nseg,
                   "series": {k_: cc[k_] for k_ in cc}, "route": "collocation"}
    info["last_dN"], info["last_dW"] = 0.0, 0.0
    Wf = W_inner(rfine)
    Wmax = max(float(np.max(np.abs(Wf))), 1e-300)
    if Wf.min() < -1e-12 * Wmax:
        info["W_negative"] = {"min_W_over_max": float(Wf.min() / Wmax), "r": float(rfine[int(np.argmin(Wf))]), "inside_gauge_intervals": bool(neg.any())}

    def W_geom_f(r, _ivs=gauge["intervals"]):
        r = np.abs(np.atleast_1d(np.asarray(r, float)))
        w = np.asarray(W_inner(np.minimum(r, r_collar)), float)
        for (a_, b_) in _ivs:
            w = np.where((r >= a_) & (r <= b_), 0.0, w)
        return np.maximum(w, 0.0)
    info["W_geom"] = W_geom_f
    info["W_aux_signed_label"] = "auxiliary signed U (the energy deficit's bookkeeping)"
    E = float(W_collar / (2.0 * g))
    _gE = g * E

    def W_field(r):
        r = np.abs(np.atleast_1d(np.asarray(r, float)))
        out = np.full(r.shape, 2.0 * _gE)
        m_ = r < r_collar
        if m_.any():
            out[m_] = W_inner(r[m_])
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
    W0c = float(W_field(rc)[0]); ser = 8.0 * np.pi * g / 3.0 * float(N_dense_f(0.0)[0]) ** 2 * float(src.rho(0.0)) * rc ** 3
    info["centre_series_ratio"] = (W0c / ser) if ser != 0 else None
    info["W_roundoff_negative_points"] = roundoff_negW
    tm["diagnostics_s"] = _rd2_time.time() - t0; t0 = _rd2_time.time()
    if r_collar < r_nodes[-1]:
        xt, wt = _rd2_leg.leggauss(64)
        n_t = max(1, int(np.ceil((r_nodes[-1] - r_collar) / ell)))
        et = np.linspace(r_collar, r_nodes[-1], n_t + 1)
        Wt = 2.0 * _gE
        for a_, b_ in zip(et[:-1], et[1:]):
            xx = 0.5 * (b_ - a_) * (xt + 1.0) + a_; ww = 0.5 * (b_ - a_) * wt
            Wt += float(np.sum(ww * 8.0 * np.pi * g * xx ** 2 * np.asarray(src.rho(xx), float)))
        info["W_tail"] = {"r_end": float(r_nodes[-1]), "E_tail_end": float(Wt / (2.0 * g)), "E_collar": E, "flagged": False}
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
    info["U_dense"] = U_dense
    info.update({"N_dense": N_dense_f, "v_dense": v_dense_f, "W_dense": W_field, "W_field": W_field, "rho_ell": src.rho, "J_ell": src.J,
                 "W_nodes": W_nodes, "N_nodes": N_nodes, "v_nodes": v_nodes, "E_collar": E, "N_min": float(N_sites.min()),
                 "spectrum": lattice_scale_power(N_sites - 1.0, v_sites, R, ell, float(N_dense_f(0.0)[0]) - 1.0)})
    tm["outputs_s"] = _rd2_time.time() - t0
    if not (np.all(np.isfinite(N_sites)) and np.all(np.isfinite(v_sites)) and np.isfinite(E)):
        return None, "negative centre: a non-finite site field or charge"
    S["negative_centre"] += 1
    U_min_sw = (float(UPseq[above[0]:].min()) / max(U_sw, 1e-300)) if PP is not None else None
    info["fields_route"] = {"route": "collocation", "branch": "negative centre (RD-2b): the flat-slice gauge on [0, r_*] by quadrature, the positive branch in t = sqrt(r - r_*)",
                            "r_star": r_star, "U_prime_at_r_star": up_star, "warm_start": False, "newton_iterations": iters, "newton_steps": hist,
                            "panels": PN.n + (PP.n if PP is not None else 0), "panels_negative": PN.n, "panels_positive": PP.n if PP is not None else 0,
                            "unknowns": (PP.n * (q + 1)) if PP is not None else 0, "p": p, "panels_per_ell": ppe, "grading_K": K,
                            "U_min_over_U_switch": U_min_sw, "seconds": _rd2_time.time() - t_all, "timing": tm}
    return "ok", (N_sites, v_sites, W_nodes, info)
# <<< v16e RD2b-negcentre'''


def edits():
    return [("RD2b-settings", ANCHOR_SETTINGS, SETTINGS),
            ("RD2b-entry", ANCHOR_ENTRY, ENTRY),
            ("RD2b-negcentre", ANCHOR_NEG, "\n\n" + NEGCENTRE + "\n" + ANCHOR_NEG)]


def md5(b):
    return hashlib.md5(b).hexdigest()


def forward(text, E):
    for name, old, new in E:
        c = text.count(old)
        if c != 1:
            raise SystemExit("edit %s: the OLD anchor occurs %d times (must be 1)" % (name, c))
        text = text.replace(old, new)
    return text


def reverse(text, E):
    for name, old, new in reversed(E):
        c = text.count(new)
        if c != 1:
            raise SystemExit("reverse %s: the NEW block occurs %d times (must be 1)" % (name, c))
        text = text.replace(new, old)
    return text


def marker_check(E):
    """every line a NEW block adds (a line not in its OLD anchor) lies between a '# >>> v16e RD2b-<X>' and a '# <<< v16e RD2b-<X>' line."""
    bad, n_new = [], 0
    for name, old, new in E:
        olds = set(old.splitlines())
        inside = False
        for ln in new.splitlines():
            s = ln.strip()
            if s.startswith("# >>> " + MARK):
                if inside:
                    bad.append((name, "<nested block>"))
                inside = True; n_new += 1; continue
            if s.startswith("# <<< " + MARK):
                inside = False; n_new += 1; continue
            if inside:
                n_new += 1; continue
            if s == "" or ln in olds:
                continue
            bad.append((name, ln))
        if inside:
            bad.append((name, "<block not closed>"))
    return bad, n_new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="engine_fields_v16d.py (build) or engine_fields_v16e.py (--reverse)")
    ap.add_argument("--dst", default=None)
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--expect", default=None)
    ap.add_argument("--expect-src-md5", default=SRC_MD5_PREFIX)
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    E = edits()
    src = open(a.src, "rb").read()
    text = src.decode("utf-8")
    if a.reverse:
        out = reverse(text, E).encode("utf-8")
        if a.dst and not a.check_only:
            if os.path.exists(a.dst):
                raise SystemExit("%s exists; nothing written (never overwritten)" % a.dst)
            open(a.dst, "wb").write(out)
        print("reverse: %s %s (md5 %s)" % ("would write" if (a.check_only or not a.dst) else "wrote", a.dst, md5(out)))
        if a.expect:
            ok = out == open(a.expect, "rb").read()
            print("restored equals %s byte for byte: %s" % (a.expect, ok))
            raise SystemExit(0 if ok else 1)
        return
    print("source %s: md5 %s, %d bytes" % (a.src, md5(src), len(src)))
    if a.expect_src_md5 and not md5(src).startswith(a.expect_src_md5):
        raise SystemExit("the source's md5 does not start with %s: not engine_fields_v16d.py; nothing written" % a.expect_src_md5)
    if MARK in text:
        raise SystemExit("the source already carries %r; nothing written" % MARK)
    for name, old, new in E:
        print("  anchor %-15s occurs %d time(s) in the input (%d -> %d lines)" % (name, text.count(old), len(old.splitlines()), len(new.splitlines())))
    bad, n_new = marker_check(E)
    if bad:
        for name, ln in bad:
            print("  UNMARKED new line in %s: %r" % (name, ln[:160]))
        raise SystemExit("new lines outside the marked blocks")
    print("every new line lies in a '# >>> %s-X' ... '# <<< %s-X' block (%d lines incl. the markers)" % (MARK, MARK, n_new))
    built = forward(text, E)
    out = built.encode("utf-8")
    back = reverse(built, E).encode("utf-8")
    print("reversible (NEW -> OLD restores the input byte for byte): %s" % (back == src))
    if back != src:
        raise SystemExit("the reverse pass does not restore the input; nothing written")
    try:
        compile(built, a.dst or "engine_fields_v16e.py", "exec")
    except SyntaxError as e:
        raise SystemExit("the result does not compile (%s); nothing written" % e)
    if a.dst is None:
        a.dst = os.path.join(os.path.dirname(os.path.abspath(a.src)), "engine_fields_v16e.py")
    if a.check_only:
        print("check only: %s would be %d bytes, md5 %s" % (a.dst, len(out), md5(out)))
        return
    if os.path.exists(a.dst):
        if open(a.dst, "rb").read() == out:
            print("%s exists and equals the build byte for byte (md5 %s); not rewritten" % (a.dst, md5(out)))
            return
        raise SystemExit("%s exists and differs from the build; nothing written (never overwritten)" % a.dst)
    open(a.dst, "wb").write(out)
    written = open(a.dst, "rb").read()
    if written != out or reverse(written.decode("utf-8"), E).encode("utf-8") != src:
        raise SystemExit("the written file does not reverse to the input")
    compile(written.decode("utf-8"), a.dst, "exec")
    print("wrote %s: %d bytes, md5 %s; compiles; the reverse of the written file equals the source byte for byte" % (a.dst, len(out), md5(out)))


if __name__ == "__main__":
    main()
