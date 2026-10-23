#!/usr/bin/env python
"""build_engine_fields_v16d.py (busi, analyst; PERF rows RD-2 and RD-5; REVIEWA_algorithm_redesign_v1 sec. 2 and 6): engine_fields_v16d.py from
engine_fields_v16c.py (md5 fa84dad3...) by two marked insertion blocks:
  RD2-colloc   before "def solve_constraints_ivp(": the collocation route (fast_fields_v1 0f9d2b24's method, ported to return every key of v16's info),
               the RD-5 warm start, the settings (_RD2, read from QI2_FIELDS_ROUTE / QI2_FIELDS_WARM at import; rd2_configure, rd2_stats);
  RD2-dispatch after solve_constraints_ivp's docstring: when the route is "collocation" (never by default) the call goes to _rd2_solve, which
               returns the collocation result or falls back to the unchanged ivp code below (one printed line per fallback).
With QI2_FIELDS_ROUTE unset (or "ivp") no line of either block runs inside a solve: the default is v16c's arithmetic and control flow exactly.
Every new line lies between a "# >>> v16d RD2-<X>" and a "# <<< v16d RD2-<X>" line (checked); the reverse pass (NEW -> OLD) restores v16c byte
for byte (in memory and on the written file; --reverse --expect); it compiles. The file is never overwritten. The name: engine_fields_v17 and
v18 exist (the empty-interior rule and its fallback), so this is v16d, deployed like v16c under the name engine_fields_v16.py in a NEW code dir.
    python build_engine_fields_v16d.py --src .../engine_fields_v16c.py [--dst .../engine_fields_v16d.py] [--check-only]
    python build_engine_fields_v16d.py --reverse --src .../engine_fields_v16d.py --dst .../restored.py --expect .../engine_fields_v16c.py
"""
import argparse
import hashlib
import os

MARK = "v16d RD2"
SRC_MD5_PREFIX = "fa84dad3"

COLLOC = r'''# >>> v16d RD2-colloc: engine_fields_v16d = engine_fields_v16c + the RD-2 collocation route and the RD-5 warm start (build_engine_fields_v16d.py;
# REVIEWA_algorithm_redesign_v1 sec. 2 and 6; PERF rows RD-2, RD-5). OFF by default: with QI2_FIELDS_ROUTE unset or "ivp" nothing in the RD2 blocks
# runs inside a solve and every solve is v16c's. QI2_FIELDS_ROUTE=collocation (or rd2_configure("collocation", warm), which also sets the variable
# for spawned workers): a solve at the engines' tolerance (rtol 1e-12, atol 1e-16 exactly) is the global Gauss-Legendre collocation of
# fast_fields_v1 (0f9d2b24): U' = f + g1 sqrt(U), f = 8 pi g r^2 rho_ell, g1 = -8 pi g r^(3/2) J_ell, on panels from r_0 = ell r0_frac (geometric
# doubling, then ell/2), p = 8 Gauss nodes per panel, Newton on the banded global system (scipy solve_banded) to a relative step below 1e-14;
# ln N and W by the same cumulative quadrature; every dense value by the panel's collocation polynomial; info carries every key v16's solve
# returns, by v16's own formulas on these dense functions. Any other tolerance (the indicator's rtol 1e-10 re-solve) takes the ivp route.
# Automatic fallback to the ivp route (one printed line each): the centre outside the regular series; the collar margin missing; U <= 0 at a
# node or panel end; U below the production's switch level 1e-6 U_scale after first reaching it (where v16 leaves the r-integration: the
# flat-slice continuation stays on the ivp route); Newton not converged in 30 iterations; a non-finite value; any exception in the route.
# QI2_FIELDS_WARM=on (RD-5): Newton starts from the latest converged solve on the same mesh (the last 4 meshes kept; else, and on a failed
# warm Newton, from the quadrature of f alone). Counters in _RD2["stats"] (rd2_stats()). The fallback re-entry flag is a module global: solves on one thread only.
import time as _rd2_time
import scipy.linalg as _rd2_sla
from numpy.polynomial import legendre as _rd2_leg
from numpy.polynomial import polynomial as _rd2_poly


def _rd2_env(name, default, allowed):
    val = os.environ.get(name, default)
    if val not in allowed:
        raise ValueError("engine_fields_v16d: %s=%r is not one of %s" % (name, val, allowed))
    return val


_RD2 = {"route": _rd2_env("QI2_FIELDS_ROUTE", "ivp", ("ivp", "collocation")), "warm": _rd2_env("QI2_FIELDS_WARM", "off", ("off", "on")) == "on",
        "inner": False, "p": 8, "ppe": 2, "newton_tol": 1e-14, "newton_max": 30, "tol": (1e-12, 1e-16), "eps_switch": 1e-6, "centre_quad": 24,
        "last": None, "gl": {},
        "stats": {"collocation": 0, "fallback": 0, "ivp_by_tolerance": 0, "warm_used": 0, "warm_retry": 0, "newton_iterations": 0,
                  "collocation_s": 0.0, "ivp_s": 0.0, "fallback_reasons": {}}}
if _RD2["warm"] and _RD2["route"] != "collocation":
    raise ValueError("engine_fields_v16d: QI2_FIELDS_WARM=on needs QI2_FIELDS_ROUTE=collocation")


def rd2_configure(route, warm="off"):
    """the RD-2 route ("ivp" | "collocation") and the RD-5 warm start ("off" | "on") in this process and in os.environ (spawned workers read
    the variables when they import the module)."""
    if route not in ("ivp", "collocation") or warm not in ("off", "on") or (warm == "on" and route != "collocation"):
        raise ValueError("rd2_configure(%r, %r): route ivp|collocation, warm off|on (on needs collocation)" % (route, warm))
    os.environ["QI2_FIELDS_ROUTE"] = route
    os.environ["QI2_FIELDS_WARM"] = warm
    _RD2["route"] = route
    _RD2["warm"] = warm == "on"
    _RD2["last"] = None


def rd2_stats():
    s = dict(_RD2["stats"])
    s["fallback_reasons"] = dict(s["fallback_reasons"])
    return s


def _rd2_gauss(p):
    """fast_fields_v1.gauss_colloc: nodes c in (0, 1), the cumulative integration matrix A, the end weights b, the interpolation points
    {0, c_1..c_p, 1} and their barycentric weights (cached per p)."""
    got = _RD2["gl"].get(p)
    if got is not None:
        return got
    x, w = _rd2_leg.leggauss(p)
    c = 0.5 * (x + 1.0)
    A = np.empty((p, p)); b = np.empty(p)
    for l in range(p):
        roots = np.delete(c, l)
        coef = _rd2_poly.polyfromroots(roots)
        coef = coef / _rd2_poly.polyval(c[l], coef)
        ic = _rd2_poly.polyint(coef)
        A[:, l] = _rd2_poly.polyval(c, ic)
        b[l] = _rd2_poly.polyval(1.0, ic)
    xs = np.concatenate([[0.0], c, [1.0]])
    wb = np.empty(p + 2)
    for k in range(p + 2):
        wb[k] = 1.0 / np.prod(xs[k] - np.delete(xs, k))
    got = (c, A, b, xs, wb)
    _RD2["gl"][p] = got
    return got


def _rd2_mesh(r0, r_collar, ell, panels_per_ell):
    """fast_fields_v1.make_mesh: geometric doubling from r_0 up to the uniform width ell/panels_per_ell, then uniform to r_collar."""
    h_u = ell / float(panels_per_ell)
    bnd = [r0]
    r = r0
    while 2.0 * r - bnd[-1] < h_u and 2.0 * r < r_collar:
        r = 2.0 * r
        bnd.append(r)
    n_u = max(1, int(np.ceil((r_collar - bnd[-1]) / h_u)))
    return np.concatenate([np.array(bnd[:-1]), np.linspace(bnd[-1], r_collar, n_u + 1)])


class _RD2Panels:
    """fast_fields_v1.Panels: the mesh, the node positions and the panel-wise collocation operators."""

    def __init__(self, bnd, p):
        self.bnd = np.asarray(bnd, float)
        self.n = len(bnd) - 1
        self.h = np.diff(self.bnd)
        self.c, self.A, self.b, self.xs, self.wb = _rd2_gauss(p)
        self.p = p
        self.r = self.bnd[:-1, None] + self.h[:, None] * self.c[None, :]
        self._wcache = {}

    def cumulative(self, F, y0):
        inc = self.h * (F @ self.b)
        yend = y0 + np.cumsum(inc)
        ystart = np.concatenate([[y0], yend[:-1]])
        ynode = ystart[:, None] + self.h[:, None] * (F @ self.A.T)
        return ynode, yend, ystart

    def evaluate(self, ynode, ystart, yend, r):
        """fast_fields_v1.Panels.evaluate (the same arithmetic); the panel index, the barycentric terms and their sums are kept per point set
        (keyed by the points' bytes, 16 sets per solve), since U, ln N and W are evaluated on the same points."""
        r = np.asarray(r, float)
        key = r.tobytes()
        got = self._wcache.get(key)
        if got is None:
            i = np.clip(np.searchsorted(self.bnd, r, side="right") - 1, 0, self.n - 1)
            tau = (r - self.bnd[i]) / self.h[i]
            d = tau[:, None] - self.xs[None, :]
            exact = np.abs(d) < 1e-15
            with np.errstate(divide="ignore", invalid="ignore"):
                terms = self.wb[None, :] / d
                tsum = terms.sum(axis=1)
            hit = exact.any(axis=1)
            kh = np.argmax(exact[hit], axis=1) if hit.any() else None
            got = (i, terms, tsum, hit, kh)
            if len(self._wcache) >= 16:
                self._wcache.pop(next(iter(self._wcache)))
            self._wcache[key] = got
        i, terms, tsum, hit, kh = got
        vals = np.concatenate([ystart[i][:, None], ynode[i], yend[i][:, None]], axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = (terms * vals).sum(axis=1) / tsum
        if kh is not None:
            out[hit] = vals[hit, kh]
        return out


def _rd2_newton(P, f, g1, U0, y):
    """fast_fields_v1's Newton loop on U at the nodes and panel ends: the banded Jacobian (lower bandwidth p + 1, upper p), every position
    written once; returns (y, converged, the relative steps)."""
    n, q = P.n, P.p
    N_unk = n * (q + 1)
    rows_blk = np.arange(q + 1)
    hA = np.concatenate([P.h[:, None, None] * P.A[None, :, :], P.h[:, None, None] * P.b[None, None, :]], axis=1)
    ab = np.zeros((2 * q + 2, N_unk))
    base = (np.arange(n) * (q + 1))[:, None]
    rr_ = base + rows_blk[None, :]
    hist = []
    for _it in range(_RD2["newton_max"]):
        Y = y.reshape(n, q + 1)
        Un = Y[:, :q]; Ue = Y[:, q]
        Us = np.concatenate([[U0], Ue[:-1]])
        sq = np.sqrt(np.maximum(Un, 0.0))
        F = f + g1 * sq
        dF = np.where(Un > 0.0, g1 / (2.0 * np.maximum(sq, 1e-300)), 0.0)
        res = Y - Us[:, None] - np.einsum("ijl,il->ij", hA, F)
        ab.fill(0.0)
        ab[q, base[:, 0] + q] = 1.0
        for l in range(q):
            cidx = base[:, 0] + l
            vals = -hA[:, :, l] * dF[:, l][:, None]
            vals[:, l] += 1.0
            c_g = np.broadcast_to(cidx[:, None], rr_.shape)
            ab[q + rr_ - c_g, c_g] = vals
        cprev = np.broadcast_to((base[1:, 0] - 1)[:, None], rr_[1:].shape)
        ab[q + rr_[1:] - cprev, cprev] = -1.0
        dy = _rd2_sla.solve_banded((q + 1, q), ab, -res.ravel(), check_finite=False)
        y = y + dy
        step = float(np.max(np.abs(dy))) / max(float(np.max(np.abs(y))), 1e-300)
        hist.append(step)
        if not np.isfinite(step):
            return y, False, hist
        if step < _RD2["newton_tol"]:
            return y, True, hist
    return y, False, hist


def _rd2_collocation(eps_tot, J_tot, g, R, ell, rtol, atol, collar_policy, support_rel, r0_frac):
    """the collocation solve with v16's interface and info; returns ("ok", (N_sites, v_sites, W_nodes, info)) or (None, the reason)."""
    t_all = _rd2_time.time(); tm = {}
    R = np.asarray(R, float)
    r_nodes = coarse_grid(R, ell)
    src = SmoothedSources5(eps_tot, J_tot, R, ell, r_nodes[-1], support_rel)
    info = {"it": 0, "converged": False, "ell": ell, "r_nodes": r_nodes, "R_last": src.R_last, "failed": None, "dictionary": DICTIONARY,
            "site_fields": "dense", "source_spline_error": src.spline_error, "collar_policy": collar_policy, "support_rel": support_rel,
            "charge_conservation_quadrature": (src.total_eps_quadrature / src.total_eps - 1.0) if src.total_eps else 0.0,
            "hist": [], "src": src, "rtol": rtol, "atol": atol, "module": "engine_fields_v16"}
    i_need = int(np.ceil((src.R_last + 8.0 * ell) / ell))
    if i_need > len(r_nodes) - 1:
        if collar_policy == "fail":
            return None, "the collar margin unavailable (collar_policy fail)"
        last8 = R > r_nodes[-1] - 8.0 * ell
        tot_e = max(float(np.abs(np.asarray(eps_tot, float)).sum()), 1e-300)
        info["collar_margin"] = {"needed_node": i_need, "last_node": len(r_nodes) - 1, "deficit_ell": i_need - (len(r_nodes) - 1),
                                 "eps_share_last_8ell": float(np.abs(np.asarray(eps_tot, float)[last8]).sum()) / tot_e,
                                 "J_abs_last_8ell": float(np.abs(np.asarray(J_tot, float)[last8]).sum())}
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
    info["centre_flag"] = None if cc["regular"] else "centre outside the regular series (rho_0 <= 0)"
    if not cc["regular"]:
        return None, "the centre is outside the regular series (rho_0 <= 0)"
    fpi = 4.0 * np.pi * g
    rg = np.linspace(0.0, r_collar, 4001)
    U_scale = float(np.sum(0.5 * (np.abs(src.rho(rg[1:])) * rg[1:] ** 2 + np.abs(src.rho(rg[:-1])) * rg[:-1] ** 2) * np.diff(rg))) * 2.0 * fpi
    U_sw = _RD2["eps_switch"] * U_scale
    U0 = cc["a"] * r0 ** 3 + cc["b"] * r0 ** 5
    L0 = cc["L2"] * r0 ** 2 + cc["L4"] * r0 ** 4
    p, ppe = _RD2["p"], _RD2["ppe"]
    bnd = _rd2_mesh(r0, r_collar, ell, ppe)
    P = _RD2Panels(bnd, p)
    rr = P.r.ravel()
    rho = np.asarray(src.rho(rr), float).reshape(P.r.shape); J = np.asarray(src.J(rr), float).reshape(P.r.shape)
    f = 2.0 * fpi * P.r ** 2 * rho
    g1 = -2.0 * fpi * P.r ** 1.5 * J
    tm["sources_s"] = _rd2_time.time() - t_all; t0 = _rd2_time.time()
    n, q = P.n, p
    key = (p, ppe, bnd.tobytes())
    warm = False
    last = _RD2["last"] or {}
    if _RD2["warm"] and key in last:
        y = last[key].copy(); warm = True
    else:
        Unode0, Uend0, _ = P.cumulative(f, U0)
        y = np.empty(n * (q + 1))
        y.reshape(n, q + 1)[:, :q] = Unode0; y.reshape(n, q + 1)[:, q] = Uend0
    y, conv, hist = _rd2_newton(P, f, g1, U0, y)
    iters = len(hist)
    if warm and not conv:
        _RD2["stats"]["warm_retry"] += 1
        Unode0, Uend0, _ = P.cumulative(f, U0)
        y = np.empty(n * (q + 1))
        y.reshape(n, q + 1)[:, :q] = Unode0; y.reshape(n, q + 1)[:, q] = Uend0
        y, conv, hist2 = _rd2_newton(P, f, g1, U0, y)
        iters += len(hist2); hist = hist + ["cold"] + hist2; warm = False
    _RD2["stats"]["newton_iterations"] += iters
    tm["newton_s"] = _rd2_time.time() - t0; t0 = _rd2_time.time()
    if not conv:
        return None, "Newton did not converge (last steps %s)" % [("%.1e" % s_ if isinstance(s_, float) else s_) for s_ in hist[-3:]]
    if not np.all(np.isfinite(y)):
        return None, "a non-finite Newton solution"
    Y = y.reshape(n, q + 1); Unode = Y[:, :q].copy(); Uend = Y[:, q].copy(); Ustart = np.concatenate([[U0], Uend[:-1]])
    Useq = np.concatenate([np.concatenate([Unode, Uend[:, None]], axis=1).ravel()])
    if float(Useq.min()) <= 0.0:
        return None, "U <= 0 on (r_0, r_collar] (min U/U_scale %.3e): the flat-slice gauge continuation (ivp route)" % (float(Useq.min()) / max(U_scale, 1e-300))
    above = np.where(Useq >= U_sw)[0]
    if len(above) == 0 or float(Useq[above[0]:].min()) < U_sw:
        return None, "U falls below the switch level 1e-6 U_scale after reaching it (min U/U_scale %.3e there): the stretch region (ivp route)" % (
            (float(Useq[above[0]:].min()) if len(above) else float(Useq.max())) / max(U_scale, 1e-300))
    if _RD2["warm"]:
        last = _RD2["last"] if _RD2["last"] is not None else {}
        last.pop(key, None)
        last[key] = y.copy()
        while len(last) > 4:
            last.pop(next(iter(last)))
        _RD2["last"] = last
    Gl = fpi * P.r ** 1.5 * J / np.sqrt(Unode)
    Lnode, Lend, Lstart = P.cumulative(Gl, L0)
    lnN_c = float(Lend[-1])
    Nn = np.exp(Lnode - lnN_c)
    xg, wg = _rd2_leg.leggauss(_RD2["centre_quad"])
    rq = 0.5 * r0 * (xg + 1.0); wq = 0.5 * r0 * wg
    Nq = np.exp(cc["L2"] * rq ** 2 + cc["L4"] * rq ** 4 - lnN_c)
    rhoq = np.asarray(src.rho(rq), float)
    W0 = float(np.sum(wq * 2.0 * fpi * rq ** 2 * Nq ** 2 * rhoq))
    Hw = 2.0 * fpi * P.r ** 2 * Nn ** 2 * rho
    Wnode, Wend, Wstart = P.cumulative(Hw, W0)
    if not (np.all(np.isfinite(Lnode)) and np.all(np.isfinite(Wnode))):
        return None, "a non-finite lapse or W quadrature"
    tm["quadratures_s"] = _rd2_time.time() - t0; t0 = _rd2_time.time()

    def U_dense(r):
        r = np.abs(np.atleast_1d(np.asarray(r, float)))
        out = np.empty_like(r)
        lo = r < r0
        out[lo] = cc["a"] * r[lo] ** 3 + cc["b"] * r[lo] ** 5
        hi = ~lo
        if hi.any():
            out[hi] = P.evaluate(Unode, Ustart, Uend, np.minimum(r[hi], r_collar))
        return out

    def lnN_dense(r):
        r = np.abs(np.atleast_1d(np.asarray(r, float)))
        out = np.empty_like(r)
        lo = r < r0
        out[lo] = cc["L2"] * r[lo] ** 2 + cc["L4"] * r[lo] ** 4
        hi = ~lo
        if hi.any():
            out[hi] = P.evaluate(Lnode, Lstart, Lend, np.minimum(r[hi], r_collar))
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
        hi = ~lo
        if hi.any():
            w[hi] = P.evaluate(Wnode, Wstart, Wend, r[hi])
        return w

    Uf = U_dense(rfine)
    Umax = max(float(np.max(Uf)), 1e-300)
    neg = Uf < -1e-12 * Umax
    gauge = {"label": "intact", "intervals": [], "min_U_over_U_ref": float(Uf.min() / Umax), "delta_L_bound": 0.0, "lapse_unresolved": False,
             "segments": 1, "zeros": [], "stretches": 0, "near_tangencies": 0, "U_scale": U_scale, "U_switch": U_sw}
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
    info["ivp"] = {"r0": r0, "U0": float(U0), "lnN0": float(L0), "lnN_collar": lnN_c, "N_centre": float(np.exp(-lnN_c)), "nfev": 0, "segments": 1,
                   "series": {k_: cc[k_] for k_ in cc}, "route": "collocation"}
    info["last_dN"], info["last_dW"] = 0.0, 0.0
    Wf = W_inner(rfine)
    Wmax = max(float(np.max(np.abs(Wf))), 1e-300)
    if Wf.min() < -1e-12 * Wmax:
        info["W_negative"] = {"min_W_over_max": float(Wf.min() / Wmax), "r": float(rfine[int(np.argmin(Wf))]), "inside_gauge_intervals": bool(neg.any())}

    def W_geom_f(r):
        r = np.abs(np.atleast_1d(np.asarray(r, float)))
        return np.maximum(np.asarray(W_inner(np.minimum(r, r_collar)), float), 0.0)
    info["W_geom"] = W_geom_f
    info["W_aux_signed_label"] = "auxiliary signed U (the energy deficit's bookkeeping)"
    E = float(Wend[-1] / (2.0 * g))
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
        return None, "a non-finite site field or charge"
    info["fields_route"] = {"route": "collocation", "warm_start": warm, "newton_iterations": iters, "newton_steps": hist, "panels": n,
                            "unknowns": n * (q + 1), "p": p, "panels_per_ell": ppe, "U_min_over_U_switch": float(Useq[above[0]:].min()) / max(U_sw, 1e-300),
                            "seconds": _rd2_time.time() - t_all, "timing": tm}
    return "ok", (N_sites, v_sites, W_nodes, info)


def _rd2_solve(eps_tot, J_tot, g, R, ell, kw):
    """the dispatch under QI2_FIELDS_ROUTE=collocation: the collocation result, or the ivp route (the unchanged code of solve_constraints_ivp,
    re-entered with the dispatch bypassed) with info["fields_route"] naming why."""
    S = _RD2["stats"]
    t0 = _rd2_time.time()
    if (kw["rtol"], kw["atol"]) != _RD2["tol"]:
        S["ivp_by_tolerance"] += 1
        reason = "the call's tolerance (rtol %g, atol %g) is not the route's class (rtol 1e-12, atol 1e-16)" % (kw["rtol"], kw["atol"])
        print("v16d RD2 ivp by tolerance: %s" % reason, flush=True)
        kind = "tolerance"
    else:
        try:
            st, out = _rd2_collocation(eps_tot, J_tot, g, R, ell, kw["rtol"], kw["atol"], kw["collar_policy"], kw["support_rel"], kw["r0_frac"])
        except Exception as ex:
            st, out = None, "the collocation route raised %s: %s" % (type(ex).__name__, str(ex)[:300])
        if st == "ok":
            S["collocation"] += 1
            S["collocation_s"] += _rd2_time.time() - t0
            if out[3]["fields_route"]["warm_start"]:
                S["warm_used"] += 1
            return out
        reason = out
        S["fallback"] += 1
        cls = reason.split(" (")[0][:80]
        S["fallback_reasons"][cls] = S["fallback_reasons"].get(cls, 0) + 1
        print("v16d RD2 fallback: collocation -> ivp: %s" % reason, flush=True)
        kind = "fallback"
    _RD2["inner"] = True
    t1 = _rd2_time.time()
    try:
        res = solve_constraints_ivp(eps_tot, J_tot, g, R, ell, **kw)
    finally:
        _RD2["inner"] = False
    S["ivp_s"] += _rd2_time.time() - t1
    res[3]["fields_route"] = {"route": "ivp", "why": kind, "reason": reason}
    return res
# <<< v16d RD2-colloc


def solve_constraints_ivp('''

DISPATCH = r'''    """version 14 (docstring): the same interface and returned diagnostics as version 12's, plus info["regime"], info["gauge"],
    info["centre_series_check"] and the series coefficients in info["ivp"]."""
    # >>> v16d RD2-dispatch: the collocation route only when QI2_FIELDS_ROUTE=collocation (never by default); _rd2_solve falls back to the code below
    if _RD2["route"] == "collocation" and not _RD2["inner"]:
        return _rd2_solve(eps_tot, J_tot, g, R, ell, dict(N0=N0, tol=tol, maxit=maxit, rtol=rtol, atol=atol, collar_policy=collar_policy, support_rel=support_rel,
                                                          site_fields=site_fields, N0_func=N0_func, on_maxit=on_maxit, stall_cap=stall_cap, r0_frac=r0_frac))
    # <<< v16d RD2-dispatch
'''


def edits():
    old1 = "\n\ndef solve_constraints_ivp("
    new1 = "\n\n" + COLLOC
    old2 = ('    """version 14 (docstring): the same interface and returned diagnostics as version 12\'s, plus info["regime"], info["gauge"],\n'
            '    info["centre_series_check"] and the series coefficients in info["ivp"]."""\n')
    return [("RD2-colloc", old1, new1), ("RD2-dispatch", old2, DISPATCH)]


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
    """every line a NEW block adds (a line not in its OLD anchor) lies between a '# >>> v16d RD2-<X>' and a '# <<< v16d RD2-<X>' line."""
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
    ap.add_argument("--src", required=True, help="engine_fields_v16c.py (build) or engine_fields_v16d.py (--reverse)")
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
        raise SystemExit("the source's md5 does not start with %s: not engine_fields_v16c.py; nothing written" % a.expect_src_md5)
    if MARK in text:
        raise SystemExit("the source already carries %r; nothing written" % MARK)
    for name, old, new in E:
        print("  anchor %-13s occurs %d time(s) in the input (%d -> %d lines)" % (name, text.count(old), len(old.splitlines()), len(new.splitlines())))
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
        compile(built, a.dst or "engine_fields_v16d.py", "exec")
    except SyntaxError as e:
        raise SystemExit("the result does not compile (%s); nothing written" % e)
    if a.dst is None:
        a.dst = os.path.join(os.path.dirname(os.path.abspath(a.src)), "engine_fields_v16d.py")
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
