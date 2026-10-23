#!/usr/bin/env python
"""fast_fields_v1.py (busi lane, reviewer A redesign prototype, REVIEWA_algorithm_redesign_v1 proposal R2): the field solve of
engine_fields_v16.solve_constraints_ivp (the two constraints as an outward initial-value problem) as a global collocation
problem solved by Newton, vectorized, in place of the adaptive scalar-stepping integrator.

The structure the production integrator does not use: the constraint system is triangular. With U the signed energy
function (U = 2 g M on the positive branch), q = -sqrt(U/r) and the smoothed sources rho_ell, J_ell:
    U'    = 8 pi g r^2 (rho + q J)         = f(r) + g1(r) sqrt(U),   f = 8 pi g r^2 rho,  g1 = -8 pi g r^(3/2) J   (one scalar nonlinear ODE)
    lnN'  = -4 pi g r J / q                = 4 pi g r^(3/2) J / sqrt(U)                                           (a quadrature given U)
    W'    = 8 pi g r^2 N^2 rho,  N = exp(lnN - lnN(r_collar))                                                       (a quadrature given lnN)
so one scalar equation carries the whole nonlinearity and the lapse and the final W are quadratures of smooth functions.
Discretization: panels from r_0 = ell/64 (the production's series start; the series below it) to r_collar, geometric near
r_0 (sqrt U ~ r^(3/2) there) and uniform of width ell/panels_per_ell beyond, p Gauss-Legendre nodes per panel, the
cumulative integration matrix A (A_jl = int_0^{c_j} l_l) and the end row b: U_ij = U_{i-1}^end + h_i sum_l A_jl F(r_il, U_il),
U_i^end = U_{i-1}^end + h_i sum_l b_l F(r_il, U_il). Newton on the global system (banded Jacobian, bandwidth p, scipy
solve_banded) from the explicit quadrature of f alone; lnN and W by the same A and b on their integrands; every dense value
by the panel's degree-p collocation polynomial (barycentric Lagrange on {0, c_1..c_p}). Accuracy: Gauss collocation of
order 2p at the panel ends and p + 1 inside, on sources smooth on the scale ell (Gaussian kernels), so with p = 10 and 8
panels per ell the discretization error is below 1e-13 of U; the solution is as accurate as the production integrator at
rtol 1e-12 (checked against a production run at rtol 1e-14, fast_fields_proto_v1.py).
Scope of this prototype: the intact gauge (U > 0 on (r_0, r_collar]); a solve whose U falls below zero reports
info["failed"] naming the production route (the flat-slice continuation in s = sqrt(U) is the same machinery on a
different independent variable and is the builder's port).
"""
import numpy as np
import scipy.linalg as sla
from numpy.polynomial import legendre as npleg
from numpy.polynomial import polynomial as nppoly


def gauss_colloc(p):
    """nodes c (p, in (0, 1)), the cumulative integration matrix A (p x p), the end weights b (p), and the barycentric weights of
    the p + 1 interpolation points {0, c_1, ..., c_p}."""
    x, w = npleg.leggauss(p)
    c = 0.5 * (x + 1.0)
    A = np.empty((p, p)); b = np.empty(p)
    for l in range(p):
        roots = np.delete(c, l)
        coef = nppoly.polyfromroots(roots)
        coef = coef / nppoly.polyval(c[l], coef)
        ic = nppoly.polyint(coef)
        A[:, l] = nppoly.polyval(c, ic)
        b[l] = nppoly.polyval(1.0, ic)
    xs = np.concatenate([[0.0], c, [1.0]])
    wb = np.empty(p + 2)
    for k in range(p + 2):
        wb[k] = 1.0 / np.prod(xs[k] - np.delete(xs, k))
    return c, A, b, xs, wb


def make_mesh(r0, r_collar, ell, panels_per_ell=8):
    """panel boundaries: geometric doubling from r_0 up to the uniform width, then uniform to r_collar."""
    h_u = ell / float(panels_per_ell)
    bnd = [r0]
    r = r0
    while 2.0 * r - bnd[-1] < h_u and 2.0 * r < r_collar:
        r = 2.0 * r
        bnd.append(r)
    n_u = max(1, int(np.ceil((r_collar - bnd[-1]) / h_u)))
    bnd = np.concatenate([np.array(bnd[:-1]), np.linspace(bnd[-1], r_collar, n_u + 1)])
    return bnd


class Panels:
    """the mesh, the node positions and the panel-wise collocation operators."""

    def __init__(self, bnd, p):
        self.bnd = np.asarray(bnd, float)
        self.n = len(bnd) - 1
        self.h = np.diff(self.bnd)
        self.c, self.A, self.b, self.xs, self.wb = gauss_colloc(p)
        self.p = p
        self.r = self.bnd[:-1, None] + self.h[:, None] * self.c[None, :]        # (n, p) node positions

    def cumulative(self, F, y0):
        """y at the nodes and panel ends from y(r_0) = y0 and the integrand values F (n, p): y_ij = y_{i-1}^end + h_i sum_l A_jl F_il."""
        inc = self.h * (F @ self.b)                       # (n,) panel increments
        yend = y0 + np.cumsum(inc)
        ystart = np.concatenate([[y0], yend[:-1]])
        ynode = ystart[:, None] + self.h[:, None] * (F @ self.A.T)
        return ynode, yend, ystart

    def evaluate(self, ynode, ystart, yend, r):
        """the collocation polynomial (degree p + 1 through the panel start, its p nodes and its end) at the points r in [bnd[0], bnd[-1]]."""
        r = np.asarray(r, float)
        i = np.clip(np.searchsorted(self.bnd, r, side="right") - 1, 0, self.n - 1)
        tau = (r - self.bnd[i]) / self.h[i]
        vals = np.concatenate([ystart[i][:, None], ynode[i], yend[i][:, None]], axis=1)          # (m, p + 2)
        d = tau[:, None] - self.xs[None, :]
        exact = np.abs(d) < 1e-15
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = self.wb[None, :] / d
            out = (terms * vals).sum(axis=1) / terms.sum(axis=1)
        hit = exact.any(axis=1)
        if hit.any():
            k = np.argmax(exact[hit], axis=1)
            out[hit] = vals[hit, k]
        return out


def solve_constraints_collocation(eps_tot, J_tot, g, R, ell, F12, FLD, p=8, panels_per_ell=2, support_rel=1e-14, collar_policy="flag",
                                  newton_tol=1e-14, newton_max=30, r0_frac=1.0 / 64.0, centre_quad=24, timing=None, diagnostics=True):
    import time as _time
    tm = {} if timing is None else timing
    _t0 = _time.time()
    """the same inputs as engine_fields_v16.solve_constraints_ivp and the same primary outputs (N_sites, v_sites, W_nodes, info with the
    dense callables, E_collar, r_collar, N_min, the residual diagnostics); F12 and FLD the production modules (sources, grids, series)."""
    R = np.asarray(R, float)
    r_nodes = F12.coarse_grid(R, ell)
    src = F12.SmoothedSources5(eps_tot, J_tot, R, ell, r_nodes[-1], support_rel)
    info = {"module": "fast_fields_v1", "failed": None, "r_nodes": r_nodes, "R_last": src.R_last, "src": src, "ell": ell, "p": p, "panels_per_ell": panels_per_ell}
    tm["sources_s"] = _time.time() - _t0; _t0 = _time.time()
    i_need = int(np.ceil((src.R_last + 8.0 * ell) / ell))
    if i_need > len(r_nodes) - 1:
        if collar_policy == "fail":
            info["failed"] = "the collar margin unavailable"
            return None, None, None, info
        i_collar = len(r_nodes) - 1
    else:
        i_collar = i_need
    r_collar = float(r_nodes[i_collar])
    info.update({"i_collar": i_collar, "r_collar": r_collar})
    rfine = np.arange(0.0, r_collar + 1e-9, ell / 16.0)
    r0 = ell * r0_frac
    cc = FLD.centre_coefficients(src, g, ell)
    if not cc["regular"]:
        info["failed"] = "fast route: the centre is outside the regular series (rho_0 <= 0): production route"
        return None, None, None, info
    U0 = cc["a"] * r0 ** 3 + cc["b"] * r0 ** 5
    L0 = cc["L2"] * r0 ** 2 + cc["L4"] * r0 ** 4
    bnd = make_mesh(r0, r_collar, ell, panels_per_ell)
    P = Panels(bnd, p)
    rr = P.r.ravel()
    rho = np.asarray(src.rho(rr), float).reshape(P.r.shape); J = np.asarray(src.J(rr), float).reshape(P.r.shape)
    fpi = 4.0 * np.pi * g
    f = 2.0 * fpi * P.r ** 2 * rho
    g1 = -2.0 * fpi * P.r ** 1.5 * J
    tm["mesh_sources_s"] = _time.time() - _t0; _t0 = _time.time()
    # Newton on U at the nodes (the panel ends eliminated: U_i^end = U_{i-1}^end + h_i b . F_i is kept as an unknown for a banded system)
    n, q = P.n, p
    N_unk = n * (q + 1)
    # initial guess: the quadrature of f alone
    Unode, Uend, Ustart = P.cumulative(f, U0)
    y = np.empty(N_unk)
    y.reshape(n, q + 1)[:, :q] = Unode; y.reshape(n, q + 1)[:, q] = Uend
    # the fixed sparsity: row (i, j) depends on the previous end (column (i-1, q)) and the panel's q + 1 unknowns
    rows_blk = np.arange(q + 1)
    hA = np.concatenate([P.h[:, None, None] * P.A[None, :, :], P.h[:, None, None] * P.b[None, None, :]], axis=1)      # (n, q + 1, q)
    conv = False; it_hist = []
    t_asm = 0.0; t_sol = 0.0
    ab = np.zeros((2 * q + 2, N_unk))
    for it in range(newton_max):
        _ta = _time.time()
        Y = y.reshape(n, q + 1)
        Un = Y[:, :q]; Ue = Y[:, q]
        Us = np.concatenate([[U0], Ue[:-1]])
        sq = np.sqrt(np.maximum(Un, 0.0))
        F = f + g1 * sq
        dF = np.where(Un > 0.0, g1 / (2.0 * np.maximum(sq, 1e-300)), 0.0)
        res = Y - Us[:, None] - np.einsum("ijl,il->ij", hA, F)
        # banded Jacobian: ab[u + r - c, c] with u = q, l = q + 1; every (row, column) position written once
        ab.fill(0.0)
        base = (np.arange(n) * (q + 1))[:, None]
        rr_ = base + rows_blk[None, :]                                  # (n, q + 1) global row indices
        ab[q, base[:, 0] + q] = 1.0                                      # the end unknowns' diagonal
        for l in range(q):
            cidx = base[:, 0] + l
            vals = -hA[:, :, l] * dF[:, l][:, None]                       # (n, q + 1): rows r = 0..q of panel i against node column l
            vals[:, l] += 1.0                                              # the node unknown's own diagonal
            c_g = np.broadcast_to(cidx[:, None], rr_.shape)
            ab[q + rr_ - c_g, c_g] = vals
        cprev = np.broadcast_to((base[1:, 0] - 1)[:, None], rr_[1:].shape)
        ab[q + rr_[1:] - cprev, cprev] = -1.0
        _tb = _time.time(); t_asm += _tb - _ta
        dy = sla.solve_banded((q + 1, q), ab, -res.ravel(), check_finite=False)
        t_sol += _time.time() - _tb
        y = y + dy
        step = float(np.max(np.abs(dy))) / max(float(np.max(np.abs(y))), 1e-300)
        it_hist.append(step)
        if step < newton_tol:
            conv = True
            break
    Y = y.reshape(n, q + 1); Unode = Y[:, :q]; Uend = Y[:, q]; Ustart = np.concatenate([[U0], Uend[:-1]])
    info["newton"] = {"iterations": len(it_hist), "converged": conv, "steps": it_hist, "unknowns": N_unk, "panels": n}
    tm["newton_s"] = _time.time() - _t0; tm["newton_assembly_s"] = t_asm; tm["newton_solve_s"] = t_sol; _t0 = _time.time()
    if not conv:
        info["failed"] = "fast route: Newton did not converge (%s)" % it_hist[-3:]
        return None, None, None, info
    if Unode.min() <= 0.0 or Uend.min() <= 0.0:
        info["failed"] = "fast route: U reaches zero on (r_0, r_collar] (min U/U_max %.3e): the flat-slice gauge continuation (production route)" % (min(Unode.min(), Uend.min()) / max(Unode.max(), 1e-300))
        return None, None, None, info
    # lnN by quadrature of 4 pi g r^(3/2) J / sqrt(U)
    Gl = fpi * P.r ** 1.5 * J / np.sqrt(Unode)
    Lnode, Lend, Lstart = P.cumulative(Gl, L0)
    lnN_c = float(Lend[-1])
    Nn = np.exp(Lnode - lnN_c)
    # W: from 0 to r_0 by a Gauss quadrature of the series integrand, then by the collocation quadrature
    xg, wg = npleg.leggauss(centre_quad)
    rq = 0.5 * r0 * (xg + 1.0); wq = 0.5 * r0 * wg
    Nq = np.exp(cc["L2"] * rq ** 2 + cc["L4"] * rq ** 4 - lnN_c)
    rhoq = np.asarray(src.rho(rq), float)
    W0 = float(np.sum(wq * 2.0 * fpi * rq ** 2 * Nq ** 2 * rhoq))
    Hw = 2.0 * fpi * P.r ** 2 * Nn ** 2 * rho
    Wnode, Wend, Wstart = P.cumulative(Hw, W0)
    E = float(Wend[-1]) / (2.0 * g)
    _gE = g * E
    tm["quadratures_s"] = _time.time() - _t0; _t0 = _time.time()

    def U_dense(r):
        r = np.abs(np.atleast_1d(np.asarray(r, float)))
        out = np.empty_like(r)
        lo = r < r0
        out[lo] = cc["a"] * r[lo] ** 3 + cc["b"] * r[lo] ** 5
        hi = ~lo
        out[hi] = P.evaluate(Unode, Ustart, Uend, np.minimum(r[hi], r_collar))
        return out

    def lnN_dense(r):
        r = np.abs(np.atleast_1d(np.asarray(r, float)))
        out = np.empty_like(r)
        lo = r < r0
        out[lo] = cc["L2"] * r[lo] ** 2 + cc["L4"] * r[lo] ** 4
        hi = ~lo
        out[hi] = P.evaluate(Lnode, Lstart, Lend, np.minimum(r[hi], r_collar))
        return out

    def N_dense_f(r):
        r = np.abs(np.atleast_1d(np.asarray(r, float)))
        return np.where(r <= r_collar, np.exp(lnN_dense(r) - lnN_c), 1.0)

    def W_field(r):
        r = np.abs(np.atleast_1d(np.asarray(r, float)))
        out = np.full(r.shape, 2.0 * _gE)
        m_ = r < r_collar
        if m_.any():
            rm = r[m_]
            w = np.empty_like(rm)
            lo = rm < r0
            if lo.any():
                # the series integrand integrated from 0 to r (small r only)
                rl = rm[lo]
                xx = 0.5 * rl[:, None] * (xg[None, :] + 1.0); ww = 0.5 * rl[:, None] * wg[None, :]
                Nx = np.exp(cc["L2"] * xx ** 2 + cc["L4"] * xx ** 4 - lnN_c)
                w[lo] = np.sum(ww * 2.0 * fpi * xx ** 2 * Nx ** 2 * (cc["rho0"] + cc["rho2"] * xx ** 2), axis=1)
            hi = ~lo
            w[hi] = P.evaluate(Wnode, Wstart, Wend, rm[hi])
            out[m_] = w
        return out

    def v_dense_f(r):
        r = np.atleast_1d(np.asarray(r, float))
        a_ = np.abs(r)
        W = W_field(a_)
        out = -np.sqrt(np.where(W > 0.0, W, 0.0) / np.maximum(a_, 1e-300))
        out = np.where(a_ > 0.0, out, 0.0)
        return np.where(r < 0, -out, out)
    cd = lambda Fn, x, hh: (Fn(x + hh) - Fn(x - hh)) / (2 * hh)
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
        return float(np.max(np.abs(dWdr - termW))) / sW, float(np.max(np.abs(dinvN - termN))) / sN
    if diagnostics:
        rW, rN = residuals_at(ell / 64.0)
        info["residual_W"], info["residual_invN"] = rW, rN
        Uf = U_dense(rfine)
        info["U_min_over_max"] = float(Uf.min() / max(Uf.max(), 1e-300))
    else:
        info["residual_W"], info["residual_invN"] = None, None
        info["U_min_over_max"] = float(min(Unode.min(), Uend.min()) / max(Unode.max(), 1e-300))
    tm["diagnostics_s"] = _time.time() - _t0; _t0 = _time.time()
    N_sites = np.array(N_dense_f(R), float); v_sites = np.array(v_dense_f(R), float)
    ext = R >= r_collar
    N_sites[ext] = 1.0; v_sites[ext] = -np.sqrt(2.0 * g * E / R[ext])
    W_nodes = W_field(r_nodes)
    N_nodes = np.array(N_dense_f(r_nodes), float)
    v_nodes = np.zeros(len(r_nodes)); v_nodes[1:] = -np.sqrt(np.where(W_nodes[1:] > 0, W_nodes[1:], 0.0) / r_nodes[1:])
    if r_collar < r_nodes[-1]:
        # the W tail with N = 1 beyond the collar: a Gauss quadrature of 8 pi g r^2 rho on [r_collar, r_end]
        xt, wt = npleg.leggauss(64)
        n_t = max(1, int(np.ceil((r_nodes[-1] - r_collar) / ell)))
        et = np.linspace(r_collar, r_nodes[-1], n_t + 1)
        Wt = 2.0 * _gE
        for a_, b_ in zip(et[:-1], et[1:]):
            xx = 0.5 * (b_ - a_) * (xt + 1.0) + a_; ww = 0.5 * (b_ - a_) * wt
            Wt += float(np.sum(ww * 2.0 * fpi * xx ** 2 * np.asarray(src.rho(xx), float)))
        info["W_tail"] = {"r_end": float(r_nodes[-1]), "E_tail_end": float(Wt / (2.0 * g)), "E_collar": E, "flagged": False}
    tm["outputs_s"] = _time.time() - _t0
    info["timing"] = tm
    info.update({"N_dense": N_dense_f, "v_dense": v_dense_f, "W_dense": W_field, "W_field": W_field, "U_dense": U_dense, "rho_ell": src.rho, "J_ell": src.J,
                 "W_nodes": W_nodes, "N_nodes": N_nodes, "v_nodes": v_nodes, "E_collar": E, "N_min": float(N_sites.min()), "N_centre": float(np.exp(-lnN_c)),
                 "lnN_collar": lnN_c, "rfine": rfine, "it": 1, "converged": True, "gauge": {"label": "intact", "intervals": []}, "site_fields": "dense"})
    return N_sites, v_sites, W_nodes, info
