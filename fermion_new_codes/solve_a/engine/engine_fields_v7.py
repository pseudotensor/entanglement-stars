#!/usr/bin/env python
"""engine_fields_v7.py (builder analyst; version 7: the iteration's convergence test accepts a stall at the integrator's floor. Version 6
required both changes (dN, and dW relative to max|W|) below tol = 1e-12, equal to the integrator's rtol, so a re-solve whose changes sat at the
integrator's noise (the r_s 64 primary at T/rs 0.25: dN 7.6e-13, dW 1.46e-12 after 200 iterations) was rejected and stopped the run. Now: converged
when both are below tol (as before), or when the larger change stays below the floor 10 max(tol, rtol) = 1e-11 for three consecutive
iterations (info["converged_at_floor"] records the changes, the floor and the iteration; BUILDER, labelled). A change above the floor at the
iteration limit is still rejected. Nothing else changes. Version 6's docstring follows.
engine_fields_v6.py (builder analyst; version 6: trapping_radius scans to the chain's last node, not to the collar: the dense functions
continue analytically beyond the collar (N = 1, W = 2 g E), and a surface there (a clean collar inside R_t) was missed by version 5's scan;
found on obs_plateau_v6's small-ball control. Nothing else changes. Version 5's docstring follows.
engine_fields_v5.py (builder analyst; the lead's "engine_fields_v4" of SPEC_final4b_engine.md sec. 16.2 items (1) to (4), (6) and (9); the
name v4 was taken by the dense-site-field version, so this is version 5 of engine_fields_v2's line). Changes against engine_fields_v4:

(1) Convergence on the dense functions: per iteration the largest change over a grid of spacing ell/16 (to the collar) of N_k - N_{k-1} and of
    (W_k - W_{k-1})/max|W|, both below tol; after convergence W is re-integrated with the final lapse (the returned pair is then the final
    N with its own W) and the coupled residuals of both equations are printed (dW/dr - 8 pi g r^2 N^2 rho_ell and d(1/N)/dr - 4 pi g r J_ell/v
    on the fine grid by Richardson-combined central differences of the dense functions, relative to the largest term). Rejected (info["failed"] with the reason,
    nothing clamped): an integrator failure, the iteration limit, a W below -1e-12 max|W| anywhere on the fine grid (values in
    [-1e-12 max|W|, 0), the integrator's rounding at the centre, read as 0 and counted), a nonpositive 1/N anywhere on the fine grid, the
    collar margin unavailable (policy "fail"; see (3)).
(2) The smoothed sources are splined on the parity-extended grid (rho even, J odd through the centre); the splines' largest error against the
    direct kernel sums at 200 test points between the grid nodes is printed (info["source_spline_error"]); the centre's series
    W = (8 pi g/3) N_0^2 rho_0 r^3 is checked at r = ell/16 (info["centre_series_ratio"]).
(3) R_last from the supports of both sources (|eps| or |J| above support_rel of their maxima). The collar is R_last + 8 ell on the coarse grid;
    if that lies beyond the chain: policy "fail" (the lead's rule) sets info["failed"]; policy "flag" (BUILDER, labelled: the collapse's
    continued sea polarizes the whole chain, so its support reaches the chain's end at every re-solve; every v5 record has its collar at the
    last node) places the collar at the last node and prints the margin's deficit and the sources' share in the last 8 ell
    (info["collar_margin"]). W beyond the collar is the exterior constant (the field: mass_function and W_field return it); the integration of
    W through the tail is kept as a separate diagnostic (info["W_tail"]).
(4) trapping_radius on the dense functions: N - |v| on a grid of spacing ell/64 from the centre, sign changes and sampled zeros bracketed
    and refined by brentq, tangencies (local minima within 1e-10 of zero) listed, the minimum of N - |v| refined by bounded minimization.
(6) The export helper: the shell totals, g, ell and the solver's parameters are what a consumer needs to re-solve the dense functions
    (deterministic, seconds); resolve_dense(totals...) does it.
(9) residuals and gauss_sum on the dense pair; the spectral diagnostic's FFT with the centre inserted (N - 1 even with its central value, v odd
    with v(0) = 0); the mass labels: mass_function = W/(2g) (the collar charge's form), misner_sharp = W/(2 g N^2) (equal to R/(2g) at a
    trapping zero); the dictionary line amended: "the goal's display's sign of v amended per engine 12.2; the owner's confirmation pending".
The kernels, the coarse grid, the momentum law's sign (engine 12.2's dictionary), the collar normalization N = 1 and the dense site fields are
version 4's. Interface: solve_constraints(eps_tot, J_tot, g, R, ell, N0=None, tol=1e-12, maxit=200, rtol=1e-12, atol=1e-16,
collar_policy="fail", support_rel=1e-14) -> (N_sites, v_sites, W_nodes, info); charge, gauss_sum, trapping_radius, mass_function,
misner_sharp, exterior_check, residuals, lattice_scale_power."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import make_interp_spline
from scipy.optimize import brentq, minimize_scalar

import engine_fields_v4 as F4

kernels = F4.kernels
coarse_grid = F4.coarse_grid
parity_spline = F4.parity_spline
DICTIONARY = F4.DICTIONARY + "; the goal's display's sign of v amended per engine 12.2; the owner's confirmation pending"


class SmoothedSources5:
    """rho_ell(r), J_ell(r): the kernel sums tabulated on a grid of spacing ell/16 and splined (quintic) on the parity-extended grid."""

    def __init__(self, eps_tot, J_tot, R, ell, r_max, support_rel=1e-14):
        R = np.asarray(R, float); eps_tot = np.asarray(eps_tot, float); J_tot = np.asarray(J_tot, float)
        self.R, self.eps_tot, self.J_tot, self.ell = R, eps_tot, J_tot, ell
        sel = (np.abs(eps_tot) > 0.0) | (np.abs(J_tot) > 0.0)
        be = max(float(np.abs(eps_tot).max()) if len(eps_tot) else 0.0, 1e-300)
        bj = max(float(np.abs(J_tot).max()) if len(J_tot) else 0.0, 1e-300)
        sig = (np.abs(eps_tot) > support_rel * be) | (np.abs(J_tot) > support_rel * bj)
        self.R_last = float(R[sig].max()) if sig.any() else 0.0
        self.sel = sel
        rf = np.arange(0.0, r_max + ell / 16.0, ell / 16.0)
        rho, J = self.direct(rf)
        self.rf, self.rho_f, self.J_f = rf, rho, J
        xx = np.concatenate([-rf[:0:-1], rf])
        self.rho = make_interp_spline(xx, np.concatenate([rho[:0:-1], rho]), k=5)
        self.J = make_interp_spline(xx, np.concatenate([-J[:0:-1], J]), k=5)
        self.total_eps = float(eps_tot.sum())
        self.total_eps_quadrature = float(np.trapezoid(4.0 * np.pi * rf ** 2 * rho, rf)) if hasattr(np, "trapezoid") else float(np.trapz(4.0 * np.pi * rf ** 2 * rho, rf))
        idx = np.linspace(0, len(rf) - 2, min(200, len(rf) - 1)).astype(int)
        rt = 0.5 * (rf[idx] + rf[idx + 1])
        rd, Jd = self.direct(rt)
        sr = max(float(np.abs(rho).max()), 1e-300); sj = max(float(np.abs(J).max()), 1e-300)
        self.spline_error = {"rho": float(np.max(np.abs(self.rho(rt) - rd))) / sr, "J": float(np.max(np.abs(self.J(rt) - Jd))) / sj}

    def direct(self, r):
        r = np.asarray(r, float)
        rho = np.zeros_like(r); J = np.zeros_like(r)
        if self.sel.any():
            Rs, es, Js = self.R[self.sel], self.eps_tot[self.sel], self.J_tot[self.sel]
            chunk = max(1, int(2e6 // max(len(Rs), 1)))
            for i0 in range(0, len(r), chunk):
                K0, K1 = kernels(r[i0:i0 + chunk], Rs, self.ell)
                rho[i0:i0 + chunk] = K0 @ es
                J[i0:i0 + chunk] = K1 @ Js
        return rho, J


def solve_constraints(eps_tot, J_tot, g, R, ell, N0=None, tol=1e-12, maxit=200, rtol=1e-12, atol=1e-16, collar_policy="fail", support_rel=1e-14,
                      site_fields="dense"):
    R = np.asarray(R, float)
    r_nodes = coarse_grid(R, ell)
    src = SmoothedSources5(eps_tot, J_tot, R, ell, r_nodes[-1], support_rel)
    info = {"it": 0, "converged": False, "ell": ell, "r_nodes": r_nodes, "R_last": src.R_last, "failed": None, "dictionary": DICTIONARY,
            "site_fields": "dense", "source_spline_error": src.spline_error, "collar_policy": collar_policy, "support_rel": support_rel,
            "charge_conservation_quadrature": (src.total_eps_quadrature / src.total_eps - 1.0) if src.total_eps else 0.0}
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
    if N0 is None:
        Nprev = lambda r: np.ones_like(np.atleast_1d(np.asarray(r, float)))
    else:
        N0a = np.asarray(N0, float)
        Nprev = lambda r: np.interp(np.abs(np.atleast_1d(np.asarray(r, float))), R, N0a, left=N0a[0], right=1.0)
    Wprev_f = None
    roundoff_negW = 0
    floor = 10.0 * max(tol, rtol)
    below = 0

    def integrate_W(Nf):
        rhsW = lambda r, W: [8.0 * np.pi * g * r * r * float(Nf(r)[0]) ** 2 * float(src.rho(r))]
        return solve_ivp(rhsW, (0.0, r_collar), [0.0], method="DOP853", dense_output=True, rtol=rtol, atol=atol)

    for it in range(maxit):
        solW = integrate_W(Nprev)
        if not solW.success:
            info["failed"] = "the W integration failed: %s" % solW.message
            return None, None, None, info
        Wf = solW.sol(rfine)[0]
        Wmax = max(float(np.max(np.abs(Wf))), 1e-300)
        if Wf.min() < -1e-12 * Wmax:
            info["failed"] = "negative W %.3e at r = %.4g (dense grid)" % (Wf.min(), rfine[int(np.argmin(Wf))])
            return None, None, None, info
        roundoff_negW = int((Wf < 0).sum())
        Jmax = max(float(np.max(np.abs(src.J_f))), 1e-300)
        vscale = np.sqrt(Wmax / max(r_collar, 1e-300))

        def v_of(r, _s=solW):
            if r <= 0.0:
                return 0.0
            W = float(_s.sol(r)[0])
            return -np.sqrt(W / r) if W > 0.0 else 0.0

        flags = {}

        def rhs_invN(r, y):
            if r <= 0.0:
                return [0.0]
            v = v_of(r); J = float(src.J(r))
            if abs(v) < 1e-12 * vscale:
                if abs(J) > 1e-10 * Jmax:
                    flags["momentum_without_flow"] = r
                return [0.0]
            return [4.0 * np.pi * g * r * J / v]
        sol_inv = solve_ivp(rhs_invN, (r_collar, 0.0), [1.0], method="DOP853", dense_output=True, rtol=rtol, atol=atol)
        if not sol_inv.success:
            info["failed"] = "the 1/N integration failed: %s" % sol_inv.message
            return None, None, None, info
        if flags.get("momentum_without_flow") is not None:
            info["failed"] = "momentum without flow at r = %.6g" % flags["momentum_without_flow"]
            return None, None, None, info
        invNf = sol_inv.sol(rfine)[0]
        if invNf.min() <= 0.0:
            info["failed"] = "nonpositive 1/N %.3e at r = %.4g (dense grid)" % (invNf.min(), rfine[int(np.argmin(invNf))])
            return None, None, None, info
        Nnew = lambda r, _si=sol_inv: np.where(np.abs(np.atleast_1d(np.asarray(r, float))) <= r_collar,
                                                1.0 / _si.sol(np.minimum(np.abs(np.atleast_1d(np.asarray(r, float))), r_collar))[0], 1.0)
        dN = float(np.max(np.abs(Nnew(rfine) - Nprev(rfine))))
        dW = float(np.max(np.abs(Wf - Wprev_f))) / Wmax if Wprev_f is not None else np.inf
        info["it"] = it + 1
        Nprev, Wprev_f = Nnew, Wf
        if dN < tol and dW < tol:
            info["converged"] = True
            break
        below = below + 1 if max(dN, dW) < floor else 0
        if below >= 3:
            info["converged"] = True
            info["converged_at_floor"] = {"dN": dN, "dW": dW, "floor": floor, "it": it + 1}
            break
    info["last_dN"], info["last_dW"] = dN, dW
    if not info["converged"]:
        info["failed"] = "not converged in %d iterations (dN %.2e, dW %.2e)" % (maxit, dN, dW)
        return None, None, None, info
    # the returned pair: the final lapse with W re-integrated from it
    Nfin = Nprev
    solW = integrate_W(Nfin)
    if not solW.success:
        info["failed"] = "the final W integration failed: %s" % solW.message
        return None, None, None, info
    Wf = solW.sol(rfine)[0]
    Wmax = max(float(np.max(np.abs(Wf))), 1e-300)
    if Wf.min() < -1e-12 * Wmax:
        info["failed"] = "negative W %.3e (final)" % Wf.min()
        return None, None, None, info
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
        a = np.abs(r)
        W = W_field(a)
        out = -np.sqrt(np.where(W > 0.0, W, 0.0) / np.maximum(a, 1e-300))
        out = np.where(a > 0.0, out, 0.0)
        return np.where(r < 0, -out, out)
    # the coupled residuals of the returned pair on the fine grid (central differences at h = ell/64 and ell/128, Richardson-combined)
    h = ell / 64.0
    ri = rfine[(rfine > 2 * h) & (rfine < r_collar - 2 * h)]
    rho_i = src.rho(ri); J_i = src.J(ri)
    Nr = N_dense_f(ri); vr = v_dense_f(ri)
    cd = lambda F, x, hh: (F(x + hh) - F(x - hh)) / (2 * hh)
    dWdr = (4.0 * cd(W_field, ri, h / 2) - cd(W_field, ri, h)) / 3.0
    termW = 8.0 * np.pi * g * ri ** 2 * Nr ** 2 * rho_i
    invNf = lambda x: 1.0 / N_dense_f(x)
    dinvN = (4.0 * cd(invNf, ri, h / 2) - cd(invNf, ri, h)) / 3.0
    termN = np.where(np.abs(vr) > 1e-12 * max(float(np.abs(vr).max()), 1e-300), 4.0 * np.pi * g * ri * J_i / np.where(vr != 0, vr, 1.0), 0.0)
    info["residual_W"] = float(np.max(np.abs(dWdr - termW))) / max(float(np.max(np.abs(termW))), 1e-300)
    info["residual_invN"] = float(np.max(np.abs(dinvN - termN))) / max(float(np.max(np.abs(termN))), 1e-300)
    # the centre's series
    rc = ell / 16.0
    W0 = float(W_field(rc)[0]); ser = 8.0 * np.pi * g / 3.0 * float(N_dense_f(0.0)[0]) ** 2 * float(src.rho(0.0)) * rc ** 3
    info["centre_series_ratio"] = (W0 / ser) if ser != 0 else None
    info["W_roundoff_negative_points"] = roundoff_negW
    # the tail integration beyond the collar (diagnostic only)
    if r_collar < r_nodes[-1]:
        rhsT = lambda r, W: [8.0 * np.pi * g * r * r * float(src.rho(r))]
        solT = solve_ivp(rhsT, (r_collar, r_nodes[-1]), [2.0 * _gE], method="DOP853", rtol=rtol, atol=atol)
        info["W_tail"] = {"r_end": float(r_nodes[-1]), "E_tail_end": float(solT.y[0][-1] / (2.0 * g)), "E_collar": E}
    # the sites
    N_sites = np.array(N_dense_f(R), float); v_sites = np.array(v_dense_f(R), float)
    ext = R >= r_collar
    N_sites[ext] = 1.0; v_sites[ext] = -np.sqrt(2.0 * g * E / R[ext])
    # the node values and the spline diagnostic (the interpolation floor), as before
    W_nodes = W_field(r_nodes)
    N_nodes = np.array(N_dense_f(r_nodes), float)
    v_nodes = np.zeros(len(r_nodes)); v_nodes[1:] = -np.sqrt(np.where(W_nodes[1:] > 0, W_nodes[1:], 0.0) / r_nodes[1:])
    inner = ~ext
    if inner.any():
        N_spl = parity_spline(r_nodes, N_nodes, +1); v_spl = parity_spline(r_nodes, v_nodes, -1)
        info["interp_floor_N"] = float(np.max(np.abs(N_spl(R[inner]) - N_sites[inner])))
        info["interp_floor_v"] = float(np.max(np.abs(v_spl(R[inner]) - v_sites[inner])))
    info.update({"N_dense": N_dense_f, "v_dense": v_dense_f, "W_dense": W_field, "W_field": W_field, "rho_ell": src.rho, "J_ell": src.J,
                 "W_nodes": W_nodes, "N_nodes": N_nodes, "v_nodes": v_nodes, "E_collar": E, "N_min": float(N_sites.min()),
                 "spectrum": lattice_scale_power(N_sites - 1.0, v_sites, R, ell, float(N_dense_f(0.0)[0]) - 1.0)})
    return N_sites, v_sites, W_nodes, info


def charge(info, g, offset_nodes=0):
    """E = W(r_collar)/(2g); offset_nodes > 0 reads the tail diagnostic's value at the chain's end instead (the collar control)."""
    if offset_nodes and info.get("W_tail"):
        return float(info["W_tail"]["E_tail_end"])
    return float(info["E_collar"])


def gauss_sum(eps_tot, R, ell, N_of_r, r_max):
    """the hydrodynamic Gauss identity on the dense lapse: sum_n eps_n^tot <N^2>_n, <N^2>_n = int 4 pi r^2 N(r)^2 K_0(r, R_n) dr (trapezoid)."""
    return F4.gauss_sum(eps_tot, R, ell, lambda r: np.asarray(N_of_r(r), float), r_max)


def trapping_radius(N, v, R, info, fine=64):
    """zeros of N - |v| on the dense functions: a grid of spacing ell/fine from the centre; sign changes and sampled zeros refined by brentq;
    tangencies (local minima within 1e-10 of zero); the minimum refined by bounded minimization. {"zeros", "tangencies", "min", "r_min"}."""
    ell = info["ell"]; rc = float(info["r_nodes"][-1])
    f = lambda x: float(info["N_dense"](x)[0] - abs(info["v_dense"](x)[0]))
    rr = np.arange(0.0, rc + 1e-9, ell / fine)
    fv = info["N_dense"](rr) - np.abs(info["v_dense"](rr))
    zeros = []
    for i in range(len(rr) - 1):
        if fv[i] == 0.0:
            zeros.append(float(rr[i]))
        elif fv[i] * fv[i + 1] < 0.0:
            zeros.append(float(brentq(f, rr[i], rr[i + 1], xtol=1e-12)))
    tang = []
    for i in range(1, len(rr) - 1):
        if fv[i] <= fv[i - 1] and fv[i] <= fv[i + 1] and abs(fv[i]) < 1e-10:
            tang.append(float(rr[i]))
    i_min = int(np.argmin(fv))
    lo, hi = rr[max(i_min - 1, 0)], rr[min(i_min + 1, len(rr) - 1)]
    if hi > lo:
        res = minimize_scalar(f, bounds=(lo, hi), method="bounded", options={"xatol": 1e-10})
        fmin, rmin = float(res.fun), float(res.x)
    else:
        fmin, rmin = float(fv[i_min]), float(rr[i_min])
    return {"zeros": sorted(zeros), "tangencies": tang, "min": fmin, "r_min": rmin}


def mass_function(info, g, r):
    """W(r)/(2g): the collar charge's form (equal to the geometric mass where N = 1); beyond the collar the exterior constant."""
    return float(info["W_field"](r)[0] / (2.0 * g))


def misner_sharp(info, g, r):
    """W/(2 g N^2) = r v^2/(2 g N^2): the Misner-Sharp form, equal to R/(2g) at a trapping zero N = |v|."""
    return float(info["W_field"](r)[0] / (2.0 * g * float(info["N_dense"](r)[0]) ** 2))


def exterior_check(N, v, R, info, g):
    return F4.exterior_check(N, v, R, info, g)


def residuals(info, g, eps_tot=None, J_tot=None, R=None):
    """the coupled residuals of the returned dense pair (computed in solve_constraints): (residual_W, residual_invN), relative."""
    return info.get("residual_W"), info.get("residual_invN")


def lattice_scale_power(dN, v, R, ell, dN0=0.0):
    """the spectral fractions above pi/ell and pi/2 of the site fields extended through the centre with the centre inserted (dN even with its
    central value dN0, v odd with v(0) = 0), Hann-windowed."""
    out = {}
    for name, f, par, f0 in (("N", dN, +1, dN0), ("v", v, -1, 0.0)):
        ff = np.concatenate([par * f[::-1], [f0], f])
        ff = ff * np.hanning(len(ff))
        Fq = np.abs(np.fft.rfft(ff)) ** 2
        k = np.fft.rfftfreq(len(ff), d=1.0) * 2.0 * np.pi
        tot = Fq.sum() if Fq.sum() > 0 else 1.0
        out[name] = {"above_pi_over_ell": float(Fq[k > np.pi / ell].sum() / tot), "above_pi_over_2": float(Fq[k > np.pi / 2.0].sum() / tot)}
    return out


def resolve_dense(eps_tot, J_tot, g, R, ell, N0=None, **kw):
    """(6) the consumer's re-solve from the exported shell totals: the same deterministic solve, returning the dense functions in info."""
    return solve_constraints(eps_tot, J_tot, g, R, ell, N0=N0, **kw)


if __name__ == "__main__":
    # J0' on version 5: the prescribed Gaussian source (J = 0 and the iterated static-frame momentum), ell = 2, 4, 8, 16; the residuals,
    # the source splines' error, the centre series, the collar policy, the trapping-radius refinement.
    R = np.arange(1, 401, dtype=float)
    eps = 0.5 * np.exp(-((R - 20.0) / 6.0) ** 2); g = 0.25
    for ell in (2, 4, 8, 16):
        N, v, W, info = solve_constraints(eps, 0 * eps, g, R, ell)
        print("ell", ell, "J=0: E", info["E_collar"], "sum eps", eps.sum(), "it", info["it"], "res", info["residual_W"], info["residual_invN"],
              "src err", info["source_spline_error"], "centre", info["centre_series_ratio"], "collar", info["collar_margin"])
        for it in range(40):
            J = np.abs(v) * eps / N
            N, v, W, info = solve_constraints(eps, J, g, R, ell, N0=N)
            if info["failed"]:
                print("FAILED", info["failed"]); break
        tz = trapping_radius(N, v, R, info)
        print("   sign check: N_min %.6f E %.9f it %d dN %.1e dW %.1e res %.1e %.1e floor %.1e %.1e tz %s" % (
            N.min(), info["E_collar"], info["it"], info["last_dN"], info["last_dW"], info["residual_W"], info["residual_invN"],
            info.get("interp_floor_N", 0), info.get("interp_floor_v", 0), tz))
