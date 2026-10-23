#!/usr/bin/env python
"""engine_fields_v4.py (builder analyst; SPEC_final4b_engine.md sec. 15.1: the dense site fields). Version 4 of engine_fields_v3.py:
site_fields = "dense" (the default) takes the fields at every site from the integrators' dense solution of the continuum constraints with the
analytic smoothed sources (W from its DOP853 dense output, 1/N from its own), and the W integrand's lapse is the previous iteration's dense
lapse (version 3 splined the node values into it), so the coupled two-point problem is solved by the dense functions themselves; the quintic
parity spline through the coarse nodes is computed only as the diagnostic (its distance from the dense solution at the sites, the
interpolation floor, printed as before). info carries N_dense(r) and v_dense(r) (vectorized), and trapping_radius and the surface
derivatives use them. site_fields = "spline" is version 3's behaviour (for the finished families' comparability). Version 3's docstring
follows.
engine_fields_v3.py (builder analyst; version 3 of the lead's engine_fields_v2 per SPEC_final4b_engine.md sec. 12.4: (1) no clipping:
a negative W (below -1e-12 of max |W|) or a nonpositive 1/N sets info["failed"] with the reason and returns the unclipped fields; (2) the
flow-free rule by the undivided residual: where |v| < 1e-12 of max |v| the momentum source must satisfy |J_ell| < 1e-10 of max |J_ell|
and the lapse is continued with N' = 0, else failed ("momentum without flow"); (3) trapping_radius on the splines, both zeros, and when
no sign change exists the minimum of N - |v| and its radius; (4) mass_function(info, g, r) = W(r)/(2 g) at any r (the dense W); (5) the
dictionary line of sec. 12.2 in every info.  Version 2's docstring follows.
engine_fields_v2.py (lead's draft; SPEC_final4b_engine.md sec. 11.1): the hydrodynamic geometry of the declared model in spherical
symmetry.  Inputs: the shell totals eps_n^tot and J_n^tot on the chain's sites R_n (the Hartree-level normal-ordered sources after the
channel sum, spec 2.2), g = G/a^2, the averaging scale ell (lattice units, an integer).  The sources are averaged with the 3-dimensional
Gaussian of standard deviation ell written in r (the l = 0 kernel for the energy density, the l = 1 kernel for the radial momentum density;
both regular at the centre, the first charge conserving); the continuum constraints
    dW/dr = 8 pi g r^2 N^2 rho_ell(r),  W = r v^2,  W(0) = 0,  v = -sqrt(W/r)        (spec 3.1, continuous form)
    d(1/N)/dr = 4 pi g r J_ell(r)/v,  1/N = 1 at the collar                           (spec 3.2, continuous form)
are integrated on the coarse grid r_i = i ell (DOP853 with dense output between the nodes, the analytic smoothed sources), the pair
iterated; N and v are interpolated to the sites by quintic splines with the parity extension through the centre (N even, v odd); beyond
the collar the exterior is written analytically (N = 1, W = 2 g E).  No lattice-scale form of the geometric law appears in this file.
Interface (spec 11.10): solve_constraints(eps_tot, J_tot, g, R, ell, N0=None, ...) -> (N_sites, v_sites, W_nodes, info);
charge(info, g, offset_nodes=0); trapping_radius(N, v, R); exterior_check(N, v, R, info, g); residuals(info, g); lattice_scale_power(...);
gauss_sum(eps_tot, R, ell, N_of_r, r_max)."""
import numpy as np
from scipy.integrate import solve_ivp

DICTIONARY = ("dictionary: H = N E + v.p, ds^2 = -N^2 dt^2 + (dx - v dt)^2, the ADM sign of the momentum law "
              "(J_i = -(1/8 pi G) d_j (K^j_i - delta^j_i K) for K_ij = (d_i v_j + d_j v_i)/(2N))")
from scipy.interpolate import make_interp_spline


def kernels(r, R, ell):
    """K_0(r, R), K_1(r, R): the density at r per unit volume from a unit total on the shell of radius R, for a scalar density (l = 0) and a
    radial vector density (l = 1): K_l = (2 pi ell^2)^(-3/2) exp(-(r^2 + R^2)/(2 ell^2)) i_l(r R/ell^2), i_0 = sinh z/z,
    i_1 = cosh z/z - sinh z/z^2, in the exponentially scaled form; the series below z = 1e-2.  r: (n_r,), R: (n_R,) -> (n_r, n_R)."""
    r = np.asarray(r, float)[:, None]
    R = np.asarray(R, float)[None, :]
    Em = np.exp(-(r - R) ** 2 / (2.0 * ell ** 2))
    Ep = np.exp(-(r + R) ** 2 / (2.0 * ell ** 2))
    EA = np.exp(-(r ** 2 + R ** 2) / (2.0 * ell ** 2))
    z = r * R / ell ** 2
    small = z < 1e-2
    zs = np.where(small, 1.0, z)
    i0 = np.where(small, EA * (1.0 + z ** 2 / 6.0), (Em - Ep) / (2.0 * zs))
    i1 = np.where(small, EA * (z / 3.0) * (1.0 + z ** 2 / 10.0), ((zs - 1.0) * Em + (zs + 1.0) * Ep) / (2.0 * zs ** 2))
    pref = 1.0 / ((2.0 * np.pi) ** 1.5 * ell ** 3)
    return pref * i0, pref * i1


class SmoothedSources:
    """rho_ell(r), J_ell(r): the kernel sums over the shells with a nonzero source, tabulated on a fine grid of spacing ell/16 and
    interpolated by quintic splines (the sums are analytic; the splines reproduce them to 1e-12), so that the integrators' calls are cheap."""

    def __init__(self, eps_tot, J_tot, R, ell, r_max):
        R = np.asarray(R, float); eps_tot = np.asarray(eps_tot, float); J_tot = np.asarray(J_tot, float)
        sel = (np.abs(eps_tot) > 0.0) | (np.abs(J_tot) > 0.0)
        big = max(np.abs(eps_tot).max() if sel.any() else 0.0, 1e-300)
        sig = np.abs(eps_tot) > 1e-14 * big
        self.R_last = float(R[sig].max()) if sig.any() else 0.0
        rf = np.arange(0.0, r_max + ell / 16.0, ell / 16.0)
        rho = np.zeros_like(rf); J = np.zeros_like(rf)
        if sel.any():
            Rs, es, Js = R[sel], eps_tot[sel], J_tot[sel]
            chunk = max(1, int(2e6 // max(len(Rs), 1)))           # BUILDER: the kernel matrix in chunks of 2e6 entries
            for i0 in range(0, len(rf), chunk):
                K0, K1 = kernels(rf[i0:i0 + chunk], Rs, ell)
                rho[i0:i0 + chunk] = K0 @ es
                J[i0:i0 + chunk] = K1 @ Js
        self.rf, self.rho_f, self.J_f = rf, rho, J
        self.rho = make_interp_spline(rf, rho, k=5)
        self.J = make_interp_spline(rf, J, k=5)
        self.total_eps = float(eps_tot.sum())
        self.total_eps_quadrature = float(np.trapz(4.0 * np.pi * rf ** 2 * rho, rf))    # BUILDER: Simpson; the conservation check


def coarse_grid(R, ell):
    """the nodes r_i = i ell up to the chain's last site; ell integer so that r_i = R_{i ell - 1} for i >= 1 and r_0 = 0 is the centre."""
    n_c = int(np.floor(float(np.max(R)) / ell))
    return np.arange(n_c + 1, dtype=float) * ell


def parity_spline(r_nodes, y_nodes, parity):
    """quintic spline through the nodes with the data extended through the centre by its parity (+1 even, -1 odd)."""
    rr = np.concatenate([-r_nodes[:0:-1], r_nodes])
    yy = np.concatenate([parity * y_nodes[:0:-1], y_nodes])
    return make_interp_spline(rr, yy, k=5)


def solve_constraints(eps_tot, J_tot, g, R, ell, N0=None, tol=1e-13, maxit=200, rtol=1e-12, atol=1e-16, site_fields="dense"):
    R = np.asarray(R, float)
    r_nodes = coarse_grid(R, ell)
    src = SmoothedSources(eps_tot, J_tot, R, ell, r_nodes[-1])
    i_collar = min(len(r_nodes) - 1, int(np.ceil((src.R_last + 8.0 * ell) / ell)))
    r_collar = r_nodes[i_collar]
    N_nodes = np.ones(len(r_nodes)) if N0 is None else np.interp(r_nodes, R, np.asarray(N0, float))
    info = {"it": 0, "converged": False, "ell": ell, "r_nodes": r_nodes, "i_collar": i_collar, "r_collar": r_collar,
            "R_last": src.R_last, "charge_conservation_quadrature": (src.total_eps_quadrature / src.total_eps - 1.0) if src.total_eps else 0.0}
    solW = sol_inv = None
    N_dense_prev = None
    for it in range(maxit):
        if site_fields == "dense" and N_dense_prev is not None:
            N_of_r = N_dense_prev                                            # version 4: the previous iteration's dense lapse
        else:
            N_of_r = parity_spline(r_nodes, N_nodes, +1)
        rhsW = lambda r, W: [8.0 * np.pi * g * r * r * float(N_of_r(r)) ** 2 * float(src.rho(r))]
        solW = solve_ivp(rhsW, (0.0, r_nodes[-1]), [0.0], method="DOP853", t_eval=r_nodes, dense_output=True, rtol=rtol, atol=atol)
        W_nodes = solW.y[0]
        if W_nodes.min() < -1e-12 * max(float(np.max(np.abs(W_nodes))), 1e-300):
            info["negative_W"] = float(W_nodes.min())               # version 3: rejected, never clipped
            info["failed"] = "negative W %.3e" % W_nodes.min()
        W_dense = lambda r: float(solW.sol(r)[0])
        v_of = lambda r: (-np.sqrt(max(W_dense(r), 0.0) / r) if r > 0.0 else 0.0)

        Wmax = max(float(np.max(np.abs(W_nodes))), 1e-300)
        vmax = np.sqrt(Wmax / max(r_nodes[-1], 1e-300))
        Jmax = max(float(np.max(np.abs(src.J_f))), 1e-300)

        def rhs_invN(r, y):
            if r <= 0.0:
                return [0.0]
            v = v_of(r); J = float(src.J(r))
            if abs(v) < 1e-12 * vmax:                                # version 3: the flow-free rule by the undivided residual
                if abs(J) > 1e-10 * Jmax:
                    info["failed"] = "momentum without flow at r = %.6g (|J_ell| %.3e)" % (r, abs(J))
                return [0.0]                                          # N' = 0 where the flow vanishes
            return [4.0 * np.pi * g * r * J / v]

        t_eval = r_nodes[:i_collar + 1][::-1]
        sol_inv = solve_ivp(rhs_invN, (r_collar, 0.0), [1.0], method="DOP853", t_eval=t_eval, dense_output=True, rtol=rtol, atol=atol)
        invN = np.ones(len(r_nodes)); invN[:i_collar + 1] = sol_inv.y[0][::-1]
        if invN.min() <= 0.0:                                       # version 3: rejected, never clipped
            info["nonpositive_lapse"] = True
            info["failed"] = "nonpositive 1/N (min %.3e)" % invN.min()
            info.update({"W_nodes": W_nodes, "N_nodes": None, "dictionary": DICTIONARY})
            return None, None, W_nodes, info
        N_new = 1.0 / invN
        d = float(np.max(np.abs(N_new - N_nodes)))
        N_nodes = N_new
        if site_fields == "dense":
            _si, _rc = sol_inv, r_collar
            N_dense_prev = (lambda r, _si=_si, _rc=_rc: float(1.0 / _si.sol(min(max(r, 0.0), _rc))[0]) if r <= _rc else 1.0)
        info["it"] = it + 1
        if d < tol:
            info["converged"] = True
            break
    info["last_dN"] = d
    v_nodes = np.zeros(len(r_nodes)); v_nodes[1:] = -np.sqrt(np.maximum(W_nodes[1:], 0.0) / r_nodes[1:])
    E = float(W_nodes[i_collar] / (2.0 * g))
    N_spl = parity_spline(r_nodes, N_nodes, +1); v_spl = parity_spline(r_nodes, v_nodes, -1)
    N_sites = np.array(N_spl(R), float); v_sites = np.array(v_spl(R), float)
    ext = R >= r_collar
    N_sites[ext] = 1.0; v_sites[ext] = -np.sqrt(2.0 * g * E / R[ext])
    inner = ~ext
    _si, _sw, _rc, _gE = sol_inv, solW, r_collar, g * E

    def N_dense_f(r):
        r = np.atleast_1d(np.asarray(r, float))
        out = np.ones_like(r)
        m_ = r < _rc
        if m_.any():
            out[m_] = 1.0 / _si.sol(np.maximum(r[m_], 0.0))[0]
        return out

    def v_dense_f(r):
        r = np.atleast_1d(np.asarray(r, float))
        out = -np.sqrt(2.0 * _gE / np.maximum(r, 1e-300))
        m_ = r < _rc
        if m_.any():
            rr = np.maximum(r[m_], 1e-300)
            out[m_] = -np.sqrt(np.maximum(_sw.sol(rr)[0], 0.0) / rr)
        return out
    info["N_dense"] = N_dense_f; info["v_dense"] = v_dense_f; info["site_fields"] = site_fields
    if inner.any():                                              # the interpolation floor: the spline against the dense output at the sites
        N_dense = N_dense_f(R[inner]); v_dense = v_dense_f(R[inner])
        info["interp_floor_N"] = float(np.max(np.abs(N_sites[inner] - N_dense)))
        info["interp_floor_v"] = float(np.max(np.abs(v_sites[inner] - v_dense)))
        if site_fields == "dense":                               # version 4: the dense solution at the sites
            N_sites[inner] = N_dense; v_sites[inner] = v_dense
    info["W_dense"] = solW.sol                                     # version 3: the dense W for mass_function
    info.setdefault("failed", None)
    info["dictionary"] = DICTIONARY
    info.update({"W_nodes": W_nodes, "N_nodes": N_nodes, "v_nodes": v_nodes, "E_collar": E, "N_min": float(N_sites.min()),
                 "E_collar_plus_2ell": float(W_nodes[min(i_collar + 2, len(r_nodes) - 1)] / (2.0 * g)),
                 "E_collar_plus_4ell": float(W_nodes[min(i_collar + 4, len(r_nodes) - 1)] / (2.0 * g)),
                 "spectrum": lattice_scale_power(N_sites - 1.0, v_sites, R, ell)})
    return N_sites, v_sites, W_nodes, info


def charge(info, g, offset_nodes=0):
    """E read at the collar node (W/(2g)); offset_nodes moves the reading outward by that many nodes (the collar control)."""
    i = min(info["i_collar"] + offset_nodes, len(info["W_nodes"]) - 1)
    return float(info["W_nodes"][i] / (2.0 * g))


def gauss_sum(eps_tot, R, ell, N_of_r, r_max):
    """the hydrodynamic Gauss identity: sum_n eps_n^tot <N^2>_n with <N^2>_n = int 4 pi r^2 N(r)^2 K_0(r, R_n) dr (to compare with E)."""
    rf = np.arange(0.0, r_max, ell / 16.0)
    Nf2 = np.asarray(N_of_r(rf), float) ** 2 * 4.0 * np.pi * rf ** 2
    tot = 0.0
    eps_tot = np.asarray(eps_tot, float); R = np.asarray(R, float)
    sel = np.abs(eps_tot) > 0.0
    for i0 in range(0, int(sel.sum()), 256):                      # BUILDER: Simpson weights in place of trapz
        K0, _ = kernels(rf, R[sel][i0:i0 + 256], ell)
        w = np.trapz(Nf2[:, None] * K0, rf, axis=0)
        tot += float(w @ eps_tot[sel][i0:i0 + 256])
    return tot


def trapping_radius(N, v, R, info=None, fine=64):
    """version 4: with info["N_dense"] present (dense site fields) the zeros and the minimum on the dense functions; else version 3's."""
    if info is not None and info.get("N_dense") is not None and info.get("site_fields") == "dense":
        r = info["r_nodes"]
        rr = np.linspace(max(r[1] * 0.25, 1e-6), r[-1], int(fine * r[-1] / info["ell"]) + 1)
        f = info["N_dense"](rr) - np.abs(info["v_dense"](rr))
        s_ = np.nonzero(f[:-1] * f[1:] < 0)[0]
        zeros = [float(rr[i] - f[i] * (rr[i + 1] - rr[i]) / (f[i + 1] - f[i])) for i in s_]
        i_min = int(np.argmin(f))
        return {"zeros": zeros, "min": float(f[i_min]), "r_min": float(rr[i_min])}
    return trapping_radius_v3(N, v, R, info, fine)


def trapping_radius_v3(N, v, R, info=None, fine=64):
    """version 3: the zeros of N - |v| on the smooth fields (the parity splines of the node values on a grid of ell/fine), all of them
    (the outermost is the trapping surface), and when none exists the minimum of N - |v| and its radius (the tangential first appearance).
    Without info (the sites only) the site interpolation of version 2.  Returns {"zeros": [...], "min": value, "r_min": radius}."""
    if info is not None and info.get("N_nodes") is not None:
        r = info["r_nodes"]
        Ns = parity_spline(r, info["N_nodes"], +1); vs = parity_spline(r, info["v_nodes"], -1)
        rr = np.linspace(r[1] * 0.25, r[-1], int(fine * r[-1] / info["ell"]) + 1)
        f = Ns(rr) - np.abs(vs(rr))
    else:
        rr, f = np.asarray(R, float), np.asarray(N, float) - np.abs(np.asarray(v, float))
    s = np.nonzero(f[:-1] * f[1:] < 0)[0]
    zeros = [float(rr[i] - f[i] * (rr[i + 1] - rr[i]) / (f[i + 1] - f[i])) for i in s]
    i_min = int(np.argmin(f))
    return {"zeros": zeros, "min": float(f[i_min]), "r_min": float(rr[i_min])}


def mass_function(info, g, r):
    """version 3: M(r) = W(r)/(2 g) from the dense output of W (the geometric mass r v^2/(2 g N^2) times N^2; equal to it where N = 1)."""
    return float(info["W_dense"](r)[0] / (2.0 * g))


def exterior_check(N, v, R, info, g):
    """beyond the collar: N = 1 and r v^2 = 2 g E to rounding, by construction (printed every time)."""
    sel = R >= info["r_collar"]; E = info["E_collar"]
    return {"max_abs_N_minus_1": float(np.max(np.abs(N[sel] - 1.0))) if sel.any() else None,
            "max_abs_rv2_minus_2gE": float(np.max(np.abs(R[sel] * v[sel] ** 2 - 2.0 * g * E))) if sel.any() else None}


def residuals(info, g, eps_tot=None, J_tot=None, R=None):
    """the continuum equations at the nodes with the splines' derivatives: dW/dr - 8 pi g r^2 N^2 rho_ell and d(1/N)/dr - 4 pi g r J_ell/v.
    BUILDER: pass the sources to rebuild SmoothedSources; returns the two residual arrays over the interior nodes (i >= 1)."""
    r = info["r_nodes"]; ell = info["ell"]
    src = SmoothedSources(eps_tot, J_tot, R, ell, r[-1])
    W_spl = parity_spline(r, info["W_nodes"], -1); N_spl = parity_spline(r, info["N_nodes"], +1)
    ri = r[1:info["i_collar"] + 1]
    r1 = W_spl.derivative()(ri) - 8.0 * np.pi * g * ri ** 2 * N_spl(ri) ** 2 * src.rho(ri)
    invN_spl = parity_spline(r, 1.0 / info["N_nodes"], +1)
    v = -np.sqrt(np.maximum(W_spl(ri), 0.0) / ri)
    r2 = invN_spl.derivative()(ri) - np.where(np.abs(v) > 1e-300, 4.0 * np.pi * g * ri * src.J(ri) / np.where(np.abs(v) > 1e-300, v, 1.0), 0.0)
    return r1, r2


def lattice_scale_power(dN, v, R, ell):
    """the fraction of the spectral power of the interpolated fields above k = pi/ell and above k = pi/2 (lattice units): the Hann-windowed
    DFT over the chain extended through the centre by parity (dN = N - 1 even, v odd).  The verification of 'no lattice-scale components'."""
    out = {}
    for name, f, par in (("N", dN, +1), ("v", v, -1)):
        ff = np.concatenate([par * f[::-1], f])
        ff = ff * np.hanning(len(ff))
        F = np.abs(np.fft.rfft(ff)) ** 2
        k = np.fft.rfftfreq(len(ff), d=1.0) * 2.0 * np.pi
        tot = F.sum() if F.sum() > 0 else 1.0
        out[name] = {"above_pi_over_ell": float(F[k > np.pi / ell].sum() / tot), "above_pi_over_2": float(F[k > np.pi / 2.0].sum() / tot)}
    return out


if __name__ == "__main__":
    # J0' (spec 11.4): the prescribed Gaussian source with J = 0 (N = 1, E = sum eps at every ell), then the iterated static-frame momentum
    # J_n = |v_n| eps_n/N_n (the sign check N < 1), at ell = 2, 4, 8, 16; the collar controls; the spectra; the interpolation floor.
    R = np.arange(1, 401, dtype=float)
    eps = 0.5 * np.exp(-((R - 20.0) / 6.0) ** 2); g = 0.25
    for ell in (2, 4, 8, 16):
        N, v, W, info = solve_constraints(eps, 0 * eps, g, R, ell)
        print("ell", ell, "J=0: E", info["E_collar"], "sum eps", eps.sum(), "N-1 max", np.abs(N - 1).max(), "cons", info["charge_conservation_quadrature"])
        for it in range(40):
            J = np.abs(v) * eps / N
            N, v, W, info = solve_constraints(eps, J, g, R, ell, N0=N)
        print("   sign check: N_min %.6f E %.9f E+2ell %.9f gauss %.9f floor %.1e %.1e spectrum %s it %d" % (
            N.min(), info["E_collar"], info["E_collar_plus_2ell"], gauss_sum(eps, R, ell, parity_spline(info["r_nodes"], info["N_nodes"], +1), R[-1]),
            info.get("interp_floor_N", 0), info.get("interp_floor_v", 0), info["spectrum"], info["it"]))
    # BUILDER: C1.9, the OV-PG hydrodynamic check (spec 11.4 item 8) from engine_ovpg_v1's profile; the solver timing at n_sites 2048, 4096.
