#!/usr/bin/env python
"""page_scatter_v4.py. Changes against page_scatter_v3.py (builder analyst; SPEC_final4b_engine.md 31.4 (8) and 31.6 BUILDER item 1 (ii);
the GAP03 review review_GAP03_page_curve.tex, R10), each with the item it implements; imports page_ops_v4 and page_core_v4.
(31.4 (8)) The transmission the comparator uses is the stationary scattering's energy-resolved Gamma_lat(omega, kappa) of the frozen stage
    operator with the interior cut inside the trapped annulus, J3' version 6's (d) routine (obs_raymap_solved_v6.py, md5
    d946cf237aab17523e0290ee2c36f6d5; its outer_root, inner_root, Profile, blended_profile_chain and scatter_on are copied below verbatim
    and name the libraries they call): the stage's site fields (N = 1, v the averaged flow) as degree-7 parity splines
    (obs_plateau_v5.site_spline) inside the last site - 8 with the analytic exterior N = 1, v = -sqrt(2 g E/r) beyond; r_H,m and kappa_m
    by the five-site quadratic fit of N + v (outer_root), r_in,m by inner_root; cut A at r_in,m + 0.5 (r_H,m - r_in,m) the reading and
    cut B at 0.35 the check, w = min(4, (r_H,m - R_cut)/6); the chain of obs_plateau_v4.blended_chain at spacing a (R_L = 64 r_s, outer
    width 0.05, kappa_ref the spline's slope at the root); Gamma = alpha^2 + beta^2 of obs_plateau_v4.hawking_reading on
    obs_greybody_radial_v1.scatter_y (y = kappa/rho, the first and last 4 sites fixed), unitarity printed. It is averaged over the packet's
    exact energy distribution (stationary_gamma, packet_average). The libraries are imported by path from --obs-path (then
    $PAGE_OBS_PATH, ../obs, ../obs/code, ../../obs/code beside this file, <internal>/code), the directory printed.
(31.4 (8), R10) v3's time-domain branch-resolved measure is kept and labelled "the finite-window branch-current diagnostic" (not a
    transmission, not a gate); beside the stationary Gamma_lat each row prints the end-norm inside r_t at the end of the passage (the
    non-returned share, to compare with C-capture version 6's f_nr), the reconversion fraction 1 - (end-norm)/Gamma_lat, the continuum
    comparator and the ratio; 8.3 (2) blocks nothing.
(R10) the continuum comparator is evaluated at the packet's own frequencies, the same equal-weight quadrature over the packet's
    eigenmodes as the stationary Gamma (packet_average: every mode in a bin, each bin at its weighted mean energy, no clipping to a
    tabulation's ends; the half-resolution average printed as the quadrature check; frequencies without a Hawking-row reading, below
    the outer lead's angular gap, excluded with their weight printed); every row is labelled "particle
    half, massless" and --validate-scattering refuses m != 0 (the implemented control is massless); the lattice-corrected luminosity of v3
    (three packet ratios interpolated and extended) is replaced by the thermal comparator with the stationary lattice Gamma evaluated at
    the luminosity's own Gauss-Legendre nodes (f_H inserted: a comparator, not an evolved occupation).

page_scatter_v3.py. Changes against page_scatter_v2.py (builder analyst, BRIEF_page_v3_builder.md), each with the item of
SPEC_final4b_engine.md 30.1 it implements: imports page_ops_v3 and page_core_v3. (5) lattice_gamma is the branch-resolved transmission:
Gamma_lat = the time integral over the whole passage of the inward number current of the ingoing-branch component through r_t (the
launch's local ingoing-velocity projection on the window of 4 sites about r_t; branch_current) over the packet's norm, its change over the
last 2 r_s printed in place of the flatness (counted below 1 percent), the inside maximum printed beside as a lower bound, the
reconversion fraction (the crossed norm less the norm still inside r_t at the end, over the crossed norm) and the returned outgoing-branch
current printed; the packet's standard deviation raised by factors 1.25 from 1.5 r_s until its energy distribution's standard deviation
is below 0.15 in x, launched 3 standard deviations outside the barrier (1.5 r_s); the comparator the continuum Gamma averaged over the
packet's exact energy distribution; a calibration of the same measure through a cut outside the barrier is printed beside (a diagnostic,
not a criterion). The first-passage plateau, the flatness gate and the dwell estimate are withdrawn; the ray-map window fit is withdrawn
(ray_map is kept in the file and no longer called; kappa_p comes from the exact-profile ladder's record, task 3); in the validation table a
ratio outside 10 percent at x 0.3 or 0.6 carries the label "continuum Gamma; the lattice first-passage transmission differs by the printed
ratio" (8.3 (2), the comparator library's acceptance, blocking nothing).

page_scatter_v2.py. Changes against page_scatter_v1.py (builder analyst, BRIEF_page_debug_v1.md): imports page_ops_v2 and
page_core_v2 and names page_recursion_v2.py in the usage line; lattice_gamma and ray_map record and print whether the stage's sites have a
trapping zero (v1 replaced a missing r_t by r_s silently: at T6's settings, r_s/a 8 with ell 2, and at r_s/a 16 with ell 4 the averaged
flow peaks at |v| 0.9749, below 1, so those runs have no trapping surface); every lattice row carries the spec 5.2 label (counted when the
plateau flatness exceeds 0.9, the gate of SPEC_final4b_engine.md 29.9; otherwise "no plateau"), shown in the validation table. The packet,
the first-passage maximum, the continuum comparator, the ray map and every printed number are v1's, unchanged (T6 is not a code defect
of them; the builder's report gives the measured reason: where a trapping zero exists the transmitted part leaves the inside again
within 2 to 4 r_s in the launched valley, so the maximum is not a plateau).

page_scatter_v1.py (lead reviewer A, GAP03 integration; builder analyst runs it): the scattering library check of S5.
(1) The continuum comparator (GAP03_integrated_v1.md D9): the massless radial Dirac field on the Schwarzschild exterior,
    d psi/dr* = [[-W, x], [-x, W]] psi, W = kappa sqrt(f)/r, f = 1 - 1/r (r in units of r_s, x = omega r_s); the horizon-regular ingoing
    column integrated outward and decomposed on the local in/out eigenvectors of unit sigma_y current; Gamma = 1/|A_in|^2. The luminosity
    L r_s^2 = sum_kappa (64 kappa/2 pi) int x Gamma f_H dx and b = cS/(4 pi cL) against 0.0052373 and 1.6391 (REVIEWA_REVIEW_web40e.md 1.11).
(2) The lattice transmission by first passage: a narrow-band ingoing packet on the frozen stage-1 operator; Gamma_lat = the plateau of
    the number inside r_t - 2a before the re-emergence (the dwell 2 r_t + 2 r_s ln(1/(a omega)) exceeds the packet's duration for
    x <= 1), compared with the continuum Gamma averaged over the packet's exact energy distribution; the lattice-corrected luminosity.
(3) The ray-map exponent kappa_p from the back-trace of an outgoing packet (the lane's flow_page_v2 P3(a) method) against 1/(2 r_s).
(4) P36's level-1 amplitude |U(t)_{yx}|^2 across the surface.
Usage through page_recursion_v4.py --validate-scattering --out DIR --tag TAG [--rs 32 --g 0.25 --nsites 8192 --channels 1,2]."""
import json
import math
import os
import time

import numpy as np
from scipy.integrate import solve_ivp

import torch

import page_ops_v4 as OPS
import page_core_v4 as CORE

CD = OPS.CD
RD = OPS.RD
SIGY = np.array([[0.0, -1j], [1j, 0.0]])
L_CONT_RS2 = 0.0052373
B_CONT = 1.6391


# ----------------------------------------------------------------------------------------------------------- continuum (D9)
def gamma_continuum(x, kappa, rtol=1e-10, atol=1e-13, s0=-40.0, r_match=5.0, r_max_factor=200.0):
    """Gamma_kappa(x) with x = omega r_s > 0 for the massless field (any sign of kappa; the massless system is symmetric under
    kappa -> -kappa with psi -> sigma_x psi-type relabelling, so only |kappa| matters). Returns (Gamma, info)."""
    k = float(kappa); x = float(x)
    if x <= 0:
        raise ValueError("x must be positive")

    def rhs_s(s, z):                      # d/ds = r d/dr*, s = ln(r - 1): stable exponentially close to the surface
        zeta = math.exp(s); r = 1.0 + zeta; W = k * math.sqrt(zeta / r) / r
        return [r * (-W * z[0] + x * z[1]), r * (-x * z[0] + W * z[1])]

    def rhs_r(r, z):
        f = 1.0 - 1.0 / r; W = k * math.sqrt(f) / r
        return [(-W * z[0] + x * z[1]) / f, (-x * z[0] + W * z[1]) / f]

    psi0 = np.array([1.0, -1j], complex) / math.sqrt(2.0)          # unit ingoing current at the surface (psi^dagger sigma_y psi = -1)
    s1 = math.log(r_match - 1.0)
    sol1 = solve_ivp(rhs_s, (s0, s1), psi0, method="DOP853", rtol=rtol, atol=atol)
    if not sol1.success:
        raise RuntimeError(sol1.message)
    r_max = max(60.0, r_max_factor * (abs(k) + 1.0) / x)
    sol2 = solve_ivp(rhs_r, (r_match, r_max), sol1.y[:, -1], method="DOP853", rtol=rtol, atol=atol)
    if not sol2.success:
        raise RuntimeError(sol2.message)
    psi = sol2.y[:, -1]
    f = 1.0 - 1.0 / r_max; W = k * math.sqrt(f) / r_max; q = math.sqrt(x * x - W * W)
    nrm = math.sqrt(2.0 * x * q)
    u_in = np.array([x, W - 1j * q], complex) / nrm                 # local eigenvectors: current -1 (in) and +1 (out), mutually sigma_y-orthogonal
    u_out = np.array([x, W + 1j * q], complex) / nrm
    A_in = -np.vdot(u_in, SIGY @ psi)
    A_out = np.vdot(u_out, SIGY @ psi)
    gamma = 1.0 / abs(A_in) ** 2
    refl = abs(A_out) ** 2 / abs(A_in) ** 2
    return gamma, {"reflection": refl, "unitarity_error": abs(gamma + refl - 1.0), "current_in": float(np.vdot(u_in, SIGY @ u_in).real),
                   "current_out": float(np.vdot(u_out, SIGY @ u_out).real), "r_max": r_max, "nfev": int(sol1.nfev + sol2.nfev),
                   "horizon_current": float(np.vdot(psi0, SIGY @ psi0).real)}


def s_binary(n):
    n = np.clip(np.asarray(n, float), 0.0, 1.0)
    out = np.zeros_like(n)
    ok = (n > 0) & (n < 1)
    out[ok] = -n[ok] * np.log(n[ok]) - (1 - n[ok]) * np.log1p(-n[ok])
    return out


def continuum_totals(kmax=6, n_nodes=48, x_max=1.5, verbose=False):
    """L r_s^2, the stationary entropy flux S_dot r_s, the number flux and b for the 8-valley massless fermion (weights 64 kappa with the
    valleys and particle-hole counted once each inside the chain's spectrum), Gauss-Legendre on [0, x_max]."""
    xs, ws = np.polynomial.legendre.leggauss(n_nodes)
    x = 0.5 * x_max * (xs + 1.0); w = 0.5 * x_max * ws
    fH = 1.0 / (np.exp(4.0 * math.pi * x) + 1.0)
    per = {}
    cL = cS = cN = 0.0
    worst_unit = 0.0
    tab = {}
    for kappa in range(1, kmax + 1):
        G = np.zeros_like(x)
        for i, xi in enumerate(x):
            G[i], info = gamma_continuum(xi, kappa)
            worst_unit = max(worst_unit, info["unitarity_error"])
        n = G * fH
        mult = 64.0 * kappa / (2.0 * math.pi)
        Lk = mult * float(np.sum(w * x * n)); Sk = mult * float(np.sum(w * s_binary(n))); Nk = mult * float(np.sum(w * n))
        per[str(kappa)] = {"L_rs2": Lk, "S_dot_rs": Sk, "N_dot_rs": Nk}
        tab[str(kappa)] = {"x": x.tolist(), "Gamma": G.tolist()}
        cL += Lk; cS += Sk; cN += Nk
        if verbose:
            print("continuum kappa %d: L r_s^2 %.6e (cum %.6e), S_dot r_s %.6e, N_dot r_s %.6e" % (kappa, Lk, cL, Sk, Nk), flush=True)
    b = cS / (4.0 * math.pi * cL)
    return {"L_rs2": cL, "S_dot_rs": cS, "N_dot_rs": cN, "b": b, "mean_omega_over_TH": 4.0 * math.pi * cL / cN, "per_kappa": per,
            "unitarity_error_max": worst_unit, "kmax": kmax, "n_nodes": n_nodes, "x_max": x_max, "table": tab,
            "reference_L_rs2": L_CONT_RS2, "reference_b": B_CONT, "rel_dev_L": cL / L_CONT_RS2 - 1.0, "rel_dev_b": b / B_CONT - 1.0}


def gamma_table(kappa, xs):
    return np.array([gamma_continuum(float(xi), kappa)[0] if xi > 1e-6 else 0.0 for xi in xs])


# ------------------------------------------------------------------------------------------------- lattice first passage
def window_subspaces(ch, st, lo, hi, v_thr=0.1):
    """the outgoing (v > v_thr) and ingoing (v < -v_thr) subspaces of the Hann-sandwiched velocity operator on the window [lo, hi)."""
    R = ch.R
    sites = np.where((R >= lo) & (R < hi))[0]
    idx = np.concatenate([[2 * s, 2 * s + 1] for s in sites])
    hann = 0.5 - 0.5 * np.cos(2.0 * math.pi * (R[sites] - lo) / (hi - lo))
    T = np.repeat(hann, 2)
    Vel = OPS.velocity_csr(st["h"], R)
    A = Vel[idx][:, idx].toarray() * T[:, None] * T[None, :]
    Ad = torch.as_tensor(A, dtype=CD, device=ch.device)
    lam, U = torch.linalg.eigh(0.5 * (Ad + Ad.mH))
    idx_t = torch.as_tensor(idx, dtype=torch.int64, device=ch.device)
    def embed(Uw):
        full = torch.zeros((ch.n2, Uw.shape[1]), dtype=CD, device=ch.device)
        full[idx_t, :] = Uw
        return full
    out = embed(U[:, lam > v_thr]); inn = embed(U[:, lam < -v_thr])
    return out, inn, idx_t


def local_spinor(kappa, r0, kc, m=0.0, sign=+1):
    """the eigenspinor of the local symbol sin(kc) sigma_x + (kappa/r0) sigma_y + m sigma_z with the chosen sign of the energy."""
    H = math.sin(kc) * np.array([[0, 1], [1, 0]], complex) + (kappa / r0) * SIGY + m * np.diag([1.0, -1.0])
    w, U = np.linalg.eigh(H)
    return U[:, 1] if sign > 0 else U[:, 0]


def packet(ch, st, r0, sigma_x, kc, spinor, direction, energy_half=0):
    """a Gaussian packet at r0 with carrier kc and the given spinor, projected onto the ingoing (direction = -1) or outgoing (+1)
    velocity subspace of a window of 4 sigma_x about r0, then (energy_half = +-1) onto that spectral half of the stage operator,
    normalized; returns the packet, its eigenbasis coefficients and its exact energy distribution."""
    R = ch.R
    env = np.exp(-0.5 * ((R - r0) / sigma_x) ** 2) * np.exp(1j * kc * np.arange(len(R)))
    psi = (env[:, None] * spinor[None, :]).reshape(-1)
    psi_t = torch.as_tensor(psi, dtype=CD, device=ch.device)
    out, inn, _ = window_subspaces(ch, st, r0 - 4 * sigma_x, r0 + 4 * sigma_x)
    P = out if direction > 0 else inn
    psi_t = P @ (P.mH @ psi_t)
    c = st["V"].mH @ psi_t
    if energy_half > 0:
        c[st["e"] < 0] = 0.0
    elif energy_half < 0:
        c[st["e"] > 0] = 0.0
    c = c / torch.linalg.norm(c)
    psi_t = st["V"] @ c
    wgt = c.abs().pow(2)
    e = st["e"]
    return psi_t, c, {"omega_mean": float((e * wgt).sum()), "omega_std": float(torch.sqrt(((e - (e * wgt).sum()) ** 2 * wgt).sum())),
                      "weight_positive": float(wgt[e > 0].sum()), "weights": wgt, "energies": e}


def branch_current(ch, st, r_c, v_thr=0.1):
    """v3 (30.1 (5)): the ingoing- and outgoing-branch projections of the window of 4 sites about the cut r_c (window_subspaces, the
    launch's own local velocity projection) and the outward number current through the cut, i[h, Pi_{R > r_c}], restricted to each:
    W_in = in^dagger V, J_in = in^dagger J in (and the same for out), so that the current of the ingoing component at time t is
    -Re x^dagger J_in x with x = W_in exp(-i e t) c."""
    out_w, in_w, _ = window_subspaces(ch, st, r_c - 2.0, r_c + 2.0, v_thr=v_thr)
    Jc = OPS.number_current_csr(st["h"], ch.R, r_c)
    res = {}
    for lab, P in (("in", in_w), ("out", out_w)):
        Pn = P.cpu().numpy()
        JP = torch.as_tensor(Jc @ Pn, dtype=CD, device=ch.device)
        res["J_" + lab] = P.mH @ JP
        res["W_" + lab] = P.mH @ st["V"]
        res["k_" + lab] = int(P.shape[1])
    return res


# --------------------------------------------------------------------------------- v4 (31.4 (8)): the stationary scattering's libraries
_LIB = {}


def obs_dirs(obs_path=None):
    """v4: the candidate directories of the obs libraries, in order."""
    here = os.path.dirname(os.path.abspath(__file__))
    c = []
    if obs_path:
        c.append(obs_path)
    if os.environ.get("PAGE_OBS_PATH"):
        c += os.environ["PAGE_OBS_PATH"].split(os.pathsep)
    c += [os.path.join(here, "..", "obs"), os.path.join(here, "..", "obs", "code"), os.path.join(here, "..", "..", "obs", "code"), "<internal>/code"]
    return [os.path.abspath(x) for x in c]


def obs_libs(obs_path=None):
    """v4 (31.4 (8)): obs_plateau_v2, obs_plateau_v4, obs_plateau_v5 and obs_greybody_radial_v1 imported by path (each registered under its
    own name, so their imports of one another resolve to the same files). Returns {"P4", "P5", "GR", "dir"}."""
    if _LIB:
        return _LIB
    import importlib.util
    import sys
    need = ("obs_plateau_v2.py", "obs_plateau_v4.py", "obs_plateau_v5.py", "obs_greybody_radial_v1.py")
    dirs = obs_dirs(obs_path)
    found = [d for d in dirs if all(os.path.isfile(os.path.join(d, f)) for f in need)]
    if not found:
        raise ImportError("the obs libraries %s not found in %s" % (need, dirs))
    d = found[0]
    if d not in sys.path:
        sys.path.insert(0, d)
    mods = {}
    for name in ("obs_plateau_v2", "obs_plateau_v4", "obs_plateau_v5", "obs_greybody_radial_v1"):
        if name in sys.modules and os.path.abspath(getattr(sys.modules[name], "__file__", "")) == os.path.join(d, name + ".py"):
            mods[name] = sys.modules[name]
            continue
        spec = importlib.util.spec_from_file_location(name, os.path.join(d, name + ".py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        mods[name] = mod
    _LIB.update({"P4": mods["obs_plateau_v4"], "P5": mods["obs_plateau_v5"], "GR": mods["obs_greybody_radial_v1"], "dir": d})
    return _LIB


# ------------------------------------------------- J3' version 6's (d) routine (obs_raymap_solved_v6.py, copied verbatim; P4, P5, GR by path)
def outer_root(R, f):
    """obs_j3p_scope_v2's: the outermost negative-to-positive sign change of f on R, refined by a quadratic fit through the nearest five
    sites; (root, slope)."""
    s = np.nonzero((f[:-1] < 0) & (f[1:] >= 0))[0]
    if len(s) == 0:
        return None, None
    i = int(s[-1])
    lo = max(i - 2, 0); hi = min(i + 3, len(R))
    x = R[lo:hi] - R[i]; c = np.polyfit(x, f[lo:hi], 2)
    roots = np.roots(c); roots = roots[np.isreal(roots)].real
    roots = roots[(roots >= R[i] - R[i] - 1e-9) & (roots <= R[i + 1] - R[i] + 1e-9)]
    if len(roots) == 0:
        r0 = -f[i] * (R[i + 1] - R[i]) / (f[i + 1] - f[i])
    else:
        r0 = float(roots[0])
    return float(R[i] + r0), float(2 * c[0] * r0 + c[1])


def inner_root(R, f, below):
    """the annulus' inner edge: the outermost positive-to-negative sign change of f below `below` (linear interpolation); None when the
    trapped region reaches the first site."""
    s = np.nonzero((f[:-1] >= 0) & (f[1:] < 0) & (R[1:] <= below))[0]
    if len(s) == 0:
        return None
    i = int(s[-1])
    return float(R[i] - f[i] * (R[i + 1] - R[i]) / (f[i + 1] - f[i]))


class Profile:
    """the frozen record's (N, v) as smooth functions of radius: obs_plateau_v5's dense route on the propagated site fields (the degree-7
    parity splines through the site values) inside the engine's last site - 8, the analytic exterior N = 1, v = -sqrt(2 g E/R) beyond; a
    grid of 1/16 site for the symbol's ray. (v4: P5 passed in, the library imported by path.)"""

    def __init__(self, Rs, Ns, vs, g, E, P5):
        self.Rs = np.asarray(Rs, float); self.Ns = np.asarray(Ns, float); self.vs = np.asarray(vs, float)
        self.Nf = P5.site_spline(self.Rs, self.Ns, +1, 7); self.vf = P5.site_spline(self.Rs, self.vs, -1, 7)
        self.dNf = self.Nf.derivative(); self.dvf = self.vf.derivative()
        self.rmax = float(self.Rs[-1]) - 8.0
        self.c = 2.0 * g * E
        self.grid = None

    def fn(self, rho):
        rho = np.asarray(rho, float)
        ins = rho < self.rmax
        r_ = np.minimum(rho, self.rmax)
        return (np.where(ins, self.Nf(r_), 1.0), np.where(ins, self.vf(r_), -np.sqrt(self.c / np.maximum(rho, 1e-12))))

    def dfn(self, rho):
        rho = np.asarray(rho, float)
        ins = rho < self.rmax
        r_ = np.minimum(rho, self.rmax)
        return (np.where(ins, self.dNf(r_), 0.0), np.where(ins, self.dvf(r_), 0.5 * np.sqrt(self.c) * np.maximum(rho, 1e-12) ** -1.5))

    def root(self, near):
        """the spline's outer root of N + v near `near` and the slope d_R(N + v) there."""
        from scipy.optimize import brentq
        f = lambda r: float(np.sum(self.fn(np.array([r]))))
        lo, hi = near - 2.0, near + 2.0
        if f(lo) * f(hi) > 0:
            return None, None
        rr = brentq(f, lo, hi, xtol=1e-13)
        dN, dv = self.dfn(np.array([rr]))
        return float(rr), float(dN[0] + dv[0])


def blended_profile_chain(prof, s, rs, Rcut, w, RL, outer_w, kappa_ref, P4):
    """obs_plateau_v4's blended chain on the frozen record's back-averaged profile at spacing a/s (s = 1 the lattice stencil, s = 4 the
    continuum proxy): every length in units of a/s; returns (v, N, y-map rho, info) with info's kappa in units of s/a."""
    fn_s = (lambda rho: prof.fn(np.asarray(rho, float) / s))
    Rg, vg_, Ng_, gi = P4.blended_chain(fn_s, s * rs, s * Rcut, s * w, s * RL, outer_w, kappa_ref / s)
    rho = s * Rcut + s * w * np.logaddexp(0.0, (Rg - s * Rcut) / (s * w))
    wL = outer_w * s * RL
    rho = s * RL - wL * np.logaddexp(0.0, (s * RL - rho) / wL)
    return vg_, Ng_, rho, gi


def scatter_on(vg_, Ng_, rho, m, kj, omega, kappa_chain, P4, GR):
    y = kj / rho
    y[:4] = y[4]; y[-4:] = y[-5]
    sc = GR.scatter_y(vg_, Ng_, m, y, float(omega))
    rd = P4.hawking_reading(sc, float(omega), kappa_chain / (2 * math.pi))
    rd["unitarity"] = sc["unitarity"]
    rd.pop("channels", None)
    return rd


# ----------------------------------------------------------------------------------- v4 (31.4 (8)): the stationary Gamma_lat
def stationary_setup(ch, st, cfg, cut_fractions=(0.5, 0.35), w_max=4.0, room=6.0, rlead_over_rs=64.0, outer_w=0.05):
    """v4 (31.4 (8)): J3' version 6's (d) set-up on the frozen stage operator: the profile, the roots, cut A (the reading) and cut B (the
    check) with their blended chains at spacing a. Returns {"ok", "cuts": {"A": {...}, "B": {...}}, the roots, "lib_dir"} or ok False
    with the reason (no trapping root on the sites: the routine needs the trapped annulus)."""
    LIB = obs_libs(getattr(cfg, "obs_path", None))
    Rs = ch.R; vs = np.asarray(st["v_sites"], float); Ns = np.ones_like(vs)
    E = st["E"]; rs = 2.0 * cfg.g * E
    f = Ns + vs
    rHm, km = outer_root(Rs, f)
    out = {"ok": False, "r_Hm": rHm, "kappa_m": km, "lib_dir": LIB["dir"], "rs": rs}
    if rHm is None:
        out["reason"] = "no negative-to-positive root of N + v on the sites (no trapped annulus; max |v| %.4f)" % float(np.max(np.abs(vs)))
        return out
    rin = inner_root(Rs, f, rHm)
    prof = Profile(Rs, Ns, vs, cfg.g, E, LIB["P5"])
    rr_, ks_ = prof.root(rHm)
    kappa_ref = ks_ or km or 1.0 / (2.0 * rs)
    out.update({"r_in_m": rin, "r_Hm_spline": rr_, "kappa_m_spline": ks_, "kappa_ref": kappa_ref, "cuts": {}})
    rin_ = rin if rin is not None else 1.0
    for lab, fr in (("A", cut_fractions[0]), ("B", cut_fractions[1])):
        Rc = rin_ + fr * (rHm - rin_)
        w_ = min(w_max, (rHm - Rc) / room)
        Nc, vc = (float(q[0]) for q in prof.fn(np.array([Rc])))
        cu = {"R_cut": Rc, "w": w_, "fraction": fr, "N_plus_v_at_cut": Nc + vc, "trapped": bool((Nc + vc) < 0), "analyticity_estimate": math.exp(-0.5 * math.pi * math.pi * w_)}
        try:
            vg_, Ng_, rho, gi = blended_profile_chain(prof, 1, rs, Rc, w_, rlead_over_rs * rs, outer_w, kappa_ref, LIB["P4"])
            cu.update({"chain": (vg_, Ng_, rho, gi), "kappa_chain": gi["kappa"], "R_h_chain": gi["R_h"], "nsites_chain": gi["nsites"]})
        except RuntimeError as exc:
            cu["error"] = str(exc)
        out["cuts"][lab] = cu
    out["ok"] = any("chain" in c_ for c_ in out["cuts"].values())
    return out


def stationary_gamma(setup, kappa, m, omegas):
    """v4 (31.4 (8)): Gamma_lat(omega) = alpha^2 + beta^2 per cut on the blended chain (scatter_on), and the scattering's unitarity."""
    LIB = obs_libs()
    res = {}
    for lab, cu in setup.get("cuts", {}).items():
        if "chain" not in cu:
            res[lab] = {"error": cu.get("error", "no chain")}
            continue
        vg_, Ng_, rho, gi = cu["chain"]
        G, U, err = [], [], 0
        for o in omegas:
            rd = scatter_on(vg_, Ng_, rho, m, float(kappa), float(o), gi["kappa"], LIB["P4"], LIB["GR"])
            G.append(rd.get("Gamma", np.nan)); U.append(rd.get("unitarity", np.nan))
            err += int("error" in rd)
        res[lab] = {"Gamma": np.array(G, float), "unitarity": np.array(U, float), "n_errors": err}
    return res


def packet_average(en, wgt, values_fn, n_bulk=40, tails=(1e-6, 1e-4, 1e-3, 1e-2)):
    """v4 (31.4 (8), R10): the average of values_fn over the packet's energy distribution (positive energies, the particle half) by an
    equal-weight quadrature over the packet's own eigenmodes: the modes sorted by energy are cut into bins at the cumulative weights
    tails, then n_bulk equal steps, then the mirrored tails; each bin's value is taken at the bin's weighted mean energy (no
    clipping: every mode lies in a bin, the extreme bins included); the same average on every second bin edge is printed as the
    quadrature check. A value that is not finite (no Hawking-row reading at that frequency) leaves its bin out; the excluded weight
    is returned."""
    en = np.asarray(en, float); wgt = np.asarray(wgt, float)
    pos = en > 0
    e_p, w_p = en[pos], wgt[pos]
    tot = float(w_p.sum())
    if tot <= 0 or len(e_p) == 0:
        return None
    order = np.argsort(e_p); es = e_p[order]; ws = w_p[order] / tot
    cw = np.cumsum(ws)
    levels = sorted(set([0.0] + list(tails) + [float(x) for x in np.linspace(tails[-1], 1.0 - tails[-1], n_bulk + 1)] + [1.0 - t for t in tails] + [1.0]))
    cut = np.unique(np.searchsorted(cw, np.array(levels[1:-1]), side="left"))
    bounds = np.concatenate([[0], cut + 1, [len(es)]])
    bounds = np.unique(np.clip(bounds, 0, len(es)))
    reps, bw = [], []
    for i0, i1 in zip(bounds[:-1], bounds[1:]):
        if i1 <= i0:
            continue
        wb = float(ws[i0:i1].sum())
        if wb <= 0:
            continue
        reps.append(float((es[i0:i1] * ws[i0:i1]).sum() / wb)); bw.append(wb)
    reps = np.array(reps); bw = np.array(bw)
    vals = values_fn(reps)
    out = {"grid": reps, "bin_weights": bw, "n_points": int(len(reps)), "outside_share": 0.0, "positive_weight": tot}
    for key, arr in vals.items():
        arr = np.asarray(arr, float)
        ok = np.isfinite(arr)
        out[key + "_grid"] = arr
        out[key + "_unresolved_weight"] = float(bw[~ok].sum())
        if not ok.any():
            out[key] = None
            continue
        out[key] = float(np.sum(bw[ok] * arr[ok]) / bw[ok].sum())
        # the quadrature check: the same average with every pair of neighbouring bins merged (its value at the merged mean energy is the
        # interpolation of the two bins' values by weight, so the check measures the variation of the value across a bin)
        if ok.sum() >= 4:
            i = np.arange(0, (len(reps) // 2) * 2, 2)
            wm = bw[i] + bw[i + 1]
            em = (bw[i] * reps[i] + bw[i + 1] * reps[i + 1]) / wm
            okm = ok[i] & ok[i + 1]
            vm = np.interp(em[okm], reps[ok], arr[ok])
            out[key + "_half_resolution"] = float(np.sum(wm[okm] * vm) / max(wm[okm].sum(), 1e-300))
    return out


# ------------------------------------------------------------------------------------------------- lattice first passage
def lattice_gamma(ch, st, cfg, x_targets, x_std_max=0.15, sigma0_over_rs=1.5, sigma_step=1.25, r_barrier_over_rs=1.5, dt_over_rs=0.05,
                  conv_over_rs=2.0, tail_over_rs=8.0, sign=+1, n_bulk=40):
    """v4 (31.4 (8)): per target x = omega r_s, v3's packet (the sigma search, the launch 3 standard deviations outside 1.5 r_s, the
    particle half, massless) and (i) the transmission the comparator uses, the stationary Gamma_lat of the frozen stage operator (cut A the
    reading, cut B the check, unitarity printed) averaged over the packet's exact energy distribution, with the continuum Gamma averaged
    the same way and their ratio; (ii) v3's time-domain measure kept as "the finite-window branch-current diagnostic" (not a transmission,
    not a gate) with its convergence, inside maximum, returned current and calibration; (iii) the end-norm inside r_t at the end of the
    passage (the non-returned share; compare with C-capture version 6's f_nr) and the reconversion fraction 1 - (end-norm)/Gamma_lat."""
    rs = st["rs"]; R = ch.R; e, V = st["e"], st["V"]
    rt_found = st["r_t"] is not None
    rt = st["r_t"] if rt_found else rs
    if not rt_found:
        print("kappa %d: no trapping zero on the sites (max |v| %.4f): r_t taken as r_s = %.2f, as in v1" % (
            ch.kappa, float(np.max(np.abs(np.asarray(st["v_sites"], float)))), rs), flush=True)
    setup = stationary_setup(ch, st, cfg)
    if not setup["ok"]:
        print("kappa %d: the stationary Gamma_lat is not available: %s" % (ch.kappa, setup.get("reason", "no chain")), flush=True)
    bc = branch_current(ch, st, rt)
    R2 = np.repeat(R, 2)
    V_in_rt = V[torch.as_tensor(np.where(R2 < rt)[0], device=ch.device)]               # the rows inside r_t
    V_in_rt2 = V[torch.as_tensor(np.where(R2 < rt - 2.0)[0], device=ch.device)]       # the rows inside r_t - 2 (v1's inside)
    ec = e.to(CD)
    rows = []
    for xt in x_targets:
        omega_c = xt / rs
        sigma = sigma0_over_rs * rs
        search = []
        while True:
            r0 = r_barrier_over_rs * rs + 3.0 * sigma
            u0 = math.sqrt(rs / r0)
            kc = math.pi - omega_c / (1.0 + u0)                              # the ingoing second-valley carrier at the launch radius
            sp = local_spinor(ch.kappa, r0, kc, m=0.0, sign=sign)
            psi, c, info = packet(ch, st, r0, sigma, kc, sp, direction=-1, energy_half=sign)
            xstd = info["omega_std"] * rs
            search.append({"sigma_over_rs": sigma / rs, "r0_over_rs": r0 / rs, "x_mean": info["omega_mean"] * rs, "x_std": xstd})
            fits = r0 + 4.0 * sigma < R[-1] - 8.0 * rs
            if xstd < x_std_max or not fits:
                break
            sigma *= sigma_step
        T_end = (r0 + 4.0 * sigma - rt) + tail_over_rs * rs                    # the slowest inward speed is 1 + u >= 1
        dt = dt_over_rs * rs
        ts = np.arange(0.0, T_end + 0.5 * dt, dt)
        r_cal = r0 - 2.0 * sigma
        bcal = branch_current(ch, st, r_cal)
        j_in = np.zeros(len(ts)); j_out = np.zeros(len(ts)); j_cal = np.zeros(len(ts)); n_rt = np.zeros(len(ts)); n_rt2 = np.zeros(len(ts))
        for i, t in enumerate(ts):
            ct = torch.exp(-1j * ec * float(t)) * c
            xi = bc["W_in"] @ ct; xo = bc["W_out"] @ ct; xc = bcal["W_in"] @ ct
            j_in[i] = -float((xi.conj() @ (bc["J_in"] @ xi)).real)
            j_out[i] = float((xo.conj() @ (bc["J_out"] @ xo)).real)
            j_cal[i] = -float((xc.conj() @ (bcal["J_in"] @ xc)).real)
            n_rt[i] = float((V_in_rt @ ct).abs().pow(2).sum()); n_rt2[i] = float((V_in_rt2 @ ct).abs().pow(2).sum())
        norm = float(c.abs().pow(2).sum())
        G = np.concatenate([[0.0], np.cumsum(0.5 * (j_in[1:] + j_in[:-1]) * dt)]) / norm
        Gout = float(np.sum(0.5 * (j_out[1:] + j_out[:-1]) * dt)) / norm
        Gcal = float(np.sum(0.5 * (j_cal[1:] + j_cal[:-1]) * dt)) / norm
        out_cal0 = float((V[torch.as_tensor(np.where(R2 > r_cal)[0], device=ch.device)] @ c).abs().pow(2).sum()) / norm
        k_conv = max(1, int(round(conv_over_rs / dt_over_rs)))
        Gam = float(G[-1]); conv = abs(G[-1] - G[-1 - k_conv]) / max(abs(G[-1]), 1e-300)
        i_max = int(np.argmax(n_rt2))
        wgt = info["weights"].cpu().numpy(); en = info["energies"].cpu().numpy()
        end_norm = float(n_rt[-1]) / norm
        # (i) the stationary Gamma_lat and the continuum, both at the packet's own frequencies (31.4 (8); R10: no clipping)
        def vals(grid):
            out = {"Gamma_cont": np.array([gamma_continuum(float(o * rs), ch.kappa)[0] for o in grid])}
            if setup["ok"]:
                sg = stationary_gamma(setup, ch.kappa, 0.0, grid)
                for lab_, r_ in sg.items():
                    if "Gamma" in r_:
                        out["Gamma_lat_" + lab_] = r_["Gamma"]; out["unitarity_" + lab_] = r_["unitarity"]
            return out
        pa = packet_average(en, wgt, vals, n_bulk=n_bulk)
        GA = pa.get("Gamma_lat_A") if pa else None; GB = pa.get("Gamma_lat_B") if pa else None; Gc_avg = pa.get("Gamma_cont") if pa else None
        unitA = float(np.nanmax(np.abs(pa["unitarity_A_grid"]))) if (pa and pa.get("unitarity_A_grid") is not None) else None
        unitB = float(np.nanmax(np.abs(pa["unitarity_B_grid"]))) if (pa and pa.get("unitarity_B_grid") is not None) else None
        sub = slice(None, None, max(1, len(ts) // 80))
        rows.append({"label": "particle half, massless", "r_t_found": rt_found, "r_t_used": rt, "x_target": xt, "sigma_over_rs": sigma / rs, "r0_over_rs": r0 / rs,
                     "sigma_search": search, "packet_fits_chain": bool(fits), "x_mean": info["omega_mean"] * rs, "x_std": xstd, "weight_positive_energy": info["weight_positive"],
                     # the transmission the comparator uses (31.4 (8))
                     "Gamma_lat": GA, "Gamma_lat_cut_B": GB, "unitarity_cut_A_max": unitA, "unitarity_cut_B_max": unitB,
                     "Gamma_cont_avg": Gc_avg, "ratio_lat_over_cont": (GA / Gc_avg) if (GA is not None and Gc_avg) else None,
                     "packet_weight_outside_grid": pa["outside_share"] if pa else None, "grid_x": (pa["grid"] * rs).tolist() if pa else None,
                     "grid_bin_weights": pa["bin_weights"].tolist() if pa else None, "n_quadrature_points": pa["n_points"] if pa else None,
                     "Gamma_lat_half_resolution": pa.get("Gamma_lat_A_half_resolution") if pa else None,
                     "Gamma_cont_half_resolution": pa.get("Gamma_cont_half_resolution") if pa else None,
                     "Gamma_lat_unresolved_weight": pa.get("Gamma_lat_A_unresolved_weight") if pa else None,
                     "Gamma_lat_grid": pa["Gamma_lat_A_grid"].tolist() if (pa and "Gamma_lat_A_grid" in pa) else None,
                     "Gamma_cont_grid": pa["Gamma_cont_grid"].tolist() if pa else None,
                     "stationary": {k_: v_ for k_, v_ in setup.items() if k_ != "cuts"},
                     "cuts": {l_: {k_: v_ for k_, v_ in c_.items() if k_ != "chain"} for l_, c_ in setup.get("cuts", {}).items()},
                     "end_norm_inside_rt": end_norm, "reconversion_fraction": (1.0 - end_norm / GA) if GA else None,
                     # the finite-window branch-current diagnostic (v3's measure, labelled; not a transmission, not a gate)
                     "diagnostic_branch_current": {"label": "the finite-window branch-current diagnostic (not a transmission, not a gate)",
                                                   "Gamma_branch": Gam, "convergence_last_2rs": conv, "inside_max": float(n_rt2[i_max]) / norm,
                                                   "t_inside_max_over_rs": float(ts[i_max] / rs), "reconversion_fraction_v3": (Gam - end_norm) / max(Gam, 1e-300),
                                                   "returned_current_fraction": Gout / max(Gam, 1e-300), "calibration_ratio": Gcal / max(out_cal0, 1e-300),
                                                   "calibration_cut_over_rs": r_cal / rs, "norm_outside_cal_cut_t0": out_cal0, "branch_dims": [bc["k_in"], bc["k_out"]],
                                                   "T_end_over_rs": T_end / rs},
                     "t_trace_over_rs": (ts[sub] / rs).tolist(), "Gamma_branch_trace": G[sub].tolist(), "inside_rt_trace": (n_rt[sub] / norm).tolist()})
        rr = rows[-1]; dg = rr["diagnostic_branch_current"]
        print("kappa %d x %.2f (particle half, massless): packet sigma %.3f r_s at %.2f r_s (x %.3f +- %.3f; %d sigma steps%s); stationary Gamma_lat %s (cut B %s; unitarity %s, %s), "
              "continuum %s, ratio %s (%s quadrature points; half resolution: lattice %s, continuum %s; weight without a reading %s); end-norm inside r_t %.4f, "
              "reconversion fraction %s; finite-window branch-current diagnostic %.4f (change over the last 2 r_s %.2e, calibration %.4f, returned current %.4f)" % (
                  ch.kappa, xt, sigma / rs, r0 / rs, info["omega_mean"] * rs, xstd, len(search), "" if fits else ", stopped by the chain length",
                  _f(GA, "%.4f"), _f(GB, "%.4f"), _f(unitA, "%.1e"), _f(unitB, "%.1e"), _f(Gc_avg, "%.4f"), _f(rr["ratio_lat_over_cont"], "%.4f"),
                  rr["n_quadrature_points"], _f(rr["Gamma_lat_half_resolution"], "%.4f"), _f(rr["Gamma_cont_half_resolution"], "%.4f"),
                  _f(rr["Gamma_lat_unresolved_weight"], "%.1e"), end_norm, _f(rr["reconversion_fraction"], "%.4f"), Gam, conv, dg["calibration_ratio"],
                  dg["returned_current_fraction"]), flush=True)
    return rows


def _f(x, fmt):
    if x is None:
        return "-"
    try:
        return fmt % x
    except (TypeError, ValueError):
        return str(x)


def ray_map(ch, st, omega_over_TH=3.0, r0_over_rs=4.0, sigma_over_rs=0.5, n_efold=8.0, samples_per_efold=4):
    """kappa_p from the back-trace exp(+i h tau) of an outgoing packet: x(tau) - r_t = (x_0 - r_t) e^{-kappa_p tau} fitted over
    2a < x - r_t < r_t/4 (the lane's flow_page_v2 P3(a) method); against 1/(2 r_s)."""
    rs = st["rs"]; R = ch.R; e, V = st["e"], st["V"]
    rt = st["r_t"] if st["r_t"] is not None else rs
    TH = 1.0 / (4.0 * math.pi * rs); kappa0 = 1.0 / (2.0 * rs)
    r0 = r0_over_rs * rs; sx = sigma_over_rs * rs
    u0 = math.sqrt(rs / r0)
    omega_c = omega_over_TH * TH
    kc = omega_c / (1.0 - u0)                                        # the outgoing first-valley carrier
    sp = local_spinor(ch.kappa, r0, kc, sign=+1)
    psi, c, info = packet(ch, st, r0, sx, kc, sp, direction=+1)
    taus = np.arange(0.0, n_efold / kappa0 + 1e-9, 1.0 / (samples_per_efold * kappa0))
    R2 = np.repeat(R, 2)
    xs = []
    for tau in taus:
        x = (V @ (torch.exp(1j * e.to(CD) * float(tau)) * c)).abs().pow(2).cpu().numpy()
        ext = R2 > rt
        if x[ext].sum() <= 0:
            xs.append(np.nan); continue
        ip = int(np.argmax(np.where(ext, x, 0.0)))
        sel = ext.copy()
        for _ in range(3):
            rr = x * sel
            cpos = float((rr * R2).sum() / rr.sum()); wd = float(np.sqrt((rr * (R2 - cpos) ** 2).sum() / rr.sum()))
            sel = ext & (np.abs(R2 - R2[ip]) <= max(3 * wd, 2.0))
        rr = x * sel
        xs.append(float((rr * R2).sum() / rr.sum()))
    xs = np.array(xs)
    m = np.isfinite(xs) & (xs - rt > 2.0) & (xs - rt < rt / 4.0)
    out = {"r_t_found": st["r_t"] is not None, "r_t_used": rt,
           "omega_over_TH": omega_over_TH, "x0_over_rs": r0_over_rs, "kappa_declared": kappa0, "n_points": int(m.sum()),
           "tau": taus.tolist(), "x_minus_rt": (xs - rt).tolist()}
    if m.sum() >= 3:
        slope, icpt = np.polyfit(taus[m], np.log(xs[m] - rt), 1)
        out.update({"kappa_p": -slope, "kappa_p_times_2rs": -slope * 2.0 * rs})
        print("kappa %d ray map at %.1f T_H: kappa_p x 2 r_s = %.4f over %d points" % (ch.kappa, omega_over_TH, -slope * 2 * rs, int(m.sum())), flush=True)
    return out


def level1_amplitude(ch, st, times_over_rs=(0.5, 1, 2, 3, 4, 6)):
    """|U(t)_{yx}|^2 for x = r_t - a, y = r_t + a (both components summed) and the share beyond 2 r_t (P36 level 1)."""
    rs = st["rs"]; R = ch.R; e, V = st["e"], st["V"]
    rt = st["r_t"] if st["r_t"] is not None else rs
    ix = int(np.argmin(np.abs(R - (rt - 1.0)))); iy = int(np.argmin(np.abs(R - (rt + 1.0))))
    R2 = torch.as_tensor(np.repeat(R, 2), dtype=RD, device=ch.device)
    rows = []
    for cx in (0, 1):
        x = torch.zeros(ch.n2, dtype=CD, device=ch.device); x[2 * ix + cx] = 1.0
        c = V.mH @ x
        for tr in times_over_rs:
            g = V @ (torch.exp(-1j * e.to(CD) * float(tr * rs)) * c)
            rows.append({"component": cx, "t_over_rs": tr, "U_yx_sq": float(g[2 * iy: 2 * iy + 2].abs().pow(2).sum()),
                         "share_beyond_2rt": float((g.abs().pow(2) * (R2 > 2 * rt).to(RD)).sum())})
    return rows


# ------------------------------------------------------------------------------------------------------------------- driver
def validate(a, x_targets=(0.3, 0.6, 1.2)):
    """--validate-scattering: the continuum totals, per channel the stationary Gamma_lat at the packets (31.4 (8)) with the finite-window
    branch-current diagnostic beside it, the thermal comparator with the stationary lattice Gamma at the luminosity's own nodes, the
    level-1 table. v4 (R10): the massless control only."""
    if a.m != 0.0:
        raise SystemExit("--validate-scattering implements the massless control only (R10): m = %g given" % a.m)
    dev = torch.device(a.device if (a.device.startswith("cpu") or torch.cuda.is_available()) else "cpu")
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    cont = continuum_totals(verbose=True)
    print("continuum: L r_s^2 %.7e (reference 0.0052373, rel dev %.2e), b %.5f (reference 1.6391, rel dev %.2e), <omega>/T_H %.4f, unitarity %.1e" % (
        cont["L_rs2"], cont["rel_dev_L"], cont["b"], cont["rel_dev_b"], cont["mean_omega_over_TH"], cont["unitarity_error_max"]), flush=True)
    res = {"continuum": {k_: v_ for k_, v_ in cont.items() if k_ != "table"}, "lattice": {}, "level1": {}, "luminosity_stationary": {},
           "settings": {k_: v_ for k_, v_ in vars(a).items() if not callable(v_)}, "label": "particle half, massless"}
    Lstat = 0.0; Lstat_B = 0.0
    xs_gl, ws_gl = np.polynomial.legendre.leggauss(cont["n_nodes"])
    xg = 0.5 * cont["x_max"] * (xs_gl + 1.0); wg = 0.5 * cont["x_max"] * ws_gl
    fH = 1.0 / (np.exp(4.0 * math.pi * xg) + 1.0)
    for kappa in a.channel_list:
        ch = CORE.Channel(kappa, a.m, a.nsites, bool(a.half_offset), dev)
        ch.set_averager(a.ell, a.fields)
        st = ch.build_stage(a, a.E1, a.Rb * a.rs, label="stage 1 (scattering)")
        rows = lattice_gamma(ch, st, a, x_targets)
        res["lattice"][str(kappa)] = rows
        res["level1"][str(kappa)] = level1_amplitude(ch, st)                   # v3: the ray-map window fit is withdrawn (30.1 (5))
        # v4 (R10): the thermal comparator with the stationary lattice Gamma at the luminosity's own nodes (no interpolation of packet ratios)
        setup = stationary_setup(ch, st, a)
        if setup["ok"]:
            sg = stationary_gamma(setup, kappa, 0.0, xg / st["rs"])
            ent = {"x": xg.tolist()}
            for lab_, r_ in sg.items():
                if "Gamma" not in r_:
                    continue
                Lk = 64.0 * kappa / (2.0 * math.pi) * float(np.nansum(wg * xg * r_["Gamma"] * fH))
                bad = ~np.isfinite(r_["Gamma"])
                Lc_bad = 64.0 * kappa / (2.0 * math.pi) * float(np.sum((wg * xg * fH * np.array([gamma_continuum(float(x_), kappa)[0] for x_ in xg]))[bad])) if bad.any() else 0.0
                ent[lab_] = {"L_rs2": Lk, "unitarity_max": float(np.nanmax(np.abs(r_["unitarity"]))), "n_errors": r_["n_errors"], "Gamma": r_["Gamma"].tolist(),
                             "nodes_without_a_reading": int(bad.sum()), "x_without_a_reading": xg[bad].tolist(), "continuum_L_rs2_at_those_nodes": Lc_bad}
            Lstat += ent.get("A", {}).get("L_rs2", 0.0); Lstat_B += ent.get("B", {}).get("L_rs2", 0.0)
            res["luminosity_stationary"][str(kappa)] = ent
            print("kappa %d: the thermal comparator with the stationary lattice Gamma (f_H inserted; not an evolved occupation): L r_s^2 cut A %s, cut B %s; continuum %.6e; "
                  "nodes without a Hawking-row reading %s (the continuum's share there %s)" % (
                kappa, _f(ent.get("A", {}).get("L_rs2"), "%.6e"), _f(ent.get("B", {}).get("L_rs2"), "%.6e"), cont["per_kappa"][str(kappa)]["L_rs2"],
                ent.get("A", {}).get("nodes_without_a_reading"), _f(ent.get("A", {}).get("continuum_L_rs2_at_those_nodes"), "%.2e")), flush=True)
        else:
            res["luminosity_stationary"][str(kappa)] = {"error": setup.get("reason")}
        del ch, st
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    res["L_rs2_stationary_lattice_measured_channels_cut_A"] = Lstat
    res["L_rs2_stationary_lattice_measured_channels_cut_B"] = Lstat_B
    res["L_rs2_continuum_same_channels"] = sum(cont["per_kappa"][str(k)]["L_rs2"] for k in a.channel_list)
    res["ratio_L_stationary_lattice_over_continuum_same_channels"] = Lstat / max(res["L_rs2_continuum_same_channels"], 1e-300)
    res["obs_lib_dir"] = obs_libs(getattr(a, "obs_path", None))["dir"]
    res["wall_s"] = time.time() - t0
    path = os.path.join(a.out, a.tag + "_scattering.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    md = [("# %s: scattering library check (r_s/a %g, G/a^2 %g, channels %s; particle half, massless)\n" % (a.tag, a.rs, a.g, a.channel_list)),
          "continuum: L r_s^2 %.7e against 0.0052373 (rel dev %.2e); b %.5f against 1.6391 (rel dev %.2e); <omega>/T_H %.4f; kmax %d." % (
              cont["L_rs2"], cont["rel_dev_L"], cont["b"], cont["rel_dev_b"], cont["mean_omega_over_TH"], cont["kmax"]),
          "", "Gamma_lat is the stationary scattering's transmission of the frozen stage operator (J3' version 6's (d) routine; cut A the reading, cut B the check) averaged over the packet's energy distribution (31.4 (8)); the continuum Gamma is averaged the same way at the same frequencies (R10). The end-norm is the packet's norm inside r_t at the end of the passage (the non-returned share; compare with C-capture version 6's f_nr) and the reconversion fraction is 1 - end-norm/Gamma_lat. The branch current is the finite-window branch-current diagnostic (not a transmission, not a gate). Libraries: %s." % res["obs_lib_dir"],
          "", "| kappa | x target | packet sigma/r_s, r0/r_s | x mean +- std | Gamma_lat (cut A) | cut B | unitarity A, B | Gamma_cont avg | ratio | quadrature points (half-resolution lattice) | end-norm inside r_t | reconversion fraction | branch-current diagnostic (change last 2 r_s; calibration) | label | r_t on the sites |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k_, rows in res["lattice"].items():
        for r in rows:
            lab = r["label"]
            if (abs(r["x_target"] - 0.3) < 1e-9 or abs(r["x_target"] - 0.6) < 1e-9) and r["ratio_lat_over_cont"] is not None and abs(r["ratio_lat_over_cont"] - 1.0) > 0.1:
                lab += "; continuum Gamma; the lattice transmission differs by the printed ratio (8.3 (2) blocks nothing)"
            dg = r["diagnostic_branch_current"]
            md.append("| %s | %.2f | %.3f, %.2f | %.3f +- %.3f | %s | %s | %s, %s | %s | %s | %s | %.4f | %s | %.4f (%.2e; %.4f) | %s | %s |" % (
                k_, r["x_target"], r["sigma_over_rs"], r["r0_over_rs"], r["x_mean"], r["x_std"], _f(r["Gamma_lat"], "%.4f"), _f(r["Gamma_lat_cut_B"], "%.4f"),
                _f(r["unitarity_cut_A_max"], "%.1e"), _f(r["unitarity_cut_B_max"], "%.1e"), _f(r["Gamma_cont_avg"], "%.4f"), _f(r["ratio_lat_over_cont"], "%.4f"),
                "%s (%s)" % (r.get("n_quadrature_points"), _f(r.get("Gamma_lat_half_resolution"), "%.4f")), r["end_norm_inside_rt"], _f(r["reconversion_fraction"], "%.4f"), dg["Gamma_branch"], dg["convergence_last_2rs"],
                dg["calibration_ratio"], lab, ("%.2f" % r["r_t_used"]) if r["r_t_found"] else "none (r_s used)"))
    md.append("\nThe thermal comparator with the stationary lattice Gamma at the luminosity's Gauss-Legendre nodes (f_H inserted; a comparator, not an evolved occupation): L r_s^2 over the measured channels %.6e (cut A; cut B %.6e) against the continuum's %.6e for the same channels (ratio %.4f)." % (
        Lstat, Lstat_B, res["L_rs2_continuum_same_channels"], res["ratio_L_stationary_lattice_over_continuum_same_channels"]))
    md.append("ray map: the window fit is withdrawn (30.1 (5)); kappa_p comes from the exact-profile ladder's record at kappa a = 1/64 (task 3).")
    with open(os.path.join(a.out, a.tag + "_scattering.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("wrote %s (%.0f s)" % (path, res["wall_s"]), flush=True)
    return res
