#!/usr/bin/env python
"""page_scatter_v3.py. Changes against page_scatter_v2.py (builder analyst, BRIEF_page_v3_builder.md), each with the item of
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
Usage through page_recursion_v3.py --validate-scattering --out DIR --tag TAG [--rs 32 --g 0.25 --nsites 8192 --channels 1,2]."""
import json
import math
import os
import time

import numpy as np
from scipy.integrate import solve_ivp

import torch

import page_ops_v3 as OPS
import page_core_v3 as CORE

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


def lattice_gamma(ch, st, cfg, x_targets, x_std_max=0.15, sigma0_over_rs=1.5, sigma_step=1.25, r_barrier_over_rs=1.5, dt_over_rs=0.05,
                  conv_over_rs=2.0, tail_over_rs=8.0, sign=+1):
    """v3 (30.1 (5)), the branch-resolved lattice transmission: for each target x = omega r_s an ingoing packet (page_scatter_v1's packet:
    carrier pi - omega/(1 + u(r_0)), the local positive-energy spinor, projected on the ingoing velocity subspace of a window of 4
    standard deviations and on the positive spectral half) whose standard deviation is raised from 1.5 r_s by factors 1.25 until its exact
    energy distribution has a standard deviation below 0.15 in x, launched 3 standard deviations outside the barrier (r = 1.5 r_s, the
    peak of f/r^2); Gamma_lat = the time integral over the whole passage of the inward number current of the ingoing-branch component
    through r_t (branch_current) over the packet's norm, with its change over the last 2 r_s (counted below 1 percent); beside it the
    inside maximum (v1's measure, the norm inside r_t - 2, a lower bound), the reconversion fraction (the crossed norm less the norm still
    inside r_t at the end, over the crossed norm), the returned current (the outgoing-branch outward current through r_t integrated, over
    the crossed norm), a calibration (the same measure through a cut 2 standard deviations inside the launch radius, over the norm outside
    that cut at t = 0; 1 when the measure counts a full crossing), and the comparator, the continuum Gamma averaged over the packet's exact
    energy distribution. The first-passage plateau, the flatness and the dwell estimate are withdrawn (30.1 (5))."""
    rs = st["rs"]; R = ch.R; e, V = st["e"], st["V"]
    rt_found = st["r_t"] is not None
    rt = st["r_t"] if rt_found else rs
    if not rt_found:
        print("kappa %d: no trapping zero on the sites (max |v| %.4f): r_t taken as r_s = %.2f, as in v1" % (
            ch.kappa, float(np.max(np.abs(np.asarray(st["v_sites"], float)))), rs), flush=True)
    bc = branch_current(ch, st, rt)
    R2 = np.repeat(R, 2)
    V_in_rt = V[torch.as_tensor(np.where(R2 < rt)[0], device=ch.device)]               # the rows inside r_t
    V_in_rt2 = V[torch.as_tensor(np.where(R2 < rt - 2.0)[0], device=ch.device)]       # the rows inside r_t - 2 (v1's inside)
    ec = e.to(CD)
    xs_grid = np.linspace(0.02, 4.0, 60)
    Gc_grid = gamma_table(ch.kappa, xs_grid)
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
        pos = en > 0
        xe = np.clip(en[pos] * rs, xs_grid[0], xs_grid[-1])
        Gc_avg = float(np.sum(wgt[pos] * np.interp(xe, xs_grid, Gc_grid)) / max(np.sum(wgt[pos]), 1e-300))
        sub = slice(None, None, max(1, len(ts) // 80))
        rows.append({"r_t_found": rt_found, "r_t_used": rt, "x_target": xt, "sigma_over_rs": sigma / rs, "r0_over_rs": r0 / rs, "sigma_search": search,
                     "packet_fits_chain": bool(fits), "x_mean": info["omega_mean"] * rs, "x_std": xstd, "weight_positive_energy": info["weight_positive"],
                     "Gamma_lat": Gam, "convergence_last_2rs": conv, "counted": bool(conv < 0.01), "T_end_over_rs": T_end / rs,
                     "inside_max": float(n_rt2[i_max]) / norm, "t_inside_max_over_rs": float(ts[i_max] / rs), "inside_rt_end": float(n_rt[-1]) / norm,
                     "reconversion_fraction": (Gam - float(n_rt[-1]) / norm) / max(Gam, 1e-300), "returned_current_fraction": Gout / max(Gam, 1e-300),
                     "calibration_ratio": Gcal / max(out_cal0, 1e-300), "calibration_cut_over_rs": r_cal / rs, "norm_outside_cal_cut_t0": out_cal0,
                     "branch_dims": [bc["k_in"], bc["k_out"]], "Gamma_cont_avg": Gc_avg, "ratio_lat_over_cont": Gam / max(Gc_avg, 1e-300),
                     "t_trace_over_rs": (ts[sub] / rs).tolist(), "Gamma_trace": G[sub].tolist(), "j_in_trace": j_in[sub].tolist(),
                     "inside_rt_trace": (n_rt[sub] / norm).tolist()})
        print("kappa %d x %.2f: packet sigma %.3f r_s at %.2f r_s (x %.3f +- %.3f; sigma search %d steps%s); Gamma_lat %.4f (change over the last 2 r_s %.2e, %s); "
              "inside maximum %.4f at %.1f r_s; reconversion fraction %.4f (returned current %.4f); calibration %.4f; continuum average %.4f; ratio %.4f" % (
                  ch.kappa, xt, sigma / rs, r0 / rs, info["omega_mean"] * rs, xstd, len(search), "" if fits else ", stopped by the chain length", Gam, conv,
                  "counted" if conv < 0.01 else "not counted", rows[-1]["inside_max"], rows[-1]["t_inside_max_over_rs"], rows[-1]["reconversion_fraction"],
                  rows[-1]["returned_current_fraction"], rows[-1]["calibration_ratio"], Gc_avg, rows[-1]["ratio_lat_over_cont"]), flush=True)
    return rows


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
    """--validate-scattering: the continuum totals, the branch-resolved lattice Gamma per channel (v3, 30.1 (5)), the level-1 table (the
    ray-map fit is withdrawn)."""
    dev = torch.device(a.device if (a.device.startswith("cpu") or torch.cuda.is_available()) else "cpu")
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    cont = continuum_totals(verbose=True)
    print("continuum: L r_s^2 %.7e (reference 0.0052373, rel dev %.2e), b %.5f (reference 1.6391, rel dev %.2e), <omega>/T_H %.4f, unitarity %.1e" % (
        cont["L_rs2"], cont["rel_dev_L"], cont["b"], cont["rel_dev_b"], cont["mean_omega_over_TH"], cont["unitarity_error_max"]), flush=True)
    res = {"continuum": {k_: v_ for k_, v_ in cont.items() if k_ != "table"}, "lattice": {}, "ray_map": {}, "level1": {},
           "settings": {k_: v_ for k_, v_ in vars(a).items() if not callable(v_)}}
    Lcorr = 0.0
    for kappa in a.channel_list:
        ch = CORE.Channel(kappa, a.m, a.nsites, bool(a.half_offset), dev)
        ch.set_averager(a.ell, a.fields)
        st = ch.build_stage(a, a.E1, a.Rb * a.rs, label="stage 1 (scattering)")
        rows = lattice_gamma(ch, st, a, x_targets)
        res["lattice"][str(kappa)] = rows
        res["level1"][str(kappa)] = level1_amplitude(ch, st)                   # v3: the ray-map window fit is withdrawn (30.1 (5))
        # the lattice-corrected thermal luminosity: the continuum integrand times the piecewise ratio from the packets
        order = np.argsort([r["x_mean"] for r in rows])
        xs = np.array([rows[i]["x_mean"] for i in order]); ratios = np.array([rows[i]["ratio_lat_over_cont"] for i in order])
        tab = cont["table"][str(kappa)]
        xg = np.array(tab["x"]); Gg = np.array(tab["Gamma"])
        rho = np.interp(xg, xs, ratios, left=ratios[0], right=ratios[-1])
        xs_gl, ws_gl = np.polynomial.legendre.leggauss(cont["n_nodes"]); w = 0.5 * cont["x_max"] * ws_gl
        fH = 1.0 / (np.exp(4.0 * math.pi * xg) + 1.0)
        Lk = 64.0 * kappa / (2.0 * math.pi) * float(np.sum(w * xg * Gg * rho * fH))
        Lcorr += Lk
        res["lattice"][str(kappa) + "_L_rs2_corrected"] = Lk
        del ch, st
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    res["L_rs2_lattice_corrected_measured_channels"] = Lcorr
    res["L_rs2_continuum_same_channels"] = sum(cont["per_kappa"][str(k)]["L_rs2"] for k in a.channel_list)
    res["ratio_L_lattice_over_continuum_same_channels"] = Lcorr / max(res["L_rs2_continuum_same_channels"], 1e-300)
    res["wall_s"] = time.time() - t0
    path = os.path.join(a.out, a.tag + "_scattering.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    md = [("# %s: scattering library check (r_s/a %g, G/a^2 %g, channels %s)\n" % (a.tag, a.rs, a.g, a.channel_list)),
          "continuum: L r_s^2 %.7e against 0.0052373 (rel dev %.2e); b %.5f against 1.6391 (rel dev %.2e); <omega>/T_H %.4f; kmax %d." % (
              cont["L_rs2"], cont["rel_dev_L"], cont["b"], cont["rel_dev_b"], cont["mean_omega_over_TH"], cont["kmax"]),
          "", "Branch-resolved lattice transmission (30.1 (5)): Gamma_lat is the time-integrated inward current of the ingoing-branch component through r_t over the packet's norm.",
          "", "| kappa | x target | packet sigma/r_s, r0/r_s | x mean +- std | Gamma_lat | change over the last 2 r_s | inside maximum (lower bound) | reconversion fraction | returned current | calibration | Gamma_cont avg | ratio | label | r_t on the sites |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k_, rows in res["lattice"].items():
        if not k_.isdigit():
            continue
        for r in rows:
            lab = "counted" if r["counted"] else "not counted (change above 1 percent)"
            if abs(r["x_target"] - 0.3) < 1e-9 or abs(r["x_target"] - 0.6) < 1e-9:
                if abs(r["ratio_lat_over_cont"] - 1.0) > 0.1:
                    lab += "; continuum Gamma; the lattice first-passage transmission differs by the printed ratio"
            md.append("| %s | %.2f | %.3f, %.2f | %.3f +- %.3f | %.4f | %.2e | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %s | %s |" % (
                k_, r["x_target"], r["sigma_over_rs"], r["r0_over_rs"], r["x_mean"], r["x_std"], r["Gamma_lat"], r["convergence_last_2rs"], r["inside_max"],
                r["reconversion_fraction"], r["returned_current_fraction"], r["calibration_ratio"], r["Gamma_cont_avg"], r["ratio_lat_over_cont"], lab,
                ("%.2f" % r["r_t_used"]) if r["r_t_found"] else "none (r_s used)"))
    md.append("\nlattice-corrected thermal luminosity over the measured channels: %.6e against the continuum's %.6e for the same channels (ratio %.4f)." % (
        Lcorr, res["L_rs2_continuum_same_channels"], res["ratio_L_lattice_over_continuum_same_channels"]))
    md.append("ray map: the window fit is withdrawn (30.1 (5)); kappa_p comes from the exact-profile ladder's record at kappa a = 1/64 (task 3).")
    with open(os.path.join(a.out, a.tag + "_scattering.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("wrote %s (%.0f s)" % (path, res["wall_s"]), flush=True)
    return res
