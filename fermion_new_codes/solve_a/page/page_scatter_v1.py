#!/usr/bin/env python
"""page_scatter_v1.py (lead reviewer A, GAP03 integration; builder analyst runs it): the scattering library check of S5.
(1) The continuum comparator (GAP03_integrated_v1.md D9): the massless radial Dirac field on the Schwarzschild exterior,
    d psi/dr* = [[-W, x], [-x, W]] psi, W = kappa sqrt(f)/r, f = 1 - 1/r (r in units of r_s, x = omega r_s); the horizon-regular ingoing
    column integrated outward and decomposed on the local in/out eigenvectors of unit sigma_y current; Gamma = 1/|A_in|^2. The luminosity
    L r_s^2 = sum_kappa (64 kappa/2 pi) int x Gamma f_H dx and b = cS/(4 pi cL) against 0.0052373 and 1.6391 (REVIEWA_REVIEW_web40e.md 1.11).
(2) The lattice transmission by first passage: a narrow-band ingoing packet on the frozen stage-1 operator; Gamma_lat = the plateau of
    the number inside r_t - 2a before the re-emergence (the dwell 2 r_t + 2 r_s ln(1/(a omega)) exceeds the packet's duration for
    x <= 1), compared with the continuum Gamma averaged over the packet's exact energy distribution; the lattice-corrected luminosity.
(3) The ray-map exponent kappa_p from the back-trace of an outgoing packet (the lane's flow_page_v2 P3(a) method) against 1/(2 r_s).
(4) P36's level-1 amplitude |U(t)_{yx}|^2 across the surface.
Usage through page_recursion_v1.py --validate-scattering --out DIR --tag TAG [--rs 32 --g 0.25 --nsites 8192 --channels 1,2]."""
import json
import math
import os
import time

import numpy as np
from scipy.integrate import solve_ivp

import torch

import page_ops_v1 as OPS
import page_core_v1 as CORE

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


def lattice_gamma(ch, st, cfg, x_targets, r0_over_rs=12.0, sigma_over_rs=1.5, dt_over_rs=0.25, sign=+1):
    """Gamma_lat at the targets x = omega r_s: for each, a narrow-band ingoing packet; the number inside r_t - 2a against time; the plateau
    (its maximum before the re-emergence) over the packet's norm; the continuum Gamma averaged over the packet's exact energy distribution
    (positive energies; the hole half is reported separately)."""
    rs = st["rs"]; R = ch.R; e, V = st["e"], st["V"]
    rt = st["r_t"] if st["r_t"] is not None else rs
    r0 = r0_over_rs * rs; sx = sigma_over_rs * rs
    u0 = math.sqrt(rs / r0)
    inside = torch.as_tensor(np.repeat(R < rt - 2.0, 2), dtype=RD, device=ch.device)
    rows = []
    xs_grid = np.linspace(0.02, 4.0, 60)
    Gc_grid = gamma_table(ch.kappa, xs_grid)
    for xt in x_targets:
        omega_c = xt / rs
        kc = math.pi - omega_c / (1.0 + u0)                          # the ingoing second-valley carrier at the launch radius
        sp = local_spinor(ch.kappa, r0, kc, m=0.0, sign=sign)
        psi, c, info = packet(ch, st, r0, sx, kc, sp, direction=-1, energy_half=sign)
        t_max = (r0 + 2.0 * rt) * 1.5 + 12.0 * rs
        ts = np.arange(0.0, t_max, dt_over_rs * rs)
        Nin = []
        for t in ts:
            x = V @ (torch.exp(-1j * e.to(CD) * float(t)) * c)
            Nin.append(float((x.abs().pow(2) * inside).sum()))
        Nin = np.array(Nin)
        i_max = int(np.argmax(Nin))
        plateau = float(Nin[i_max])
        later = Nin[min(i_max + int(2.0 / dt_over_rs), len(Nin) - 1)]
        # the continuum Gamma averaged over the packet's exact energy distribution (particle half)
        wgt = info["weights"].cpu().numpy(); en = info["energies"].cpu().numpy()
        pos = en > 0
        xe = np.clip(en[pos] * rs, xs_grid[0], xs_grid[-1])
        Gc_avg = float(np.sum(wgt[pos] * np.interp(xe, xs_grid, Gc_grid)) / max(np.sum(wgt[pos]), 1e-300))
        rows.append({"x_target": xt, "x_mean": info["omega_mean"] * rs, "x_std": info["omega_std"] * rs, "weight_positive_energy": info["weight_positive"],
                     "Gamma_lat": plateau, "t_plateau_over_rs": float(ts[i_max] / rs), "plateau_flatness_2rs_later": float(later / max(plateau, 1e-300)),
                     "Gamma_cont_avg": Gc_avg, "ratio_lat_over_cont": plateau / max(Gc_avg, 1e-300), "N_inside_trace": Nin[:: max(1, len(Nin) // 60)].tolist(),
                     "t_trace_over_rs": (ts[:: max(1, len(ts) // 60)] / rs).tolist()})
        print("kappa %d x %.2f (mean %.2f +- %.2f): Gamma_lat %.4f (plateau at %.1f r_s, flatness %.3f) against the continuum average %.4f: ratio %.4f" % (
            ch.kappa, xt, rows[-1]["x_mean"], rows[-1]["x_std"], plateau, rows[-1]["t_plateau_over_rs"], rows[-1]["plateau_flatness_2rs_later"], Gc_avg, rows[-1]["ratio_lat_over_cont"]), flush=True)
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
    out = {"omega_over_TH": omega_over_TH, "x0_over_rs": r0_over_rs, "kappa_declared": kappa0, "n_points": int(m.sum()),
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
    """--validate-scattering: the continuum totals, the lattice first-passage Gamma per channel, the ray map, the level-1 table."""
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
        res["ray_map"][str(kappa)] = ray_map(ch, st)
        res["level1"][str(kappa)] = level1_amplitude(ch, st)
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
          "", "| kappa | x target | x mean +- std | Gamma_lat (plateau) | flatness | Gamma_cont avg | ratio |", "|---|---|---|---|---|---|---|"]
    for k_, rows in res["lattice"].items():
        if not k_.isdigit():
            continue
        for r in rows:
            md.append("| %s | %.2f | %.2f +- %.2f | %.4f | %.3f | %.4f | %.4f |" % (k_, r["x_target"], r["x_mean"], r["x_std"], r["Gamma_lat"], r["plateau_flatness_2rs_later"], r["Gamma_cont_avg"], r["ratio_lat_over_cont"]))
    md.append("\nlattice-corrected thermal luminosity over the measured channels: %.6e against the continuum's %.6e for the same channels (ratio %.4f)." % (
        Lcorr, res["L_rs2_continuum_same_channels"], res["ratio_L_lattice_over_continuum_same_channels"]))
    for k_, rm in res["ray_map"].items():
        md.append("ray map kappa %s: kappa_p x 2 r_s = %s over %d points (omega = %.1f T_H)." % (k_, ("%.4f" % rm["kappa_p_times_2rs"]) if "kappa_p_times_2rs" in rm else "no fit", rm["n_points"], rm["omega_over_TH"]))
    with open(os.path.join(a.out, a.tag + "_scattering.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("wrote %s (%.0f s)" % (path, res["wall_s"]), flush=True)
    return res
