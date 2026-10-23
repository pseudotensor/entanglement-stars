#!/usr/bin/env python3
"""flow_chart_v1.py: the flow-chart one-particle operator of H_* on a chain (first draft, lead; builder completes).

Chart: flat ruler psi = 1, lapse N = 1 (simpler version: the flow v(R) is a declared input, marked in every record).
Chain: two components per site, index 2n + s. On site m_n SZ + y_n SY; sine hop h[n, n+1] = -(i/2) SX;
flow hop h[n, n+2] = -(i/4) vbar_n 1, vbar_n = (v_n + v_{n+2})/2 (horizon_cut.py / reviewer B F2.4; the range-two hop is
the law's own momentum current sin(2p)/2, JUDGMENT_web40c sec. 12 item 1).
Uniform symbol: h(k) = m SZ + sin k SX + v sin 2k /2; bands +-sqrt(m^2 + sin^2 k) + v sin 2k /2.
Planar chain: m_n = sqrt(m^2 + sin^2 ky + sin^2 kz), y_n = 0; radial channel: m_n = m, y_n = kappa_j / R_n, R_n = n + 1.
Units a = 1. Specs: solve/specs/SPEC_task3_prescribed_flow.md, SPEC_task4_injection.md.
BUILDER: complete the modes marked TODO, add the GPU eigh (torch) for 2N > 4000, the JSON records and the launchers.
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import brentq

SX = np.array([[0, 1], [1, 0]], complex)
SY = np.array([[0, -1j], [1j, 0]], complex)
SZ = np.diag([1.0, -1.0]).astype(complex)
I2 = np.eye(2, dtype=complex)
ETA = np.array([1.0, -1.0, -1.0, 1.0])
RECT = {"light": np.array([0.0, 0.5, 0.5, 1.0]), "1223": np.array([1.0, 2.0, 2.0, 3.0]) / 3.0}


# ---------------------------------------------------------------- operator
def chain_hamiltonian(mz, yy, v):
    """Dense 2N x 2N Hermitian chain generator (sec. 1 of the task 3 spec)."""
    mz, yy, v = (np.asarray(x, float) for x in (mz, yy, v))
    n = len(mz)
    h = np.zeros((2 * n, 2 * n), complex)
    for i in range(n):
        h[2 * i:2 * i + 2, 2 * i:2 * i + 2] = mz[i] * SZ + yy[i] * SY
    for i in range(n - 1):
        h[2 * i:2 * i + 2, 2 * i + 2:2 * i + 4] = -0.5j * SX
    for i in range(n - 2):
        vbar = 0.5 * (v[i] + v[i + 2])
        h[2 * i:2 * i + 2, 2 * i + 4:2 * i + 6] = -0.25j * vbar * I2
    h = h + np.triu(h, 1).conj().T
    return h


def radial_fields(nsites, m, kappa_j):
    R = np.arange(1, nsites + 1, dtype=float)
    return np.full(nsites, float(m)), kappa_j / R, R


def planar_fields(nsites, m, ky=0.0, kz=0.0):
    mc = np.sqrt(m * m + np.sin(ky) ** 2 + np.sin(kz) ** 2)
    x = np.arange(nsites, dtype=float) - (nsites - 1) / 2
    return np.full(nsites, mc), np.zeros(nsites), x


def pg_profile(R, rs, Rm):
    """Declared input: v = -sqrt(rs/R) outside the matter ball, -sqrt(rs/Rm^3) R inside (regular centre)."""
    R = np.asarray(R, float)
    return np.where(R >= Rm, -np.sqrt(rs / np.maximum(R, 1e-300)), -np.sqrt(rs / Rm ** 3) * R)


def tanh_profile(x, delta, width):
    """The record's planar profile; kappa = delta/width; r_s := 1/(2 kappa)."""
    return -1.0 + delta * np.tanh(np.asarray(x, float) / width)


# ---------------------------------------------------------------- symbol (part a)
def bands(k, mc, v):
    E = np.sqrt(mc * mc + np.sin(k) ** 2)
    dop = v * np.sin(2 * k) / 2
    return E + dop, -E + dop, E


def group_velocity(k, mc, v):
    E = np.sqrt(mc * mc + np.sin(k) ** 2)
    return np.sin(k) * np.cos(k) / E + v * np.cos(2 * k)


def turning_ratio(k, mc=0.0):
    """R_turn/r_s = cos^2 2k/cos^2 k (1 + mc^2/sin^2 k); massless: the record's cos^2(2ap)/cos^2(ap)."""
    k = np.asarray(k, float)
    return np.cos(2 * k) ** 2 / np.cos(k) ** 2 * (1 + mc * mc / np.sin(k) ** 2)


def vg_max_at_surface(mc=0.0, w=1.0, ngrid=200001):
    k = np.linspace(1e-6, np.pi / 2 - 1e-6, ngrid)
    vg = group_velocity(k, mc, -w)
    i = int(np.argmax(vg))
    return float(vg[i]), float(k[i])


def ring_dispersion_check(nsites, mc, w):
    """C1: periodic uniform-flow ring against the symbol (control only; never for cuts or conversions)."""
    h = chain_hamiltonian(np.full(nsites, mc), np.zeros(nsites), np.full(nsites, -w))
    # periodic closure of the two hops
    for d, blk in ((1, -0.5j * SX), (2, -0.25j * (-w) * I2)):
        for i in range(nsites - d, nsites):
            j = (i + d) % nsites
            h[2 * i:2 * i + 2, 2 * j:2 * j + 2] = blk
            h[2 * j:2 * j + 2, 2 * i:2 * i + 2] = blk.conj().T
    ev = np.sort(eigh(h, eigvals_only=True))
    k = 2 * np.pi * np.arange(nsites) / nsites
    op, om, _ = bands(k, mc, -w)
    return float(np.max(np.abs(ev - np.sort(np.concatenate([op, om])))))


# ---------------------------------------------------------------- states and evolution
def spectrum(h):
    """BUILDER: route 2N > 4000 to torch.linalg.eigh on a free GPU (paper's lattice_entropy.spectrum pattern)."""
    return eigh(h, driver="evr", check_finite=False)


def flat_sea(h0):
    e, Q = spectrum(h0)
    return Q[:, e < 0], (e < 0).astype(float)


def ramp_unitary(mz, yy, vfinal, ramp_time, dt, envelope=lambda s: np.sin(np.pi * s / 2) ** 2):
    """Dense product of exact midpoint exponentials over the formation ramp (horizon_cut's ramp, F2.32)."""
    steps = max(1, int(np.ceil(ramp_time / dt)))
    step = ramp_time / steps
    U = np.eye(2 * len(mz), dtype=complex)
    for i in range(steps):
        s = min((i + 0.5) * step / ramp_time, 1.0)
        e, Q = spectrum(chain_hamiltonian(mz, yy, envelope(s) * vfinal))
        U = (Q * np.exp(-1j * step * e)) @ (Q.conj().T @ U)
    return U


def propagate(Q, e, t, vec):
    return Q @ (np.exp(-1j * e * t)[:, None] * (Q.conj().T @ vec)) if vec.ndim == 2 else \
        Q @ (np.exp(-1j * e * t) * (Q.conj().T @ vec))


# ---------------------------------------------------------------- packets
def positive_spinor(k, mc):
    E = np.sqrt(mc * mc + np.sin(k) ** 2)
    u = np.stack([E + mc, np.sin(k) * np.ones_like(E)], axis=-1)
    return u / np.linalg.norm(u, axis=-1, keepdims=True)


def branch_momenta(omega_c, sigma, mc, vext, outgoing=True, nk=4001, kmax=np.pi / 2 - 1e-6):
    """Momenta on the chosen branch of the uniform exterior with Killing frequency in omega_c +- 5 sigma.

    Returns k, omega(k), |A(k)|^2 with |A|^2 dk = Gaussian(omega) domega (packet measure of the spec, sec. 5)."""
    sgn = 1.0 if outgoing else -1.0
    k = sgn * np.linspace(1e-6, kmax, nk)
    om = bands(k, mc, vext)[0]
    vg = group_velocity(k, mc, vext)
    ok = (vg > 0) if outgoing else (vg < 0)
    ok &= np.abs(om - omega_c) < 5 * sigma
    if not np.any(ok):
        raise ValueError("no momenta on the branch inside the frequency window (below threshold?)")
    k, om, vg = k[ok], om[ok], vg[ok]
    a2 = np.exp(-(om - omega_c) ** 2 / (2 * sigma * sigma)) * np.abs(vg)
    return k, om, a2 / np.trapz(a2, k)


def build_packet(nsites, x_index, k, a2, mc):
    """w_n = sum_k A(k) e^{ik(n - x0)} u_+(k), normalized; x_index is the site index of the centre."""
    n = np.arange(nsites)
    dk = np.gradient(k)
    A = np.sqrt(a2 * dk)
    phase = np.exp(1j * np.outer(n - x_index, k))
    u = positive_spinor(k, mc)
    w = (phase * A) @ u
    w = w.reshape(-1)
    return w / np.linalg.norm(w)


def threshold(mc, vext):
    """omega_th = mc sqrt(1 - vext^2) on a uniform exterior (min over k of the outgoing band); PG exterior: m."""
    return mc * np.sqrt(max(0.0, 1 - vext * vext))


def fermi_average(k, om, a2, theta, kappa):
    n = 1.0 / (np.exp(2 * np.pi * om / (kappa * theta)) + 1.0)
    return float(np.trapz(a2 * n, k))


def invert_theta(nw, k, om, a2, kappa, lo=0.2, hi=3.0):
    return brentq(lambda th: fermi_average(k, om, a2, th, kappa) - nw, lo, hi, xtol=1e-10)


def occupation_backward(w, Uramp, Qf, ef, t_after_ramp, V0):
    """n_w = ||P_0 U(0, t_l) w||^2, P_0 the initial normal-sea projector (the record's protocol)."""
    u = Qf @ (np.exp(+1j * ef * t_after_ramp) * (Qf.conj().T @ w))
    u = Uramp.conj().T @ u
    return float(np.sum(np.abs(V0.conj().T @ u) ** 2))


# ---------------------------------------------------------------- entropies (part c)
def binary_entropy(p):
    p = np.clip(np.asarray(p).real, 0.0, 1.0)
    q = p[(p > 0) & (p < 1)]
    return float(np.sum(-q * np.log(q) - (1 - q) * np.log1p(-q)))


def cut_entropies(orbitals, occupations, cut_site):
    """S_A, S_B, S_all, half mutual information for the cut between sites cut_site-1 and cut_site."""
    C = orbitals @ orbitals.conj().T
    c = 2 * cut_site
    SA = binary_entropy(eigh(C[:c, :c], eigvals_only=True, driver="evr", check_finite=False))
    SB = binary_entropy(eigh(C[c:, c:], eigvals_only=True, driver="evr", check_finite=False))
    Sall = binary_entropy(occupations)
    return SA, SB, Sall, 0.5 * (SA + SB - Sall)


def quadrant_quadrature(nk):
    """The record's transverse quadrature (horizon_cut.run): midpoints in (0, pi/2), weight 1 or 2 over nk^2."""
    ks = (np.arange(nk) + 0.5) * np.pi / (2 * nk)
    out = []
    for i in range(nk):
        for j in range(i, nk):
            out.append((ks[i], ks[j], (1.0 if i == j else 2.0) / nk ** 2))
    return out


def channel_masses(rect, aM, ky, kz):
    return np.sqrt((aM ** 2) * RECT[rect] + np.sin(ky) ** 2 + np.sin(kz) ** 2)


def assemble_packets(table, rect, aM, nk):
    """table: dict channel mass (rounded 1e-12) -> entropy; returns S_s, S_gr, S_*  (factor 2 for the two blocks)."""
    S = np.zeros(4)
    for ky, kz, wgt in quadrant_quadrature(nk):
        for s, mc in enumerate(channel_masses(rect, aM, ky, kz)):
            S[s] += 2 * wgt * table[round(float(mc), 12)]
    return S, float(ETA @ S), float(np.sum(S))


# ---------------------------------------------------------------- packet diagnostics (parts a, d; task 4)
def density(psi):
    return (np.abs(psi.reshape(-1, 2)) ** 2).sum(axis=1)


def centre_width(psi, X):
    rho = density(psi)
    rho = rho / rho.sum()
    c = float(rho @ X)
    return c, float(np.sqrt(rho @ (X - c) ** 2))


def momentum_distribution(psi, nfft=None):
    """Per-component FFT of the packet over the chain (open chain: approximate; use for carrier readings)."""
    a = psi.reshape(-1, 2)
    n = nfft or 4 * len(a)
    k = 2 * np.pi * np.fft.fftfreq(n)
    p = np.abs(np.fft.fft(a, n=n, axis=0)) ** 2
    p = p.sum(axis=1)
    return k, p / p.sum()


def track(Q, e, psi0, times, X, regions):
    """Centre, width, carrier and region norms along the evolution under a fixed generator."""
    rec = []
    for t in times:
        psi = propagate(Q, e, t, psi0)
        c, wdt = centre_width(psi, X)
        k, p = momentum_distribution(psi)
        rho = density(psi)
        rec.append({"t": float(t), "centre": c, "width": wdt, "carrier": float(k @ p),
                    **{name: float(rho[sel].sum()) for name, sel in regions.items()}})
    return rec


# ---------------------------------------------------------------- modes
def mode_symbol(args):
    out = {"C1_ring_max_error": {}, "vg_max_surface": {}, "turning_table": {}}
    for mc in (0.0, 0.1, 0.3):
        for w in (0.0, 0.5, 1.0, 1.5, 2.0):
            out["C1_ring_max_error"][f"mc={mc},w={w}"] = ring_dispersion_check(64, mc, w)
        out["vg_max_surface"][f"mc={mc}"] = vg_max_at_surface(mc)
        ks = np.pi * np.array([1 / 32, 1 / 16, 1 / 8, 3 / 16, 1 / 4, 5 / 16, 3 / 8])
        out["turning_table"][f"mc={mc}"] = {"k": ks.tolist(), "R_turn_over_rs": turning_ratio(ks, mc).tolist()}
    out["bound_check_9_over_8"] = out["vg_max_surface"]["mc=0.0"][0] - 9 / 8
    return out


def mode_convert(args):
    """Part (b) on the planar chain: one channel mass, the massless control or the massive scan."""
    kappa = 1.0 / args.inv_kappa
    delta, width = args.delta, args.delta * args.inv_kappa
    N = args.nsites
    mz, yy, x = planar_fields(N, args.mass)
    vfinal = tanh_profile(x, delta, width)
    vext = -1.0 + delta
    h0 = chain_hamiltonian(mz, yy, np.zeros(N))
    V0, occ0 = flat_sea(h0)
    Uramp = ramp_unitary(mz, yy, vfinal, args.ramp_time, args.dt)
    ef, Qf = spectrum(chain_hamiltonian(mz, yy, vfinal))
    unit = float(np.linalg.norm(Uramp.conj().T @ Uramp - np.eye(2 * N)))
    mc = float(mz[0])
    om_th = threshold(mc, vext)
    x0 = int(round((N - 1) / 2 + args.x0_over_kappa / kappa))
    results = {}
    for r in args.ratios:
        omega_c = r * kappa
        sigma = min(omega_c / 4, max((omega_c - om_th) / 3, 1e-12))
        if omega_c - 3 * sigma <= om_th:
            results[f"r={r}"] = {"skipped": "below threshold", "omega_th": om_th}
            continue
        k, om, a2 = branch_momenta(omega_c, sigma, mc, vext, outgoing=True)
        w = build_packet(N, x0, k, a2, mc)
        plateau = {}
        for kt in np.arange(140, 201, 10):
            tl = kt / kappa - args.ramp_time
            nw = occupation_backward(w, Uramp, Qf, ef, tl, V0)
            plateau[str(int(kt))] = {"n_w": nw, "theta_r": invert_theta(nw, k, om, a2, kappa)}
        th = np.array([p["theta_r"] for p in plateau.values()])
        results[f"r={r}"] = {"plateau": plateau, "theta_mean": float(th.mean()), "theta_spread": float(np.ptp(th)),
                             "c_r": float((1 - th.mean()) * args.inv_kappa ** 2), "c_pred": 1 / 3 + 5 * r * r / 18,
                             "sigma_omega": sigma, "omega_th": om_th, "F_w(1)": fermi_average(k, om, a2, 1.0, kappa)}
    return {"label": "prescribed flow (simpler version); planar tanh; v declared", "kappa_a": kappa,
            "mass": args.mass, "m_over_kappa": args.mass / kappa, "delta": delta, "width": width, "N": N,
            "x0_index": x0, "ramp_time": args.ramp_time, "dt": args.dt, "C2_unitarity": unit, "results": results}


def mode_cut(args):
    """Part (c): the per-channel entropy table at the surface cut, flat and formed, at the hold times."""
    kappa = 1.0 / args.inv_kappa
    delta, width = args.delta, args.delta * args.inv_kappa
    N = args.nsites
    masses = set()
    for rect in ("light", "1223"):
        for aM in args.aM:
            for ky, kz, _ in quadrant_quadrature(args.nk):
                for mc in channel_masses(rect, aM, ky, kz):
                    masses.add(round(float(mc), 12))
    masses.add(0.0)
    table = {"flat": {}, "formed": {str(th): {} for th in args.hold_over_kappa}}
    for mc in sorted(masses):
        mz, yy, x = planar_fields(N, mc)
        vfinal = tanh_profile(x, delta, width)
        V0, occ0 = flat_sea(chain_hamiltonian(mz, yy, np.zeros(N)))
        table["flat"][mc] = cut_entropies(V0, occ0, N // 2)
        Uramp = ramp_unitary(mz, yy, vfinal, args.ramp_time, args.dt)
        ef, Qf = spectrum(chain_hamiltonian(mz, yy, vfinal))
        Phi = Uramp @ V0
        for th in args.hold_over_kappa:
            table["formed"][str(th)][mc] = cut_entropies(propagate(Qf, ef, th / kappa, Phi), occ0, N // 2)
    assembled = {}
    for rect in ("light", "1223"):
        for aM in args.aM:
            key = f"{rect},aM={aM}"
            flat = {m: v[0] for m, v in table["flat"].items()}
            assembled[key] = {"flat": assemble_packets(flat, rect, aM, args.nk)}
            for th in args.hold_over_kappa:
                formed = {m: v[0] for m, v in table["formed"][str(th)].items()}
                assembled[key][f"formed,hold={th}"] = assemble_packets(formed, rect, aM, args.nk)
    return {"label": "prescribed flow (simpler version); planar tanh; v declared", "kappa_a": kappa, "N": N,
            "nk": args.nk, "delta": delta, "width": width, "ramp_time": args.ramp_time, "dt": args.dt,
            "massless_flat_SA": table["flat"][0.0][0], "massless_formed_SA": {th: table["formed"][str(th)][0.0][0]
                                                                              for th in args.hold_over_kappa},
            "table_flat": {str(m): v for m, v in table["flat"].items()},
            "table_formed": {th: {str(m): v for m, v in d.items()} for th, d in table["formed"].items()},
            "assembled": assembled}


def mode_radial_packets(args):
    """Parts (a) dynamic, (d) and task 4 flow half: one fixed PG channel, packets launched and tracked.

    BUILDER: wire the launch lists of the spec (A2, A3, D1 to D4; task 4 sec. 2) through --launch; this draft
    does one launch per call."""
    N = args.nsites
    mz, yy, R = radial_fields(N, args.mass, args.kappa_j)
    v = pg_profile(R, args.rs, args.Rm)
    e, Q = spectrum(chain_hamiltonian(mz, yy, v))
    if args.launch == "dust":
        k = np.array([0.0]); a2 = np.array([1.0])
        n = np.arange(N); env = np.exp(-(R - args.R0) ** 2 / (4 * args.sigma_x ** 2))
        u = positive_spinor(0.0, args.mass)
        psi0 = (env[:, None] * u[None, :]).reshape(-1); psi0 /= np.linalg.norm(psi0)
        times = np.arange(0, args.tmax + 1e-9, args.tsample)
        law = (args.R0 ** 1.5 - 1.5 * np.sqrt(args.rs) * times) ** (2 / 3)
        rec = track(Q, e, psi0, times, R, {"inside_rs": R < args.rs, "inside_Rm": R < args.Rm})
        return {"label": "prescribed flow (simpler version); PG radial; v declared", "launch": "dust",
                "rs": args.rs, "Rm": args.Rm, "mass": args.mass, "sigma_x": args.sigma_x, "R0": args.R0,
                "dust_law": law.tolist(), "track": rec}
    if args.launch == "carrier":   # outgoing or ingoing Gaussian at carrier k0 (parts a, d4)
        n = np.arange(N); env = np.exp(-(R - args.R0) ** 2 / (4 * args.sigma_x ** 2))
        u = positive_spinor(args.k0, args.mass)
        psi0 = (env * np.exp(1j * args.k0 * R))[:, None] * u[None, :]
        psi0 = psi0.reshape(-1); psi0 /= np.linalg.norm(psi0)
        times = np.arange(0, args.tmax + 1e-9, args.tsample)
        rec = track(Q, e, psi0, times, R, {"inside_rs": R < args.rs, "beyond_2rs": R > 2 * args.rs})
        c0, c1 = rec[0]["centre"], rec[min(len(rec) - 1, 1)]["centre"]
        vloc = float(pg_profile(args.R0, args.rs, args.Rm))
        kk = np.linspace(args.k0 - 4 / (2 * args.sigma_x), args.k0 + 4 / (2 * args.sigma_x), 801)
        wk = np.exp(-(kk - args.k0) ** 2 * 2 * args.sigma_x ** 2); wk /= wk.sum()
        return {"label": "prescribed flow (simpler version); PG radial; v declared", "launch": "carrier",
                "k0": args.k0, "R0": args.R0, "R_turn_law": float(args.rs * turning_ratio(abs(args.k0), args.mass)),
                "vg_symbol_packet_avg": float(wk @ group_velocity(kk, args.mass, vloc)),
                "vg_measured_first_step": (c1 - c0) / max(rec[1]["t"], 1e-12) if len(rec) > 1 else None,
                "track": rec}
    if args.launch == "ingoing":   # Killing-frequency packet on the ingoing branch, built in the far exterior
        vext = float(pg_profile(args.R0, args.rs, args.Rm))
        k, om, a2 = branch_momenta(args.omega, args.omega / 4, args.mass, vext, outgoing=False)
        psi0 = build_packet(N, int(round(args.R0 - 1)), k, a2, args.mass)
        times = np.arange(0, args.tmax + 1e-9, args.tsample)
        rec = track(Q, e, psi0, times, R, {"inside_rs": R < args.rs, "beyond_Rref": R > 2 * args.rs})
        return {"label": "prescribed flow (simpler version); PG radial; v declared", "launch": "ingoing",
                "omega": args.omega, "omega_rs": args.omega * args.rs, "R0": args.R0, "track": rec}
    raise ValueError(args.launch)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["symbol", "convert", "cut", "radial"])
    p.add_argument("--inv-kappa", type=float, default=8.0)
    p.add_argument("--delta", type=float, default=0.5)
    p.add_argument("--nsites", type=int, default=1024)
    p.add_argument("--mass", type=float, default=0.0)
    p.add_argument("--ramp-time", type=float, default=30.0)
    p.add_argument("--dt", type=float, default=0.5)
    p.add_argument("--x0-over-kappa", type=float, default=4.0)
    p.add_argument("--ratios", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    p.add_argument("--nk", type=int, default=12)
    p.add_argument("--aM", type=float, nargs="+", default=[0.3, 0.1, 0.03])
    p.add_argument("--hold-over-kappa", type=float, nargs="+", default=[0.0, 12.0, 24.0, 36.0])
    p.add_argument("--rs", type=float, default=32.0)
    p.add_argument("--Rm", type=float, default=8.0)
    p.add_argument("--kappa-j", type=float, default=-1.0)
    p.add_argument("--launch", default="dust")
    p.add_argument("--R0", type=float, default=128.0)
    p.add_argument("--k0", type=float, default=0.0)
    p.add_argument("--omega", type=float, default=0.03125)
    p.add_argument("--sigma-x", type=float, default=8.0)
    p.add_argument("--tmax", type=float, default=200.0)
    p.add_argument("--tsample", type=float, default=4.0)
    p.add_argument("--output", default=None)
    a = p.parse_args()
    t0 = time.time()
    out = {"symbol": mode_symbol, "convert": mode_convert, "cut": mode_cut, "radial": mode_radial_packets}[a.mode](a)
    out["wall_s"] = time.time() - t0
    out["code_md5"] = hashlib.md5(open(__file__, "rb").read()).hexdigest()
    out["args"] = vars(a)
    text = json.dumps(out, indent=1, default=float)
    if a.output:
        os.makedirs(os.path.dirname(a.output) or ".", exist_ok=True)
        with open(a.output, "w") as f:
            f.write(text + "\n")
    print(text[:4000])


if __name__ == "__main__":
    main()
