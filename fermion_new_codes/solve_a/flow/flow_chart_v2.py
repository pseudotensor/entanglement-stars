#!/usr/bin/env python3
"""flow_chart_v2.py: the flow-chart one-particle operator of H_* on a chain (the lead's first draft flow_chart_v1.py completed by the builder
analyst; every addition labelled BUILDER in the code or here).

Chart: flat ruler psi = 1, lapse N = 1 (simpler version: the flow v(R) is a declared input, marked in every record).
Chain: two components per site, index 2n + s. On site m_n SZ + y_n SY; sine hop h[n, n+1] = -(i/2) SX;
flow hop h[n, n+2] = -(i/4) vbar_n 1, vbar_n = (v_n + v_{n+2})/2 (horizon_cut.py / reviewer B F2.4; the range-two hop is
the law's own momentum current sin(2p)/2, JUDGMENT_web40c sec. 12 item 1).
Uniform symbol: h(k) = m SZ + sin k SX + v sin 2k /2; bands +-sqrt(m^2 + sin^2 k) + v sin 2k /2.
Planar chain: m_n = sqrt(m^2 + sin^2 ky + sin^2 kz), y_n = 0; radial channel: m_n = m, y_n = kappa_j / R_n, R_n = n + 1.
Units a = 1. Specs: solve/specs/SPEC_task3_prescribed_flow.md, SPEC_task4_injection.md.
Builder's completion (version 2): (1) mode checks: Hermiticity, C0 against the paper's four-component chain_generator (p = [1,0,0], the same
velocity array: every eigenvalue twice), C1 (the periodic ring against the symbol), C2 (the ramp product's unitarity), the sea's idempotency,
and the flat planar cut against verify.py's entropy_test (nx = 128, nk = 24: the record's SA = 0.7864825818); (2) the radial packets: the launch
lists A2, A3, D1, D2, D3, D4 of the spec in one call per (mass, kappa_j) on one eigendecomposition; the packet spinor from the local on-site
symbol with the centrifugal term (the record's packet code), the symbol's velocities with m_eff^2 = m^2 + (kappa_j/R)^2 (WKB); BUILDER choices
labelled in the records: A2's k0 = pi/4 carrier has R_turn = 0, so it is launched at 0.25, 0.5, 0.75 r_s; the chain lengths hold the packets'
4 sigma margins (A2 N = 512, D3 N = 1024); D3 at m = 0.1 uses omega r_s in {4, 8} with sigma_omega = min(omega/4, (omega - m)/3) (the spec's
frequencies are below the threshold m there); (3) the cut tables in parallel over the channel masses (a process pool), the record's
forming-surface parameters as the defaults (delta 0.4, width delta/kappa, ramp 40, dt 0.5, nx 128: RESULT_T128.md Item 2); (4) records carry
the script name, never a hash (the goal's guard); (5) the GPU route for 2N > 4000 (torch.linalg.eigh on --gpu).
"""
import argparse
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
GPU = {"dev": None}


def spectrum(h):
    """Dense Hermitian eigendecomposition; 2N > 4000 goes to torch.linalg.eigh on --gpu when given (the paper's spectrum() pattern)."""
    if GPU["dev"] is not None and h.shape[0] > 4000:
        import torch
        a = torch.as_tensor(h, device="cuda:%d" % GPU["dev"])
        e, Q = torch.linalg.eigh(a)
        out = e.cpu().numpy(), Q.cpu().numpy()
        del a, e, Q
        torch.cuda.empty_cache()
        return out
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


def _cut_channel(job):
    """one chain of the cut table: flat and formed S_A at the hold times (the record's horizon_cut protocol, the hold exact on h_f)."""
    mc, N, delta, width, ramp_time, dt, holds, kappa = job
    mz, yy, x = planar_fields(N, mc)
    vfinal = tanh_profile(x, delta, width)
    V0, occ0 = flat_sea(chain_hamiltonian(mz, yy, np.zeros(N)))
    flat = cut_entropies(V0, occ0, N // 2)
    Uramp = ramp_unitary(mz, yy, vfinal, ramp_time, dt)
    ef, Qf = spectrum(chain_hamiltonian(mz, yy, vfinal))
    Phi = Uramp @ V0
    formed = {str(th): cut_entropies(propagate(Qf, ef, th / kappa, Phi), occ0, N // 2) for th in holds}
    return mc, flat, formed


def mode_cut(args):
    """Part (c): the per-channel entropy table at the surface cut, flat and formed, at the hold times; the 4-packet sums per rectangle and aM."""
    from concurrent.futures import ProcessPoolExecutor
    kappa = 1.0 / args.inv_kappa
    delta, width = args.delta, args.delta * args.inv_kappa
    N = args.nsites
    masses = set()
    for rect in ("light", "1223"):
        for aM in args.aM:
            for ky, kz, _ in quadrant_quadrature(args.nk):
                for mc in channel_masses(rect, aM, ky, kz):
                    masses.add(round(float(mc), 12))
    for ky, kz, _ in quadrant_quadrature(args.nk):
        masses.add(round(float(np.sqrt(np.sin(ky) ** 2 + np.sin(kz) ** 2)), 12))
    masses.add(0.0)
    jobs = [(mc, N, delta, width, args.ramp_time, args.dt, args.hold_over_kappa, kappa) for mc in sorted(masses)]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(_cut_channel, jobs, chunksize=1))
    table = {"flat": {}, "formed": {str(th): {} for th in args.hold_over_kappa}}
    for mc, flat, formed in res:
        table["flat"][mc] = flat
        for th in args.hold_over_kappa:
            table["formed"][str(th)][mc] = formed[str(th)]
    assembled = {}
    for rect in ("light", "1223"):
        for aM in args.aM:
            key = f"{rect},aM={aM}"
            flat = {m: v[0] for m, v in table["flat"].items()}
            assembled[key] = {"flat": assemble_packets(flat, rect, aM, args.nk)}
            for th in args.hold_over_kappa:
                formed = {m: v[0] for m, v in table["formed"][str(th)].items()}
                assembled[key][f"formed,hold={th}"] = assemble_packets(formed, rect, aM, args.nk)
    # the massless sea alone (the record's control): the sum over the quadrant quadrature with the factor 2
    ml_flat = sum(2 * w * table["flat"][round(float(np.sqrt(np.sin(ky) ** 2 + np.sin(kz) ** 2)), 12)][0] for ky, kz, w in quadrant_quadrature(args.nk))
    ml_formed = {str(th): sum(2 * w * table["formed"][str(th)][round(float(np.sqrt(np.sin(ky) ** 2 + np.sin(kz) ** 2)), 12)][0]
                              for ky, kz, w in quadrant_quadrature(args.nk)) for th in args.hold_over_kappa}
    return {"label": "prescribed flow (simpler version); planar tanh; v declared", "kappa_a": kappa, "N": N,
            "nk": args.nk, "delta": delta, "width": width, "ramp_time": args.ramp_time, "dt": args.dt,
            "massless_sea_flat_SA": ml_flat, "massless_sea_formed_SA": ml_formed,
            "massless_sea_excess": {th: ml_formed[th] / ml_flat - 1 for th in ml_formed},
            "table_flat": {str(m): v for m, v in table["flat"].items()},
            "table_formed": {th: {str(m): v for m, v in d.items()} for th, d in table["formed"].items()},
            "assembled": assembled}


def mode_checks(args):
    """Unit checks before any run (the hand-back's job 1)."""
    import sys as _sys
    _sys.path.insert(0, os.environ.get("ORIENT_DIR", "<internal>/orient"))
    import lattice_entropy as LE
    rng = np.random.default_rng(3)
    out = {}
    n = 48
    v = -1.0 + 0.4 * np.tanh((np.arange(n) - (n - 1) / 2) / 6.0)
    h = chain_hamiltonian(np.full(n, 0.2), np.zeros(n), v)
    out["hermiticity"] = float(np.max(np.abs(h - h.conj().T)))
    H4 = LE.chain_generator((1, 0, 0), (0.0, 0.0), n, mass=0.2, velocity=v).toarray()
    e4 = np.sort(np.linalg.eigvalsh(H4))
    e2 = np.sort(np.linalg.eigvalsh(h))
    out["C0_chain_generator_twice"] = float(np.max(np.abs(e4 - np.sort(np.concatenate([e2, e2])))))
    out["C1_ring"] = {f"mc={mc},w={w}": ring_dispersion_check(64, mc, w) for mc in (0.0, 0.1, 0.3) for w in (0.0, 0.5, 1.0, 1.5, 2.0)}
    mz, yy, x = planar_fields(128, 0.1)
    U = ramp_unitary(mz, yy, tanh_profile(x, 0.4, 0.4 * 16), 8.0, 0.5)
    out["C2_unitarity"] = float(np.linalg.norm(U.conj().T @ U - np.eye(256)))
    V0, occ = flat_sea(chain_hamiltonian(mz, yy, np.zeros(128)))
    P = V0 @ V0.conj().T
    out["sea_idempotency"] = float(np.linalg.norm(P @ P - P))
    # verify.py's entropy_test: ff-initial at nx = 128, nk = 24, the quadrant quadrature with the factor 2
    tot = 0.0
    for ky, kz, w in quadrant_quadrature(24):
        mzc, yyc, _ = planar_fields(128, 0.0, ky, kz)
        V0c, occc = flat_sea(chain_hamiltonian(mzc, yyc, np.zeros(128)))
        tot += 2 * w * cut_entropies(V0c, occc, 64)[0]
    out["flat_cut_nx128_nk24"] = tot
    out["record_verify_SA"] = 0.7864825818
    out["flat_cut_minus_record"] = tot - 0.7864825818
    out["verify_threshold_0.786582_pm_0.015"] = bool(abs(tot - 0.786582) < 0.015)
    return out


def onsite_spinor(k, m, a):
    """the positive-energy spinor of the local symbol m SZ + a SY + sin k SX (the record's packet construction)."""
    H = np.array([[m, np.sin(k) - 1j * a], [np.sin(k) + 1j * a, -m]], complex)
    return np.linalg.eigh(H)[1][:, -1]


def symbol_vg(k, m, a, v):
    E = np.sqrt(m * m + a * a + np.sin(k) ** 2)
    return np.sin(k) * np.cos(k) / E + v * np.cos(2 * k)


def gaussian_carrier(R, R0, sigma, k0, m, a):
    env = np.exp(-(R - R0) ** 2 / (4 * sigma ** 2) + 1j * k0 * R)
    u = onsite_spinor(k0, m, a)
    psi = (env[:, None] * u[None, :]).reshape(-1)
    return psi / np.linalg.norm(psi)


def packet_avg_vg(k0, sigma, m, a, v):
    kk = np.linspace(k0 - 4 / (2 * sigma), k0 + 4 / (2 * sigma), 801)
    wk = np.exp(-2 * sigma ** 2 * (kk - k0) ** 2)
    wk /= wk.sum()
    return float(wk @ symbol_vg(kk, m, a, v))


def initial_velocity(Q, e, psi0, R, tmax=4.0):
    ts = np.linspace(0, tmax, 9)
    cs = [centre_width(propagate(Q, e, t, psi0), R)[0] for t in ts]
    return float(np.polyfit(ts, cs, 1)[0]), cs


def dust_law(t, R0, rs, Rm):
    """the marginally bound dust: R^(3/2) = R0^(3/2) - (3/2) sqrt(rs) t outside the ball; inside, dR/dt = -sqrt(rs/Rm^3) R."""
    t = np.asarray(t, float)
    tm = (2.0 / 3.0) * (R0 ** 1.5 - Rm ** 1.5) / np.sqrt(rs)
    out = np.where(t <= tm, np.maximum(R0 ** 1.5 - 1.5 * np.sqrt(rs) * t, 0.0) ** (2.0 / 3.0),
                   Rm * np.exp(-np.sqrt(rs / Rm ** 3) * (t - tm)))
    return out


def symbol_ode(R0, p0, m, kj, rs, Rm, ts):
    """Hamilton's equations of omega = sqrt(m^2 + (kj/R)^2 + sin^2 p) + v(R) sin(2p)/2."""
    from scipy.integrate import solve_ivp

    def rhs(t, y):
        R, pp = y
        a = kj / R
        E = np.sqrt(m * m + a * a + np.sin(pp) ** 2)
        v = float(pg_profile(R, rs, Rm))
        dv = (float(pg_profile(R + 1e-6, rs, Rm)) - float(pg_profile(R - 1e-6, rs, Rm))) / 2e-6
        dR = np.sin(pp) * np.cos(pp) / E + v * np.cos(2 * pp)
        dp = (a * a / R) / E - dv * np.sin(2 * pp) / 2
        return [dR, dp]
    sol = solve_ivp(rhs, (ts[0], ts[-1]), [R0, p0], t_eval=ts, rtol=1e-10, atol=1e-12)
    return sol.y[0].tolist(), sol.y[1].tolist()


def mode_radial_packets(args):
    """Parts (a) dynamic, (d) and task 4's flow half on the PG radial channel: one channel (mass, kappa_j), the launch list of --launch, all
    packets on one eigendecomposition (the lead's draft did one launch per call)."""
    N = args.nsites
    mz, yy, R = radial_fields(N, args.mass, args.kappa_j)
    v = pg_profile(R, args.rs, args.Rm)
    e, Q = spectrum(chain_hamiltonian(mz, yy, v))
    rs, Rm, m, kj = args.rs, args.Rm, args.mass, args.kappa_j
    lab = {"label": "prescribed flow (simpler version); PG radial; v declared", "launch": args.launch, "rs": rs, "Rm": Rm, "mass": m,
           "kappa_j": kj, "N": N, "R_max": float(R[-1])}
    loc = lambda R0: (float(pg_profile(R0, rs, Rm)), kj / R0)
    if args.launch == "A2":
        rows = []
        carriers = [(np.pi / 16, 32.0), (np.pi / 8, 16.0), (3 * np.pi / 16, 16.0), (np.pi / 4, 16.0), (-3 * np.pi / 8, 8.0)]
        for k0, sg in carriers:
            kabs = abs(k0)
            Rt = rs * float(turning_ratio(kabs, m))
            if Rt < 1e-9:
                launches = [0.25 * rs, 0.5 * rs, 0.75 * rs]
                note = "R_turn = 0: launched at 0.25, 0.5, 0.75 r_s (BUILDER)"
            else:
                launches = [f * Rt for f in (0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.25)]
                note = None
            for R0 in launches:
                if R0 - 4 * sg < 2 or R0 + 4 * sg > R[-1] - 2:
                    rows.append({"k0": k0, "sigma_x": sg, "R_launch": R0, "skipped": "outside the chain's 4 sigma margin"})
                    continue
                vloc, aloc = loc(R0)
                psi0 = gaussian_carrier(R, R0, sg, k0, m, aloc)
                vm, cs = initial_velocity(Q, e, psi0, R)
                rows.append({"k0": k0, "sigma_x": sg, "R_launch": R0, "R_turn_law": Rt, "note": note, "v_measured": vm,
                             "v_symbol_packet": packet_avg_vg(k0, sg, m, aloc, vloc), "centres": cs})
        # the measured turning radius per carrier: the sign change of the measured velocity along the launch scan
        turn = {}
        for k0, sg in carriers:
            rr = [r for r in rows if r.get("k0") == k0 and "v_measured" in r]
            for r1, r2 in zip(rr[:-1], rr[1:]):
                if r1["v_measured"] * r2["v_measured"] < 0:
                    x1, x2, y1, y2 = r1["R_launch"], r2["R_launch"], r1["v_measured"], r2["v_measured"]
                    turn["%.6f" % k0] = {"measured": x1 - y1 * (x2 - x1) / (y2 - y1), "law": rr[0]["R_turn_law"]}
        return dict(lab, rows=rows, turning=turn)
    if args.launch == "A3":
        k0 = float(np.arccos(0.25))
        sg = 16.0
        R0 = rs
        vloc, aloc = loc(R0)
        psi0 = gaussian_carrier(R, R0, sg, k0, m, aloc)
        vm, cs = initial_velocity(Q, e, psi0, R)
        kk = np.linspace(1e-6, np.pi / 2 - 1e-6, 200001)
        return dict(lab, k0=k0, R_launch=R0, v_measured=vm, v_symbol_packet=packet_avg_vg(k0, sg, m, aloc, vloc),
                    max_k_vg=float(np.max(symbol_vg(kk, m, aloc, vloc))), bound_9_8=9 / 8, centres=cs)
    if args.launch in ("D1", "D2"):
        R0, sg = 4 * rs, 8.0
        vloc, aloc = loc(R0)
        if args.launch == "D1":
            p0 = 0.0
        else:
            from scipy.optimize import brentq
            f = lambda pp: np.sin(pp) * np.cos(pp) / np.sqrt(m * m + aloc * aloc + np.sin(pp) ** 2) + vloc
            p0 = float(brentq(f, 1e-9, np.pi / 4))
        psi0 = gaussian_carrier(R, R0, sg, p0, m, aloc)
        tc = (2.0 / 3.0) * (R0 ** 1.5 - Rm ** 1.5) / np.sqrt(rs)
        times = np.arange(0, tc + 1e-9, args.tsample)
        rec = track(Q, e, psi0, times, R, {"inside_rs": R < rs, "inside_Rm": R < Rm})
        out = dict(lab, R0=R0, p0=p0, sigma_x=sg, times=times.tolist(), track=rec)
        if args.launch == "D1":
            out["dust_law"] = dust_law(times, R0, rs, Rm).tolist()
            out["crossing_time_law"] = (2.0 / 3.0) * (R0 ** 1.5 - rs ** 1.5) / np.sqrt(rs)
        Rs, Ps = symbol_ode(R0, p0, m, kj, rs, Rm, times)
        out["symbol_ode_R"], out["symbol_ode_p"] = Rs, Ps
        return out
    if args.launch == "D3":
        rows = []
        oms = [0.5, 1.0, 2.0, 4.0] if m == 0 else [4.0, 8.0]
        for omr in oms:
            om = omr / rs
            sgw = min(om / 4, (om - m) / 3) if m > 0 else om / 4
            if om - 3 * sgw <= m:
                rows.append({"omega_rs": omr, "skipped": "below the threshold window"})
                continue
            R0 = R[-1] - 2.0
            vext, aext = loc(R0)
            try:
                k, omk, a2 = branch_momenta(om, sgw, float(np.sqrt(m * m + aext * aext)), vext, outgoing=False)
            except ValueError as ex:
                rows.append({"omega_rs": omr, "skipped": str(ex)})
                continue
            vg0 = abs(float(np.sum(a2 * group_velocity(k, float(np.sqrt(m * m + aext * aext)), vext) * np.gradient(k))))
            sx = vg0 / (2 * sgw)
            R0 = min(R[-1] - 4 * sx - 2, R[-1] - 2)
            if R0 - 4 * sx < 2 * rs:
                rows.append({"omega_rs": omr, "skipped": "the packet does not fit outside 2 r_s (sigma_x %.1f)" % sx})
                continue
            psi0 = build_packet(N, int(round(R0 - 1)), k, a2, float(np.sqrt(m * m + aext * aext)))
            T = (R0 - Rm) / max(vg0, 1e-3) * 1.5
            times = np.linspace(0, T, 41)
            rec = track(Q, e, psi0, times, R, {"inside_rs": R < rs, "beyond_Rref": R > 2 * rs})
            rows.append({"omega_rs": omr, "omega": om, "sigma_omega": sgw, "sigma_x": sx, "R0": R0, "times": times.tolist(), "track": rec})
        return dict(lab, rows=rows)
    if args.launch == "D4":
        k0, sg, R0 = np.pi / 2, 8.0, rs / 2
        vloc, aloc = loc(R0)
        psi0 = gaussian_carrier(R, R0, sg, k0, m, aloc)
        times = np.arange(0, 3 * rs + 1e-9, args.tsample)
        rec = track(Q, e, psi0, times, R, {"inside_rs": R < rs, "beyond_2rs": R > 2 * rs})
        return dict(lab, k0=k0, R0=R0, sigma_x=sg, times=times.tolist(), track=rec,
                    w_along=[float(-pg_profile(r["centre"], rs, Rm)) for r in rec])
    raise ValueError(args.launch)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["symbol", "convert", "cut", "radial", "checks"])
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
    p.add_argument("--cut-delta", type=float, default=None, help="unused; the cut mode reads --delta (record default 0.4 set by the launcher)")
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
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--output", default=None)
    a = p.parse_args()
    t0 = time.time()
    GPU["dev"] = a.gpu
    out = {"symbol": mode_symbol, "convert": mode_convert, "cut": mode_cut, "radial": mode_radial_packets, "checks": mode_checks}[a.mode](a)
    out["wall_s"] = time.time() - t0
    out["script"] = os.path.basename(__file__)
    out["args"] = vars(a)
    text = json.dumps(out, indent=1, default=float)
    if a.output:
        os.makedirs(os.path.dirname(a.output) or ".", exist_ok=True)
        with open(a.output, "w") as f:
            f.write(text + "\n")
    print(text[:4000])


if __name__ == "__main__":
    main()
