#!/usr/bin/env python3
"""flow_chart_v5.py: task 5 of final4 (item 5, the black object of the model) on the flow chart.  flow_chart_v4.py is imported unchanged
(its records A1 to D4 stand); this file adds the lead's sec. 7 drafts of solve/specs/SPEC_task5_black_object.md, completed by the builder
analyst.  Every record carries the label "prescribed flow (simpler version); PG radial; v declared" and the script name, never a hash.

Modes (one JSON per call):
  e0  checks: (i) the cut current J_in against the norm inside on a D3 state (r_s = 32, R_m = 8, N = 1024, omega r_s = 2, j = 1/2,
      kappa_j = -1, m = 0): 64-point Gauss-Legendre integrals of J_in over intervals of length 8 against the change of the norm inside, and
      the Richardson central difference at the interval centres, both signs printed (the derivation gives dN_in/dt = +2 Im psi_in^H h_{in,out}
      psi_out; the lead's draft wrote -2 Im; the check fixes the sign); (ii) the Killing readout of the launched packet; (iii) the
      eigenvectors' orthonormality on the device (Pi = Q_+ Q_+^H idempotent iff Q_+^H Q_+ = 1) and the flat sea's idempotency.
  e1  the escape probability by wavelength (sec. 3 E1): P_+ = ||Pi_{h > m} w||^2, the symbol's P_sym, the dynamical first-passage P_dyn.
  e4  the clock (sec. 3 E4): packets emitted on the dust law's events; omega_bar, E_loc, 1 + z, the arrival at R_obs, the late-time fits.
  e3  what crosses from inside (sec. 3 E3): D3's packets rerun with the per-annulus momentum distributions.
  e2  the capture cross section (sec. 3 E2): per (m, kappa_j) one eigendecomposition, the ingoing packets, T_kappa, the return, sigma_cap.

BUILDER choices (each also written into the records):
  (B1) the cut between the sites with R_n < R_cut and R_n >= R_cut (R_n = n + 1); J_in from the 4 x 4 block h[2c-4:2c, 2c:2c+4], which holds
       the sine hop and both range-two hops across the cut.
  (B2) E1's gate: the earliest return through R_g is t_min = R_g/3 + R_g/(9/8) (inward speeds are below 1 + w <= 3 in the declared profile,
       outward below 9/8); t_gate is the first time after t_min at which the outward flux through R_g, integrated from t_min, exceeds 1e-3;
       the outward flux through R_g before t_min (the packet's own tail) is recorded as the gate's cleanliness.  P_dyn is the norm beyond
       R_ref at t_gate minus that at t = 0 (equal to the flux integral, printed beside it), and P_dyn_ext the same at t_gate + (R_ref -
       R_g)/(9/8), the earliest time the return can reach R_ref.  For launches at R_0 > 1.5 r_s, R_ref = R_0 + 4 sigma_x + r_s/2.
  (B3) E1's launch lists: pi/4 adds 0.3 and 0.4 r_s (R_esc(pi/4) = r_s/2 is the spec list's lowest point); m = 0.03 adds 1.1 to 2.5 r_s
       and m = 0.1 adds 1.1 to 4 r_s (the spec's "extended to 2.5" and "to 4"); the control sigma_x = 32 at pi/8 runs every launch.
  (B4) the symbol's packet averages use the Gaussian weights of the launched packet in (k, R) (sigma_k = 1/(2 sigma_x)) and the WKB symbol
       omega = sqrt(m^2 + (kappa_j/R)^2 + sin^2 k) + v sin(2k)/2 (version 4's convention); "the packet average of 1/(1 - w cos k_0)" is read
       as the ratio of the symbol's averages <E>/<omega>, the measured quantity's own form, with the point law 1/(1 - w(R_e) cos k_0) beside.
  (B5) E4 runs at N = 4096 (R_max = 32 r_s) on its own decompositions: at the spec's N = 2048 the wall's reflection returns to R_obs = 8 r_s
       at about 2350 lattice units, before the near-surface arrivals (2700 to 3100); the N = 2048 run is kept as the control with its clean
       window printed.  Beside the spec's 1 + z = E_loc/omega_bar (the whole packet), the escaping part's 1 + z_+ = E_loc/omega_bar_+
       (omega_bar_+ the mean of h over the weight above the threshold) and the flux-weighted mean arrival; the symbol's arrival uses
       omega_bar_+.
  (B6) E3's per-annulus momentum: the masked state's per-component FFT zero-padded to 4096, its weight at k > 0 and the weighted mean k there.
  (B7) E2's packets: sigma_omega = omega/8, at m > 0 min(omega/8, (omega - m)/4) (version 4's D3 rule, the window above the threshold);
       R_0 = max(16 r_s, 10 |kappa_j|, 2 r_s + 4 sigma_x); the gate at r_s/2 as (B2) with the front time (inward flux through r_s/2 above
       1e-6) plus (r_s/2)/3 + (r_s/2)/(9/8), threshold 1e-3 relative to the inward flux through the gate; T_kappa by the flux integral and by
       the norm inside r_s (printed beside); sigma_cap with pi/omega^2 (the spec) and, at m > 0, pi/p^2 with p = sqrt(omega^2 - m^2) beside.
"""
import argparse
import json
import math
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import flow_chart_v4 as V4  # noqa: E402

LABEL = "prescribed flow (simpler version); PG radial; v declared"
VMAX_OUT = 9.0 / 8.0
VMAX_IN = 3.0


def _torch(gpu):
    import torch
    torch.set_num_threads(1)
    return torch, (torch.device("cuda:%d" % gpu) if gpu is not None else torch.device("cpu"))


def cut_index(Rcut):
    """(B1): the sites n < c have R_n = n + 1 < R_cut."""
    return int(math.ceil(Rcut - 1e-9)) - 1


def current_in(A8, Hb):
    """J_in(t) = dN_in/dt = 2 Im psi_in^H h_{in,out} psi_out from the rows 2c-4 .. 2c+3 (columns: times)."""
    a, b = A8[:4], A8[4:]
    return 2.0 * np.imag(np.sum(a.conj() * (Hb @ b), axis=0))


def cumtrapz0(y, dt):
    return np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * dt)])


class Chain:
    """The PG radial channel (m, kappa_j) on N sites: h (numpy, dense), its eigendecomposition on the device (torch)."""

    def __init__(self, N, m, kj, rs, Rm, gpu, with_h0=False):
        torch, dev = _torch(gpu)
        self.torch, self.dev = torch, dev
        self.N, self.m, self.kj, self.rs, self.Rm = int(N), float(m), float(kj), float(rs), float(Rm)
        mz, yy, R = V4.radial_fields(N, m, kj)
        self.R = R
        self.v = V4.pg_profile(R, rs, Rm)
        self.h = V4.chain_hamiltonian(mz, yy, self.v)
        t0 = time.time()
        e, Q = torch.linalg.eigh(torch.as_tensor(self.h, device=dev))
        self.e, self.Q = e, Q
        self.eig_s = time.time() - t0
        self.h0 = V4.chain_hamiltonian(mz, yy, np.zeros(N)) if with_h0 else None

    def block(self, c):
        return self.h[2 * c - 4:2 * c, 2 * c:2 * c + 4]

    def coef(self, w):
        return self.Q.conj().T @ self.torch.as_tensor(np.ascontiguousarray(w), device=self.dev)

    def readout(self, c, thr):
        p = c.abs() ** 2
        tot = float(p.sum())
        p = p / tot
        e = self.e
        mean = float((e * p).sum())
        var = float((e * e * p).sum()) - mean * mean
        sel = e > thr
        pa = float(p[sel].sum())
        mean_a = float((e[sel] * p[sel]).sum()) / pa if pa > 0 else None
        return {"P_above": pa, "omega_mean": mean, "omega_spread": math.sqrt(max(var, 0.0)), "omega_mean_above": mean_a,
                "weight_omega_neg": float(p[e < 0].sum()), "norm": tot}

    def state(self, c, t):
        return (self.Q @ (self.torch.exp(-1j * self.e * float(t)) * c)).cpu().numpy()

    def rows_series(self, c, rows, times, chunk=512):
        torch = self.torch
        Qr = self.Q[list(rows), :]
        out = []
        times = np.asarray(times, float)
        for i in range(0, len(times), chunk):
            tt = torch.as_tensor(times[i:i + chunk], device=self.dev)
            ph = torch.exp(-1j * self.e[:, None] * tt[None, :]) * c[:, None]
            out.append((Qr @ ph).cpu().numpy())
        return np.concatenate(out, axis=1)

    def free(self):
        del self.Q, self.e
        if self.dev.type == "cuda":
            self.torch.cuda.empty_cache()


def orthonormality(ch):
    G = ch.Q.conj().T @ ch.Q
    G.diagonal().sub_(1.0)
    v = float(G.abs().max())
    del G
    return {"QhQ_minus_1_max": v}


# ---------------------------------------------------------------- symbol (sec. 2)
def symbol_omega(k, R, m, kj, rs, Rm):
    E = np.sqrt(m * m + (kj / R) ** 2 + np.sin(k) ** 2)
    return E + V4.pg_profile(R, rs, Rm) * np.sin(2 * k) / 2, E


def packet_symbol(k0, sx, R0, m, kj, rs, Rm, thr, n=241):
    """(B4) the symbol's packet averages over the Gaussian weights of the launched packet."""
    sk = 1.0 / (2.0 * sx)
    k = k0 + sk * np.linspace(-6, 6, n)
    R = R0 + sx * np.linspace(-6, 6, n)
    R = R[R >= 1.0]
    W = np.outer(np.exp(-(k - k0) ** 2 / (2 * sk * sk)), np.exp(-(R - R0) ** 2 / (2 * sx * sx)))
    W /= W.sum()
    om, E = symbol_omega(k[:, None], R[None, :], m, kj, rs, Rm)
    mean = float((W * om).sum())
    return {"P_sym": float(W[om > thr].sum()), "omega_sym": mean, "spread_sym": float(np.sqrt(max((W * om * om).sum() - mean * mean, 0.0))),
            "E_sym": float((W * E).sum())}


def escape_surface(k, m, rs):
    """the lead's R_esc(k, m) = r_s (sin k cos k)^2/(E - m)^2 (massless r_s cos^2 k)."""
    E = np.sqrt(m * m + np.sin(k) ** 2)
    return rs * (np.sin(k) * np.cos(k)) ** 2 / (E - m) ** 2


def esc_budget(k0, sx, m, rs):
    dk = 1e-6
    d = abs(float(escape_surface(k0 + dk, m, rs)) - float(escape_surface(k0 - dk, m, rs))) / (2 * dk)
    return d / (2 * sx) + sx


def outgoing_root(omega, R, rs, Rm, m, kj, k_prev, step=0.02):
    """k(R; omega) on the outgoing branch continuous from k_prev (the lead's draft: bracket from k_prev by steps of 0.02; None if none)."""
    v = float(V4.pg_profile(R, rs, Rm))
    a = kj / R
    F = lambda k: math.sqrt(m * m + a * a + math.sin(k) ** 2) + v * math.sin(2 * k) / 2 - omega
    f1 = F(k_prev)
    if f1 == 0.0:
        return k_prev
    d = -step if f1 > 0 else step
    k1 = k_prev
    for _ in range(int(math.pi / step) + 2):
        k2 = k1 + d
        if k2 <= 1e-12 or k2 >= math.pi - 1e-12:
            return None
        f2 = F(k2)
        if f1 * f2 <= 0:
            r = brentq(F, min(k1, k2), max(k1, k2), xtol=1e-14)
            return r if float(V4.symbol_vg(r, m, a, v)) > 0 else None
        k1, f1 = k2, f2
    return None


def arrival_time(omega, Re, Robs, rs, Rm, m, kj, k0, n=3000):
    """t_arr - t_e = int_{R_e}^{R_obs} dR/v_g(k(R; omega), R) along the stationary outgoing root (grid dense near R_e)."""
    s = np.linspace(0.0, 1.0, n)
    Rg = Re + (Robs - Re) * s ** 2
    k = outgoing_root(omega, Re, rs, Rm, m, kj, k0)
    if k is None:
        return None, "no outgoing root at R_e"
    inv = []
    for R in Rg:
        k = outgoing_root(omega, R, rs, Rm, m, kj, k)
        if k is None:
            return None, "the root disappears at R = %.4f" % R
        inv.append(1.0 / float(V4.symbol_vg(k, m, kj / R, float(V4.pg_profile(R, rs, Rm)))))
    return float(np.trapz(inv, Rg)), None


def root_inside(omega, R, rs, Rm, m, kj):
    """the positive-omega outgoing root inside the trapped region: k in [arccos(1/w), pi/2]; outside it the long-wave root in (0, pi/2]."""
    v = float(V4.pg_profile(R, rs, Rm))
    w = -v
    a = kj / R
    F = lambda k: math.sqrt(m * m + a * a + math.sin(k) ** 2) + v * math.sin(2 * k) / 2 - omega
    lo = math.acos(1.0 / w) if w > 1 else 1e-9
    hi = math.pi / 2
    if F(lo) * F(hi) > 0:
        return None
    return brentq(F, lo, hi, xtol=1e-14)


def ingoing_root(omega, R, rs, Rm, m, kj):
    v = float(V4.pg_profile(R, rs, Rm))
    a = kj / R
    F = lambda k: math.sqrt(m * m + a * a + math.sin(k) ** 2) + v * math.sin(2 * k) / 2 - omega
    lo, hi = -1.2, -1e-9
    if F(lo) * F(hi) > 0:
        return None
    return brentq(F, lo, hi, xtol=1e-14)


def d3_packet(N, rs, Rm, m, kj, omr):
    """version 4's D3 launch, unchanged: the ingoing packet at the box edge."""
    _, _, R = V4.radial_fields(N, m, kj)
    om = omr / rs
    sgw = min(om / 4, (om - m) / 4) if m > 0 else om / 4
    R0 = R[-1] - 2.0
    vext, aext = float(V4.pg_profile(R0, rs, Rm)), kj / R0
    me = float(np.sqrt(m * m + aext * aext))
    k, omk, a2 = V4.branch_momenta(om, sgw, me, vext, outgoing=False)
    vg0 = abs(float(np.sum(a2 * V4.group_velocity(k, me, vext) * np.gradient(k))))
    sx = vg0 / (2 * sgw)
    R0 = min(R[-1] - 4 * sx - 2, R[-1] - 2)
    psi0 = V4.build_packet(N, int(round(R0 - 1)), k, a2, me, m=m, a=kj / R0)
    return psi0, R0, sx, om, sgw, vg0


# ---------------------------------------------------------------- E0
def mode_e0(args):
    rs, Rm, N, kj = 32.0, 8.0, 1024, -1.0
    ch = Chain(N, 0.0, kj, rs, Rm, None)
    psi0, R0, sx, om, sgw, vg0 = d3_packet(N, rs, Rm, 0.0, kj, 2.0)
    c = ch.coef(psi0)
    c1 = cut_index(rs)
    rows = range(2 * c1 - 4, 2 * c1 + 4)
    Nin = lambda t: float((np.abs(ch.state(c, t)[:2 * c1]) ** 2).sum())
    T = (R0 - Rm) / vg0 * 1.5
    ts = np.linspace(0, T, 121)
    nin = np.array([Nin(t) for t in ts])
    d = np.gradient(nin, ts)
    t_up, t_dn = float(ts[np.argmax(d)]), float(ts[np.argmin(d)])
    xg, wg = np.polynomial.legendre.leggauss(64)
    rec = []
    for tc in (t_up, t_dn, t_up - 10.0, t_up + 10.0, t_dn + 10.0):
        t1, t2 = tc - 4.0, tc + 4.0
        tt = 0.5 * (t2 - t1) * xg + 0.5 * (t1 + t2)
        J = current_in(ch.rows_series(c, rows, tt), ch.block(c1))
        integ = 0.5 * (t2 - t1) * float(wg @ J)
        dN = Nin(t2) - Nin(t1)
        cd = lambda hh: (Nin(tc + hh) - Nin(tc - hh)) / (2 * hh)
        rich = (4 * cd(0.005) - cd(0.01)) / 3
        Jc = float(current_in(ch.rows_series(c, rows, [tc]), ch.block(c1))[0])
        rec.append({"t1": t1, "t2": t2, "dN_in": dN, "int_J_plus": integ, "err_plus": integ - dN, "err_minus": -integ - dN, "t_c": tc,
                    "J_plus_at_t_c": Jc, "richardson_dN_dt": rich, "err_richardson_plus": Jc - rich, "err_richardson_minus": -Jc - rich})
    plus = max(max(abs(r["err_plus"]), abs(r["err_richardson_plus"])) for r in rec)
    minus = max(max(abs(r["err_minus"]), abs(r["err_richardson_minus"])) for r in rec)
    mz, yy, R = V4.radial_fields(N, 0.0, kj)
    e0, Q0 = V4.spectrum(V4.chain_hamiltonian(mz, yy, np.zeros(N)))
    V0 = Q0[:, e0 < 0]
    P = V0 @ V0.conj().T
    return {"label": LABEL, "mode": "e0", "setup": {"rs": rs, "Rm": Rm, "N": N, "kappa_j": kj, "m": 0.0, "omega_rs": 2.0, "R0": R0, "sigma_x": sx,
            "sigma_omega": sgw}, "i_cut_current": {"intervals": rec, "max_err_plus": plus, "max_err_minus": minus,
            "sign_fixed": "+2 Im" if plus < minus else "-2 Im"},
            "ii_killing_readout": {"omega_target": om, **ch.readout(c, 0.0)},
            "iii_orthonormality": orthonormality(ch), "iii_flat_sea_idempotency_max": float(np.max(np.abs(P @ P - P)))}


# ---------------------------------------------------------------- E1
E1_CARRIERS = [(np.pi / 32, 32.0), (np.pi / 16, 16.0), (np.pi / 8, 16.0), (np.pi / 4, 8.0)]
E1_BASE = [0.5, 0.7, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975, 1.0, 1.05, 1.2]


def e1_launches(m, k0):
    L = list(E1_BASE)
    if abs(k0 - np.pi / 4) < 1e-12:
        L += [0.3, 0.4]
    if m >= 0.1 - 1e-12:
        L += [1.1, 1.3, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0]
    elif m > 0:
        L += [1.1, 1.3, 1.5, 1.75, 2.0, 2.25, 2.5]
    return sorted(set(L))


def crossings(x, y, level=0.5):
    out = []
    for i in range(len(x) - 1):
        a, b = y[i] - level, y[i + 1] - level
        if a == 0:
            out.append(float(x[i]))
        elif a * b < 0:
            out.append(float(x[i] + (x[i + 1] - x[i]) * (-a) / (b - a)))
    return out


def e1_dynamic(ch, c, R0, sx, tmax, dt):
    """(B2) the first-passage escape through R_ref before the centre's return crosses the gate R_g outward."""
    rs = ch.rs
    Rg = min(R0 - 3 * sx, rs / 2)
    Rref = 2 * rs if R0 <= 1.5 * rs else R0 + 4 * sx + 0.5 * rs
    out = {"R_g": Rg, "R_ref": Rref}
    if Rg < 4:
        out["skipped"] = "gate below R = 4"
        return out
    cg, cr = cut_index(Rg), cut_index(Rref)
    rows = list(range(2 * cg - 4, 2 * cg + 4)) + list(range(2 * cr - 4, 2 * cr + 4))
    times = np.arange(0.0, tmax + 1e-9, dt)
    A = ch.rows_series(c, rows, times)
    Jg_out = -current_in(A[:8], ch.block(cg))
    Jr_out = -current_in(A[8:], ch.block(cr))
    tmin = Rg / VMAX_IN + Rg / VMAX_OUT
    cum = cumtrapz0(np.maximum(Jg_out, 0.0), dt)
    i0 = min(int(np.searchsorted(times, tmin)), len(times) - 1)
    after = cum - cum[i0]
    hit = np.where((times > tmin) & (after > 1e-3))[0]
    tg = float(times[hit[0]]) if len(hit) else None
    beyond = lambda t: float(V4.density(ch.state(c, t))[ch.R >= Rref].sum())
    n0 = beyond(0.0)
    Fr = cumtrapz0(Jr_out, dt)
    out.update({"t_min_return": tmin, "gate_outward_before_t_min": float(cum[i0]), "t_gate": tg, "beyond_R_ref_t0": n0,
                "beyond_R_ref_tmax": beyond(tmax), "tmax": tmax, "dt": dt})
    if tg is not None:
        out["P_dyn"] = beyond(tg) - n0
        out["P_dyn_flux"] = float(Fr[hit[0]])
        te = min(tg + (Rref - Rg) / VMAX_OUT, tmax)
        out["t_ext"] = te
        out["P_dyn_ext"] = beyond(te) - n0
    else:
        out["P_dyn_tmax"] = beyond(tmax) - n0
        out["P_dyn_flux_tmax"] = float(Fr[-1])
    return out


def mode_e1(args):
    rs, Rm, N = args.rs, args.Rm, args.nsites
    res = {"label": LABEL, "mode": "e1", "rs": rs, "Rm": Rm, "N": N, "R_max": float(N), "R_inner": math.sqrt(Rm ** 3 / rs),
           "rmax_control": bool(args.rmax_control), "runs": []}
    for m in args.masses:
        for kj in args.kappas:
            ch = Chain(N, m, kj, rs, Rm, args.gpu)
            run = {"mass": m, "kappa_j": kj, "eig_s": ch.eig_s, "checks": orthonormality(ch), "series": []}
            carriers = [(np.pi / 16, 16.0)] if args.rmax_control else list(E1_CARRIERS) + [(np.pi / 8, 32.0)]
            for k0, sx in carriers:
                if m >= 0.1 - 1e-12 and k0 < np.pi / 16 - 1e-12:
                    continue
                rows = []
                for f in e1_launches(m, k0):
                    R0 = f * rs
                    if R0 - 4 * sx < 1 or R0 + 4 * sx > N - 2:
                        rows.append({"R0_over_rs": f, "skipped": "outside the chain's 4 sigma margin"})
                        continue
                    w = V4.gaussian_carrier(ch.R, R0, sx, k0, m, kj / R0)
                    c = ch.coef(w)
                    ro = ch.readout(c, m)
                    row = {"R0_over_rs": f, "R0": R0, "P_plus": ro["P_above"], **ro, **packet_symbol(k0, sx, R0, m, kj, rs, Rm, m)}
                    om = ro["omega_mean"]
                    row["killing_spread_over_abs_omega"] = ro["omega_spread"] / abs(om) if om != 0 else None
                    row["dyn"] = e1_dynamic(ch, c, R0, sx, args.tmax_over_rs * rs, args.dt)
                    rows.append(row)
                ok = [r for r in rows if "P_plus" in r]
                Resc = float(escape_surface(k0, m, rs))
                run["series"].append({"k0": k0, "sigma_x": sx, "control": "sigma_x doubled at pi/8" if (abs(k0 - np.pi / 8) < 1e-12 and sx == 32.0) else None,
                                      "R_esc": Resc, "R_esc_over_rs": Resc / rs, "budget": esc_budget(k0, sx, m, rs),
                                      "R_turn": rs * float(V4.turning_ratio(k0, m)),
                                      "R_half_measured": crossings([r["R0"] for r in ok], [r["P_plus"] for r in ok]),
                                      "R_half_symbol": crossings([r["R0"] for r in ok], [r["P_sym"] for r in ok]), "rows": rows})
                print("e1 m %g kj %g k0 %.4f sx %g: R_esc %.2f, R_1/2 %s (%.0f s)" % (m, kj, k0, sx, Resc, run["series"][-1]["R_half_measured"],
                                                                                      time.time() - T0), flush=True)
            res["runs"].append(run)
            ch.free()
    return res


# ---------------------------------------------------------------- E4
E4_RE = [2.0, 1.5, 1.25, 1.1, 1.05, 1.02, 1.01, 1.005, 1.0, 0.98]
E4_CARRIERS = [(np.pi / 32, 32.0), (np.pi / 16, 16.0), (np.pi / 8, 16.0)]


def dust_time(Re, R0, rs):
    return (2.0 / 3.0) * (R0 ** 1.5 - Re ** 1.5) / math.sqrt(rs)


def linfit(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2:
        return None
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ coef
    return {"slope": float(coef[0]), "intercept": float(coef[1]), "n": int(len(x)), "rms_residual": float(np.sqrt(np.mean(res ** 2)))}


def e4_fits(rows, kappa, sat):
    out = {}
    pts = sorted([r for r in rows if "t_e" in r], key=lambda r: r["t_e"])
    variants = {"measured_peak": ("one_plus_z", "t_arr_peak"), "measured_half": ("one_plus_z", "t_arr_half"),
                "escaping_mean": ("one_plus_z_plus", "t_arr_mean"), "symbol": ("one_plus_z_point", "t_arr_symbol")}
    for name, (zk, tk) in variants.items():
        good = [r for r in pts if r.get(zk) is not None and r.get(tk) is not None and r[zk] > 0 and r.get("clean", True)]
        win = [r for r in good if 3.0 <= r[zk] <= sat / 3.0]
        f1 = linfit([kappa * r[tk] for r in win], [math.log(r[zk]) for r in win])
        pairs = []
        for r1, r2 in zip(good[:-1], good[1:]):
            if r2["t_e"] > r1["t_e"] and r2[tk] > r1[tk] and 3.0 <= r1[zk] <= sat / 3.0 and 3.0 <= r2[zk] <= sat / 3.0:
                pairs.append((0.5 * kappa * (r1[tk] + r2[tk]), math.log((r2[tk] - r1[tk]) / (r2["t_e"] - r1["t_e"]))))
        f2 = linfit([p[0] for p in pairs], [p[1] for p in pairs])
        out[name] = {"window_points": [r["Re_over_rs"] for r in win], "ln_1pz_vs_kappa_t_arr": f1, "ln_dtarr_dte_vs_kappa_t_arr": f2}
    return out


def e4_packet(ch, k0, sx, Re, Robs, R0d, tau_max, dt):
    rs, Rm, m, kj = ch.rs, ch.Rm, ch.m, ch.kj
    w = V4.gaussian_carrier(ch.R, Re, sx, k0, m, kj / Re)
    c = ch.coef(w)
    ro = ch.readout(c, m)
    Eloc = float(np.real(np.vdot(w, ch.h0 @ w)))
    sym = packet_symbol(k0, sx, Re, m, kj, rs, Rm, m)
    E0 = math.sqrt(m * m + math.sin(k0) ** 2)
    wRe = -float(V4.pg_profile(Re, rs, Rm))
    row = {"Re_over_rs": Re / rs, "Re": Re, "t_e": dust_time(Re, R0d, rs), **ro, "E_loc": Eloc, **sym,
           "one_plus_z": Eloc / ro["omega_mean"] if ro["omega_mean"] != 0 else None,
           "one_plus_z_plus": Eloc / ro["omega_mean_above"] if ro["omega_mean_above"] else None,
           "one_plus_z_symbol_avg": sym["E_sym"] / sym["omega_sym"] if sym["omega_sym"] != 0 else None,
           "one_plus_z_point": E0 / (E0 - wRe * math.sin(2 * k0) / 2) if E0 - wRe * math.sin(2 * k0) / 2 > 0 else None,
           "threshold_margin": ro["omega_mean"] - m}
    co = cut_index(Robs)
    times = np.arange(0.0, tau_max + 1e-9, dt)
    J = -current_in(ch.rows_series(c, range(2 * co - 4, 2 * co + 4), times), ch.block(co))
    Rw = float(ch.N)
    t_wall = (Rw - (Re + 4 * sx)) / VMAX_OUT + (Rw - Robs) / (1.0 + math.sqrt(rs / Robs))
    ok = times < t_wall
    row["t_wall_clean"] = t_wall
    if not np.any(J[ok] > 0):
        row["arrival"] = "no outward flux at R_obs in the clean window"
        return row
    Jc, tc = J[ok], times[ok]
    i = int(np.argmax(Jc))
    tp = float(tc[i])
    if 0 < i < len(Jc) - 1:
        y0, y1, y2 = Jc[i - 1], Jc[i], Jc[i + 1]
        den = y0 - 2 * y1 + y2
        if den != 0:
            tp = float(tc[i] + 0.5 * dt * (y0 - y2) / den)
    F = cumtrapz0(Jc, dt)
    half = np.where(F >= 0.5 * F[-1])[0]
    tpos = np.maximum(Jc, 0)
    row.update({"tau_peak": tp, "tau_half": float(tc[half[0]]) if len(half) else None, "tau_mean": float((tc * tpos).sum() / tpos.sum()),
                "escaped_flux_clean": float(F[-1]), "peak_at_window_edge": bool(i >= len(Jc) - 2)})
    te = row["t_e"]
    row["t_arr_peak"] = te + tp
    row["t_arr_half"] = te + row["tau_half"] if row["tau_half"] is not None else None
    row["t_arr_mean"] = te + row["tau_mean"]
    om_sym = ro["omega_mean_above"]
    if om_sym is not None and om_sym > m:
        ta, why = arrival_time(om_sym, Re, Robs, rs, Rm, m, kj, k0)
        row["tau_symbol"], row["tau_symbol_note"] = ta, why
        row["t_arr_symbol"] = te + ta if ta is not None else None
    om_pt = E0 - wRe * math.sin(2 * k0) / 2 + 0.0
    if om_pt > m:
        ta, why = arrival_time(om_pt, Re, Robs, rs, Rm, m, kj, k0)
        row["tau_symbol_point"], row["tau_symbol_point_note"] = ta, why
    row["clean"] = bool(tp < t_wall and not row["peak_at_window_edge"])
    return row


def mode_e4(args):
    rs, Rm, N = args.rs, args.Rm, args.nsites
    Robs = args.robs_over_rs * rs
    R0d = 4 * rs
    kappa = 1.0 / (2 * rs)
    res = {"label": LABEL, "mode": "e4", "rs": rs, "Rm": Rm, "N": N, "R_obs": Robs, "dust_R0": R0d, "t_cross": dust_time(rs, R0d, rs),
           "kappa": kappa, "builder_B5": "N = 4096 is the builder's production size (the wall); N = 2048 is the spec's, kept as the control",
           "runs": []}
    for m in args.masses:
        for kj in args.kappas:
            ch = Chain(N, m, kj, rs, Rm, args.gpu, with_h0=True)
            run = {"mass": m, "kappa_j": kj, "eig_s": ch.eig_s, "checks": orthonormality(ch), "series": []}
            carriers = list(E4_CARRIERS) + [(np.pi / 16, 32.0)]
            for k0, sx in carriers:
                rows = []
                for f in E4_RE:
                    Re = f * rs
                    if Re + 4 * sx > N - 2:
                        continue
                    rows.append(e4_packet(ch, k0, sx, Re, Robs, R0d, args.tau_max_over_rs * rs, args.dt))
                E0 = math.sqrt(m * m + math.sin(k0) ** 2)
                sat = E0 / (E0 - math.sin(2 * k0) / 2)
                run["series"].append({"k0": k0, "sigma_x": sx, "control": "sigma_x doubled at pi/16" if sx == 32.0 and abs(k0 - np.pi / 16) < 1e-12 else None,
                                      "saturation": sat, "rows": rows, "fits": e4_fits(rows, kappa, sat)})
                print("e4 m %g kj %g k0 %.4f sx %g: fits %s (%.0f s)" % (m, kj, k0, sx, {k: (v["ln_1pz_vs_kappa_t_arr"] or {}).get("slope")
                      for k, v in run["series"][-1]["fits"].items()}, time.time() - T0), flush=True)
            res["runs"].append(run)
            ch.free()
    return res


# ---------------------------------------------------------------- E3
def mode_e3(args):
    rs, Rm, N = 32.0, 8.0, 1024
    Rin = math.sqrt(Rm ** 3 / rs)
    ann = [(Rin, rs / 2), (rs / 2, 0.75 * rs), (0.75 * rs, rs)]
    nfft = 4096
    kf = 2 * np.pi * np.fft.fftfreq(nfft)
    res = {"label": LABEL, "mode": "e3", "rs": rs, "Rm": Rm, "N": N, "R_inner": Rin, "annuli": ann, "runs": []}
    for kj in args.kappas:
        ch = Chain(N, 0.0, kj, rs, Rm, None)
        for omr in (1.0, 2.0, 4.0):
            psi0, R0, sx, om, sgw, vg0 = d3_packet(N, rs, Rm, 0.0, kj, omr)
            c = ch.coef(psi0)
            ro = ch.readout(c, 0.0)
            omb = ro["omega_mean"]
            T = (R0 - Rm) / max(vg0, 1e-3) * 1.5
            times = np.linspace(0, T, 241)
            track = []
            for t in times:
                psi = ch.state(c, t)
                rho = V4.density(psi)
                row = {"t": float(t), "in_rs": float(rho[ch.R < rs].sum()), "in_half": float(rho[ch.R < rs / 2].sum()),
                       "in_Rinner": float(rho[ch.R < Rin].sum()), "beyond_2rs": float(rho[ch.R >= 2 * rs].sum()), "annuli": []}
                a2 = psi.reshape(-1, 2)
                for lo, hi in ann:
                    sel = (ch.R >= lo) & (ch.R < hi)
                    nrm = float(rho[sel].sum())
                    if nrm < 1e-4:
                        row["annuli"].append({"norm": nrm})
                        continue
                    a = np.where(sel[:, None], a2, 0.0)
                    pk = (np.abs(np.fft.fft(a, n=nfft, axis=0)) ** 2).sum(1)
                    pk = pk / pk.sum()
                    pos = kf > 0
                    wpos = float(pk[pos].sum())
                    row["annuli"].append({"norm": nrm, "weight_k_pos": wpos, "mean_k_pos": float((kf[pos] * pk[pos]).sum() / max(wpos, 1e-300)),
                                          "peak_k": float(kf[int(np.argmax(pk))])})
                track.append(row)
            nin = np.array([r["in_rs"] for r in track])
            ip = int(np.argmax(nin))
            up = np.where(nin[:ip + 1] >= 0.5 * nin[ip])[0]
            dn = np.where(nin[ip:] <= 0.5 * nin[ip])[0]
            roots = {}
            for lo, hi in ann:
                Rmid = 0.5 * (lo + hi)
                roots["%.1f-%.1f" % (lo, hi)] = {"mid": root_inside(omb, Rmid, rs, Rm, 0.0, kj), "lo": root_inside(omb, lo + 1e-6, rs, Rm, 0.0, kj),
                                                  "hi": root_inside(omb, hi - 1e-6, rs, Rm, 0.0, kj)}
            p0 = ingoing_root(omb, R0, rs, Rm, 0.0, kj)
            sym = {}
            if p0 is not None:
                ts = np.linspace(0, T, 4001)
                Rs, Ps = V4.symbol_ode(R0, p0, 0.0, kj, rs, Rm, ts)
                Rs = np.asarray(Rs)
                cin = np.where((Rs[:-1] >= rs) & (Rs[1:] < rs))[0]
                cout = np.where((Rs[:-1] < rs) & (Rs[1:] >= rs))[0]
                sym = {"p0": p0, "t_entry": float(ts[cin[0]]) if len(cin) else None, "t_exit": float(ts[cout[0]]) if len(cout) else None,
                       "R_min": float(Rs.min())}
            psiT = ch.state(c, T)
            mask = np.repeat(ch.R >= 2 * rs, 2)
            ret = np.where(mask, psiT, 0.0)
            rret = ch.readout(ch.coef(ret), 0.0)
            res["runs"].append({"kappa_j": kj, "omega_rs": omr, "omega": om, "sigma_omega": sgw, "R0": R0, "sigma_x": sx, "launched": ro,
                                "returned_beyond_2rs": rret, "t_entry_half": float(times[up[0]]) if len(up) else None,
                                "t_exit_half": float(times[ip + dn[0]]) if len(dn) else None, "symbol_ray": sym, "roots_inside": roots,
                                "track": track})
            print("e3 kj %g omr %g: entry %s exit %s, symbol %s (%.0f s)" % (kj, omr, res["runs"][-1]["t_entry_half"], res["runs"][-1]["t_exit_half"],
                                                                               sym, time.time() - T0), flush=True)
        ch.free()
    return res


# ---------------------------------------------------------------- E2
def e2_packet(N, rs, Rm, m, kj, omr, frac):
    """(B7) the ingoing packet far out on the local ingoing branch with the local spinor."""
    om = omr / rs
    sgw = om * frac
    if m > 0:
        sgw = min(sgw, (om - m) / 4)
    if sgw <= 0:
        return None
    R0 = max(16 * rs, 10 * abs(kj))
    for _ in range(3):
        vext, aext = float(V4.pg_profile(R0, rs, Rm)), kj / R0
        me = float(np.sqrt(m * m + aext * aext))
        try:
            k, omk, a2 = V4.branch_momenta(om, sgw, me, vext, outgoing=False)
        except ValueError:
            return None
        vg0 = abs(float(np.sum(a2 * V4.group_velocity(k, me, vext) * np.gradient(k))))
        sx = vg0 / (2 * sgw)
        R0n = max(16 * rs, 10 * abs(kj), 2 * rs + 4 * sx)
        if abs(R0n - R0) < 1e-9:
            break
        R0 = R0n
    if R0 + 4 * sx > N - 2:
        return None
    psi0 = V4.build_packet(N, int(round(R0 - 1)), k, a2, me, m=m, a=kj / R0)
    return psi0, R0, sx, om, sgw, vg0


def e2_run(ch, psi0, R0, sx, vg0, dt):
    rs = ch.rs
    c = ch.coef(psi0)
    T = 1.5 * (R0 + 4 * sx) / max(vg0, 0.05) + 8 * rs
    times = np.arange(0.0, T + 1e-9, dt)
    c1, c2 = cut_index(rs), cut_index(rs / 2)
    A = ch.rows_series(c, range(0, 2 * c1 + 4), times)
    Nin = (np.abs(A[:2 * c1]) ** 2).sum(0)
    J1 = current_in(A[2 * c1 - 4:2 * c1 + 4], ch.block(c1))
    J2 = current_in(A[2 * c2 - 4:2 * c2 + 4], ch.block(c2))
    Fin2 = cumtrapz0(np.maximum(J2, 0.0), dt)
    out = {"T_run": T, "dt": dt}
    front = np.where(Fin2 > 1e-6)[0]
    tg = None
    ig = None
    if len(front):
        tmin = float(times[front[0]]) + (rs / 2) / VMAX_IN + (rs / 2) / VMAX_OUT
        Fo = cumtrapz0(np.maximum(-J2, 0.0), dt)
        i0 = min(int(np.searchsorted(times, tmin)), len(times) - 1)
        after = Fo - Fo[i0]
        hit = np.where((times > tmin) & (after > 1e-3 * np.maximum(Fin2, 1e-300)))[0]
        if len(hit):
            ig = int(hit[0])
            tg = float(times[ig])
        out.update({"t_front_gate": float(times[front[0]]), "t_min_return": tmin, "gate_outward_before_t_min": float(Fo[i0])})
    F1 = cumtrapz0(J1, dt)
    iT = ig if ig is not None else len(times) - 1
    out.update({"t_gate": tg, "T_first_passage": float(F1[iT]), "T_norm_inside": float(Nin[iT] - Nin[0]), "N_inside_end": float(Nin[-1]),
                "N_inside_max": float(Nin.max()), "inflow_gate": float(Fin2[iT])})
    out["return_after_gate"] = float(-(F1[-1] - F1[iT])) if ig is not None else None
    return out


def mode_e2(args):
    rs, Rm = args.rs, args.Rm
    bc = 1.5 * math.sqrt(3.0) * rs
    jobs = []
    for m in args.masses:
        oms = [o for o in args.omega_rs if o / rs > m]
        Kmax = max(int(math.floor(1.5 * (o / rs) * bc + 2)) for o in oms)
        for ka in range(1, Kmax + 1):
            for sg in (-1.0, 1.0):
                jobs.append((m, sg * ka))
    if args.e2_rmax_control:
        jobs = [(args.masses[0], -1.0)]
    jobs = jobs[args.shard::args.shards]
    res = {"label": LABEL, "mode": "e2", "rs": rs, "Rm": Rm, "b_c": bc, "shard": args.shard, "shards": args.shards,
           "rmax_control": bool(args.e2_rmax_control), "channels": []}
    for m, kj in jobs:
        oms = [o for o in args.omega_rs if o / rs > m and abs(kj) <= math.floor(1.5 * (o / rs) * bc + 2)]
        groups = {}
        for o in oms:
            N = 4096 if o <= 2 else 2048
            if args.e2_rmax_control:
                N = 4096
            groups.setdefault(N, []).append(o)
        for N, olist in sorted(groups.items()):
            ch = Chain(N, m, kj, rs, Rm, args.gpu)
            for o in olist:
                variants = [("base", 1.0 / 8, args.dt)]
                if o == 4 and m == 0:
                    variants.append(("sigma_omega_halved", 1.0 / 16, args.dt))
                    if kj == -1.0:
                        variants.append(("dt_halved", 1.0 / 8, args.dt / 2))
                for vname, frac, dtv in variants:
                    pk = e2_packet(N, rs, Rm, m, kj, o, frac)
                    rec = {"mass": m, "kappa_j": kj, "omega_rs": o, "N": N, "variant": vname}
                    if pk is None:
                        rec["skipped"] = "the packet does not fit or no momenta in the window"
                        res["channels"].append(rec)
                        continue
                    psi0, R0, sx, om, sgw, vg0 = pk
                    ro = ch.readout(ch.coef(psi0), m)
                    rec.update({"R0": R0, "sigma_x": sx, "omega": om, "sigma_omega": sgw, "vg0": vg0, "launched": ro,
                                "killing_flag_spread_above_omega_over_6": bool(ro["omega_spread"] > om / 6)})
                    rec.update(e2_run(ch, psi0, R0, sx, vg0, dtv))
                    res["channels"].append(rec)
            ch.free()
        print("e2 m %g kj %g done (%.0f s)" % (m, kj, time.time() - T0), flush=True)
    return res


T0 = time.time()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["e0", "e1", "e2", "e3", "e4"])
    p.add_argument("--rs", type=float, default=128.0)
    p.add_argument("--Rm", type=float, default=32.0)
    p.add_argument("--nsites", type=int, default=2048)
    p.add_argument("--masses", type=float, nargs="+", default=[0.0])
    p.add_argument("--kappas", type=float, nargs="+", default=[-1.0, 1.0])
    p.add_argument("--dt", type=float, default=0.5)
    p.add_argument("--tmax-over-rs", type=float, default=10.0)
    p.add_argument("--rmax-control", action="store_true")
    p.add_argument("--robs-over-rs", type=float, default=8.0)
    p.add_argument("--tau-max-over-rs", type=float, default=36.0)
    p.add_argument("--omega-rs", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0, 16.0])
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--e2-rmax-control", action="store_true")
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = {"e0": mode_e0, "e1": mode_e1, "e2": mode_e2, "e3": mode_e3, "e4": mode_e4}[a.mode](a)
    out["wall_s"] = time.time() - T0
    out["script"] = os.path.basename(__file__)
    out["args"] = vars(a)
    os.makedirs(os.path.dirname(a.output) or ".", exist_ok=True)
    with open(a.output, "w") as f:
        json.dump(out, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("wrote %s (%.0f s)" % (a.output, out["wall_s"]), flush=True)


if __name__ == "__main__":
    main()
