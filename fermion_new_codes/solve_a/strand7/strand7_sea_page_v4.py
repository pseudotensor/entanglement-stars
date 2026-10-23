#!/usr/bin/env python
"""Strand 7, item 7: the page curve and the correlations of the light sector along the external-frame collapse
(SPEC_strand7_page.md; the lead's first draft, reviewer A on the provider API, cut at the API's output limit inside connection_control;
completed by the builder, analyst: connection_control, run_channel, main, and shell_series's record keys; marked "builder's completion").

Builds on strand7_sea_v1.py (Background: the S7a fields in T and r with the exterior forms beyond the S7a grid; Channel.sectors: the two
2n x 2n sectors of the massless light packet's chiral h per channel j, with parts=True also U = X G^-1/2 and B_0 = G^1/4 L G^1/4).
The sea is evolved as in the core (one exact exponential of h_s at the midpoint of each refresh interval tau); at the output times the
item 7 quantities are added per sector (SPEC_strand7_page.md sec. 4):
  (a) the Peschel entropy S_X = sum h2(eig C_X), C = Phi Phi^dag, of the interior side of each cut (the exterior side at the check
      times) for the evolved sea, the initial vacuum Phi(0) and the instantaneous vacuum Phi_vac(T); dS0 = S[Phi(T)] - S[Phi(0)],
      dSad = S[Phi(T)] - S[Phi_vac(T)];
  (b) the (d, e)-frame entropy of the interior domain nodes (Phi' = V Phi, V = (1/sqrt2)[[1, U^T], [1, -U^T]]; the frame's vacuum has S' = 0);
  (c) the mutual information between the slabs [rho - w, rho) and [rho, rho + w), w = 4h, for the three states;
  (d) the two-point function across the cut: G(d) = sum over the nearest domain and range nodes at rho - d and rho + d of |C_ab|^2;
  (e) the occupations n_a of all 2n eigenmodes of h_s(T), <r>_a, the exterior (<r>_a > rho_rad) and near-surface diagonal entropies,
      N_ext, the bins of 1 T_H over [0, 8] T_H (particles and holes), the raw-odds line over [1, 6] T_H (labelled: contains the transmission).
Cuts (coordinate radii): throat A(T), shell edge r_p99(T) + 2h (from the S7a JSON's monitors), barrier 4A(T), radiation 8A(T) = 2 r_s(T).
Controls: --flat and --static-from-over-rs 0 (no production; the eigendecomposition cached), the frame identity and S'(vac) = 0 at every
output, the purity S_int = S_ext at the check times, --connection-control (the moving-frame step with Gam = (i/2) U^T U_dot against the
fixed-basis step at tau and tau/2: the difference ratio about 1/8 with the connection and about 1/2 without).
Writes <out>/s7p_<tag>.json (every scalar per channel and output) and <out>/s7p_<tag>_modes.npz (omega, n_a, <r>_a per channel, sector, time).
Units: the fields file's lattice units (h = 1 or 0.5; r_s0 = 2 E_0 read from the S7a JSON meta).

Version 2 (builder analyst; SPEC_strand7_page.md sec. 12.7, the lead's code review, corrections (1) to (7)): (1) LockedBackground, the
prescribed locked collapse L of sec. 12.2 (psi = 1 + A_0/max(r, s(T)), chi = 1 - A_0/max(r, s(T)), A = B = A_0; s(T) the one-particle
characteristic on that exterior by DOP853 with dense output, rtol 1e-12, atol 1e-14; its energy control N W = N(r_0) on a 0.1 r_s0 grid;
static_from for S0 on L); (2) Cuts for L from s(T) and N_s(T) on a 0.1 r_s0 grid, the window with T_H = 1/(4 pi 4 A_0); (3) main's
--background {fields, locked, flat}, --A0-over-rs0, --R0-over-rs, --rs0; (4) the slab width in the fixed length unit, w = 4 (h if h >= 1
else 1); (5) a "shell" entry per output (L: s, N_s; C: r_sh, r_p99, N_sh by interpolation of the S7a monitors; the shell cut's areal
radius); (6) --check-times per run (as before); (7) the fields background's last time in the record (as before).
Version 3 (builder): the sectors through ChannelSVD, the same polar factors computed stably.  At j >= 3/2 the radial matrix X (2560 nodes)
has a near-null edge mode at the outer wall (smallest singular value of order n^-(j + 1)), so G = X^T X has an eigenvalue below rounding;
version 1's G^-1/4 from eigh(G) then gives NaN (the GPU solver's failures at j = 3/2 and 5/2 were on those NaN matrices).  With X = W S V^T
(SVD), U = X G^-1/2 = W V^T and B_0 = G^1/4 L G^1/4 = V S^1/2 V^T L V S^1/2 V^T, so the sector block T = U B_0 needs no negative power;
identical to version 1's route where both are finite (checked at j = 1/2 before the runs).
Version 4 (builder; speed): the same polar factors through eigh(G) instead of the GPU SVD (5.5 s per 2560 x 2560 matrix on the A6000 against
0.5 s): with G = X^T X = Z diag(w) Z^T, Y = X Z has orthogonal columns of norms sigma_k (the singular values, taken from the column norms, not
from w, so the near-null one keeps its size), U = (Y/sigma) Z^T, the columns with sigma below 1e-6 sigma_max re-orthogonalized against the
others (Gram-Schmidt, two passes: the near-null edge mode at the outer wall), G^1/4 = Z diag(sigma^1/2) Z^T; agreement with the SVD route 1e-12
and U^T U - 1 at 1e-13 in the builder's benchmark (bench_polar_v1.py at j = 1/2, 3/2, 7/2).  Also (builder): every
dense symmetric eigendecomposition goes through eigh_safe, which falls back to the CPU (LAPACK) when the GPU's syevd does not converge
(version 1's runs at j = 3/2 and 5/2 stopped at T = 0 on that error) and counts the fallbacks in the record."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import math
import sys
import time

import numpy as np
import torch
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(HERE, "..", "strand5"))
from strand7_sea_v1 import Background, Channel  # noqa: E402
import strand7_sea_v1 as SV1  # noqa: E402

C128 = torch.complex128
CUT_DISTANCES = {"throat": (1, 2, 4), "shell": (2, 4, 8, 16), "barrier": (2, 4, 8, 16), "rad": (2, 4, 8, 16)}  # in units of h at h = 1


def h2(x):
    """binary entropy in nats; equal to the paper's lattice_entropy.h2 (control C-pg-6)."""
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    out = np.zeros_like(x)
    ok = (x > 0) & (x < 1)
    out[ok] = -x[ok] * np.log(x[ok]) - (1 - x[ok]) * np.log1p(-x[ok])
    return out


EIGH_FALLBACKS = {"count": 0}


def eigh_safe(H):
    """torch.linalg.eigh on H's device; on a convergence failure of the GPU solver, the same on the CPU (LAPACK), moved back (counted)."""
    try:
        return torch.linalg.eigh(H)
    except Exception:
        if H.device.type == "cpu":
            raise
        EIGH_FALLBACKS["count"] += 1
        lam, V = torch.linalg.eigh(H.cpu())
        return lam.to(H.device), V.to(H.device)


def eigvalsh_safe(H):
    try:
        return torch.linalg.eigvalsh(H)
    except Exception:
        if H.device.type == "cpu":
            raise
        EIGH_FALLBACKS["count"] += 1
        return torch.linalg.eigvalsh(H.cpu()).to(H.device)


class ChannelSVD(Channel):
    """Channel with the sectors' polar factors from the SVD of X (version 3; see the docstring)."""

    def sectors(self, bg, T, parts=False):
        psi_i, chi_i, A, B = bg.at(T, self.r_int)
        psi_h, chi_h, _, _ = bg.at(T, self.r_half)
        Ap, k0 = SV1.V1.channel_matrix(self.r_int, self.r_half, psi_i, psi_h, 0.0, self.j, self.h)
        if k0 != 0:
            raise SystemExit("channel cut k0 = %d at j = %g (not expected for j <= 9/2)" % (k0, self.j))
        n = self.n
        P = torch.as_tensor(Ap[:n, n:], dtype=torch.float64, device=self.dev)
        Q = torch.as_tensor(Ap[n:, :n], dtype=torch.float64, device=self.dev)
        out, prt = {}, {}
        for name, X, Ldom in (("alpha", Q, chi_i), ("beta", P, chi_h)):
            try:
                Wl, S, Vh = torch.linalg.svd(X, full_matrices=False)
            except Exception:
                EIGH_FALLBACKS["count"] += 1
                Wl, S, Vh = (t.to(X.device) for t in torch.linalg.svd(X.cpu(), full_matrices=False))
            U = Wl @ Vh
            Gp = (Vh.T * S ** 0.5) @ Vh
            L = torch.as_tensor(Ldom, dtype=torch.float64, device=self.dev)
            B0 = Gp @ (L[:, None] * Gp)
            B0 = 0.5 * (B0 + B0.T)
            Tm = U @ B0
            H = torch.zeros(2 * n, 2 * n, dtype=torch.float64, device=self.dev)
            H[:n, n:] = Tm.T
            H[n:, :n] = Tm
            out[name] = H
            if parts:
                prt[name] = (U, B0)
        return (out, A, B, prt) if parts else (out, A, B)


class ChannelFast(Channel):
    """Channel with the sectors' polar factors through eigh(G) and normalized columns (version 4; see the docstring)."""

    def sectors(self, bg, T, parts=False):
        psi_i, chi_i, A, B = bg.at(T, self.r_int)
        psi_h, chi_h, _, _ = bg.at(T, self.r_half)
        Ap, k0 = SV1.V1.channel_matrix(self.r_int, self.r_half, psi_i, psi_h, 0.0, self.j, self.h)
        if k0 != 0:
            raise SystemExit("channel cut k0 = %d at j = %g (not expected for j <= 9/2)" % (k0, self.j))
        n = self.n
        P = torch.as_tensor(Ap[:n, n:], dtype=torch.float64, device=self.dev)
        Q = torch.as_tensor(Ap[n:, :n], dtype=torch.float64, device=self.dev)
        out, prt = {}, {}
        for name, X, Ldom in (("alpha", Q, chi_i), ("beta", P, chi_h)):
            w, Z = eigh_safe(X.T @ X)
            Y = X @ Z
            sig = torch.linalg.norm(Y, dim=0)
            Yn = Y / sig
            small = torch.nonzero(sig < 1e-6 * sig.max()).flatten().tolist()
            for k in small:
                keep = torch.ones(Yn.shape[1], dtype=torch.bool, device=self.dev)
                keep[k] = False
                Yo = Yn[:, keep]
                v = Yn[:, k].clone()
                for _ in range(2):
                    v = v - Yo @ (Yo.T @ v)
                Yn[:, k] = v / torch.linalg.norm(v)
            U = Yn @ Z.T
            Gp = (Z * sig ** 0.5) @ Z.T
            L = torch.as_tensor(Ldom, dtype=torch.float64, device=self.dev)
            B0 = Gp @ (L[:, None] * Gp)
            B0 = 0.5 * (B0 + B0.T)
            Tm = U @ B0
            H = torch.zeros(2 * n, 2 * n, dtype=torch.float64, device=self.dev)
            H[:n, n:] = Tm.T
            H[n:, :n] = Tm
            out[name] = H
            if parts:
                prt[name] = (U, B0)
        return (out, A, B, prt) if parts else (out, A, B)


class LockedBackground:
    """the prescribed locked collapse L (SPEC_strand7_page.md sec. 12.2): the exterior form of the fixed charge A_0 outside the shell radius
    s(T), the continuity values inside; s(T) the test-particle characteristic of the one-particle law on that exterior (m = 1, j = 0) from
    rest at the coordinate radius r_0 with r_0 (1 + A_0/r_0)^2 = R_0 (the root outside the throat)."""

    def __init__(self, A0, rs0, R0, T_end, static_from=None):
        self.A0, self.rs0, self.R0 = float(A0), float(rs0), float(R0)
        self.flat, self.static_from, self.Tlast = False, static_from, np.inf
        # r (1 + A0/r)^2 = R0  <=>  r^2 + (2 A0 - R0) r + A0^2 = 0, the larger root
        b = 2 * A0 - R0
        self.r0 = float((-b + math.sqrt(b * b - 4 * A0 * A0)) / 2)
        self.eps = self.N(self.r0)
        self.T_int = float(T_end)
        sol = solve_ivp(self._rhs, (0.0, self.T_int), [self.r0, 0.0], method="DOP853", rtol=1e-12, atol=1e-14, dense_output=True)
        if not sol.success:
            raise SystemExit("LockedBackground: the characteristic's integration failed: %s" % sol.message)
        self.sol = sol
        Tg = np.arange(0.0, self.T_int + 1e-9, 0.1 * rs0)
        y = sol.sol(Tg)
        r, p = y[0], y[1]
        Wg = np.sqrt(1 + self.psi(r) ** -4 * p * p)
        self.energy_control = {"eps": self.eps, "max_rel_dev_NW": float(np.max(np.abs(self.N(r) * Wg / self.eps - 1))), "grid_step_over_rs0": 0.1,
                               "T_int_over_rs0": self.T_int / rs0, "nfev": int(sol.nfev), "r0": self.r0, "r0_over_A0": self.r0 / self.A0}

    def psi(self, r):
        return 1 + self.A0 / r

    def N(self, r):
        return (1 - self.A0 / r) / (1 + self.A0 / r)

    def _rhs(self, T, y):
        r, p = y
        A = self.A0
        ps = 1 + A / r
        ch = 1 - A / r
        N = ch / ps
        dps = -A / r ** 2
        dch = A / r ** 2
        dN = (dch * ps - ch * dps) / ps ** 2
        q = ps ** -4
        dq = -4 * ps ** -5 * dps
        W = math.sqrt(1 + q * p * p)
        return [N * q * p / W, -(dN * W + N * dq * p * p / (2 * W))]

    def s(self, T):
        Tc = min(T, self.static_from) if self.static_from is not None else T
        Tc = min(Tc, self.T_int)
        return float(self.sol.sol(Tc)[0])

    def N_s(self, T):
        return float(self.N(self.s(T)))

    def at(self, T, rr):
        sT = self.s(T)
        rm = np.maximum(np.asarray(rr, float), sT)
        return 1 + self.A0 / rm, 1 - self.A0 / rm, self.A0, self.A0


def peschel(rows):
    """S_X of the Slater state from its orbitals restricted to the region's rows (|X| x n complex, on the GPU)."""
    if rows.shape[0] == 0:
        return 0.0
    C = rows @ rows.conj().T
    lam = eigvalsh_safe(C).real.cpu().numpy()
    return float(h2(lam).sum())


def two_point(Phi, a, b):
    """sum over a in in(d), b in out(d) of |C_ab|^2 with C = Phi Phi^dag."""
    M = Phi[a, :] @ Phi[b, :].conj().T
    return float((M.abs() ** 2).sum().item())


def de_frame(Phi, U, n):
    """Phi' = V Phi: the d and the e components on the domain nodes (2n x n)."""
    Ut = U.T.to(C128)
    d = (Phi[:n] + Ut @ Phi[n:]) / math.sqrt(2)
    e = (Phi[:n] - Ut @ Phi[n:]) / math.sqrt(2)
    return torch.cat([d, e], 0)


def shell_series(S7):
    """the S7a JSON's per-output monitors (strand7_shell_v2/v3.py: run.records with T, r_sh (the median), r_p99, N_sh = N(r_sh));
    the builder's completion of the record keys."""
    mon = S7["run"]["records"]
    T = np.array([m["T"] for m in mon], dtype=float)
    r_sh = np.array([m["r_sh"] for m in mon], dtype=float)
    r_p99 = np.array([m["r_p99"] for m in mon], dtype=float)
    N_sh = np.array([m["N_sh"] for m in mon], dtype=float)
    return T, r_sh, r_p99, N_sh


class Cuts:
    """the four cuts at the clock T (coordinate radii); the shell edge from the S7a monitors, the others from A(T)."""

    def __init__(self, S7, h, flat=False, locked=None, T_end=None):
        self.h, self.flat, self.locked = h, flat, locked
        if locked is not None:
            self.Tm = np.arange(0.0, T_end + 1e-9, 0.1 * locked.rs0)
            self.r_p99 = np.array([locked.s(t) for t in self.Tm])
            self.r_sh = self.r_p99.copy()
            self.N_sh = np.array([locked.N_s(t) for t in self.Tm])
        elif S7 is not None and not flat:
            self.Tm, self.r_sh, self.r_p99, self.N_sh = shell_series(S7)
        else:
            self.Tm = None

    def shell(self, T):
        """the shell's record at T: L, s and N_s from the class; C, r_sh, r_p99 and N_sh interpolated from the S7a monitors."""
        if self.Tm is None:
            return None
        if self.locked is not None:
            return {"s": self.locked.s(T), "r_p99": self.locked.s(T), "N_s": self.locked.N_s(T)}
        return {"r_sh": float(np.interp(T, self.Tm, self.r_sh)), "r_p99": float(np.interp(T, self.Tm, self.r_p99)),
                "N_s": float(np.interp(T, self.Tm, self.N_sh))}

    def at(self, T, A, rs0):
        if self.flat or A <= 0:
            A_eff = rs0 / 4.0  # the flat control: the cuts at the primary's nominal radii
        else:
            A_eff = A
        cuts = {"throat": A_eff, "barrier": 4 * A_eff, "rad": 8 * A_eff}
        if self.Tm is not None:
            cuts["shell"] = float(np.interp(T, self.Tm, self.r_p99)) + 2 * self.h
        return cuts

    def window(self, TH_of_T):
        """T_1 (N(r_sh) < 0.5) and T_2(omega) (N(r_sh) < omega h) for omega = 1, 2, 4, 8 T_H; TH_of_T a callable."""
        if self.Tm is None:
            return {}
        out = {}
        i1 = np.where(self.N_sh < 0.5)[0]
        out["T_1"] = float(self.Tm[i1[0]]) if len(i1) else None
        for k in (1, 2, 4, 8):
            thr = np.array([k * TH_of_T(t) * self.h for t in self.Tm])
            i2 = np.where(self.N_sh < thr)[0]
            out["T_2_%dTH" % k] = float(self.Tm[i2[0]]) if len(i2) else None
        return out


def schedule_tau(T_end, tau, outs):
    b = set(np.round(np.arange(0.0, T_end + 1e-9, tau), 10).tolist()) | set(np.round(outs, 10).tolist())
    return sorted(x for x in b if x <= T_end + 1e-9)


def cut_analysis(Phi, Phi0, Phivac, U, radii, ch, name, cuts, h, check_ext):
    """the entanglement, the (d, e)-frame entropy, the mutual information and the two-point functions at every cut (one sector)."""
    n = ch.n
    out = {}
    Phip = de_frame(Phi, U, n)
    Pvp = de_frame(Phivac, U, n)
    rdom = ch.r_int if name == "alpha" else ch.r_half
    rrng = ch.r_half if name == "alpha" else ch.r_int
    states = (("state", Phi), ("vac0", Phi0), ("vacad", Phivac))
    for cname, rho in cuts.items():
        rin = np.where(radii < rho)[0]
        rext = np.where(radii >= rho)[0]
        row = {"rho": float(rho), "n_int": int(len(rin))}
        S = {}
        for k, P in states:
            S[k] = peschel(P[rin, :])
            row["S_int_" + k] = S[k]
            if check_ext:
                row["S_ext_" + k] = peschel(P[rext, :])
        row["dS0"] = S["state"] - S["vac0"]
        row["dSad"] = S["state"] - S["vacad"]
        din = np.where(rdom < rho)[0]
        de_rows = np.concatenate([din, n + din])
        row["S_de_int_state"] = peschel(Phip[de_rows, :])
        row["S_de_int_vacad"] = peschel(Pvp[de_rows, :])  # the frame's vacuum: 0 to rounding (C-pg-5)
        w = 4 * (h if h >= 1 else 1.0)  # the fixed length unit (sec. 12.7 (4))
        Aw = np.where((radii >= rho - w) & (radii < rho))[0]
        Bw = np.where((radii >= rho) & (radii < rho + w))[0]
        AB = np.concatenate([Aw, Bw])
        for k, P in states:
            row["MI_" + k] = peschel(P[Aw, :]) + peschel(P[Bw, :]) - peschel(P[AB, :])
        for d_units in CUT_DISTANCES[cname]:
            d = d_units * (h if h >= 1 else 1.0)  # the same distances in units of r_s0 when h is halved (spec 3.2)
            a = [int(np.argmin(np.abs(rdom - (rho - d)))), n + int(np.argmin(np.abs(rrng - (rho - d))))]
            b = [int(np.argmin(np.abs(rdom - (rho + d)))), n + int(np.argmin(np.abs(rrng - (rho + d))))]
            for k, P in states:
                row["G_%s_d%g" % (k, d)] = two_point(P, a, b)
            row["g_d%g" % d] = row["G_state_d%g" % d] / row["G_vac0_d%g" % d] if row["G_vac0_d%g" % d] > 0 else None
        out[cname] = row
    return out


def spectrum_rows(om, na, rr, rho_rad, TH, bin_TH=1.0, bins_max_TH=8.0):
    """the exterior and near-surface diagonal entropies, N_ext, the bins (particles and holes) and the raw-odds line (labelled)."""
    ext = rr > rho_rad
    pos = om > 0
    out = {"S_diag_rad": float(h2(na[ext]).sum()), "S_diag_near": float(h2(na[~ext]).sum()),
           "N_ext": float(na[ext & pos].sum()), "N_near": float(na[~ext & pos].sum()),
           "near_count": int(np.sum(~ext)), "exterior_total_all": float(na[ext].sum())}
    edges = np.arange(0.0, bins_max_TH + 1e-9, bin_TH) * TH
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sp = ext & pos & (om >= lo) & (om < hi)
        sh = ext & (~pos) & (-om >= lo) & (-om < hi)
        bins.append({"omega_lo_TH": float(lo / TH), "omega_hi_TH": float(hi / TH),
                     "particles": {"modes": int(sp.sum()), "sum": float(na[sp].sum()), "mean": float(na[sp].mean()) if sp.any() else None},
                     "holes": {"modes": int(sh.sum()), "sum": float((1 - na[sh]).sum()), "mean": float((1 - na[sh]).mean()) if sh.any() else None}})
    out["bins"] = bins
    xs, ys = [], []
    for b in bins:
        mid = 0.5 * (b["omega_lo_TH"] + b["omega_hi_TH"])
        nb = b["particles"]["mean"]
        if 1.0 <= mid <= 6.0 and nb is not None and 0 < nb < 1:
            xs.append(mid)
            ys.append(math.log(1.0 / nb - 1.0))
    if len(xs) >= 2:
        slope, icpt = np.polyfit(xs, ys, 1)
        out["raw_odds"] = {"slope_per_TH": float(slope), "intercept": float(icpt), "bins_used": len(xs),
                           "label": "contains the transmission Gamma_j(omega): not a temperature (P12)"}
    else:
        out["raw_odds"] = None
    return out




def connection_control(ch, bg, T0, tau, dev, Phi):
    """C-pg-11 (builder's completion of the lead's draft): on sector alpha at T0, the fixed-basis step V(T0 + t) e^{-i h(T0 + t/2) t} Phi
    against the moving-frame step e^{-i G_mov t} V(T0) Phi with G_mov = [[B_0 - Gam, Gam], [Gam, -B_0 - Gam]], Gam = (i/2) U^T U_dot at the
    midpoint (U_dot by the central difference over t/20), and against the same with Gam dropped, at t = tau and tau/2; returns the
    differences' norms and their ratios (expected about 1/8 with the connection and about 1/2 without)."""
    n = ch.n
    I = torch.eye(n, dtype=torch.float64, device=dev)

    def Vmat(U):
        return torch.cat([torch.cat([I, U.T], 1), torch.cat([I, -U.T], 1)], 0) / math.sqrt(2)
    res = {}
    for tt in (tau, 0.5 * tau):
        H0, _, _, p0 = ch.sectors(bg, T0, parts=True)
        H1, _, _, p1 = ch.sectors(bg, T0 + tt, parts=True)
        Hm, _, _, pm = ch.sectors(bg, T0 + 0.5 * tt, parts=True)
        eps = tt / 20.0
        _, _, _, pp = ch.sectors(bg, T0 + 0.5 * tt + eps, parts=True)
        _, _, _, pq = ch.sectors(bg, T0 + 0.5 * tt - eps, parts=True)
        U0, U1, Um = p0["alpha"][0], p1["alpha"][0], pm["alpha"][0]
        Bm = pm["alpha"][1]
        Udot = (pp["alpha"][0] - pq["alpha"][0]) / (2 * eps)
        lam, V = eigh_safe(Hm["alpha"])
        Vc = V.to(C128)
        fixed = Vmat(U1).to(C128) @ (Vc @ (torch.exp(-1j * lam * tt).to(C128)[:, None] * (Vc.T @ Phi)))
        Gam = 0.5j * (Um.T @ Udot).to(C128)
        Bc = Bm.to(C128)
        out = {}
        for lab, G_ in (("with", Gam), ("without", torch.zeros_like(Gam))):
            Gm = torch.cat([torch.cat([Bc - G_, G_], 1), torch.cat([G_, -Bc - G_], 1)], 0)
            Gm = 0.5 * (Gm + Gm.conj().T)
            w, Q = eigh_safe(Gm)
            mov = Q @ (torch.exp(-1j * w * tt)[:, None] * (Q.conj().T @ (Vmat(U0).to(C128) @ Phi)))
            out[lab] = torch.linalg.norm(fixed - mov).item()
        res["tau_%g" % tt] = out
    k1, k2 = "tau_%g" % tau, "tau_%g" % (0.5 * tau)
    res["ratio_with"] = res[k2]["with"] / res[k1]["with"] if res[k1]["with"] > 0 else None
    res["ratio_without"] = res[k2]["without"] / res[k1]["without"] if res[k1]["without"] > 0 else None
    return res


def run_channel(a, bg, S7, j, dev, outs, rs0, cuts_obj, check_times):
    """the sea of channel j: evolution, and at every output time the item 7 quantities per sector (builder's completion)."""
    ch = ChannelFast(j, a.n, a.h, dev)
    Hs, A0, B0_ = ch.sectors(bg, 0.0)
    Phi, Phi0 = {}, {}
    for name, H in Hs.items():
        lam, V = eigh_safe(H)
        Phi[name] = V[:, :a.n].to(C128)
        Phi0[name] = Phi[name].clone()
    bounds = schedule_tau(a.T_end, a.tau, outs)
    recs, modes, conn = [], {}, None
    cache = {}
    static = bg.flat or (bg.static_from is not None and bg.static_from <= 0)
    t0 = time.time()
    nref = 0
    for ta, tb in zip(bounds[:-1], bounds[1:]):
        tm = 0.5 * (ta + tb)
        if static and cache:
            eig = cache
        else:
            Hs_m, _, _ = ch.sectors(bg, tm)
            eig = {name: eigh_safe(H) for name, H in Hs_m.items()}
            if static:
                cache = eig
        for name, (lam, V) in eig.items():
            Vc = V.to(C128)
            Phi[name] = Vc @ (torch.exp(-1j * lam * (tb - ta)).to(C128)[:, None] * (Vc.T @ Phi[name]))
        nref += 1
        if not any(abs(tb - o) < 1e-9 for o in outs):
            continue
        Hs_T, A, B, prt = ch.sectors(bg, tb, parts=True)
        rs = 4 * A if A > 0 else rs0
        TH = 1.0 / (4 * math.pi * rs)
        cuts = cuts_obj.at(tb, A, rs0)
        areal = {k: float(r_ * bg.at(tb, np.array([r_]))[0][0] ** 2) for k, r_ in cuts.items()}
        check_ext = any(abs(tb - c) < 1e-9 for c in check_times)
        row = {"j": j, "T": tb, "T_over_rs0": tb / rs0, "A": A, "B": B, "r_s": rs, "T_H": TH, "cuts": cuts, "cuts_areal": areal,
               "refreshes": nref, "seconds": time.time() - t0, "sectors": {}, "eigh_cpu_fallbacks": EIGH_FALLBACKS["count"]}
        sh = cuts_obj.shell(tb)
        if sh is not None:
            sh["areal_shell_cut"] = areal.get("shell")
            row["shell"] = sh
        for name, H in Hs_T.items():
            lam, V = eigh_safe(H)
            Phivac = V[:, :a.n].to(C128)
            U, B0m = prt[name]
            radii = ch.radii(name)
            ca = cut_analysis(Phi[name], Phi0[name], Phivac, U, radii, ch, name, cuts, a.h, check_ext)
            ov = V.T.to(C128) @ Phi[name]
            na = torch.sum(torch.abs(ov) ** 2, dim=1).cpu().numpy()
            om = lam.cpu().numpy()
            rr = (torch.as_tensor(radii, dtype=torch.float64, device=dev)[:, None] * V ** 2).sum(0).cpu().numpy()
            sp = spectrum_rows(om, na, rr, cuts["rad"], TH)
            I_n = torch.eye(a.n, dtype=torch.float64, device=dev)
            Vf = torch.cat([torch.cat([I_n, U.T], 1), torch.cat([I_n, -U.T], 1)], 0) / math.sqrt(2)
            Zb = torch.zeros_like(B0m)
            tgt = torch.cat([torch.cat([B0m, Zb], 1), torch.cat([Zb, -B0m], 1)], 0)
            ctrl = {"orthonormality": torch.linalg.norm(Phi[name].conj().T @ Phi[name] - torch.eye(a.n, dtype=C128, device=dev)).item(),
                    "frame_identity_rel": (torch.linalg.norm(Vf @ H @ Vf.T - tgt) / torch.linalg.norm(B0m)).item(),
                    "U_orthogonality": torch.linalg.norm(U.T @ U - I_n).item(),
                    "near_zero_modes": int(np.sum(np.abs(om) < 1e-6)), "smallest_abs_omega": float(np.min(np.abs(om)))}
            row["sectors"][name] = {"cuts": ca, "spectrum": sp, "controls": ctrl}
            modes["j%g_%s_T%g" % (j, name, tb)] = np.stack([om, na, rr]).astype(np.float32)
        if a.connection_control and conn is None and tb >= a.connection_T0 * rs0 - 1e-9:
            conn = connection_control(ch, bg, tb, a.tau, dev, Phi["alpha"])
            row["connection_control"] = conn
        recs.append(row)
        ths = row["sectors"]
        dsr = sum(ths[s]["cuts"]["rad"]["dSad"] for s in ths)
        dst = sum(ths[s]["cuts"]["throat"]["dSad"] for s in ths)
        print("j %g T/rs0 %.3g: dSad rad %.4e throat %.4e N_ext %.4e (%.0f s)" % (
            j, tb / rs0, dsr, dst, sum(ths[s]["spectrum"]["N_ext"] for s in ths), time.time() - t0), flush=True)
    return recs, modes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--background", default="fields", choices=["fields", "locked", "flat"],
                    help="fields: an S7a fields file (C); locked: the prescribed locked collapse L of sec. 12.2; flat: the flat control")
    ap.add_argument("--fields", default=None, help="the S7a fields file (strand7_shell_v3.py --save-fields 1); --background fields only")
    ap.add_argument("--s7a-json", default=None, help="the S7a run's JSON (meta r_s, the monitors for the shell edge and the window); fields only")
    ap.add_argument("--rs0", type=float, default=None, help="r_s0 in lattice units; required for locked and flat")
    ap.add_argument("--A0-over-rs0", type=float, default=0.234465, help="L's clock charge A_0 in units of r_s0 (sec. 12.2)")
    ap.add_argument("--R0-over-rs", type=float, default=4.0, help="L's release areal radius in units of r_s0")
    ap.add_argument("--channels", default="0.5,1.5,2.5")
    ap.add_argument("--h", type=float, default=1.0)
    ap.add_argument("--Rmax-over-rs", type=float, default=80.0)
    ap.add_argument("--tau-over-rs", type=float, default=0.125)
    ap.add_argument("--T-end-over-rs", type=float, default=40.0)
    ap.add_argument("--out-times", default="spec", help="'spec' (0, 2, ..., T_end and every 1 over [12, 36]) or a comma list in r_s0")
    ap.add_argument("--check-times", default="0,24,40", help="the outputs (r_s0) with the exterior sides computed (C-pg-1)")
    ap.add_argument("--static-from-over-rs", type=float, default=None)
    ap.add_argument("--connection-control", type=int, default=0)
    ap.add_argument("--connection-T0", type=float, default=24.0)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    dev = torch.device(a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu")
    S7 = None
    if a.background == "fields":
        if not a.fields or not a.s7a_json:
            raise SystemExit("--background fields needs --fields and --s7a-json")
        with open(a.s7a_json) as fh:
            S7 = json.load(fh)
        rs0 = float(S7["meta"]["r_s"])
    else:
        if a.rs0 is None:
            raise SystemExit("--background %s needs --rs0" % a.background)
        rs0 = float(a.rs0)
    a.n = int(round(a.Rmax_over_rs * rs0 / a.h))
    a.tau = a.tau_over_rs * rs0
    a.T_end = a.T_end_over_rs * rs0
    if a.out_times == "spec":
        o = set(np.arange(0, a.T_end_over_rs + 1e-9, 2.0).tolist()) | set(x for x in np.arange(12, 37, 1.0).tolist() if x <= a.T_end_over_rs)
        outs = sorted(x * rs0 for x in o)
    else:
        outs = [float(x) * rs0 for x in a.out_times.split(",")]
    check_times = [float(x) * rs0 for x in a.check_times.split(",")]
    sf = None if a.static_from_over_rs is None else a.static_from_over_rs * rs0
    locked = None
    if a.background == "flat":
        bg = Background.__new__(Background)
        bg.flat, bg.static_from, bg.Tlast, bg.r, bg.T = True, None, 0.0, np.array([0.0, 1.0]), np.array([0.0])
        cuts_obj = Cuts(None, a.h, flat=True)
    elif a.background == "locked":
        locked = LockedBackground(a.A0_over_rs0 * rs0, rs0, a.R0_over_rs * rs0, a.T_end + 2 * a.tau, static_from=sf)
        bg = locked
        cuts_obj = Cuts(None, a.h, locked=locked, T_end=a.T_end)
    else:
        bg = Background(a.fields, flat=False, static_from=sf)
        cuts_obj = Cuts(S7, a.h)
    os.makedirs(a.out, exist_ok=True)
    win = {}
    if locked is not None:
        TH = 1.0 / (4 * math.pi * 4 * locked.A0)
        win = cuts_obj.window(lambda t: TH)
        win["T_H"] = TH
    elif S7 is not None:
        recsA = S7["run"]["records"]
        TA = np.array([q["T"] for q in recsA])
        AA = np.array([q["A"] for q in recsA])
        win = cuts_obj.window(lambda t: 1.0 / (4 * math.pi * 4 * float(np.interp(t, TA, AA))))
    rec = {"meta": dict(vars(a), r_s0=rs0, device_used=str(dev), outs_over_rs0=[x / rs0 for x in outs], builder="analyst (versions 2 to 4: the lead's review, the stable sectors, their fast route)",
                        script="strand7_sea_page_v4.py", background_last_T_over_rs0=(bg.Tlast / rs0 if a.background == "fields" else None)),
           "window": win, "channels": {}}
    if locked is not None:
        rec["locked"] = {"A0": locked.A0, "r_h": 4 * locked.A0, "R0": locked.R0, "energy_control": locked.energy_control,
                         "label": "the prescribed locked collapse L (SPEC_strand7_page.md sec. 12.2): the exterior form of the fixed charge outside the "
                                  "shell, the continuity values inside, the shell on the one-particle characteristic; carries the lock by construction",
                         "s_over_A0": [[t / rs0, locked.s(t) / locked.A0, locked.N_s(t)] for t in np.arange(0.0, a.T_end + 1e-9, 0.5 * rs0)]}
    allmodes = {}
    t0 = time.time()
    for j in [float(x) for x in a.channels.split(",")]:
        r_, m_ = run_channel(a, bg, S7, j, dev, outs, rs0, cuts_obj, check_times)
        rec["channels"][str(j)] = r_
        allmodes.update(m_)
        rec["meta"]["eigh_cpu_fallbacks"] = EIGH_FALLBACKS["count"]
        with open(os.path.join(a.out, "s7p_%s.json" % a.tag), "w") as fh:
            json.dump(rec, fh, indent=1, default=float)
    rec["meta"]["seconds"] = time.time() - t0
    rec["meta"]["eigh_cpu_fallbacks"] = EIGH_FALLBACKS["count"]
    with open(os.path.join(a.out, "s7p_%s.json" % a.tag), "w") as fh:
        json.dump(rec, fh, indent=1, default=float)
    np.savez_compressed(os.path.join(a.out, "s7p_%s_modes.npz" % a.tag), **allmodes)
    print("done %s %.0f s (CPU eigh fallbacks %d)" % (a.tag, time.time() - t0, EIGH_FALLBACKS["count"]))


if __name__ == "__main__":
    main()
