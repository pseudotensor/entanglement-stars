#!/usr/bin/env python3
"""
3D cubic-lattice vacuum solver — CPU version.

Demonstrates that the 3D two-state closure naturally produces Newtonian
(1/r) gravity via the proxy equation, and tests whether the full
self-consistent equation has smooth solutions in 3D (where CDW is
suppressed by the non-nested Fermi surface).

Key differences from the spherical (1D) case:
  - Exact discrete degeneracy G_n (not 4*pi*n^2 approximation)
  - Intra-shell bonds contribute to conductances and energy
  - 3D Fermi surface: no Peierls/CDW instability at 2k_F

Usage: PYTHONUNBUFFERED=1 python3 full_numerics/solve_3d_vacuum.py
"""

import sys
import os
import time
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import least_squares

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

FIGDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "fermion", "numerical", "figures")
os.makedirs(FIGDIR, exist_ok=True)


# ============================================================================
# Binary entropy
# ============================================================================

def binary_entropy(x):
    x = np.clip(x, 1e-30, 1.0 - 1e-30)
    return -x * np.log(x) - (1.0 - x) * np.log(1.0 - x)


# ============================================================================
# 3D cubic lattice (CPU)
# ============================================================================

class CubicLattice3D:
    """3D cubic lattice within sphere of radius R_max."""

    def __init__(self, R_max):
        self.R_max = R_max
        t0 = time.time()

        coords = np.mgrid[-R_max:R_max+1, -R_max:R_max+1, -R_max:R_max+1]
        coords = coords.reshape(3, -1).T
        r2 = np.sum(coords**2, axis=1)
        mask = r2 <= R_max * R_max
        self.sites = coords[mask]
        self.N_sites = len(self.sites)
        self.r = np.sqrt(np.sum(self.sites**2, axis=1).astype(float))
        self.shell_of = np.round(self.r).astype(int)
        self.N_shell = self.shell_of.max() + 1

        self.shell_sites = [np.where(self.shell_of == n)[0]
                            for n in range(self.N_shell)]
        self.G_n = np.array([len(s) for s in self.shell_sites])

        # Build neighbor pairs
        coord_to_idx = {}
        for i in range(self.N_sites):
            coord_to_idx[tuple(self.sites[i])] = i

        row, col = [], []
        for i in range(self.N_sites):
            x, y, z = self.sites[i]
            for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                j = coord_to_idx.get((x+dx, y+dy, z+dz), -1)
                if j >= 0:
                    row.append(i)
                    col.append(j)
        self.row = np.array(row, dtype=np.int64)
        self.col = np.array(col, dtype=np.int64)

        self.shell_row = self.shell_of[self.row]
        self.shell_col = self.shell_of[self.col]

        # Forward bonds (shell n -> shell n+1)
        fwd = self.shell_col == self.shell_row + 1
        self.fwd_row = self.row[fwd]
        self.fwd_col = self.col[fwd]
        self.fwd_shell = self.shell_row[fwd]

        # Intra-shell bonds (unique pairs within same shell)
        intra = (self.shell_col == self.shell_row) & (self.row < self.col)
        self.intra_row = self.row[intra]
        self.intra_col = self.col[intra]
        self.intra_shell = self.shell_row[intra]

        elapsed = time.time() - t0
        print(f"  Lattice: R={R_max}, {self.N_sites} sites, {self.N_shell} shells, "
              f"{len(self.row)//2} bonds ({len(self.fwd_row)} fwd, "
              f"{len(self.intra_row)} intra) [{elapsed:.1f}s]", flush=True)
        print(f"  G_n: {list(self.G_n)}", flush=True)


# ============================================================================
# 3D Model (CPU)
# ============================================================================

class ThreeDModelCPU:
    """Full 3D self-consistent closure equation — CPU version."""

    def __init__(self, lat, t0=1.0, V0=0.01, n_core=2, beta0=0.1,
                 cstar_sq=0.5):
        self.lat = lat
        self.t0 = t0
        self.V0 = V0
        self.n_core = n_core
        self.beta0 = beta0
        self.cstar_sq = cstar_sq
        self.N_shell = lat.N_shell
        self.N = lat.N_shell  # for compatibility with extract_rs
        self.r = np.arange(1, self.N_shell + 1, dtype=float)
        self.n_call = 0

        # Precompute bg/tgt
        t_s = time.time()
        self.rho_bg = self._energy_profile(
            np.zeros(self.N_shell), smeared=False, add_V0=False)
        self.rho_tgt = self._energy_profile(
            np.zeros(self.N_shell), smeared=False, add_V0=True)
        self.rho_src = self.rho_bg - self.rho_tgt

        # Background MI per forward bond for normalization
        self.mi_bg_bonds = self._compute_mi_background()

        print(f"  bg/tgt in {time.time()-t_s:.1f}s  "
              f"src_norm={np.sum(np.abs(self.rho_src)):.6f}", flush=True)

    def _compute_mi_background(self):
        """Background MI per forward bond at Phi=0 (uniform chain).

        Used to normalize conductances: kappa_bond = t0^2 * MI/MI_bg.
        This removes the spurious 1/beta0^2 temperature factor of the
        old formula kappa = (4/beta0^2)*MI.
        """
        H = self._build_H(np.zeros(self.N_shell), smeared=False, add_V0=False)
        evals, evecs = eigh(H)
        be = np.clip(self.beta0 * evals, -500, 500)
        f = 1.0 / (np.exp(be) + 1.0)

        V_i = evecs[self.lat.fwd_row]
        V_j = evecs[self.lat.fwd_col]
        a = (V_i * V_i) @ f
        d = (V_j * V_j) @ f
        b = (V_i * V_j) @ f

        tr = a + d
        det = a * d - b * b
        disc = np.maximum(tr * tr - 4.0 * det, 0.0)
        lam1 = 0.5 * (tr + np.sqrt(disc))
        lam2 = 0.5 * (tr - np.sqrt(disc))
        mi = (binary_entropy(a) + binary_entropy(d)
              - binary_entropy(lam1) - binary_entropy(lam2))
        return np.maximum(mi, 1e-30)

    def _build_H(self, Phi_shell, smeared=False, add_V0=False):
        """Build dense Hamiltonian."""
        N = self.lat.N_sites
        H = np.zeros((N, N))

        if smeared:
            lapse = 1.0 + Phi_shell / self.cstar_sq
            Nbar = 0.5 * (lapse[self.lat.shell_of[self.lat.row]]
                        + lapse[self.lat.shell_of[self.lat.col]])
            H[self.lat.row, self.lat.col] = -self.t0 * np.abs(Nbar)
        else:
            H[self.lat.row, self.lat.col] = -self.t0

        if add_V0:
            for n in range(min(self.n_core, self.N_shell)):
                idx = self.lat.shell_sites[n]
                H[idx, idx] += self.V0

        return H

    def _energy_profile(self, Phi_shell, smeared=False, add_V0=False):
        """Energy per shell from correlation matrix."""
        H = self._build_H(Phi_shell, smeared=smeared, add_V0=add_V0)
        evals, evecs = eigh(H)
        be = np.clip(self.beta0 * evals, -500, 500)
        f = 1.0 / (np.exp(be) + 1.0)

        rho = np.zeros(self.N_shell)

        # Forward bonds
        V_i = evecs[self.lat.fwd_row]
        V_j = evecs[self.lat.fwd_col]
        G_fwd = (V_i * V_j) @ f

        if smeared:
            lapse = 1.0 + Phi_shell / self.cstar_sq
            Nbar_fwd = 0.5 * (lapse[self.lat.shell_of[self.lat.fwd_row]]
                            + lapse[self.lat.shell_of[self.lat.fwd_col]])
            hop_fwd = self.t0 * np.abs(Nbar_fwd)
        else:
            hop_fwd = self.t0
        np.add.at(rho, self.lat.fwd_shell, 2.0 * hop_fwd * G_fwd)

        # Intra-shell bonds
        V_intra_i = evecs[self.lat.intra_row]
        V_intra_j = evecs[self.lat.intra_col]
        G_intra = (V_intra_i * V_intra_j) @ f

        if smeared:
            hop_intra = self.t0 * np.abs(lapse[self.lat.intra_shell])
        else:
            hop_intra = self.t0
        np.add.at(rho, self.lat.intra_shell, 2.0 * hop_intra * G_intra)

        # On-site V0
        if add_V0:
            for n in range(min(self.n_core, self.N_shell)):
                idx = self.lat.shell_sites[n]
                V_core = evecs[idx]
                G_diag = (V_core * V_core) @ f
                rho[n] += self.V0 * G_diag.sum()

        return rho

    def _conductances_and_energy(self, Phi_shell):
        """Compute both MI conductances and rho_sigma in one pass."""
        lapse = 1.0 + Phi_shell / self.cstar_sq

        H = self._build_H(Phi_shell, smeared=True, add_V0=False)
        evals, evecs = eigh(H)
        be = np.clip(self.beta0 * evals, -500, 500)
        f = 1.0 / (np.exp(be) + 1.0)

        # -- Conductances (MI) --
        V_i = evecs[self.lat.fwd_row]
        V_j = evecs[self.lat.fwd_col]
        a = (V_i * V_i) @ f
        d = (V_j * V_j) @ f
        b = (V_i * V_j) @ f

        tr = a + d
        det = a * d - b * b
        disc = np.maximum(tr * tr - 4.0 * det, 0.0)
        lam1 = 0.5 * (tr + np.sqrt(disc))
        lam2 = 0.5 * (tr - np.sqrt(disc))
        mi = (binary_entropy(a) + binary_entropy(d)
              - binary_entropy(lam1) - binary_entropy(lam2))

        mi_ratio = mi / self.mi_bg_bonds
        kappa = np.zeros(self.N_shell - 1)
        np.add.at(kappa, self.lat.fwd_shell, self.t0**2 * mi_ratio)

        # -- Energy profile (smeared, no V0) --
        rho = np.zeros(self.N_shell)
        Nbar_fwd = 0.5 * (lapse[self.lat.shell_of[self.lat.fwd_row]]
                        + lapse[self.lat.shell_of[self.lat.fwd_col]])
        np.add.at(rho, self.lat.fwd_shell, 2.0 * self.t0 * np.abs(Nbar_fwd) * b)

        V_intra_i = evecs[self.lat.intra_row]
        V_intra_j = evecs[self.lat.intra_col]
        G_intra = (V_intra_i * V_intra_j) @ f
        np.add.at(rho, self.lat.intra_shell,
                  2.0 * self.t0 * np.abs(lapse[self.lat.intra_shell]) * G_intra)

        return kappa, rho

    def _conductances_only(self, Phi_shell):
        """MI conductances only (for proxy residual)."""
        H = self._build_H(Phi_shell, smeared=True, add_V0=False)
        evals, evecs = eigh(H)
        be = np.clip(self.beta0 * evals, -500, 500)
        f = 1.0 / (np.exp(be) + 1.0)

        V_i = evecs[self.lat.fwd_row]
        V_j = evecs[self.lat.fwd_col]
        a = (V_i * V_i) @ f
        d = (V_j * V_j) @ f
        b = (V_i * V_j) @ f

        tr = a + d
        det = a * d - b * b
        disc = np.maximum(tr * tr - 4.0 * det, 0.0)
        lam1 = 0.5 * (tr + np.sqrt(disc))
        lam2 = 0.5 * (tr - np.sqrt(disc))
        mi = (binary_entropy(a) + binary_entropy(d)
              - binary_entropy(lam1) - binary_entropy(lam2))

        mi_ratio = mi / self.mi_bg_bonds
        kappa = np.zeros(self.N_shell - 1)
        np.add.at(kappa, self.lat.fwd_shell, self.t0**2 * mi_ratio)
        return kappa

    def residual(self, Phi_shell):
        """Full self-consistent residual.

        Both rho_sigma and rho_tgt are evaluated in the same gravitational
        field (lapse-smeared hopping), so their difference is localized to
        the core.  This eliminates the Yukawa screening that arises when
        rho_tgt is computed at flat Phi=0.
        """
        self.n_call += 1
        kappa, rho_sigma = self._conductances_and_energy(Phi_shell)
        rho_tgt_sm = self._energy_profile(Phi_shell, smeared=True, add_V0=True)

        N = self.N_shell
        lhs = np.zeros(N)
        lhs[:N-1] += kappa * (Phi_shell[:N-1] - Phi_shell[1:N])
        lhs[1:N]  += kappa * (Phi_shell[1:N]  - Phi_shell[:N-1])

        F = lhs - (self.beta0 / self.cstar_sq) * (rho_sigma - rho_tgt_sm)
        F[N-1] = Phi_shell[N-1]
        return F

    def residual_proxy(self, Phi_shell):
        """Proxy: MI conductances, frozen source."""
        self.n_call += 1
        kappa = self._conductances_only(Phi_shell)

        N = self.N_shell
        lhs = np.zeros(N)
        lhs[:N-1] += kappa * (Phi_shell[:N-1] - Phi_shell[1:N])
        lhs[1:N]  += kappa * (Phi_shell[1:N]  - Phi_shell[:N-1])

        F = lhs - (self.beta0 / self.cstar_sq) * self.rho_src
        F[N-1] = Phi_shell[N-1]
        return F


# ============================================================================
# Solver
# ============================================================================

def solve_trf(model, mode='proxy', seed=None, tol=1e-7):
    """Solve via scipy least_squares (TRF)."""
    N = model.N_shell
    lb = np.full(N, -model.cstar_sq * 5.0)
    ub = np.full(N, model.cstar_sq * 0.5)
    lb[-1] = -1e-8; ub[-1] = 1e-8

    x0 = np.clip(seed if seed is not None else np.zeros(N), lb + 1e-10, ub - 1e-10)
    func = model.residual if mode == 'full' else model.residual_proxy
    model.n_call = 0

    result = least_squares(func, x0, method='trf', bounds=(lb, ub),
                          ftol=1e-12, xtol=1e-12, max_nfev=300)
    F = func(result.x)
    res = np.max(np.abs(F[:-1]))
    return result.x, res < tol, res, model.n_call


def solve_lm(model, mode='proxy', seed=None, tol=1e-7):
    """Solve via Levenberg-Marquardt (unbounded)."""
    N = model.N_shell
    x0 = seed if seed is not None else np.zeros(N)
    func = model.residual if mode == 'full' else model.residual_proxy
    model.n_call = 0

    result = least_squares(func, x0, method='lm', max_nfev=500)
    F = func(result.x)
    res = np.max(np.abs(F[:-1]))
    return result.x, res < tol, res, model.n_call


# ============================================================================
# Vacuum GM extraction (BC-corrected)
# ============================================================================

def extract_gm_vacuum(Phi, r):
    """BC-corrected GM extraction: GM = -Phi * r * R / (R - r)."""
    R = r[-1]
    with np.errstate(divide='ignore', invalid='ignore'):
        gm = -Phi * r * R / (R - r)
    return gm


def extract_rs_vacuum(Phi, r, n_core, cstar_sq):
    """Median-based rs extraction from vacuum-BC-corrected GM profile."""
    R = r[-1]
    N = len(Phi)
    i_lo = n_core + 1
    i_hi = max(i_lo + 2, min(N - 2, N * 3 // 4))
    if i_lo >= i_hi:
        return 0.0, 0.0

    r_slice = r[i_lo:i_hi]
    GM_est = -Phi[i_lo:i_hi] * r_slice * R / (R - r_slice)
    GM = np.median(GM_est)
    rs = 2.0 * GM / cstar_sq
    return rs, GM


def stagger_amplitude(Phi, n_inner=8):
    """Max |Phi[2k] - Phi[2k+1]| in inner shells."""
    n = min(n_inner, len(Phi) - 1)
    if n < 4:
        return 0.0
    pairs = n // 2
    return np.max(np.abs(Phi[:2*pairs:2] - Phi[1:2*pairs:2]))


# ============================================================================
# Main
# ============================================================================

def main():
    R_MAX = 10
    t0_hop = 1.0
    cstar_sq = 0.5
    n_core = 3
    beta0V0 = 0.003
    bt0_targets = [0.10, 0.50, 1.00, 2.11]

    print("=" * 80, flush=True)
    print(f"3D CUBIC LATTICE VACUUM SOLVER — CPU (R_max={R_MAX}, n_core={n_core})")
    print("=" * 80, flush=True)

    lat = CubicLattice3D(R_MAX)
    r = np.arange(1, lat.N_shell + 1, dtype=float)
    R = r[-1]

    # ==================================================================
    # PART 1: Temperature sweep — proxy equation (Newtonian asymptotics)
    # ==================================================================
    print(f"\n{'='*80}")
    print("PART 1: Proxy equation — temperature sweep")
    print(f"{'='*80}")
    print(f"{'bt0':>6s}  {'|F|':>10s}  {'min_N':>8s}  {'rs':>8s}  "
          f"{'GM':>10s}  {'stag':>10s}  {'nfev':>5s}  {'time':>6s}")
    print("-" * 80)

    proxy_results = {}
    for bt0 in bt0_targets:
        beta0 = bt0 / t0_hop
        V0 = beta0V0 / beta0

        model = ThreeDModelCPU(lat, t0=t0_hop, V0=V0, n_core=n_core,
                               beta0=beta0, cstar_sq=cstar_sq)

        t_s = time.time()
        Phi_p, conv_p, res_p, nfev_p = solve_trf(model, mode='proxy')
        dt = time.time() - t_s

        lapse = 1.0 + Phi_p / cstar_sq
        rs, GM = extract_rs_vacuum(Phi_p, r, n_core, cstar_sq)
        stag = stagger_amplitude(Phi_p)

        print(f"{bt0:6.2f}  {res_p:10.2e}  {lapse.min():8.5f}  {rs:8.4f}  "
              f"{GM:10.6f}  {stag:10.2e}  {nfev_p:5d}  {dt:5.1f}s", flush=True)

        proxy_results[bt0] = {
            'Phi': Phi_p, 'model': model, 'lapse': lapse,
            'rs': rs, 'GM': GM, 'res': res_p, 'stagger': stag,
        }

    print("-" * 80)

    # ==================================================================
    # PART 2: Full equation — does 3D suppress CDW?
    # ==================================================================
    print(f"\n{'='*80}")
    print("PART 2: Full equation — CDW test in 3D")
    print(f"{'='*80}")
    print(f"{'bt0':>6s}  {'|F_full|':>10s}  {'min_N':>8s}  {'rs':>8s}  "
          f"{'GM':>10s}  {'stag':>10s}  {'nfev':>5s}  {'time':>6s}")
    print("-" * 80)

    full_results = {}
    for bt0 in bt0_targets:
        beta0 = bt0 / t0_hop
        V0 = beta0V0 / beta0
        model = proxy_results[bt0]['model']
        Phi_seed = proxy_results[bt0]['Phi']

        # Try full equation from proxy seed (TRF)
        t_s = time.time()
        Phi_f, conv_f, res_f, nfev_f = solve_trf(
            model, mode='full', seed=Phi_seed)
        dt_trf = time.time() - t_s

        lapse_f = 1.0 + Phi_f / cstar_sq
        rs_f, GM_f = extract_rs_vacuum(Phi_f, r, n_core, cstar_sq)
        stag_f = stagger_amplitude(Phi_f)

        print(f"{bt0:6.2f}  {res_f:10.2e}  {lapse_f.min():8.5f}  {rs_f:8.4f}  "
              f"{GM_f:10.6f}  {stag_f:10.2e}  {nfev_f:5d}  {dt_trf:5.1f}s"
              f"  [TRF]", flush=True)

        # Also try LM from proxy seed
        t_s = time.time()
        Phi_lm, conv_lm, res_lm, nfev_lm = solve_lm(
            model, mode='full', seed=Phi_seed)
        dt_lm = time.time() - t_s

        lapse_lm = 1.0 + Phi_lm / cstar_sq
        rs_lm, GM_lm = extract_rs_vacuum(Phi_lm, r, n_core, cstar_sq)
        stag_lm = stagger_amplitude(Phi_lm)

        print(f"        {res_lm:10.2e}  {lapse_lm.min():8.5f}  {rs_lm:8.4f}  "
              f"{GM_lm:10.6f}  {stag_lm:10.2e}  {nfev_lm:5d}  {dt_lm:5.1f}s"
              f"  [LM]", flush=True)

        # Pick better
        if res_lm < res_f and stag_lm < 0.05:
            Phi_best, res_best = Phi_lm, res_lm
            method = "LM"
        else:
            Phi_best, res_best = Phi_f, res_f
            method = "TRF"

        lapse_best = 1.0 + Phi_best / cstar_sq
        rs_best, GM_best = extract_rs_vacuum(Phi_best, r, n_core, cstar_sq)
        stag_best = stagger_amplitude(Phi_best)

        full_results[bt0] = {
            'Phi': Phi_best, 'lapse': lapse_best,
            'rs': rs_best, 'GM': GM_best, 'res': res_best,
            'stagger': stag_best, 'method': method,
        }

    print("-" * 80)

    # ==================================================================
    # PART 3: Cross-residual (proxy ≈ full?)
    # ==================================================================
    print(f"\n{'='*80}")
    print("PART 3: Cross-residual — proxy vs full")
    print(f"{'='*80}")
    print(f"{'bt0':>6s}  {'|F_px(Phi_px)|':>14s}  {'|F_full(Phi_px)|':>16s}  "
          f"{'|F_full(Phi_f)|':>15s}  {'stag_full':>10s}")
    print("-" * 80)

    for bt0 in bt0_targets:
        model = proxy_results[bt0]['model']
        Phi_px = proxy_results[bt0]['Phi']
        Phi_f = full_results[bt0]['Phi']

        F_px = model.residual_proxy(Phi_px)
        F_full_at_px = model.residual(Phi_px)
        F_full_at_f = model.residual(Phi_f)

        print(f"{bt0:6.2f}  {np.max(np.abs(F_px[:-1])):14.2e}  "
              f"{np.max(np.abs(F_full_at_px[:-1])):16.2e}  "
              f"{np.max(np.abs(F_full_at_f[:-1])):15.2e}  "
              f"{full_results[bt0]['stagger']:10.2e}", flush=True)

    print("-" * 80)

    # ==================================================================
    # PART 4: GM profile and 1/r verification
    # ==================================================================
    print(f"\n{'='*80}")
    print("PART 4: GM profile flatness (1/r test)")
    print(f"{'='*80}")

    for bt0 in bt0_targets:
        Phi_px = proxy_results[bt0]['Phi']
        gm = extract_gm_vacuum(Phi_px, r)
        i_lo = n_core + 1
        i_hi = max(i_lo + 2, min(len(r) - 2, len(r) * 3 // 4))
        gm_mid = gm[i_lo:i_hi]
        gm_valid = gm_mid[np.isfinite(gm_mid)]
        GM = proxy_results[bt0]['GM']
        if len(gm_valid) > 0 and GM > 0:
            spread = (np.max(gm_valid) - np.min(gm_valid)) / GM
            print(f"  bt0={bt0:.2f}: GM={GM:.6f}, "
                  f"spread/GM={spread:.2e}, "
                  f"GM_profile=[{', '.join(f'{g:.5f}' for g in gm_valid)}]",
                  flush=True)
        else:
            print(f"  bt0={bt0:.2f}: GM={GM:.6f} (too few points for flatness)",
                  flush=True)

    # ==================================================================
    # PART 5: Summary plots
    # ==================================================================
    print(f"\n{'='*80}")
    print("PART 5: Plots")
    print(f"{'='*80}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # (a) Proxy potential at bt0=2.11 vs Newtonian
    bt_show = 2.11
    ax = axes[0, 0]
    pr = proxy_results[bt_show]
    phi_norm = pr['Phi'] / cstar_sq
    ax.plot(r, phi_norm, 'bo-', ms=5, lw=1.5, label='3D proxy')
    rs = pr['rs']
    if rs > 0:
        phi_newt = (rs / 2.0) * (1.0 / R - 1.0 / r)
        ax.plot(r, phi_newt, 'k--', lw=1.2, alpha=0.7,
                label=f'Newtonian+BC ($r_s={rs:.2f}$)')
    ax.set_xlabel('$r/a$')
    ax.set_ylabel(r'$\Phi / c_*^2$')
    ax.set_title(rf'(a) 3D Proxy potential at $\beta_0 t_0 = {bt_show}$')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (b) Lapse comparison
    ax = axes[0, 1]
    ax.plot(r, pr['lapse'], 'bo-', ms=5, lw=1.5, label='3D proxy')
    if rs > 0:
        lapse_schw = np.sqrt(np.maximum(1.0 - rs / r, 0.0))
        ax.plot(r, lapse_schw, 'k--', lw=1.2, alpha=0.7,
                label=rf'$\sqrt{{1-r_s/r}}$')
    ax.set_xlabel('$r/a$')
    ax.set_ylabel('$N(r)$')
    ax.set_title(rf'(b) 3D Lapse at $\beta_0 t_0 = {bt_show}$')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # (c) GM vs temperature
    ax = axes[0, 2]
    bts = sorted(proxy_results.keys())
    gm_px = [proxy_results[b]['GM'] for b in bts]
    gm_f = [full_results[b]['GM'] for b in bts]
    ax.plot(bts, gm_px, 'bo-', ms=6, lw=1.5, label='Proxy (vacuum)')
    ax.plot(bts, gm_f, 'r^--', ms=5, lw=1, label='Full')
    ax.set_xlabel(r'$\beta_0 t_0$')
    ax.set_ylabel('$GM$')
    ax.set_title('(c) Gravitational mass: proxy vs full')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (d) BC-corrected GM profile at bt0=2.11 (flatness = 1/r)
    ax = axes[1, 0]
    gm = extract_gm_vacuum(pr['Phi'], r)
    mask = np.isfinite(gm) & (np.arange(len(r)) >= n_core) & (np.arange(len(r)) < len(r) - 1)
    if mask.any():
        ax.plot(r[mask], gm[mask], 'bo-', ms=5, lw=1.5)
        ax.axhline(pr['GM'], color='k', ls=':', lw=1.2, alpha=0.7,
                   label=f'$GM = {pr["GM"]:.5f}$')
    ax.set_xlabel('$r/a$')
    ax.set_ylabel(r'$-\Phi \cdot rR/(R-r)$')
    ax.set_title(rf'(d) GM extraction at $\beta_0 t_0 = {bt_show}$ (flat $=$ 1/r)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (e) Full vs proxy at bt0=2.11 — stagger comparison
    ax = axes[1, 1]
    fr = full_results[bt_show]
    phi_px = pr['Phi'] / cstar_sq
    phi_f = fr['Phi'] / cstar_sq
    ax.plot(r, phi_px, 'bo-', ms=5, lw=1.5, label=f'Proxy (stag={pr["stagger"]:.1e})')
    ax.plot(r, phi_f, 'r^-', ms=4, lw=1, label=f'Full (stag={fr["stagger"]:.1e})')
    ax.set_xlabel('$r/a$')
    ax.set_ylabel(r'$\Phi / c_*^2$')
    ax.set_title(rf'(e) Proxy vs Full at $\beta_0 t_0 = {bt_show}$')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (f) Cross-residual across temperatures
    ax = axes[1, 2]
    px_res = []
    cr_res = []
    for bt0 in bts:
        model = proxy_results[bt0]['model']
        Phi_px = proxy_results[bt0]['Phi']
        px_res.append(np.max(np.abs(model.residual_proxy(Phi_px)[:-1])))
        cr_res.append(np.max(np.abs(model.residual(Phi_px)[:-1])))
    ax.semilogy(bts, px_res, 'bo-', ms=5, lw=1.5,
                label=r'$|F_{\rm proxy}(\Phi)|$')
    ax.semilogy(bts, cr_res, 'r^--', ms=5, lw=1.5,
                label=r'$|F_{\rm full}(\Phi_{\rm proxy})|$')
    ax.set_xlabel(r'$\beta_0 t_0$')
    ax.set_ylabel('Residual')
    ax.set_title(r'(f) Cross-residual')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'3D cubic lattice vacuum solver (R={R_MAX}, n_core={n_core})',
                 fontsize=14, y=1.01)
    plt.tight_layout()

    path = os.path.join(FIGDIR, "vacuum_3d_selfconsistent_summary.pdf")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.savefig(path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close()

    # ==================================================================
    # Summary
    # ==================================================================
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"\n3D cubic lattice (R={R_MAX}, {lat.N_sites} sites, "
          f"{lat.N_shell} shells)")
    print(f"\nProxy equation (Newtonian asymptotics):")
    for bt0 in bts:
        pr = proxy_results[bt0]
        print(f"  bt0={bt0:.2f}: rs={pr['rs']:.4f}, GM={pr['GM']:.6f}, "
              f"min_N={pr['lapse'].min():.5f}, stag={pr['stagger']:.1e}")
    print(f"\nFull equation (self-consistent energy feedback):")
    for bt0 in bts:
        fr = full_results[bt0]
        print(f"  bt0={bt0:.2f}: rs={fr['rs']:.4f}, GM={fr['GM']:.6f}, "
              f"|F|={fr['res']:.1e}, stag={fr['stagger']:.1e} [{fr['method']}]")
    print(f"\nPlots saved to: {FIGDIR}")


if __name__ == "__main__":
    main()
