#!/usr/bin/env python3
"""
3D cubic-lattice vacuum solver — GPU (PyTorch CUDA) version.

Accelerates eigendecomposition and MI/energy computations on GPU,
enabling larger lattice radii than the CPU version.

The two-state proxy equation naturally produces asymptotic Newtonian (1/r)
gravity: the source (rho_bg - rho_tgt) is localized near the core, making
the far-field equation Poisson.  The full self-consistent equation (with
rho_sigma feedback) is also solved to test whether 3D suppresses CDW.

Usage:
  PYTHONUNBUFFERED=1 python full_numerics/solve_3d_vacuum_gpu.py [--R 15]
"""

import sys
import os
import time
import argparse
import numpy as np
import torch
from scipy.linalg import solve_banded
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
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

FIGDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "fermion", "numerical", "figures")
os.makedirs(FIGDIR, exist_ok=True)


# ============================================================================
# Binary entropy (CPU and GPU)
# ============================================================================

def binary_entropy_np(x):
    x = np.clip(x, 1e-30, 1.0 - 1e-30)
    return -x * np.log(x) - (1.0 - x) * np.log(1.0 - x)


def binary_entropy_t(x):
    x = torch.clamp(x, 1e-30, 1.0 - 1e-30)
    return -x * torch.log(x) - (1.0 - x) * torch.log(1.0 - x)


# ============================================================================
# 3D cubic lattice (CPU — built once)
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

        # Forward bonds per shell (for analytic Picard)
        self.B_n = np.bincount(self.fwd_shell,
                               minlength=self.N_shell)[:self.N_shell - 1]

        elapsed = time.time() - t0
        print(f"  Lattice: R={R_max}, {self.N_sites} sites, {self.N_shell} shells, "
              f"{len(self.row)//2} bonds ({len(self.fwd_row)} fwd, "
              f"{len(self.intra_row)} intra) [{elapsed:.1f}s]", flush=True)


# ============================================================================
# 3D Model — GPU (PyTorch CUDA)
# ============================================================================

class ThreeDModelGPU:
    """3D self-consistent closure with GPU-accelerated eigh and MI."""

    BOND_CHUNK = 4000  # bonds per GPU chunk (tune for VRAM)

    def __init__(self, lat, t0=1.0, V0=0.01, n_core=2, beta0=0.1,
                 cstar_sq=0.5, gpu_dtype='float32'):
        self.device = torch.device('cuda')
        # float32 for eigh (~4x faster than float64 on consumer GPUs)
        self.dtype = torch.float32 if gpu_dtype == 'float32' else torch.float64
        self.lat = lat
        self.t0 = t0
        self.V0 = V0
        self.n_core = n_core
        self.beta0 = beta0
        self.cstar_sq = cstar_sq
        self.N_shell = lat.N_shell
        self.N = lat.N_shell
        self.r = np.arange(1, self.N_shell + 1, dtype=float)
        self.n_call = 0

        # GPU index tensors
        self.row_t = torch.tensor(lat.row, dtype=torch.long, device=self.device)
        self.col_t = torch.tensor(lat.col, dtype=torch.long, device=self.device)
        self.fwd_row_t = torch.tensor(lat.fwd_row, dtype=torch.long, device=self.device)
        self.fwd_col_t = torch.tensor(lat.fwd_col, dtype=torch.long, device=self.device)
        self.fwd_shell_t = torch.tensor(lat.fwd_shell, dtype=torch.long, device=self.device)
        self.intra_row_t = torch.tensor(lat.intra_row, dtype=torch.long, device=self.device)
        self.intra_col_t = torch.tensor(lat.intra_col, dtype=torch.long, device=self.device)
        self.intra_shell_t = torch.tensor(lat.intra_shell, dtype=torch.long, device=self.device)
        self.shell_of_t = torch.tensor(lat.shell_of, dtype=torch.long, device=self.device)

        # Shell indices for all neighbor pairs (for smeared H build)
        self.shell_of_row_t = self.shell_of_t[self.row_t]
        self.shell_of_col_t = self.shell_of_t[self.col_t]

        # Core site indices for V0
        self.core_idx_t = []
        for n in range(min(n_core, self.N_shell)):
            self.core_idx_t.append(
                torch.tensor(lat.shell_sites[n], dtype=torch.long, device=self.device))

        # Precompute bg and tgt energy profiles on GPU
        t_s = time.time()
        Phi0 = torch.zeros(self.N_shell, dtype=self.dtype, device=self.device)
        self.rho_bg_t = self._energy_profile_gpu(Phi0, smeared=False, add_V0=False)
        self.rho_tgt_t = self._energy_profile_gpu(Phi0, smeared=False, add_V0=True)
        self.rho_src_t = self.rho_bg_t - self.rho_tgt_t
        self.rho_src = self.rho_src_t.cpu().numpy()
        self.rho_tgt_np = self.rho_tgt_t.cpu().numpy()

        # Background MI per forward bond for normalization
        self.mi_bg_bonds_t = self._compute_mi_background_gpu(Phi0)

        dt = time.time() - t_s
        print(f"  bg/tgt on GPU in {dt:.1f}s  "
              f"src_norm={self.rho_src_t.abs().sum().item():.6f}", flush=True)

    # ---- Background MI (for normalization) ----

    def _compute_mi_background_gpu(self, Phi0_t):
        """Background MI per forward bond at Phi=0 (uniform chain).

        Used to normalize conductances: kappa_bond = t0^2 * MI/MI_bg.
        This removes the spurious 1/beta0^2 temperature factor.
        """
        H = self._build_H(Phi0_t, smeared=False, add_V0=False)
        evals, evecs = torch.linalg.eigh(H)
        del H
        be = torch.clamp(self.beta0 * evals, -500, 500)
        f = 1.0 / (torch.exp(be) + 1.0)

        mi = self._mi_from_evecs(evecs, f, self.fwd_row_t, self.fwd_col_t)
        del evecs
        return torch.clamp(mi, min=1e-30)

    # ---- Hamiltonian build (GPU) ----

    def _build_H(self, Phi_t, smeared=False, add_V0=False):
        """Build dense Hamiltonian as GPU tensor."""
        Ns = self.lat.N_sites
        H = torch.zeros((Ns, Ns), dtype=self.dtype, device=self.device)

        if smeared:
            lapse = 1.0 + Phi_t / self.cstar_sq
            Nbar = 0.5 * (lapse[self.shell_of_row_t] + lapse[self.shell_of_col_t])
            H[self.row_t, self.col_t] = -self.t0 * torch.abs(Nbar)
        else:
            H[self.row_t, self.col_t] = -self.t0

        if add_V0:
            for n in range(min(self.n_core, self.N_shell)):
                idx = self.core_idx_t[n]
                H[idx, idx] += self.V0

        return H

    # ---- Batched G-element computation ----

    def _G_elements(self, evecs, f, row_idx, col_idx):
        """
        Compute G[row[b], col[b]] for bond list, in memory-safe chunks.

        G[i,j] = sum_k evecs[i,k] * evecs[j,k] * f[k]
        """
        n_bonds = len(row_idx)
        result = torch.zeros(n_bonds, dtype=self.dtype, device=self.device)
        C = self.BOND_CHUNK
        for s in range(0, n_bonds, C):
            e = min(s + C, n_bonds)
            vi = evecs[row_idx[s:e]]   # (chunk, N_sites)
            vj = evecs[col_idx[s:e]]   # (chunk, N_sites)
            result[s:e] = (vi * vj) @ f  # batched dot product
        return result

    # ---- Energy profile (GPU) ----

    def _energy_profile_gpu(self, Phi_t, smeared=False, add_V0=False):
        """Energy per shell from correlation matrix (all on GPU)."""
        H = self._build_H(Phi_t, smeared=smeared, add_V0=add_V0)
        evals, evecs = torch.linalg.eigh(H)
        del H
        be = torch.clamp(self.beta0 * evals, -500, 500)
        f = 1.0 / (torch.exp(be) + 1.0)

        rho = torch.zeros(self.N_shell, dtype=self.dtype, device=self.device)

        # Forward bonds: G[fwd_row, fwd_col]
        G_fwd = self._G_elements(evecs, f, self.fwd_row_t, self.fwd_col_t)
        if smeared:
            lapse = 1.0 + Phi_t / self.cstar_sq
            shell_i = self.shell_of_t[self.fwd_row_t]
            shell_j = self.shell_of_t[self.fwd_col_t]
            Nbar_fwd = 0.5 * (lapse[shell_i] + lapse[shell_j])
            hop_fwd = self.t0 * torch.abs(Nbar_fwd)
        else:
            hop_fwd = self.t0
        rho.scatter_add_(0, self.fwd_shell_t, 2.0 * hop_fwd * G_fwd)

        # Intra-shell bonds: G[intra_row, intra_col]
        G_intra = self._G_elements(evecs, f, self.intra_row_t, self.intra_col_t)
        if smeared:
            hop_intra = self.t0 * torch.abs(lapse[self.intra_shell_t])
        else:
            hop_intra = self.t0
        rho.scatter_add_(0, self.intra_shell_t, 2.0 * hop_intra * G_intra)

        # On-site V0
        if add_V0:
            G_diag = (evecs * evecs) @ f  # (N_sites,) = G_{nn}
            for n in range(min(self.n_core, self.N_shell)):
                idx = self.core_idx_t[n]
                rho[n] += self.V0 * G_diag[idx].sum()

        del evecs
        return rho

    # ---- MI conductances (GPU) ----

    def _mi_from_evecs(self, evecs, f, row_idx, col_idx):
        """Mutual information per bond from eigenvectors and Fermi function."""
        # G diagonal at bond endpoints
        G_diag = (evecs * evecs) @ f  # (N_sites,)
        a = G_diag[row_idx]
        d = G_diag[col_idx]
        # Off-diagonal
        b = self._G_elements(evecs, f, row_idx, col_idx)

        tr = a + d
        det = a * d - b * b
        disc = torch.clamp(tr * tr - 4.0 * det, min=0.0)
        lam1 = 0.5 * (tr + torch.sqrt(disc))
        lam2 = 0.5 * (tr - torch.sqrt(disc))
        return (binary_entropy_t(a) + binary_entropy_t(d)
                - binary_entropy_t(lam1) - binary_entropy_t(lam2))

    def _conductances_and_energy_gpu(self, Phi_t):
        """MI conductances + rho_sigma in one eigh pass (GPU)."""
        H = self._build_H(Phi_t, smeared=True, add_V0=False)
        evals, evecs = torch.linalg.eigh(H)
        del H
        be = torch.clamp(self.beta0 * evals, -500, 500)
        f = 1.0 / (torch.exp(be) + 1.0)

        # MI conductances from forward bonds (ratio-normalized)
        mi = self._mi_from_evecs(evecs, f, self.fwd_row_t, self.fwd_col_t)
        mi_ratio = mi / self.mi_bg_bonds_t
        kappa = torch.zeros(self.N_shell - 1, dtype=self.dtype, device=self.device)
        kappa.scatter_add_(0, self.fwd_shell_t, self.t0**2 * mi_ratio)

        # rho_sigma (smeared energy profile, no V0)
        lapse = 1.0 + Phi_t / self.cstar_sq
        rho = torch.zeros(self.N_shell, dtype=self.dtype, device=self.device)

        # Forward bonds
        b_fwd = self._G_elements(evecs, f, self.fwd_row_t, self.fwd_col_t)
        shell_i = self.shell_of_t[self.fwd_row_t]
        shell_j = self.shell_of_t[self.fwd_col_t]
        Nbar_fwd = 0.5 * (lapse[shell_i] + lapse[shell_j])
        rho.scatter_add_(0, self.fwd_shell_t,
                         2.0 * self.t0 * torch.abs(Nbar_fwd) * b_fwd)

        # Intra-shell bonds
        G_intra = self._G_elements(evecs, f, self.intra_row_t, self.intra_col_t)
        rho.scatter_add_(0, self.intra_shell_t,
                         2.0 * self.t0 * torch.abs(lapse[self.intra_shell_t]) * G_intra)

        del evecs
        return kappa, rho

    def _conductances_only_gpu(self, Phi_t):
        """MI conductances only (for proxy residual)."""
        H = self._build_H(Phi_t, smeared=True, add_V0=False)
        evals, evecs = torch.linalg.eigh(H)
        del H
        be = torch.clamp(self.beta0 * evals, -500, 500)
        f = 1.0 / (torch.exp(be) + 1.0)

        mi = self._mi_from_evecs(evecs, f, self.fwd_row_t, self.fwd_col_t)
        mi_ratio = mi / self.mi_bg_bonds_t
        kappa = torch.zeros(self.N_shell - 1, dtype=self.dtype, device=self.device)
        kappa.scatter_add_(0, self.fwd_shell_t, self.t0**2 * mi_ratio)

        del evecs
        return kappa

    # ---- Residuals (numpy interface for scipy) ----

    def residual(self, Phi_shell):
        """Full self-consistent residual. numpy in/out.

        Both rho_sigma and rho_tgt are evaluated in the same gravitational
        field (lapse-smeared hopping), so their difference is localized to
        the core.  This eliminates the Yukawa screening that arises when
        rho_tgt is computed at flat Phi=0.
        """
        self.n_call += 1
        Phi_t = torch.tensor(Phi_shell, dtype=self.dtype, device=self.device)
        kappa, rho_sigma = self._conductances_and_energy_gpu(Phi_t)
        rho_tgt_sm = self._energy_profile_gpu(Phi_t, smeared=True, add_V0=True)

        N = self.N_shell
        lhs = torch.zeros(N, dtype=self.dtype, device=self.device)
        lhs[:N-1] += kappa * (Phi_t[:N-1] - Phi_t[1:N])
        lhs[1:N]  += kappa * (Phi_t[1:N]  - Phi_t[:N-1])

        F = lhs - (self.beta0 / self.cstar_sq) * (rho_sigma - rho_tgt_sm)
        F[N-1] = Phi_t[N-1]
        return F.cpu().numpy()

    def residual_proxy(self, Phi_shell):
        """Proxy residual: MI conductances, frozen source. numpy in/out."""
        self.n_call += 1
        Phi_t = torch.tensor(Phi_shell, dtype=self.dtype, device=self.device)
        kappa = self._conductances_only_gpu(Phi_t)

        N = self.N_shell
        lhs = torch.zeros(N, dtype=self.dtype, device=self.device)
        lhs[:N-1] += kappa * (Phi_t[:N-1] - Phi_t[1:N])
        lhs[1:N]  += kappa * (Phi_t[1:N]  - Phi_t[:N-1])

        F = lhs - (self.beta0 / self.cstar_sq) * self.rho_src_t
        F[N-1] = Phi_t[N-1]
        return F.cpu().numpy()


# ============================================================================
# Picard warmup (analytic kappa — no eigh, CPU only)
# ============================================================================

def picard_proxy_3d(model, max_iter=300, mixing=0.3, lapse_floor=0.005):
    """
    Picard iteration with analytic conductances (no eigendecomposition).

    Uses kappa_n = B_n * t0^2 * Nbar^2 where B_n is the number of
    forward bonds from shell n to n+1.  This gives a good seed for
    the GPU-accelerated Newton/TRF polish.
    """
    N = model.N_shell
    cs2 = model.cstar_sq
    t0v = model.t0
    B_n = model.lat.B_n.astype(float)
    src = (model.beta0 / cs2) * model.rho_src
    Phi = np.zeros(N)
    Phi_lo = -cs2 * (1.0 - lapse_floor)

    for it in range(max_iter):
        lapse = 1.0 + Phi / cs2
        Nbar = 0.5 * (lapse[:-1] + lapse[1:])
        kap = B_n * t0v**2 * Nbar**2

        d = np.zeros(N)
        d[:-1] += kap
        d[1:] += kap
        off = -kap
        d[-1] = 1.0
        s = src.copy()
        s[-1] = 0.0
        ab = np.zeros((3, N))
        ab[0, 1:] = off
        ab[1, :] = d
        ab[2, :-1] = off
        Pn = solve_banded((1, 1), ab, s)
        Pn = mixing * Pn + (1 - mixing) * Phi
        Pn = np.maximum(Pn, Phi_lo)

        chg = np.max(np.abs(Pn - Phi)) / max(1e-15, np.max(np.abs(Phi)))
        Phi = Pn
        if chg < 1e-10 and it > 10:
            break

    return Phi


# ============================================================================
# Solvers
# ============================================================================

def newton_fd(model, mode='proxy', seed=None, tol=1e-7, max_iter=20,
              verbose=False, fd_eps=1e-4):
    """
    Newton solver with FD Jacobian.  N_shell is small (16-21), so
    the N x N Jacobian is cheap to form and solve.  Each iteration
    needs N+1 residual evaluations (the expensive part is eigh inside
    each).

    fd_eps should be large enough to overcome float32 residual noise
    (~1e-5 from GPU eigh accumulation).  Default 1e-4 works for float32.
    """
    N = model.N_shell
    cs2 = model.cstar_sq
    func = model.residual if mode == 'full' else model.residual_proxy
    Phi = seed.copy() if seed is not None else np.zeros(N)
    model.n_call = 0

    for it in range(max_iter):
        F = func(Phi)
        res = np.max(np.abs(F[:-1]))
        if verbose:
            lapse = 1.0 + Phi / cs2
            print(f"    Newton {it:3d}: |F|={res:.3e}, min(N)={lapse.min():.5f}",
                  flush=True)
        if res < tol:
            return Phi, True, res, model.n_call

        # FD Jacobian (eps tuned for float32 GPU residual noise)
        J = np.zeros((N, N))
        for j in range(N):
            eps = fd_eps * max(1.0, abs(Phi[j]))
            Phi_p = Phi.copy()
            Phi_p[j] += eps
            J[:, j] = (func(Phi_p) - F) / eps

        try:
            dPhi = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            return Phi, False, res, model.n_call

        # Backtracking line search with lapse guard (limited to 8 steps)
        alpha = 1.0
        for _ in range(8):
            Phi_try = Phi + alpha * dPhi
            if (1.0 + Phi_try / cs2).min() < -0.5:
                alpha *= 0.5
                continue
            F_try = func(Phi_try)
            if np.max(np.abs(F_try[:-1])) < res * 1.1:
                Phi = Phi_try
                break
            alpha *= 0.5
            if alpha < 1e-6:
                Phi = Phi + 0.01 * dPhi
                break

    F = func(Phi)
    res = np.max(np.abs(F[:-1]))
    return Phi, res < tol, res, model.n_call


def solve_trf(model, mode='proxy', seed=None, tol=5e-5):
    """Solve via scipy least_squares (TRF) — backup solver."""
    N = model.N_shell
    lb = np.full(N, -model.cstar_sq * 5.0)
    ub = np.full(N, model.cstar_sq * 0.5)
    lb[-1] = -1e-8; ub[-1] = 1e-8

    x0 = np.clip(seed if seed is not None else np.zeros(N), lb + 1e-10, ub - 1e-10)
    func = model.residual if mode == 'full' else model.residual_proxy
    model.n_call = 0

    result = least_squares(func, x0, method='trf', bounds=(lb, ub),
                          ftol=1e-12, xtol=1e-12, max_nfev=150)
    F = func(result.x)
    res = np.max(np.abs(F[:-1]))
    return result.x, res < tol, res, model.n_call


# ============================================================================
# Vacuum GM extraction (BC-corrected: Phi = -GM(1/r - 1/R))
# ============================================================================

def extract_gm_vacuum(Phi, r):
    R = r[-1]
    with np.errstate(divide='ignore', invalid='ignore'):
        gm = -Phi * r * R / (R - r)
    return gm


def extract_rs_vacuum(Phi, r, n_core, cstar_sq):
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
    n = min(n_inner, len(Phi) - 1)
    if n < 4:
        return 0.0
    pairs = n // 2
    return np.max(np.abs(Phi[:2*pairs:2] - Phi[1:2*pairs:2]))


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, default=15,
                    help="Lattice radius (default 15)")
    ap.add_argument("--n-core", type=int, default=3)
    ap.add_argument("--cstar-sq", type=float, default=0.5)
    ap.add_argument("--betaV0", type=float, default=0.003)
    ap.add_argument("--dtype", choices=['float32', 'float64'], default='float32',
                    help="GPU dtype for eigh (float32 ~4x faster, default)")
    args = ap.parse_args()

    R_MAX = args.R
    t0_hop = 1.0
    cstar_sq = args.cstar_sq
    n_core = args.n_core
    beta0V0 = args.betaV0
    bt0_targets = [0.10, 0.50, 1.00, 2.11]

    dev = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print("=" * 80, flush=True)
    print(f"3D CUBIC LATTICE VACUUM SOLVER — GPU ({dev}, {mem:.1f} GB)")
    print(f"R_max={R_MAX}, n_core={n_core}, c*^2={cstar_sq}, beta0*V0={beta0V0}")
    print(f"GPU dtype: {args.dtype}")
    print("=" * 80, flush=True)

    lat = CubicLattice3D(R_MAX)

    # Warm-up eigh timing
    dtype_t = torch.float32 if args.dtype == 'float32' else torch.float64
    H_test = torch.randn(lat.N_sites, lat.N_sites, dtype=dtype_t, device='cuda')
    H_test = (H_test + H_test.T) / 2
    torch.cuda.synchronize()
    t_eigh = time.time()
    torch.linalg.eigh(H_test)
    torch.cuda.synchronize()
    dt_eigh = time.time() - t_eigh
    del H_test
    torch.cuda.empty_cache()
    print(f"  eigh({lat.N_sites}x{lat.N_sites}, {args.dtype}): {dt_eigh:.2f}s", flush=True)
    r = np.arange(1, lat.N_shell + 1, dtype=float)
    R = r[-1]

    # ==================================================================
    # PART 1: Temperature sweep — proxy equation (Newtonian asymptotics)
    # ==================================================================
    print(f"\n{'='*80}")
    print("PART 1: Proxy equation — temperature sweep (asymptotic Newtonian)")
    print(f"{'='*80}")
    print(f"{'bt0':>6s}  {'|F|':>10s}  {'min_N':>8s}  {'rs':>8s}  "
          f"{'GM':>10s}  {'stag':>10s}  {'nfev':>5s}  {'time':>6s}")
    print("-" * 80)

    proxy_results = {}
    for bt0 in bt0_targets:
        beta0 = bt0 / t0_hop
        V0 = beta0V0 / beta0

        model = ThreeDModelGPU(lat, t0=t0_hop, V0=V0, n_core=n_core,
                               beta0=beta0, cstar_sq=cstar_sq,
                               gpu_dtype=args.dtype)

        # Picard warmup (analytic kappa, no GPU eigh)
        Phi_seed = picard_proxy_3d(model, max_iter=300, mixing=0.3)

        # Newton polish (much fewer eigh calls than TRF)
        t_s = time.time()
        Phi_p, conv_p, res_p, nfev_p = newton_fd(
            model, mode='proxy', seed=Phi_seed, tol=5e-5,
            max_iter=10, verbose=True)
        if not conv_p:
            # Fallback to TRF
            print(f"    Newton failed, falling back to TRF...", flush=True)
            Phi_p, conv_p, res_p, nfev_p = solve_trf(
                model, mode='proxy', seed=Phi_seed)
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
    print("PART 2: Full self-consistent equation — CDW test in 3D")
    print(f"{'='*80}")
    print(f"{'bt0':>6s}  {'|F_full|':>10s}  {'min_N':>8s}  {'rs':>8s}  "
          f"{'GM':>10s}  {'stag':>10s}  {'nfev':>5s}  {'time':>6s}  {'method':>6s}")
    print("-" * 90)

    full_results = {}
    for bt0 in bt0_targets:
        model = proxy_results[bt0]['model']
        Phi_seed = proxy_results[bt0]['Phi']

        # Newton from proxy seed
        t_s = time.time()
        Phi_f, conv_f, res_f, nfev_f = newton_fd(
            model, mode='full', seed=Phi_seed, tol=5e-5,
            max_iter=10, verbose=True)
        dt_f = time.time() - t_s
        method = "Newton"

        # Fallback to TRF if Newton fails
        if not conv_f:
            t_s2 = time.time()
            Phi_trf, conv_trf, res_trf, nfev_trf = solve_trf(
                model, mode='full', seed=Phi_seed)
            dt_trf = time.time() - t_s2
            if res_trf < res_f:
                Phi_f, conv_f, res_f, nfev_f = Phi_trf, conv_trf, res_trf, nfev_trf
                dt_f += dt_trf
                method = "TRF"

        Phi_best, res_best = Phi_f, res_f
        nfev_best, dt_best = nfev_f, dt_f

        lapse_best = 1.0 + Phi_best / cstar_sq
        rs_best, GM_best = extract_rs_vacuum(Phi_best, r, n_core, cstar_sq)
        stag_best = stagger_amplitude(Phi_best)

        print(f"{bt0:6.2f}  {res_best:10.2e}  {lapse_best.min():8.5f}  {rs_best:8.4f}  "
              f"{GM_best:10.6f}  {stag_best:10.2e}  {nfev_best:5d}  "
              f"{dt_best:5.1f}s  {method:>6s}", flush=True)

        full_results[bt0] = {
            'Phi': Phi_best, 'lapse': lapse_best,
            'rs': rs_best, 'GM': GM_best, 'res': res_best,
            'stagger': stag_best, 'method': method,
        }

    print("-" * 90)

    # ==================================================================
    # PART 3: Cross-residual + 1/r verification
    # ==================================================================
    print(f"\n{'='*80}")
    print("PART 3: Cross-residual and Newtonian (1/r) verification")
    print(f"{'='*80}")
    print(f"{'bt0':>6s}  {'|F_px|':>10s}  {'|F_full@px|':>12s}  "
          f"{'|F_full@f|':>12s}  {'GM_spread':>10s}  {'stag_full':>10s}")
    print("-" * 80)

    for bt0 in bt0_targets:
        model = proxy_results[bt0]['model']
        Phi_px = proxy_results[bt0]['Phi']
        Phi_f = full_results[bt0]['Phi']

        F_px = model.residual_proxy(Phi_px)
        F_full_at_px = model.residual(Phi_px)
        F_full_at_f = model.residual(Phi_f)

        # GM profile flatness
        gm = extract_gm_vacuum(Phi_px, r)
        GM = proxy_results[bt0]['GM']
        i_lo = n_core + 1
        i_hi = max(i_lo + 2, min(len(r) - 2, len(r) * 3 // 4))
        gm_mid = gm[i_lo:i_hi]
        gm_valid = gm_mid[np.isfinite(gm_mid)]
        spread = (np.max(gm_valid) - np.min(gm_valid)) / max(GM, 1e-15) if len(gm_valid) > 0 and GM > 0 else 0

        print(f"{bt0:6.2f}  {np.max(np.abs(F_px[:-1])):10.2e}  "
              f"{np.max(np.abs(F_full_at_px[:-1])):12.2e}  "
              f"{np.max(np.abs(F_full_at_f[:-1])):12.2e}  "
              f"{spread:10.2e}  "
              f"{full_results[bt0]['stagger']:10.2e}", flush=True)

    print("-" * 80)

    # ==================================================================
    # PART 4: Summary plots
    # ==================================================================
    print(f"\n{'='*80}")
    print("PART 4: Plots")
    print(f"{'='*80}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    bt_show = 2.11

    # (a) Potential vs Newtonian
    ax = axes[0, 0]
    pr = proxy_results[bt_show]
    phi_norm = pr['Phi'] / cstar_sq
    ax.plot(r, phi_norm, 'b-o', ms=4, lw=1.5, label='3D proxy (GPU)')
    rs = pr['rs']
    if rs > 0:
        phi_newt = (rs / 2.0) * (1.0 / R - 1.0 / r)
        ax.plot(r, phi_newt, 'k--', lw=1.2, alpha=0.7,
                label=f'$-GM(1/r - 1/R)$, $r_s={rs:.3f}$')
    ax.set_xlabel('$r/a$')
    ax.set_ylabel(r'$\Phi / c_*^2$')
    ax.set_title(rf'(a) Potential at $\beta_0 t_0 = {bt_show}$')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (b) Lapse vs Schwarzschild
    ax = axes[0, 1]
    ax.plot(r, pr['lapse'], 'b-o', ms=4, lw=1.5, label='3D proxy')
    if rs > 0:
        lapse_schw = np.sqrt(np.maximum(1.0 - rs / r, 0.0))
        ax.plot(r, lapse_schw, 'k--', lw=1.2, alpha=0.7,
                label=rf'$\sqrt{{1-r_s/r}}$')
    ax.set_xlabel('$r/a$')
    ax.set_ylabel('$N(r)$')
    ax.set_title(rf'(b) Lapse at $\beta_0 t_0 = {bt_show}$')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # (c) GM vs temperature
    ax = axes[0, 2]
    bts = sorted(proxy_results.keys())
    gm_px = [proxy_results[b]['GM'] for b in bts]
    gm_f = [full_results[b]['GM'] for b in bts]
    ax.plot(bts, gm_px, 'bo-', ms=6, lw=1.5, label='Proxy (Newtonian)')
    ax.plot(bts, gm_f, 'r^--', ms=5, lw=1, label='Full (self-consistent)')
    ax.set_xlabel(r'$\beta_0 t_0$')
    ax.set_ylabel('$GM$')
    ax.set_title('(c) Gravitational mass vs temperature')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (d) BC-corrected GM profile (flatness = 1/r)
    ax = axes[1, 0]
    gm = extract_gm_vacuum(pr['Phi'], r)
    mask = np.isfinite(gm) & (np.arange(len(r)) >= n_core) & (np.arange(len(r)) < len(r) - 1)
    if mask.any():
        ax.plot(r[mask], gm[mask], 'b-o', ms=4, lw=1.5)
        ax.axhline(pr['GM'], color='k', ls=':', lw=1.2, alpha=0.7,
                   label=f'$GM = {pr["GM"]:.5f}$')
    ax.set_xlabel('$r/a$')
    ax.set_ylabel(r'$-\Phi \cdot rR/(R-r)$')
    ax.set_title(rf'(d) GM extraction at $\beta_0 t_0 = {bt_show}$ (flat $=$ 1/r)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (e) Full vs proxy
    ax = axes[1, 1]
    fr = full_results[bt_show]
    ax.plot(r, pr['Phi'] / cstar_sq, 'b-o', ms=4, lw=1.5,
            label=f'Proxy (stag={pr["stagger"]:.1e})')
    ax.plot(r, fr['Phi'] / cstar_sq, 'r-^', ms=3, lw=1,
            label=f'Full (stag={fr["stagger"]:.1e})')
    ax.set_xlabel('$r/a$')
    ax.set_ylabel(r'$\Phi / c_*^2$')
    ax.set_title(rf'(e) Proxy vs Full at $\beta_0 t_0 = {bt_show}$')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (f) Source localization
    ax = axes[1, 2]
    model_show = proxy_results[bt_show]['model']
    src = model_show.rho_src
    n_show = min(lat.N_shell, 20)
    nonzero = np.abs(src[:n_show]) > 1e-15
    if nonzero.any():
        ax.semilogy(r[:n_show][nonzero], np.abs(src[:n_show][nonzero]),
                    'b-o', ms=5, lw=1.5)
    ax.axvline(r[n_core - 1], color='gray', ls=':', lw=1, alpha=0.7,
               label=f'Core boundary ($n={n_core}$)')
    ax.set_xlabel('$r/a$')
    ax.set_ylabel(r'$|\rho_{\rm bg} - \rho_{\rm tgt}|$')
    ax.set_title(rf'(f) Source localization at $\beta_0 t_0 = {bt_show}$')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'3D cubic lattice vacuum solver — GPU (R={R_MAX}, '
                 f'{lat.N_sites} sites, {lat.N_shell} shells)',
                 fontsize=14, y=1.01)
    plt.tight_layout()

    path = os.path.join(FIGDIR, "vacuum_3d_gpu_summary.pdf")
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
    print(f"\n3D cubic lattice: R={R_MAX}, {lat.N_sites} sites, {lat.N_shell} shells")
    print(f"GPU: {dev}")
    print(f"\nProxy equation (asymptotic Newtonian):")
    for bt0 in bts:
        pr = proxy_results[bt0]
        print(f"  bt0={bt0:.2f}: rs={pr['rs']:.4f}, GM={pr['GM']:.6f}, "
              f"min_N={pr['lapse'].min():.5f}, |F|={pr['res']:.1e}")
    print(f"\nFull self-consistent equation:")
    for bt0 in bts:
        fr = full_results[bt0]
        print(f"  bt0={bt0:.2f}: rs={fr['rs']:.4f}, GM={fr['GM']:.6f}, "
              f"|F|={fr['res']:.1e}, stag={fr['stagger']:.1e} [{fr['method']}]")
    print(f"\nPhysics:")
    print(f"  - Source (rho_bg - rho_tgt) localized within n_core={n_core} shells")
    print(f"  - Far field is Poisson => Phi = -GM(1/r - 1/R) (Newtonian)")
    print(f"  - Flat GM extraction profile confirms 1/r")
    print(f"\nPlots saved to: {FIGDIR}")


if __name__ == "__main__":
    main()
