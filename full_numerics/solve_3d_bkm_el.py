#!/usr/bin/env python3
"""
3D cubic-lattice solver: KMS/BKM conductances + full Euler-Lagrange.

Upgrades solve_3d_persite.py:
  - MI → BKM bond-current covariance conductances
  - Picard (frozen κ) → full EL via JAX autograd (exact ∂Q/∂Φ)
  - Self-consistent source (both states in same lapse field)

Strategy:
  1. Compute background BKM conductances (one-time, Φ=0)
  2. Picard with analytic κ (seed, ~200 iterations)
  3. Newton with full EL residual + exact Jacobian (JAX jacfwd)
  4. Verify convergence, analyze symmetry, save results

Usage:
  python full_numerics/solve_3d_bkm_el.py [--R 10]
"""

import sys
import os
import time
import argparse
import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve

import torch

import jax
import jax.numpy as jnp
from jax import jit, jacfwd
jax.config.update("jax_enable_x64", True)

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
# 3D cubic lattice
# ============================================================================

class CubicLattice3D:
    """3D cubic lattice within sphere of radius R_max."""

    def __init__(self, R_max, device='cuda'):
        self.R_max = R_max
        self.device = torch.device(device)
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

        # Build neighbor pairs (directed: both i->j and j->i)
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
        self.N_bonds_directed = len(self.row)

        # Unique undirected bonds (i < j)
        fwd_mask = self.row < self.col
        self.bond_i = self.row[fwd_mask]
        self.bond_j = self.col[fwd_mask]
        self.N_bonds = len(self.bond_i)

        # Boundary sites (outermost shell)
        self.boundary = np.where(self.shell_of == self.N_shell - 1)[0]

        # GPU tensors
        self.row_t = torch.tensor(self.row, dtype=torch.long, device=self.device)
        self.col_t = torch.tensor(self.col, dtype=torch.long, device=self.device)
        self.bi_t = torch.tensor(self.bond_i, dtype=torch.long, device=self.device)
        self.bj_t = torch.tensor(self.bond_j, dtype=torch.long, device=self.device)

        # Core indices (set later)
        self.core_idx = None
        self.core_t = None

        elapsed = time.time() - t0
        print(f"  Lattice: R={R_max}, {self.N_sites} sites, {self.N_shell} shells, "
              f"{self.N_bonds} bonds [{elapsed:.1f}s]", flush=True)

    def set_core(self, n_core):
        core_idx = []
        for n in range(min(n_core, self.N_shell)):
            core_idx.extend(self.shell_sites[n].tolist())
        self.core_idx = np.array(core_idx)
        self.core_t = torch.tensor(self.core_idx, dtype=torch.long, device=self.device)
        print(f"  Core sites: {len(self.core_idx)}", flush=True)


# ============================================================================
# BKM bond-current covariance (PyTorch, chunked)
# ============================================================================

def bkm_kernel(evals, beta0):
    """BKM kernel K[α,β] = f_α(1-f_β) · φ(β₀(ε_α-ε_β)).

    φ(x) = (e^x - 1)/x, φ(0) = 1.
    """
    be = torch.clamp(beta0 * evals, -500.0, 500.0)
    f = 1.0 / (torch.exp(be) + 1.0)
    diff = beta0 * (evals[:, None] - evals[None, :])
    phi = torch.where(torch.abs(diff) > 1e-12,
                      torch.expm1(diff) / torch.where(torch.abs(diff) > 1e-12,
                                                       diff, torch.ones_like(diff)),
                      torch.ones_like(diff))
    return f[:, None] * (1.0 - f[None, :]) * phi


def bkm_bond_cov_3d(evecs, K, bond_i, bond_j, chunk=3000):
    """BKM covariance of the bond-current operator per undirected bond.

    cov[b] = Σ_{α,β} (V_{j,α}V_{i,β} - V_{i,α}V_{j,β})² K[α,β]
           = (V_j² @ K @ V_i².T)[diag] + (V_i² @ K @ V_j².T)[diag]
             - 2 (cross @ K @ cross.T)[diag]

    Computed in chunks to manage memory.
    """
    N_bonds = bond_i.shape[0]
    cov_chunks = []

    for s in range(0, N_bonds, chunk):
        e = min(s + chunk, N_bonds)
        Ui = evecs[bond_i[s:e]]   # (chunk, M)
        Uj = evecs[bond_j[s:e]]   # (chunk, M)
        Ui_sq = Ui * Ui
        Uj_sq = Uj * Uj
        cross = Ui * Uj
        t1 = torch.sum((Uj_sq @ K) * Ui_sq, dim=1)
        t2 = torch.sum((Ui_sq @ K) * Uj_sq, dim=1)
        t3 = torch.sum((cross @ K) * cross, dim=1)
        cov_chunks.append(t1 + t2 - 2.0 * t3)

    return torch.cat(cov_chunks)


# ============================================================================
# JAX BKM functions (for autodiff through eigh)
# ============================================================================

def jax_bkm_kernel(evals, beta0):
    """BKM kernel K[α,β] = f_α(1-f_β) · φ(β₀(ε_α-ε_β))."""
    be = jnp.clip(beta0 * evals, -500.0, 500.0)
    f = 1.0 / (jnp.exp(be) + 1.0)
    diff = beta0 * (evals[:, None] - evals[None, :])
    phi = jnp.where(jnp.abs(diff) > 1e-12,
                    jnp.expm1(diff) / jnp.where(jnp.abs(diff) > 1e-12,
                                                  diff, jnp.ones_like(diff)),
                    jnp.ones_like(diff))
    return f[:, None] * (1.0 - f[None, :]) * phi


def jax_bkm_bond_cov(evecs, K, bond_i, bond_j):
    """BKM bond-current covariance per undirected bond (JAX)."""
    Ui = evecs[bond_i]
    Uj = evecs[bond_j]
    Ui_sq = Ui * Ui
    Uj_sq = Uj * Uj
    cross = Ui * Uj
    t1 = jnp.sum((Uj_sq @ K) * Ui_sq, axis=1)
    t2 = jnp.sum((Ui_sq @ K) * Uj_sq, axis=1)
    t3 = jnp.sum((cross @ K) * cross, axis=1)
    return t1 + t2 - 2.0 * t3


def make_jax_residual(lat, t0, V0, n_core, beta0, cstar_sq, bkm_bg_np):
    """Build JIT-compiled residual and Jacobian using JAX autodiff.

    The energy functional E[Φ] = ½ Σ_b κ_b(Φ)(ΔΦ_b)² is differentiated
    exactly via jax.grad, capturing both the N̄² kinematic factor and
    the ∂Q/∂Φ dependence of the BKM covariance ratio.
    """
    N = lat.N_sites
    cs2 = cstar_sq

    # Static arrays (JAX)
    row = jnp.array(lat.row)
    col = jnp.array(lat.col)
    bond_i = jnp.array(lat.bond_i)
    bond_j = jnp.array(lat.bond_j)
    bnd_mask = jnp.zeros(N, dtype=jnp.float64)
    bnd_mask = bnd_mask.at[lat.boundary].set(1.0)
    interior_mask = 1.0 - bnd_mask
    core_idx = jnp.array(lat.core_idx)
    bkm_bg_j = jnp.array(bkm_bg_np)

    # Tiny diagonal perturbation to break eigenvalue degeneracies
    # (JAX eigh gradient is ill-defined at degenerate eigenvalues)
    _diag_perturb = jnp.array(np.random.RandomState(42).randn(N) * 1e-10,
                               dtype=jnp.float64)

    def _conductances(Phi):
        """BKM conductances κ(Φ) via JAX eigh — differentiable."""
        lapse = 1.0 + Phi / cs2
        Nbar_dir = jnp.abs(0.5 * (lapse[row] + lapse[col]))
        Nbar_und = jnp.abs(0.5 * (lapse[bond_i] + lapse[bond_j]))

        # Build lapse-smeared Hamiltonian
        H = jnp.zeros((N, N), dtype=jnp.float64)
        H = H.at[row, col].set(-t0 * Nbar_dir)
        # Break eigenvalue degeneracies for stable autodiff
        H = H + jnp.diag(_diag_perturb)

        evals, evecs = jnp.linalg.eigh(H)
        K = jax_bkm_kernel(evals, beta0)
        cov = jax_bkm_bond_cov(evecs, K, bond_i, bond_j)

        kappa = t0**2 * Nbar_und**2 * cov / jnp.maximum(bkm_bg_j, 1e-30)
        return kappa

    def _energy(Phi):
        """E[Φ] = ½ Σ_b κ_b(Φ)(ΔΦ_b)²."""
        kappa = _conductances(Phi)
        dPhi = Phi[bond_i] - Phi[bond_j]
        return 0.5 * jnp.sum(kappa * dPhi**2)

    _grad_energy = jax.grad(_energy)

    @jit
    def grad_energy(Phi):
        return _grad_energy(Phi)

    @jit
    def jac_grad_energy(Phi):
        return jacfwd(_grad_energy)(Phi)

    return grad_energy, jac_grad_energy


# ============================================================================
# Background BKM (one-time, no autograd)
# ============================================================================

def compute_bkm_bg(lat, t0, beta0):
    """Background BKM covariance per undirected bond (Φ=0)."""
    dev = lat.device
    N = lat.N_sites
    print("  Computing background BKM covariances...", flush=True)
    ts = time.time()

    H = torch.zeros((N, N), dtype=torch.float64, device=dev)
    H[lat.row_t, lat.col_t] = -t0
    evals, evecs = torch.linalg.eigh(H)
    K = bkm_kernel(evals, beta0)
    cov_bg = bkm_bond_cov_3d(evecs, K, lat.bi_t, lat.bj_t)

    del H, evals, evecs, K
    torch.cuda.empty_cache()
    print(f"    BKM background in {time.time()-ts:.1f}s, "
          f"mean={cov_bg.mean().item():.6e}", flush=True)
    return cov_bg


# ============================================================================
# BKM conductances (no autograd needed)
# ============================================================================

def compute_bkm_conductances(Phi_np, lat, t0, beta0, cstar_sq, bkm_bg):
    """Compute BKM conductances κ_{ij}(Φ) per undirected bond.

    κ = t₀² N̄² cov/cov_bg with BKM covariance.
    Returns (kappa_np, Nbar_np) as numpy arrays of shape (N_bonds,).
    """
    dev = lat.device
    N = lat.N_sites
    cs2 = cstar_sq

    lapse = 1.0 + Phi_np / cs2
    Nbar_dir = np.abs(0.5 * (lapse[lat.row] + lapse[lat.col]))
    Nbar_und = np.abs(0.5 * (lapse[lat.bond_i] + lapse[lat.bond_j]))

    # Build lapse-smeared H
    Nbar_dir_t = torch.tensor(Nbar_dir, dtype=torch.float64, device=dev)
    H = torch.zeros((N, N), dtype=torch.float64, device=dev)
    H[lat.row_t, lat.col_t] = -t0 * Nbar_dir_t
    del Nbar_dir_t

    evals, evecs = torch.linalg.eigh(H)
    del H
    torch.cuda.synchronize()

    K = bkm_kernel(evals, beta0)
    cov = bkm_bond_cov_3d(evecs, K, lat.bi_t, lat.bj_t)
    cov_np = cov.cpu().numpy()

    del evals, evecs, K, cov
    torch.cuda.empty_cache()

    bkm_bg_np = bkm_bg.cpu().numpy()
    kappa_np = t0**2 * Nbar_und**2 * cov_np / np.maximum(bkm_bg_np, 1e-30)

    return kappa_np, Nbar_und


# ============================================================================
# Full EL residual (analytic approximation — kept for comparison/diagnostics)
# ============================================================================

def compute_full_el_lhs(Phi_np, lat, kappa_np, Nbar_np, cstar_sq):
    """Compute approximate EL LHS = Laplacian(κ,Φ) + analytic EL correction.

    NOTE: This is the analytic approximation that drops ∂Q/∂Φ.
    The exact EL is computed via JAX autodiff (make_jax_residual).
    Kept here for diagnostic comparison only.

    The EL correction from ∂κ/∂Φ uses the analytic chain rule:
      ∂κ_{ij}/∂Φ_i = κ_{ij} / (N̄_{ij} c*²)
    (treating the BKM ratio Q as independent of Φ).

    Full EL LHS_i = Σ_{j~i} κ_{ij}(Φ_i - Φ_j)
                  + (1/2) Σ_{j~i} [κ_{ij}/(N̄_{ij} c*²)] (Φ_i - Φ_j)²
    """
    N = lat.N_sites
    cs2 = cstar_sq

    # Standard Laplacian action
    lhs = np.zeros(N)
    dPhi = Phi_np[lat.bond_i] - Phi_np[lat.bond_j]
    np.add.at(lhs, lat.bond_i, kappa_np * dPhi)
    np.add.at(lhs, lat.bond_j, -kappa_np * dPhi)

    # EL correction: (1/2) ∂κ/∂Φ_i (ΔΦ)²
    # ∂κ_{ij}/∂Φ_i = κ_{ij}/(N̄_{ij} c*²)  (from ∂(N̄²)/∂Φ_i = N̄/c*²)
    el_coeff = kappa_np / (np.maximum(np.abs(Nbar_np), 1e-30) * cs2)
    dPhi_sq = dPhi**2
    np.add.at(lhs, lat.bond_i, 0.5 * el_coeff * dPhi_sq)
    np.add.at(lhs, lat.bond_j, 0.5 * el_coeff * dPhi_sq)

    return lhs


# ============================================================================
# Self-consistent source (no autograd)
# ============================================================================

def compute_source(Phi_np, lat, t0, V0, n_core, beta0, cstar_sq):
    """Compute (β₀/c*²)(ρ_σ - ρ_tgt) with both states in the same lapse field.

    Returns per-site source array (numpy).
    """
    dev = lat.device
    N = lat.N_sites
    cs2 = cstar_sq

    lapse = 1.0 + Phi_np / cs2
    Nbar_dir = np.abs(0.5 * (lapse[lat.row] + lapse[lat.col]))
    Nbar_dir_t = torch.tensor(Nbar_dir, dtype=torch.float64, device=dev)

    # --- Smeared background → ρ_σ ---
    H_sm = torch.zeros((N, N), dtype=torch.float64, device=dev)
    H_sm[lat.row_t, lat.col_t] = -t0 * Nbar_dir_t
    evals, evecs = torch.linalg.eigh(H_sm)
    del H_sm
    be = torch.clamp(beta0 * evals, -500, 500)
    f = 1.0 / (torch.exp(be) + 1.0)

    G_off_d = torch.zeros(lat.N_bonds_directed, dtype=torch.float64, device=dev)
    chunk = 10000
    for s in range(0, lat.N_bonds_directed, chunk):
        e = min(s + chunk, lat.N_bonds_directed)
        G_off_d[s:e] = torch.sum(
            evecs[lat.row_t[s:e]] * evecs[lat.col_t[s:e]] * f[None, :], dim=1)

    rho_sigma = torch.zeros(N, dtype=torch.float64, device=dev)
    rho_sigma.scatter_add_(0, lat.row_t, t0 * Nbar_dir_t * G_off_d)
    del evecs, evals, f, G_off_d
    torch.cuda.empty_cache()

    # --- Smeared defect → ρ_tgt ---
    H_src = torch.zeros((N, N), dtype=torch.float64, device=dev)
    H_src[lat.row_t, lat.col_t] = -t0 * Nbar_dir_t
    H_src[lat.core_t, lat.core_t] += V0
    evals_s, evecs_s = torch.linalg.eigh(H_src)
    del H_src
    be_s = torch.clamp(beta0 * evals_s, -500, 500)
    f_s = 1.0 / (torch.exp(be_s) + 1.0)

    G_bond_s = torch.zeros(lat.N_bonds_directed, dtype=torch.float64, device=dev)
    for s in range(0, lat.N_bonds_directed, chunk):
        e = min(s + chunk, lat.N_bonds_directed)
        G_bond_s[s:e] = torch.sum(
            evecs_s[lat.row_t[s:e]] * evecs_s[lat.col_t[s:e]] * f_s[None, :], dim=1)

    rho_tgt = torch.zeros(N, dtype=torch.float64, device=dev)
    rho_tgt.scatter_add_(0, lat.row_t, t0 * Nbar_dir_t * G_bond_s)
    del G_bond_s

    # On-site V₀ in core
    G_diag_s = torch.sum(evecs_s * evecs_s * f_s[None, :], dim=1)
    rho_tgt[lat.core_t] += V0 * G_diag_s[lat.core_t]
    del evecs_s, evals_s, f_s, G_diag_s, Nbar_dir_t
    torch.cuda.empty_cache()

    source = (rho_sigma - rho_tgt).cpu().numpy()
    del rho_sigma, rho_tgt

    pref = beta0 / cs2
    return pref * source


# ============================================================================
# Picard seed (analytic κ, proxy source)
# ============================================================================

def picard_analytic_seed(lat, t0, V0, n_core, beta0, cstar_sq,
                          max_iter=300, mixing=0.3, tol=1e-8):
    """Picard iteration with analytic κ = t₀²N̄² and proxy source."""
    N = lat.N_sites
    cs2 = cstar_sq
    Phi_lo = -cs2 * 0.99

    # Proxy source at Φ=0
    source = compute_source(np.zeros(N), lat, t0, V0, n_core, beta0, cs2)

    Phi = np.zeros(N)
    bnd = lat.boundary

    for it in range(max_iter):
        lapse = 1.0 + Phi / cs2
        Nbar = np.abs(0.5 * (lapse[lat.row] + lapse[lat.col]))
        kappa = t0**2 * Nbar**2

        W = csr_matrix((kappa, (lat.row, lat.col)), shape=(N, N))
        D_diag = np.array(W.sum(axis=1)).ravel()
        L = diags(D_diag) - W

        L_mod = L.tolil()
        for b in bnd:
            L_mod[b, :] = 0
            L_mod[b, b] = 1.0
        L_mod = L_mod.tocsr()

        rhs = source.copy()
        rhs[bnd] = 0.0

        Phi_new = spsolve(L_mod, rhs)
        Phi_new = np.maximum(Phi_new, Phi_lo)

        Phi_next = mixing * Phi_new + (1 - mixing) * Phi
        Phi_next[bnd] = 0.0

        chg = np.max(np.abs(Phi_next - Phi)) / max(1e-15, np.max(np.abs(Phi)))
        Phi = Phi_next

        if it % 50 == 0 or it < 3 or chg < tol:
            lapse_now = 1.0 + Phi / cs2
            print(f"    Picard {it:4d}: chg={chg:.3e}, "
                  f"min(N)={lapse_now.min():.5f}", flush=True)

        if chg < tol and it > 10:
            print(f"    Converged at iteration {it}", flush=True)
            break

    return Phi


# ============================================================================
# Newton iteration with exact EL via JAX autodiff
# ============================================================================

def newton_jax_el(lat, Phi_init, t0, V0, n_core, beta0, cstar_sq, bkm_bg,
                  grad_energy_fn, jac_fn=None,
                  max_iter=50, tol=1e-7, verbose=True):
    """Newton iteration with exact EL via JAX autodiff.

    LHS = jax.grad(E[Φ]) — exact, includes ∂Q/∂Φ through eigh
    Jacobian: if jac_fn provided, use exact jacfwd; otherwise use
              sparse κ-Laplacian approximation (much faster for large N).
    Source still computed via PyTorch (no ∂/∂Φ needed).
    """
    N = lat.N_sites
    cs2 = cstar_sq
    bnd = lat.boundary
    Phi = Phi_init.copy()
    use_exact_jac = jac_fn is not None

    for it in range(max_iter):
        ts = time.time()

        # Exact EL gradient via JAX
        Phi_j = jnp.array(Phi)
        lhs = np.array(grad_energy_fn(Phi_j))

        # Self-consistent source (PyTorch)
        source = compute_source(Phi, lat, t0, V0, n_core, beta0, cs2)

        # BKM conductances (for Jacobian approximation and diagnostics)
        kappa_np, Nbar_np = compute_bkm_conductances(
            Phi, lat, t0, beta0, cs2, bkm_bg)

        # Residual
        F = lhs - source
        F[bnd] = Phi[bnd]

        F_inf = np.max(np.abs(F))
        dt = time.time() - ts

        if verbose:
            lapse = 1.0 + Phi / cs2
            # Compute Picard-only LHS for comparison (no EL)
            L_kappa_Phi = np.zeros(N)
            dPhi_bond = Phi[lat.bond_i] - Phi[lat.bond_j]
            np.add.at(L_kappa_Phi, lat.bond_i, kappa_np * dPhi_bond)
            np.add.at(L_kappa_Phi, lat.bond_j, -kappa_np * dPhi_bond)
            el_correction = lhs - L_kappa_Phi
            interior = ~np.isin(np.arange(N), bnd)
            el_norm = np.max(np.abs(el_correction[interior]))

            print(f"    Newton {it:3d}: |F|∞={F_inf:.3e}, "
                  f"min(N)={lapse.min():.5f}, |EL_corr|∞={el_norm:.3e}, "
                  f"[{dt:.1f}s]", flush=True)

        if F_inf < tol:
            print(f"    Converged at iteration {it}", flush=True)
            break

        if use_exact_jac:
            # Exact Jacobian of energy gradient via JAX
            ts_jac = time.time()
            J = np.array(jac_fn(Phi_j))
            J[bnd, :] = 0
            J[bnd, bnd] = 1.0
            if verbose:
                print(f"      Jacobian (exact): {time.time()-ts_jac:.1f}s", flush=True)
            delta_Phi = np.linalg.solve(J, -F)
        else:
            # Approximate Jacobian: sparse κ-Laplacian
            data_w = np.concatenate([kappa_np, kappa_np])
            rows_w = np.concatenate([lat.bond_i, lat.bond_j])
            cols_w = np.concatenate([lat.bond_j, lat.bond_i])
            W = csr_matrix((data_w, (rows_w, cols_w)), shape=(N, N))
            D_diag = np.array(W.sum(axis=1)).ravel()
            L = diags(D_diag) - W
            L_mod = L.tolil()
            for b in bnd:
                L_mod[b, :] = 0
                L_mod[b, b] = 1.0
            L_mod = L_mod.tocsr()
            delta_Phi = spsolve(L_mod, -F)

        # Line search
        alpha = 1.0
        for _ in range(8):
            Phi_trial = Phi + alpha * delta_Phi
            Phi_trial[bnd] = 0.0

            lhs_trial = np.array(grad_energy_fn(jnp.array(Phi_trial)))
            source_trial = compute_source(Phi_trial, lat, t0, V0, n_core, beta0, cs2)
            F_trial = lhs_trial - source_trial
            F_trial[bnd] = Phi_trial[bnd]
            F_trial_inf = np.max(np.abs(F_trial))
            if F_trial_inf < F_inf:
                break
            alpha *= 0.5
        else:
            if verbose:
                print(f"      Line search: no improvement (α={alpha:.3e})", flush=True)
            alpha = 0.5**8

        Phi = Phi + alpha * delta_Phi
        Phi[bnd] = 0.0

        if verbose and alpha < 1.0:
            print(f"      Line search: α={alpha:.3f}", flush=True)

    # Return final conductances for diagnostics
    kappa_final, Nbar_final = compute_bkm_conductances(
        Phi, lat, t0, beta0, cs2, bkm_bg)
    return Phi, kappa_final, Nbar_final


# ============================================================================
# Symmetry analysis
# ============================================================================

def analyze_symmetry(lat, Phi, cstar_sq):
    """Analyze spherical symmetry of the per-site solution."""
    N_shell = lat.N_shell
    mean_Phi = np.zeros(N_shell)
    std_Phi = np.zeros(N_shell)
    spread_Phi = np.zeros(N_shell)

    for n in range(N_shell):
        idx = lat.shell_sites[n]
        if len(idx) == 0:
            continue
        vals = Phi[idx]
        mean_Phi[n] = np.mean(vals)
        std_Phi[n] = np.std(vals)
        spread_Phi[n] = np.max(vals) - np.min(vals)

    lapse_mean = 1.0 + mean_Phi / cstar_sq

    # True angular anisotropy (exact r²)
    r2_int = np.sum(lat.sites**2, axis=1)
    unique_r2 = np.unique(r2_int)
    angular_spreads = []
    for r2 in unique_r2:
        mask = r2_int == r2
        count = np.sum(mask)
        if count < 2:
            continue
        vals = Phi[mask]
        mean_v = np.mean(vals)
        if abs(mean_v) < 1e-30:
            continue
        spr = np.max(vals) - np.min(vals)
        angular_spreads.append(spr / abs(mean_v))

    max_angular = max(angular_spreads) if angular_spreads else 0
    median_angular = np.median(angular_spreads) if angular_spreads else 0

    print(f"\n  {'shell':>5s}  {'G_n':>5s}  {'mean_Phi':>12s}  {'spread/|mean|':>13s}  {'lapse':>8s}")
    for n in range(min(N_shell, 12)):
        rel = spread_Phi[n] / max(abs(mean_Phi[n]), 1e-30)
        print(f"  {n:5d}  {lat.G_n[n]:5d}  {mean_Phi[n]:12.6e}  {rel:13.4e}  {lapse_mean[n]:8.5f}")

    print(f"\n  Max angular anisotropy: {max_angular:.4e}")
    print(f"  Median angular anisotropy: {median_angular:.4e}")

    return {
        'mean_Phi': mean_Phi, 'std_Phi': std_Phi,
        'spread_Phi': spread_Phi, 'lapse_mean': lapse_mean,
        'max_angular_anisotropy': max_angular,
        'median_angular_anisotropy': median_angular,
    }


# ============================================================================
# GM extraction
# ============================================================================

def extract_gm(mean_Phi, cstar_sq, n_core):
    """Extract GM from shell-averaged Newtonian potential."""
    N = len(mean_Phi)
    r = np.arange(N, dtype=float)
    r[0] = 1e-10
    R = r[-1]
    with np.errstate(divide='ignore', invalid='ignore'):
        gm = -mean_Phi * r * R / (R - r)
    i_lo = n_core + 1
    i_hi = max(i_lo + 2, min(N - 2, N * 3 // 4))
    if i_lo >= i_hi:
        return 0.0, 0.0
    gm_mid = gm[i_lo:i_hi]
    gm_valid = gm_mid[np.isfinite(gm_mid)]
    GM = np.median(gm_valid) if len(gm_valid) > 0 else 0.0
    rs = 2.0 * GM / cstar_sq
    return rs, GM


# ============================================================================
# Plots
# ============================================================================

def make_plots(lat, Phi, sym, n_core, cstar_sq, beta0_t0, R_max,
               kappa_np=None, t0=1.0, V0=0.03, beta0=0.1):
    """Summary figure."""
    N_shell = lat.N_shell
    r = np.arange(N_shell, dtype=float)
    mean_Phi = sym['mean_Phi']
    rs, GM = extract_gm(mean_Phi, cstar_sq, n_core)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # (a) Shell-averaged potential
    ax = axes[0, 0]
    ax.plot(r[1:], mean_Phi[1:] / cstar_sq, 'bo-', ms=4, lw=1.5, label='BKM+EL (shell avg)')
    if rs > 0:
        R = r[-1]
        phi_newt = (rs / 2.0) * (1.0 / R - 1.0 / r[1:])
        ax.plot(r[1:], phi_newt, 'k--', lw=1.2, alpha=0.7, label=f'Newtonian ($r_s={rs:.3f}$)')
    ax.set_xlabel('Shell $n$')
    ax.set_ylabel(r'$\langle\Phi\rangle / c_*^2$')
    ax.set_title(rf'(a) Shell-averaged potential ($\beta_0 t_0={beta0_t0}$)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (b) Lapse
    ax = axes[0, 1]
    ax.plot(r, sym['lapse_mean'], 'bo-', ms=4, lw=1.5, label='BKM+EL')
    if rs > 0:
        lapse_schw = np.sqrt(np.maximum(1.0 - rs / np.maximum(r, 0.5), 0.0))
        ax.plot(r[1:], lapse_schw[1:], 'k--', lw=1.2, alpha=0.7, label=r'$\sqrt{1-r_s/r}$')
    ax.set_xlabel('Shell $n$')
    ax.set_ylabel('$N(r)$')
    ax.set_title(rf'(b) Lapse ($\beta_0 t_0={beta0_t0}$)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # (c) Angular anisotropy
    ax = axes[0, 2]
    rel_spread = sym['spread_Phi'] / np.maximum(np.abs(mean_Phi), 1e-30)
    ax.semilogy(r[1:-1], rel_spread[1:-1], 'ro-', ms=4, lw=1.5)
    ax.axhline(0.02, color='gray', ls=':', lw=1, alpha=0.7, label='2%')
    ax.set_xlabel('Shell $n$')
    ax.set_ylabel('Spread / $|\\langle\\Phi\\rangle|$')
    ax.set_title('(c) Departure from spherical symmetry')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (d) Phi on z=0 slice
    ax = axes[1, 0]
    z0_mask = lat.sites[:, 2] == 0
    if z0_mask.any():
        x_slice = lat.sites[z0_mask, 0]
        y_slice = lat.sites[z0_mask, 1]
        phi_slice = Phi[z0_mask] / cstar_sq
        sc = ax.scatter(x_slice, y_slice, c=phi_slice, cmap='RdBu_r', s=15, edgecolors='none')
        plt.colorbar(sc, ax=ax, label=r'$\Phi/c_*^2$')
    ax.set_xlabel('$x/a$')
    ax.set_ylabel('$y/a$')
    ax.set_title(r'(d) $\Phi/c_*^2$ in $z=0$ plane')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)

    # (e) Conductance comparison (shell-averaged κ/κ_flat vs N̄²)
    ax = axes[1, 1]
    if kappa_np is not None:
        # Shell-average BKM conductance
        kappa_flat = t0**2  # per bond, without g_n
        kappa_shell = np.zeros(N_shell - 1)
        kappa_count = np.zeros(N_shell - 1)
        lapse = 1.0 + Phi / cstar_sq
        for b in range(lat.N_bonds):
            i, j = lat.bond_i[b], lat.bond_j[b]
            si = lat.shell_of[i]
            sj = lat.shell_of[j]
            s_avg = (si + sj) // 2
            if s_avg < N_shell - 1:
                Nbar_b = 0.5 * (lapse[i] + lapse[j])
                # κ without g_n factor
                kappa_shell[s_avg] += kappa_np[b]
                kappa_count[s_avg] += Nbar_b**2 * t0**2
        mask = kappa_count > 0
        ratio = np.ones(N_shell - 1)
        ratio[mask] = kappa_shell[mask] / kappa_count[mask]
        ax.plot(np.arange(N_shell - 1)[mask], ratio[mask], 'bo-', ms=4, lw=1.5)
        ax.axhline(1.0, color='gray', ls=':', lw=1)
        ax.set_xlabel('Shell')
        ax.set_ylabel(r'$\kappa^{\rm BKM} / (t_0^2 \bar{N}^2)$')
        ax.set_title('(e) BKM/analytic conductance ratio')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No conductance data', transform=ax.transAxes, ha='center')

    # (f) Source localization
    ax = axes[1, 2]
    source_final = compute_source(Phi, lat, t0, V0, n_core, beta0, cstar_sq)
    src_shell = np.zeros(N_shell)
    for n in range(N_shell):
        idx = lat.shell_sites[n]
        if len(idx) > 0:
            src_shell[n] = np.mean(np.abs(source_final[idx]))
    nz = src_shell > 1e-30
    if nz.any():
        ax.semilogy(r[nz], src_shell[nz], 'bo-', ms=4, lw=1.5)
    ax.axvline(n_core - 0.5, color='gray', ls=':', lw=1, alpha=0.7, label='Core boundary')
    ax.set_xlabel('Shell $n$')
    ax.set_ylabel(r'$\langle|{\rm source}|\rangle$')
    ax.set_title('(f) Source localization')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'3D BKM+EL solver (R={R_max}, {lat.N_sites} sites, '
                 rf'$\beta_0 t_0={beta0_t0}$)', fontsize=14, y=1.01)
    plt.tight_layout()

    path = os.path.join(FIGDIR, f"persite_3d_bkm_el_R{R_max}.pdf")
    plt.savefig(path, bbox_inches='tight')
    plt.savefig(path.replace('.pdf', '.png'), bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():

    t0_hop = 1.0

    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, default=10)
    ap.add_argument("--n-core", type=int, default=3)
    ap.add_argument("--cstar-sq", type=float, default=0.5)
    ap.add_argument("--betaV0", type=float, default=0.003)
    ap.add_argument("--bt0", type=float, default=0.10)
    ap.add_argument("--newton-iter", type=int, default=30)
    ap.add_argument("--newton-tol", type=float, default=1e-7)
    ap.add_argument("--exact-jac", action="store_true",
                    help="Use exact JAX jacfwd Jacobian (slow compile for large N)")
    args = ap.parse_args()

    R_MAX = args.R
    cstar_sq = args.cstar_sq
    n_core = args.n_core
    bt0 = args.bt0
    beta0 = bt0 / t0_hop
    V0 = args.betaV0 / beta0

    print("=" * 80)
    print(f"3D BKM + FULL EL SOLVER (R={R_MAX}, bt0={bt0})")
    print(f"  n_core={n_core}, c*²={cstar_sq}, β₀V₀={args.betaV0}, V₀={V0:.6f}")
    print("=" * 80)

    # Build lattice
    lat = CubicLattice3D(R_MAX, device='cuda')
    lat.set_core(n_core)

    # Background BKM
    bkm_bg = compute_bkm_bg(lat, t0_hop, beta0)

    # Build JAX energy gradient (and optionally exact Jacobian)
    print(f"\n--- Building JAX energy functional ---", flush=True)
    bkm_bg_np = bkm_bg.cpu().numpy()
    grad_energy_fn, jac_fn = make_jax_residual(
        lat, t0_hop, V0, n_core, beta0, cstar_sq, bkm_bg_np)

    # Warm up JIT for grad_energy
    print("  JIT compiling grad_energy...", flush=True)
    ts = time.time()
    _ = grad_energy_fn(jnp.zeros(lat.N_sites, dtype=jnp.float64))
    print(f"  grad_energy compiled in {time.time()-ts:.1f}s", flush=True)

    if args.exact_jac:
        print("  JIT compiling exact Jacobian (jacfwd)...", flush=True)
        ts = time.time()
        _ = jac_fn(jnp.zeros(lat.N_sites, dtype=jnp.float64))
        print(f"  Jacobian compiled in {time.time()-ts:.1f}s", flush=True)
    else:
        jac_fn = None
        print("  Using approximate sparse Jacobian (pass --exact-jac for exact)", flush=True)

    # Phase 1: Picard seed (analytic κ, proxy source)
    print(f"\n--- Phase 1: Picard seed (analytic κ) ---", flush=True)
    ts = time.time()
    Phi_seed = picard_analytic_seed(lat, t0_hop, V0, n_core, beta0, cstar_sq,
                                     max_iter=300, tol=1e-8)
    print(f"  Picard done in {time.time()-ts:.1f}s", flush=True)

    # Check full EL residual of the seed
    print(f"\n--- Checking full EL residual of seed ---", flush=True)
    kappa_seed, Nbar_seed = compute_bkm_conductances(
        Phi_seed, lat, t0_hop, beta0, cstar_sq, bkm_bg)
    lhs_seed = compute_full_el_lhs(Phi_seed, lat, kappa_seed, Nbar_seed, cstar_sq)
    source_seed = compute_source(Phi_seed, lat, t0_hop, V0, n_core, beta0, cstar_sq)
    F_seed = lhs_seed - source_seed
    F_seed[lat.boundary] = Phi_seed[lat.boundary]
    print(f"  Seed |F|∞ = {np.max(np.abs(F_seed)):.3e}")

    # Phase 2: Newton with full EL
    print(f"\n--- Phase 2: Newton with BKM + full EL ---", flush=True)
    ts = time.time()
    Phi_final, kappa_final, Nbar_final = newton_jax_el(
        lat, Phi_seed, t0_hop, V0, n_core, beta0, cstar_sq, bkm_bg,
        grad_energy_fn, jac_fn,
        max_iter=args.newton_iter, tol=args.newton_tol)
    print(f"  Newton done in {time.time()-ts:.1f}s", flush=True)

    # Final analysis
    print(f"\n--- Symmetry analysis ---", flush=True)
    sym = analyze_symmetry(lat, Phi_final, cstar_sq)
    rs, GM = extract_gm(sym['mean_Phi'], cstar_sq, n_core)
    print(f"\n  rs = {rs:.6f}, GM = {GM:.8f}")

    # Compute final EL residual (exact, via JAX autodiff)
    lhs_final = np.array(grad_energy_fn(jnp.array(Phi_final)))
    source_final = compute_source(Phi_final, lat, t0_hop, V0, n_core, beta0, cstar_sq)
    F_final = lhs_final - source_final
    F_final[lat.boundary] = Phi_final[lat.boundary]
    print(f"  Final |F|∞ = {np.max(np.abs(F_final)):.3e}")

    # Also compute the EL correction magnitude
    N = lat.N_sites
    L_kappa_Phi = np.zeros(N)
    dPhi_bond = Phi_final[lat.bond_i] - Phi_final[lat.bond_j]
    np.add.at(L_kappa_Phi, lat.bond_i, kappa_final * dPhi_bond)
    np.add.at(L_kappa_Phi, lat.bond_j, -kappa_final * dPhi_bond)
    el_corr = lhs_final - L_kappa_Phi
    interior = ~np.isin(np.arange(N), lat.boundary)
    print(f"  |EL correction|∞ = {np.max(np.abs(el_corr[interior])):.3e}")
    print(f"  |EL correction| / |LHS| = "
          f"{np.max(np.abs(el_corr[interior])) / max(np.max(np.abs(lhs_final[interior])), 1e-30):.3e}")

    # Source localization
    src_core = np.sum(np.abs(source_final[lat.core_idx]))
    src_total = np.sum(np.abs(source_final))
    print(f"  Source core fraction: {src_core/max(src_total,1e-30):.4f}")

    # w = N² profile check
    lapse = 1.0 + sym['mean_Phi'] / cstar_sq
    w = lapse**2
    print(f"\n  w = N² profile (should be 1 - rs/r):")
    for n in range(1, min(lat.N_shell - 1, 10)):
        w_schw = 1.0 - rs / n if n > 0 else 0
        print(f"    n={n:2d}: w={w[n]:.6f}, w_Schw={w_schw:.6f}, "
              f"diff={w[n]-w_schw:.2e}")

    # Plots
    print(f"\n--- Plots ---", flush=True)
    make_plots(lat, Phi_final, sym, n_core, cstar_sq, bt0, R_MAX,
               kappa_final, t0=t0_hop, V0=V0, beta0=beta0)

    # Save data
    outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           f"3d_bkm_el_R{R_MAX}.npz")
    np.savez(outpath,
             Phi=Phi_final,
             kappa=kappa_final,
             mean_Phi=sym['mean_Phi'],
             lapse_mean=sym['lapse_mean'],
             spread_Phi=sym['spread_Phi'],
             sites=lat.sites,
             shell_of=lat.shell_of,
             bond_i=lat.bond_i,
             bond_j=lat.bond_j,
             rs=rs, GM=GM,
             R_max=R_MAX,
             beta0=beta0, V0=V0, t0=t0_hop,
             cstar_sq=cstar_sq, n_core=n_core,
             max_angular_anisotropy=sym['max_angular_anisotropy'],
             F_inf=np.max(np.abs(F_final)),
             el_correction_inf=np.max(np.abs(el_corr[interior])))
    print(f"  Saved: {outpath}")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"  3D cubic lattice: R={R_MAX}, {lat.N_sites} sites")
    print(f"  KMS/BKM conductances + full Euler-Lagrange")
    print(f"  rs = {rs:.6f}, GM = {GM:.8f}")
    print(f"  Final |F|∞ = {np.max(np.abs(F_final)):.3e}")
    print(f"  |EL correction| / |LHS| = "
          f"{np.max(np.abs(el_corr[interior])) / max(np.max(np.abs(lhs_final[interior])), 1e-30):.3e}")
    print(f"  Max angular anisotropy: {sym['max_angular_anisotropy']:.4e}")
    print(f"  Source core fraction: {src_core/max(src_total,1e-30):.4f}")
    print(f"  Spherical symmetry: "
          f"{'YES (<5%)' if sym['max_angular_anisotropy'] < 0.05 else 'MODERATE'}")


if __name__ == "__main__":
    main()
