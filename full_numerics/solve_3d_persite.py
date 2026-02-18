#!/usr/bin/env python3
"""
3D cubic-lattice PER-SITE solver — tests whether spherical symmetry emerges.

Unlike the shell-averaged solvers, here Phi varies over every lattice site.
The self-consistent closure equation is:

  F_i = sum_{j~i} kappa_{ij}(Phi)(Phi_i - Phi_j)
        - (beta0/c*^2)(rho_bg_i - rho_tgt_i)        [proxy]

where kappa_{ij} are bond conductances and the source is per-site.

Strategy:
  1. Compute per-site source from full 3D eigh (one-time GPU cost)
  2. Picard iteration with analytic kappa (no eigh, sparse linear solve)
  3. Optionally, MI-based kappa Picard (eigh each iteration)
  4. Analyze: is the solution spherically symmetric?

Usage:
  PYTHONUNBUFFERED=1 python full_numerics/solve_3d_persite.py [--R 10]
"""

import sys
import os
import time
import argparse
import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve

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
# Binary entropy
# ============================================================================

def binary_entropy(x):
    x = np.clip(x, 1e-30, 1.0 - 1e-30)
    return -x * np.log(x) - (1.0 - x) * np.log(1.0 - x)


# ============================================================================
# 3D cubic lattice
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
        # Core sites
        self.core_mask = np.zeros(self.N_sites, dtype=bool)

        elapsed = time.time() - t0
        print(f"  Lattice: R={R_max}, {self.N_sites} sites, {self.N_shell} shells, "
              f"{self.N_bonds} bonds [{elapsed:.1f}s]", flush=True)


# ============================================================================
# Per-site source computation (requires eigh — one-time cost)
# ============================================================================

def compute_persite_source(lat, t0, V0, n_core, beta0):
    """Compute per-site source = rho_bg - rho_tgt from full 3D eigh.

    Energy per site (positive convention):
      rho_i = sum_{j~i} t0 * G_{ij}

    Target includes V0:
      rho_tgt_i = sum_{j~i} t0 * G^tgt_{ij} + V0 * G^tgt_{ii} * 1_core(i)
    """
    from scipy.linalg import eigh

    N = lat.N_sites

    # Background Hamiltonian (no V0)
    H_bg = np.zeros((N, N))
    H_bg[lat.row, lat.col] = -t0
    print(f"  Computing background eigh ({N}x{N})...", flush=True)
    t_s = time.time()
    evals_bg, evecs_bg = eigh(H_bg)
    del H_bg
    print(f"    eigh in {time.time()-t_s:.1f}s", flush=True)

    be_bg = np.clip(beta0 * evals_bg, -500, 500)
    f_bg = 1.0 / (np.exp(be_bg) + 1.0)

    # Per-site background energy: rho_bg_i = sum_{j~i} t0 * G_{ij}
    rho_bg = np.zeros(N)
    for start in range(0, lat.N_bonds_directed, 10000):
        end = min(start + 10000, lat.N_bonds_directed)
        ri, ci = lat.row[start:end], lat.col[start:end]
        G_bond = np.sum(evecs_bg[ri] * (evecs_bg[ci] * f_bg[None, :]), axis=1)
        np.add.at(rho_bg, ri, t0 * G_bond)
    del evecs_bg, evals_bg

    # Target Hamiltonian (with V0 in core)
    H_tgt = np.zeros((N, N))
    H_tgt[lat.row, lat.col] = -t0
    core_idx = []
    for n in range(min(n_core, lat.N_shell)):
        core_idx.extend(lat.shell_sites[n].tolist())
    core_idx = np.array(core_idx)
    H_tgt[core_idx, core_idx] += V0
    print(f"  Computing target eigh ({N}x{N})...", flush=True)
    t_s = time.time()
    evals_tgt, evecs_tgt = eigh(H_tgt)
    del H_tgt
    print(f"    eigh in {time.time()-t_s:.1f}s", flush=True)

    be_tgt = np.clip(beta0 * evals_tgt, -500, 500)
    f_tgt = 1.0 / (np.exp(be_tgt) + 1.0)

    # Per-site target energy
    rho_tgt = np.zeros(N)
    for start in range(0, lat.N_bonds_directed, 10000):
        end = min(start + 10000, lat.N_bonds_directed)
        ri, ci = lat.row[start:end], lat.col[start:end]
        G_bond = np.sum(evecs_tgt[ri] * (evecs_tgt[ci] * f_tgt[None, :]), axis=1)
        np.add.at(rho_tgt, ri, t0 * G_bond)
    # Add on-site V0 contribution in core
    G_diag_tgt = np.sum(evecs_tgt * (evecs_tgt * f_tgt[None, :]), axis=1)
    rho_tgt[core_idx] += V0 * G_diag_tgt[core_idx]
    del evecs_tgt, evals_tgt

    source = rho_bg - rho_tgt
    print(f"  Source: |total|={np.sum(np.abs(source)):.6f}, "
          f"core fraction={np.sum(np.abs(source[core_idx]))/np.sum(np.abs(source)):.4f}",
          flush=True)

    return source, rho_bg, rho_tgt, core_idx


def compute_persite_source_gpu(lat, t0, V0, n_core, beta0):
    """GPU-accelerated version of compute_persite_source."""
    import torch
    device = torch.device('cuda')

    N = lat.N_sites
    row_t = torch.tensor(lat.row, dtype=torch.long, device=device)
    col_t = torch.tensor(lat.col, dtype=torch.long, device=device)

    # Background
    H_bg = torch.zeros((N, N), dtype=torch.float64, device=device)
    H_bg[row_t, col_t] = -t0
    print(f"  Computing background eigh on GPU ({N}x{N})...", flush=True)
    t_s = time.time()
    evals_bg, evecs_bg = torch.linalg.eigh(H_bg)
    del H_bg
    torch.cuda.synchronize()
    print(f"    eigh in {time.time()-t_s:.1f}s", flush=True)

    be_bg = torch.clamp(beta0 * evals_bg, -500, 500)
    f_bg = 1.0 / (torch.exp(be_bg) + 1.0)

    rho_bg = torch.zeros(N, dtype=torch.float64, device=device)
    G_bond_bg = torch.sum(evecs_bg[row_t] * evecs_bg[col_t] * f_bg[None, :], dim=1)
    rho_bg.scatter_add_(0, row_t, t0 * G_bond_bg)
    del G_bond_bg

    rho_bg_np = rho_bg.cpu().numpy()
    del rho_bg, evecs_bg, evals_bg

    # Target (with V0)
    H_tgt = torch.zeros((N, N), dtype=torch.float64, device=device)
    H_tgt[row_t, col_t] = -t0
    core_idx = []
    for n in range(min(n_core, lat.N_shell)):
        core_idx.extend(lat.shell_sites[n].tolist())
    core_idx = np.array(core_idx)
    core_t = torch.tensor(core_idx, dtype=torch.long, device=device)
    H_tgt[core_t, core_t] += V0

    print(f"  Computing target eigh on GPU ({N}x{N})...", flush=True)
    t_s = time.time()
    evals_tgt, evecs_tgt = torch.linalg.eigh(H_tgt)
    del H_tgt
    torch.cuda.synchronize()
    print(f"    eigh in {time.time()-t_s:.1f}s", flush=True)

    be_tgt = torch.clamp(beta0 * evals_tgt, -500, 500)
    f_tgt = 1.0 / (torch.exp(be_tgt) + 1.0)

    rho_tgt = torch.zeros(N, dtype=torch.float64, device=device)
    G_bond_tgt = torch.sum(evecs_tgt[row_t] * evecs_tgt[col_t] * f_tgt[None, :], dim=1)
    rho_tgt.scatter_add_(0, row_t, t0 * G_bond_tgt)
    del G_bond_tgt
    # On-site V0
    G_diag = torch.sum(evecs_tgt * evecs_tgt * f_tgt[None, :], dim=1)
    rho_tgt[core_t] += V0 * G_diag[core_t]
    del evecs_tgt, evals_tgt, G_diag

    rho_tgt_np = rho_tgt.cpu().numpy()
    del rho_tgt
    torch.cuda.empty_cache()

    source = rho_bg_np - rho_tgt_np
    print(f"  Source: |total|={np.sum(np.abs(source)):.6f}, "
          f"core fraction={np.sum(np.abs(source[core_idx]))/np.sum(np.abs(source)):.4f}",
          flush=True)

    return source, rho_bg_np, rho_tgt_np, core_idx


# ============================================================================
# Picard iteration with analytic conductances (no eigh needed)
# ============================================================================

def picard_persite(lat, source, t0, beta0, cstar_sq,
                   max_iter=500, mixing=0.3, lapse_floor=0.01,
                   tol=1e-8, verbose=True):
    """Picard iteration with analytic kappa_{ij} = t0^2 * Nbar_{ij}^2.

    Solves: L_kappa(Phi) Phi = (beta0/c*^2) * source
    with BC: Phi = 0 at boundary (outermost shell).
    """
    N = lat.N_sites
    cs2 = cstar_sq
    pref = beta0 / cs2
    rhs = pref * source

    Phi = np.zeros(N)
    Phi_lo = -cs2 * (1.0 - lapse_floor)

    # Boundary mask (fix Phi=0 at outermost shell)
    bnd = lat.boundary
    interior = np.ones(N, dtype=bool)
    interior[bnd] = False
    int_idx = np.where(interior)[0]

    row = lat.row
    col = lat.col

    for it in range(max_iter):
        lapse = 1.0 + Phi / cs2
        Nbar = 0.5 * (lapse[row] + lapse[col])
        kappa = t0**2 * Nbar**2

        # Build sparse graph Laplacian
        # (L Phi)_i = sum_{j~i} kappa_{ij} (Phi_i - Phi_j)
        # L = D - W where D_ii = sum_j kappa_{ij}, W_{ij} = kappa_{ij}
        W = csr_matrix((kappa, (row, col)), shape=(N, N))
        D_diag = np.array(W.sum(axis=1)).ravel()

        # Replace boundary rows: Phi_bnd = 0
        # Build system: L @ Phi = rhs, with Phi[bnd] = 0
        # => just solve for interior, using rhs - W @ (Phi_bnd) = rhs (since Phi_bnd=0)
        L = diags(D_diag) - W

        # Set boundary rows to identity
        L_mod = L.tolil()
        for b in bnd:
            L_mod[b, :] = 0
            L_mod[b, b] = 1.0
        L_mod = L_mod.tocsr()

        rhs_mod = rhs.copy()
        rhs_mod[bnd] = 0.0

        Phi_new = spsolve(L_mod, rhs_mod)
        Phi_new = np.maximum(Phi_new, Phi_lo)

        Phi_next = mixing * Phi_new + (1 - mixing) * Phi
        Phi_next[bnd] = 0.0

        chg = np.max(np.abs(Phi_next - Phi)) / max(1e-15, np.max(np.abs(Phi)))
        Phi = Phi_next

        if verbose and (it % 50 == 0 or it < 5 or chg < tol):
            lapse = 1.0 + Phi / cs2
            print(f"    Picard {it:4d}: chg={chg:.3e}, "
                  f"min(N)={lapse.min():.5f}, max|Phi|={np.max(np.abs(Phi)):.6e}",
                  flush=True)

        if chg < tol and it > 10:
            if verbose:
                print(f"    Converged at iteration {it}", flush=True)
            break

    return Phi


# ============================================================================
# MI-based Picard iteration (fully self-consistent, no high-T approximation)
# ============================================================================

def picard_persite_mi_gpu(lat, t0, V0, n_core, beta0, cstar_sq,
                           Phi_init=None, max_iter=100, mixing=0.3,
                           lapse_floor=0.01, tol=1e-6, verbose=True):
    """MI-based Picard iteration: fully self-consistent, no high-T approximation.

    Each iteration performs two GPU eighs to compute:
    - MI-based conductances kappa_ij = t0^2 * MI_ij(Phi) / MI^bg_ij
    - Self-consistent source: rho_sigma(Phi) - rho_tgt(Phi), both in same lapse field
    Then solves the linear system L_kappa @ Phi = (beta0/c*^2) * source.
    """
    import torch
    device = torch.device('cuda')

    def bent(x):
        """Binary entropy for torch tensors."""
        x = torch.clamp(x, 1e-30, 1.0 - 1e-30)
        return -x * torch.log(x) - (1.0 - x) * torch.log(1.0 - x)

    N = lat.N_sites
    cs2 = cstar_sq
    pref = beta0 / cs2
    Phi_lo = -cs2 * (1.0 - lapse_floor)
    chunk = 10000

    row_t = torch.tensor(lat.row, dtype=torch.long, device=device)
    col_t = torch.tensor(lat.col, dtype=torch.long, device=device)
    bi_t = torch.tensor(lat.bond_i, dtype=torch.long, device=device)
    bj_t = torch.tensor(lat.bond_j, dtype=torch.long, device=device)

    # Core indices
    core_idx = []
    for n in range(min(n_core, lat.N_shell)):
        core_idx.extend(lat.shell_sites[n].tolist())
    core_idx_np = np.array(core_idx)
    core_t = torch.tensor(core_idx_np, dtype=torch.long, device=device)

    # Boundary
    bnd = lat.boundary

    # ---- Background MI (one-time, Phi=0) ----
    print("  Computing background MI on GPU...", flush=True)
    ts = time.time()
    H_bg = torch.zeros((N, N), dtype=torch.float64, device=device)
    H_bg[row_t, col_t] = -t0
    evals, evecs = torch.linalg.eigh(H_bg)
    del H_bg
    torch.cuda.synchronize()
    be = torch.clamp(beta0 * evals, -500, 500)
    f = 1.0 / (torch.exp(be) + 1.0)

    G_diag_bg = torch.sum(evecs**2 * f[None, :], dim=1)
    G_off_bg = torch.zeros(lat.N_bonds, dtype=torch.float64, device=device)
    for s in range(0, lat.N_bonds, chunk):
        e = min(s + chunk, lat.N_bonds)
        G_off_bg[s:e] = torch.sum(
            evecs[bi_t[s:e]] * evecs[bj_t[s:e]] * f[None, :], dim=1)

    a0 = G_diag_bg[bi_t]; d0 = G_diag_bg[bj_t]; b0 = G_off_bg
    tr0 = a0 + d0; det0 = a0 * d0 - b0 * b0
    disc0 = torch.clamp(tr0 * tr0 - 4 * det0, min=0)
    l10 = 0.5 * (tr0 + torch.sqrt(disc0))
    l20 = 0.5 * (tr0 - torch.sqrt(disc0))
    mi_bg = bent(a0) + bent(d0) - bent(l10) - bent(l20)
    mi_bg = torch.clamp(mi_bg, min=1e-30)

    del evecs, evals, f, G_diag_bg, G_off_bg
    del a0, d0, b0, tr0, det0, disc0, l10, l20
    torch.cuda.empty_cache()
    print(f"    Background MI in {time.time()-ts:.1f}s  "
          f"mean(MI_bg)={mi_bg.mean().item():.6e}", flush=True)

    # ---- Picard loop ----
    Phi = Phi_init.copy() if Phi_init is not None else np.zeros(N)

    for it in range(max_iter):
        ts_it = time.time()

        lapse = 1.0 + Phi / cs2
        Nbar_dir = np.abs(0.5 * (lapse[lat.row] + lapse[lat.col]))
        Nbar_dir_t = torch.tensor(Nbar_dir, dtype=torch.float64, device=device)

        # ---- Smeared background eigh ----
        H_sm = torch.zeros((N, N), dtype=torch.float64, device=device)
        H_sm[row_t, col_t] = -t0 * Nbar_dir_t
        evals, evecs = torch.linalg.eigh(H_sm)
        del H_sm
        torch.cuda.synchronize()
        be = torch.clamp(beta0 * evals, -500, 500)
        f = 1.0 / (torch.exp(be) + 1.0)

        # MI per undirected bond
        G_diag = torch.sum(evecs**2 * f[None, :], dim=1)
        G_off_u = torch.zeros(lat.N_bonds, dtype=torch.float64, device=device)
        for s in range(0, lat.N_bonds, chunk):
            e = min(s + chunk, lat.N_bonds)
            G_off_u[s:e] = torch.sum(
                evecs[bi_t[s:e]] * evecs[bj_t[s:e]] * f[None, :], dim=1)

        aa = G_diag[bi_t]; dd = G_diag[bj_t]; bb = G_off_u
        trv = aa + dd; detv = aa * dd - bb * bb
        dsc = torch.clamp(trv * trv - 4 * detv, min=0)
        l1 = 0.5 * (trv + torch.sqrt(dsc))
        l2 = 0.5 * (trv - torch.sqrt(dsc))
        mi_v = bent(aa) + bent(dd) - bent(l1) - bent(l2)
        mi_v = torch.clamp(mi_v, min=1e-30)
        kappa_und = (t0**2 * mi_v / mi_bg).cpu().numpy()

        del aa, dd, bb, trv, detv, dsc, l1, l2, mi_v, G_off_u

        # rho_sigma per site (directed bonds for scatter_add)
        G_off_d = torch.zeros(lat.N_bonds_directed, dtype=torch.float64,
                              device=device)
        for s in range(0, lat.N_bonds_directed, chunk):
            e = min(s + chunk, lat.N_bonds_directed)
            G_off_d[s:e] = torch.sum(
                evecs[row_t[s:e]] * evecs[col_t[s:e]] * f[None, :], dim=1)

        rho_sigma = torch.zeros(N, dtype=torch.float64, device=device)
        rho_sigma.scatter_add_(0, row_t, t0 * Nbar_dir_t * G_off_d)

        del evecs, evals, f, G_diag, G_off_d
        torch.cuda.empty_cache()

        # ---- Smeared defect eigh ----
        H_src = torch.zeros((N, N), dtype=torch.float64, device=device)
        H_src[row_t, col_t] = -t0 * Nbar_dir_t
        H_src[core_t, core_t] += V0
        evals_s, evecs_s = torch.linalg.eigh(H_src)
        del H_src
        torch.cuda.synchronize()
        be_s = torch.clamp(beta0 * evals_s, -500, 500)
        f_s = 1.0 / (torch.exp(be_s) + 1.0)

        G_bond_s = torch.zeros(lat.N_bonds_directed, dtype=torch.float64,
                               device=device)
        for s in range(0, lat.N_bonds_directed, chunk):
            e = min(s + chunk, lat.N_bonds_directed)
            G_bond_s[s:e] = torch.sum(
                evecs_s[row_t[s:e]] * evecs_s[col_t[s:e]] * f_s[None, :],
                dim=1)

        rho_tgt = torch.zeros(N, dtype=torch.float64, device=device)
        rho_tgt.scatter_add_(0, row_t, t0 * Nbar_dir_t * G_bond_s)
        del G_bond_s

        G_diag_s = torch.sum(evecs_s**2 * f_s[None, :], dim=1)
        rho_tgt[core_t] += V0 * G_diag_s[core_t]

        del evecs_s, evals_s, f_s, G_diag_s, Nbar_dir_t
        torch.cuda.empty_cache()

        # Source
        source_np = (rho_sigma - rho_tgt).cpu().numpy()
        del rho_sigma, rho_tgt

        dt_it = time.time() - ts_it

        # ---- Sparse solve ----
        rhs = pref * source_np

        data_w = np.concatenate([kappa_und, kappa_und])
        rows_w = np.concatenate([lat.bond_i, lat.bond_j])
        cols_w = np.concatenate([lat.bond_j, lat.bond_i])
        W = csr_matrix((data_w, (rows_w, cols_w)), shape=(N, N))
        D_diag = np.array(W.sum(axis=1)).ravel()
        L = diags(D_diag) - W

        L_mod = L.tolil()
        for b_idx in bnd:
            L_mod[b_idx, :] = 0
            L_mod[b_idx, b_idx] = 1.0
        L_mod = L_mod.tocsr()

        rhs_mod = rhs.copy()
        rhs_mod[bnd] = 0.0

        Phi_new = spsolve(L_mod, rhs_mod)
        Phi_new = np.maximum(Phi_new, Phi_lo)

        Phi_next = mixing * Phi_new + (1 - mixing) * Phi
        Phi_next[bnd] = 0.0

        chg = np.max(np.abs(Phi_next - Phi)) / max(1e-15, np.max(np.abs(Phi)))
        Phi = Phi_next

        if verbose and (it % 5 == 0 or it < 3 or chg < tol):
            lapse_now = 1.0 + Phi / cs2
            src_core = np.sum(np.abs(source_np[core_idx_np]))
            src_total = np.sum(np.abs(source_np))
            print(f"    MI-Picard {it:4d}: chg={chg:.3e}, "
                  f"min(N)={lapse_now.min():.5f}, "
                  f"|Phi|_max={np.max(np.abs(Phi)):.4e}, "
                  f"core_frac={src_core/max(src_total,1e-30):.3f}, "
                  f"gpu={dt_it:.1f}s", flush=True)

        if chg < tol and it > 5:
            if verbose:
                print(f"    Converged at iteration {it}", flush=True)
            break

    return Phi


# ============================================================================
# Spherical symmetry analysis
# ============================================================================

def analyze_symmetry(lat, Phi, cstar_sq):
    """Analyze how spherically symmetric the per-site solution is."""
    N_shell = lat.N_shell
    r_shell = np.arange(N_shell, dtype=float)

    mean_Phi = np.zeros(N_shell)
    std_Phi = np.zeros(N_shell)
    spread_Phi = np.zeros(N_shell)
    min_Phi = np.zeros(N_shell)
    max_Phi = np.zeros(N_shell)

    for n in range(N_shell):
        idx = lat.shell_sites[n]
        if len(idx) == 0:
            continue
        vals = Phi[idx]
        mean_Phi[n] = np.mean(vals)
        std_Phi[n] = np.std(vals)
        min_Phi[n] = np.min(vals)
        max_Phi[n] = np.max(vals)
        spread_Phi[n] = max_Phi[n] - min_Phi[n]

    lapse_mean = 1.0 + mean_Phi / cstar_sq

    print(f"\n  {'shell':>5s}  {'G_n':>5s}  {'mean_Phi':>12s}  {'std_Phi':>12s}  "
          f"{'spread':>12s}  {'spread/|mean|':>13s}  {'lapse':>8s}")
    print(f"  {'-'*80}")
    for n in range(N_shell):
        rel = spread_Phi[n] / max(abs(mean_Phi[n]), 1e-30)
        print(f"  {n:5d}  {lat.G_n[n]:5d}  {mean_Phi[n]:12.6e}  {std_Phi[n]:12.6e}  "
              f"{spread_Phi[n]:12.6e}  {rel:13.4e}  {lapse_mean[n]:8.5f}", flush=True)

    # Overall measure: max relative spread across non-boundary shells
    n_int = N_shell - 1  # exclude boundary
    rel_spreads = spread_Phi[1:n_int] / np.maximum(np.abs(mean_Phi[1:n_int]), 1e-30)
    max_rel_spread = np.max(rel_spreads) if len(rel_spreads) > 0 else 0

    print(f"\n  Max relative spread (shells 1..{n_int-1}): {max_rel_spread:.4e}")
    print(f"  NOTE: This includes radial binning (round(r) gives thick shells).")

    # ---- True angular anisotropy: group by exact r^2 (integer) ----
    # Sites with same x^2+y^2+z^2 have the same exact radius.
    # Spread within such a group measures pure angular anisotropy.
    r2_int = np.sum(lat.sites**2, axis=1)
    unique_r2 = np.unique(r2_int)

    print(f"\n  True angular anisotropy (grouping by exact r^2):")
    print(f"  {'r^2':>5s}  {'r':>6s}  {'count':>5s}  {'mean_Phi':>12s}  "
          f"{'spread':>12s}  {'spread/|mean|':>13s}")
    print(f"  {'-'*70}")

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
        rel = spr / abs(mean_v)
        r_exact = np.sqrt(float(r2))
        angular_spreads.append(rel)
        if count >= 6:  # only print substantial groups
            print(f"  {r2:5d}  {r_exact:6.2f}  {count:5d}  {mean_v:12.6e}  "
                  f"{spr:12.6e}  {rel:13.4e}")

    if angular_spreads:
        max_angular = max(angular_spreads)
        median_angular = np.median(angular_spreads)
        print(f"\n  Max angular anisotropy: {max_angular:.4e}")
        print(f"  Median angular anisotropy: {median_angular:.4e}")
        print(f"  => {'GOOD' if max_angular < 0.10 else 'MODERATE' if max_angular < 0.30 else 'SIGNIFICANT'} "
              f"spherical symmetry at fixed r")
    else:
        max_angular = max_rel_spread
        median_angular = max_rel_spread

    return {
        'mean_Phi': mean_Phi, 'std_Phi': std_Phi,
        'spread_Phi': spread_Phi, 'lapse_mean': lapse_mean,
        'max_rel_spread': max_rel_spread,
        'max_angular_anisotropy': max_angular,
        'median_angular_anisotropy': median_angular,
    }


# ============================================================================
# GM extraction from shell-averaged potential
# ============================================================================

def extract_gm(mean_Phi, cstar_sq, n_core):
    """Extract GM from shell-averaged Newtonian potential."""
    N = len(mean_Phi)
    r = np.arange(N, dtype=float)
    r[0] = 1e-10  # avoid /0 for shell 0
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
# Summary plots
# ============================================================================

def make_plots(lat, Phi, sym, source, n_core, cstar_sq, beta0_t0, R_max):
    """Create summary figure for per-site solver."""
    N_shell = lat.N_shell
    r = np.arange(N_shell, dtype=float)
    mean_Phi = sym['mean_Phi']

    rs, GM = extract_gm(mean_Phi, cstar_sq, n_core)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # (a) Shell-averaged potential vs Newtonian
    ax = axes[0, 0]
    ax.plot(r[1:], mean_Phi[1:] / cstar_sq, 'bo-', ms=4, lw=1.5,
            label='Per-site (shell avg)')
    if rs > 0:
        R = r[-1]
        phi_newt = (rs / 2.0) * (1.0 / R - 1.0 / r[1:])
        ax.plot(r[1:], phi_newt, 'k--', lw=1.2, alpha=0.7,
                label=f'Newtonian ($r_s={rs:.3f}$)')
    ax.set_xlabel('Shell $n$')
    ax.set_ylabel(r'$\langle\Phi\rangle / c_*^2$')
    ax.set_title(rf'(a) Shell-averaged potential ($\beta_0 t_0={beta0_t0}$)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (b) Lapse profile
    ax = axes[0, 1]
    ax.plot(r, sym['lapse_mean'], 'bo-', ms=4, lw=1.5, label='Per-site (shell avg)')
    if rs > 0:
        lapse_schw = np.sqrt(np.maximum(1.0 - rs / np.maximum(r, 0.5), 0.0))
        ax.plot(r[1:], lapse_schw[1:], 'k--', lw=1.2, alpha=0.7,
                label=rf'$\sqrt{{1-r_s/r}}$')
    ax.set_xlabel('Shell $n$')
    ax.set_ylabel('$N(r)$')
    ax.set_title(rf'(b) Lapse ($\beta_0 t_0={beta0_t0}$)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # (c) Spherical symmetry test: spread per shell
    ax = axes[0, 2]
    rel_spread = sym['spread_Phi'] / np.maximum(np.abs(mean_Phi), 1e-30)
    ax.semilogy(r[1:-1], rel_spread[1:-1], 'ro-', ms=4, lw=1.5)
    ax.axhline(0.05, color='gray', ls=':', lw=1, alpha=0.7, label='5% threshold')
    ax.set_xlabel('Shell $n$')
    ax.set_ylabel('Spread / $|\\langle\\Phi\\rangle|$')
    ax.set_title('(c) Departure from spherical symmetry')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (d) Phi on a slice through the lattice
    ax = axes[1, 0]
    # Extract z=0 plane
    z0_mask = lat.sites[:, 2] == 0
    if z0_mask.any():
        x_slice = lat.sites[z0_mask, 0]
        y_slice = lat.sites[z0_mask, 1]
        phi_slice = Phi[z0_mask] / cstar_sq
        sc = ax.scatter(x_slice, y_slice, c=phi_slice, cmap='RdBu_r',
                       s=15, edgecolors='none')
        plt.colorbar(sc, ax=ax, label=r'$\Phi/c_*^2$')
    ax.set_xlabel('$x/a$')
    ax.set_ylabel('$y/a$')
    ax.set_title(r'(d) $\Phi/c_*^2$ in $z=0$ plane')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)

    # (e) Source localization (shell-averaged)
    ax = axes[1, 1]
    src_shell = np.zeros(N_shell)
    for n in range(N_shell):
        idx = lat.shell_sites[n]
        if len(idx) > 0:
            src_shell[n] = np.mean(np.abs(source[idx]))
    nz = src_shell > 1e-30
    if nz.any():
        ax.semilogy(r[nz], src_shell[nz], 'bo-', ms=4, lw=1.5)
    ax.axvline(n_core - 0.5, color='gray', ls=':', lw=1, alpha=0.7,
               label=f'Core boundary')
    ax.set_xlabel('Shell $n$')
    ax.set_ylabel(r'$\langle|{\rm source}|\rangle$')
    ax.set_title('(e) Source localization')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (f) Angular distribution at a mid-radius shell
    ax = axes[1, 2]
    mid_shell = min(N_shell // 2, N_shell - 2)
    mid_idx = lat.shell_sites[mid_shell]
    if len(mid_idx) > 1:
        phi_mid = Phi[mid_idx] / cstar_sq
        ax.hist(phi_mid, bins=max(10, len(mid_idx) // 5),
                color='steelblue', edgecolor='white', alpha=0.8)
        ax.axvline(np.mean(phi_mid), color='red', ls='--', lw=1.5,
                   label=f'mean={np.mean(phi_mid):.6e}')
        ax.axvline(np.mean(phi_mid) + np.std(phi_mid), color='red',
                   ls=':', lw=1, alpha=0.5)
        ax.axvline(np.mean(phi_mid) - np.std(phi_mid), color='red',
                   ls=':', lw=1, alpha=0.5)
    ax.set_xlabel(r'$\Phi/c_*^2$')
    ax.set_ylabel('Count')
    ax.set_title(f'(f) Angular distribution at shell {mid_shell} '
                 f'($G_n={lat.G_n[mid_shell]}$)')
    ax.legend(fontsize=8)

    fig.suptitle(f'3D per-site solver (R={R_max}, {lat.N_sites} sites, '
                 rf'$\beta_0 t_0={beta0_t0}$)',
                 fontsize=14, y=1.01)
    plt.tight_layout()

    path = os.path.join(FIGDIR, f"persite_3d_R{R_max}_bt{beta0_t0:.2f}.pdf")
    plt.savefig(path, bbox_inches='tight')
    plt.savefig(path.replace('.pdf', '.png'), bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, default=10,
                    help="Lattice radius (default 10)")
    ap.add_argument("--n-core", type=int, default=3)
    ap.add_argument("--cstar-sq", type=float, default=0.5)
    ap.add_argument("--betaV0", type=float, default=0.003)
    ap.add_argument("--bt0", type=float, default=0.10,
                    help="beta0*t0 (default 0.10)")
    ap.add_argument("--gpu", action="store_true",
                    help="Use GPU for eigh (source computation)")
    ap.add_argument("--max-iter", type=int, default=500)
    ap.add_argument("--mi", action="store_true",
                    help="Use MI-based conductances (fully self-consistent, requires GPU)")
    args = ap.parse_args()

    R_MAX = args.R
    t0_hop = 1.0
    cstar_sq = args.cstar_sq
    n_core = args.n_core
    beta0V0 = args.betaV0
    bt0 = args.bt0
    beta0 = bt0 / t0_hop
    V0 = beta0V0 / beta0

    print("=" * 80)
    print(f"3D PER-SITE SOLVER (R={R_MAX}, bt0={bt0})")
    print(f"  n_core={n_core}, c*^2={cstar_sq}, beta0*V0={beta0V0}, V0={V0:.6f}")
    print("=" * 80)

    lat = CubicLattice3D(R_MAX)

    # Mark core sites
    for n in range(min(n_core, lat.N_shell)):
        lat.core_mask[lat.shell_sites[n]] = True
    print(f"  Core sites: {np.sum(lat.core_mask)}", flush=True)

    # Compute per-site source
    print(f"\n--- Source computation ---", flush=True)
    if args.gpu:
        source, rho_bg, rho_tgt, core_idx = compute_persite_source_gpu(
            lat, t0_hop, V0, n_core, beta0)
    else:
        source, rho_bg, rho_tgt, core_idx = compute_persite_source(
            lat, t0_hop, V0, n_core, beta0)

    # Picard with analytic kappa
    print(f"\n--- Picard iteration (analytic kappa) ---", flush=True)
    t_s = time.time()
    Phi_analytic = picard_persite(lat, source, t0_hop, beta0, cstar_sq,
                                  max_iter=args.max_iter, mixing=0.3,
                                  tol=1e-8, verbose=True)
    dt = time.time() - t_s
    print(f"  Analytic Picard done in {dt:.1f}s", flush=True)

    Phi = Phi_analytic

    # MI-based Picard (fully self-consistent)
    if args.mi:
        print(f"\n--- MI-based Picard (fully self-consistent) ---", flush=True)
        t_s = time.time()
        Phi_mi = picard_persite_mi_gpu(lat, t0_hop, V0, n_core, beta0, cstar_sq,
                                        Phi_init=Phi_analytic, max_iter=100,
                                        mixing=0.3, tol=1e-6, verbose=True)
        dt = time.time() - t_s
        print(f"  MI Picard done in {dt:.1f}s", flush=True)

        # Compare analytic vs MI
        dPhi = Phi_mi - Phi_analytic
        rel = np.max(np.abs(dPhi)) / max(np.max(np.abs(Phi_analytic)), 1e-30)
        print(f"\n  Analytic vs MI comparison:")
        print(f"    max|dPhi|       = {np.max(np.abs(dPhi)):.6e}")
        print(f"    max|dPhi/Phi|   = {rel:.6e}")
        lapse_an = 1.0 + Phi_analytic / cstar_sq
        lapse_mi = 1.0 + Phi_mi / cstar_sq
        print(f"    min(N) analytic = {lapse_an.min():.6f}")
        print(f"    min(N) MI       = {lapse_mi.min():.6f}")

        Phi = Phi_mi

    # Analyze spherical symmetry
    print(f"\n--- Spherical symmetry analysis ---", flush=True)
    sym = analyze_symmetry(lat, Phi, cstar_sq)

    # Extract GM
    rs, GM = extract_gm(sym['mean_Phi'], cstar_sq, n_core)
    print(f"\n  rs = {rs:.6f}, GM = {GM:.8f}")

    # Make plots
    print(f"\n--- Plots ---", flush=True)
    make_plots(lat, Phi, sym, source, n_core, cstar_sq, bt0, R_MAX)

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"  3D cubic lattice: R={R_MAX}, {lat.N_sites} sites, {lat.N_shell} shells")
    print(f"  Temperature: beta0*t0 = {bt0}")
    print(f"  rs = {rs:.6f}, GM = {GM:.8f}")
    print(f"  Max relative spread: {sym['max_rel_spread']:.4e}")
    print(f"  Spherical symmetry: "
          f"{'YES (<5%)' if sym['max_rel_spread'] < 0.05 else 'NO (>5%)'}")


if __name__ == "__main__":
    main()
