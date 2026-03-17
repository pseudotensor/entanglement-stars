#!/usr/bin/env python3
"""
GPU-accelerated computation of exact two-state closure solutions.

JAX port of full_numerics/compute_exact_data.py — runs all phases on GPU
using jacfwd for exact Jacobians with host-side Newton loops.
Adds a TOV phase (n_core scan at fixed mass with force balance).

Phases:
  A/A2/A3 — core/supercritical/proxy+analytic solutions
  B — floor-independence study
  B2 — temperature sweep (bt0-continuation)
  C — V0 sweep (exact + proxy + analytic)
  TOV — n_core scan at fixed mass, I_grav + I_mat + I_tot

Output: fermion/numerical/data/exact_solutions.npz

Run:  python3 compute_exact_data_gpu.py
"""

import os
import sys
import time
import numpy as np

os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, jacfwd
from scipy.linalg import solve_banded

print(f"JAX devices: {jax.devices()}")
print(f"JAX default backend: {jax.default_backend()}")


# ═══════════════════════════════════════════════════════════════════
# JAX PRIMITIVES
# ═══════════════════════════════════════════════════════════════════

def binary_entropy(x):
    x = jnp.clip(x, 1e-30, 1.0 - 1e-30)
    return -x * jnp.log(x) - (1.0 - x) * jnp.log(1.0 - x)


def build_tridiag(diag, off):
    return jnp.diag(diag) + jnp.diag(off, 1) + jnp.diag(off, -1)


@jit
def bond_correlations(diag, off, beta0):
    """G = (exp(beta0*h) + I)^{-1}, return (G_diag, G_super)."""
    H = build_tridiag(diag, off)
    evals, evecs = jnp.linalg.eigh(H)
    be = jnp.clip(beta0 * evals, -500.0, 500.0)
    f = 1.0 / (jnp.exp(be) + 1.0)
    vf = evecs * f[None, :]
    G_diag = jnp.sum(evecs * vf, axis=1)
    G_super = jnp.sum(evecs[:-1] * vf[1:], axis=1)
    return G_diag, G_super


@jit
def mi_from_bond(G_diag, G_super):
    """MI per bond from 2x2 correlation-matrix blocks."""
    a = G_diag[:-1]
    d = G_diag[1:]
    b = G_super
    tr = a + d
    det = a * d - b**2
    disc = jnp.maximum(tr**2 - 4.0 * det, 0.0)
    sqrt_disc = jnp.sqrt(disc + 1e-30)
    lam1 = 0.5 * (tr + sqrt_disc)
    lam2 = 0.5 * (tr - sqrt_disc)
    return (binary_entropy(a) + binary_entropy(d)
            - binary_entropy(lam1) - binary_entropy(lam2))


@jit
def bkm_bond_cov(diag, off, beta0):
    """BKM covariance of the bond current operator per bond (GPU).

    Returns shape (N-1,) array of covariance values.
    Same cost as bond_correlations but gives the exact KMS edge weight.
    """
    H = build_tridiag(diag, off)
    evals, evecs = jnp.linalg.eigh(H)
    be = jnp.clip(beta0 * evals, -500.0, 500.0)
    f = 1.0 / (jnp.exp(be) + 1.0)

    # BKM kernel K[α,β] = f_α(1-f_β) · φ(β₀(ε_α-ε_β))
    diff = beta0 * (evals[:, None] - evals[None, :])
    # φ(x) = expm1(x)/x, φ(0) = 1
    phi = jnp.where(jnp.abs(diff) > 1e-12,
                     jnp.expm1(diff) / jnp.where(jnp.abs(diff) > 1e-12, diff, 1.0),
                     1.0)
    K = f[:, None] * (1.0 - f[None, :]) * phi  # (M, M)

    U_n = evecs[:-1, :]   # (N-1, M)
    U_n1 = evecs[1:, :]   # (N-1, M)
    Un_sq = U_n**2
    Un1_sq = U_n1**2
    cross = U_n * U_n1

    # cov[b] = Σ_{α,β} (U_n1[b,α]U_n[b,β] - U_n[b,α]U_n1[b,β])² K[α,β]
    t1 = jnp.sum((Un1_sq @ K) * Un_sq, axis=1)
    t2 = jnp.sum((Un_sq @ K) * Un1_sq, axis=1)
    t3 = jnp.sum((cross @ K) * cross, axis=1)
    return t1 + t2 - 2.0 * t3


def compute_kms_bg(N, t0, beta0):
    """Background BKM covariance per bond for uniform chain (Phi=0)."""
    off = -t0 * jnp.ones(N - 1, dtype=jnp.float64)
    diag = jnp.zeros(N, dtype=jnp.float64)
    return bkm_bond_cov(diag, off, beta0)


@jit
def laplacian_action(kappa, Phi):
    """(L_kappa Phi)_n = sum_{m~n} kappa_{nm}(Phi_n - Phi_m)."""
    N = Phi.shape[0]
    out = jnp.zeros(N, dtype=jnp.float64)
    out = out.at[:-1].add(kappa * (Phi[:-1] - Phi[1:]))
    out = out.at[1:].add(kappa * (Phi[1:] - Phi[:-1]))
    return out


# ═══════════════════════════════════════════════════════════════════
# PHYSICS (all pure functions, JIT-compatible)
# ═══════════════════════════════════════════════════════════════════

def compute_mi_bg(N, t0, beta0):
    """Background MI per bond for uniform chain (Phi=0)."""
    off = -t0 * jnp.ones(N - 1, dtype=jnp.float64)
    diag = jnp.zeros(N, dtype=jnp.float64)
    Gd, Gs = bond_correlations(diag, off, beta0)
    return mi_from_bond(Gd, Gs)


def compute_rho_bg_tgt(N, t0, V0, n_core, beta0):
    """Precompute flat-space background and target energy profiles."""
    h0_diag = jnp.zeros(N, dtype=jnp.float64)
    h0_off = -t0 * jnp.ones(N - 1, dtype=jnp.float64)

    hV_diag = h0_diag.at[:n_core].add(V0)

    g = 4.0 * jnp.pi * jnp.arange(1, N + 1, dtype=jnp.float64)**2

    # Background
    Gd_bg, Gs_bg = bond_correlations(h0_diag, h0_off, beta0)
    rho_bg = jnp.zeros(N, dtype=jnp.float64)
    rho_bg = rho_bg.at[:-1].set(2.0 * t0 * Gs_bg * g[:-1])

    # Target
    Gd_tgt, Gs_tgt = bond_correlations(hV_diag, h0_off, beta0)
    rho_tgt = jnp.zeros(N, dtype=jnp.float64)
    rho_tgt = rho_tgt.at[:-1].set(2.0 * t0 * Gs_tgt * g[:-1])
    rho_tgt = rho_tgt.at[:n_core].add(V0 * Gd_tgt[:n_core] * g[:n_core])

    return rho_bg, rho_tgt


def _make_physics_fns(N, t0, n_core, beta0, cstar_sq):
    """Create closure-based physics functions with concrete N.

    V0 is passed as an explicit argument (not closure) so JAX compiles once.
    Returns (conductances_fn, rho_sigma_fn, rho_tgt_smeared_fn) — all JIT-safe.
    """
    # Pre-build arrays with concrete N (outside JIT trace)
    n_arr = jnp.arange(1, N + 1, dtype=jnp.float64)
    g = 4.0 * jnp.pi * n_arr**2
    diag0 = jnp.zeros(N, dtype=jnp.float64)
    core_mask = (jnp.arange(N) < n_core).astype(jnp.float64)

    def conductances_mi_fn(Phi, mi_bg):
        """MI-based conductances (legacy): κ_n = g_n t₀² MI/MI_bg."""
        lapse = 1.0 + Phi / cstar_sq
        Nbar = 0.5 * (lapse[:-1] + lapse[1:])
        # abs(Nbar): Newton-step safeguard; Nbar>0 on positive-lapse branch,
        # so abs() is a no-op for converged solutions.
        off = -t0 * jnp.abs(Nbar)
        Gd, Gs = bond_correlations(diag0, off, beta0)
        mi = mi_from_bond(Gd, Gs)
        mi_ratio = mi / jnp.maximum(mi_bg, 1e-30)
        return g[:-1] * t0**2 * mi_ratio

    def conductances_fn(Phi, kms_bg):
        """KMS conductances: κ_n = g_n t₀² N̄² cov/cov_bg."""
        lapse = 1.0 + Phi / cstar_sq
        Nbar = 0.5 * (lapse[:-1] + lapse[1:])
        off = -t0 * jnp.abs(Nbar)
        cov = bkm_bond_cov(diag0, off, beta0)
        cov_ratio = cov / jnp.maximum(kms_bg, 1e-30)
        return g[:-1] * t0**2 * Nbar**2 * cov_ratio

    def rho_sigma_fn(Phi):
        """Energy profile of reconstructed state sigma[Phi]."""
        lapse = 1.0 + Phi / cstar_sq
        Nbar = 0.5 * (lapse[:-1] + lapse[1:])
        off = -t0 * jnp.abs(Nbar)
        _, Gs = bond_correlations(diag0, off, beta0)
        prof = jnp.zeros(N, dtype=jnp.float64)
        prof = prof.at[:-1].set(2.0 * t0 * jnp.abs(Nbar) * Gs * g[:-1])
        return prof

    def rho_tgt_smeared_fn(Phi, V0):
        """Target energy profile with lapse-smeared hopping AND V0."""
        lapse = 1.0 + Phi / cstar_sq
        Nbar = 0.5 * (lapse[:-1] + lapse[1:])
        off = -t0 * jnp.abs(Nbar)
        diag = diag0 + V0 * core_mask
        Gd, Gs = bond_correlations(diag, off, beta0)
        prof = jnp.zeros(N, dtype=jnp.float64)
        prof = prof.at[:-1].set(2.0 * t0 * jnp.abs(Nbar) * Gs * g[:-1])
        prof = prof.at[:n_core].add(V0 * Gd[:n_core] * g[:n_core])
        return prof

    return conductances_fn, rho_sigma_fn, rho_tgt_smeared_fn


# ═══════════════════════════════════════════════════════════════════
# RESIDUALS
# ═══════════════════════════════════════════════════════════════════

def make_residual_proxy(N, t0, V0, n_core, beta0, cstar_sq):
    """Return a JIT'd proxy residual: fixed source, KMS conductances."""
    kms_bg = compute_kms_bg(N, t0, beta0)
    rho_bg, rho_tgt = compute_rho_bg_tgt(N, t0, V0, n_core, beta0)
    pref = beta0 / cstar_sq
    source = pref * (rho_bg - rho_tgt)
    cond_fn, _, _ = _make_physics_fns(N, t0, n_core, beta0, cstar_sq)

    @jit
    def residual(Phi):
        kappa = cond_fn(Phi, kms_bg)
        lhs = laplacian_action(kappa, Phi)
        F = lhs - source
        F = F.at[N - 1].set(Phi[N - 1])
        return F

    return residual, kms_bg, rho_bg, rho_tgt


def make_residual_fixedpoint(N, t0, n_core, beta0, cstar_sq):
    """Return a JIT'd fixedpoint residual: self-consistent source.

    V0 is an explicit argument (not closure) so JAX compiles once for all V0.

    The LHS (Laplacian + EL correction) is computed as grad of the energy
    functional E = (1/2) sum_b kappa_b(Phi) * (Phi_b - Phi_{b+1})^2.
    This gives an exact Jacobian when differentiated with jacfwd,
    unlike the old FD + stop_gradient approach.
    """
    kms_bg = compute_kms_bg(N, t0, beta0)
    cond_fn, rho_sigma_fn, rho_tgt_fn = _make_physics_fns(
        N, t0, n_core, beta0, cstar_sq)

    def energy_lhs(Phi):
        """Conductance energy: (1/2) sum_b kappa_b(Phi) * (dPhi_b)^2."""
        kappa = cond_fn(Phi, kms_bg)
        dPhi = Phi[:-1] - Phi[1:]
        return 0.5 * jnp.sum(kappa * dPhi**2)

    grad_energy = jax.grad(energy_lhs)

    @jit
    def residual(Phi, V0):
        # LHS = grad of energy = Laplacian_action + EL correction (exact)
        lhs = grad_energy(Phi)

        pref = beta0 / cstar_sq
        rs = rho_sigma_fn(Phi)
        rt = rho_tgt_fn(Phi, V0)
        rhs = pref * (rs - rt)
        F = lhs - rhs
        F = F.at[N - 1].set(Phi[N - 1])
        return F

    return residual, kms_bg


def make_residual_tempsweep(N, t0, n_core, cstar_sq):
    """Fully parametric residual(Phi, V0, beta0) for temperature sweep.

    Full EL correction via jax.grad of the conductance energy functional.
    V0 and beta0 are traced args so JAX compiles once for all combinations.
    """
    n_arr = jnp.arange(1, N + 1, dtype=jnp.float64)
    g = 4.0 * jnp.pi * n_arr**2
    diag0 = jnp.zeros(N, dtype=jnp.float64)
    core_mask = (jnp.arange(N) < n_core).astype(jnp.float64)

    def _conductances(Phi, beta0):
        """KMS conductances as function of (Phi, beta0)."""
        lapse = 1.0 + Phi / cstar_sq
        Nbar = 0.5 * (lapse[:-1] + lapse[1:])
        off = -t0 * jnp.abs(Nbar)
        kms_bg = bkm_bond_cov(diag0, -t0 * jnp.ones(N - 1, dtype=jnp.float64), beta0)
        cov = bkm_bond_cov(diag0, off, beta0)
        cov_ratio = cov / jnp.maximum(kms_bg, 1e-30)
        return g[:-1] * t0**2 * Nbar**2 * cov_ratio

    def _energy(Phi, beta0):
        """Conductance energy: (1/2) sum_b kappa_b(Phi) * (dPhi_b)^2."""
        kappa = _conductances(Phi, beta0)
        dPhi = Phi[:-1] - Phi[1:]
        return 0.5 * jnp.sum(kappa * dPhi**2)

    _grad_energy = jax.grad(_energy, argnums=0)

    @jit
    def residual(Phi, V0, beta0):
        # LHS = grad of energy functional (includes EL correction exactly)
        lhs = _grad_energy(Phi, beta0)

        # Self-consistent source
        lapse = 1.0 + Phi / cstar_sq
        Nbar = 0.5 * (lapse[:-1] + lapse[1:])
        off = -t0 * jnp.abs(Nbar)

        _, Gs_sig = bond_correlations(diag0, off, beta0)
        rho_sig = jnp.zeros(N, dtype=jnp.float64)
        rho_sig = rho_sig.at[:-1].set(2.0 * t0 * jnp.abs(Nbar) * Gs_sig * g[:-1])

        diag_tgt = diag0 + V0 * core_mask
        Gd_tgt, Gs_tgt = bond_correlations(diag_tgt, off, beta0)
        rho_tgt = jnp.zeros(N, dtype=jnp.float64)
        rho_tgt = rho_tgt.at[:-1].set(2.0 * t0 * jnp.abs(Nbar) * Gs_tgt * g[:-1])
        rho_tgt = rho_tgt.at[:n_core].add(V0 * Gd_tgt[:n_core] * g[:n_core])

        pref = beta0 / cstar_sq
        F = lhs - pref * (rho_sig - rho_tgt)
        F = F.at[N - 1].set(Phi[N - 1])
        return F

    return residual


def make_newton_solver_3arg(residual_fn, N, max_iter=60, tol=1e-8):
    """Newton solver for residual_fn(Phi, V0, beta0) -> F.

    jacfwd differentiates w.r.t. argnums=0 (Phi only).
    """
    resid_jit = jit(residual_fn)
    jac_jit = jit(jacfwd(residual_fn, argnums=0))

    def solve(Phi_init, V0, beta0):
        Phi = jnp.array(Phi_init)
        V0_jax = jnp.float64(V0)
        b0_jax = jnp.float64(beta0)
        stagnant = 0

        for it in range(max_iter):
            F = resid_jit(Phi, V0_jax, b0_jax)
            F = jax.block_until_ready(F)
            norm_F = float(jnp.max(jnp.abs(F)))

            if norm_F < tol:
                return Phi, True, it, norm_F

            J = jac_jit(Phi, V0_jax, b0_jax)
            J = jax.block_until_ready(J)

            reg = 1e-10 * jnp.maximum(jnp.abs(jnp.diag(J)), 1e-10)
            J_reg = J + jnp.diag(reg)
            dPhi = jnp.linalg.solve(J_reg, -F)

            alpha = 1.0
            best_Phi = Phi
            best_norm = norm_F
            improved = False
            for _ in range(6):
                Phi_trial = Phi + alpha * dPhi
                F_trial = resid_jit(Phi_trial, V0_jax, b0_jax)
                norm_trial = float(jnp.max(jnp.abs(jax.block_until_ready(F_trial))))
                if norm_trial < best_norm:
                    best_Phi = Phi_trial
                    best_norm = norm_trial
                    improved = True
                    break
                alpha *= 0.5

            if improved:
                Phi = best_Phi
                stagnant = 0
            else:
                Phi = Phi + 0.01 * dPhi
                stagnant += 1
                if stagnant >= 5:
                    break

            if best_norm > 1e6 or float(jnp.any(jnp.isnan(Phi))):
                break

        F = resid_jit(Phi, V0_jax, b0_jax)
        norm_F = float(jnp.max(jnp.abs(jax.block_until_ready(F))))
        return Phi, norm_F < tol, max_iter, norm_F

    return solve


# ═══════════════════════════════════════════════════════════════════
# NEWTON SOLVER (host-side loop, JIT'd residual + jacfwd per call)
# ═══════════════════════════════════════════════════════════════════

def make_newton_solver(residual_fn, N, max_iter=60, tol=1e-8):
    """Create a Newton solver using host-side Python loop.

    residual_fn(Phi) -> F  [1-arg, used by proxy solver]

    Uses individually JIT'd residual and Jacobian calls to avoid
    the massive XLA compilation cost of lax.while_loop + jacfwd.

    Returns: solve(Phi_init) -> (Phi_final, converged, iters, final_res)
    """
    resid_jit = jit(residual_fn)
    jac_jit = jit(jacfwd(residual_fn))

    def solve(Phi_init):
        Phi = jnp.array(Phi_init)
        stagnant = 0

        for it in range(max_iter):
            F = resid_jit(Phi)
            F = jax.block_until_ready(F)
            norm_F = float(jnp.max(jnp.abs(F)))

            if norm_F < tol:
                return Phi, True, it, norm_F

            J = jac_jit(Phi)
            J = jax.block_until_ready(J)

            # Levenberg-Marquardt regularization
            reg = 1e-10 * jnp.maximum(jnp.abs(jnp.diag(J)), 1e-10)
            J_reg = J + jnp.diag(reg)
            dPhi = jnp.linalg.solve(J_reg, -F)

            # Line search
            alpha = 1.0
            best_Phi = Phi
            best_norm = norm_F
            improved = False
            for _ in range(6):
                Phi_trial = Phi + alpha * dPhi
                F_trial = resid_jit(Phi_trial)
                norm_trial = float(jnp.max(jnp.abs(jax.block_until_ready(F_trial))))
                if norm_trial < best_norm:
                    best_Phi = Phi_trial
                    best_norm = norm_trial
                    improved = True
                    break
                alpha *= 0.5

            if improved:
                Phi = best_Phi
                stagnant = 0
            else:
                Phi = Phi + 0.01 * dPhi
                stagnant += 1
                if stagnant >= 5:
                    break

            if best_norm > 1e6 or float(jnp.any(jnp.isnan(Phi))):
                break

        F = resid_jit(Phi)
        norm_F = float(jnp.max(jnp.abs(jax.block_until_ready(F))))
        return Phi, norm_F < tol, max_iter, norm_F

    return solve


def make_newton_solver_2arg(residual_fn, N, max_iter=60, tol=1e-8):
    """Newton solver for residual_fn(Phi, V0) -> F.

    V0 is passed as a traced argument so JAX compiles once for all V0.
    jacfwd differentiates w.r.t. argnums=0 (Phi only).

    Returns: solve(Phi_init, V0) -> (Phi_final, converged, iters, final_res)
    """
    resid_jit = jit(residual_fn)
    jac_jit = jit(jacfwd(residual_fn, argnums=0))

    def solve(Phi_init, V0):
        Phi = jnp.array(Phi_init)
        V0_jax = jnp.float64(V0)
        stagnant = 0

        for it in range(max_iter):
            F = resid_jit(Phi, V0_jax)
            F = jax.block_until_ready(F)
            norm_F = float(jnp.max(jnp.abs(F)))

            if norm_F < tol:
                return Phi, True, it, norm_F

            J = jac_jit(Phi, V0_jax)
            J = jax.block_until_ready(J)

            # Levenberg-Marquardt regularization
            reg = 1e-10 * jnp.maximum(jnp.abs(jnp.diag(J)), 1e-10)
            J_reg = J + jnp.diag(reg)
            dPhi = jnp.linalg.solve(J_reg, -F)

            # Line search
            alpha = 1.0
            best_Phi = Phi
            best_norm = norm_F
            improved = False
            for _ in range(6):
                Phi_trial = Phi + alpha * dPhi
                F_trial = resid_jit(Phi_trial, V0_jax)
                norm_trial = float(jnp.max(jnp.abs(jax.block_until_ready(F_trial))))
                if norm_trial < best_norm:
                    best_Phi = Phi_trial
                    best_norm = norm_trial
                    improved = True
                    break
                alpha *= 0.5

            if improved:
                Phi = best_Phi
                stagnant = 0
            else:
                Phi = Phi + 0.01 * dPhi
                stagnant += 1
                if stagnant >= 5:
                    break  # not making progress

            # Divergence check
            if best_norm > 1e6 or float(jnp.any(jnp.isnan(Phi))):
                break

        F = resid_jit(Phi, V0_jax)
        norm_F = float(jnp.max(jnp.abs(jax.block_until_ready(F))))
        return Phi, norm_F < tol, max_iter, norm_F

    return solve


# ═══════════════════════════════════════════════════════════════════
# HIGH-LEVEL SOLVERS
# ═══════════════════════════════════════════════════════════════════

def solve_proxy_V0_continuation(N, t0, V0_target, n_core, beta0, cstar_sq,
                                 n_steps=12, tol=1e-8, Phi_seed=None,
                                 verbose=True):
    """Solve proxy equation via V0 continuation from 0 to V0_target."""
    V0_values = V0_target * jnp.linspace(1.0/n_steps, 1.0, n_steps)
    Phi = jnp.zeros(N, dtype=jnp.float64) if Phi_seed is None else jnp.array(Phi_seed)

    for iv, V0_step in enumerate(V0_values):
        V0_s = float(V0_step)
        resid_fn, kms_bg, rho_bg, rho_tgt = make_residual_proxy(
            N, t0, V0_s, n_core, beta0, cstar_sq)
        solver = make_newton_solver(resid_fn, N, max_iter=40, tol=tol)
        Phi, conv, nit, res = solver(Phi)
        Phi = jax.block_until_ready(Phi)

    if verbose:
        lapse = 1.0 + Phi / cstar_sq
        print(f"    proxy V0={V0_target:.5f}: |F|={float(res):.3e}, "
              f"min(N)={float(jnp.min(lapse)):.6f}, conv={bool(conv)}",
              flush=True)

    # Return kms_bg as mi_bg for backward-compatible caller signatures
    return Phi, bool(conv), float(res), kms_bg, rho_bg, rho_tgt


def solve_fixedpoint_from_proxy(Phi_proxy, V0, fp_solver, cstar_sq,
                                 verbose=True):
    """Refine proxy solution with self-consistent (fixedpoint) residual.

    fp_solver: pre-built 2-arg Newton solver (compiled once, reused for all V0).
    """
    Phi, conv, nit, res = fp_solver.solve(jnp.array(Phi_proxy), V0)
    Phi = jax.block_until_ready(Phi)

    if verbose:
        lapse = 1.0 + Phi / cstar_sq
        print(f"    exact V0={V0:.5f}: |F|={float(res):.3e}, "
              f"min(N)={float(jnp.min(lapse)):.6f}, conv={bool(conv)}",
              flush=True)

    return Phi, bool(conv), float(res), fp_solver.kms_bg


class FixedpointSolver:
    """Pre-compiled fixedpoint solver that can be reused for any V0."""

    def __init__(self, N, t0, n_core, beta0, cstar_sq, max_iter=60, tol=1e-6):
        self.resid_fn, self.kms_bg = make_residual_fixedpoint(
            N, t0, n_core, beta0, cstar_sq)
        self.solver = make_newton_solver_2arg(
            self.resid_fn, N, max_iter=max_iter, tol=tol)

    def solve(self, Phi_init, V0):
        return self.solver(Phi_init, V0)

    def warmup(self, N):
        """Force JIT compilation with a dummy call."""
        Phi0 = jnp.zeros(N, dtype=jnp.float64)
        V0_dummy = jnp.float64(0.001)
        F = jax.block_until_ready(jit(self.resid_fn)(Phi0, V0_dummy))
        J = jax.block_until_ready(
            jit(jacfwd(self.resid_fn, argnums=0))(Phi0, V0_dummy))
        return F, J


def solve_floored(V0, fp_solver, N, cstar_sq, n_core, t0, beta0,
                  lapse_floor=1e-5, tol=1e-8, Phi_seed=None, verbose=True):
    """Solve with lapse floor clamping (for supercritical V0)."""
    floor_val = -(1.0 - lapse_floor) * cstar_sq

    # Get proxy seed first
    Phi_proxy, _, _, kms_bg, _, _ = solve_proxy_V0_continuation(
        N, t0, V0, n_core, beta0, cstar_sq,
        n_steps=20, tol=1e-8, verbose=False)

    resid_fn = fp_solver.resid_fn
    resid_jit = jit(resid_fn)
    jac_jit = jit(jacfwd(resid_fn, argnums=0))
    V0_jax = jnp.float64(V0)

    Phi = np.array(Phi_proxy)
    Phi = np.maximum(Phi, floor_val)

    for it in range(50):
        clamped = Phi <= floor_val + 1e-13

        F = np.array(resid_jit(jnp.array(Phi), V0_jax))
        F[clamped] = 0.0
        F[-1] = Phi[-1]
        res = np.max(np.abs(F))

        if verbose and (it < 3 or it % 5 == 0 or res < tol):
            lapse = 1.0 + Phi / cstar_sq
            print(f"    Newton {it:3d}: |F|={res:.3e}, "
                  f"min(N)={lapse.min():.6f}, clamped={int(clamped.sum())}")

        if res < tol:
            break

        J = np.array(jac_jit(jnp.array(Phi), V0_jax))
        for n_idx in range(N):
            if clamped[n_idx]:
                J[n_idx, :] = 0.0
                J[:, n_idx] = 0.0
                J[n_idx, n_idx] = 1.0

        try:
            dPhi = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            break

        alpha = 1.0
        for _ in range(30):
            Phi_trial = np.maximum(Phi + alpha * dPhi, floor_val)
            F_trial = np.array(resid_jit(jnp.array(Phi_trial), V0_jax))
            F_trial[Phi_trial <= floor_val + 1e-13] = 0.0
            F_trial[-1] = Phi_trial[-1]
            if np.max(np.abs(F_trial)) < res * 1.1:
                break
            alpha *= 0.5
            if alpha < 1e-6:
                break

        Phi = np.maximum(Phi + alpha * dPhi, floor_val)

    lapse = 1.0 + Phi / cstar_sq
    rs = extract_rs_np(Phi, N, n_core, cstar_sq, a=1.0)
    return {
        "Phi": Phi, "lapse": lapse, "rs": rs,
        "min_N": lapse.min(), "converged": res < tol, "residual": res,
    }


# ═══════════════════════════════════════════════════════════════════
# EXTRACT RS (numpy, host-side)
# ═══════════════════════════════════════════════════════════════════

def extract_rs_np(Phi, N, n_core, cstar_sq, a=1.0):
    """Extract Schwarzschild radius from far-field Phi (vacuum BC)."""
    Phi = np.asarray(Phi)
    r = a * np.arange(1, N + 1, dtype=float)
    R = r[-1]
    i_lo = n_core + 3
    i_hi = min(N - 5, N // 2)
    if i_lo >= i_hi:
        return 0.0
    if not np.any(np.abs(Phi[i_lo:i_hi]) > 1e-15):
        return 0.0
    r_slice = r[i_lo:i_hi]
    GM_est = -Phi[i_lo:i_hi] * r_slice * R / (R - r_slice)
    GM = np.median(GM_est)
    return 2.0 * GM / cstar_sq


# ═══════════════════════════════════════════════════════════════════
# ANALYTIC TWO-STATE (use existing CPU code for comparison overlays)
# ═══════════════════════════════════════════════════════════════════

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from full_numerics.physics_twostate import TwoStateShellModel
from full_numerics.solve_twostate import solve_proxy as _solve_proxy_analytic
from full_numerics.solve_twostate import solve_full as _solve_full_analytic


def solve_analytic_proxy(N, t0, V0, n_core, beta0, cstar_sq, lapse_floor=0.01,
                         Phi_seed=None):
    """Solve proxy equation with analytic (Nbar^2) conductances."""
    model = TwoStateShellModel(N=N, t0=t0, V0=V0, n_core=n_core,
                                beta0=beta0, cstar_sq=cstar_sq)
    Phi, _ = _solve_proxy_analytic(model, Phi_seed, tol=1e-8,
                                    lapse_floor=lapse_floor)
    return Phi


def solve_analytic_full(N, t0, V0, n_core, beta0, cstar_sq, lapse_floor=0.01,
                        Phi_seed=None):
    """Solve full equation with analytic conductances (Picard+Newton)."""
    model = TwoStateShellModel(N=N, t0=t0, V0=V0, n_core=n_core,
                                beta0=beta0, cstar_sq=cstar_sq)
    Phi, _, _ = _solve_full_analytic(model, Phi_seed, tol=1e-8,
                                      lapse_floor=lapse_floor)
    return Phi


# ═══════════════════════════════════════════════════════════════════
# MATTER THERMODYNAMICS (for TOV phase)
# ═══════════════════════════════════════════════════════════════════

def matter_U_S(Phi, N, t0, V0, n_core, beta0, cstar_sq):
    """Compute (U_src, S_src, U_bg, S_bg) for lapse-smeared Hamiltonians."""
    g = 4.0 * np.pi * np.arange(1, N + 1, dtype=float)**2
    Phi_np = np.asarray(Phi)
    lapse = 1.0 + Phi_np / cstar_sq
    Nbar = 0.5 * (lapse[:-1] + lapse[1:])
    off = -t0 * np.abs(Nbar)

    diag_bg = np.zeros(N)
    diag_src = diag_bg.copy()
    diag_src[:n_core] += V0

    # Build full matrices for eigendecomposition
    H_bg = np.diag(diag_bg) + np.diag(off, 1) + np.diag(off, -1)
    H_src = np.diag(diag_src) + np.diag(off, 1) + np.diag(off, -1)

    evals_bg, evecs_bg = np.linalg.eigh(H_bg)
    evals_src, evecs_src = np.linalg.eigh(H_src)

    fb = 1.0 / (np.exp(np.clip(beta0 * evals_bg, -500, 500)) + 1.0)
    fs = 1.0 / (np.exp(np.clip(beta0 * evals_src, -500, 500)) + 1.0)

    # Bond correlations
    vf_bg = evecs_bg * fb[None, :]
    Gd_bg = np.sum(evecs_bg * vf_bg, axis=1)
    Gs_bg = np.sum(evecs_bg[:-1] * vf_bg[1:], axis=1)

    vf_src = evecs_src * fs[None, :]
    Gd_src = np.sum(evecs_src * vf_src, axis=1)
    Gs_src = np.sum(evecs_src[:-1] * vf_src[1:], axis=1)

    # Internal energies
    U_bg = np.sum(2.0 * off * Gs_bg * g[:-1])
    U_src = np.sum(2.0 * off * Gs_src * g[:-1]) + np.sum(V0 * Gd_src[:n_core] * g[:n_core])

    # Entropies — g-weighted to match g-weighted energy (fixes normalization bug)
    def binent(x):
        x = np.clip(x, 1e-30, 1.0 - 1e-30)
        return -x * np.log(x) - (1.0 - x) * np.log(1.0 - x)

    # Use local site entropy S = sum_n g_n h(G_nn) for consistent 3D counting
    S_bg = np.sum(g * binent(Gd_bg))
    S_src = np.sum(g * binent(Gd_src))

    return float(U_src), float(S_src), float(U_bg), float(S_bg)


def mismatch_energy_np(Phi, N, t0, n_core, beta0, cstar_sq, mi_bg):
    """E_mis = (1/2) sum kappa_n (DPhi_n)^2."""
    cond_fn, _, _ = _make_physics_fns(N, t0, n_core, beta0, cstar_sq)
    kappa = np.array(jit(lambda p: cond_fn(p, mi_bg))(jnp.array(Phi)))
    Phi_np = np.asarray(Phi)
    dphi = Phi_np[:-1] - Phi_np[1:]
    return 0.5 * np.sum(kappa * dphi**2)


def conductances_np(Phi, N, t0, n_core, beta0, cstar_sq, mi_bg):
    """Compute kappa as numpy array (for host-side use)."""
    cond_fn, _, _ = _make_physics_fns(N, t0, n_core, beta0, cstar_sq)
    return np.array(jit(lambda p: cond_fn(p, mi_bg))(jnp.array(Phi)))


def shell_g_sum(nc):
    return 4.0 * np.pi * nc * (nc + 1) * (2 * nc + 1) / 6.0


# ═══════════════════════════════════════════════════════════════════
# EMBEDDING
# ═══════════════════════════════════════════════════════════════════

def compute_embedding(Phi, N, cstar_sq, a=1.0, t0=1.0):
    Phi = np.asarray(Phi)
    lapse = 1.0 + Phi / cstar_sq
    r = a * np.arange(1, N + 1, dtype=float)
    Nbar = np.maximum(0.5 * (lapse[:-1] + lapse[1:]), 1e-15)
    dr = np.diff(r)
    d_rho = dr / (t0 * Nbar)
    rho = np.zeros(len(r))
    rho[1:] = np.cumsum(d_rho)
    return rho, r, lapse


def extract_kappa_H(rs, cstar_sq):
    if rs <= 0:
        return np.nan
    return cstar_sq / (2.0 * rs)


# ═══════════════════════════════════════════════════════════════════
# MAIN COMPUTATION
# ═══════════════════════════════════════════════════════════════════

def main():
    t_total = time.time()

    # Parameters
    N = 200
    t0 = 1.0
    n_core = 5
    beta0 = 0.1
    cstar_sq = 0.5
    a = 1.0

    r_arr = a * np.arange(1, N + 1, dtype=float)
    r_bond = 0.5 * (r_arr[:-1] + r_arr[1:])

    # Create shared fixedpoint solver (compiled once, reused for all V0)
    print("Building fixedpoint solver...", flush=True)
    t_warm = time.time()
    fp_solver = FixedpointSolver(N, t0, n_core, beta0, cstar_sq,
                                  max_iter=120, tol=1e-6)
    print(f"  Created in {time.time() - t_warm:.1f}s", flush=True)

    # Warm up: force JIT compilation of residual + jacfwd (one-time cost)
    print("Warming up JAX compilation...", flush=True)
    t_jit = time.time()
    _F, _J = fp_solver.warmup(N)
    print(f"  Fixedpoint jacfwd compiled in {time.time() - t_jit:.1f}s", flush=True)
    print(f"Warmup done in {time.time() - t_warm:.1f}s", flush=True)

    # ═══════════════════════════════════════════════════════════════
    # PHASE A: Core V0 solutions
    # ═══════════════════════════════════════════════════════════════

    # With full EL (jax.grad), V0* ≈ 0.0683 (confirmed by pseudo-arclength
    # continuation). All V0 below this are subcritical.
    V0_core = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.055, 0.057,
               0.06, 0.063, 0.065, 0.066, 0.067]
    V0_extended = []
    V0_supercrit = []

    print("\n" + "=" * 70)
    print("PHASE A: Solving exact two-state for core V0 values")
    print("=" * 70)

    solutions = {}
    Phi_seed = None
    # Track last two exact solutions for linear extrapolation
    exact_history = []  # list of (V0, Phi_np) pairs

    t_a = time.time()
    for V0 in sorted(V0_core + V0_extended):
        print(f"\n--- V0 = {V0:.4f} ---")
        # Proxy with V0 continuation
        Phi_p, conv_p, res_p, mi_bg, rho_bg, rho_tgt = solve_proxy_V0_continuation(
            N, t0, V0, n_core, beta0, cstar_sq,
            n_steps=max(8, int(V0 * 200)), tol=1e-8, Phi_seed=Phi_seed)

        # Build candidate seeds for fixedpoint solver
        seeds = []
        if len(exact_history) >= 2:
            # Linear extrapolation from last two exact solutions
            V0a, Phi_a = exact_history[-2]
            V0b, Phi_b = exact_history[-1]
            Phi_extrap = Phi_b + (V0 - V0b) * (Phi_b - Phi_a) / (V0b - V0a)
            seeds.append(("extrapolation", Phi_extrap))
        if len(exact_history) >= 1:
            seeds.append(("exact prev", exact_history[-1][1]))
        seeds.append(("proxy", np.array(Phi_p)))

        # Try each seed, keep best
        Phi_ex, conv_ex, res_ex, mi_bg_ex = None, False, 1e10, mi_bg
        for label, seed in seeds:
            Phi_try, conv_try, res_try, mi_try = solve_fixedpoint_from_proxy(
                seed, V0, fp_solver, cstar_sq)
            if conv_try:
                Phi_ex, conv_ex, res_ex, mi_bg_ex = Phi_try, True, res_try, mi_try
                print(f"    (converged from {label})", flush=True)
                break
            elif res_try < res_ex:
                Phi_ex, conv_ex, res_ex, mi_bg_ex = Phi_try, False, res_try, mi_try
        mi_bg = mi_bg_ex

        Phi_np = np.array(Phi_ex)
        lapse = 1.0 + Phi_np / cstar_sq
        Nbar = 0.5 * (lapse[:-1] + lapse[1:])
        rs = extract_rs_np(Phi_np, N, n_core, cstar_sq)

        kappa = conductances_np(Phi_np, N, t0, n_core, beta0, cstar_sq, mi_bg)
        g = 4.0 * np.pi * np.arange(1, N + 1, dtype=float)**2
        kappa_flat = g[:-1] * t0**2

        solutions[V0] = {
            "Phi": Phi_np, "lapse": lapse, "Nbar": Nbar,
            "kappa": kappa, "kappa_flat": kappa_flat,
            "rs": rs, "min_N": lapse.min(),
            "converged": conv_ex, "residual": res_ex,
        }
        print(f"  rs={rs:.4f}, min(N)={lapse.min():.6f}, |F|={res_ex:.2e}",
              flush=True)

        if conv_ex:
            Phi_seed = Phi_np
            exact_history.append((V0, Phi_np))
            if len(exact_history) > 3:
                exact_history = exact_history[-3:]
        elif conv_p:
            Phi_seed = np.array(Phi_p)

    dt_a = time.time() - t_a
    print(f"\nPhase A done in {dt_a:.1f}s", flush=True)

    # ═══════════════════════════════════════════════════════════════
    # PHASE A2: Near-critical fine scan + fold test
    # ═══════════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("PHASE A2: Near-critical fine scan + fold test")
    print("=" * 70)

    V0_fine = [0.0675, 0.068]  # near-critical, approaching fold at V0*≈0.0683
    for V0 in V0_fine:
        print(f"\n--- V0 = {V0:.4f} (near-critical) ---")
        Phi_p, conv_p, res_p, mi_bg, rho_bg, rho_tgt = solve_proxy_V0_continuation(
            N, t0, V0, n_core, beta0, cstar_sq,
            n_steps=max(12, int(V0 * 300)), tol=1e-8, Phi_seed=Phi_seed)

        # Build seeds from exact history + proxy
        seeds_a2 = []
        if len(exact_history) >= 2:
            V0a, Phi_a = exact_history[-2]
            V0b, Phi_b = exact_history[-1]
            seeds_a2.append(("extrap", Phi_b + (V0 - V0b) * (Phi_b - Phi_a) / (V0b - V0a)))
        if len(exact_history) >= 1:
            seeds_a2.append(("exact prev", exact_history[-1][1]))
        seeds_a2.append(("proxy", np.array(Phi_p)))

        Phi_ex, conv_ex, res_ex, mi_bg_ex = None, False, 1e10, mi_bg
        for label, seed in seeds_a2:
            Phi_try, conv_try, res_try, mi_try = solve_fixedpoint_from_proxy(
                seed, V0, fp_solver, cstar_sq)
            if conv_try:
                Phi_ex, conv_ex, res_ex, mi_bg_ex = Phi_try, True, res_try, mi_try
                print(f"    (converged from {label})", flush=True)
                break
            elif res_try < res_ex:
                Phi_ex, conv_ex, res_ex, mi_bg_ex = Phi_try, False, res_try, mi_try
        mi_bg = mi_bg_ex

        Phi_np = np.array(Phi_ex)
        lapse = 1.0 + Phi_np / cstar_sq
        Nbar = 0.5 * (lapse[:-1] + lapse[1:])
        rs = extract_rs_np(Phi_np, N, n_core, cstar_sq)
        kappa = conductances_np(Phi_np, N, t0, n_core, beta0, cstar_sq, mi_bg)
        g = 4.0 * np.pi * np.arange(1, N + 1, dtype=float)**2
        solutions[V0] = {
            "Phi": Phi_np, "lapse": lapse, "Nbar": Nbar,
            "kappa": kappa, "kappa_flat": g[:-1] * t0**2,
            "rs": rs, "min_N": lapse.min(),
            "converged": conv_ex, "residual": res_ex,
        }
        print(f"  rs={rs:.4f}, min(N)={lapse.min():.6f}, |F|={res_ex:.2e}",
              flush=True)
        if conv_ex:
            Phi_seed = Phi_np
            exact_history.append((V0, Phi_np))
            if len(exact_history) > 3:
                exact_history = exact_history[-3:]

    # Try V0=0.069 to confirm fold (expect convergence failure or neg lapse)
    print("\n--- V0 = 0.0690 (fold test) ---")
    try:
        Phi_p69, _, _, mi69, _, _ = solve_proxy_V0_continuation(
            N, t0, 0.069, n_core, beta0, cstar_sq,
            n_steps=30, tol=1e-8, Phi_seed=Phi_seed)
        seed69 = exact_history[-1][1] if exact_history else Phi_p69
        Phi_ex69, conv69, res69, _ = solve_fixedpoint_from_proxy(
            seed69, 0.069, fp_solver, cstar_sq)
        if not conv69:
            Phi_ex69b, conv69b, res69b, _ = solve_fixedpoint_from_proxy(
                Phi_p69, 0.069, fp_solver, cstar_sq)
            if res69b < res69:
                Phi_ex69, conv69, res69 = Phi_ex69b, conv69b, res69b
        lapse69 = 1.0 + np.array(Phi_ex69) / cstar_sq
        print(f"  V0=0.069: min(N)={lapse69.min():.6f}, |F|={res69:.2e}, "
              f"conv={conv69}")
        if lapse69.min() < 0 or not conv69:
            print("  => FOLD CONFIRMED: past V0*≈0.0683")
        else:
            print(f"  => Still positive lapse (min(N)={lapse69.min():.4f})")
    except Exception as e:
        print(f"  => FOLD CONFIRMED (solver failed): {e}")

    # ═══════════════════════════════════════════════════════════════
    # PHASE A3: Proxy + analytic for comparison
    # ═══════════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("PHASE A3: Proxy + analytic two-state for comparison overlays")
    print("=" * 70)

    import signal

    class TimeoutError(Exception):
        pass

    def _timeout_handler(signum, frame):
        raise TimeoutError("Analytic solver timed out")

    proxy_solutions = {}
    analytic_solutions = {}
    # Skip near-critical V0 for analytic model (Picard/Newton too slow there,
    # and the analytic Nbar^2 approximation breaks down anyway).
    V0_analytic = [v for v in sorted(set(V0_core + V0_extended)) if v <= 0.04]
    for V0 in V0_analytic:
        print(f"  V0 = {V0:.4f} ... ", end="", flush=True)
        # Use exact GPU solution as seed — dramatically speeds up Picard
        seed = solutions[V0]["Phi"] if V0 in solutions else None

        try:
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(30)  # 30s timeout per V0

            Phi_p = solve_analytic_proxy(N, t0, V0, n_core, beta0, cstar_sq,
                                         Phi_seed=seed)
            Phi_a = solve_analytic_full(N, t0, V0, n_core, beta0, cstar_sq,
                                        Phi_seed=seed)

            signal.alarm(0)  # cancel alarm

            rs_p = extract_rs_np(Phi_p, N, n_core, cstar_sq)
            rs_a = extract_rs_np(Phi_a, N, n_core, cstar_sq)

            proxy_solutions[V0] = {
                "Phi": Phi_p, "lapse": 1 + Phi_p / cstar_sq, "rs": rs_p,
            }
            analytic_solutions[V0] = {
                "Phi": Phi_a, "lapse": 1 + Phi_a / cstar_sq, "rs": rs_a,
            }
            print(f"proxy rs={rs_p:.4f}, analytic rs={rs_a:.4f}", flush=True)
        except (Exception, TimeoutError) as e:
            signal.alarm(0)
            print(f"FAILED: {e}", flush=True)
            # Use proxy-only if full fails
            try:
                signal.alarm(15)
                Phi_p = solve_analytic_proxy(N, t0, V0, n_core, beta0, cstar_sq,
                                             Phi_seed=seed)
                signal.alarm(0)
                rs_p = extract_rs_np(Phi_p, N, n_core, cstar_sq)
                proxy_solutions[V0] = {
                    "Phi": Phi_p, "lapse": 1 + Phi_p / cstar_sq, "rs": rs_p,
                }
            except Exception:
                signal.alarm(0)

    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    print("Phase A3 done.", flush=True)

    # ═══════════════════════════════════════════════════════════════
    # PHASE B: Floor-independence study
    # ═══════════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("PHASE B: Floor-independence study (exact solver)")
    print("=" * 70)

    floors = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00001]
    V0_floor_test = [0.03, 0.05, 0.057, 0.066]

    floor_data = {}
    for V0 in V0_floor_test:
        print(f"\n  V0 = {V0:.3f}")
        floor_data[V0] = []

        for floor in floors:
            res = solve_floored(V0, fp_solver, N, cstar_sq, n_core, t0, beta0,
                                lapse_floor=floor, tol=1e-8, verbose=False)
            floor_data[V0].append({
                "floor": floor, "min_N": res["min_N"],
                "rs": res["rs"], "converged": res["converged"],
            })
            print(f"    floor={floor:.1e}: min(N)={res['min_N']:.8f}, "
                  f"rs={res['rs']:.4f}, conv={res['converged']}")

    # Proxy floor study
    print("\n  Proxy floor study:")
    V0_proxy_floor = [0.05, 0.057, 0.063, 0.066]
    proxy_floor_data = {}
    for V0 in V0_proxy_floor:
        proxy_floor_data[V0] = []
        for floor in floors:
            Phi_pf = solve_analytic_proxy(N, t0, V0, n_core, beta0, cstar_sq,
                                          lapse_floor=floor)
            lapse_pf = 1 + Phi_pf / cstar_sq
            proxy_floor_data[V0].append({"floor": floor, "min_N": lapse_pf.min()})
        minNs_p = [d["min_N"] for d in proxy_floor_data[V0]]
        print(f"    V0={V0:.3f}: min(N) range [{min(minNs_p):.6f}, {max(minNs_p):.6f}]")

    # ═══════════════════════════════════════════════════════════════
    # PHASE B2: Temperature sweep
    # ═══════════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("PHASE B2: Temperature sweep (exact solver, bt0-continuation)")
    print("=" * 70)

    # Build 3-arg solver for temperature sweep (compiled once for all V0, beta0)
    print("  Compiling temperature sweep solver...", flush=True)
    t_ts = time.time()
    ts_resid = make_residual_tempsweep(N, t0, n_core, cstar_sq)
    ts_solver = make_newton_solver_3arg(ts_resid, N, max_iter=50, tol=1e-8)
    # Warmup
    _Phi0 = jnp.zeros(N, dtype=jnp.float64)
    _F0 = jax.block_until_ready(jit(ts_resid)(_Phi0, jnp.float64(0.001), jnp.float64(0.1)))
    _J0 = jax.block_until_ready(jit(jacfwd(ts_resid, argnums=0))(
        _Phi0, jnp.float64(0.001), jnp.float64(0.1)))
    print(f"  Temperature sweep solver compiled in {time.time() - t_ts:.1f}s", flush=True)

    # Extended range: V0=0.001 to bt0=3.0, V0=0.005 to fold (~2.4)
    bt0_coarse = np.arange(0.1, 1.51, 0.1)
    bt0_fine = np.arange(1.5, 3.01, 0.05)
    bt0_values = np.unique(np.round(np.concatenate([bt0_coarse, bt0_fine]), 4))
    V0_temp_list = [0.001, 0.005]
    temp_sweep = {V0t: [] for V0t in V0_temp_list}
    mask_far = (r_arr >= 60) & (r_arr <= 150)

    t_b2 = time.time()
    for V0_temp in V0_temp_list:
        print(f"\n  V0 = {V0_temp} temperature sweep")

        # Initial seed at bt0=0.1
        if V0_temp in solutions:
            Phi_cont = solutions[V0_temp]["Phi"].copy()
        else:
            Phi_p, _, _, _, _, _ = solve_proxy_V0_continuation(
                N, t0, V0_temp, n_core, beta0, cstar_sq,
                n_steps=8, tol=1e-8, verbose=False)
            Phi_ex, _, _, _ = solve_fixedpoint_from_proxy(
                Phi_p, V0_temp, fp_solver, cstar_sq, verbose=False)
            Phi_cont = np.array(Phi_ex)

        fold_reached = False
        for bt0 in bt0_values:
            if fold_reached:
                break

            this_beta0 = bt0 / t0

            # Solve fixedpoint at this temperature (3-arg solver, compiled once)
            Phi_ex_j, conv_j, nit_j, res_j = ts_solver(
                jnp.array(Phi_cont), V0_temp, this_beta0)
            Phi_ex_np = np.array(jax.block_until_ready(Phi_ex_j))
            conv_ex = bool(conv_j)
            Fex = float(res_j)

            lapse_ex = 1.0 + Phi_ex_np / cstar_sq
            minN_ex = lapse_ex.min()
            rs_ex = extract_rs_np(Phi_ex_np, N, n_core, cstar_sq)

            # Detect fold: negative-lapse branch or non-convergence
            if minN_ex < 0:
                print(f"  bt0={bt0:.2f}: FOLD — min(N)={minN_ex:.4f} (negative lapse)")
                fold_reached = True
                continue

            dPhi_core = np.max(np.abs(np.diff(Phi_ex_np[:50])))
            Phi_scale = max(np.max(np.abs(Phi_ex_np)), 1e-15)
            stagger = dPhi_core / Phi_scale

            if conv_ex or Fex < 1e-6:
                Phi_cont = Phi_ex_np.copy()

            R_outer = float(r_arr[-1])
            GM_est = -Phi_ex_np[mask_far] * r_arr[mask_far] * R_outer / (R_outer - r_arr[mask_far])
            GM_ex = float(np.mean(GM_est))
            GM_std_ex = float(np.std(GM_est) / GM_ex) if GM_ex > 0 else np.nan

            # Proxy at this temperature (analytic)
            Phi_p_T = solve_analytic_proxy(N, t0, V0_temp, n_core, this_beta0,
                                            cstar_sq)
            rs_proxy_T = extract_rs_np(Phi_p_T, N, n_core, cstar_sq)

            entry = {
                "bt0": bt0, "rs_exact": rs_ex, "conv_exact": conv_ex,
                "F_exact": Fex, "minN_exact": minN_ex,
                "stagger": stagger, "Phi_exact": Phi_ex_np,
                "GM_exact": GM_ex, "GM_std_exact": GM_std_ex,
                "rs_proxy": rs_proxy_T,
            }
            temp_sweep[V0_temp].append(entry)

            print(f"  bt0={bt0:.2f}: rs={rs_ex:.4f} "
                  f"|F|={Fex:.2e} min(N)={minN_ex:.4f} "
                  f"proxy_rs={rs_proxy_T:.4f} "
                  f"{'OK' if conv_ex else 'NEAR' if Fex < 1e-6 else 'FAIL'}")

    dt_b2 = time.time() - t_b2
    print(f"\nPhase B2 done in {dt_b2:.1f}s")

    # ═══════════════════════════════════════════════════════════════
    # PHASE C: V0 sweep for summary figure
    # ═══════════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("PHASE C: V0 sweep for summary (exact solver)")
    print("=" * 70)

    V0_sweep = np.array(sorted(set(
        list(np.linspace(0.005, 0.067, 15)) +
        list(V0_core) + list(V0_extended)
    )))

    sweep_results = []
    Phi_seed_sw = None
    for V0 in V0_sweep:
        if V0 in solutions:
            s = solutions[V0]
            sweep_results.append({"V0": V0, "rs": s["rs"], "min_N": s["min_N"]})
            Phi_seed_sw = s["Phi"].copy()
            continue

        Phi_p, conv_p, _, _, _, _ = solve_proxy_V0_continuation(
            N, t0, V0, n_core, beta0, cstar_sq,
            n_steps=max(8, int(V0 * 200)), tol=1e-8,
            Phi_seed=Phi_seed_sw, verbose=False)
        Phi_ex, conv_ex, res_ex, _ = solve_fixedpoint_from_proxy(
            Phi_p, V0, fp_solver, cstar_sq, verbose=False)
        Phi_np = np.array(Phi_ex)
        lapse = 1.0 + Phi_np / cstar_sq
        rs = extract_rs_np(Phi_np, N, n_core, cstar_sq)
        sweep_results.append({"V0": V0, "rs": rs, "min_N": lapse.min()})
        if conv_ex:
            Phi_seed_sw = Phi_np
        print(f"  V0={V0:.4f}: rs={rs:.4f}, min(N)={lapse.min():.6f}")

    # Proxy + analytic V0 sweep (skip near-critical V0 where analytic is slow)
    V0_sweep_analytic = V0_sweep[V0_sweep <= 0.04]
    print(f"\n  Proxy + analytic V0 sweep ({len(V0_sweep_analytic)} pts up to 0.04):")
    sweep_proxy_rs = []
    sweep_analytic_rs = []
    sweep_proxy_minN = []
    sweep_analytic_minN = []
    for V0 in V0_sweep_analytic:
        seed = solutions[V0]["Phi"] if V0 in solutions else None
        Phi_p = solve_analytic_proxy(N, t0, V0, n_core, beta0, cstar_sq,
                                     Phi_seed=seed)
        Phi_a = solve_analytic_full(N, t0, V0, n_core, beta0, cstar_sq,
                                    Phi_seed=seed)
        sweep_proxy_rs.append(extract_rs_np(Phi_p, N, n_core, cstar_sq))
        sweep_analytic_rs.append(extract_rs_np(Phi_a, N, n_core, cstar_sq))
        sweep_proxy_minN.append((1 + Phi_p / cstar_sq).min())
        sweep_analytic_minN.append((1 + Phi_a / cstar_sq).min())

    sweep_proxy_rs = np.array(sweep_proxy_rs)
    sweep_analytic_rs = np.array(sweep_analytic_rs)
    sweep_proxy_minN = np.array(sweep_proxy_minN)
    sweep_analytic_minN = np.array(sweep_analytic_minN)

    # ═══════════════════════════════════════════════════════════════
    # PHASE TOV: n_core scan at fixed mass
    # ═══════════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("PHASE TOV: n_core scan at fixed mass")
    print("=" * 70)

    tov_configs = [
        (0.1, 0.03, 5),
        (0.1, 0.05, 5),
        (0.5, 0.005, 5),
        (1.0, 0.003, 5),
    ]
    nc_min, nc_max = 2, 20
    tov_results = {}

    for beta0_tov, V0_ref, nc_ref in tov_configs:
        M0 = (V0_ref / (2.0 * cstar_sq)) * shell_g_sum(nc_ref)
        tag = f"b{beta0_tov:.1f}_V{V0_ref}"
        print(f"\n--- beta0={beta0_tov}, V0_ref={V0_ref}, M0={M0:.4f} ---")

        ncs = np.arange(nc_min, nc_max + 1, dtype=int)
        E_mis_arr = []
        rs_arr = []
        minN_arr = []
        dU_arr = []
        dF_arr = []
        I_grav_arr = []
        I_mat_arr = []
        I_tot_arr = []
        # New: corrected quantities
        I_grav_new_arr = []  # I_grav = E_mis (no beta^2 prefactor)
        I_mat_surf_arr = []  # surface-reduced I_mat
        I_tot_new_arr = []   # I_grav_new + I_mat
        I_tot_surf_arr = []  # I_grav_new + I_mat_surf
        conv_arr = []

        V0_cap = 0.015 / max(beta0_tov, 0.05)

        for nc in ncs:
            V0_nc = M0 * 2.0 * cstar_sq / shell_g_sum(nc)

            if V0_nc > V0_cap:
                print(f"    nc={nc:2d}  V0={V0_nc:.6f} > cap, SKIP")
                for arr in [E_mis_arr, rs_arr, minN_arr, dU_arr, dF_arr,
                            I_grav_arr, I_mat_arr, I_tot_arr,
                            I_grav_new_arr, I_mat_surf_arr,
                            I_tot_new_arr, I_tot_surf_arr]:
                    arr.append(np.nan)
                conv_arr.append(False)
                continue

            print(f"    nc={nc:2d}  V0={V0_nc:.6f}", end="", flush=True)

            try:
                n_steps = max(8, int(np.ceil(V0_nc * beta0_tov / 0.001)))
                n_steps = min(n_steps, 40)

                # Step 1: proxy seed via V0 continuation
                Phi_p, conv_p, _, mi_bg_tov, _, _ = solve_proxy_V0_continuation(
                    N, t0, V0_nc, nc, beta0_tov, cstar_sq,
                    n_steps=n_steps, tol=1e-8, verbose=False)

                # Step 2: fully self-consistent fixedpoint
                # TOV uses varying n_core, so build per-nc solver
                fp_tov_nc = FixedpointSolver(N, t0, nc, beta0_tov, cstar_sq,
                                              max_iter=60, tol=1e-6)
                Phi_ex, conv_ex, res_ex, _ = solve_fixedpoint_from_proxy(
                    Phi_p, V0_nc, fp_tov_nc, cstar_sq, verbose=False)

                Phi_np = np.array(jax.block_until_ready(Phi_ex))
                lapse = 1.0 + Phi_np / cstar_sq
                minN_val = float(lapse.min())

                rs_val = extract_rs_np(Phi_np, N, nc, cstar_sq)
                Em = mismatch_energy_np(Phi_np, N, t0, nc, beta0_tov, cstar_sq, mi_bg_tov)

                U_src, S_src, U_bg, S_bg = matter_U_S(
                    Phi_np, N, t0, V0_nc, nc, beta0_tov, cstar_sq)
                dU = U_src - U_bg
                dF = (U_src - S_src / beta0_tov) - (U_bg - S_bg / beta0_tov)

                # Old prefactor (kept for comparison)
                I_g = (beta0_tov / cstar_sq)**2 * Em
                I_m = beta0_tov * dF
                I_t = I_g + I_m

                # NEW: corrected gravity prefactor — I_grav = E_mis
                I_g_new = Em
                # Surface-reduced matter: subtract bulk rest piece
                bulk_rest = 0.5 * V0_nc * shell_g_sum(nc)
                dF_surf = dF - bulk_rest
                I_m_surf = beta0_tov * dF_surf
                I_t_new = I_g_new + I_m
                I_t_surf = I_g_new + I_m_surf

                E_mis_arr.append(Em)
                rs_arr.append(rs_val)
                minN_arr.append(minN_val)
                dU_arr.append(dU)
                dF_arr.append(dF)
                I_grav_arr.append(I_g)
                I_mat_arr.append(I_m)
                I_tot_arr.append(I_t)
                I_grav_new_arr.append(I_g_new)
                I_mat_surf_arr.append(I_m_surf)
                I_tot_new_arr.append(I_t_new)
                I_tot_surf_arr.append(I_t_surf)
                conv_arr.append(bool(conv_ex))

                print(f"  rs={rs_val:.4f}  Em={Em:.3e}  I_g_new={I_g_new:.3e}"
                      f"  I_m={I_m:.3e}  I_m_surf={I_m_surf:.3e}"
                      f"  I_t_surf={I_t_surf:.3e}  |F|={float(res_ex):.1e}")

            except Exception as e:
                print(f"  FAILED: {e}")
                for arr in [E_mis_arr, rs_arr, minN_arr, dU_arr, dF_arr,
                            I_grav_arr, I_mat_arr, I_tot_arr,
                            I_grav_new_arr, I_mat_surf_arr,
                            I_tot_new_arr, I_tot_surf_arr]:
                    arr.append(np.nan)
                conv_arr.append(False)

        tov_results[tag] = {
            "ncs": ncs, "M0": M0, "beta0": beta0_tov, "V0_ref": V0_ref,
            "E_mis": np.array(E_mis_arr), "rs": np.array(rs_arr),
            "minN": np.array(minN_arr), "dU": np.array(dU_arr),
            "dF": np.array(dF_arr),
            "I_grav": np.array(I_grav_arr), "I_mat": np.array(I_mat_arr),
            "I_tot": np.array(I_tot_arr),
            "I_grav_new": np.array(I_grav_new_arr),
            "I_mat_surf": np.array(I_mat_surf_arr),
            "I_tot_new": np.array(I_tot_new_arr),
            "I_tot_surf": np.array(I_tot_surf_arr),
            "converged": np.array(conv_arr),
        }

        # Report minima for all functional variants
        for label, key in [("old (beta^2 pref)", "I_tot"),
                           ("new (E_mis pref)", "I_tot_new"),
                           ("new+surf", "I_tot_surf")]:
            It = tov_results[tag][key]
            valid = np.isfinite(It)
            if np.any(valid):
                i_star = np.argmin(It[valid])
                nc_star = ncs[valid][i_star]
                is_interior = (i_star > 0) and (i_star < np.sum(valid) - 1)
                print(f"  => {label}: nc*={nc_star}, interior={is_interior}, "
                      f"I_tot*={It[valid][i_star]:.4e}")

    # ═══════════════════════════════════════════════════════════════
    # Compute derived quantities
    # ═══════════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("Computing derived quantities...")
    print("=" * 70)

    # Embeddings
    V0_embed = [0.01, 0.05, 0.066]
    embed_rho = {}
    embed_R = {}
    proxy_embed_rho = {}
    proxy_embed_R = {}
    for V0 in V0_embed:
        rho, R, _ = compute_embedding(solutions[V0]["Phi"], N, cstar_sq)
        embed_rho[V0] = rho
        embed_R[V0] = R
        if V0 in proxy_solutions:
            rho_p, R_p, _ = compute_embedding(proxy_solutions[V0]["Phi"], N, cstar_sq)
            proxy_embed_rho[V0] = rho_p
            proxy_embed_R[V0] = R_p

    # Thermodynamics
    V0_thermo = [V0 for V0 in sorted(solutions.keys())
                 if 0.01 <= V0 <= 0.068]
    thermo_data = []
    for V0 in V0_thermo:
        s = solutions[V0]
        rs = s["rs"]
        if rs <= 0:
            continue
        kH = extract_kappa_H(rs, cstar_sq)
        T_H = kH / (2 * np.pi * np.sqrt(cstar_sq))

        kH_proxy = np.nan
        kH_analytic = np.nan
        if V0 in proxy_solutions and proxy_solutions[V0]["rs"] > 0:
            kH_proxy = extract_kappa_H(proxy_solutions[V0]["rs"], cstar_sq)
        if V0 in analytic_solutions and analytic_solutions[V0]["rs"] > 0:
            kH_analytic = extract_kappa_H(analytic_solutions[V0]["rs"], cstar_sq)

        thermo_data.append({
            "V0": V0, "rs": rs, "kappa_H": kH, "T_H": T_H,
            "min_N": s["min_N"], "kH_proxy": kH_proxy, "kH_analytic": kH_analytic,
        })

    # Cross-residuals
    V0_comp = sorted(set(V0_core + V0_extended))
    cross_proxy = []
    cross_analytic = []
    for V0 in V0_comp:
        if V0 in solutions and V0 in proxy_solutions and V0 in analytic_solutions:
            Phi_exact = solutions[V0]["Phi"]
            resid_fn_ex = fp_solver.resid_fn
            V0_j = jnp.float64(V0)
            F_p = np.array(resid_fn_ex(jnp.array(proxy_solutions[V0]["Phi"]), V0_j))
            F_a = np.array(resid_fn_ex(jnp.array(analytic_solutions[V0]["Phi"]), V0_j))
            cross_proxy.append(np.max(np.abs(F_p[:-1])))
            cross_analytic.append(np.max(np.abs(F_a[:-1])))
        else:
            cross_proxy.append(np.nan)
            cross_analytic.append(np.nan)

    # T^QI profiles
    V0_tqi = [0.01, 0.02, 0.03, 0.05, 0.057]
    tqi_sc_dict = {}
    tqi_pr_dict = {}
    for V0 in V0_tqi:
        s = solutions[V0]
        rs = s["rs"]
        if rs < 0.01:
            continue
        Nbar = s["Nbar"]
        Nbar_ext = np.concatenate([Nbar, [1.0]])
        la = beta0 * np.maximum(np.abs(Nbar_ext), 1e-10) * t0 / 4.0
        lt = np.log(np.maximum(la, 1e-30))
        tqi_sc = np.abs(Nbar_ext)**6 / np.maximum(lt**2, 1e-30)

        f_s = np.maximum(1.0 - rs / r_arr, 1e-10)
        la_s = beta0 * t0 * np.sqrt(np.maximum(f_s, 1e-10)) / 4.0
        lt_s = np.log(np.maximum(np.abs(la_s), 1e-30))
        tqi_pr = f_s**3 / np.maximum(lt_s**2, 1e-30)

        tqi_sc_dict[V0] = tqi_sc
        tqi_pr_dict[V0] = tqi_pr

    # Verification data (V0=0.05)
    V0_verify = 0.05
    s_v = solutions[V0_verify]
    Phi_v = s_v["Phi"]
    kappa_v = s_v["kappa"]
    LHS_v = np.zeros(N)
    LHS_v[:-1] += kappa_v * (Phi_v[:-1] - Phi_v[1:])
    LHS_v[1:] += kappa_v * (Phi_v[1:] - Phi_v[:-1])

    pref_v = beta0 / cstar_sq
    _, rho_sigma_fn_v, rho_tgt_fn_v = _make_physics_fns(
        N, t0, n_core, beta0, cstar_sq)
    rho_sigma_v = np.array(jit(rho_sigma_fn_v)(jnp.array(Phi_v)))
    rho_tgt_v = np.array(jit(lambda p: rho_tgt_fn_v(p, jnp.float64(V0_verify)))(jnp.array(Phi_v)))
    RHS_v = pref_v * (rho_sigma_v - rho_tgt_v)
    RHS_v[-1] = 0.0

    # ═══════════════════════════════════════════════════════════════
    # Save to npz
    # ═══════════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("Saving to npz...")
    print("=" * 70)

    DATADIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "fermion", "numerical", "data")
    os.makedirs(DATADIR, exist_ok=True)

    data = {}

    # Parameters
    data["N"] = N
    data["t0"] = t0
    data["n_core"] = n_core
    data["beta0"] = beta0
    data["cstar_sq"] = cstar_sq
    data["a"] = a
    data["r_arr"] = r_arr
    data["r_bond"] = r_bond

    # V0 lists
    data["V0_core"] = np.array(V0_core)
    data["V0_extended"] = np.array(V0_extended)
    data["V0_supercrit"] = np.array(V0_supercrit)

    # Per-V0 solutions
    all_V0 = sorted(solutions.keys())
    data["solution_V0s"] = np.array(all_V0)
    for V0 in all_V0:
        s = solutions[V0]
        tag = f"sol_{V0:.4f}"
        data[f"{tag}_Phi"] = s["Phi"]
        data[f"{tag}_lapse"] = s["lapse"]
        data[f"{tag}_Nbar"] = s["Nbar"]
        data[f"{tag}_kappa"] = s["kappa"]
        data[f"{tag}_kappa_flat"] = s["kappa_flat"]
        data[f"{tag}_rs"] = s["rs"]
        data[f"{tag}_min_N"] = s["min_N"]
        data[f"{tag}_converged"] = s["converged"]
        data[f"{tag}_residual"] = s["residual"]

    # Proxy + analytic solutions
    proxy_V0s = sorted(proxy_solutions.keys())
    data["proxy_V0s"] = np.array(proxy_V0s)
    for V0 in proxy_V0s:
        tag = f"proxy_{V0:.4f}"
        data[f"{tag}_Phi"] = proxy_solutions[V0]["Phi"]
        data[f"{tag}_lapse"] = proxy_solutions[V0]["lapse"]
        data[f"{tag}_rs"] = proxy_solutions[V0]["rs"]
        lapse_p = proxy_solutions[V0]["lapse"]
        Nbar_p = 0.5 * (lapse_p[:-1] + lapse_p[1:])
        data[f"{tag}_Nbar"] = Nbar_p
        data[f"{tag}_kappa_ratio"] = Nbar_p**2

    analytic_V0s = sorted(analytic_solutions.keys())
    data["analytic_V0s"] = np.array(analytic_V0s)
    for V0 in analytic_V0s:
        tag = f"analytic_{V0:.4f}"
        data[f"{tag}_Phi"] = analytic_solutions[V0]["Phi"]
        data[f"{tag}_lapse"] = analytic_solutions[V0]["lapse"]
        data[f"{tag}_rs"] = analytic_solutions[V0]["rs"]
        lapse_a = analytic_solutions[V0]["lapse"]
        Nbar_a = 0.5 * (lapse_a[:-1] + lapse_a[1:])
        data[f"{tag}_Nbar"] = Nbar_a
        data[f"{tag}_kappa_ratio"] = Nbar_a**2

    # Floor study — exact
    data["floors"] = np.array(floors)
    data["V0_floor_test"] = np.array(V0_floor_test)
    for V0 in V0_floor_test:
        tag = f"floor_{V0:.4f}"
        data[f"{tag}_min_N"] = np.array([d["min_N"] for d in floor_data[V0]])
        data[f"{tag}_rs"] = np.array([d["rs"] for d in floor_data[V0]])
        data[f"{tag}_converged"] = np.array([d["converged"] for d in floor_data[V0]])

    # Floor study — proxy
    data["V0_proxy_floor"] = np.array(V0_proxy_floor)
    for V0 in V0_proxy_floor:
        tag = f"pfloor_{V0:.4f}"
        data[f"{tag}_min_N"] = np.array([d["min_N"] for d in proxy_floor_data[V0]])

    # Sweep
    data["sweep_V0"] = np.array([d["V0"] for d in sweep_results])
    data["sweep_rs"] = np.array([d["rs"] for d in sweep_results])
    data["sweep_min_N"] = np.array([d["min_N"] for d in sweep_results])
    data["sweep_proxy_rs"] = sweep_proxy_rs
    data["sweep_analytic_rs"] = sweep_analytic_rs
    data["sweep_proxy_minN"] = sweep_proxy_minN
    data["sweep_analytic_minN"] = sweep_analytic_minN

    # Embeddings
    data["embed_V0s"] = np.array(V0_embed)
    for V0 in V0_embed:
        tag = f"embed_{V0:.4f}"
        data[f"{tag}_rho"] = embed_rho[V0]
        data[f"{tag}_R"] = embed_R[V0]
    for V0 in V0_embed:
        if V0 in proxy_embed_rho:
            ptag = f"pembed_{V0:.4f}"
            data[f"{ptag}_rho"] = proxy_embed_rho[V0]
            data[f"{ptag}_R"] = proxy_embed_R[V0]

    # Thermodynamics
    data["thermo_V0"] = np.array([d["V0"] for d in thermo_data])
    data["thermo_rs"] = np.array([d["rs"] for d in thermo_data])
    data["thermo_kappa_H"] = np.array([d["kappa_H"] for d in thermo_data])
    data["thermo_T_H"] = np.array([d["T_H"] for d in thermo_data])
    data["thermo_min_N"] = np.array([d["min_N"] for d in thermo_data])
    data["thermo_kH_proxy"] = np.array([d["kH_proxy"] for d in thermo_data])
    data["thermo_kH_analytic"] = np.array([d["kH_analytic"] for d in thermo_data])

    # Cross-residuals
    data["cross_V0"] = np.array(V0_comp)
    data["cross_proxy"] = np.array(cross_proxy)
    data["cross_analytic"] = np.array(cross_analytic)

    # T^QI profiles
    data["tqi_V0s"] = np.array(sorted(tqi_sc_dict.keys()))
    for V0 in sorted(tqi_sc_dict.keys()):
        tag = f"tqi_{V0:.4f}"
        data[f"{tag}_sc"] = tqi_sc_dict[V0]
        data[f"{tag}_pr"] = tqi_pr_dict[V0]

    # Temperature sweep (per-V0 bt0 arrays may differ in length due to fold)
    data["temp_V0s"] = np.array(V0_temp_list)
    data["temp_bt0"] = bt0_values  # full grid (for reference)
    for V0_temp in V0_temp_list:
        tag = f"temp_{V0_temp:.4f}"
        sw = temp_sweep[V0_temp]
        data[f"{tag}_bt0"] = np.array([e["bt0"] for e in sw])
        data[f"{tag}_rs"] = np.array([e["rs_exact"] for e in sw])
        data[f"{tag}_conv"] = np.array([e["conv_exact"] for e in sw])
        data[f"{tag}_F"] = np.array([e["F_exact"] for e in sw])
        data[f"{tag}_minN"] = np.array([e["minN_exact"] for e in sw])
        data[f"{tag}_stagger"] = np.array([e["stagger"] for e in sw])
        data[f"{tag}_Phi"] = np.array([e["Phi_exact"] for e in sw])
        data[f"{tag}_GM"] = np.array([e["GM_exact"] for e in sw])
        data[f"{tag}_GM_std"] = np.array([e["GM_std_exact"] for e in sw])
        data[f"{tag}_proxy_rs"] = np.array([e["rs_proxy"] for e in sw])

    # Verification
    data["verify_V0"] = V0_verify
    data["verify_LHS"] = LHS_v
    data["verify_RHS"] = RHS_v

    # TOV phase
    tov_tags = sorted(tov_results.keys())
    data["tov_n_configs"] = len(tov_tags)
    data["tov_beta0s"] = np.array([tov_results[t]["beta0"] for t in tov_tags])
    data["tov_V0_refs"] = np.array([tov_results[t]["V0_ref"] for t in tov_tags])
    for tag in tov_tags:
        res = tov_results[tag]
        prefix = f"tov_{tag}"
        data[f"{prefix}_ncs"] = res["ncs"]
        data[f"{prefix}_M0"] = res["M0"]
        data[f"{prefix}_beta0"] = res["beta0"]
        data[f"{prefix}_V0_ref"] = res["V0_ref"]
        data[f"{prefix}_E_mis"] = res["E_mis"]
        data[f"{prefix}_rs"] = res["rs"]
        data[f"{prefix}_minN"] = res["minN"]
        data[f"{prefix}_dU"] = res["dU"]
        data[f"{prefix}_dF"] = res["dF"]
        data[f"{prefix}_I_grav"] = res["I_grav"]
        data[f"{prefix}_I_mat"] = res["I_mat"]
        data[f"{prefix}_I_tot"] = res["I_tot"]
        data[f"{prefix}_converged"] = res["converged"]

    outpath = os.path.join(DATADIR, "exact_solutions.npz")
    np.savez_compressed(outpath, **data)
    print(f"Saved: {outpath}")
    print(f"  Keys: {len(data)}")
    print(f"  Size: {os.path.getsize(outpath) / 1024:.0f} KB")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'V0':>8s} {'rs/a':>8s} {'min_N':>8s} {'kH':>10s} {'TH':>10s}")
    print("-" * 50)
    for d in thermo_data:
        print(f"{d['V0']:8.4f} {d['rs']:8.4f} {d['min_N']:8.4f} "
              f"{d['kappa_H']:10.4e} {d['T_H']:10.4e}")

    dt_total = time.time() - t_total
    print(f"\nTotal time: {dt_total:.0f}s")
    print("Done.")


if __name__ == "__main__":
    main()
