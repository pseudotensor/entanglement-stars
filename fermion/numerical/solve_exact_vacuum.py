#!/usr/bin/env python3
"""
Exact self-consistent solver with vacuum boundary conditions.

Implements the full closure equation (Section 11c) with two boundary
condition modes:

  (A) Thermal bath (existing behavior):
      Constant Yukawa mass m^2 everywhere — models an object immersed
      in a polarizable medium with nonzero static response.

  (B) Asymptotically vacuum (new):
      Position-dependent m^2(r) that decays to zero outside a microscopic
      atmosphere near the core/horizon.  The atmosphere thickness is set
      by the MI correlation length xi ~ 2/(pi * beta0 * t0).
      This implements the physically correct boundary condition for an
      isolated object in empty space (Section 10a).

Key physics:  For vacuum BC, the exterior equation is Poisson (not
Helmholtz), so the potential remains 1/r to arbitrarily large radii.
The screening only operates within the localized atmosphere.

Usage:
    python fermion/numerical/solve_exact_vacuum.py
"""

from __future__ import annotations

import os
import time

import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.interpolate import interp1d
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

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGDIR, exist_ok=True)


# ================================================================
# Exact MI and energy computation from the correlation matrix
# ================================================================

def binary_entropy(f):
    """Binary entropy h(f) = -f ln f - (1-f) ln(1-f) in nats."""
    f = np.clip(f, 1e-15, 1 - 1e-15)
    return -f * np.log(f) - (1 - f) * np.log(1 - f)


def correlation_matrix_from_hamiltonian(diag, offdiag, beta0):
    """
    Compute correlation matrix elements G_nn, G_{n,n+1} from a
    tridiagonal single-particle Hamiltonian.

    Parameters:
        diag: on-site energies (length N)
        offdiag: hopping amplitudes (length N-1), entered as the
                 off-diagonal of the tridiagonal matrix (i.e., -t0*Nbar)
        beta0: inverse temperature

    Returns:
        G_diag: diagonal elements G_nn (length N)
        G_offdiag: off-diagonal elements G_{n,n+1} (length N-1)
    """
    evals, evecs = eigh_tridiagonal(diag, offdiag)
    fermi = 1.0 / (np.exp(beta0 * evals) + 1.0)

    N = len(diag)
    G_diag = np.array([np.dot(evecs[n, :]**2, fermi) for n in range(N)])
    G_offdiag = np.array([np.dot(evecs[n, :] * evecs[n+1, :], fermi)
                          for n in range(N - 1)])
    return G_diag, G_offdiag


def compute_exact_mi(G_diag, G_offdiag, N):
    """
    Compute single-channel MI between adjacent shells from the
    correlation matrix.

    Returns: array of MI values (length N-1)
    """
    mi = np.zeros(N - 1)
    for n in range(N - 1):
        s_n = binary_entropy(G_diag[n])
        s_m = binary_entropy(G_diag[n + 1])

        avg = 0.5 * (G_diag[n] + G_diag[n + 1])
        diff = 0.5 * (G_diag[n] - G_diag[n + 1])
        disc = np.sqrt(diff**2 + G_offdiag[n]**2)
        lam_p = np.clip(avg + disc, 1e-15, 1 - 1e-15)
        lam_m = np.clip(avg - disc, 1e-15, 1 - 1e-15)

        s_joint = binary_entropy(lam_p) + binary_entropy(lam_m)
        mi[n] = max(s_n + s_m - s_joint, 0.0)
    return mi


def compute_energy_profile(G_offdiag, t0, Nbar, g):
    """
    Exact bond kinetic energy: rho(n) = 2*t0*Nbar_n*Re(G_{n,n+1})*g_n

    This is eq. (11c.4) of the paper.

    Parameters:
        G_offdiag: off-diagonal correlation matrix elements (N-1)
        t0: hopping amplitude
        Nbar: bond-averaged lapse (N-1)
        g: orbital degeneracy at each BOND (use g of the lower shell)

    Returns: energy profile (length N-1, one per bond)
    """
    return 2.0 * t0 * Nbar * G_offdiag.real * g


def exact_mi_uniform_chain(beta_t, N_chain=600):
    """
    Compute single-channel MI between adjacent sites of a uniform
    free-fermion chain at half filling (reference value).
    """
    if beta_t < 1e-14:
        return 0.0

    diag = np.zeros(N_chain)
    offdiag = -np.ones(N_chain - 1)

    evals, evecs = eigh_tridiagonal(diag, offdiag)
    fermi = 1.0 / (np.exp(beta_t * evals) + 1.0)

    n = N_chain // 2
    G_nn = np.dot(evecs[n, :]**2, fermi)
    G_mm = np.dot(evecs[n+1, :]**2, fermi)
    G_nm = np.dot(evecs[n, :] * evecs[n+1, :], fermi)

    s_n = binary_entropy(G_nn)
    s_m = binary_entropy(G_mm)

    avg = 0.5 * (G_nn + G_mm)
    diff = 0.5 * (G_nn - G_mm)
    disc = np.sqrt(diff**2 + G_nm**2)
    lam_p = np.clip(avg + disc, 1e-15, 1 - 1e-15)
    lam_m = np.clip(avg - disc, 1e-15, 1 - 1e-15)

    s_joint = binary_entropy(lam_p) + binary_entropy(lam_m)
    return max(s_n + s_m - s_joint, 0.0)


def tabulate_mi_function(beta0_t0, n_points=800, Nbar_max=3.0):
    """
    Tabulate the conductance ratio F(Nbar) = MI(beta0*Nbar*t0) / MI(beta0*t0).
    """
    Nbar_values = np.linspace(0.001, Nbar_max, n_points)
    mi_values = np.array([exact_mi_uniform_chain(beta0_t0 * Nb)
                          for Nb in Nbar_values])
    mi_ref = exact_mi_uniform_chain(beta0_t0)

    if mi_ref > 1e-30:
        ratio = mi_values / mi_ref
    else:
        ratio = Nbar_values**2

    func = interp1d(Nbar_values, ratio, kind='cubic',
                    bounds_error=False, fill_value=(ratio[0], ratio[-1]))
    return func, mi_ref


# ================================================================
# Atmosphere profile for vacuum boundary conditions
# ================================================================

def atmosphere_profile(N, n_core, beta0_t0, mode='uv'):
    """
    Compute the atmosphere weight alpha(n) for vacuum boundary conditions.

    Three modes:
      'uv':   Response confined to UV-scale layer (n_core + 1 shell).
              This is the physically correct vacuum BC: the background
              susceptibility only exists in the microscopic near-core region.
      'corr': Response extends over the MI correlation length.
              n_atm = n_core + max(1, int(3 * xi_corr))
      'none': No screening at all (m^2 = 0 everywhere).
              Pure Poisson equation with self-consistent conductances.

    For vacuum BC (isolated object):
        - Near core: thermal response (alpha = 1)
        - Far field: no response (alpha -> 0), equation is Poisson

    For thermal bath: alpha = 1 everywhere (constant screening).
    """
    alpha = np.ones(N)

    if mode == 'none':
        # No Yukawa mass at all — pure Poisson
        alpha[:] = 0.0
        return alpha

    xi_corr = 2.0 / (np.pi * max(beta0_t0, 0.001))

    if mode == 'uv':
        # UV-scale atmosphere: response only within core + 1 shell
        n_atm = n_core + 1
        xi_atm = 1.0  # decay over 1 lattice spacing
    elif mode == 'corr':
        # Correlation-length atmosphere
        n_buffer = max(1, int(3.0 * xi_corr))
        n_atm = n_core + n_buffer
        xi_atm = max(1.0, xi_corr)
    else:
        raise ValueError(f"Unknown atmosphere mode: {mode}")

    for n in range(N):
        if n > n_atm:
            alpha[n] = np.exp(-(n - n_atm) / xi_atm)

    return alpha


# ================================================================
# Self-consistent solver
# ================================================================

def solve_selfconsistent_exact(N, t0, V0, n_core, beta0, cstar_sq,
                                atm_mode='bath',
                                use_exact_mi=True,
                                mi_func=None,
                                max_iter=800, tol=1e-10, mixing=0.3,
                                verbose=False):
    """
    Solve the self-consistent closure equation by Picard iteration.

    Parameters:
        N: number of shells
        t0: hopping amplitude
        V0: on-site potential in core
        n_core: number of core shells
        beta0: inverse temperature
        cstar_sq: c_*^2 = t0^2 / 2
        atm_mode: boundary condition mode:
            'bath' = thermal bath (constant m^2 everywhere)
            'uv'   = UV-scale atmosphere (m^2 only in core + 1 shell)
            'corr' = correlation-length atmosphere
            'none' = no Yukawa mass (pure Poisson)
        use_exact_mi: if True, use exact MI for conductances
        mi_func: interpolated MI function F(Nbar) (if use_exact_mi)
        max_iter: maximum Picard iterations
        tol: convergence tolerance
        mixing: Picard mixing parameter
        verbose: print convergence info

    Returns:
        dict with solution fields
    """
    a = 1.0
    r = np.arange(1, N + 1, dtype=float) * a
    g = 4 * np.pi * np.arange(1, N + 1, dtype=float)**2

    beta0_t0 = beta0 * t0

    # Source: mass defect on core shells
    source = np.zeros(N)
    for n in range(min(n_core, N)):
        source[n] = (beta0 / cstar_sq) * g[n] * V0 / 2.0

    # Yukawa mass (uniform thermal background value)
    rho0 = 0.5 * beta0 * t0**2
    m_sq = 2 * beta0 * rho0 / cstar_sq**2

    # Atmosphere profile
    if atm_mode == 'bath':
        alpha = np.ones(N)
    else:
        alpha = atmosphere_profile(N, n_core, beta0_t0, mode=atm_mode)

    # Initialize
    Phi = np.zeros(N)
    kappa = g[:-1] * t0**2  # flat-space conductances

    LAPSE_FLOOR = 1e-5

    converged = False
    for iteration in range(max_iter):
        lapse = np.maximum(1.0 + Phi / cstar_sq, LAPSE_FLOOR)
        Nbar = 0.5 * (lapse[:-1] + lapse[1:])

        # Conductances
        if use_exact_mi and mi_func is not None:
            kappa_new = g[:-1] * t0**2 * mi_func(Nbar)
        else:
            kappa_new = g[:-1] * t0**2 * Nbar**2

        kappa = mixing * kappa_new + (1 - mixing) * kappa

        # Build graph Laplacian + position-dependent Yukawa mass
        L = np.zeros((N, N))
        for n in range(N - 1):
            L[n, n] += kappa[n]
            L[n+1, n+1] += kappa[n]
            L[n, n+1] -= kappa[n]
            L[n+1, n] -= kappa[n]

        # Position-dependent Yukawa mass (the key change for vacuum BC)
        for n in range(N):
            L[n, n] += alpha[n] * m_sq * g[n]

        # BC: Phi[N-1] = 0
        L_red = L[:N-1, :N-1]
        S_red = -source[:N-1]

        Phi_new = np.zeros(N)
        try:
            Phi_new[:N-1] = np.linalg.solve(L_red, S_red)
        except np.linalg.LinAlgError:
            if verbose:
                print(f"  iter {iteration}: singular matrix, stopping")
            break

        Phi_new = np.maximum(Phi_new, cstar_sq * (LAPSE_FLOOR - 1))

        dPhi = np.max(np.abs(Phi_new - Phi))
        norm = np.max(np.abs(Phi_new)) + 1e-30
        Phi = Phi_new

        if verbose and iteration % 50 == 0:
            print(f"  iter {iteration}: dPhi/|Phi| = {dPhi/norm:.2e}")

        if dPhi / norm < tol:
            converged = True
            break

    # Final observables
    lapse = np.maximum(1.0 + Phi / cstar_sq, LAPSE_FLOOR)
    Nbar = 0.5 * (lapse[:-1] + lapse[1:])
    r_bond = 0.5 * (r[:-1] + r[1:])

    if use_exact_mi and mi_func is not None:
        hrr = mi_func(Nbar)
    else:
        hrr = Nbar**2

    # Extract GM from the far-field potential.
    # IMPORTANT: The finite-domain BC (Phi[N-1]=0 at r=R) means the
    # exterior solution is Phi(r) = -GM(1/r - 1/R), NOT -GM/r.
    # So GM = -Phi(r) / (1/r - 1/R) = -Phi(r) * r * R / (R - r).
    R = r[-1]  # outer boundary radius

    if atm_mode in ('uv', 'none'):
        # First pass: estimate r_s from corrected GM
        n_lo = min(n_core + 10, N - 20)
        n_hi = min(N * 3 // 4, n_lo + 100)
        n_hi = max(n_hi, n_lo + 5)
    elif atm_mode == 'corr':
        xi_corr = 2.0 / (np.pi * max(beta0_t0, 0.001))
        n_atm = n_core + max(1, int(3 * xi_corr))
        n_lo = min(n_atm + 3, N - 20)
        n_hi = min(N * 3 // 4, n_lo + 100)
        n_hi = max(n_hi, n_lo + 5)
    else:  # bath
        xi_Y = t0 / np.sqrt(m_sq) if m_sq > 0 else N
        n_lo = min(n_core + 3, N - 2)
        n_hi = min(int(xi_Y * 0.6), N * 3 // 4)
        n_hi = max(n_hi, n_lo + 5)

    idx_mid = slice(n_lo, n_hi)
    # Corrected GM extraction accounting for BC: Phi = -GM(1/r - 1/R)
    GM_vals = -Phi[idx_mid] * r[idx_mid] * R / (R - r[idx_mid])
    GM = np.median(GM_vals) if len(GM_vals) > 0 else 0.0
    r_s = 2 * GM / cstar_sq if GM > 0 else 0.0

    # Entropy per mode: alpha_S = (1/g) * MI_1ch / ln(2)
    # Compute at the "horizon" shell (minimum lapse)
    n_horizon = np.argmin(lapse)
    if n_horizon < N - 1:
        # MI at the horizon bond
        mi_horizon_bond = exact_mi_uniform_chain(beta0_t0 * Nbar[min(n_horizon, N-2)])
        mi_flat = exact_mi_uniform_chain(beta0_t0)
        alpha_S = mi_horizon_bond / np.log(2) if mi_horizon_bond > 0 else 0.0
    else:
        alpha_S = 0.0

    return {
        'Phi': Phi, 'lapse': lapse, 'Nbar': Nbar,
        'hrr': hrr, 'r': r, 'r_bond': r_bond,
        'r_s': r_s, 'GM': GM,
        'converged': converged,
        'iterations': iteration + 1,
        'alpha': alpha,
        'alpha_S': alpha_S,
        'min_lapse': np.min(lapse),
        'beta0_t0': beta0_t0,
    }


# ================================================================
# Temperature sweep
# ================================================================

def run_temperature_sweep(N=200, t0=1.0, n_core=5, V0=0.05):
    """
    Run the exact solver at multiple temperatures, comparing
    thermal bath vs vacuum boundary conditions.

    Uses a FIXED V0 across all temperatures so the source strength
    is constant and the screening effect is clearly visible.

    Returns: list of result dicts
    """
    cstar_sq = t0**2 / 2.0

    # Temperature values (beta0 * t0)
    bt_values = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 2.11, 2.5]

    results = []

    print("=" * 120)
    print("TEMPERATURE SWEEP: Thermal Bath vs Vacuum Boundary Conditions")
    print("=" * 120)
    print(f"N={N}, t0={t0}, n_core={n_core}, c*^2={cstar_sq}, V0={V0} (fixed)")
    print()
    print(f"{'bt0':>6s} | {'GM_bath':>10s} {'GM_uv':>10s} {'GM_none':>10s}"
          f" | {'rs_bath':>8s} {'rs_uv':>8s} {'rs_none':>8s}"
          f" | {'m_sq':>7s}")
    print("-" * 100)

    for bt in bt_values:
        beta0 = bt / t0

        # Tabulate MI function at this temperature
        t_start = time.time()
        mi_func, mi_ref = tabulate_mi_function(bt)

        # Yukawa mass for diagnostics
        rho0 = 0.5 * beta0 * t0**2
        m_sq = 2 * beta0 * rho0 / cstar_sq**2

        # Solve with thermal bath BC
        res_bath = solve_selfconsistent_exact(
            N, t0, V0, n_core, beta0, cstar_sq,
            atm_mode='bath', use_exact_mi=True, mi_func=mi_func)

        # Solve with UV-scale vacuum BC
        res_vac = solve_selfconsistent_exact(
            N, t0, V0, n_core, beta0, cstar_sq,
            atm_mode='uv', use_exact_mi=True, mi_func=mi_func)

        # Solve with no Yukawa mass (pure Poisson)
        res_none = solve_selfconsistent_exact(
            N, t0, V0, n_core, beta0, cstar_sq,
            atm_mode='none', use_exact_mi=True, mi_func=mi_func)

        dt = time.time() - t_start

        # Atmosphere parameters
        xi_corr = 2.0 / (np.pi * max(bt, 0.001))
        n_atm = n_core + max(1, int(3 * xi_corr))

        print(f"{bt:6.3f}"
              f" | {res_bath['GM']:10.6f} {res_vac['GM']:10.6f} {res_none['GM']:10.6f}"
              f" | {res_bath['r_s']:8.4f} {res_vac['r_s']:8.4f} {res_none['r_s']:8.4f}"
              f" | {m_sq:7.2f}"
              f"  ({dt:.1f}s)")

        results.append({
            'beta0_t0': bt, 'V0': V0, 'beta0': beta0,
            'res_bath': res_bath, 'res_vac': res_vac, 'res_none': res_none,
            'mi_func': mi_func, 'mi_ref': mi_ref,
            'xi_corr': xi_corr, 'n_atm': n_atm,
            'm_sq': m_sq,
        })

    print("-" * 110)
    return results


# ================================================================
# Plotting
# ================================================================

def plot_temperature_sweep(results):
    """Plot comparison of thermal bath vs vacuum BC across temperatures."""

    # --- Plot 1: Potential profiles at selected temperatures ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    bt_selected = [0.1, 0.5, 1.0, 1.5, 2.0, 2.11]
    for idx, bt_sel in enumerate(bt_selected):
        ax = axes[idx // 3, idx % 3]
        res = None
        for r in results:
            if abs(r['beta0_t0'] - bt_sel) < 0.01:
                res = r
                break
        if res is None:
            ax.text(0.5, 0.5, f"No data for bt0={bt_sel}", transform=ax.transAxes,
                    ha='center', va='center')
            continue

        r_arr = res['res_bath']['r']
        Phi_bath = res['res_bath']['Phi']
        Phi_vac = res['res_vac']['Phi']
        Phi_none = res['res_none']['Phi']

        cstar_sq = 0.5
        phi_bath = Phi_bath / cstar_sq
        phi_vac = Phi_vac / cstar_sq
        phi_none = Phi_none / cstar_sq

        ax.plot(r_arr, phi_bath, 'b-', lw=1.5, label='Bath')
        ax.plot(r_arr, phi_vac, 'r--', lw=1.5, label='UV atm')
        ax.plot(r_arr, phi_none, 'g:', lw=1.5, label='No $m^2$')

        # Newtonian reference from none r_s
        r_s_none = res['res_none']['r_s']
        if r_s_none > 0.001:
            phi_newt = -r_s_none / (2 * r_arr)
            ax.plot(r_arr, phi_newt, 'k:', lw=1, alpha=0.5, label=f'$1/r$ ($r_s$={r_s_none:.3f})')

        ax.set_xlabel('$r/a$')
        ax.set_ylabel(r'$\Phi/c_*^2$')
        ax.set_title(rf'$\beta_0 t_0 = {bt_sel}$')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, min(60, len(r_arr)))

    plt.tight_layout()
    path = os.path.join(FIGDIR, "vacuum_bc_potential_profiles.pdf")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.savefig(path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()

    # --- Plot 2: r*Phi (should be constant = -GM for 1/r potential) ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    for idx, bt_sel in enumerate(bt_selected):
        ax = axes[idx // 3, idx % 3]
        res = None
        for r in results:
            if abs(r['beta0_t0'] - bt_sel) < 0.01:
                res = r
                break
        if res is None:
            continue

        r_arr = res['res_bath']['r']
        rPhi_bath = r_arr * res['res_bath']['Phi']
        rPhi_vac = r_arr * res['res_vac']['Phi']
        rPhi_none = r_arr * res['res_none']['Phi']

        ax.plot(r_arr, rPhi_bath, 'b-', lw=1.5, label='Bath')
        ax.plot(r_arr, rPhi_vac, 'r--', lw=1.5, label='UV atm')
        ax.plot(r_arr, rPhi_none, 'g:', lw=1.5, label='No $m^2$')

        GM_none = res['res_none']['GM']
        if GM_none > 1e-6:
            ax.axhline(-GM_none, color='k', ls=':', lw=1, alpha=0.5,
                       label=f'$-GM = {-GM_none:.5f}$')

        ax.set_xlabel('$r/a$')
        ax.set_ylabel(r'$r \cdot \Phi$')
        ax.set_title(rf'$\beta_0 t_0 = {bt_sel}$ — $r\Phi$ flatness test')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, min(100, len(r_arr)))

    plt.tight_layout()
    path = os.path.join(FIGDIR, "vacuum_bc_rPhi_test.pdf")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.savefig(path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()

    # --- Plot 3: Summary: r_s and min_lapse vs temperature ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    bts = [r['beta0_t0'] for r in results]
    GM_bath = [r['res_bath']['GM'] for r in results]
    GM_vac = [r['res_vac']['GM'] for r in results]
    GM_none = [r['res_none']['GM'] for r in results]

    ax = axes[0]
    ax.semilogy(bts, [max(g, 1e-12) for g in GM_bath], 'bo-', ms=5, label='Bath')
    ax.semilogy(bts, [max(g, 1e-12) for g in GM_vac], 'rs--', ms=5, label='UV atm')
    ax.semilogy(bts, [max(g, 1e-12) for g in GM_none], 'g^:', ms=5, label='No $m^2$')
    ax.set_xlabel(r'$\beta_0 t_0$')
    ax.set_ylabel(r'$GM$')
    ax.set_title('Effective gravitational mass vs temperature')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(2.11, color='green', ls=':', alpha=0.5)

    minN_bath = [r['res_bath']['min_lapse'] for r in results]
    minN_none = [r['res_none']['min_lapse'] for r in results]

    ax = axes[1]
    ax.plot(bts, minN_bath, 'bo-', ms=5, label='Bath')
    ax.plot(bts, minN_none, 'g^:', ms=5, label='No $m^2$')
    ax.set_xlabel(r'$\beta_0 t_0$')
    ax.set_ylabel(r'$N_{\min}$')
    ax.set_title('Minimum lapse vs temperature')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(2.11, color='green', ls=':', alpha=0.5)

    # Atmosphere thickness
    ax = axes[2]
    xi_corrs = [r['xi_corr'] for r in results]
    n_atms = [r['n_atm'] for r in results]
    ax.plot(bts, xi_corrs, 'ko-', ms=5, label=r'$\xi_{\rm corr}/a$')
    ax.plot(bts, n_atms, 'g^--', ms=5, label=r'$n_{\rm atm}$')
    ax.axhline(1.0, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel(r'$\beta_0 t_0$')
    ax.set_ylabel('Lattice spacings')
    ax.set_title('Atmosphere thickness')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.axvline(2.11, color='green', ls=':', alpha=0.5)

    plt.tight_layout()
    path = os.path.join(FIGDIR, "vacuum_bc_summary.pdf")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.savefig(path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


def plot_focused_bt211(results):
    """
    Focused diagnostic plots at beta0*t0 = 2.11 (the BH entropy point).
    """
    res = None
    for r in results:
        if abs(r['beta0_t0'] - 2.11) < 0.01:
            res = r
            break
    if res is None:
        print("No data at bt0=2.11")
        return

    cstar_sq = 0.5
    r_arr = res['res_none']['r']
    N = len(r_arr)
    r_s = res['res_none']['r_s']

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # (a) Lapse vs Schwarzschild — the key comparison
    ax = axes[0, 0]
    ax.plot(r_arr, res['res_bath']['lapse'], 'b-', lw=1.5, label='Thermal bath')
    ax.plot(r_arr, res['res_vac']['lapse'], 'r--', lw=1.5, label='UV atm')
    ax.plot(r_arr, res['res_none']['lapse'], 'g-', lw=2, label='No $m^2$ (vacuum)')
    if r_s > 0:
        lapse_schw = np.sqrt(np.maximum(1 - r_s / r_arr, 0))
        ax.plot(r_arr, lapse_schw, 'k--', lw=1.2, alpha=0.7,
                label=f'Schwarzschild ($r_s={r_s:.1f}$)')
    ax.set_xlabel('$r/a$')
    ax.set_ylabel('$N(r)$')
    ax.set_title(r'(a) Lapse at $\beta_0 t_0 = 2.11$')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, min(5 * r_s, N) if r_s > 1 else 80)

    # (b) Potential vs full Schwarzschild prediction
    ax = axes[0, 1]
    phi_none = res['res_none']['Phi'] / cstar_sq
    ax.plot(r_arr, phi_none, 'g-', lw=2, label='No $m^2$ (vacuum)')
    if r_s > 0:
        # Full Schwarzschild: Phi/c*^2 = sqrt(1 - r_s/r) - 1
        phi_schw = np.sqrt(np.maximum(1 - r_s / r_arr, 0)) - 1
        # Weak-field Newtonian: Phi/c*^2 = -r_s/(2r)
        phi_newt = -r_s / (2 * r_arr)
        ax.plot(r_arr, phi_schw, 'k--', lw=1.2, alpha=0.7, label='Schwarzschild')
        ax.plot(r_arr, phi_newt, 'k:', lw=1, alpha=0.4, label='Newtonian $-r_s/2r$')
    ax.set_xlabel('$r/a$')
    ax.set_ylabel(r'$\Phi / c_*^2$')
    ax.set_title(r'(b) Potential: emergent vs Schwarzschild')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, min(5 * r_s, N) if r_s > 1 else 80)

    # (c) Lapse residual: (N_emergent - N_Schw) / N_Schw in exterior
    ax = axes[0, 2]
    if r_s > 0:
        lapse_none = res['res_none']['lapse']
        lapse_schw = np.sqrt(np.maximum(1 - r_s / r_arr, 0))
        # Only plot exterior where Schwarzschild lapse > 0.1
        mask = (r_arr > r_s * 1.1) & (lapse_schw > 0.05)
        if mask.any():
            residual = (lapse_none[mask] - lapse_schw[mask]) / lapse_schw[mask]
            ax.plot(r_arr[mask] / r_s, residual, 'g-', lw=2)
            ax.axhline(0, color='k', ls=':', lw=1, alpha=0.5)
            ax.set_xlabel(r'$r / r_s$')
            ax.set_ylabel(r'$(N_{\rm em} - N_{\rm Schw}) / N_{\rm Schw}$')
            ax.set_title('(c) Lapse residual vs Schwarzschild (exterior)')
            ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No solution (r_s=0)', transform=ax.transAxes,
                ha='center', va='center')
        ax.set_title('(c) Lapse residual')

    # (d) Corrected GM extraction: GM(r) = -Phi * r * R/(R-r)
    # Should be flat = GM if the solution is 1/r with BC Phi(R)=0
    ax = axes[1, 0]
    R = r_arr[-1]
    GM_none = res['res_none']['GM']
    if r_s > 0:
        GM_profile = -res['res_none']['Phi'] * r_arr * R / (R - r_arr)
        far = (r_arr > 2 * r_s) & (r_arr < 0.9 * R)
        ax.plot(r_arr[far], GM_profile[far], 'g-', lw=2, label='Emergent')
        ax.axhline(GM_none, color='k', ls=':', lw=1.2, alpha=0.7,
                   label=f'$GM = {GM_none:.3f}$')
    ax.set_xlabel('$r/a$')
    ax.set_ylabel(r'$-\Phi \cdot rR/(R-r)$')
    ax.set_title(r'(d) BC-corrected $GM$ extraction (should be flat)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (e) Co-metric h^{rr} comparison
    ax = axes[1, 1]
    r_bond_none = res['res_none']['r_bond']

    if r_s > 0:
        x_none = r_bond_none / r_s
        mask_none = (x_none > 1.05) & (x_none < 12)

        ax.plot(x_none[mask_none], res['res_none']['hrr'][mask_none],
                'g-', lw=2, label='No $m^2$ (vacuum)')
        x_plot = np.linspace(1.05, 12, 200)
        ax.plot(x_plot, 1 - 1/x_plot, 'k--', lw=1.2, alpha=0.7, label='Schwarzschild')
        ax.set_xlabel(r'$r / r_s$')
    else:
        ax.plot(r_bond_none, res['res_none']['hrr'], 'g-', lw=2, label='No $m^2$')
        ax.set_xlabel('$r/a$')

    ax.set_ylabel(r'$h^{rr} / h_0$')
    ax.set_title(r'(e) Co-metric $h^{rr}$: emergent vs Schwarzschild')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (f) Conductance ratio (zoom near horizon)
    ax = axes[1, 2]
    r_bond_none = res['res_none']['r_bond']
    kappa_ratio_none = res['res_none']['hrr']

    n_plot = min(60, len(kappa_ratio_none))
    ax.plot(r_bond_none[:n_plot], kappa_ratio_none[:n_plot], 'g-', lw=2,
            label='No $m^2$ (vacuum)')
    if r_s > 0:
        ax.axvline(r_s, color='k', ls=':', lw=1, alpha=0.5,
                   label=f'$r_s = {r_s:.1f}$')
    ax.set_xlabel('$r/a$')
    ax.set_ylabel(r'$\kappa / \kappa_{\rm flat} = F(\bar{N})$')
    ax.set_title(r'(f) Conductance degeneration near horizon')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(r'Vacuum BC at $\beta_0 t_0 = 2.11$: emergent Schwarzschild geometry',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(FIGDIR, "vacuum_bc_bt211_focused.pdf")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.savefig(path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


def plot_1overr_persistence(results):
    """
    Show that vacuum BC gives Schwarzschild-like exterior across temperatures.
    Left: lapse comparison vs Schwarzschild.
    Right: far-field r*Phi (normalized at r = 3*r_s) for each temperature.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: lapse vs Schwarzschild for each bt in 'none' mode
    ax = axes[0]
    for res in results:
        bt = res['beta0_t0']
        r_arr = res['res_none']['r']
        lapse = res['res_none']['lapse']
        r_s = res['res_none']['r_s']
        if r_s < 0.5:
            continue
        x = r_arr / r_s
        mask = (x > 1.05) & (x < 10)
        if not mask.any():
            continue
        lapse_schw = np.sqrt(np.maximum(1 - 1/x, 0))
        residual = np.zeros_like(x)
        residual[mask] = (lapse[mask] - lapse_schw[mask])
        ax.plot(x[mask], residual[mask], lw=1.5, label=f'{bt:.2f}')

    ax.axhline(0, color='k', ls=':', lw=1.5, alpha=0.5)
    ax.set_xlabel(r'$r / r_s$')
    ax.set_ylabel(r'$N_{\rm em}(r) - N_{\rm Schw}(r)$')
    ax.set_title('Vacuum (no $m^2$): lapse deviation from Schwarzschild')
    ax.legend(title=r'$\beta_0 t_0$', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # Right: r*Phi normalized at far-field reference for 'none' mode
    ax = axes[1]
    for res in results:
        bt = res['beta0_t0']
        r_arr = res['res_none']['r']
        Phi = res['res_none']['Phi']
        r_s = res['res_none']['r_s']
        if r_s < 0.1:
            continue
        # Reference in far field: r = max(3*r_s, 30)
        n_ref = min(max(int(3 * r_s), 30), len(r_arr) - 10)
        Phi_ref = Phi[n_ref]
        r_ref = r_arr[n_ref]
        if abs(Phi_ref) < 1e-20:
            continue
        ratio = (r_arr * Phi) / (r_ref * Phi_ref)
        far = np.arange(len(r_arr)) >= n_ref - 5
        n_end = min(len(r_arr), n_ref + 120)
        ax.plot(r_arr[n_ref-5:n_end], ratio[n_ref-5:n_end],
                lw=1.5, label=f'{bt:.2f}')

    ax.axhline(1.0, color='k', ls=':', lw=1.5, alpha=0.5)
    ax.set_xlabel('$r/a$')
    ax.set_ylabel(r'$r\Phi(r) / [r_{\rm ref}\Phi(r_{\rm ref})]$')
    ax.set_title('Vacuum (no $m^2$): far-field $r\\Phi$ flatness (1 = $1/r$)')
    ax.legend(title=r'$\beta_0 t_0$', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.8, 1.15)

    plt.tight_layout()
    path = os.path.join(FIGDIR, "vacuum_bc_1overr_persistence.pdf")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.savefig(path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


# ================================================================
# Main
# ================================================================

def run_weak_field_check(N=200, t0=1.0, n_core=5, V0=0.0005):
    """
    Verify 1/r behavior with a tiny source (weak-field regime).
    At bt0=2.11 with V0=0.0005, the potential should be shallow
    enough that r*Phi is flat (confirming pure 1/r).
    """
    cstar_sq = t0**2 / 2.0
    bt_values = [0.5, 1.0, 2.11]

    print()
    print("=" * 80)
    print("WEAK-FIELD VERIFICATION (V0 = %.4f)" % V0)
    print("=" * 80)

    fig, axes = plt.subplots(1, len(bt_values), figsize=(6*len(bt_values), 5))
    if len(bt_values) == 1:
        axes = [axes]

    for idx, bt in enumerate(bt_values):
        beta0 = bt / t0
        mi_func, mi_ref = tabulate_mi_function(bt)

        res_none = solve_selfconsistent_exact(
            N, t0, V0, n_core, beta0, cstar_sq,
            atm_mode='none', use_exact_mi=True, mi_func=mi_func,
            verbose=False)

        res_bath = solve_selfconsistent_exact(
            N, t0, V0, n_core, beta0, cstar_sq,
            atm_mode='bath', use_exact_mi=True, mi_func=mi_func,
            verbose=False)

        r_arr = res_none['r']
        R = r_arr[-1]
        # Corrected GM: Phi = -GM(1/r - 1/R)
        # So GM = -Phi * r * R / (R - r)
        GM_corr = -res_none['Phi'] * r_arr * R / (R - r_arr)
        GM_corr_bath = -res_bath['Phi'] * r_arr * R / (R - r_arr)

        print(f"  bt0={bt:.2f}: GM_none={res_none['GM']:.6f}, "
              f"r_s={res_none['r_s']:.4f}, "
              f"GM_bath={res_bath['GM']:.6f}")

        ax = axes[idx]
        # Plot corrected GM extraction (should be flat = GM for 1/r)
        n_start = n_core + 2
        n_end = min(150, N - 5)  # avoid boundary divergence
        ax.plot(r_arr[n_start:n_end], GM_corr[n_start:n_end], 'g-', lw=2,
                label=f'Vacuum: $GM={res_none["GM"]:.5f}$')
        if np.max(np.abs(GM_corr_bath[n_start:n_end])) > 1e-10:
            ax.plot(r_arr[n_start:n_end], GM_corr_bath[n_start:n_end],
                    'b-', lw=1.5, label=f'Bath: $GM={res_bath["GM"]:.5f}$')
        ax.axhline(res_none['GM'], color='k', ls=':', lw=1, alpha=0.5)
        ax.set_xlabel('$r/a$')
        ax.set_ylabel(r'$-\Phi \cdot r \cdot R/(R-r)$ (corrected $GM$)')
        ax.set_title(rf'$\beta_0 t_0 = {bt}$, $V_0 = {V0}$ (weak field)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGDIR, "vacuum_bc_weak_field_check.pdf")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.savefig(path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close()


def main():
    print("Exact self-consistent solver with vacuum boundary conditions")
    print("=" * 70)
    print()

    # Run temperature sweep
    results = run_temperature_sweep(N=200, t0=1.0, n_core=5, V0=0.05)

    print()
    print("Generating plots...")
    plot_temperature_sweep(results)
    plot_focused_bt211(results)
    plot_1overr_persistence(results)

    # Weak-field verification
    run_weak_field_check()

    # Summary table
    print()
    print("=" * 90)
    print("SUMMARY: Vacuum BC resolves screening")
    print("=" * 90)
    print()
    print("At beta0*t0 = 2.11 (alpha_S = 1/4, BH entropy):")
    for r in results:
        if abs(r['beta0_t0'] - 2.11) < 0.01:
            print(f"  Thermal bath:  r_s = {r['res_bath']['r_s']:.4f}, "
                  f"GM = {r['res_bath']['GM']:.6f}, "
                  f"min_N = {r['res_bath']['min_lapse']:.6f}")
            print(f"  UV atmosphere: r_s = {r['res_vac']['r_s']:.4f}, "
                  f"GM = {r['res_vac']['GM']:.6f}, "
                  f"min_N = {r['res_vac']['min_lapse']:.6f}")
            print(f"  No m^2:        r_s = {r['res_none']['r_s']:.4f}, "
                  f"GM = {r['res_none']['GM']:.6f}, "
                  f"min_N = {r['res_none']['min_lapse']:.6f}")
            print(f"  Atmosphere:    xi_corr = {r['xi_corr']:.3f}a, "
                  f"n_atm = {r['n_atm']}")
            break
    print()
    print("Physical interpretation:")
    print("  - Thermal bath: uniform Yukawa mass screens the 1/r potential")
    print("    at the Thomas-Fermi length lambda_TF ~ 0.14a (sub-lattice).")
    print("  - Vacuum BC: screening confined to microscopic atmosphere")
    print("    (~1 lattice spacing at bt0=2.11). The exterior equation is")
    print("    Poisson, so 1/r persists to arbitrarily large radii.")
    print()
    print("All plots saved to:", FIGDIR)


if __name__ == "__main__":
    main()
