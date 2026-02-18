"""
Numerical solver for the *analytic high-temperature* self-consistent
closure equation on a 1D radial shell chain.

This script solves a reduced, analytic form of the two-state stationarity
equation in which:
  - conductances use the leading high-T formula:
        kappa_n(Phi) = g_n * t0^2 * Nbar_n^2
  - the EL correction term (1/2)(dkappa/dPhi)(DeltaPhi)^2 is included,
    converting the fixed-point equation into the true EL and yielding
    exact Schwarzschild in the sub-Yukawa regime
  - the RHS source is the core defect contrast:
        tilde_rho(n) = g_n * V0/2  for n <= n_core

It is *not* the "exact MI-based" solver of the paper's Section 11c
(Exact self-consistent solution), which computes conductances from the
full correlation matrix with no high-T expansion.  This script serves as:
  (i)  a fast prototype for the analytic regime, and
  (ii) a cross-check of the continuum EL mechanism (w = N^2 linearization)
       at finite N.

Model: 1D radial shell chain with N shells at r_n = a*n (n=1,...,N).
  - Orbital degeneracy g_n = 4*pi*n^2
  - Uniform hopping t_0, on-site potential V_0 for n <= n_core
  - High temperature: beta_0 * t_0 << 1

Method: Picard iteration (fixed-point) on the conductances, starting
from the linearized Poisson solution, then Newton refinement.

Usage:
    python solve_selfconsistent.py
"""

import numpy as np
from scipy.linalg import eigh_tridiagonal, solve_banded
from numba import njit, prange
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
import time

# ---------------------------------------------------------------------------
# Matplotlib settings
# ---------------------------------------------------------------------------
rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.figsize": (6, 4.5),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "lines.linewidth": 1.5,
})

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGDIR, exist_ok=True)

# ===================================================================
# Core physics: the self-consistent Poisson equation
# ===================================================================
#
# From the high-T expansion (eqs 9.3-9.6, 10.1-10.2):
#
# Energy in reconstructed state:
#   <h_n>_{G^(Phi)} = g_n * Nbar_n^2 * beta0*t0^2/2
#                     + V0*g_n*N_n/2 * 1_{core}
#
# Reference data:
#   rho(n) = g_n * beta0*t0^2/2 + V0*g_n/2 * 1_{core}
#
# Mismatch:
#   <h_n> - rho(n) = g_n*(Nbar_n^2 - 1)*beta0*t0^2/2
#                   + V0*g_n*(N_n - 1)/2 * 1_{core}
#
# Write N_n = 1 + Phi_n/c*^2 = 1 + phi_n where phi_n = Phi_n/c*^2.
# Nbar_n = 1 + (phi_n + phi_{n+1})/2.
# Then Nbar_n^2 - 1 = (phi_n+phi_{n+1}) + (phi_n+phi_{n+1})^2/4.
# N_n - 1 = phi_n.
#
# The closure equation (9.8) in Poisson form:
#
#   sum kappa[Phi]*(Phi_n - Phi_m) = (beta0/c*^2)*[mismatch]
#
# Rearranging: separate the linear-in-Phi part of the mismatch as a
# "Yukawa mass" term on the LHS:
#
#   sum kappa[Phi]*(Phi_n-Phi_m) - (beta0/c*^2)*[linear response]*Phi
#     = -(beta0/c*^2)*tilde_rho(n)  + (beta0/c*^2)*[nonlinear response]
#
# At each iteration, we freeze kappa and the nonlinear correction at the
# current Phi, and solve the resulting linear system for the new Phi.

@njit(cache=True)
def compute_lapse_nbar(Phi, cstar_sq, N):
    """N_n = 1 + Phi_n/c*^2, Nbar_n = (N_n + N_{n+1})/2."""
    lapse = np.empty(N)
    Nbar = np.empty(N - 1)
    for n in range(N):
        lapse[n] = 1.0 + Phi[n] / cstar_sq
    for n in range(N - 1):
        Nbar[n] = 0.5 * (lapse[n] + lapse[n + 1])
    return lapse, Nbar


def solve_poisson_with_conductances(N, t0, V0, n_core, beta0, cstar_sq, g,
                                     kappa_bonds, include_mass_term=True,
                                     extra_source=None):
    """Solve the self-consistent Poisson equation for given conductances.

    [L_kappa - M_diag] Phi = -S + extra_source

    where:
      L_kappa = graph Laplacian with bond conductances kappa_bonds
      M_diag = diagonal Yukawa mass from the energy response (diagonal only,
               to preserve diagonal dominance and prevent oscillations)
      S = source from the mass defect tilde_rho
      extra_source = EL correction term (1/2)sum(dkappa/dPhi)(DeltaPhi)^2

    Parameters:
      kappa_bonds: array of length N-1, conductance for bond (n, n+1)
      include_mass_term: whether to include the Yukawa screening
      extra_source: optional array of length N, added to the RHS
    """
    A = np.zeros((N, N))
    b = np.zeros(N)
    prefactor = beta0 / cstar_sq

    for n in range(N):
        # Graph Laplacian with given conductances
        if n < N - 1:
            A[n, n] += kappa_bonds[n]
            A[n, n + 1] -= kappa_bonds[n]
        if n > 0:
            A[n, n] += kappa_bonds[n - 1]
            A[n, n - 1] -= kappa_bonds[n - 1]

        if include_mass_term:
            # Yukawa mass from the linearized energy response.
            # The hopping energy at bond n is g_n * Nbar_n^2 * beta0*t0^2/2.
            # Nbar_n^2 - 1 = 2*(phi_n+phi_{n+1})/2 + ... where phi=Phi/c*^2
            # The diagonal part of the response (dNbar^2/dPhi_n) gives:
            #   d/dPhi_n of Nbar_{n-1}^2 and Nbar_n^2 contributes to row n.
            # We use only the diagonal mass to preserve stability:
            #   mu_n = (beta0/c*^2) * beta0*t0^2/c*^2 * sum of g's at adjacent bonds
            mass_diag = 0.0
            if n < N - 1:
                mass_diag += prefactor * g[n] * beta0 * t0**2 / (2.0 * cstar_sq)
            if n > 0:
                mass_diag += prefactor * g[n - 1] * beta0 * t0**2 / (2.0 * cstar_sq)
            A[n, n] -= mass_diag

            # On-site mass from core potential:
            if n < n_core:
                onsite_mass = prefactor * V0 * g[n] / (2.0 * cstar_sq)
                A[n, n] -= onsite_mass

        # Source: -(beta0/c*^2) * tilde_rho(n)
        if n < n_core:
            b[n] = -prefactor * g[n] * V0 / 2.0

    # Add EL correction (extra source) to RHS
    if extra_source is not None:
        b -= extra_source

    # Pin outer boundary: Phi_N = 0
    A[-1, :] = 0
    A[-1, -1] = 1.0
    b[-1] = 0.0

    # A is tridiagonal — use banded solver for O(N) complexity
    ab = np.zeros((3, N))
    ab[0, 1:] = np.array([A[i, i+1] for i in range(N-1)])   # upper diagonal
    ab[1, :]  = np.diag(A)                                    # main diagonal
    ab[2, :-1] = np.array([A[i+1, i] for i in range(N-1)])   # lower diagonal
    return solve_banded((1, 1), ab, b)


def solve_selfconsistent(N, t0, V0, n_core, beta0, cstar_sq, g,
                          max_iter=200, tol=1e-10, mixing=0.5,
                          verbose=True):
    """Solve the self-consistent closure equation by Picard iteration
    on the conductances.

    1. Start with flat conductances: kappa_n = g_n * t0^2
    2. Solve Poisson equation with current conductances -> Phi
    3. Update conductances: kappa_n = g_n * t0^2 * Nbar_n^2
    4. Mix old and new: kappa <- mixing*kappa_new + (1-mixing)*kappa_old
    5. Repeat until convergence

    Returns: (Phi, converged, history)
    """
    # Initial flat conductances
    kappa = g[:-1] * t0**2

    # Step 1: Solve linearized (flat conductances, mass term for Yukawa seed)
    Phi = solve_poisson_with_conductances(
        N, t0, V0, n_core, beta0, cstar_sq, g, kappa,
        include_mass_term=True)
    # If the seed is too deep (lapse < 0.1), try without mass term
    if (1.0 + Phi / cstar_sq).min() < 0.1:
        Phi = solve_poisson_with_conductances(
            N, t0, V0, n_core, beta0, cstar_sq, g, kappa,
            include_mass_term=False)

    # Clamp potential to enforce lapse floor
    LAPSE_FLOOR = 0.01
    Phi_floor = -(1.0 - LAPSE_FLOOR) * cstar_sq
    Phi = np.maximum(Phi, Phi_floor)

    history = []

    if verbose:
        lapse0, Nbar0 = compute_lapse_nbar(Phi, cstar_sq, N)
        print(f"  Initial (linearized): min(N)={lapse0.min():.6f}, "
              f"max|Phi/c*^2|={np.max(np.abs(Phi))/cstar_sq:.6f}")

    for it in range(max_iter):
        # Current lapse and Nbar
        lapse, Nbar = compute_lapse_nbar(Phi, cstar_sq, N)

        # Check lapse positivity
        if lapse.min() <= 0.0:
            if verbose:
                print(f"  Lapse non-positive at iter {it}: min(N)={lapse.min():.6f}")
            return Phi, False, history

        # New conductances from current Phi
        kappa_new = g[:-1] * t0**2 * Nbar**2

        # Mix
        kappa_mixed = mixing * kappa_new + (1.0 - mixing) * kappa

        # Compute the EL correction: (1/2) sum (dkappa/dPhi_n)(Phi_n - Phi_m)^2
        # Since kappa_nm = g_n * t0^2 * Nbar_nm^2 and Nbar = (N_n+N_{n+1})/2,
        # dkappa_nm/dPhi_n = g_n * t0^2 * Nbar_nm / cstar_sq.
        # This term converts the fixed-point equation into the true EL.
        el_correction = np.zeros(N)
        for n in range(N - 1):
            dPhi = Phi[n] - Phi[n + 1]
            # dkappa/dPhi = g_n * t0^2 * Nbar / c*^2 (same for both sites)
            dk_n = g[n] * t0**2 * Nbar[n] / cstar_sq
            el_correction[n] += 0.5 * dk_n * dPhi**2
            el_correction[n + 1] += 0.5 * dk_n * dPhi**2

        # Solve Poisson with updated conductances.
        # The EL correction is moved to the RHS as an additional source.
        Phi_new = solve_poisson_with_conductances(
            N, t0, V0, n_core, beta0, cstar_sq, g, kappa_mixed,
            include_mass_term=False,
            extra_source=el_correction)

        # Clamp potential to enforce lapse floor (elementwise, no scaling)
        Phi_new = np.maximum(Phi_new, Phi_floor)

        # Convergence check: relative change in Phi
        dPhi = np.max(np.abs(Phi_new - Phi))
        Phi_scale = max(np.max(np.abs(Phi)), 1e-15)
        rel_change = dPhi / Phi_scale
        history.append((it, rel_change, lapse.min()))

        if verbose and (it % 10 == 0 or it < 5):
            print(f"  iter {it:4d}: |dPhi|/|Phi| = {rel_change:.3e}, "
                  f"min(N) = {lapse.min():.6f}")

        if rel_change < tol:
            if verbose:
                print(f"  Converged at iter {it}: |dPhi|/|Phi| = {rel_change:.3e}")
            return Phi_new, True, history

        Phi = Phi_new
        kappa = kappa_mixed

    if verbose:
        print(f"  Not converged after {max_iter} iters: "
              f"|dPhi|/|Phi| = {rel_change:.3e}")
    return Phi, rel_change < 1e-6, history


# ===================================================================
# Exact Fermi-Dirac for validation
# ===================================================================

def fermi_dirac_tridiag(diag, offdiag, beta):
    """G = (exp(beta*h) + I)^{-1} for tridiagonal h."""
    evals, evecs = eigh_tridiagonal(diag, offdiag)
    fermi = 1.0 / (np.exp(beta * evals) + 1.0)
    return (evecs * fermi[np.newaxis, :]) @ evecs.T


# ===================================================================
# Physical observables
# ===================================================================

def compute_observables(Phi, N, t0, V0, n_core, beta0, cstar_sq, g, a=1.0):
    """Compute all physical observables from converged Phi."""
    r = a * np.arange(1, N + 1, dtype=float)
    lapse, Nbar = compute_lapse_nbar(Phi, cstar_sq, N)

    hrr = Nbar**2  # co-metric, normalized so hrr -> 1 at large r
    r_bond = 0.5 * (r[:-1] + r[1:])
    kappa = g[:-1] * t0**2 * Nbar**2

    # Extract GM from far-field potential.
    # Two regimes:
    #  (a) Yukawa-screened (bath): Phi ~ -(GM/r)*exp(-r/xi_Y)
    #      -> use sub-Yukawa plateau of -Phi*r
    #  (b) Vacuum BC: Phi = -GM*(1/r - 1/R) with R = outer boundary
    #      -> use BC-corrected GM = -Phi*r*R/(R-r)
    # The Picard solver uses include_mass_term=False in iterations,
    # so the solution is vacuum-like; we use the BC-corrected extraction.
    R = r[-1]
    i_lo = max(n_core + 5, 15)
    i_hi = min(N - 5, int(N * 0.5))
    r_w = r[i_lo:i_hi]
    Phi_w = Phi[i_lo:i_hi]
    if np.any(np.abs(Phi_w) > 1e-15):
        # BC-corrected extraction: Phi = -GM*(1/r - 1/R)
        GM_est = -Phi_w * r_w * R / (R - r_w)
        GM = np.median(GM_est)
    else:
        GM = 0.0
    r_s = 2.0 * GM / cstar_sq

    M_total = np.sum(g[:n_core] * V0 / 2.0)

    # T^QI_sc (eq 10.4): proportional to Nbar^6 / [log(beta0*Nbar*t0)]^2
    Nbar_ext = np.concatenate([Nbar, [1.0]])
    la = beta0 * np.maximum(np.abs(Nbar_ext), 1e-10) * t0
    lt = np.log(np.maximum(la, 1e-30))
    tqi_sc = np.abs(Nbar_ext)**6 / np.maximum(lt**2, 1e-30)

    # Prescribed-background T^QI (section 8)
    f_s = np.maximum(1.0 - r_s / r, 1e-10) if r_s > 0 else np.ones(N)
    la_s = beta0 * t0 * np.sqrt(np.maximum(f_s, 1e-10))
    lt_s = np.log(np.maximum(np.abs(la_s), 1e-30))
    tqi_pr = f_s**3 / np.maximum(lt_s**2, 1e-30)

    return {
        "r": r, "r_bond": r_bond, "Phi": Phi, "lapse": lapse, "Nbar": Nbar,
        "hrr": hrr, "kappa": kappa, "GM": GM, "r_s": r_s, "M_total": M_total,
        "tqi_sc": tqi_sc, "tqi_prescribed": tqi_pr, "g": g,
    }


# ===================================================================
# Plotting
# ===================================================================

def plot_potential_profile(results_list, labels, cstar_sq, figdir=FIGDIR):
    fig, ax = plt.subplots()
    cm = plt.cm.viridis(np.linspace(0.1, 0.9, len(results_list)))
    for res, lab, c in zip(results_list, labels, cm):
        r, Phi = res["r"], res["Phi"]
        ax.plot(r, np.maximum(Phi / cstar_sq, -1.25), "-", color=c,
                label=lab, zorder=3)
        GM = res["GM"]
        if abs(GM) > 1e-12:
            newt = np.maximum(-GM / (cstar_sq * r), -1.25)
            ax.plot(r, newt, "--", color=c, alpha=0.4, lw=1.0)
    ax.plot([], [], "--", color="gray", alpha=0.5, lw=1,
            label=r"Newtonian $-GM/(c_*^2 r)$")
    ax.set_xlabel(r"$r / a$"); ax.set_ylabel(r"$\Phi(r) / c_*^2$")
    ax.set_title("Self-consistent gravitational potential")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0, 100)
    ax.set_ylim(-1.25, 0.15)
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.axhline(-1, color="red", lw=0.5, ls="--", alpha=0.5)
    ax.text(50, -1.07, r"horizon: $\Phi = -c_*^2$", fontsize=8,
            color="red", alpha=0.7, ha="center")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "potential_profile.pdf"))
    plt.close(fig); print("  Saved potential_profile.pdf")


def plot_cometric_comparison(results_list, labels, figdir=FIGDIR):
    fig, ax = plt.subplots()
    cm = plt.cm.viridis(np.linspace(0.1, 0.9, len(results_list)))
    has = False
    for res, lab, c in zip(results_list, labels, cm):
        rs = res["r_s"]
        if rs < 1e-6: continue
        has = True
        x = res["r_bond"] / rs
        # Only show where Nbar is above the lapse floor (physical region)
        nbar = res["Nbar"]
        m = (x > 1.0) & (x < 120) & (nbar > 0.02)
        if np.any(m):
            ax.plot(x[m], res["hrr"][m], "-", color=c, lw=1.5,
                    label=f"SC, {lab}")
    if has:
        xr = np.linspace(1.5, 120.0, 500)
        ax.plot(xr, 1.0 - 1.0/xr, "k--", lw=2, alpha=0.6,
                label=r"Schwarzschild $1-r_s/r$")
    ax.set_xlabel(r"$r / r_s$"); ax.set_ylabel(r"$h^{rr}/h_0$")
    ax.set_title("Emergent co-metric vs. Schwarzschild")
    ax.set_xlim(0.8, 50); ax.set_ylim(-0.05, 1.1)
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "cometric_comparison.pdf"))
    plt.close(fig); print("  Saved cometric_comparison.pdf")


def plot_lapse_profile(results_list, labels, figdir=FIGDIR):
    fig, ax = plt.subplots()
    cm = plt.cm.viridis(np.linspace(0.1, 0.9, len(results_list)))
    for res, lab, c in zip(results_list, labels, cm):
        ax.plot(res["r"], res["lapse"], "-", color=c, lw=1.5, label=lab)
        # Analytic: N = sqrt(1 - r_s/r) (exact Schwarzschild lapse)
        rs = res["r_s"]
        if rs > 1e-6:
            r_an = res["r"]
            f_an = np.maximum(1.0 - rs / r_an, 0.0)
            ax.plot(r_an, np.sqrt(f_an), "--", color=c, alpha=0.5, lw=1.0)
    ax.plot([], [], "--", color="gray", alpha=0.5,
            label=r"$\sqrt{1-r_s/r}$")
    ax.set_xlabel(r"$r / a$"); ax.set_ylabel(r"$N(r)=1+\Phi/c_*^2$")
    ax.set_title("Lapse function profile")
    ax.axhline(0, color="red", lw=0.8, ls="--", alpha=0.5, label="Horizon $N=0$")
    ax.axhline(1, color="gray", lw=0.5, ls=":")
    ax.set_xlim(0, 100); ax.set_ylim(-0.05, 1.08)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "lapse_profile.pdf"))
    plt.close(fig); print("  Saved lapse_profile.pdf")


def plot_conductance_profile(results_list, labels, figdir=FIGDIR):
    fig, ax = plt.subplots()
    cm = plt.cm.viridis(np.linspace(0.1, 0.9, len(results_list)))
    for res, lab, c in zip(results_list, labels, cm):
        # Normalize by flat-space conductance: kappa_flat = g_n * t0^2
        kf = res["g"][:-1] * 1.0  # t0=1
        ratio = res["kappa"] / kf
        ax.plot(res["r_bond"], ratio, "-", color=c, lw=1.5, label=lab)
        # Analytic: 1 - r_s/r (Schwarzschild conductance = Nbar^2 ~ 1-r_s/r)
        rs = res["r_s"]
        if rs > 1e-6:
            r_an = res["r_bond"]
            an = np.maximum(1.0 - rs / r_an, 0.0)
            ax.plot(r_an, an, "--", color=c, alpha=0.5, lw=1.0)
    ax.plot([], [], "--", color="gray", alpha=0.5,
            label=r"$1-r_s/r$")
    ax.set_xlabel(r"$r / a$")
    ax.set_ylabel(r"$\kappa_n[\Phi]/\kappa_n^{\mathrm{flat}}$")
    ax.set_title("Self-consistent conductance suppression")
    ax.axhline(1, color="gray", lw=0.5, ls=":")
    ax.axhline(0, color="red", lw=0.5, ls="--", alpha=0.3)
    ax.set_xlim(0, 100); ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "conductance_profile.pdf"))
    plt.close(fig); print("  Saved conductance_profile.pdf")


def plot_tqi_profile(results_list, labels, figdir=FIGDIR):
    fig, ax = plt.subplots()
    cm = plt.cm.viridis(np.linspace(0.1, 0.9, len(results_list)))
    has = False
    for res, lab, c in zip(results_list, labels, cm):
        if res["r_s"] < 1e-6: continue
        has = True
        r = res["r"]
        tsc, tpr = res["tqi_sc"], res["tqi_prescribed"]
        # Normalize to flat-space (large r) value
        ns = tsc[-1] if abs(tsc[-1]) > 1e-30 else 1.0
        np_ = tpr[-1] if abs(tpr[-1]) > 1e-30 else 1.0
        ax.plot(r, tsc/ns, "-", color=c, lw=1.5, label=lab)
        ax.plot(r, tpr/np_, "--", color=c, alpha=0.5, lw=1.0)
    if has:
        ax.plot([], [], "--", color="gray", alpha=0.5,
                label=r"Prescribed $f^3/\ln^2(\beta_0 t_0\sqrt{f})$")
    ax.set_xlabel(r"$r / a$")
    ax.set_ylabel(r"$T^{\mathrm{QI}}/T^{\mathrm{QI}}_\infty$")
    ax.set_title(r"Self-consistent vs. prescribed $T^{\mathrm{QI}}$")
    ax.set_xlim(0, 100); ax.set_ylim(-0.05, 1.15)
    if has: ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "tqi_profile.pdf"))
    plt.close(fig); print("  Saved tqi_profile.pdf")


def plot_pn_comparison(results_list, labels, figdir=FIGDIR):
    fig, ax = plt.subplots()
    cm = plt.cm.viridis(np.linspace(0.1, 0.9, len(results_list)))
    has = False
    for res, lab, c in zip(results_list, labels, cm):
        rs = res["r_s"]
        if rs < 0.05: continue
        has = True
        x = res["r_bond"] / rs
        nbar = res["Nbar"]
        residual = res["hrr"] - (1.0 - 1.0/x)
        # Only show where Nbar is above lapse floor (physical region)
        m = (x > 2.0) & (x < 100) & (nbar > 0.02)
        if np.any(m):
            ax.plot(x[m], residual[m], "-", color=c, lw=1.5,
                    label=f"Numerical, {lab}")
    if has:
        ax.axhline(0, color="k", lw=1, alpha=0.4, ls="--",
                   label="Exact Schwarzschild (zero residual)")
    ax.set_xlabel(r"$r / r_s$")
    ax.set_ylabel(r"$h^{rr} - h^{rr}_{\mathrm{Schw}}$")
    ax.set_title(r"Post-Newtonian residual")
    ax.set_xlim(1, 60); ax.legend(fontsize=7)
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "pn_comparison.pdf"))
    plt.close(fig); print("  Saved pn_comparison.pdf")


def plot_cometric_conductance_pn(results_list, labels, figdir=FIGDIR):
    """Combined 3-panel figure: co-metric, conductance suppression, PN residual."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    cm = plt.cm.viridis(np.linspace(0.1, 0.9, len(results_list)))

    # (a) Co-metric
    ax = axes[0]
    has = False
    for res, lab, c in zip(results_list, labels, cm):
        rs = res["r_s"]
        if rs < 1e-6: continue
        has = True
        x = res["r_bond"] / rs
        nbar = res["Nbar"]
        m = (x > 1.0) & (x < 120) & (nbar > 0.02)
        if np.any(m):
            ax.plot(x[m], res["hrr"][m], "-", color=c, lw=1.2, label=lab)
    if has:
        xr = np.linspace(1.5, 120.0, 500)
        ax.plot(xr, 1.0 - 1.0/xr, "k--", lw=1.5, alpha=0.6,
                label=r"$1-r_s/r$")
    ax.set_xlabel(r"$r / r_s$"); ax.set_ylabel(r"$h^{rr}/h_0 = \bar{N}^2$")
    ax.set_title("(a) Emergent co-metric", fontsize=10)
    ax.set_xlim(0.8, 50); ax.set_ylim(-0.05, 1.1)
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.legend(fontsize=6, loc="lower right")

    # (b) Conductance suppression
    ax = axes[1]
    for res, lab, c in zip(results_list, labels, cm):
        kf = res["g"][:-1] * 1.0
        ratio = res["kappa"] / kf
        ax.plot(res["r_bond"], ratio, "-", color=c, lw=1.2, label=lab)
        rs = res["r_s"]
        if rs > 1e-6:
            r_an = res["r_bond"]
            an = np.maximum(1.0 - rs / r_an, 0.0)
            ax.plot(r_an, an, "--", color=c, alpha=0.5, lw=0.8)
    ax.plot([], [], "--", color="gray", alpha=0.5, label=r"$1-r_s/r$")
    ax.set_xlabel(r"$r / a$")
    ax.set_ylabel(r"$\kappa_n[\Phi]/\kappa_n^{\mathrm{flat}}$")
    ax.set_title("(b) Conductance suppression", fontsize=10)
    ax.axhline(1, color="gray", lw=0.5, ls=":")
    ax.set_xlim(0, 100); ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=6)

    # (c) PN residual
    ax = axes[2]
    has = False
    for res, lab, c in zip(results_list, labels, cm):
        rs = res["r_s"]
        if rs < 0.05: continue
        has = True
        x = res["r_bond"] / rs
        nbar = res["Nbar"]
        residual = res["hrr"] - (1.0 - 1.0/x)
        m = (x > 2.0) & (x < 100) & (nbar > 0.02)
        if np.any(m):
            ax.plot(x[m], residual[m], "-", color=c, lw=1.2, label=lab)
    if has:
        ax.axhline(0, color="k", lw=1, alpha=0.4, ls="--",
                   label="Exact Schwarzschild")
    ax.set_xlabel(r"$r / r_s$")
    ax.set_ylabel(r"$h^{rr} - h^{rr}_{\mathrm{Schw}}$")
    ax.set_title("(c) Post-Newtonian residual", fontsize=10)
    ax.set_xlim(1, 60); ax.legend(fontsize=6)
    ax.axhline(0, color="gray", lw=0.5, ls=":")

    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "cometric_conductance_pn.pdf"))
    plt.close(fig); print("  Saved cometric_conductance_pn.pdf")


def plot_max_mass(conv_data, r_core, V0_crit=None, figdir=FIGDIR):
    V0s, fres, mlapse, cflags, GMs, rss = conv_data
    ca = cflags.astype(bool)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    gc, fc = "#2ca02c", "#d62728"

    ax = axes[0]
    for v, f, c in zip(V0s, fres, ca):
        ax.semilogy(v, max(f, 1e-16), "o", color=gc if c else fc, ms=4)
    ax.set_xlabel(r"$V_0/t_0$"); ax.set_ylabel(r"Final $\|\delta\Phi\|/\|\Phi\|$")
    ax.set_title("Picard convergence"); ax.axhline(1e-10, color="gray", ls="--", lw=0.5)

    ax = axes[1]
    for v, ml, c in zip(V0s, mlapse, ca):
        ax.plot(v, ml, "s", color=gc if c else fc, ms=4)
    # Weak-field analytic: N_min ≈ 1 - V0/V0_crit
    if V0_crit is not None:
        v_an = np.linspace(0, V0_crit, 100)
        ax.plot(v_an, 1.0 - v_an / V0_crit, "k--", lw=1, alpha=0.5,
                label=r"$1-V_0/V_0^*$")
    ax.set_xlabel(r"$V_0/t_0$"); ax.set_ylabel(r"$\min_n N_n$")
    ax.set_title("Minimum lapse"); ax.axhline(0, color="red", ls="--", lw=0.8, alpha=0.7)

    ax = axes[2]
    if np.any(ca):
        ax.plot(V0s[ca], rss[ca]/r_core, "o-", color="steelblue", ms=4)
        # Weak-field analytic: r_s ∝ V0 (linear regime)
        # Fit slope from the weakest converged points
        wk = ca & (rss / r_core < 0.5)
        if np.sum(wk) >= 2:
            slope = np.mean((rss[wk] / r_core) / V0s[wk])
            v_an = np.linspace(0, V0s[ca].max(), 100)
            ax.plot(v_an, slope * v_an, "k--", lw=1, alpha=0.5,
                    label=fr"Linear: ${slope:.1f}\,V_0$")
    ax.set_xlabel(r"$V_0/t_0$"); ax.set_ylabel(r"$r_s/r_{\mathrm{core}}$")
    ax.set_title(r"Schwarzschild radius vs. $V_0$")
    ax.axhline(1.0, color="red", ls="--", lw=0.5, alpha=0.5,
               label=r"$r_s=r_{\mathrm{core}}$"); ax.legend(fontsize=8)

    from matplotlib.lines import Line2D
    leg = [Line2D([0],[0], marker="o", color="w", markerfacecolor=gc, ms=6, label="Converged"),
           Line2D([0],[0], marker="o", color="w", markerfacecolor=fc, ms=6, label="Failed"),
           Line2D([0],[0], ls="--", color="k", alpha=0.5, label="Weak-field")]
    for a in axes[:2]: a.legend(handles=leg, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "max_mass.pdf"))
    plt.close(fig); print("  Saved max_mass.pdf")


# ===================================================================
# Main
# ===================================================================

def run_parameter_sweep():
    N = 200; a = 1.0; n_core = 5; beta0 = 0.1; t0 = 1.0
    cstar_sq = t0**2 / 2.0; r_core = n_core * a
    n_arr = np.arange(1, N+1, dtype=float)
    g = 4.0 * np.pi * n_arr**2
    g_core_sum = np.sum(g[:n_core])

    # V0 scale where Phi ~ c*^2 at core
    V0_crit = cstar_sq * 4*np.pi * r_core / (beta0 * g_core_sum)

    print("=" * 65)
    print("Self-consistent closure equation: numerical solver")
    print("=" * 65)
    print(f"  N={N}, n_core={n_core}, r_core={r_core}")
    print(f"  beta0*t0={beta0*t0:.2f}, c*^2={cstar_sq}")
    print(f"  V0_crit (Phi~c*^2 estimate) = {V0_crit:.4f}")
    print()

    # Phase 1: Scan with dense sampling across the full V0 range
    print("Phase 1: Scanning V0 ...")
    # Use log-spaced points at low V0 plus linear at high V0
    V0_low = np.geomspace(0.001, V0_crit * 0.3, 30)
    V0_high = np.linspace(V0_crit * 0.3, V0_crit * 4.0, 50)
    V0_scan = np.unique(np.concatenate([V0_low, V0_high]))
    sd = {k: [] for k in ["V0","GM","r_s","min_lapse","conv","fres"]}

    for V0 in V0_scan:
        Phi, conv, hist = solve_selfconsistent(
            N, t0, V0, n_core, beta0, cstar_sq, g,
            max_iter=400, tol=1e-10, mixing=0.3, verbose=False)
        obs = compute_observables(Phi, N, t0, V0, n_core, beta0, cstar_sq, g, a)
        fr = hist[-1][1] if hist else 1.0
        sd["V0"].append(V0); sd["GM"].append(obs["GM"]); sd["r_s"].append(obs["r_s"])
        sd["min_lapse"].append(obs["lapse"].min()); sd["conv"].append(conv)
        sd["fres"].append(fr)

    for k in sd: sd[k] = np.array(sd[k])
    cm = sd["conv"].astype(bool)
    if np.any(cm):
        print(f"  Converged V0: [{sd['V0'][cm][0]:.4f}, {sd['V0'][cm][-1]:.4f}]")
        print(f"  r_s: [{sd['r_s'][cm].min():.4f}, {sd['r_s'][cm].max():.4f}]")
        print(f"  r_s/r_core: [{sd['r_s'][cm].min()/r_core:.4f}, "
              f"{sd['r_s'][cm].max()/r_core:.4f}]")
        print(f"  min(N): [{sd['min_lapse'][cm].min():.4f}, "
              f"{sd['min_lapse'][cm].max():.4f}]")
    print()

    # Phase 2: Select 5 well-spaced cases by r_s/r_core
    conv_V0 = sd["V0"][cm]; conv_rs = sd["r_s"][cm]; conv_ratio = conv_rs/r_core
    min_ratio = conv_ratio.min() if len(conv_ratio) > 0 else 0
    max_ratio = conv_ratio.max() if len(conv_ratio) > 0 else 1
    # Pick 5 evenly-spaced targets spanning the range
    n_cases = min(5, len(conv_V0))
    if n_cases >= 3:
        # Use log-spacing to get good spread at both ends
        targets = np.geomspace(max(min_ratio, 0.01), max(max_ratio, 0.02), n_cases)
    else:
        targets = np.linspace(min_ratio, max_ratio, n_cases)
    sel_V0, sel_lab = [], []
    used_idx = set()
    for t in targets:
        if len(conv_ratio) == 0: break
        d = np.abs(conv_ratio - t)
        # Don't reuse the same index
        for j in np.argsort(d):
            if j not in used_idx:
                used_idx.add(j)
                sel_V0.append(conv_V0[j])
                sel_lab.append(f"$r_s/r_c={conv_ratio[j]:.2f}$")
                break

    print(f"Phase 2: Detailed solutions for {len(sel_V0)} cases ...")
    results = []
    for V0, lab in zip(sel_V0, sel_lab):
        print(f"\n--- V0={V0:.5f} ({lab}) ---")
        Phi, conv, hist = solve_selfconsistent(
            N, t0, V0, n_core, beta0, cstar_sq, g,
            max_iter=500, tol=1e-12, mixing=0.3, verbose=True)
        obs = compute_observables(Phi, N, t0, V0, n_core, beta0, cstar_sq, g, a)
        obs.update({"cstar_sq": cstar_sq, "V0": V0, "converged": conv, "history": hist})
        results.append(obs)
        if conv:
            print(f"  GM={obs['GM']:.6f}, r_s={obs['r_s']:.4f}, "
                  f"r_s/r_core={obs['r_s']/r_core:.4f}, min(N)={obs['lapse'].min():.6f}")

    # Phase 3: Max-mass sweep
    print(f"\nPhase 3: Fine V0 sweep ...")
    V0f = np.linspace(0.002, V0_crit*4.0, 80)
    mm = {k: [] for k in ["V0","fres","lapse","conv","GM","rs"]}
    for V0 in V0f:
        Phi, conv, hist = solve_selfconsistent(
            N, t0, V0, n_core, beta0, cstar_sq, g,
            max_iter=300, tol=1e-10, mixing=0.3, verbose=False)
        obs = compute_observables(Phi, N, t0, V0, n_core, beta0, cstar_sq, g, a)
        fr = hist[-1][1] if hist else 1.0
        mm["V0"].append(V0); mm["fres"].append(fr)
        mm["lapse"].append(obs["lapse"].min()); mm["conv"].append(conv)
        mm["GM"].append(obs["GM"]); mm["rs"].append(obs["r_s"])
    cdata = tuple(np.array(mm[k]) for k in ["V0","fres","lapse","conv","GM","rs"])

    # Figures
    print("\n" + "=" * 65)
    print("Generating figures ...")
    print("=" * 65)
    plot_potential_profile(results, sel_lab, cstar_sq)
    plot_cometric_comparison(results, sel_lab)
    plot_lapse_profile(results, sel_lab)
    plot_conductance_profile(results, sel_lab)
    plot_tqi_profile(results, sel_lab)
    plot_max_mass(cdata, r_core, V0_crit=V0_crit)
    plot_pn_comparison(results, sel_lab)
    plot_cometric_conductance_pn(results, sel_lab)

    # Summary
    print("\n" + "=" * 65)
    print("Summary:")
    print("=" * 65)
    hdr = f"{'V0':>10s}  {'GM':>10s}  {'r_s':>8s}  {'r_s/r_c':>8s}  {'min(N)':>8s}  {'conv':>5s}"
    print(hdr); print("-" * len(hdr))
    for r in results:
        print(f"{r['V0']:10.5f}  {r['GM']:10.6f}  {r['r_s']:8.4f}  "
              f"{r['r_s']/r_core:8.4f}  {r['lapse'].min():8.6f}  "
              f"{'Y' if r['converged'] else 'N':>5s}")

    mca = np.array(mm["conv"])
    if np.any(mca) and not np.all(mca):
        lc = np.where(mca)[0][-1]
        print(f"\nMax-mass threshold: V0={mm['V0'][lc]:.5f}, "
              f"min(N)={mm['lapse'][lc]:.6f}, r_s/r_c={mm['rs'][lc]/r_core:.4f}")
    elif np.all(mca):
        print(f"\nAll V0 converged up to {mm['V0'][-1]:.4f}")

    print(f"\nFigures: {FIGDIR}")
    return results, cdata


if __name__ == "__main__":
    t0 = time.time()
    results, cdata = run_parameter_sweep()
    print(f"\nTotal: {time.time()-t0:.1f}s")
