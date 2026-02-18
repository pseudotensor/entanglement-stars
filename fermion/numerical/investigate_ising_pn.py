"""
Investigate post-Newtonian coefficients for the 2D Ising model "black hole".

Physics background
------------------
For the free fermion case, the 2PN coefficient alpha_2 = -1/4 is structural:
it comes from the quadratic conductance-lapse coupling F(Nbar) = Nbar^2.
The key question: is the 2PN departure universal or model-specific?

For the 2D Ising model at inverse temperature beta*J, the MI conductance
between adjacent blocks separated by distance ell (in lattice units) is:

    kappa ~ C_conn(ell)^2

where C_conn is the connected spin-spin correlator.  In the high-T phase:

    C_conn(r) ~ exp(-r/xi)

with xi the Onsager correlation length:

    1/xi = -log(tanh(beta*J)) - 2*beta*J

When a lapse modulates the local temperature, beta -> beta_0 * Nbar, the
conductance ratio is:

    F(Nbar) = kappa(beta_0*J*Nbar) / kappa(beta_0*J)
            = exp(-2*ell*[1/xi(Nbar) - 1/xi(1)])

where xi(Nbar) uses the Onsager formula with beta_0*J*Nbar.

Key result: log F is NOT linear in (Nbar-1) because 1/xi is a convex
function of beta*J.  This means F''(1) - F'(1)^2 < 0 always, giving
alpha_2 < 0 for the Ising model too.

This script:
1. Computes F(Nbar) for 2D Ising using the exact Onsager formula
2. Extracts F'(1), F''(1) and predicts alpha_1, alpha_2
3. Scans over beta*J and block size ell
4. Compares with the free fermion case
5. Identifies the universality of the 2PN departure
"""

import numpy as np
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
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# ============================================================
# 1. Onsager correlation length
# ============================================================

def onsager_inv_xi(beta_J):
    """
    Inverse correlation length from the exact Onsager formula:
        1/xi = -log(tanh(beta*J)) - 2*beta*J
    Valid for beta*J < beta_c*J.
    """
    if beta_J <= 0:
        return np.inf
    if beta_J >= 0.5 * np.log(1 + np.sqrt(2)):
        return 0.0  # at or beyond criticality
    return -np.log(np.tanh(beta_J)) - 2 * beta_J


def onsager_xi(beta_J):
    """Correlation length from the Onsager formula."""
    inv = onsager_inv_xi(beta_J)
    if inv <= 0:
        return np.inf
    return 1.0 / inv


# ============================================================
# 2. Ising conductance ratio F(Nbar)
# ============================================================

def ising_F(Nbar, beta0_J, ell):
    """
    Conductance ratio F(Nbar) for the Ising model.

    kappa ~ exp(-2*ell/xi(beta*J*Nbar))
    F(Nbar) = kappa(Nbar) / kappa(1)
            = exp(-2*ell*[1/xi(Nbar) - 1/xi(1)])

    Parameters:
        Nbar: bond-averaged lapse (array or scalar)
        beta0_J: dimensionless coupling at the reference point
        ell: block separation in lattice units
    """
    Nbar = np.atleast_1d(np.asarray(Nbar, dtype=float))
    result = np.zeros_like(Nbar)
    inv_xi_ref = onsager_inv_xi(beta0_J)

    for i, nb in enumerate(Nbar):
        bJ_eff = beta0_J * nb
        inv_xi = onsager_inv_xi(bJ_eff)
        if np.isinf(inv_xi) or np.isinf(inv_xi_ref):
            result[i] = 0.0
        elif inv_xi_ref <= 0:
            result[i] = 1.0
        else:
            result[i] = np.exp(-2 * ell * (inv_xi - inv_xi_ref))

    return result


def ising_F_derivatives(beta0_J, ell, h=1e-5):
    """
    Compute F'(1) and F''(1) for the Ising conductance ratio.

    Uses the exact Onsager formula + numerical derivatives.
    Also computes analytically from d/dN of 1/xi(beta0_J * N).
    """
    # --- Numerical derivatives via central differences ---
    F_p1 = ising_F(1.0 + h, beta0_J, ell)[0]
    F_m1 = ising_F(1.0 - h, beta0_J, ell)[0]
    F_p2 = ising_F(1.0 + 2*h, beta0_J, ell)[0]
    F_m2 = ising_F(1.0 - 2*h, beta0_J, ell)[0]
    F_0 = 1.0  # F(1) = 1 by definition

    Fp_num = (-F_p2 + 8*F_p1 - 8*F_m1 + F_m2) / (12 * h)
    Fpp_num = (-F_p2 + 16*F_p1 - 30*F_0 + 16*F_m1 - F_m2) / (12 * h**2)

    # --- Analytical derivatives ---
    # log F(Nbar) = -2*ell * [phi(Nbar) - phi(1)]
    # where phi(Nbar) = 1/xi(beta0_J * Nbar)
    #
    # phi(N) = -log(tanh(K*N)) - 2*K*N  where K = beta0_J
    #
    # phi'(N) = K * [-2/sinh(2KN) - 2]
    # phi''(N) = K^2 * [4*cosh(2KN)/sinh^2(2KN)]
    K = beta0_J
    x = 2 * K  # = 2*beta0_J at Nbar=1
    sinh_x = np.sinh(x)
    cosh_x = np.cosh(x)

    phi_p = K * (-2.0 / sinh_x - 2.0)
    phi_pp = K**2 * (4.0 * cosh_x / sinh_x**2)

    # F'(1) = -2*ell * phi'(1) * F(1) = -2*ell*phi_p
    Fp_anal = -2 * ell * phi_p
    # F''(1) = -2*ell*phi''(1) + (2*ell*phi'(1))^2
    Fpp_anal = -2 * ell * phi_pp + (2 * ell * phi_p)**2

    return {
        'Fp_num': Fp_num, 'Fpp_num': Fpp_num,
        'Fp_anal': Fp_anal, 'Fpp_anal': Fpp_anal,
        'phi_p': phi_p, 'phi_pp': phi_pp,
    }


# ============================================================
# 3. Free fermion comparison (from investigate_pn_vs_temperature.py)
# ============================================================

def binary_entropy(f):
    f = np.clip(f, 1e-15, 1 - 1e-15)
    return -f * np.log(f) - (1 - f) * np.log(1 - f)


def exact_mi_uniform_chain(beta_t, N_chain=600):
    """Single-channel MI between adjacent sites of a uniform free-fermion chain."""
    from scipy.linalg import eigh_tridiagonal
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


def fermion_F_derivatives(beta0_t0, h=1e-4):
    """F'(1) and F''(1) for the free fermion conductance ratio."""
    mi_0 = exact_mi_uniform_chain(beta0_t0)
    if mi_0 < 1e-30:
        return 2.0, 2.0  # high-T limit

    mi_p1 = exact_mi_uniform_chain(beta0_t0 * (1 + h))
    mi_m1 = exact_mi_uniform_chain(beta0_t0 * (1 - h))
    mi_p2 = exact_mi_uniform_chain(beta0_t0 * (1 + 2*h))
    mi_m2 = exact_mi_uniform_chain(beta0_t0 * (1 - 2*h))

    Fp = (-mi_p2 + 8*mi_p1 - 8*mi_m1 + mi_m2) / (12 * h * mi_0)
    Fpp = (-mi_p2 + 16*mi_p1 - 30*mi_0 + 16*mi_m1 - mi_m2) / (12 * h**2 * mi_0)
    return Fp, Fpp


# ============================================================
# 4. Main investigation
# ============================================================

def main():
    beta_c_J = 0.5 * np.log(1 + np.sqrt(2))

    print("=" * 95)
    print("ISING MODEL POST-NEWTONIAN COEFFICIENTS")
    print("=" * 95)
    print()
    print(f"Critical coupling: beta_c * J = {beta_c_J:.6f}")
    print()
    print("The MI conductance ratio for the 2D Ising model is:")
    print("    F(Nbar) = exp(-2*ell*[1/xi(beta0*J*Nbar) - 1/xi(beta0*J)])")
    print("where xi is the exact Onsager correlation length.")
    print()

    # --------------------------------------------------------
    # Part A: Ising PN coefficients vs temperature at ell=1
    # --------------------------------------------------------
    print("PART A: Ising model PN coefficients (ell=1, single-lattice-spacing blocks)")
    print("-" * 95)
    print(f"{'beta_J':>8s} {'beta/beta_c':>11s} {'xi':>8s} {'Fp':>9s} {'Fpp':>9s} "
          f"{'alpha1':>9s} {'alpha2':>9s} {'Fpp-Fp^2':>9s}")
    print("-" * 95)

    beta_J_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
                     0.38, 0.40, 0.42, 0.43, 0.435, 0.438, 0.4395]
    ell = 1  # single lattice spacing

    ising_results_ell1 = []
    for bJ in beta_J_values:
        xi = onsager_xi(bJ)
        derivs = ising_F_derivatives(bJ, ell)
        Fp = derivs['Fp_anal']
        Fpp = derivs['Fpp_anal']
        alpha1 = -Fp / 2.0
        alpha2 = (Fpp - Fp**2) / 8.0
        ratio = bJ / beta_c_J

        print(f"{bJ:8.4f} {ratio:11.4f} {xi:8.3f} {Fp:9.4f} {Fpp:9.4f} "
              f"{alpha1:9.5f} {alpha2:9.5f} {Fpp-Fp**2:9.4f}")
        ising_results_ell1.append({
            'beta_J': bJ, 'xi': xi, 'ratio': ratio,
            'Fp': Fp, 'Fpp': Fpp,
            'alpha1': alpha1, 'alpha2': alpha2,
        })

    # --------------------------------------------------------
    # Part B: Block-size dependence (ell = 1, 2, 4, 8)
    # --------------------------------------------------------
    print()
    print("PART B: Block-size dependence at fixed beta*J = 0.30")
    print("-" * 95)
    print(f"{'ell':>5s} {'xi/ell':>8s} {'Fp':>9s} {'Fpp':>9s} "
          f"{'alpha1':>9s} {'alpha2':>9s} {'Fpp-Fp^2':>9s}")
    print("-" * 95)

    bJ_fixed = 0.30
    xi_fixed = onsager_xi(bJ_fixed)
    ell_values = [1, 2, 4, 8, 16]
    ising_results_ell = []

    for ell_val in ell_values:
        derivs = ising_F_derivatives(bJ_fixed, ell_val)
        Fp = derivs['Fp_anal']
        Fpp = derivs['Fpp_anal']
        alpha1 = -Fp / 2.0
        alpha2 = (Fpp - Fp**2) / 8.0

        print(f"{ell_val:5d} {xi_fixed/ell_val:8.3f} {Fp:9.4f} {Fpp:9.4f} "
              f"{alpha1:9.5f} {alpha2:9.5f} {Fpp-Fp**2:9.4f}")
        ising_results_ell.append({
            'ell': ell_val, 'xi_over_ell': xi_fixed / ell_val,
            'Fp': Fp, 'Fpp': Fpp,
            'alpha1': alpha1, 'alpha2': alpha2,
        })

    # --------------------------------------------------------
    # Part C: Direct comparison with free fermions
    # --------------------------------------------------------
    print()
    print("PART C: Ising vs Free Fermion comparison (single-site/shell)")
    print("-" * 95)
    print(f"{'T param':>10s} {'Model':>10s} {'Fp':>9s} {'Fpp':>9s} "
          f"{'alpha1':>9s} {'alpha2':>9s}")
    print("-" * 95)

    # Ising at a few representative temperatures (ell=1)
    for bJ in [0.10, 0.20, 0.30, 0.40]:
        derivs = ising_F_derivatives(bJ, 1)
        Fp, Fpp = derivs['Fp_anal'], derivs['Fpp_anal']
        a1, a2 = -Fp/2, (Fpp - Fp**2)/8
        print(f"{'bJ='+f'{bJ:.2f}':>10s} {'Ising':>10s} {Fp:9.4f} {Fpp:9.4f} "
              f"{a1:9.5f} {a2:9.5f}")

    # Free fermions at corresponding dimensionless temperatures
    for bt in [0.01, 0.10, 0.50, 1.00]:
        Fp, Fpp = fermion_F_derivatives(bt)
        a1, a2 = -Fp/2, (Fpp - Fp**2)/8
        print(f"{'bt='+f'{bt:.2f}':>10s} {'Fermion':>10s} {Fp:9.4f} {Fpp:9.4f} "
              f"{a1:9.5f} {a2:9.5f}")

    print("-" * 95)

    # --------------------------------------------------------
    # Part D: The universality argument
    # --------------------------------------------------------
    print()
    print("=" * 95)
    print("UNIVERSALITY ANALYSIS")
    print("=" * 95)
    print()
    print("The key quantity is F''(1) - F'(1)^2, which equals 8*alpha_2.")
    print()
    print("For the Ising model with blocks of size ell:")
    print("    log F(Nbar) = -2*ell * [phi(Nbar) - phi(1)]")
    print("    where phi(N) = 1/xi(beta0*J*N) is the inverse correlation length.")
    print()
    print("    F''(1) - F'(1)^2 = -2*ell * phi''(1)")
    print()
    print("    phi''(1) = (beta0*J)^2 * 4*cosh(2*beta0*J) / sinh^2(2*beta0*J)")
    print()
    print("Since phi''(1) > 0 always (1/xi is convex), F''(1) - F'(1)^2 < 0,")
    print("and therefore alpha_2 < 0 for the Ising model at ALL temperatures")
    print("and ALL block sizes.")
    print()
    print("For free fermions:")
    print("    F(Nbar) = MI(beta0*Nbar*t0) / MI(beta0*t0)")
    print("    MI is bounded (MI <= 2*ln(2) per channel), so F is bounded,")
    print("    and d^2(log F)/dN^2 < 0 at N=1 (concavity of bounded functions)")
    print("    => F''(1) - F'(1)^2 < 0 => alpha_2 < 0")
    print()
    print("UNIVERSAL MECHANISM: For ANY model where the MI conductance comes from")
    print("a connected correlator that decays with distance,")
    print("    kappa ~ exp(-2*d/xi(beta))")
    print("the logarithm of F(Nbar) inherits the concavity of -1/xi(beta*Nbar).")
    print("Since 1/xi is convex in beta for standard statistical mechanics models")
    print("(the inverse correlation length has positive curvature away from Tc),")
    print("alpha_2 < 0 universally.")
    print()
    print("Schwarzschild (alpha_2 = 0) would require d^2(1/xi)/d(beta)^2 = 0")
    print("at the operating temperature -- an inflection point of the inverse")
    print("correlation length -- which does not occur in the 2D Ising model")
    print("or in free fermions.")

    # --------------------------------------------------------
    # Part E: Plots
    # --------------------------------------------------------
    outdir = "fermion/numerical"

    # --- Plot 1: F(Nbar) comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    Nbar_plot = np.linspace(0.5, 1.5, 200)

    ax = axes[0]
    # Ising at various temperatures (ell=1)
    for bJ in [0.10, 0.20, 0.30, 0.40]:
        F_vals = ising_F(Nbar_plot, bJ, 1)
        xi = onsager_xi(bJ)
        ax.plot(Nbar_plot, F_vals,
                label=rf'Ising $\beta J={bJ:.2f}$ ($\xi={xi:.1f}$)')

    # Free fermion high-T
    ax.plot(Nbar_plot, Nbar_plot**2, 'k--', lw=1.5,
            label=r'Fermion high-T: $\bar{N}^2$')

    # Schwarzschild reference (exponential)
    ax.plot(Nbar_plot, np.exp(2*(Nbar_plot - 1)), 'k:', lw=1.5,
            label=r'Schwarzschild: $e^{2(\bar{N}-1)}$')

    ax.set_xlabel(r'$\bar{N}$ (bond-averaged lapse)')
    ax.set_ylabel(r'$F(\bar{N})$')
    ax.set_title(r'Conductance ratio: Ising ($\ell=1$) vs Fermion')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 3)

    # --- Plot 2: alpha_2 comparison ---
    ax = axes[1]

    # Ising alpha_2 vs beta_J
    betas_i = [r['beta_J'] for r in ising_results_ell1]
    a2_i = [r['alpha2'] for r in ising_results_ell1]
    ax.plot(betas_i, a2_i, 'bo-', lw=2, ms=5, label=r'Ising ($\ell=1$)')

    # Free fermion alpha_2 vs beta_0*t_0 (on a separate axis concept,
    # but we'll put them on the same plot since both are alpha_2 vs coupling)
    ff_betas = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.3]
    ff_a2 = []
    for bt in ff_betas:
        Fp, Fpp = fermion_F_derivatives(bt)
        ff_a2.append((Fpp - Fp**2) / 8.0)
    ax.plot(ff_betas, ff_a2, 'rs-', lw=2, ms=5, label=r'Fermion')

    ax.axhline(0.0, color='green', ls='--', alpha=0.6, label='Schwarzschild')
    ax.axhline(-0.25, color='gray', ls=':', alpha=0.5, label=r'$-1/4$ (high-T limit)')
    ax.set_xlabel(r'Coupling ($\beta J$ for Ising, $\beta_0 t_0$ for Fermion)')
    ax.set_ylabel(r'$\alpha_2$ (2PN coefficient)')
    ax.set_title(r'2PN departure: always $\alpha_2 < 0$')
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.axvline(beta_c_J, color='blue', ls=':', alpha=0.3,
               label=r'Ising $\beta_c$')

    plt.tight_layout()
    plt.savefig(f"{outdir}/ising_pn_comparison.pdf", dpi=150, bbox_inches='tight')
    plt.savefig(f"{outdir}/ising_pn_comparison.png", dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {outdir}/ising_pn_comparison.pdf")

    # --- Plot 3: Block-size dependence and convexity ---
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes2[0]
    # alpha_2 vs ell at fixed beta*J
    ells = [r['ell'] for r in ising_results_ell]
    a2_ells = [r['alpha2'] for r in ising_results_ell]
    ax.plot(ells, a2_ells, 'ko-', lw=2, ms=7)
    ax.set_xlabel(r'Block size $\ell$ (lattice units)')
    ax.set_ylabel(r'$\alpha_2$')
    ax.set_title(rf'$\alpha_2$ vs block size ($\beta J = {bJ_fixed}$, $\xi = {xi_fixed:.2f}$)')
    ax.axhline(0, color='green', ls='--', alpha=0.5, label='Schwarzschild')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # --- Convexity of 1/xi ---
    ax = axes2[1]
    bJ_range = np.linspace(0.01, 0.435, 300)
    inv_xi_vals = [onsager_inv_xi(bJ) for bJ in bJ_range]
    ax.plot(bJ_range, inv_xi_vals, 'k-', lw=2)
    ax.set_xlabel(r'$\beta J$')
    ax.set_ylabel(r'$1/\xi$ (inverse correlation length)')
    ax.set_title(r'Onsager $1/\xi$ is convex $\Rightarrow \alpha_2 < 0$')
    ax.axvline(beta_c_J, color='red', ls='--', alpha=0.5, label=r'$\beta_c J$')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{outdir}/ising_pn_blocksize.pdf", dpi=150, bbox_inches='tight')
    plt.savefig(f"{outdir}/ising_pn_blocksize.png", dpi=150, bbox_inches='tight')
    print(f"Plot saved: {outdir}/ising_pn_blocksize.pdf")

    # --- Plot 4: F(Nbar) at different block sizes ---
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    for ell_val in [1, 2, 4, 8]:
        F_vals = ising_F(Nbar_plot, bJ_fixed, ell_val)
        ax3.plot(Nbar_plot, F_vals,
                 label=rf'$\ell={ell_val}$, $\xi/\ell={xi_fixed/ell_val:.2f}$')

    ax3.plot(Nbar_plot, np.exp(2*(Nbar_plot-1)), 'k:', lw=1.5,
             label=r'Schwarzschild: $e^{2(\bar{N}-1)}$')
    ax3.set_xlabel(r'$\bar{N}$')
    ax3.set_ylabel(r'$F(\bar{N})$')
    ax3.set_title(rf'Ising $F(\bar{{N}})$ at $\beta J = {bJ_fixed}$ vs block size')
    ax3.legend(loc='upper left', framealpha=0.9)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 4)

    plt.tight_layout()
    plt.savefig(f"{outdir}/ising_pn_F_blocksize.pdf", dpi=150, bbox_inches='tight')
    plt.savefig(f"{outdir}/ising_pn_F_blocksize.png", dpi=150, bbox_inches='tight')
    print(f"Plot saved: {outdir}/ising_pn_F_blocksize.pdf")


if __name__ == '__main__':
    main()
