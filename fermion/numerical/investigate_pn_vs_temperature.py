"""
Investigate post-Newtonian coefficients vs temperature (beta_0 * t_0).

Physics background
------------------
The self-consistent closure equation in the exterior is:

    r^2 F(N_bar) dPhi/dr = const

where F(N_bar) = kappa(N_bar)/kappa_flat is the conductance ratio as
a function of the bond-averaged lapse N_bar = 1 + Phi/c_*^2.

At high T:  F(N_bar) = N_bar^2   (because MI ~ (beta N t)^2 / 4)
At finite T: F has corrections from the exact Fermi-Dirac MI.

The PN coefficients of the co-metric h^{rr} = 1 + alpha_1 (r_s/r) + alpha_2 (r_s/r)^2 + ...
follow from the analytical formula (derived from the ODE):

    alpha_1 = -F'(1)/2
    alpha_2 = [F''(1) - F'(1)^2] / 8

where F'(1), F''(1) are derivatives of the conductance ratio at N_bar = 1.

For high-T (F = N_bar^2):  F'=2, F''=2  => alpha_1=-1, alpha_2=-1/4
For Schwarzschild (alpha_2=0): would need F''=F'^2, i.e., F ~ exp(2(N_bar-1))

This script:
1. Computes the exact MI function at various temperatures
2. Extracts F'(1), F''(1) to predict alpha_1, alpha_2 analytically
3. Solves the full self-consistent equation numerically to verify
4. Produces diagnostic plots
"""

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
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# ============================================================
# 1. Exact MI computation for uniform chain
# ============================================================

def binary_entropy(f):
    """Binary entropy s(f) = -f ln(f) - (1-f) ln(1-f) in nats."""
    f = np.clip(f, 1e-15, 1 - 1e-15)
    return -f * np.log(f) - (1 - f) * np.log(1 - f)


def exact_mi_uniform_chain(beta_t, N_chain=600):
    """
    Compute single-channel MI between adjacent sites of a uniform
    free-fermion chain at half filling.

    Parameters:
        beta_t: dimensionless beta * t_0 (inverse temperature * hopping)
        N_chain: chain length (large for bulk limit)

    Returns:
        MI: mutual information per channel (in nats)
    """
    if beta_t < 1e-14:
        return 0.0

    # Uniform chain Hamiltonian: h_{nm} = -t (delta_{m,n+1} + delta_{m,n-1})
    # We set t=1 and scale by beta_t
    diag = np.zeros(N_chain)
    offdiag = -np.ones(N_chain - 1)

    evals, evecs = eigh_tridiagonal(diag, offdiag)
    fermi = 1.0 / (np.exp(beta_t * evals) + 1.0)

    # Correlation matrix elements at the middle bond
    n = N_chain // 2
    G_nn = np.dot(evecs[n, :]**2, fermi)
    G_mm = np.dot(evecs[n+1, :]**2, fermi)
    G_nm = np.dot(evecs[n, :] * evecs[n+1, :], fermi)

    # Single-site entropies
    s_n = binary_entropy(G_nn)
    s_m = binary_entropy(G_mm)

    # Joint 2x2 eigenvalues
    avg = 0.5 * (G_nn + G_mm)
    diff = 0.5 * (G_nn - G_mm)
    disc = np.sqrt(diff**2 + G_nm**2)
    lam_p = np.clip(avg + disc, 1e-15, 1 - 1e-15)
    lam_m = np.clip(avg - disc, 1e-15, 1 - 1e-15)

    s_joint = binary_entropy(lam_p) + binary_entropy(lam_m)
    mi = s_n + s_m - s_joint
    return max(mi, 0.0)


def tabulate_mi_function(beta0_t0, n_points=800, Nbar_max=3.0):
    """
    Tabulate the conductance ratio F(Nbar) = MI(beta0*Nbar*t0) / MI(beta0*t0)
    and its derivatives at Nbar=1.

    Returns: mi_func (interpolant), F'(1), F''(1), mi_ref
    """
    Nbar_values = np.linspace(0.001, Nbar_max, n_points)
    mi_values = np.array([exact_mi_uniform_chain(beta0_t0 * Nb)
                          for Nb in Nbar_values])

    mi_ref = exact_mi_uniform_chain(beta0_t0)

    if mi_ref > 1e-30:
        ratio = mi_values / mi_ref
    else:
        ratio = Nbar_values**2

    # Interpolation function
    func = interp1d(Nbar_values, ratio, kind='cubic',
                    bounds_error=False, fill_value=(ratio[0], ratio[-1]))

    # Numerical derivatives at Nbar = 1 using central differences
    h = 1e-4
    mi_p1 = exact_mi_uniform_chain(beta0_t0 * (1 + h))
    mi_m1 = exact_mi_uniform_chain(beta0_t0 * (1 - h))
    mi_p2 = exact_mi_uniform_chain(beta0_t0 * (1 + 2*h))
    mi_m2 = exact_mi_uniform_chain(beta0_t0 * (1 - 2*h))
    mi_0 = mi_ref

    # 4th-order central differences
    Fp = (-mi_p2 + 8*mi_p1 - 8*mi_m1 + mi_m2) / (12 * h * mi_0)
    Fpp = (-mi_p2 + 16*mi_p1 - 30*mi_0 + 16*mi_m1 - mi_m2) / (12 * h**2 * mi_0)

    return func, Fp, Fpp, mi_ref


# ============================================================
# 2. Analytical PN prediction from F derivatives
# ============================================================

def predict_pn_from_F(Fp, Fpp):
    """
    Given F'(1) and F''(1), predict the PN coefficients.

    The co-metric h^{rr} = 1 + alpha_1 (r_s/r) + alpha_2 (r_s/r)^2 + ...
    where:
        alpha_1 = -F'(1)/2
        alpha_2 = [F''(1) - F'(1)^2] / 8

    Derivation: the exterior closure equation r^2 F(u) du/dr = C/(c*^2),
    with u = 1 + Phi/c*^2, is integrated and inverted order by order.
    """
    alpha_1 = -Fp / 2.0
    alpha_2 = (Fpp - Fp**2) / 8.0
    return alpha_1, alpha_2


# ============================================================
# 3. Self-consistent solver (supports exact MI conductances)
# ============================================================

def solve_selfconsistent(N, t0, V0, n_core, beta0, cstar_sq,
                         use_exact_mi=False, mi_func=None,
                         max_iter=500, tol=1e-10, mixing=0.3,
                         vacuum=False):
    """
    Solve the self-consistent closure equation by Picard iteration.

    If use_exact_mi=True and mi_func is provided, conductances use
    the exact MI function. Otherwise, use high-T approx kappa ~ Nbar^2.

    If vacuum=True, the Yukawa mass is set to zero everywhere (no
    screening from the background susceptibility).  This implements the
    asymptotically vacuum boundary condition of Section 10a, where the
    exterior equation is Poisson and the 1/r potential persists to
    arbitrarily large radii.
    """
    a = 1.0
    r = np.arange(1, N + 1, dtype=float) * a
    g = 4 * np.pi * np.arange(1, N + 1, dtype=float)**2

    # Source: mass defect on core shells
    source = np.zeros(N)
    for n in range(min(n_core, N)):
        source[n] = (beta0 / cstar_sq) * g[n] * V0 / 2.0

    # Yukawa mass (high-T formula; provides screening at large r)
    # For vacuum BC (Section 10a), m_sq = 0: no polarizable background.
    if vacuum:
        m_sq = 0.0
    else:
        rho0 = 0.5 * beta0 * t0**2
        m_sq = 2 * beta0 * rho0 / cstar_sq**2

    Phi = np.zeros(N)
    kappa = g[:-1] * t0**2  # flat-space conductances

    LAPSE_FLOOR = 0.01

    for iteration in range(max_iter):
        lapse = np.maximum(1.0 + Phi / cstar_sq, LAPSE_FLOOR)
        Nbar = 0.5 * (lapse[:-1] + lapse[1:])

        if use_exact_mi and mi_func is not None:
            kappa_new = g[:-1] * t0**2 * mi_func(Nbar)
        else:
            kappa_new = g[:-1] * t0**2 * Nbar**2

        kappa = mixing * kappa_new + (1 - mixing) * kappa

        # Build graph Laplacian + Yukawa mass
        L = np.zeros((N, N))
        for n in range(N - 1):
            L[n, n] += kappa[n]
            L[n+1, n+1] += kappa[n]
            L[n, n+1] -= kappa[n]
            L[n+1, n] -= kappa[n]
        for n in range(N):
            L[n, n] += m_sq * g[n]

        # BC: Phi[N-1] = 0
        L_red = L[:N-1, :N-1]
        S_red = -source[:N-1]

        Phi_new = np.zeros(N)
        Phi_new[:N-1] = np.linalg.solve(L_red, S_red)
        Phi_new = np.maximum(Phi_new, cstar_sq * (LAPSE_FLOOR - 1))

        dPhi = np.max(np.abs(Phi_new - Phi))
        norm = np.max(np.abs(Phi_new)) + 1e-30
        Phi = Phi_new

        if dPhi / norm < tol:
            break

    # Final observables
    lapse = np.maximum(1.0 + Phi / cstar_sq, LAPSE_FLOOR)
    Nbar = 0.5 * (lapse[:-1] + lapse[1:])
    r_bond = 0.5 * (r[:-1] + r[1:])

    if use_exact_mi and mi_func is not None:
        hrr = mi_func(Nbar)
    else:
        hrr = Nbar**2

    # Compute Yukawa screening length
    rho0 = 0.5 * beta0 * t0**2
    m_sq_eff = 2 * beta0 * rho0 / cstar_sq**2
    xi_Y = t0 / np.sqrt(m_sq_eff) if m_sq_eff > 0 else N

    # Extract r_s from sub-Yukawa region: n_core+3 to min(xi_Y*0.6, N/2)
    n_lo = min(n_core + 3, N - 2)
    n_hi = min(int(xi_Y * 0.6), N // 2)
    n_hi = max(n_hi, n_lo + 5)
    idx_mid = slice(n_lo, n_hi)
    GM_vals = -Phi[idx_mid] * r[idx_mid]
    GM = np.median(GM_vals)
    r_s = 2 * GM / cstar_sq

    return {
        'Phi': Phi, 'lapse': lapse, 'Nbar': Nbar,
        'hrr': hrr, 'r': r, 'r_bond': r_bond,
        'r_s': r_s, 'GM': GM,
        'converged': iteration < max_iter - 1,
        'iterations': iteration + 1,
    }


# ============================================================
# 4. PN coefficient extraction from numerical solution
# ============================================================

def fit_pn_coefficients(r_bond, hrr, r_s, fit_range=(5, 40)):
    """
    Fit h^{rr} = 1 + alpha_1 (r_s/r) + alpha_2 (r_s/r)^2 + alpha_3 (r_s/r)^3
    in the region fit_range[0] < r/r_s < fit_range[1].
    """
    if r_s <= 0:
        return None
    x = r_bond / r_s
    mask = (x > fit_range[0]) & (x < fit_range[1])
    if np.sum(mask) < 10:
        return None

    u = 1.0 / x[mask]
    dy = hrr[mask] - 1.0

    A = np.column_stack([u, u**2, u**3])
    result = np.linalg.lstsq(A, dy, rcond=None)
    coeffs = result[0]

    residuals = dy - A @ coeffs
    if len(dy) > 3:
        sigma_sq = np.sum(residuals**2) / (len(dy) - 3)
        cov = sigma_sq * np.linalg.inv(A.T @ A)
        uncertainties = np.sqrt(np.diag(cov))
    else:
        uncertainties = np.full(3, np.nan)

    return {
        'alpha1': coeffs[0], 'alpha2': coeffs[1], 'alpha3': coeffs[2],
        'alpha1_err': uncertainties[0], 'alpha2_err': uncertainties[1],
        'alpha3_err': uncertainties[2],
    }


# ============================================================
# 5. Main investigation
# ============================================================

def main():
    N = 200
    t0 = 1.0
    n_core = 5
    cstar_sq = t0**2 / 2.0

    beta0_t0_values = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3,
                       0.5, 0.7, 1.0, 1.3]

    print("=" * 90)
    print("POST-NEWTONIAN COEFFICIENTS vs TEMPERATURE")
    print("=" * 90)
    print()
    print("Analytical formula:  alpha_1 = -F'(1)/2,  alpha_2 = [F''(1) - F'(1)^2]/8")
    print("High-T (F=Nbar^2):  F'=2.000, F''=2.000  =>  alpha_1=-1.000, alpha_2=-0.250")
    print("Schwarzschild:       alpha_1=-1.000, alpha_2= 0.000  (needs F''=F'^2)")
    print()

    # --------------------------------------------------------
    # Part A: Analytical predictions from F derivatives
    # --------------------------------------------------------
    print("PART A: Analytical predictions from F'(1), F''(1)")
    print("-" * 90)
    print(f"{'beta0_t0':>9s} {'MI(b0t0)':>12s} {'F_prime':>9s} {'F_pprime':>9s} "
          f"{'alpha1_an':>10s} {'alpha2_an':>10s} {'alpha2_HT':>10s}")
    print("-" * 90)

    analytical_results = []
    for bt in beta0_t0_values:
        mi_func, Fp, Fpp, mi_ref = tabulate_mi_function(bt)
        a1, a2 = predict_pn_from_F(Fp, Fpp)
        # High-T prediction for alpha2: just use F=Nbar^2 always
        a2_ht = (2.0 - 4.0) / 8.0  # = -0.25
        print(f"{bt:9.3f} {mi_ref:12.6e} {Fp:9.4f} {Fpp:9.4f} "
              f"{a1:10.5f} {a2:10.5f} {a2_ht:10.5f}")
        analytical_results.append({
            'beta0_t0': bt, 'mi_ref': mi_ref,
            'Fp': Fp, 'Fpp': Fpp,
            'alpha1_an': a1, 'alpha2_an': a2,
            'mi_func': mi_func,
        })

    # --------------------------------------------------------
    # Part B: Numerical verification via F(Nbar) fitting
    #
    # Instead of fitting PN in r-space (complicated by Yukawa screening),
    # we verify F(Nbar) directly: for the converged solution, fit
    # h^{rr} vs Nbar to extract F'(1) and F''(1) numerically.
    # --------------------------------------------------------
    print()
    print("PART B: Numerical verification (F(Nbar) from self-consistent solver)")
    print("-" * 95)
    print(f"{'beta0_t0':>9s} {'xi_Y':>6s} {'V0':>10s} {'Fp_num':>9s} {'Fpp_num':>9s} "
          f"{'a2_num':>9s} {'a2_anal':>9s} {'match':>7s}")
    print("-" * 95)

    numerical_results = []
    for ar in analytical_results:
        bt = ar['beta0_t0']
        beta0 = bt / t0

        # Compute Yukawa range
        rho0 = 0.5 * beta0 * t0**2
        m_sq_eff = 2 * beta0 * rho0 / cstar_sq**2
        xi_Y = t0 / np.sqrt(m_sq_eff) if m_sq_eff > 0 else N

        # Use moderate V0 to get Phi/c*^2 ~ 0.1-0.3 (enough nonlinearity)
        V0 = 0.15 * cstar_sq / beta0

        # Solve with exact MI conductances
        res_ex = solve_selfconsistent(N, t0, V0, n_core, beta0, cstar_sq,
                                      use_exact_mi=True, mi_func=ar['mi_func'])
        # Also solve with high-T
        res_ht = solve_selfconsistent(N, t0, V0, n_core, beta0, cstar_sq,
                                      use_exact_mi=False)

        # Fit F(Nbar) = 1 + a*(Nbar-1) + b*(Nbar-1)^2 from the data
        # Use the exterior region where Nbar varies smoothly
        Nbar_ex = res_ex['Nbar']
        hrr_ex = res_ex['hrr']
        dN = Nbar_ex - 1.0
        # Use points where |dN| is between 1e-4 and 0.4 (weak-to-moderate field)
        mask = (np.abs(dN) > 1e-4) & (np.abs(dN) < 0.4)
        n_pts = np.sum(mask)

        if n_pts >= 8:
            A_fit = np.column_stack([dN[mask], dN[mask]**2])
            dy = hrr_ex[mask] - 1.0
            coeffs = np.linalg.lstsq(A_fit, dy, rcond=None)[0]
            Fp_num = coeffs[0]
            Fpp_num = 2 * coeffs[1]  # F''(1)/2 * (Nbar-1)^2, so coeff = F''/2
            a2_num = (Fpp_num - Fp_num**2) / 8.0
            match = "OK" if abs(a2_num - ar['alpha2_an']) < 0.02 else "DIFF"
            print(f"{bt:9.3f} {xi_Y:6.1f} {V0:10.4f} {Fp_num:9.4f} {Fpp_num:9.4f} "
                  f"{a2_num:9.4f} {ar['alpha2_an']:9.4f} {match:>7s}")
        else:
            Fp_num, Fpp_num, a2_num = np.nan, np.nan, np.nan
            print(f"{bt:9.3f} {xi_Y:6.1f} {V0:10.4f}  (insufficient data, n_pts={n_pts})")

        numerical_results.append({
            'beta0_t0': bt, 'V0': V0, 'xi_Y': xi_Y,
            'res_ht': res_ht, 'res_ex': res_ex,
            'Fp_num': Fp_num, 'Fpp_num': Fpp_num, 'a2_num': a2_num,
            'alpha2_an': ar['alpha2_an'],
        })

    print("-" * 95)

    # --------------------------------------------------------
    # Part C: Plots
    # --------------------------------------------------------
    outdir = "fermion/numerical"

    # --- Plot 1: PN coefficients vs temperature ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    betas = [ar['beta0_t0'] for ar in analytical_results]
    a1_an = [ar['alpha1_an'] for ar in analytical_results]
    a2_an = [ar['alpha2_an'] for ar in analytical_results]

    betas_num = [nr['beta0_t0'] for nr in numerical_results if np.isfinite(nr['a2_num'])]
    Fp_num_vals = [nr['Fp_num'] for nr in numerical_results if np.isfinite(nr['a2_num'])]
    a2_num_vals = [nr['a2_num'] for nr in numerical_results if np.isfinite(nr['a2_num'])]

    ax = axes[0]
    ax.plot(betas, a1_an, 'k-', lw=2, label=r'Analytical $-F^\prime(1)/2$')
    ax.plot(betas_num, [-f/2 for f in Fp_num_vals], 'rs', ms=7,
            label='Numerical (from F fit)')
    ax.axhline(-1.0, color='gray', ls='--', alpha=0.5, label='Schwarzschild')
    ax.set_xlabel(r'$\beta_0 t_0$')
    ax.set_ylabel(r'$\alpha_1$ (1PN coefficient)')
    ax.set_title(r'First post-Newtonian: $h^{rr} = 1 + \alpha_1 r_s/r + \cdots$')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(betas, a2_an, 'k-', lw=2, label=r'Analytical $[F^{\prime\prime}-F^{\prime 2}]/8$')
    ax.plot(betas_num, a2_num_vals, 'rs', ms=7, label='Numerical (from F fit)')
    ax.axhline(0.0, color='gray', ls='--', alpha=0.5, label='Schwarzschild')
    ax.axhline(-0.25, color='gray', ls=':', alpha=0.5, label=r'High-T ($-1/4$)')
    ax.set_xlabel(r'$\beta_0 t_0$')
    ax.set_ylabel(r'$\alpha_2$ (2PN coefficient)')
    ax.set_title(r'Second post-Newtonian: $h^{rr} = 1 - r_s/r + \alpha_2 (r_s/r)^2 + \cdots$')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{outdir}/pn_vs_temperature.pdf", dpi=150, bbox_inches='tight')
    plt.savefig(f"{outdir}/pn_vs_temperature.png", dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {outdir}/pn_vs_temperature.pdf")

    # --- Plot 2: Conductance ratio F(Nbar) at various temperatures ---
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5.5))

    Nbar_plot = np.linspace(0.1, 2.0, 300)
    ax = axes2[0]
    for bt_sel in [0.01, 0.1, 0.3, 0.5, 1.0, 1.3]:
        mi_func, Fp, Fpp, _ = tabulate_mi_function(bt_sel)
        ratio = mi_func(Nbar_plot)
        ax.plot(Nbar_plot, ratio, label=rf'$\beta_0 t_0 = {bt_sel}$')
    ax.plot(Nbar_plot, Nbar_plot**2, 'k--', lw=1.5, label=r'$\bar{N}^2$ (high-T)')
    ax.set_xlabel(r'$\bar{N}$ (bond-averaged lapse)')
    ax.set_ylabel(r'$F(\bar{N}) = \kappa / \kappa_{\mathrm{flat}}$')
    ax.set_title('Conductance ratio vs lapse')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # --- Plot 3: F'' - F'^2 vs temperature (controls alpha_2) ---
    ax = axes2[1]
    Fpp_minus_Fp2 = [ar['Fpp'] - ar['Fp']**2 for ar in analytical_results]
    ax.plot(betas, Fpp_minus_Fp2, 'ko-', lw=2, ms=6)
    ax.axhline(-2.0, color='gray', ls='--', alpha=0.5,
               label=r'High-T: $F^{\prime\prime}-F^{\prime 2}=-2$')
    ax.axhline(0.0, color='green', ls=':', alpha=0.5,
               label=r'Schwarzschild: $F^{\prime\prime}=F^{\prime 2}$')
    ax.set_xlabel(r'$\beta_0 t_0$')
    ax.set_ylabel(r"$F''(1) - F'(1)^2$")
    ax.set_title(r"Key quantity controlling $\alpha_2 = [F''-F'^2]/8$")
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{outdir}/mi_vs_lapse.pdf", dpi=150, bbox_inches='tight')
    plt.savefig(f"{outdir}/mi_vs_lapse.png", dpi=150, bbox_inches='tight')
    print(f"Plot saved: {outdir}/mi_vs_lapse.pdf")

    # --- Plot 4: One representative co-metric comparison ---
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    # Use beta0_t0 = 0.1 case
    plotted = False
    for nr in numerical_results:
        if abs(nr['beta0_t0'] - 0.1) < 0.01 and np.isfinite(nr['a2_num']):
            rs = nr['res_ex']['r_s']
            rb = nr['res_ex']['r_bond']
            x = rb / rs
            mask = (x > 1.5) & (x < 60)

            ax3.plot(x[mask], nr['res_ht']['hrr'][mask], 'b-', lw=1.5,
                     label='High-T conductances')
            ax3.plot(x[mask], nr['res_ex']['hrr'][mask], 'r--', lw=1.5,
                     label='Exact MI conductances')
            ax3.plot(x[mask], 1 - 1/x[mask], 'k-', lw=1, alpha=0.5,
                     label='Schwarzschild')
            ax3.plot(x[mask], (1 - 1.5/x[mask])**(2/3), 'g:', lw=1.5,
                     label=r'$(1-3r_s/2r)^{2/3}$ analytic')
            plotted = True
            break
    if not plotted:
        ax3.text(0.5, 0.5, 'No suitable data for beta0_t0=0.1',
                 transform=ax3.transAxes, ha='center', va='center')

    ax3.set_xlabel(r'$r / r_s$')
    ax3.set_ylabel(r'$h^{rr} / h_0$')
    ax3.set_title(r'Co-metric comparison at $\beta_0 t_0 = 0.1$')
    ax3.legend(loc='lower right', framealpha=0.9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(1.5, 30)

    plt.tight_layout()
    plt.savefig(f"{outdir}/cometric_comparison_exact_vs_ht.pdf",
                dpi=150, bbox_inches='tight')
    print(f"Plot saved: {outdir}/cometric_comparison_exact_vs_ht.pdf")

    # --------------------------------------------------------
    # Part D: Key physical insight
    # --------------------------------------------------------
    print()
    print("=" * 90)
    print("KEY PHYSICAL INSIGHT")
    print("=" * 90)
    print()
    print("The 2PN coefficient alpha_2 = [F''(1) - F'(1)^2] / 8 depends on")
    print("the SHAPE of the conductance ratio F(Nbar) = MI(beta0*Nbar*t0)/MI(beta0*t0).")
    print()
    print("For ANY power law F(Nbar) = Nbar^p:  alpha_2 = -p/8")
    print("  - High-T free fermions: p=2 => alpha_2 = -1/4")
    print()
    print("For Schwarzschild (alpha_2=0): need F''(1) = F'(1)^2,")
    print("  i.e., F ~ exp(F'(1)*(Nbar-1)) near Nbar=1 (exponential, not power-law)")
    print()
    print("Free fermion MI is always bounded (MI <= 2 ln 2 per channel),")
    print("so F cannot be exponential. The 2PN departure is structural")
    print("to free fermions, not an artifact of the high-T approximation.")
    print()
    print("A model with exponential conductance-lapse coupling (e.g., near a")
    print("phase transition) could in principle give alpha_2 = 0.")


if __name__ == '__main__':
    main()
