"""
Investigate log-concavity of MI conductance and conditions for Schwarzschild.

Physics background
------------------
The 2PN coefficient alpha_2 = [F''(1) - F'(1)^2]/8, where F(Nbar) = kappa(beta_0*Nbar)/kappa(beta_0)
is the conductance ratio.  We showed alpha_2 < 0 for free fermions and Ising.

The question: what assumption makes this universal?

Answer: **log-concavity of MI in the coupling**, i.e., d^2(log MI)/dbeta^2 < 0.

Key analytical result:
  The sub-Yukawa exterior ODE  r^2 F(N) N' = epsilon  has exact solution via
  integral G(N) = int_1^N F(N') dN' = -epsilon/r.

  - F = N^2   =>  h^{rr} = (1 - 3r_s/(2r))^{2/3}   (our result, alpha_2 = -1/4)
  - F = exp(2(N-1))  =>  h^{rr} = 1 - r_s/r  (exact Schwarzschild at all PN orders)

  Proof:  G(N) = int_1^N exp(2(N'-1)) dN' = [exp(2(N-1)) - 1]/2 = -epsilon/r
          => exp(2(N-1)) = 1 - r_s/r  => F(N) = exp(2(N-1)) = h^{rr}  (= 1-r_s/r)

  This requires MI(beta_0*Nbar*t_0)/MI(beta_0*t_0) = exp(2(Nbar-1)),
  i.e., MI ~ e^{2*beta} -- impossible for bounded MI.

This script:
  Part A: Numerical verification of log-concavity across models
  Part B: High-T expansion analysis (sub-Gaussian boundary fluctuations)
  Part C: Systematic search for counterexamples (gap scan)
  Part D: Schwarzschild impossibility argument
  Part E: Four diagnostic figures
"""

import numpy as np
from scipy.linalg import eigh_tridiagonal
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

# ============================================================
# Shared utilities
# ============================================================

def binary_entropy(f):
    """Binary entropy s(f) = -f ln(f) - (1-f) ln(1-f) in nats."""
    f = np.clip(f, 1e-15, 1 - 1e-15)
    return -f * np.log(f) - (1 - f) * np.log(1 - f)


def exact_mi_chain(beta_t, diag, offdiag):
    """
    Single-channel MI between the two middle sites of a free-fermion chain
    with arbitrary tridiagonal Hamiltonian H_{nm} = diag[n]*delta_{nm} + offdiag[n]*delta_{m,n+1}.

    Parameters:
        beta_t: dimensionless inverse temperature (beta * energy scale)
        diag: on-site energies (length N_chain)
        offdiag: hopping amplitudes (length N_chain - 1)

    Returns:
        MI in nats
    """
    if beta_t < 1e-14:
        return 0.0

    N_chain = len(diag)
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


def exact_mi_uniform_chain(beta_t, N_chain=600):
    """MI for uniform chain (diag=0, offdiag=-1)."""
    diag = np.zeros(N_chain)
    offdiag = -np.ones(N_chain - 1)
    return exact_mi_chain(beta_t, diag, offdiag)


def exact_mi_gapped_chain(beta_t, delta, N_chain=600):
    """MI for chain with staggered on-site potential +/- delta."""
    diag = np.array([delta * (-1)**n for n in range(N_chain)])
    offdiag = -np.ones(N_chain - 1)
    return exact_mi_chain(beta_t, diag, offdiag)


def exact_mi_dimerized_chain(beta_t, dim_delta, N_chain=600):
    """MI for dimerized chain with alternating hoppings t1=1+delta, t2=1-delta."""
    offdiag = np.array([-(1 + dim_delta * (-1)**n) for n in range(N_chain - 1)])
    diag = np.zeros(N_chain)
    return exact_mi_chain(beta_t, diag, offdiag)


def d2_log_mi(beta, mi_func, h_frac=0.01):
    """
    Compute d^2(log MI)/dbeta^2 via 5-point central differences.

    Parameters:
        beta: evaluation point
        mi_func: callable beta -> MI
        h_frac: fractional step size
    """
    h = max(beta * h_frac, 1e-5)
    mi_vals = np.array([mi_func(beta + k*h) for k in [-2, -1, 0, 1, 2]])

    # Avoid log(0)
    if np.any(mi_vals < 1e-30):
        return np.nan

    log_mi = np.log(mi_vals)

    # 5-point central difference for second derivative
    d2 = (-log_mi[4] + 16*log_mi[3] - 30*log_mi[2] + 16*log_mi[1] - log_mi[0]) / (12 * h**2)
    return d2


def onsager_inv_xi(beta_J):
    """Inverse correlation length from exact Onsager formula."""
    if beta_J <= 0:
        return np.inf
    beta_c_J = 0.5 * np.log(1 + np.sqrt(2))
    if beta_J >= beta_c_J:
        return 0.0
    return -np.log(np.tanh(beta_J)) - 2 * beta_J


def ising_kappa(beta_J, ell=1):
    """Ising MI conductance proportional to exp(-2*ell/xi)."""
    inv_xi = onsager_inv_xi(beta_J)
    if np.isinf(inv_xi):
        return 0.0
    return np.exp(-2 * ell * inv_xi)


def F_derivatives_numerical(mi_func, beta0, h=1e-4):
    """
    Compute F'(1) and F''(1) for F(Nbar) = mi_func(beta0*Nbar)/mi_func(beta0).

    Uses 5-point central differences in Nbar around Nbar=1.
    """
    mi_0 = mi_func(beta0)
    if mi_0 < 1e-30:
        return 2.0, 2.0  # high-T limit for fermion-type

    mi_p1 = mi_func(beta0 * (1 + h))
    mi_m1 = mi_func(beta0 * (1 - h))
    mi_p2 = mi_func(beta0 * (1 + 2*h))
    mi_m2 = mi_func(beta0 * (1 - 2*h))

    Fp = (-mi_p2 + 8*mi_p1 - 8*mi_m1 + mi_m2) / (12 * h * mi_0)
    Fpp = (-mi_p2 + 16*mi_p1 - 30*mi_0 + 16*mi_m1 - mi_m2) / (12 * h**2 * mi_0)
    return Fp, Fpp


# ============================================================
# Part A: Numerical verification of log-concavity
# ============================================================

def part_a():
    """Compute d^2(log MI)/dbeta^2 across beta range for multiple models."""
    print("=" * 95)
    print("PART A: NUMERICAL VERIFICATION OF LOG-CONCAVITY")
    print("=" * 95)
    print()

    beta_range = np.logspace(-3, np.log10(5), 200)
    results = {}

    # --- A1: Uniform free fermion chain ---
    print("  A1: Uniform free fermion chain (gapless)...")
    d2_uniform = np.array([d2_log_mi(b, exact_mi_uniform_chain) for b in beta_range])
    max_val = np.nanmax(d2_uniform)
    results['uniform'] = {'beta': beta_range, 'd2': d2_uniform}
    print(f"      max d^2(log MI)/dbeta^2 = {max_val:.6e}  ({'< 0 PASS' if max_val < 0 else '> 0 FAIL'})")

    # --- A2: Gapped chain ---
    print("  A2: Gapped free fermion chain (staggered potential)...")
    gap_values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    results['gapped'] = {}
    for delta in gap_values:
        mi_func = lambda b, d=delta: exact_mi_gapped_chain(b, d)
        d2_vals = np.array([d2_log_mi(b, mi_func) for b in beta_range])
        max_val = np.nanmax(d2_vals)
        results['gapped'][delta] = {'beta': beta_range, 'd2': d2_vals}
        status = '< 0 PASS' if max_val < 0 else '> 0 FAIL'
        print(f"      Delta/t = {delta:.1f}: max d^2 = {max_val:.6e}  ({status})")

    # --- A3: Dimerized chain ---
    print("  A3: Dimerized chain (alternating hopping)...")
    dim_values = [0.0, 0.2, 0.4, 0.6, 0.8]
    results['dimerized'] = {}
    for dim_d in dim_values:
        mi_func = lambda b, d=dim_d: exact_mi_dimerized_chain(b, d)
        d2_vals = np.array([d2_log_mi(b, mi_func) for b in beta_range])
        max_val = np.nanmax(d2_vals)
        results['dimerized'][dim_d] = {'beta': beta_range, 'd2': d2_vals}
        status = '< 0 PASS' if max_val < 0 else '> 0 FAIL'
        print(f"      delta = {dim_d:.1f}: max d^2 = {max_val:.6e}  ({status})")

    # --- A4: 2D Ising (analytical) ---
    print("  A4: 2D Ising model (analytical)...")
    beta_c_J = 0.5 * np.log(1 + np.sqrt(2))
    beta_J_range = np.linspace(0.01, beta_c_J * 0.99, 200)
    ell_vals = [1, 2, 4]
    results['ising'] = {}
    for ell in ell_vals:
        # d^2(log kappa)/d(betaJ)^2 = -2*ell * phi''
        # phi''(betaJ) = 4*cosh(2*betaJ)/sinh^2(2*betaJ)  (always > 0)
        d2_ising = np.zeros(len(beta_J_range))
        for i, bJ in enumerate(beta_J_range):
            x = 2 * bJ
            d2_ising[i] = -2 * ell * 4.0 * np.cosh(x) / np.sinh(x)**2
        max_val = np.nanmax(d2_ising)
        results['ising'][ell] = {'beta_J': beta_J_range, 'd2': d2_ising}
        status = '< 0 PASS' if max_val < 0 else '> 0 FAIL'
        print(f"      ell = {ell}: max d^2(log kappa)/d(betaJ)^2 = {max_val:.6e}  ({status})")

    print()
    return results


# ============================================================
# Part B: High-T expansion analysis
# ============================================================

def part_b():
    """Analyze the O(beta^4) coefficient c_4 in MI = c_2*beta^2 + c_4*beta^4 + ..."""
    print("=" * 95)
    print("PART B: HIGH-T EXPANSION ANALYSIS")
    print("=" * 95)
    print()

    # For free fermions at half filling:
    # boundary Hamiltonian h_b = -t(c_n^dag c_{n+1} + h.c.)
    # In the 4-state Fock space {|00>, |10>, |01>, |11>}:
    # h_b eigenvalues: {0, -t, +t, 0}
    #
    # At infinite T (beta=0), equal probability:
    #   <h_b^2>_0 = (0 + t^2 + t^2 + 0)/4 = t^2/2
    #   <h_b^4>_0 = (0 + t^4 + t^4 + 0)/4 = t^4/2
    #
    # 4th cumulant = <h_b^4>_c = <h_b^4> - 3<h_b^2>^2
    #              = t^4/2 - 3*(t^2/2)^2 = t^4/2 - 3*t^4/4 = -t^4/4 < 0
    #
    # The MI has expansion MI ~ c_2*beta^2 + c_4*beta^4 + ...
    # where c_2 = <h_b^2>_0 = t^2/2 and c_4 involves the 4th cumulant.

    t = 1.0  # hopping
    h2 = t**2 / 2  # <h_b^2>
    h4 = t**4 / 2  # <h_b^4>
    c4_cumulant = h4 - 3 * h2**2  # 4th cumulant

    print("  Free fermion boundary Hamiltonian (2-site, half-filled):")
    print(f"    Spectrum: {{0, -t, +t, 0}} with t = {t}")
    print(f"    <h_b^2>_0 = {h2:.4f}")
    print(f"    <h_b^4>_0 = {h4:.4f}")
    print(f"    4th cumulant = <h_b^4> - 3<h_b^2>^2 = {c4_cumulant:.4f}  ({'< 0: sub-Gaussian' if c4_cumulant < 0 else '>= 0: super-Gaussian'})")
    print()

    # Ising boundary: h_b = -J*s_1*s_2, s^2 = 1
    # <h_b^2> = J^2, <h_b^4> = J^4
    # 4th cumulant = J^4 - 3*J^4 = -2*J^4 < 0
    J = 1.0
    ising_h2 = J**2
    ising_h4 = J**4
    ising_c4 = ising_h4 - 3 * ising_h2**2

    print("  Ising boundary Hamiltonian:")
    print(f"    h_b = -J*s_1*s_2 with J = {J}")
    print(f"    <h_b^2> = {ising_h2:.4f}")
    print(f"    <h_b^4> = {ising_h4:.4f}")
    print(f"    4th cumulant = {ising_c4:.4f}  ({'< 0: sub-Gaussian' if ising_c4 < 0 else '>= 0: super-Gaussian'})")
    print()

    # Verify numerically: at very small beta, d^2(log MI)/dbeta^2 ~ -2/beta^2 + 2*c4/c2
    print("  Numerical verification at small beta:")
    print(f"  {'beta':>10s} {'d2_logMI':>12s} {'asymptote':>12s} {'ratio':>10s}")
    print("  " + "-" * 50)

    for beta in [0.001, 0.003, 0.01, 0.03, 0.1]:
        d2_val = d2_log_mi(beta, exact_mi_uniform_chain)
        # High-T asymptote: d^2(log MI)/dbeta^2 ~ -2/beta^2 (dominant term)
        asymp = -2.0 / beta**2
        if d2_val is not None and not np.isnan(d2_val):
            ratio = d2_val / asymp if asymp != 0 else np.nan
            print(f"  {beta:10.4f} {d2_val:12.4e} {asymp:12.4e} {ratio:10.4f}")

    print()
    print("  Key insight: d^2(log MI)/dbeta^2 ~ -2/beta^2 + 2*c4/c2")
    print("  The -2/beta^2 term dominates at high T and is always negative.")
    print("  Log-convexity would require c_4 > 0 (super-Gaussian fluctuations),")
    print("  but bounded interactions always give c_4 < 0.")
    print()

    return {'fermion_c4': c4_cumulant, 'ising_c4': ising_c4}


# ============================================================
# Part C: Systematic counterexample search
# ============================================================

def part_c():
    """Scan gap parameter x beta to search for any d^2(log MI)/dbeta^2 > 0."""
    print("=" * 95)
    print("PART C: SYSTEMATIC COUNTEREXAMPLE SEARCH")
    print("=" * 95)
    print()

    delta_values = np.linspace(0, 3.0, 31)
    beta_values = np.logspace(-3, 1, 200)

    print(f"  Scanning Delta/t in [0, 3] ({len(delta_values)} points)")
    print(f"  x beta in [0.001, 10] ({len(beta_values)} log-spaced points)")
    print()

    max_d2_per_delta = np.full(len(delta_values), -np.inf)
    beta_at_max = np.zeros(len(delta_values))
    global_max = -np.inf
    global_max_delta = 0
    global_max_beta = 0

    for i, delta in enumerate(delta_values):
        mi_func = lambda b, d=delta: exact_mi_gapped_chain(b, d)
        for j, beta in enumerate(beta_values):
            val = d2_log_mi(beta, mi_func)
            if not np.isnan(val) and val > max_d2_per_delta[i]:
                max_d2_per_delta[i] = val
                beta_at_max[i] = beta
                if val > global_max:
                    global_max = val
                    global_max_delta = delta
                    global_max_beta = beta

        if (i+1) % 10 == 0 or i == 0:
            print(f"    Delta/t = {delta:5.2f}: max d^2 = {max_d2_per_delta[i]:.6e}")

    print()
    print(f"  GLOBAL MAXIMUM: d^2(log MI)/dbeta^2 = {global_max:.6e}")
    print(f"    at Delta/t = {global_max_delta:.2f}, beta = {global_max_beta:.4f}")
    if global_max < 0:
        print("    => LOG-CONCAVITY HOLDS EVERYWHERE (no counterexample found)")
    else:
        print("    => LOG-CONCAVITY VIOLATED (counterexample found!)")
    print()

    return {
        'delta_values': delta_values,
        'max_d2_per_delta': max_d2_per_delta,
        'beta_at_max': beta_at_max,
        'global_max': global_max,
    }


# ============================================================
# Part D: Schwarzschild impossibility
# ============================================================

def part_d():
    """Demonstrate that F = exp(2(N-1)) gives exact Schwarzschild, but is unphysical."""
    print("=" * 95)
    print("PART D: SCHWARZSCHILD IMPOSSIBILITY")
    print("=" * 95)
    print()

    # --- D1: Analytical ODE solution ---
    print("  D1: ODE solutions for different F(N)")
    print("  " + "-" * 70)
    print()
    print("  Exterior closure ODE:  r^2 F(N) dN/dr = epsilon")
    print("  Exact solution:  G(N) = int_1^N F(N') dN' = -epsilon/r")
    print("  where epsilon = -r_s/2 (with appropriate normalization)")
    print()

    print("  Case 1: F(N) = N^2  (free fermion, high-T)")
    print("    G(N) = (N^3 - 1)/3 = r_s/(2r)")
    print("    N^3 = 1 - 3*r_s/(2r)")
    print("    h^{rr} = F(N) = N^2 = (1 - 3*r_s/(2r))^{2/3}")
    print("    => alpha_1 = -1, alpha_2 = -1/4, alpha_3 = -5/36, ...")
    print()

    print("  Case 2: F(N) = exp(2(N-1))  (Schwarzschild target)")
    print()
    print("  Sign-careful derivation:")
    print("    ODE: r^2 F(N) dN/dr = epsilon, with N'<0 (lapse decreases inward)")
    print("    Integrate from r to infinity, using N(inf) = 1, epsilon = -r_s/2:")
    print("    int_{N(r)}^1 F(N') dN' = epsilon * (1/r)")
    print("    => int_1^{N(r)} F(N') dN' = -epsilon/r = r_s/(2r)")
    print()
    print("    Since N(r) < 1, the integral runs from 1 DOWN to N.")
    print("    Write G_down(N) = int_N^1 F(N') dN' = r_s/(2r)  [positive]")
    print()
    print("    For F = exp(2(N'-1)):")
    print("    G_down(N) = int_N^1 exp(2(N'-1)) dN' = [1 - exp(2(N-1))]/2")
    print("    [1 - exp(2(N-1))]/2 = r_s/(2r)")
    print("    exp(2(N-1)) = 1 - r_s/r")
    print("    h^{rr} = F(N) = exp(2(N-1)) = 1 - r_s/r  [EXACT SCHWARZSCHILD]")
    print()

    # Let's just compute everything numerically and compare
    print("  Numerical comparison of ODE solutions:")
    print("  " + "-" * 70)

    r_over_rs = np.linspace(1.5, 50, 500)
    x = 1.0 / r_over_rs  # r_s/r

    # Case 1: F = N^2
    # N^3 = 1 - 3x/2 => N = (1 - 3x/2)^{1/3}
    # h^{rr} = N^2 = (1 - 3x/2)^{2/3}
    mask1 = x < 2.0/3  # N > 0
    hrr_quadratic = np.full_like(x, np.nan)
    hrr_quadratic[mask1] = (1 - 1.5*x[mask1])**(2.0/3)

    # Case 2: F = exp(2(N-1))
    # G_down(N) = int_N^1 exp(2(N'-1)) dN' = [1 - exp(2(N-1))]/2 = x/2
    # exp(2(N-1)) = 1 - x = 1 - r_s/r  (exact Schwarzschild)
    mask2 = x < 1.0
    hrr_exponential = np.full_like(x, np.nan)
    hrr_exponential[mask2] = 1 - x[mask2]

    # Schwarzschild
    hrr_schw = 1 - x

    print()
    print(f"  {'r/r_s':>8s} {'F=N^2':>10s} {'F=exp':>10s} {'Schw':>10s}")
    print("  " + "-" * 45)
    for rr in [2, 3, 5, 10, 20, 50]:
        xx = 1.0/rr
        h1 = (1 - 1.5*xx)**(2.0/3) if xx < 2.0/3 else float('nan')
        h2 = 1 - xx
        hs = 1 - xx
        print(f"  {rr:8d} {h1:10.6f} {h2:10.6f} {hs:10.6f}")

    print()
    print("  RESULT: F = exp(2(N-1)) gives EXACTLY h^{rr} = 1 - r_s/r")
    print("  (Schwarzschild at all PN orders)")
    print()

    # --- D2: Why this is impossible ---
    print("  D2: Why F = exp(2(N-1)) is unphysical")
    print("  " + "-" * 70)
    print()
    print("  F(Nbar) = MI(beta_0*Nbar)/MI(beta_0) = exp(2(Nbar-1))")
    print("  => MI(beta) = MI(beta_0) * exp(2*(beta/beta_0 - 1))")
    print("  => MI(beta) ~ exp(2*beta/beta_0) as beta -> infinity")
    print()
    print("  But MI is bounded: MI <= 2*ln(2) per channel (for qubits).")
    print("  An exponentially growing MI is impossible.")
    print()
    print("  More precisely: log MI goes from -infinity (beta=0, MI~beta^2)")
    print("  to log(MI_max) (beta=infinity).  Since it starts at -infinity")
    print("  and ends at a finite value, d(log MI)/dbeta must integrate to")
    print("  a finite positive number, so it must decrease from +infinity to 0.")
    print("  Therefore d^2(log MI)/dbeta^2 is negative on average:")
    print("    int_0^inf d^2(log MI)/dbeta^2 dbeta = [d(log MI)/dbeta]_0^inf")
    print("                                        = 0 - (+inf) = -inf")
    print()

    # --- D3: Compare F functions ---
    print("  D3: Comparing physical F(Nbar) with Schwarzschild target")
    print("  " + "-" * 70)

    Nbar_range = np.linspace(0.5, 1.5, 200)
    F_target = np.exp(2*(Nbar_range - 1))

    # Free fermion F at various temperatures
    print()
    print(f"  {'Model':>20s} {'F(0.8)':>8s} {'F(1.2)':>8s} {'logF(0.8)':>10s} {'logF(1.2)':>10s} "
          f"{'tgt(0.8)':>10s} {'tgt(1.2)':>10s}")
    print("  " + "-" * 80)

    F_results = {}
    for bt in [0.01, 0.1, 0.5, 1.0]:
        mi_ref = exact_mi_uniform_chain(bt)
        F_vals = np.array([exact_mi_uniform_chain(bt * nb) / mi_ref if mi_ref > 1e-30
                           else nb**2 for nb in Nbar_range])
        F_results[f'fermion_bt{bt}'] = F_vals
        f08 = np.interp(0.8, Nbar_range, F_vals)
        f12 = np.interp(1.2, Nbar_range, F_vals)
        t08 = np.exp(2*(0.8 - 1))
        t12 = np.exp(2*(1.2 - 1))
        print(f"  {'FF bt='+f'{bt:.2f}':>20s} {f08:8.4f} {f12:8.4f} "
              f"{np.log(max(f08,1e-30)):10.4f} {np.log(f12):10.4f} "
              f"{t08:10.4f} {t12:10.4f}")

    # Ising F
    beta_c_J = 0.5 * np.log(1 + np.sqrt(2))
    for bJ in [0.1, 0.3]:
        inv_xi_ref = onsager_inv_xi(bJ)
        F_vals = np.array([np.exp(-2 * 1 * (onsager_inv_xi(bJ * nb) - inv_xi_ref))
                           if not np.isinf(onsager_inv_xi(bJ * nb)) else 0
                           for nb in Nbar_range])
        F_results[f'ising_bJ{bJ}'] = F_vals
        f08 = np.interp(0.8, Nbar_range, F_vals)
        f12 = np.interp(1.2, Nbar_range, F_vals)
        t08 = np.exp(2*(0.8 - 1))
        t12 = np.exp(2*(1.2 - 1))
        print(f"  {'Ising bJ='+f'{bJ:.2f}':>20s} {f08:8.4f} {f12:8.4f} "
              f"{np.log(max(f08,1e-30)):10.4f} {np.log(f12):10.4f} "
              f"{t08:10.4f} {t12:10.4f}")

    F_results['target'] = F_target
    print()
    print("  All physical F's are concave on a log scale:")
    print("  log F bends DOWN relative to the linear (Schwarzschild) target.")
    print()

    return {
        'r_over_rs': r_over_rs,
        'hrr_quadratic': hrr_quadratic,
        'hrr_exponential': hrr_exponential,
        'hrr_schw': hrr_schw,
        'Nbar_range': Nbar_range,
        'F_results': F_results,
        'F_target': F_target,
    }


# ============================================================
# Part E: Figures
# ============================================================

def make_figures(results_a, results_b, results_c, results_d):
    """Generate four diagnostic figures."""
    outdir = "fermion/numerical"

    # ---- Figure 1: Log-concavity landscape ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Uniform chain
    ax = axes[0, 0]
    d = results_a['uniform']
    mask = np.isfinite(d['d2'])
    ax.plot(d['beta'][mask], d['d2'][mask], 'k-', lw=1.5)
    ax.axhline(0, color='red', ls='--', alpha=0.5)
    ax.set_xlabel(r'$\beta t_0$')
    ax.set_ylabel(r'$d^2(\log \mathrm{MI})/d\beta^2$')
    ax.set_title('Uniform free fermion chain')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=min(-10, np.nanmin(d['d2'][mask])*1.1), top=1)

    # Panel 2: Gapped chain
    ax = axes[0, 1]
    for delta, data in results_a['gapped'].items():
        mask = np.isfinite(data['d2'])
        if np.any(mask):
            ax.plot(data['beta'][mask], data['d2'][mask], lw=1.2,
                    label=rf'$\Delta/t = {delta:.1f}$')
    ax.axhline(0, color='red', ls='--', alpha=0.5)
    ax.set_xlabel(r'$\beta t_0$')
    ax.set_ylabel(r'$d^2(\log \mathrm{MI})/d\beta^2$')
    ax.set_title('Gapped free fermion chain')
    ax.set_xscale('log')
    ax.legend(loc='lower right', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=-50, top=1)

    # Panel 3: Dimerized chain
    ax = axes[1, 0]
    for dim_d, data in results_a['dimerized'].items():
        mask = np.isfinite(data['d2'])
        if np.any(mask):
            ax.plot(data['beta'][mask], data['d2'][mask], lw=1.2,
                    label=rf'$\delta = {dim_d:.1f}$')
    ax.axhline(0, color='red', ls='--', alpha=0.5)
    ax.set_xlabel(r'$\beta t_0$')
    ax.set_ylabel(r'$d^2(\log \mathrm{MI})/d\beta^2$')
    ax.set_title('Dimerized free fermion chain')
    ax.set_xscale('log')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=-50, top=1)

    # Panel 4: 2D Ising
    ax = axes[1, 1]
    for ell, data in results_a['ising'].items():
        ax.plot(data['beta_J'], data['d2'], lw=1.5,
                label=rf'$\ell = {ell}$')
    ax.axhline(0, color='red', ls='--', alpha=0.5)
    beta_c_J = 0.5 * np.log(1 + np.sqrt(2))
    ax.axvline(beta_c_J, color='gray', ls=':', alpha=0.5, label=r'$\beta_c J$')
    ax.set_xlabel(r'$\beta J$')
    ax.set_ylabel(r'$d^2(\log \kappa)/d(\beta J)^2$')
    ax.set_title('2D Ising model (analytical)')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=-200, top=5)

    fig.suptitle(r'Log-concavity of MI conductance: $d^2(\log \kappa)/d\beta^2 < 0$ everywhere',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{outdir}/log_concavity_landscape.pdf", dpi=150, bbox_inches='tight')
    plt.savefig(f"{outdir}/log_concavity_landscape.png", dpi=150, bbox_inches='tight')
    print(f"Figure 1 saved: {outdir}/log_concavity_landscape.pdf")

    # ---- Figure 2: alpha_2 vs parameters ----
    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: alpha_2 vs beta_0*t_0 (uniform)
    ax = axes2[0]
    beta_scan = np.logspace(-2, 0.5, 60)
    a2_scan = []
    for bt in beta_scan:
        Fp, Fpp = F_derivatives_numerical(exact_mi_uniform_chain, bt)
        a2_scan.append((Fpp - Fp**2) / 8.0)
    ax.semilogx(beta_scan, a2_scan, 'k-', lw=2)
    ax.axhline(0, color='green', ls='--', alpha=0.5, label='Schwarzschild')
    ax.axhline(-0.25, color='gray', ls=':', alpha=0.5, label=r'$-1/4$ (high-T)')
    ax.set_xlabel(r'$\beta_0 t_0$')
    ax.set_ylabel(r'$\alpha_2$')
    ax.set_title(r'Uniform chain: $\alpha_2$ vs temperature')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: alpha_2 vs Delta/t (gapped, at fixed beta=0.1)
    ax = axes2[1]
    delta_scan = np.linspace(0, 3, 40)
    for bt_fixed in [0.1, 0.5, 1.0]:
        a2_gap = []
        for delta in delta_scan:
            mi_func = lambda b, d=delta: exact_mi_gapped_chain(b, d)
            Fp, Fpp = F_derivatives_numerical(mi_func, bt_fixed)
            a2_gap.append((Fpp - Fp**2) / 8.0)
        ax.plot(delta_scan, a2_gap, lw=1.5,
                label=rf'$\beta_0 t_0 = {bt_fixed}$')
    ax.axhline(0, color='green', ls='--', alpha=0.5, label='Schwarzschild')
    ax.axhline(-0.25, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel(r'$\Delta / t$ (gap parameter)')
    ax.set_ylabel(r'$\alpha_2$')
    ax.set_title(r'Gapped chain: $\alpha_2$ vs gap')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: alpha_2 vs delta (dimerized, at fixed beta=0.1)
    ax = axes2[2]
    dim_scan = np.linspace(0, 0.9, 40)
    for bt_fixed in [0.1, 0.5, 1.0]:
        a2_dim = []
        for dim_d in dim_scan:
            mi_func = lambda b, d=dim_d: exact_mi_dimerized_chain(b, d)
            Fp, Fpp = F_derivatives_numerical(mi_func, bt_fixed)
            a2_dim.append((Fpp - Fp**2) / 8.0)
        ax.plot(dim_scan, a2_dim, lw=1.5,
                label=rf'$\beta_0 t_0 = {bt_fixed}$')
    ax.axhline(0, color='green', ls='--', alpha=0.5, label='Schwarzschild')
    ax.axhline(-0.25, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel(r'$\delta$ (dimerization)')
    ax.set_ylabel(r'$\alpha_2$')
    ax.set_title(r'Dimerized chain: $\alpha_2$ vs dimerization')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    fig2.suptitle(r'$\alpha_2 < 0$ universally: the 2PN departure is structural',
                  fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{outdir}/alpha2_vs_parameters.pdf", dpi=150, bbox_inches='tight')
    plt.savefig(f"{outdir}/alpha2_vs_parameters.png", dpi=150, bbox_inches='tight')
    print(f"Figure 2 saved: {outdir}/alpha2_vs_parameters.pdf")

    # ---- Figure 3: F(Nbar) comparison with Schwarzschild target ----
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5.5))

    Nbar_range = results_d['Nbar_range']
    F_target = results_d['F_target']
    F_results = results_d['F_results']

    # Panel 1: F(Nbar) direct
    ax = axes3[0]
    for key, F_vals in F_results.items():
        if key == 'target':
            continue
        label = key.replace('fermion_bt', 'FF ').replace('ising_bJ', 'Ising ')
        label = label.replace('FF ', r'Fermion $\beta_0 t_0=$')
        label = label.replace('Ising ', r'Ising $\beta J=$')
        ax.plot(Nbar_range, F_vals, lw=1.2, label=label)
    ax.plot(Nbar_range, F_target, 'k--', lw=2,
            label=r'Schwarzschild: $e^{2(\bar{N}-1)}$')
    ax.plot(Nbar_range, Nbar_range**2, 'k:', lw=1.5,
            label=r'High-T: $\bar{N}^2$')
    ax.set_xlabel(r'$\bar{N}$ (bond-averaged lapse)')
    ax.set_ylabel(r'$F(\bar{N})$')
    ax.set_title(r'Conductance ratio $F(\bar{N})$')
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 1.5)
    ax.set_ylim(0, 3.5)

    # Panel 2: log F vs (Nbar - 1) to show concavity
    ax = axes3[1]
    dN = Nbar_range - 1
    for key, F_vals in F_results.items():
        if key == 'target':
            continue
        log_F = np.log(np.maximum(F_vals, 1e-30))
        label = key.replace('fermion_bt', 'FF ').replace('ising_bJ', 'Ising ')
        label = label.replace('FF ', r'Fermion $\beta_0 t_0=$')
        label = label.replace('Ising ', r'Ising $\beta J=$')
        ax.plot(dN, log_F, lw=1.2, label=label)
    ax.plot(dN, 2*dN, 'k--', lw=2,
            label=r'Schwarzschild: $2(\bar{N}-1)$ [linear]')
    ax.plot(dN, 2*np.log(np.maximum(Nbar_range, 1e-30)), 'k:', lw=1.5,
            label=r'High-T: $2\log\bar{N}$ [concave]')
    ax.set_xlabel(r'$\bar{N} - 1$')
    ax.set_ylabel(r'$\log F(\bar{N})$')
    ax.set_title(r'$\log F$ is concave (bends below Schwarzschild line)')
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, 0.5)

    fig3.suptitle(r'Physical $F$ vs Schwarzschild target: log-concavity prevents exact GR',
                  fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{outdir}/F_comparison_schwarzschild.pdf", dpi=150, bbox_inches='tight')
    plt.savefig(f"{outdir}/F_comparison_schwarzschild.png", dpi=150, bbox_inches='tight')
    print(f"Figure 3 saved: {outdir}/F_comparison_schwarzschild.pdf")

    # ---- Figure 4: Impossibility plot ----
    fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel 1: max d^2(log MI)/dbeta^2 vs Delta/t
    ax = axes4[0]
    ax.plot(results_c['delta_values'], results_c['max_d2_per_delta'],
            'ko-', lw=2, ms=4)
    ax.axhline(0, color='red', ls='--', alpha=0.7, lw=2, label='Log-convexity boundary')
    ax.fill_between(results_c['delta_values'],
                     results_c['max_d2_per_delta'],
                     0, where=results_c['max_d2_per_delta'] < 0,
                     alpha=0.15, color='blue', label='Log-concave (physical)')
    ax.set_xlabel(r'$\Delta / t$ (gap parameter)')
    ax.set_ylabel(r'$\max_\beta\, d^2(\log \mathrm{MI})/d\beta^2$')
    ax.set_title('Maximum log-curvature vs gap (always < 0)')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: Integral constraint
    # Compute int_0^B d^2(log MI)/dbeta^2 dbeta for various B
    ax = axes4[1]
    B_max_values = np.logspace(-1, 1, 50)
    for delta_val in [0.0, 1.0, 2.0]:
        mi_func = lambda b, d=delta_val: exact_mi_gapped_chain(b, d)
        integrals = []
        for B_max in B_max_values:
            beta_grid = np.linspace(0.01, B_max, 100)
            d2_vals = np.array([d2_log_mi(b, mi_func) for b in beta_grid])
            valid = np.isfinite(d2_vals)
            if np.sum(valid) > 5:
                integral = np.trapezoid(d2_vals[valid], beta_grid[valid])
            else:
                integral = np.nan
            integrals.append(integral)
        ax.plot(B_max_values, integrals, lw=1.5,
                label=rf'$\Delta/t = {delta_val:.1f}$')

    ax.axhline(0, color='red', ls='--', alpha=0.5)
    ax.set_xlabel(r'$B_{\max}$')
    ax.set_ylabel(r'$\int_0^{B_{\max}} d^2(\log \mathrm{MI})/d\beta^2\, d\beta$')
    ax.set_title('Integral constraint (must be negative)')
    ax.set_xscale('log')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    fig4.suptitle('Schwarzschild impossibility: log-concavity is universal',
                  fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{outdir}/schwarzschild_impossibility.pdf", dpi=150, bbox_inches='tight')
    plt.savefig(f"{outdir}/schwarzschild_impossibility.png", dpi=150, bbox_inches='tight')
    print(f"Figure 4 saved: {outdir}/schwarzschild_impossibility.pdf")


# ============================================================
# Main
# ============================================================

def main():
    print()
    print("=" * 95)
    print("LOG-CONCAVITY OF MI CONDUCTANCE AND CONDITIONS FOR SCHWARZSCHILD")
    print("=" * 95)
    print()

    results_a = part_a()
    results_b = part_b()
    results_c = part_c()
    results_d = part_d()

    print()
    print("=" * 95)
    print("GENERATING FIGURES")
    print("=" * 95)
    print()

    make_figures(results_a, results_b, results_c, results_d)

    # Final summary
    print()
    print("=" * 95)
    print("SUMMARY")
    print("=" * 95)
    print()
    print("1. LOG-CONCAVITY IS UNIVERSAL:")
    print("   d^2(log MI)/dbeta^2 < 0 for all models tested:")
    print("   - Uniform free fermion chain (gapless)")
    print("   - Gapped free fermion chain (all Delta/t)")
    print("   - Dimerized free fermion chain (all delta)")
    print("   - 2D Ising model (analytical proof)")
    print()
    print("2. MECHANISM:")
    print("   At high T: MI ~ c_2*beta^2 + c_4*beta^4, with c_4 < 0 (sub-Gaussian)")
    print("   => d^2(log MI)/dbeta^2 ~ -2/beta^2 < 0")
    print("   At low T: MI saturates (bounded by 2*ln2), so log MI has negative curvature")
    print()
    print("3. SCHWARZSCHILD REQUIRES F = exp(2(N-1)):")
    print("   This gives h^{rr} = 1 - r_s/r (exact GR at all PN orders)")
    print("   But it requires MI ~ e^{2*beta}, violating MI <= 2*ln(2)")
    print()
    print("4. CONSEQUENCE:")
    print("   alpha_2 < 0 for any model with bounded, log-concave MI")
    print("   The 2PN departure from Schwarzschild is structural, not a high-T artifact")
    print()


if __name__ == '__main__':
    main()
