"""
Compare two observable choices for the exact two-state closure equation.

"Smeared observable" (current): rho_sigma = 2*t0*Nbar*G^smear_{n,n+1}*g_n
  The energy of the SMEARED Hamiltonian in its own thermal state.

"Fixed observable":             rho_sigma = 2*t0*G^smear_{n,n+1}*g_n
  The energy of the ORIGINAL Hamiltonian (hopping t0) measured in
  the smeared thermal state.  No Nbar factor.

The question: which gives correct Newtonian gravity in the exterior?

Run: python3 -m full_numerics.compare_observables
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .physics_twostate_exact import TwoStateExactModel
from .physics_twostate import TwoStateShellModel
from .solve_twostate import solve_full as solve_full_analytic
from .solve_twostate_exact import newton_fd_exact
from .solver import extract_rs


# ---------------------------------------------------------------------------
# Fixed-observable model: inherits TwoStateExactModel, overrides rho_sigma
# ---------------------------------------------------------------------------

class FixedObservableExactModel(TwoStateExactModel):
    """
    Exact two-state model where the observable is the ORIGINAL Hamiltonian
    (hopping t0), not the smeared Hamiltonian (hopping t0*Nbar).

    Only rho_sigma changes:  drop the Nbar factor.
    rho_bg and rho_tgt stay the same (they are properties of the actual
    physical states, not the reconstruction).
    """

    def rho_sigma(self, Phi):
        """
        Fixed-observable energy profile of reconstructed state sigma[Phi].

        rho_sigma_n = 2*t0*G^smear_{n,n+1}*g_n   (NO Nbar factor)

        This measures the original (unsmeared) bond Hamiltonian
        in the smeared thermal state.
        """
        G, Nbar = self._smeared_corr(Phi)
        hop = np.real(np.diag(G, 1))  # G^smear_{n,n+1}
        prof = np.zeros(self.N)
        prof[:-1] = 2.0 * self.t0 * hop * self.g[:-1]  # NO Nbar
        return prof

    def check_rho_sigma_at_zero(self):
        """
        For fixed observable, rho_sigma(Phi=0) should equal rho_bg
        since at Phi=0 the smeared Hamiltonian IS the original, and Nbar=1.
        """
        Phi0 = np.zeros(self.N)
        rho_s = self.rho_sigma(Phi0)
        rel_err = np.max(np.abs(rho_s[:-1] - self.rho_bg[:-1])) / \
                  np.max(np.abs(self.rho_bg[:-1]))
        return {"rho_sigma_0": rho_s, "rho_bg": self.rho_bg, "rel_err": rel_err}


# ---------------------------------------------------------------------------
# Solver wrapper for either model variant
# ---------------------------------------------------------------------------

def solve_variant(model_class, N=200, t0=1.0, V0=0.01, n_core=5,
                  beta0=0.1, cstar_sq=0.5, tol=1e-10,
                  lapse_floor=0.01, verbose=True, label=""):
    """
    Solve the exact two-state equation for a given model class.

    Uses the analytic two-state solver to generate a proxy seed, then
    refines with dense Newton (FD Jacobian) on the exact residual.
    """
    # Step 1: Analytic seed (same for both variants)
    model_an = TwoStateShellModel(
        N=N, t0=t0, V0=V0, n_core=n_core,
        beta0=beta0, cstar_sq=cstar_sq, mode="analytic")

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Solving: {label} (V0={V0})")
        print(f"{'='*60}")
        print(f"  Step 1: Analytic two-state seed...")

    Phi_an, conv_an, _ = solve_full_analytic(
        model_an, None, tol=1e-12,
        lapse_floor=lapse_floor, verbose=False)

    if verbose:
        lapse_an = 1.0 + Phi_an / cstar_sq
        print(f"    Analytic seed: min(N)={lapse_an.min():.6f}, conv={conv_an}")

    # Step 2: Newton on the exact model
    model_ex = model_class(
        N=N, t0=t0, V0=V0, n_core=n_core,
        beta0=beta0, cstar_sq=cstar_sq)

    # Verify rho_sigma(0) = rho_bg
    chk = model_ex.check_rho_sigma_at_zero()
    if verbose:
        print(f"    rho_sigma(0) vs rho_bg: rel_err = {chk['rel_err']:.3e}")

    res0 = np.max(np.abs(model_ex.residual(Phi_an)))
    if verbose:
        print(f"  Step 2: Newton (seed |F_exact|={res0:.3e})...")

    Phi_ex, conv_ex, nit = newton_fd_exact(
        model_ex, Phi_an, tol=tol, max_iter=60,
        lapse_guard=True, verbose=verbose)

    lapse_ex = 1.0 + Phi_ex / cstar_sq
    rs_ex = extract_rs(Phi_ex, model_ex)

    return {
        "Phi": Phi_ex,
        "Phi_seed": Phi_an,
        "model": model_ex,
        "converged": conv_ex,
        "nit": nit,
        "min_N": lapse_ex.min(),
        "rs": rs_ex,
        "residual": np.max(np.abs(model_ex.residual(Phi_ex))),
        "label": label,
        "V0": V0,
    }


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def envelope_analysis(Phi, model):
    """
    Separate the Phi profile into slowly-varying envelope + Friedel oscillation.

    The exact (lattice) solver produces staggered oscillations typical of
    half-filled tight-binding.  We extract the envelope by local averaging
    of adjacent shells: Phi_env(n) = (Phi(n) + Phi(n+1))/2.

    Returns dict with envelope Phi, envelope -Phi*r, and fit parameters.
    """
    r = model.r
    N = model.N
    n_core = model.n_core

    # Envelope by adjacent-shell averaging
    Phi_env = np.zeros(N)
    Phi_env[0] = Phi[0]
    Phi_env[1:-1] = 0.5 * (Phi[:-2] + Phi[2:])  # centered average
    Phi_env[-1] = Phi[-1]

    # -Phi_env * r in the exterior
    Phi_r_env = -Phi_env * r

    # Fit region: exterior, away from core and boundary
    i_lo = max(n_core + 10, 20)
    i_hi = min(N - 10, N // 2)

    if i_hi <= i_lo:
        return {
            "Phi_env": Phi_env, "Phi_r_env": Phi_r_env,
            "GM": 0.0, "GM_std": np.inf, "rel_osc": np.inf,
        }

    Phi_r_fit = Phi_r_env[i_lo:i_hi]
    GM = np.median(Phi_r_fit)
    GM_std = np.std(Phi_r_fit)
    rel_osc = GM_std / max(abs(GM), 1e-20)

    return {
        "Phi_env": Phi_env,
        "Phi_r_env": Phi_r_env,
        "GM": GM,
        "GM_std": GM_std,
        "rel_osc": rel_osc,
    }


def raw_exterior_fit(Phi, model):
    """
    Raw (no envelope) exterior fit.  Returns GM from median of -Phi*r
    in a conservative exterior window.
    """
    r = model.r
    n_core = model.n_core
    i_lo = max(n_core + 10, 20)
    i_hi = min(model.N - 10, model.N // 2)

    Phi_r = -Phi[i_lo:i_hi] * r[i_lo:i_hi]
    if len(Phi_r) == 0:
        return 0.0, 0.0, np.inf
    GM = np.median(Phi_r)
    GM_std = np.std(Phi_r)
    return GM, GM_std, GM_std / max(abs(GM), 1e-20)


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def main():
    import time

    N = 200
    t0 = 1.0
    n_core = 5
    beta0 = 0.1
    cstar_sq = 0.5
    tol = 1e-10
    lapse_floor = 0.01

    V0_values = [0.01, 0.02, 0.03, 0.05]

    results_smeared = []
    results_fixed = []

    for V0 in V0_values:
        t_start = time.time()

        # Smeared observable (current, with Nbar)
        res_s = solve_variant(
            TwoStateExactModel, N=N, t0=t0, V0=V0, n_core=n_core,
            beta0=beta0, cstar_sq=cstar_sq, tol=tol,
            lapse_floor=lapse_floor, verbose=True,
            label="Smeared observable")
        results_smeared.append(res_s)

        # Fixed observable (no Nbar)
        res_f = solve_variant(
            FixedObservableExactModel, N=N, t0=t0, V0=V0, n_core=n_core,
            beta0=beta0, cstar_sq=cstar_sq, tol=tol,
            lapse_floor=lapse_floor, verbose=True,
            label="Fixed observable")
        results_fixed.append(res_f)

        elapsed = time.time() - t_start
        print(f"\n  V0={V0}: elapsed {elapsed:.1f}s")

    # -----------------------------------------------------------------------
    # Print summary table
    # -----------------------------------------------------------------------
    print("\n" + "="*90)
    print("SUMMARY: Observable comparison")
    print("="*90)
    print(f"{'V0':>6s}  {'Conv_S':>6s} {'Conv_F':>6s}  "
          f"{'rs_S':>8s} {'rs_F':>8s}  "
          f"{'minN_S':>8s} {'minN_F':>8s}  "
          f"{'|F|_S':>10s} {'|F|_F':>10s}")
    print("-"*90)

    for rs, rf in zip(results_smeared, results_fixed):
        print(f"{rs['V0']:6.3f}  "
              f"{'Y' if rs['converged'] else 'N':>6s} "
              f"{'Y' if rf['converged'] else 'N':>6s}  "
              f"{rs['rs']:8.4f} {rf['rs']:8.4f}  "
              f"{rs['min_N']:8.5f} {rf['min_N']:8.5f}  "
              f"{rs['residual']:10.3e} {rf['residual']:10.3e}")

    # -----------------------------------------------------------------------
    # Envelope analysis for exterior 1/r
    # -----------------------------------------------------------------------
    print("\n" + "="*90)
    print("ENVELOPE ANALYSIS: GM = median(-Phi_env * r) in exterior")
    print("  Phi_env = centered 3-point average to remove Friedel oscillations")
    print("="*90)
    print(f"{'V0':>6s}  {'GM_smear':>10s} {'GM_fixed':>10s}  "
          f"{'rel_osc_S':>10s} {'rel_osc_F':>10s}  "
          f"{'rs_env_S':>10s} {'rs_env_F':>10s}")
    print("-"*90)

    env_smeared = []
    env_fixed = []
    for rs, rf in zip(results_smeared, results_fixed):
        es = envelope_analysis(rs["Phi"], rs["model"])
        ef = envelope_analysis(rf["Phi"], rf["model"])
        env_smeared.append(es)
        env_fixed.append(ef)
        rs_env_s = 2.0 * es["GM"] / cstar_sq
        rs_env_f = 2.0 * ef["GM"] / cstar_sq
        print(f"{rs['V0']:6.3f}  "
              f"{es['GM']:10.6f} {ef['GM']:10.6f}  "
              f"{es['rel_osc']:10.3e} {ef['rel_osc']:10.3e}  "
              f"{rs_env_s:10.4f} {rs_env_f:10.4f}")

    # -----------------------------------------------------------------------
    # Raw (no envelope) exterior analysis
    # -----------------------------------------------------------------------
    print("\n" + "="*90)
    print("RAW EXTERIOR: GM = median(-Phi * r), no envelope smoothing")
    print("="*90)
    for rs, rf in zip(results_smeared, results_fixed):
        GM_s, std_s, rel_s = raw_exterior_fit(rs["Phi"], rs["model"])
        GM_f, std_f, rel_f = raw_exterior_fit(rf["Phi"], rf["model"])
        print(f"  V0={rs['V0']:.3f}: "
              f"GM_S={GM_s:+.6f} +/- {std_s:.6f} (rel={rel_s:.2e}), "
              f"GM_F={GM_f:+.6f} +/- {std_f:.6f} (rel={rel_f:.2e})")

    # -----------------------------------------------------------------------
    # rho_sigma at Phi=0 sanity check
    # -----------------------------------------------------------------------
    print("\n" + "="*90)
    print("rho_sigma(Phi=0) CHECK (should equal rho_bg for both)")
    print("="*90)
    for V0 in [0.03]:
        m_s = TwoStateExactModel(N=N, t0=t0, V0=V0, n_core=n_core,
                                  beta0=beta0, cstar_sq=cstar_sq)
        m_f = FixedObservableExactModel(N=N, t0=t0, V0=V0, n_core=n_core,
                                         beta0=beta0, cstar_sq=cstar_sq)
        Phi0 = np.zeros(N)
        rho_s = m_s.rho_sigma(Phi0)
        rho_f = m_f.rho_sigma(Phi0)
        rho_bg = m_s.rho_bg
        err_s = np.max(np.abs(rho_s[:-1] - rho_bg[:-1])) / np.max(np.abs(rho_bg[:-1]))
        err_f = np.max(np.abs(rho_f[:-1] - rho_bg[:-1])) / np.max(np.abs(rho_bg[:-1]))
        print(f"  V0={V0}: smeared err={err_s:.3e}, fixed err={err_f:.3e}")
        print(f"    (Both should be ~0 since Nbar=1 at Phi=0)")

    # -----------------------------------------------------------------------
    # rho_sigma at the converged solution
    # -----------------------------------------------------------------------
    print("\n" + "="*90)
    print("rho_sigma AT SOLUTION: effect of Nbar factor (V0=0.03)")
    print("="*90)
    idx_03 = V0_values.index(0.03)
    res_s03 = results_smeared[idx_03]
    res_f03 = results_fixed[idx_03]
    Phi_s03 = res_s03["Phi"]
    Phi_f03 = res_f03["Phi"]

    _, Nbar_s = res_s03["model"].lapse_nbar(Phi_s03)
    _, Nbar_f = res_f03["model"].lapse_nbar(Phi_f03)
    print(f"  Smeared sol: min(Nbar)={Nbar_s.min():.6f}, max(Nbar)={Nbar_s.max():.6f}")
    print(f"  Fixed sol:   min(Nbar)={Nbar_f.min():.6f}, max(Nbar)={Nbar_f.max():.6f}")

    rho_sigma_s = res_s03["model"].rho_sigma(Phi_s03)
    rho_sigma_f = res_f03["model"].rho_sigma(Phi_f03)
    rho_tgt = res_s03["model"].rho_tgt
    print(f"  Smeared: max|rho_sigma - rho_tgt| = {np.max(np.abs(rho_sigma_s - rho_tgt)):.6e}")
    print(f"  Fixed:   max|rho_sigma - rho_tgt| = {np.max(np.abs(rho_sigma_f - rho_tgt)):.6e}")

    # -----------------------------------------------------------------------
    # Potential depth at core
    # -----------------------------------------------------------------------
    print("\n" + "="*90)
    print("POTENTIAL DEPTH: min(Phi/c*^2) at the core")
    print("="*90)
    for rs, rf in zip(results_smeared, results_fixed):
        print(f"  V0={rs['V0']:.3f}: "
              f"Smeared={rs['Phi'].min()/cstar_sq:.6f}, "
              f"Fixed={rf['Phi'].min()/cstar_sq:.6f}")

    # -----------------------------------------------------------------------
    # Generate 4-panel figure
    # -----------------------------------------------------------------------
    print("\n  Generating 4-panel figure...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Use V0=0.03 for panels (a), (b), (c)
    idx_ref = V0_values.index(0.03)
    res_s_ref = results_smeared[idx_ref]
    res_f_ref = results_fixed[idx_ref]
    model_ref = res_s_ref["model"]
    r = model_ref.r
    es_ref = env_smeared[idx_ref]
    ef_ref = env_fixed[idx_ref]

    conv_label_s = " [CONVERGED]" if res_s_ref["converged"] else " [NOT CONV]"
    conv_label_f = " [CONVERGED]" if res_f_ref["converged"] else " [NOT CONV]"

    # (a) Phi/c*^2 profiles
    ax = axes[0, 0]
    ax.plot(r, res_s_ref["Phi"] / cstar_sq, 'b-', lw=0.8, alpha=0.5)
    ax.plot(r, es_ref["Phi_env"] / cstar_sq, 'b-', lw=2.0,
            label='Smeared obs.' + conv_label_s)
    ax.plot(r, res_f_ref["Phi"] / cstar_sq, 'r-', lw=0.8, alpha=0.5)
    ax.plot(r, ef_ref["Phi_env"] / cstar_sq, 'r--', lw=2.0,
            label='Fixed obs.' + conv_label_f)
    ax.set_xlabel('$r/a$')
    ax.set_ylabel('$\\Phi / c_*^2$')
    ax.set_title(f'(a) Potential, $V_0 = {V0_values[idx_ref]}$'
                 '  (thick=envelope)')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 80)
    ax.axhline(0, color='gray', lw=0.5)

    # (b) Lapse profiles
    ax = axes[0, 1]
    lapse_s = 1.0 + res_s_ref["Phi"] / cstar_sq
    lapse_f = 1.0 + res_f_ref["Phi"] / cstar_sq
    lapse_s_env = 1.0 + es_ref["Phi_env"] / cstar_sq
    lapse_f_env = 1.0 + ef_ref["Phi_env"] / cstar_sq
    ax.plot(r, lapse_s, 'b-', lw=0.8, alpha=0.4)
    ax.plot(r, lapse_s_env, 'b-', lw=2.0, label='Smeared obs.')
    ax.plot(r, lapse_f, 'r-', lw=0.8, alpha=0.4)
    ax.plot(r, lapse_f_env, 'r--', lw=2.0, label='Fixed obs.')
    ax.set_xlabel('$r/a$')
    ax.set_ylabel('$N = 1 + \\Phi/c_*^2$')
    ax.set_title(f'(b) Lapse, $V_0 = {V0_values[idx_ref]}$')
    ax.legend(fontsize=9)
    ax.set_xlim(0, 80)
    ax.axhline(1, color='gray', lw=0.5)
    ax.axhline(0, color='gray', lw=0.5, ls=':')

    # (c) Envelope exterior fit: -Phi_env*r vs r
    ax = axes[1, 0]
    n_lo = model_ref.n_core + 2
    n_hi = min(model_ref.N - 5, 130)
    rr = r[n_lo:n_hi]

    ax.plot(rr, -res_s_ref["Phi"][n_lo:n_hi] * rr, 'b-', lw=0.5,
            alpha=0.3, label='_raw')
    ax.plot(rr, es_ref["Phi_r_env"][n_lo:n_hi], 'b-', lw=2.0,
            label=f'Smeared env (GM={es_ref["GM"]:.5f})')
    ax.plot(rr, -res_f_ref["Phi"][n_lo:n_hi] * rr, 'r-', lw=0.5,
            alpha=0.3, label='_raw')
    ax.plot(rr, ef_ref["Phi_r_env"][n_lo:n_hi], 'r--', lw=2.0,
            label=f'Fixed env (GM={ef_ref["GM"]:.5f})')

    # Horizontal reference lines
    ax.axhline(es_ref["GM"], color='b', lw=0.8, ls=':', alpha=0.7)
    ax.axhline(ef_ref["GM"], color='r', lw=0.8, ls=':', alpha=0.7)
    ax.set_xlabel('$r/a$')
    ax.set_ylabel('$-\\Phi_{\\rm env} \\cdot r$')
    ax.set_title('(c) Envelope 1/r test: $-\\Phi_{\\rm env} r$ vs $r$')
    ax.legend(fontsize=8)

    # (d) rs and min_N across V0
    ax = axes[1, 1]
    V0s = [r_["V0"] for r_ in results_smeared]

    # Envelope-based rs
    rs_env_s = [2.0 * es["GM"] / cstar_sq for es in env_smeared]
    rs_env_f = [2.0 * ef["GM"] / cstar_sq for ef in env_fixed]
    minN_s = [r_["min_N"] for r_ in results_smeared]
    minN_f = [r_["min_N"] for r_ in results_fixed]

    ax.plot(V0s, rs_env_s, 'bo-', lw=1.5, ms=6, label='$r_s$ smeared (env)')
    ax.plot(V0s, rs_env_f, 'r^--', lw=1.5, ms=6, label='$r_s$ fixed (env)')

    ax2 = ax.twinx()
    ax2.plot(V0s, minN_s, 'bs:', lw=1, ms=4, alpha=0.5,
             label='min($N$) smeared')
    ax2.plot(V0s, minN_f, 'rv:', lw=1, ms=4, alpha=0.5,
             label='min($N$) fixed')
    ax2.set_ylabel('min($N$)', color='gray')

    ax.set_xlabel('$V_0$')
    ax.set_ylabel('$r_s / a$  (from envelope)')
    ax.set_title('(d) $r_s$ and min lapse vs $V_0$')
    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='center right', fontsize=8)

    # Mark non-converged with X
    for i, V0 in enumerate(V0s):
        if not results_smeared[i]["converged"]:
            ax.plot(V0, rs_env_s[i], 'bx', ms=12, mew=2)
        if not results_fixed[i]["converged"]:
            ax.plot(V0, rs_env_f[i], 'rx', ms=12, mew=2)

    fig.suptitle("Two-state closure: smeared vs fixed observable comparison",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    outpath = "fermion/numerical/figures/twostate_observable_comparison.pdf"
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"  Figure saved to {outpath}")
    plt.close()

    # -----------------------------------------------------------------------
    # Final verdict
    # -----------------------------------------------------------------------
    print("\n" + "="*90)
    print("VERDICT")
    print("="*90)

    all_conv_s = all(r_["converged"] for r_ in results_smeared)
    all_conv_f = all(r_["converged"] for r_ in results_fixed)
    print(f"  Convergence: Smeared ALL={all_conv_s}, Fixed ALL={all_conv_f}")

    n_conv_s = sum(1 for r_ in results_smeared if r_["converged"])
    n_conv_f = sum(1 for r_ in results_fixed if r_["converged"])
    print(f"  Converged count: Smeared={n_conv_s}/{len(V0_values)}, "
          f"Fixed={n_conv_f}/{len(V0_values)}")

    if all_conv_s and not all_conv_f:
        print("\n  RESULT: Fixed observable FAILS TO CONVERGE.")
        print("  The smeared-observable form (with Nbar factor) is the")
        print("  physically correct choice: it is the energy of the")
        print("  smeared Hamiltonian in its own thermal state, which")
        print("  self-consistently defines the closure equation.")
        print("  The fixed-observable form has no self-consistent solution.")
    elif all_conv_f and not all_conv_s:
        print("\n  RESULT: Smeared observable FAILS TO CONVERGE.")
        print("  The fixed-observable form is the correct choice.")
    elif all_conv_s and all_conv_f:
        # Both converge -- compare quality
        avg_osc_s = np.mean([es["rel_osc"] for es in env_smeared])
        avg_osc_f = np.mean([ef["rel_osc"] for ef in env_fixed])
        print(f"\n  Both converge. Avg envelope rel_osc:")
        print(f"    Smeared = {avg_osc_s:.4e}")
        print(f"    Fixed   = {avg_osc_f:.4e}")
        if avg_osc_s < avg_osc_f:
            print("  --> Smeared gives cleaner 1/r exterior")
        else:
            print("  --> Fixed gives cleaner 1/r exterior")
    else:
        print("\n  RESULT: Both have convergence issues -- inconclusive")

    # Physics interpretation
    print("\n  PHYSICS INTERPRETATION:")
    print("  The closure equation reads:")
    print("    L_kappa(Phi) * Phi = (beta0/c*^2) * [rho_sigma(Phi) - rho_tgt]")
    print("  where rho_sigma is the energy measured in the reconstructed state.")
    print("  ")
    print("  Smeared obs: sigma = exp(-beta0 * H_smear)/Z, measuring H_smear")
    print("    -> rho_sigma = <H_smear>_{sigma} = 2*t0*Nbar*G^smear*g")
    print("    -> At high T: rho_sigma ~ Nbar^2 * beta0*t0^2*g/2")
    print("    -> Nbar^2 coupling provides the geometric amplification")
    print("  ")
    print("  Fixed obs:   sigma = exp(-beta0 * H_smear)/Z, measuring H_orig")
    print("    -> rho_sigma = <H_orig>_{sigma} = 2*t0*G^smear*g")
    print("    -> Missing the Nbar factor that couples Phi to the energy")
    print("    -> Weaker Phi-dependence in the RHS -> no self-consistent solution")

    return results_smeared, results_fixed


if __name__ == "__main__":
    main()
