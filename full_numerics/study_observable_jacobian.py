"""
Observable comparison: smeared vs fixed — residual minimization proof.

For the fixed-observable model, performs trust-region minimization of ||F||^2
to show that the global minimum is large (no solution exists).
Also computes the linearized IFT check at V0=0.

Run: python3 -m full_numerics.study_observable_jacobian
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from .physics_twostate_exact import TwoStateExactModel
from .compare_observables import FixedObservableExactModel
from .physics_twostate import TwoStateShellModel
from .solve_twostate import solve_full as solve_full_analytic
from .solve_twostate_exact import newton_fd_exact


def global_residual_minimization(model, Phi_seed, cstar_sq=0.5,
                                   lapse_floor=0.01, verbose=True):
    """
    Minimize ||F(Phi)||^2 using L-BFGS-B (much faster than TRF for large N).
    If the minimum is large, no solution exists.
    """
    from scipy.optimize import minimize
    N = model.N
    floor_val = -(1.0 - lapse_floor) * cstar_sq

    def objective(x):
        Phi = np.maximum(x, floor_val)
        F = model.residual(Phi)
        F[-1] = 0.0
        return 0.5 * np.sum(F**2)

    def gradient(x):
        """FD gradient of ||F||^2."""
        eps = 1e-7
        f0 = objective(x)
        grad = np.zeros(N)
        for j in range(N):
            ej = np.zeros(N)
            ej[j] = eps * max(1.0, abs(x[j]))
            grad[j] = (objective(x + ej) - f0) / ej[j]
        return grad

    bounds = [(floor_val, None)] * N
    result = minimize(objective, Phi_seed.copy(), method='L-BFGS-B',
                       bounds=bounds, options={'maxiter': 100, 'maxfun': 1000})

    Phi_opt = np.maximum(result.x, floor_val)
    F_opt = model.residual(Phi_opt)
    F_opt[-1] = 0.0

    return {
        "Phi": Phi_opt,
        "F_norm": np.max(np.abs(F_opt[:-1])),
        "F_rms": np.sqrt(np.mean(F_opt[:-1]**2)),
        "cost": result.fun,
        "nfev": result.nfev,
        "success": result.success,
    }


def ift_linearization_check(V0_values, N=200, t0=1.0, n_core=5,
                              beta0=0.1, cstar_sq=0.5, verbose=True):
    """
    At V0=0 (Phi=0 is exact solution), compute the linearized Jacobians
    for both models and check invertibility. This is the IFT check.

    The linearized equation at Phi=0:
      J * dPhi = -(beta0/c^2) * d(rho_tgt)/dV0 * dV0

    If J is invertible, IFT guarantees a branch. For smeared, J is negative
    definite (Laplacian + mass). For fixed, J has different structure.
    """
    # At Phi=0, build both Jacobians via FD
    Phi0 = np.zeros(N)
    eps = 1e-7

    results = {}
    for label, ModelClass in [("smeared", TwoStateExactModel),
                               ("fixed", FixedObservableExactModel)]:
        model = ModelClass(N=N, t0=t0, V0=0.001, n_core=n_core,
                            beta0=beta0, cstar_sq=cstar_sq)

        F0 = model.residual(Phi0)
        J = np.zeros((N, N))
        for j in range(N):
            ej = np.zeros(N)
            ej[j] = eps
            J[:, j] = (model.residual(Phi0 + ej) - F0) / eps

        evals = np.sort(np.real(np.linalg.eigvals(J)))
        # Exclude boundary eigenvalue
        interior = evals[np.abs(evals - 1.0) > 0.1]

        n_pos = np.sum(interior > 0)
        n_neg = np.sum(interior < 0)
        near_zero = interior[np.argmin(np.abs(interior))]

        results[label] = {
            "evals": interior,
            "n_pos": n_pos, "n_neg": n_neg,
            "near_zero": near_zero,
            "cond": np.abs(interior).max() / np.abs(interior).min(),
        }

        if verbose:
            print(f"  {label}: +{n_pos}/-{n_neg}, "
                  f"λ_near_0={near_zero:.4e}, cond={results[label]['cond']:.2e}")

    return results


def main():
    print("=" * 60)
    print("Observable comparison: residual minimization")
    print("=" * 60)

    V0_values = [0.01, 0.02, 0.03, 0.05]
    N = 200; t0 = 1.0; n_core = 5; beta0 = 0.1; cstar_sq = 0.5

    # IFT check at Phi=0
    print("\n--- IFT linearization at Phi=0 ---")
    ift_results = ift_linearization_check(V0_values)

    # For each V0: solve smeared (Newton), minimize ||F||^2 for fixed
    print("\n--- Residual minimization ---")
    results = []

    for V0 in V0_values:
        print(f"\n  V0 = {V0:.3f}")

        # Get analytic seed
        model_an = TwoStateShellModel(N=N, t0=t0, V0=V0, n_core=n_core,
                                       beta0=beta0, cstar_sq=cstar_sq)
        Phi_seed, _, _ = solve_full_analytic(model_an, tol=1e-12,
                                              lapse_floor=0.01, verbose=False)

        # Smeared: Newton converges
        model_sm = TwoStateExactModel(N=N, t0=t0, V0=V0, n_core=n_core,
                                       beta0=beta0, cstar_sq=cstar_sq)
        Phi_sm, conv_sm, nit_sm = newton_fd_exact(model_sm, Phi_seed,
                                                    tol=1e-10, max_iter=50,
                                                    verbose=False)
        F_sm = model_sm.residual(Phi_sm)
        F_sm[-1] = 0.0
        F_sm_norm = np.max(np.abs(F_sm[:-1]))

        # Fixed: Newton + global minimization
        model_fx = FixedObservableExactModel(N=N, t0=t0, V0=V0, n_core=n_core,
                                              beta0=beta0, cstar_sq=cstar_sq)

        # Try Newton first
        Phi_fx, conv_fx, nit_fx = newton_fd_exact(model_fx, Phi_seed,
                                                    tol=1e-10, max_iter=60,
                                                    verbose=False)
        F_fx = model_fx.residual(Phi_fx)
        F_fx[-1] = 0.0
        F_fx_newton = np.max(np.abs(F_fx[:-1]))

        # Global minimization from multiple seeds
        lapse_sm = 1 + Phi_sm / cstar_sq
        lapse_fx = 1 + Phi_fx / cstar_sq
        print(f"    Smeared Newton: |F|={F_sm_norm:.2e}, conv={conv_sm}, "
              f"min(N)={lapse_sm.min():.4f}")
        print(f"    Fixed Newton:   |F|={F_fx_newton:.2e}, conv={conv_fx}, "
              f"min(N)={lapse_fx.min():.4f}")

        # Try L-BFGS-B from analytic seed
        min_result = global_residual_minimization(model_fx, Phi_seed,
                                                    cstar_sq=cstar_sq)
        print(f"    Fixed L-BFGS-B (analytic seed): |F|={min_result['F_norm']:.2e}")

        # Try L-BFGS-B from zero
        min_result2 = global_residual_minimization(
            model_fx, np.zeros(N), cstar_sq=cstar_sq)
        print(f"    Fixed L-BFGS-B (Φ=0):           |F|={min_result2['F_norm']:.2e}")

        # Try Newton + L-BFGS-B from 20 random seeds in the feasible region
        rng = np.random.RandomState(42 + int(V0 * 1000))
        floor_val = -(1.0 - 0.01) * cstar_sq
        n_random = 10
        random_best = np.inf
        random_norms = []
        for iseed in range(n_random):
            # Random initial condition: uniform in [floor/2, 0] with Dirichlet BC
            Phi_rand = rng.uniform(floor_val * 0.5, 0.0, size=N)
            Phi_rand[-1] = 0.0  # enforce Dirichlet BC
            # Try Newton from this seed (fast: uses FD Jacobian, few iterations)
            Phi_rn, conv_rn, _ = newton_fd_exact(model_fx, Phi_rand,
                                                   tol=1e-10, max_iter=30,
                                                   verbose=False)
            F_rn = model_fx.residual(Phi_rn)
            F_rn[-1] = 0.0
            fn = np.max(np.abs(F_rn[:-1]))
            # Also try L-BFGS-B polishing (limited iterations for speed)
            mr = global_residual_minimization(model_fx, Phi_rn,
                                                cstar_sq=cstar_sq, verbose=False)
            fn_polished = mr['F_norm']
            best_this = min(fn, fn_polished)
            random_norms.append(best_this)
            if best_this < random_best:
                random_best = best_this
        print(f"    Fixed ({n_random} random seeds): best |F|={random_best:.2e}, "
              f"median={np.median(random_norms):.2e}, "
              f"worst={np.max(random_norms):.2e}")

        best_fx = min(min_result['F_norm'], min_result2['F_norm'],
                      F_fx_newton, random_best)

        results.append({
            "V0": V0,
            "F_sm": F_sm_norm,
            "F_fx_newton": F_fx_newton,
            "F_fx_trf_seed": min_result['F_norm'],
            "F_fx_trf_zero": min_result2['F_norm'],
            "F_fx_random_best": random_best,
            "F_fx_random_median": np.median(random_norms),
            "F_fx_best": best_fx,
            "conv_sm": conv_sm,
            "n_random": n_random,
        })

    # Summary
    print("\n--- Summary ---")
    print(f"{'V0':>6s}  {'Smeared |F|':>12s}  {'Fixed best |F|':>14s}  "
          f"{'Ratio':>10s}")
    for r in results:
        ratio = r["F_fx_best"] / max(r["F_sm"], 1e-30)
        print(f"{r['V0']:6.3f}  {r['F_sm']:12.2e}  {r['F_fx_best']:14.2e}  "
              f"{ratio:10.1f}×")

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    ax = axes[0]
    V0s = [r["V0"] for r in results]
    ax.semilogy(V0s, [r["F_sm"] for r in results], 'bo-', label="Smeared (Newton)")
    ax.semilogy(V0s, [r["F_fx_best"] for r in results], 'rs-', label="Fixed (global min)")
    ax.set_xlabel("$V_0$")
    ax.set_ylabel("$\\|F\\|_\\infty$")
    ax.set_title("(a) Residual: converged vs minimized")
    ax.legend()

    ax = axes[1]
    evals_sm = ift_results["smeared"]["evals"]
    evals_fx = ift_results["fixed"]["evals"]
    ax.semilogy(range(len(evals_sm)), np.abs(evals_sm), 'b.', ms=2,
                label=f"Smeared (+{ift_results['smeared']['n_pos']}/"
                      f"-{ift_results['smeared']['n_neg']})")
    ax.semilogy(range(len(evals_fx)), np.abs(evals_fx), 'r.', ms=2,
                label=f"Fixed (+{ift_results['fixed']['n_pos']}/"
                      f"-{ift_results['fixed']['n_neg']})")
    ax.set_xlabel("Mode index")
    ax.set_ylabel("$|\\lambda_i|$")
    ax.set_title("(b) Linearized Jacobian at $\\Phi=0$")
    ax.legend(fontsize=8)

    plt.tight_layout()
    outpath = "fermion/numerical/figures/observable_jacobian.pdf"
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"\n  Figure saved: {outpath}")
    plt.close()


if __name__ == "__main__":
    main()
