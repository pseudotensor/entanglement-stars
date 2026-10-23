"""
Newton solver with FD Jacobian for the exact two-state closure equation.

Uses the analytic two-state solver to generate a proxy seed, then
refines with dense Newton (FD Jacobian) on the exact residual.
"""

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from .physics_twostate_exact import TwoStateExactModel
from .physics_twostate import TwoStateShellModel
from .solve_twostate import solve_full as solve_full_analytic
from .solver import extract_rs


def newton_fd_exact(model, Phi0, tol=1e-10, max_iter=50,
                    eps=1e-7, lapse_guard=True, verbose=True,
                    jacobian_func=None, residual_func=None):
    """
    Newton with FD Jacobian (dense) for the exact two-state equation.

    If residual_func is provided, it replaces model.residual for both the
    RHS and convergence check.  This allows solving the proxy equation
    (fixed source) instead of the full self-consistent equation.

    If jacobian_func is provided, it is used for the FD Jacobian columns
    (inexact Newton) while the residual_func is used for the RHS.

    Each Newton step requires N+1 residual evaluations (each O(N^2)),
    so each step costs O(N^3). For N=200 this is ~seconds per step.
    """
    N = model.N
    Phi = Phi0.copy()
    rfunc = residual_func if residual_func is not None else model.residual
    jfunc = jacobian_func if jacobian_func is not None else rfunc

    for it in range(max_iter):
        F = rfunc(Phi)
        res = np.max(np.abs(F))
        lapse = 1.0 + Phi / model.cstar_sq

        if verbose and (it % 5 == 0 or it < 3 or res < tol):
            print(f"  Newton {it:3d}: |F|={res:.3e}, min(N)={lapse.min():.6f}",
                  flush=True)

        if res < tol:
            if verbose:
                print(f"  Converged at iter {it}", flush=True)
            return Phi, True, it

        # FD Jacobian (dense) — uses jfunc (possibly fast approximate)
        F_jac = jfunc(Phi)
        J = np.zeros((N, N))
        for j in range(N):
            Phi_p = Phi.copy()
            step = eps * max(1.0, abs(Phi[j]))
            Phi_p[j] += step
            J[:, j] = (jfunc(Phi_p) - F_jac) / step

        try:
            dPhi = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            if verbose:
                print(f"  Singular Jacobian at iter {it}", flush=True)
            return Phi, False, it

        # Line search with lapse guard
        alpha = 1.0
        for _ in range(30):
            Phi_trial = Phi + alpha * dPhi
            lapse_trial = 1.0 + Phi_trial / model.cstar_sq

            if lapse_guard and lapse_trial.min() < -0.5:
                alpha *= 0.5
                if alpha < 1e-6:
                    break
                continue

            F_trial = rfunc(Phi_trial)
            if np.max(np.abs(F_trial)) < res * 1.1:
                break
            alpha *= 0.5
            if alpha < 1e-6:
                break

        Phi = Phi + alpha * dPhi

    if verbose:
        print(f"  Not converged: |F|={np.max(np.abs(rfunc(Phi))):.3e}",
              flush=True)
    return Phi, False, max_iter


def _compute_jacobian(model, Phi, eps=1e-7, residual_func=None):
    """FD Jacobian at Phi (defaults to residual_fixedpoint for cheap columns)."""
    rfunc = residual_func if residual_func is not None else model.residual_fixedpoint
    N = model.N
    F0 = rfunc(Phi)
    if not np.all(np.isfinite(F0)):
        F0 = np.nan_to_num(F0, nan=0.0, posinf=1e10, neginf=-1e10)
    J = np.zeros((N, N))
    for j in range(N):
        step = eps * max(1.0, abs(Phi[j]))
        Phi_p = Phi.copy()
        Phi_p[j] += step
        Fj = rfunc(Phi_p)
        if not np.all(np.isfinite(Fj)):
            Fj = np.nan_to_num(Fj, nan=0.0, posinf=1e10, neginf=-1e10)
        J[:, j] = (Fj - F0) / step
    return J


def solve_full_diis(model, Phi_proxy, Phi_seed=None,
                    tol=1e-10, max_iter=500, lapse_floor=0.005,
                    jac_refresh=40, verbose=True,
                    stagger_project=False):
    """
    Damped Newton with line search, periodic Jacobian refresh, and lapse
    clamping for the full self-consistent closure equation.

    The "diis" in the name is historical: there is no DIIS/Pulay mixing here,
    and never was.  Six modules import this symbol, so the name is kept rather
    than churned; what it does is exactly what this docstring says.

    Strategy:
      1. Compute full FD Jacobian at starting point
      2. Newton steps with non-monotone line search (allows occasional increases)
      3. Refresh Jacobian every `jac_refresh` steps
      4. When line search fails, take a small damped step and continue
      5. Clamp lapse >= lapse_floor to prevent unphysical states
    """
    N = model.N
    cstar_sq = model.cstar_sq
    Phi_lo = -cstar_sq * (1.0 - lapse_floor)

    Phi = (Phi_seed if Phi_seed is not None else Phi_proxy).copy()

    lu_piv = None
    best_res = np.inf
    best_Phi = Phi.copy()
    jac_age = jac_refresh + 1  # force initial computation
    # Non-monotone reference: max residual over last M steps
    res_history = []
    M_nonmono = 10

    for it in range(max_iter):
        F = model.residual(Phi)
        res = np.max(np.abs(F))

        if res < best_res:
            best_res = res
            best_Phi = Phi.copy()

        res_history.append(res)
        if len(res_history) > M_nonmono:
            res_history.pop(0)
        res_ref = max(res_history)  # non-monotone reference

        if verbose and (it % 20 == 0 or it < 3 or res < tol):
            lapse = 1.0 + Phi / cstar_sq
            print(f"  Newton {it:4d}: |F|={res:.3e}, "
                  f"min(N)={lapse.min():.6f}", flush=True)

        if res < tol:
            if verbose:
                print(f"  Converged at iter {it}", flush=True)
            return Phi, True, it

        # Compute/refresh Jacobian periodically
        if jac_age >= jac_refresh:
            J = _compute_jacobian(model, Phi)
            try:
                lu_piv = lu_factor(J)
            except Exception:
                if verbose:
                    print(f"  Jacobian singular at iter {it}", flush=True)
                return best_Phi, False, it
            jac_age = 0
            if verbose and it > 0:
                print(f"  Jacobian refreshed at iter {it}", flush=True)

        # Newton direction (protect against NaN)
        if not np.all(np.isfinite(F)):
            Phi = best_Phi.copy()
            jac_age = jac_refresh + 1
            continue
        dPhi = -lu_solve(lu_piv, F)
        if not np.all(np.isfinite(dPhi)):
            jac_age = jac_refresh + 1
            continue

        # Project out the (-1)^n staggering mode from Newton step
        if stagger_project:
            v = np.array([(-1.0)**n for n in range(N)])
            dPhi -= (np.dot(v, dPhi) / np.dot(v, v)) * v

        # Non-monotone backtracking line search with lapse clamp
        alpha = 1.0
        best_alpha = 0.0
        best_trial_res = res
        for _ in range(30):
            Phi_trial = np.maximum(Phi + alpha * dPhi, Phi_lo)
            F_trial = model.residual(Phi_trial)
            res_trial = np.max(np.abs(F_trial))

            # Track best alpha seen
            if res_trial < best_trial_res:
                best_trial_res = res_trial
                best_alpha = alpha

            # Non-monotone Armijo: accept if better than recent worst
            if res_trial < res_ref * (1.0 - 1e-4 * alpha):
                break
            alpha *= 0.5
            if alpha < 1e-8:
                break

        # Use best alpha found (even if Armijo wasn't satisfied)
        if best_alpha > 0:
            alpha = best_alpha
        else:
            alpha = 0.01  # fallback: tiny step

        Phi = np.maximum(Phi + alpha * dPhi, Phi_lo)
        jac_age += 1

    if verbose:
        print(f"  Not converged: best |F|={best_res:.3e}", flush=True)
    return best_Phi, False, max_iter



def solve_proxy_exact(N=200, t0=1.0, V0=0.03, n_core=5, beta0=0.1,
                      cstar_sq=0.5, tol=1e-10, Phi_seed=None,
                      n_V0_steps=10, verbose=True,
                      w_perp=0.0, n_mu=1, smooth_eps=0.0):
    """
    Solve the proxy equation with exact MI conductances via V0-continuation.

    Strategy:
      1. Start from V0_small, solve proxy (Newton with FD Jacobian)
      2. Ramp V0 uniformly to target, using each solution as seed

    The proxy equation has a fixed source (no energy response), so it is
    well-conditioned at any beta0.  The V0-continuation ensures smooth
    evolution from the trivial solution (V0=0 -> Phi=0).

    Returns: dict with Phi, model, convergence info
    """
    if Phi_seed is None:
        Phi_seed = np.zeros(N)

    # V0 ramp: uniform spacing from V0/n_V0_steps up to V0.
    V0_min = V0 / n_V0_steps
    V0_values = np.linspace(V0_min, V0, n_V0_steps)

    Phi = Phi_seed.copy()
    conv_final = False

    for iv, V0_step in enumerate(V0_values):
        model = TwoStateExactModel(
            N=N, t0=t0, V0=V0_step, n_core=n_core,
            beta0=beta0, cstar_sq=cstar_sq,
            w_perp=w_perp, n_mu=n_mu, smooth_eps=smooth_eps)

        Phi, conv, nit = newton_fd_exact(
            model, Phi, tol=tol, max_iter=30,
            lapse_guard=True, verbose=False,
            residual_func=model.residual_proxy)

        if verbose and (iv == len(V0_values) - 1 or not conv):
            lapse = 1.0 + Phi / model.cstar_sq
            res = np.max(np.abs(model.residual_proxy(Phi)))
            print(f"    V0={V0_step:.5f}: |F|={res:.3e}, min(N)={lapse.min():.6f}, "
                  f"conv={conv}, nit={nit}", flush=True)

        if not conv and iv < len(V0_values) - 1:
            # Not converged at intermediate step — try smaller increments
            if verbose:
                print(f"    WARNING: not converged at V0={V0_step:.5f}, "
                      f"trying finer steps", flush=True)
            # Insert 5 sub-steps
            V0_sub = np.linspace(V0_values[max(0, iv-1)], V0_step, 6)[1:]
            for V0_s in V0_sub:
                model_s = TwoStateExactModel(
                    N=N, t0=t0, V0=V0_s, n_core=n_core,
                    beta0=beta0, cstar_sq=cstar_sq,
                    w_perp=w_perp, n_mu=n_mu, smooth_eps=smooth_eps)
                Phi, conv, _ = newton_fd_exact(
                    model_s, Phi, tol=tol, max_iter=30,
                    lapse_guard=True, verbose=False,
                    residual_func=model_s.residual_proxy)
            # Continue with next V0 step

        conv_final = conv

    # Final model at target V0
    model_final = TwoStateExactModel(
        N=N, t0=t0, V0=V0, n_core=n_core,
        beta0=beta0, cstar_sq=cstar_sq,
        w_perp=w_perp, n_mu=n_mu, smooth_eps=smooth_eps)

    lapse = 1.0 + Phi / cstar_sq
    rs = extract_rs(Phi, model_final, vacuum=True)

    return {
        "Phi": Phi,
        "model": model_final,
        "conv": conv_final,
        "min_N": lapse.min(),
        "rs": rs,
    }


def solve_exact(N=200, t0=1.0, V0=0.01, n_core=5, beta0=0.1,
                cstar_sq=0.5, tol=1e-6, lapse_floor=0.01,
                Phi_seed=None, verbose=True,
                w_perp=0.0, n_mu=1, smooth_eps=0.0):
    """
    Solve the exact two-state closure equation.

    Strategy:
      1. Solve the analytic two-state equation -> proxy seed
      2. Newton with FD Jacobian on exact residual -> converged solution

    Returns: dict with Phi_exact, Phi_analytic, model_exact, etc.
    """
    # Step 1: Get analytic two-state solution as seed
    model_an = TwoStateShellModel(
        N=N, t0=t0, V0=V0, n_core=n_core,
        beta0=beta0, cstar_sq=cstar_sq, mode="analytic")

    if verbose:
        print(f"  Step 1: Solving analytic two-state (V0={V0})...")
    Phi_an, conv_an, Phi_proxy = solve_full_analytic(
        model_an, Phi_seed, tol=1e-12,
        lapse_floor=lapse_floor, verbose=False)

    if verbose:
        lapse_an = 1.0 + Phi_an / cstar_sq
        res_an = np.max(np.abs(model_an.residual(Phi_an)))
        print(f"    Analytic: min(N)={lapse_an.min():.6f}, |F|={res_an:.2e}, conv={conv_an}")

    # Step 2: Exact Newton from analytic seed
    model_ex = TwoStateExactModel(
        N=N, t0=t0, V0=V0, n_core=n_core,
        beta0=beta0, cstar_sq=cstar_sq,
        w_perp=w_perp, n_mu=n_mu, smooth_eps=smooth_eps)

    if verbose:
        res0 = np.max(np.abs(model_ex.residual(Phi_an)))
        print(f"  Step 2: Exact Newton (seed |F_exact|={res0:.3e})...")

    Phi_ex, conv_ex, nit = newton_fd_exact(
        model_ex, Phi_an, tol=tol, max_iter=50,
        lapse_guard=True, verbose=verbose,
        jacobian_func=model_ex.residual_fixedpoint_analytic)

    lapse_ex = 1.0 + Phi_ex / cstar_sq
    rs_ex = extract_rs(Phi_ex, model_ex, vacuum=True)
    rs_an = extract_rs(Phi_an, model_an, vacuum=True)

    return {
        "Phi_exact": Phi_ex,
        "Phi_analytic": Phi_an,
        "Phi_proxy": Phi_proxy,
        "model_exact": model_ex,
        "model_analytic": model_an,
        "conv_exact": conv_ex,
        "conv_analytic": conv_an,
        "nit_exact": nit,
        "min_N_exact": lapse_ex.min(),
        "min_N_analytic": (1.0 + Phi_an / cstar_sq).min(),
        "rs_exact": rs_ex,
        "rs_analytic": rs_an,
        "F_exact": np.max(np.abs(model_ex.residual(Phi_ex))),
        "F_analytic": np.max(np.abs(model_an.residual(Phi_an))),
    }
