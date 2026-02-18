"""
Robust solver for the two-state closure equation.

Strategy:
  1. Proxy Picard+Newton (stable, fixed RHS) -> converged proxy solution
  2. Full Newton from proxy seed (analytic tridiag Jacobian, lapse floor,
     clamped-shell handling) -> converged full solution

Direct Picard on the full equation is UNSTABLE because the energy response
rho_sigma ~ Nbar^2 creates positive feedback. The proxy Picard has fixed
RHS and is stable.

The lapse floor is a regularization. The floor-independence study
(solve with decreasing floors) tests the self-regularization
Proposition of Section 10.
"""

import numpy as np
from scipy.linalg import solve_banded
from .physics_twostate import TwoStateShellModel
from .solver import extract_rs


def _solve_tridiag(sub, diag, sup, rhs):
    """Solve tridiagonal system."""
    N = len(diag)
    ab = np.zeros((3, N))
    ab[0, 1:] = sup
    ab[1, :] = diag
    ab[2, :-1] = sub
    return solve_banded((1, 1), ab, rhs)


def picard_proxy(model, Phi0, tol=1e-8, max_iter=2000,
                  mixing=0.3, lapse_floor=0.01, verbose=False):
    """
    Picard iteration for the PROXY equation (fixed RHS, stable).
    """
    N = model.N
    Phi = Phi0.copy()
    pref = model.beta0 / model.cstar_sq
    floor_val = -(1.0 - lapse_floor) * model.cstar_sq if lapse_floor else None

    # Fixed RHS for proxy: (beta0/c*^2) * (rho_bg - rho_tgt)
    rhs_source = pref * (model.rho_bg - model.rho_tgt)

    for it in range(max_iter):
        lapse, Nbar = model.lapse_nbar(Phi)
        kappa = model.conductances(Nbar)

        # Build tridiagonal Laplacian
        sub_L = np.zeros(N - 1)
        diag_L = np.zeros(N)
        sup_L = np.zeros(N - 1)
        for n in range(N - 1):
            diag_L[n] += kappa[n]
            sup_L[n] -= kappa[n]
            diag_L[n + 1] += kappa[n]
            sub_L[n] -= kappa[n]

        rhs = rhs_source.copy()

        # Boundary condition
        diag_L[N - 1] = 1.0
        sub_L[N - 2] = 0.0
        rhs[N - 1] = 0.0

        Phi_new = _solve_tridiag(sub_L, diag_L, sup_L, rhs)

        if floor_val is not None:
            Phi_new = np.maximum(Phi_new, floor_val)

        Phi_mixed = mixing * Phi_new + (1.0 - mixing) * Phi

        rel = np.max(np.abs(Phi_mixed - Phi)) / max(np.max(np.abs(Phi)), 1e-15)
        Phi = Phi_mixed

        if verbose and (it % 200 == 0 or it < 3):
            lapse_now = 1.0 + Phi / model.cstar_sq
            print(f"    Picard {it:5d}: rel={rel:.3e}, min(N)={lapse_now.min():.6f}")

        if rel < tol:
            if verbose:
                print(f"    Picard converged at iter {it}")
            return Phi, True

    if verbose:
        print(f"    Picard not converged after {max_iter} iters: rel={rel:.3e}")
    return Phi, False


def newton_twostate(model, Phi0, proxy=False, tol=1e-12, max_iter=200,
                     lapse_floor=None, verbose=False):
    """
    Newton with analytic tridiag Jacobian for the two-state equation.

    When lapse_floor is set, shells at the floor are treated as
    Dirichlet conditions (fixed Phi) to avoid fighting the constraint.
    """
    N = model.N
    Phi = Phi0.copy()
    floor_val = -(1.0 - lapse_floor) * model.cstar_sq if lapse_floor else None

    for it in range(max_iter):
        F = model.residual(Phi, proxy=proxy)
        lapse = 1.0 + Phi / model.cstar_sq

        # Identify clamped shells (at floor)
        if floor_val is not None:
            clamped = Phi <= floor_val + 1e-13
            F[clamped] = 0.0
        else:
            clamped = np.zeros(N, dtype=bool)

        res = np.max(np.abs(F))

        if verbose and (it % 10 == 0 or it < 5 or res < tol):
            nc = int(np.sum(clamped))
            extra = f", clamped={nc}" if nc > 0 else ""
            print(f"    Newton {it:4d}: |F|={res:.3e}, "
                  f"min(N)={lapse.min():.6f}{extra}")

        if res < tol:
            return Phi, True, it

        # Jacobian
        sub, diag, sup = model.jacobian(Phi, proxy=proxy)

        # Replace clamped shells with identity (dPhi=0)
        for n in range(N):
            if clamped[n]:
                diag[n] = 1.0
                if n > 0:
                    sub[n - 1] = 0.0
                if n < N - 1:
                    sup[n] = 0.0

        try:
            dPhi = _solve_tridiag(sub, diag, sup, -F)
        except Exception as e:
            if verbose:
                print(f"    Tridiag solve failed: {e}")
            return Phi, False, it

        # Line search: require strict decrease (Armijo condition)
        alpha = 1.0
        for _ in range(40):
            Phi_trial = Phi + alpha * dPhi
            if floor_val is not None:
                Phi_trial = np.maximum(Phi_trial, floor_val)

            F_trial = model.residual(Phi_trial, proxy=proxy)
            if floor_val is not None:
                F_trial[Phi_trial <= floor_val + 1e-13] = 0.0

            if np.max(np.abs(F_trial)) < res * (1.0 - 1e-4 * alpha):
                break
            alpha *= 0.5
            if alpha < 1e-12:
                break

        if alpha < 1e-12:
            # Newton step makes no progress; give up
            if verbose:
                print(f"    Newton stalled at iter {it}: |F|={res:.3e}")
            return Phi, False, it

        Phi = Phi + alpha * dPhi
        if floor_val is not None:
            Phi = np.maximum(Phi, floor_val)

    F = model.residual(Phi, proxy=proxy)
    if floor_val is not None:
        F[Phi <= floor_val + 1e-13] = 0.0
    if verbose:
        print(f"    Newton not converged: |F|={np.max(np.abs(F)):.3e}")
    return Phi, False, max_iter


def solve_proxy(model, Phi_seed=None, tol=1e-12, lapse_floor=0.01,
                verbose=False):
    """Solve the proxy equation: Picard warmup -> Newton polish."""
    if Phi_seed is None:
        Phi_seed = np.zeros(model.N)

    Phi, _ = picard_proxy(
        model, Phi_seed, tol=1e-8, max_iter=2000,
        mixing=0.3, lapse_floor=lapse_floor, verbose=verbose)

    Phi, conv, _ = newton_twostate(
        model, Phi, proxy=True, tol=tol, max_iter=200,
        lapse_floor=lapse_floor, verbose=verbose)

    return Phi, conv


def solve_full(model, Phi_seed=None, tol=1e-12, lapse_floor=0.01,
               verbose=False):
    """
    Solve the full two-state equation.

    Strategy: proxy Picard+Newton -> full Newton from proxy seed.
    """
    if Phi_seed is None:
        Phi_seed = np.zeros(model.N)

    # Step 1: Solve proxy (stable Picard + Newton)
    Phi_proxy, _ = solve_proxy(
        model, Phi_seed, tol=tol, lapse_floor=lapse_floor, verbose=False)

    # Step 2: Full Newton from proxy seed
    Phi_full, conv, _ = newton_twostate(
        model, Phi_proxy, proxy=False, tol=tol, max_iter=200,
        lapse_floor=lapse_floor, verbose=verbose)

    return Phi_full, conv, Phi_proxy


def solve_full_constrained(model, Phi_seed=None, tol=1e-12,
                           lapse_floor=0.01, verbose=False):
    """
    Solve the two-state closure equation with beta0-stationarity constraint.

    The augmented system is (N+1) x (N+1):
      F(Phi, beta0) = 0   (N equations: closure)
      G(Phi, beta0) = 0   (1 equation:  global energy matching)

    Strategy: trust-region least-squares (scipy) on the combined system.
    The solution manifold F=0 has a fold near the G=0 crossing, making
    simple bordered Newton unstable.  Trust-region methods handle the
    near-singular Jacobian gracefully via Levenberg-Marquardt damping.

    Phase 1: least_squares finds (Phi, beta0) with both F~0 and G~0.
    Phase 2: Newton polish at fixed beta0 to reach machine precision.

    Returns: (Phi, beta0_final, conv, Phi_unconstrained)
    """
    from scipy.optimize import least_squares

    beta0_orig = model.beta0
    N = model.N

    # Step 1: solve unconstrained as seed
    Phi_unc, conv_unc, _ = solve_full(
        model, Phi_seed, tol=tol, lapse_floor=lapse_floor, verbose=False)
    if not conv_unc and verbose:
        print("  Warning: unconstrained solve did not converge; using as seed anyway")

    floor_val = -(1.0 - lapse_floor) * model.cstar_sq if lapse_floor else None

    G_unc = model.global_constraint(Phi_unc)
    # Weight G so that (G*w)^2 ~ O(1) at seed, comparable to sum(F_i^2)
    # at nearby beta0 values where F~O(1). Use w = 0.01 empirically.
    G_weight = 0.01

    def combined_residual(x):
        """Combined (N+1)-vector residual: [F(Phi,beta0), G*weight]."""
        Phi = np.maximum(x[:N], floor_val) if floor_val is not None else x[:N].copy()
        beta0 = max(x[N], 1e-6)
        model.set_beta0(beta0)
        F = model.residual(Phi, proxy=False)
        G = model.global_constraint(Phi)
        if floor_val is not None:
            clamped = Phi <= floor_val + 1e-13
            F[clamped] = 0.0
        return np.append(F, G * G_weight)

    # Phase 1: trust-region least-squares
    x0 = np.append(Phi_unc, beta0_orig)
    lb = np.full(N + 1, -np.inf)
    ub = np.full(N + 1, np.inf)
    if floor_val is not None:
        lb[:N] = floor_val
    lb[N] = 0.01  # beta0 > 0
    ub[N] = 1.0

    if verbose:
        print(f"    Phase 1: trust-region least-squares (N+1={N+1} unknowns)")
        print(f"    Seed: beta0={beta0_orig:.6f}, |G|={abs(G_unc):.3e}")

    result = least_squares(
        combined_residual, x0, method='trf', bounds=(lb, ub),
        ftol=1e-14, xtol=1e-14, gtol=1e-14, max_nfev=200000, verbose=0)

    beta0_lsq = result.x[N]
    Phi_lsq = result.x[:N].copy()
    if floor_val is not None:
        Phi_lsq = np.maximum(Phi_lsq, floor_val)

    model.set_beta0(beta0_lsq)
    G_lsq = model.global_constraint(Phi_lsq)
    F_lsq = model.residual(Phi_lsq, proxy=False)
    if floor_val is not None:
        F_lsq[Phi_lsq <= floor_val + 1e-13] = 0.0

    if verbose:
        print(f"    LSQ done: beta0={beta0_lsq:.8f}, |F|={np.max(np.abs(F_lsq)):.3e}, "
              f"|G|={abs(G_lsq):.3e}, nfev={result.nfev}")

    # Phase 2: Newton polish at fixed beta0
    Phi_pol, conv_pol, nit = newton_twostate(
        model, Phi_lsq, proxy=False, tol=tol, max_iter=200,
        lapse_floor=lapse_floor, verbose=False)

    G_pol = model.global_constraint(Phi_pol)
    F_pol = model.residual(Phi_pol, proxy=False)
    if floor_val is not None:
        F_pol[Phi_pol <= floor_val + 1e-13] = 0.0
    F_pol[-1] = 0.0  # boundary

    # Use whichever solution is better (Newton polish may or may not improve)
    res_lsq = max(np.max(np.abs(F_lsq)), abs(G_lsq))
    res_pol = max(np.max(np.abs(F_pol)), abs(G_pol))

    if conv_pol and res_pol < res_lsq:
        Phi_final = Phi_pol
        if verbose:
            print(f"    Newton polish: |F|={np.max(np.abs(F_pol)):.3e}, "
                  f"|G|={abs(G_pol):.3e} (improved)")
    else:
        Phi_final = Phi_lsq
        if verbose:
            status = "kept LSQ" if not conv_pol else "LSQ was better"
            print(f"    Newton polish: {status}")

    # Convergence: interior |F| < tol and |G| < 1
    F_final = model.residual(Phi_final, proxy=False)
    F_final[-1] = 0.0
    if floor_val is not None:
        F_final[Phi_final <= floor_val + 1e-13] = 0.0
    G_final = model.global_constraint(Phi_final)
    # Interior residual (exclude boundary)
    res_interior = np.max(np.abs(F_final[:N - 1]))
    conv = (res_interior < max(tol, 1e-7)) and (abs(G_final) < 1.0)

    if verbose:
        lapse = 1.0 + Phi_final / model.cstar_sq
        print(f"    Final: beta0={beta0_lsq:.8f}, |F_int|={res_interior:.3e}, "
              f"|G|={abs(G_final):.3e}, min(N)={lapse.min():.6f}, conv={conv}")

    # Restore model to final beta0
    model.set_beta0(beta0_lsq)
    return Phi_final, beta0_lsq, conv, Phi_unc


def twostate_sweep(V0_values, N=200, t0=1.0, n_core=5, beta0=0.1,
                    cstar_sq=0.5, lapse_floor=0.01, tol=1e-12, verbose=False):
    """
    Continuation sweep in V0: solve both proxy and full equations.
    """
    results = []
    Phi_seed = np.zeros(N)

    for i, V0 in enumerate(V0_values):
        model = TwoStateShellModel(N=N, t0=t0, V0=V0, n_core=n_core,
                                    beta0=beta0, cstar_sq=cstar_sq)

        if verbose:
            print(f"\n  [{i+1}/{len(V0_values)}] V0 = {V0:.4f}")

        # Solve proxy
        Phi_proxy, conv_p = solve_proxy(
            model, Phi_seed, tol=tol, lapse_floor=lapse_floor, verbose=False)

        # Solve full (proxy seed is computed internally; pass continuation seed)
        Phi_full, conv_f, _ = solve_full(
            model, Phi_seed, tol=tol, lapse_floor=lapse_floor, verbose=verbose)

        lapse_p = 1.0 + Phi_proxy / cstar_sq
        lapse_f = 1.0 + Phi_full / cstar_sq
        rs_p = extract_rs(Phi_proxy, model)
        rs_f = extract_rs(Phi_full, model)

        # Cross-residual: proxy solution in full equation
        F_cross = model.residual(Phi_proxy, proxy=False)
        F_cross[-1] = 0.0
        floor_val = -(1.0 - lapse_floor) * cstar_sq
        F_cross[Phi_proxy <= floor_val + 1e-13] = 0.0

        results.append({
            "V0": V0, "model": model,
            "Phi_proxy": Phi_proxy.copy(),
            "Phi_full": Phi_full.copy(),
            "min_N_proxy": lapse_p.min(),
            "min_N_full": lapse_f.min(),
            "rs_proxy": rs_p,
            "rs_full": rs_f,
            "F_cross": np.max(np.abs(F_cross)),
            "conv_proxy": conv_p,
            "conv_full": conv_f,
        })

        if verbose:
            dPhi = np.max(np.abs(Phi_full - Phi_proxy))
            print(f"    proxy: min(N)={lapse_p.min():.6f}, rs={rs_p:.3f}")
            print(f"    full:  min(N)={lapse_f.min():.6f}, rs={rs_f:.3f}, "
                  f"|dPhi|={dPhi:.3e}, conv={conv_f}")

        # Update seed for continuation
        if conv_f:
            Phi_seed = Phi_full.copy()
        elif conv_p:
            Phi_seed = Phi_proxy.copy()

    return results


def floor_independence_study(V0, floors, N=200, t0=1.0, n_core=5,
                              beta0=0.1, cstar_sq=0.5, verbose=False):
    """
    Solve at fixed V0 with decreasing lapse floors.

    Tests self-regularization: if min(N) plateaus above all floors,
    the solution is floor-independent and the lapse is naturally positive.
    """
    results = []
    Phi_seed = np.zeros(N)

    for floor in floors:
        model = TwoStateShellModel(N=N, t0=t0, V0=V0, n_core=n_core,
                                    beta0=beta0, cstar_sq=cstar_sq)

        if verbose:
            print(f"\n  floor={floor:.1e}")

        Phi_full, conv, Phi_proxy = solve_full(
            model, Phi_seed, tol=1e-12,
            lapse_floor=floor, verbose=verbose)

        lapse = 1.0 + Phi_full / cstar_sq
        rs = extract_rs(Phi_full, model)

        results.append({
            "floor": floor,
            "Phi": Phi_full.copy(),
            "Phi_proxy": Phi_proxy.copy(),
            "min_N": lapse.min(),
            "rs": rs,
            "conv": conv,
            "model": model,
        })

        if verbose:
            print(f"    min(N)={lapse.min():.8f}, rs={rs:.4f}, conv={conv}")

        Phi_seed = Phi_full.copy()

    return results
