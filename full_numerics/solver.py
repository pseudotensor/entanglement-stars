"""
Newton solver and continuation for the self-consistent closure equation.
"""

import numpy as np
from .physics import ShellModel


def newton_solve(model, Phi0, proxy=False, tol=1e-12, max_iter=200,
                 damping=1.0, min_damping=1e-4, verbose=True):
    """
    Newton's method with tridiagonal Jacobian and optional line search.

    Returns: (Phi, converged, n_iters, history)
    """
    Phi = Phi0.copy()
    history = []

    for it in range(max_iter):
        F = model.residual(Phi, proxy=proxy)
        res_norm = np.max(np.abs(F))
        lapse = 1.0 + Phi / model.cstar_sq
        min_lapse = lapse.min()

        history.append({
            "iter": it, "res": res_norm, "min_lapse": min_lapse,
        })

        if verbose and (it % 10 == 0 or it < 5 or res_norm < tol):
            print(f"  iter {it:4d}: |F|_inf = {res_norm:.3e}, "
                  f"min(N) = {min_lapse:.6f}")

        if res_norm < tol:
            if verbose:
                print(f"  Converged at iter {it}")
            return Phi, True, it, history

        # Newton step
        sub, diag, sup = model.jacobian(Phi, proxy=proxy)
        try:
            dPhi = model.solve_tridiag(sub, diag, sup, -F)
        except Exception as e:
            if verbose:
                print(f"  Tridiag solve failed at iter {it}: {e}")
            return Phi, False, it, history

        # Line search: backtrack if residual increases
        alpha = damping
        for _ in range(20):
            Phi_trial = Phi + alpha * dPhi
            F_trial = model.residual(Phi_trial, proxy=proxy)
            if np.max(np.abs(F_trial)) < res_norm * 1.1:
                break
            alpha *= 0.5
            if alpha < min_damping:
                break

        Phi = Phi + alpha * dPhi

    if verbose:
        print(f"  Not converged after {max_iter} iters: |F| = {res_norm:.3e}")
    return Phi, False, max_iter, history


def picard_solve(model, Phi0, proxy=False, tol=1e-10, max_iter=500,
                 mixing=0.3, lapse_floor=None, verbose=True):
    """
    Picard (fixed-point) iteration: freeze nonlinear terms, solve linear system.
    Optionally clamp lapse at a floor.
    """
    N = model.N
    Phi = Phi0.copy()
    history = []

    for it in range(max_iter):
        lapse, Nbar = model.lapse_nbar(Phi)
        kappa = model.conductances(Nbar)
        min_lapse = lapse.min()

        # Build linear system: L_kappa * Phi_new = source
        # For proxy: source = -(beta0/c*^2) * rho_tilde
        # For full: source = (beta0/c*^2) * mismatch(Phi_current)
        pref = model.beta0 / model.cstar_sq

        # Assemble tridiagonal Laplacian
        sub_L = np.zeros(N - 1)
        diag_L = np.zeros(N)
        sup_L = np.zeros(N - 1)
        rhs = np.zeros(N)

        for n in range(N - 1):
            diag_L[n] += kappa[n]
            sup_L[n] -= kappa[n]
            diag_L[n + 1] += kappa[n]
            sub_L[n] -= kappa[n]

        if proxy:
            rhs[:model.n_core] = -pref * model.rho_tilde[:model.n_core]
        else:
            mismatch = model.energy_mismatch(Phi)
            rhs[:N-1] = pref * mismatch[:N-1]
            # Note: we freeze mismatch at current Phi for the linear solve

        # Boundary
        diag_L[N - 1] = 1.0
        sub_L[N - 2] = 0.0
        sup_L[N - 2] = 0.0  # Already 0 but be explicit
        rhs[N - 1] = 0.0

        Phi_new = model.solve_tridiag(sub_L, diag_L, sup_L, rhs)

        # Optional lapse floor
        if lapse_floor is not None:
            Phi_floor = -(1.0 - lapse_floor) * model.cstar_sq
            Phi_new = np.maximum(Phi_new, Phi_floor)

        # Mix
        Phi_mixed = mixing * Phi_new + (1.0 - mixing) * Phi

        # Convergence check
        rel_change = np.max(np.abs(Phi_mixed - Phi)) / max(np.max(np.abs(Phi)), 1e-15)
        history.append({
            "iter": it, "rel_change": rel_change, "min_lapse": min_lapse,
        })

        if verbose and (it % 20 == 0 or it < 3):
            print(f"  iter {it:4d}: |dPhi|/|Phi| = {rel_change:.3e}, "
                  f"min(N) = {min_lapse:.6f}")

        if rel_change < tol:
            if verbose:
                print(f"  Converged at iter {it}")
            return Phi_mixed, True, it, history

        Phi = Phi_mixed

    if verbose:
        print(f"  Not converged after {max_iter} iters")
    return Phi, False, max_iter, history


def continuation_sweep(N=200, t0=1.0, n_core=5, beta0=0.1, cstar_sq=0.5,
                       V0_values=None, proxy=False, tol=1e-12, verbose=True,
                       lapse_floor=None):
    """
    Sweep V0 from small to large, using previous solution as initial guess.
    Uses Picard to get a seed, then Newton to polish.
    """
    if V0_values is None:
        V0_values = np.concatenate([
            np.linspace(0.001, 0.05, 10),
            np.linspace(0.06, 0.3, 15),
            np.linspace(0.35, 1.0, 15),
            np.linspace(1.1, 2.0, 10),
        ])

    results = []
    Phi = np.zeros(N)

    for i, V0 in enumerate(V0_values):
        model = ShellModel(N=N, t0=t0, V0=V0, n_core=n_core,
                           beta0=beta0, cstar_sq=cstar_sq)

        if verbose:
            lapse0 = 1.0 + Phi / cstar_sq
            print(f"\n--- V0 = {V0:.4f} ({i+1}/{len(V0_values)}) "
                  f"[seed min(N) = {lapse0.min():.4f}] ---")

        # Step 1: Picard to get a reasonable guess (few iters)
        Phi_picard, conv_p, _, _ = picard_solve(
            model, Phi, proxy=proxy, tol=1e-6, max_iter=100,
            mixing=0.3, lapse_floor=lapse_floor, verbose=False)

        # Step 2: Newton to polish
        Phi_newton, conv_n, n_iters, hist = newton_solve(
            model, Phi_picard, proxy=proxy, tol=tol, max_iter=200,
            verbose=verbose)

        lapse = 1.0 + Phi_newton / model.cstar_sq
        F = model.residual(Phi_newton, proxy=proxy)

        results.append({
            "V0": V0,
            "Phi": Phi_newton.copy(),
            "converged": conv_n,
            "n_iters": n_iters,
            "min_lapse": lapse.min(),
            "residual": np.max(np.abs(F)),
            "model": model,
        })

        if verbose:
            rs = extract_rs(Phi_newton, model)
            print(f"  Result: min(N)={lapse.min():.6f}, "
                  f"r_s/r_c={rs/model.r[n_core-1]:.3f}, "
                  f"|F|={np.max(np.abs(F)):.2e}")

        if conv_n:
            Phi = Phi_newton.copy()  # use as seed for next V0
        else:
            if verbose:
                print(f"  Newton failed at V0={V0:.4f}, stopping sweep")
            break

    return results


def extract_rs(Phi, model, vacuum=False):
    """Extract Schwarzschild radius from far-field Phi.

    Two modes controlled by the `vacuum` flag:

    vacuum=False (default, thermal bath):
        Assumes Yukawa-screened potential Phi ~ -(GM/r)*exp(-r/xi_Y).
        The product GM_est = -Phi(r)*r peaks in the sub-Yukawa plateau.
        We average over ±5 shells around this peak.

    vacuum=True (asymptotically vacuum BC):
        The potential obeys Phi(R)=0 at the outer boundary, so the
        exterior solution is Phi = -GM*(1/r - 1/R), not -GM/r.
        We correct for this: GM_est = -Phi(r)*r*R/(R-r), which
        should be constant in the far field.
    """
    r = model.r
    n_core = model.n_core
    R = r[-1]  # outer boundary radius
    i_lo = n_core + 3
    i_hi = min(model.N - 5, model.N // 2)
    if i_lo >= i_hi:
        return 0.0
    if not np.any(np.abs(Phi[i_lo:i_hi]) > 1e-15):
        return 0.0

    if vacuum:
        # BC-corrected extraction: Phi = -GM*(1/r - 1/R)
        # => GM = -Phi * r * R / (R - r)
        r_slice = r[i_lo:i_hi]
        GM_est = -Phi[i_lo:i_hi] * r_slice * R / (R - r_slice)
        GM = np.median(GM_est)
    else:
        GM_est = -Phi[i_lo:i_hi] * r[i_lo:i_hi]
        # Find peak of GM_est (the sub-Yukawa plateau)
        i_peak = np.argmax(GM_est)
        # Average over ±5 shells around the peak (clipped to window)
        hw = 5
        j_lo = max(0, i_peak - hw)
        j_hi = min(len(GM_est), i_peak + hw + 1)
        GM = np.mean(GM_est[j_lo:j_hi])

    return 2.0 * GM / model.cstar_sq
