#!/usr/bin/env python
"""engine_fields_v1.py (lead's draft; SPEC_final4b_engine.md sec. 3): the discretized constraints in the radial variable, lattice units,
g = G/a^2.  Hamiltonian constraint on faces: W_{n+1/2} - W_{n-1/2} = 2 g N_n^2 eps_n^tot, W_{1/2} = 0, W = r v^2, v_n = -sqrt(W_n/R_n)
with W_n the face average.  Momentum constraint on links: (1/N)_{n+1} - (1/N)_n = g (J_n^tot + J_{n+1}^tot)/(2 R_{n+1/2} v_{n+1/2}),
(1/N)_collar = 1, integrated inward.  The pair iterated to tol.  eps_n^tot and J_n^tot are the shell totals (spec 2.2)."""
import numpy as np


def solve_constraints(eps_tot, J_tot, g, R, N0=None, tol=1e-13, maxit=500):
    n = len(R)
    N = np.ones(n) if N0 is None else np.array(N0, float)
    info = {"it": 0, "converged": False}
    for it in range(maxit):
        Wf = np.concatenate([[0.0], np.cumsum(2.0 * g * N ** 2 * eps_tot)])          # faces R_{-1/2}.. : Wf[0] = W_{1/2} = 0
        Wn = 0.5 * (Wf[:-1] + Wf[1:])
        if np.any(Wn < -1e-14):
            info["negative_W"] = float(Wn.min())                                     # negative energy inside: flagged, not clipped silently
        v = -np.sqrt(np.maximum(Wn, 0.0) / R)
        Jl = 0.5 * (J_tot[:-1] + J_tot[1:])
        Rl = 0.5 * (R[:-1] + R[1:])
        vl = 0.5 * (v[:-1] + v[1:])
        safe = np.abs(vl) > 1e-14
        inc = np.zeros(n - 1)
        inc[safe] = g * Jl[safe] / (Rl[safe] * vl[safe])
        invN = np.ones(n)
        invN[:-1] = 1.0 - np.cumsum(inc[::-1])[::-1]                                  # (1/N)_n = 1 - sum_{k >= n} inc_k
        if np.any(invN <= 0):
            info["nonpositive_lapse"] = True
            invN = np.maximum(invN, 1e-12)
        Nn = 1.0 / invN
        d = float(np.max(np.abs(Nn - N)))
        N = Nn
        info["it"] = it + 1
        if d < tol:
            info["converged"] = True
            break
    return N, v, Wf, info


def charge(Wf, g, collar):
    """E read at the collar face (W_collar/(2g)); equal to sum_{n < collar} N_n^2 eps_n^tot by construction (the Gauss identity)."""
    return float(Wf[collar] / (2.0 * g))


def trapping_radius(N, v, R):
    """zeros of N - |v| (linear interpolation); the outermost is the trapping surface of the external frame."""
    f = N - np.abs(v)
    s = np.nonzero(f[:-1] * f[1:] < 0)[0]
    return [float(R[i] - f[i] * (R[i + 1] - R[i]) / (f[i + 1] - f[i])) for i in s]


def exterior_check(N, v, R, Wf, g, E, r_star):
    sel = R > r_star + 10
    return {"max_abs_N_minus_1": float(np.max(np.abs(N[sel] - 1))) if sel.any() else None,
            "max_abs_rv2_minus_2gE": float(np.max(np.abs(R[sel] * v[sel] ** 2 - 2 * g * E))) if sel.any() else None}


def residuals(eps_tot, J_tot, g, R, N, v, Wf):
    """the two discrete constraints evaluated at (N, v, Wf) with the given sources: the record's residual per shell."""
    r1 = (Wf[1:] - Wf[:-1]) - 2.0 * g * N ** 2 * eps_tot
    invN = 1.0 / N
    Jl = 0.5 * (J_tot[:-1] + J_tot[1:]); Rl = 0.5 * (R[:-1] + R[1:]); vl = 0.5 * (v[:-1] + v[1:])
    r2 = (invN[1:] - invN[:-1]) - np.where(np.abs(vl) > 1e-14, g * Jl / (Rl * np.where(np.abs(vl) > 1e-14, vl, 1.0)), 0.0)
    return r1, r2


if __name__ == "__main__":
    # BUILDER (J0): a Gaussian source at rest, J = 0: N = 1 everywhere (no momentum, no lapse), r v^2 = 2 g E beyond it; then a source
    # with J_n = +|v_n| eps_n/N_n (the static-frame momentum, spec 3.2): N < 1 inside (the sign check 3.3), N = 1 beyond.
    R = np.arange(1, 201, dtype=float)
    eps = 0.5 * np.exp(-((R - 20) / 6) ** 2); g = 0.25
    N, v, Wf, info = solve_constraints(eps, 0 * eps, g, R)
    E = charge(Wf, g, 150)
    print(info, E, exterior_check(N, v, R, Wf, g, E, 40.0))
    for it in range(50):
        J = np.abs(v) * eps / N
        N, v, Wf, info = solve_constraints(eps, J, g, R, N0=N)
    print("N_min", N.min(), "at", R[np.argmin(N)], "E", charge(Wf, g, 150), "trap", trapping_radius(N, v, R))
