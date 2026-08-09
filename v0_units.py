#!/usr/bin/env python3
"""The one place the shorthand -> derived V0 conversion lives.

The defect-model scripts were all written against the solver's lattice-unit
shorthand pref = beta0/c_*^2.  The closure of eq:sc-closure actually sources
with the DERIVED prefactor

    P = c_*^2 t0^2 / (lambda eps_b),

lambda the per-channel alpha_S^bond and eps_b the flat-chain BULK bond energy
of eq:sc-sumrule, so the same physical point is

    V0_derived = V0_shorthand * (beta0 / c_*^2) / P,

which is 4.33e-3 at the production parameters -- a factor of 231.  Every V0
literal, npz key and figure selection in the defect path has to agree on that
number, and each place that reimplemented it became a way for them to disagree
silently: a plotter that guessed the scale by nearest match selected
NEIGHBOURING solutions in seven of eight calls while its labels kept claiming
the requested ones.

So it is computed once, here, from parameters read out of the data file rather
than remembered.  compute_exact_data_gpu.py holds the jax original; this is its
numpy transcription for consumers that must not drag jax in, and
tests/test_tk_closure.py pins the two together so they cannot drift.

The conversion is a RELABEL OF THE GRID, not of a solution: the closure is
nonlinear in V0, so a point converted and re-solved differs from the same point
rescaled in place by 0.03% at weak field rising to 1.36% at V0 = 0.066.  Convert
the grid, then solve.  Never rescale a stored Phi.
"""
import numpy as np

# per-channel alpha_S^bond; must match LAM_CLOSURE in compute_exact_data_gpu.py
LAM_CLOSURE = 0.217

# npz key format.  Was "%.4f", under which the converted grid (2.2e-5 .. 2.9e-4)
# formats as 0.0000/0.0001/0.0002/0.0003: 15 distinct solutions collide onto 4
# tags and overwrite each other, losing 135 of 351 keys with no error anywhere.
V0_KEY_FMT = ".8f"


def flat_bond_energy(N, t0, beta0):
    """eps_b = 2 t0 G_super at the BULK bond of the flat (Phi = 0) chain.

    numpy transcription of bond_correlations() in compute_exact_data_gpu.py:
    G = (exp(beta0 h) + I)^{-1} for the tridiagonal hopping h with off-diagonal
    -t0.  Read at the middle bond because the open-chain ends differ.
    """
    off = -t0 * np.ones(N - 1)
    H = np.diag(off, 1) + np.diag(off, -1)
    evals, evecs = np.linalg.eigh(H)
    f = 1.0 / (np.exp(np.clip(beta0 * evals, -500.0, 500.0)) + 1.0)
    vf = evecs * f[None, :]
    G_super = np.sum(evecs[:-1] * vf[1:], axis=1)
    return 2.0 * t0 * G_super[(N - 1) // 2]


def shorthand_to_derived(N, t0, beta0, cstar_sq, lam=LAM_CLOSURE):
    """The factor f with V0_derived = f * V0_shorthand.

    f depends on beta0 through eps_b: it is 4.33e-3, 0.102, 0.352, 1.57 at
    beta0 = 0.1, 0.5, 1, 3 and crosses 1 near beta0 t0 = 2, so there is no single
    number and every grid must be converted at ITS OWN temperature.
    """
    eps_b = flat_bond_energy(N, t0, beta0)
    P = cstar_sq * t0**2 / (lam * eps_b)
    return (beta0 / cstar_sq) / P


def scale_from_npz(D, lam=LAM_CLOSURE):
    """The factor, built from the parameters the data file itself records."""
    return shorthand_to_derived(int(D["N"]), float(D["t0"]),
                                float(D["beta0"]), float(D["cstar_sq"]), lam)


def sol_tag(V0_derived, prefix="sol"):
    """npz key prefix for a solved (derived-units) V0."""
    return f"{prefix}_{V0_derived:{V0_KEY_FMT}}"


def resolve_shorthand(D, *wanted, prefix="sol"):
    """Solved V0 values for points named in SHORTHAND units.

    Every requested point must be present in the solved grid to a relative
    1e-9.  A wrong scale cannot hit a whole grid to that tolerance, so the
    exact-match requirement is itself the check that the conversion is right --
    and a missing point raises, naming what is absent, rather than being
    dropped into an empty axes or substituted by a neighbour.
    """
    avail = np.asarray(sorted(np.atleast_1d(D["solution_V0s"]).astype(float)))
    if avail.size == 0:
        raise SystemExit("resolve_shorthand: data file contains no solutions")
    scale = scale_from_npz(D)
    out, missing = [], []
    for w in wanted:
        target = float(w) * scale
        j = int(np.argmin(np.abs(avail - target)))
        if abs(avail[j] - target) > 1e-9 * max(abs(target), 1e-30):
            missing.append((w, target, float(avail[j])))
        else:
            out.append(float(avail[j]))
    if missing:
        lines = "\n".join(
            f"    shorthand {w:g} -> {t:.8e}, nearest solved {n:.8e} "
            f"(off by {abs(n - t) / t:.2%})" for w, t, n in missing)
        raise SystemExit(
            f"resolve_shorthand: {len(missing)} requested V0 not in the solved "
            f"grid:\n{lines}\n  scale in use: {scale:.12e}\n"
            f"  solved grid: {avail.min():.8e} .. {avail.max():.8e} "
            f"({avail.size} points)\n"
            f"  Either the data file came from a different V0 grid, or the "
            f"closure prefactor changed and the grids need regenerating.")
    return out
