#!/usr/bin/env python3
"""Colored (banded) forward-mode Jacobian for the TK closure solvers.

The closure Jacobian is 51% of a Newton iteration (472 ms of 925 ms at N=200 on
an A6000) and is numerically BANDED, because the correlations that build it
decay exponentially in shell separation.  Measured at beta0 t0 = 5, N = 200:

    bandwidth  12 : largest entry outside the band = 5.9e-09 of the peak
    bandwidth  20 :                                  4.5e-11
    bandwidth  30 :                                  9.3e-14
    bandwidth  40 :                                  4.2e-15   (machine epsilon)

A banded Jacobian does not need N forward passes.  Columns further apart than
2b have disjoint row supports, so the columns split into p = 2b+1 groups by
j mod p, and p forward passes recover every entry.  The unscrambling is EXACT
rather than a least-squares fit: within one residue class consecutive columns
differ by p > 2b, so at most one of them can reach any given row.

Measured against the full N-tangent jacfwd, N = 200:

    b= 8 ( 17 passes)  6.71x on the Jacobian,  2.36x end to end
    b=12 ( 25 passes)  5.48x                   2.12x
    b=20 ( 41 passes)  4.19x                   2.15x
    b=30 ( 61 passes)  4.61x                   2.03x
    b=40 ( 81 passes)  3.53x                   2.02x

every one of which converged to the same Phi as the exact Jacobian to ~1e-10,
which is the solver's own tolerance.  (b=30 beating b=20 despite more passes is
a GPU tiling effect, which is why calibrate_bandwidth measures rather than
guesses.)

WHY THIS CANNOT CHANGE A PUBLISHED NUMBER.  Truncating the Jacobian perturbs
the Newton DIRECTION only.  The fixed point is defined by the full fp64
residual test, which is untouched, so a slightly wrong direction can cost an
iteration but cannot move where the iteration lands.  That is the whole safety
argument, and it is why the truncation tolerance can be loose while the
residual tolerance stays at 1e-10.

ALWAYS CALIBRATE, NEVER HARD-CODE.  The bandwidth tracks the correlation
length, so a colder beta0 t0 spreads the Jacobian and needs a wider band.  A
hard-coded 30 taken from the beta0 t0 = 5 scan would silently truncate the
temperature chain.  calibrate_bandwidth returns None when no band is found, and
callers must then keep the exact Jacobian -- costing speed, not accuracy.

Four other optimizations were tried and measured SLOWER; see the header of
tests/bench_tk_banded_jac.py before attempting any of them again.
"""
from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp
from jax import jit


def calibrate_bandwidth(jac_full, probe_args, N, tol=1e-13, margin=1.6):
    """Smallest b whose truncation is below tol, times a safety margin.

    jac_full   : the exact Jacobian, callable as jac_full(*probe_args)
    probe_args : a REPRESENTATIVE state.  Calibrating at Phi = 0 would measure
                 the flat chain, whose band is narrower than a star's.
    Returns None if no band reaches tol, meaning: keep the exact Jacobian.
    """
    J = np.asarray(jac_full(*probe_args))
    scale = float(np.abs(J).max())
    if not np.isfinite(scale) or scale <= 0.0:
        return None
    off = np.abs(np.subtract.outer(np.arange(N), np.arange(N)))
    for cand in range(2, N):
        sel = J[off > cand]
        if sel.size == 0:                      # band covers the whole matrix
            return None
        if float(np.abs(sel).max()) / scale < tol:
            b = int(np.ceil(cand * margin))
            # p = 2b+1 passes must beat N, or the colouring costs more than the
            # exact Jacobian it replaces.  At beta0 t0 = 12-15 the honest
            # bandwidth is ~199, where p = 399 against N = 200.
            if 2 * b + 1 >= N:
                return None
            return b
    return None


def make_colored_jac(residual, N, b):
    """p = 2b+1 forward passes instead of N.

    residual(P, *rest) -> (N,).  Differentiated with respect to P only; *rest
    is passed through untouched, so this works for the (Phi, mu),
    (Phi, mu, beta0) and (Phi, mu, mask) signatures alike.
    """
    p = 2 * b + 1
    cols = jnp.arange(N)
    V = (cols[None, :] % p == jnp.arange(p)[:, None]).astype(jnp.float64)
    rows = jnp.arange(N)
    band = jnp.abs(rows[:, None] - cols[None, :]) <= b
    pick = jnp.broadcast_to((cols % p)[None, :], (N, N))
    rowi = jnp.broadcast_to(rows[:, None], (N, N))

    @jit
    def jac(P, *rest):
        JV = jax.vmap(
            lambda v: jax.jvp(lambda x: residual(x, *rest), (P,), (v,))[1]
        )(V)                                    # (p, N):  JV[c, i]
        return jnp.where(band, JV[pick, rowi], 0.0)

    return jac


def install(residual, jac_full, probe_args, N, label="", tol=1e-13, margin=1.6):
    """Calibrate and build, or fall back to the exact Jacobian.  Returns
    (jacobian_callable, bandwidth_or_None) and prints which branch was taken --
    a silent fallback would look like a mysterious slowdown."""
    b = calibrate_bandwidth(jac_full, probe_args, N, tol=tol, margin=margin)
    if b is None:
        print(f"  {label}Jacobian is NOT banded at {tol:g}; "
              f"keeping the exact jacfwd", flush=True)
        return jac_full, None
    jac = make_colored_jac(residual, N, b)
    jax.block_until_ready(jac(*probe_args))
    print(f"  {label}Colored Jacobian: bandwidth {b}, {2*b+1} forward passes "
          f"instead of {N}", flush=True)
    return jac, b
