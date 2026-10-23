#!/usr/bin/env python3
"""Is matrix-free Newton-Krylov faster and smaller than jacfwd, on the GPU?

The current solver forms the Jacobian with jacfwd: N tangents through two dense
eigendecompositions.  That is O(N^4) work and O(N^3) memory -- measured, the
memory is 104*N^3 bytes, which asked for 49.71 GiB at N=800 and matches
104*800^3 to within 1%.  So N ~ 780 is a hard wall on a 49 GB card.

The alternative never forms J.  GMRES needs only J.v, which is one jvp, so
memory falls to O(N^2) and the work per Newton step falls from N tangents to k,
the Krylov count.  Everything then rests on k, i.e. on the preconditioner.

Two preconditioners are compared, both TRIDIAGONAL and both solved with the
native lax.linalg.tridiagonal_solve (cuSPARSE on GPU, O(N) and parallel -- not
a sequential Thomas recurrence, which is the one thing a GPU is bad at):

  Mflat  the flat-conductance (proxy) Laplacian, kappa = g t0^2.  It depends on
         NEITHER Phi NOR beta0, so it is built once and reused for an entire
         sweep.  The proxy reproduces the full solve to 0.3-1.4% in weak field,
         so this should be excellent when cold and degrade when hot.

  Mkappa the same Laplacian built from the CURRENT conductances,
         kappa = g t0^2 Nbar^2 (BKM ratio).  It tracks the field, so it should
         hold up at the hot end where Mflat degrades.  Rebuilt per Newton step,
         which is O(N) and therefore free.

Reported: k for each, wall time per Newton step, peak device memory, and -- the
part that matters -- whether the Krylov solve lands on the same Phi as the
direct solve to the solver's own tolerance.

    python3 bench_newton_krylov.py [--sizes 200,400,800,1600]
"""
import argparse
import os
import sys
import time

os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, jacfwd

from compute_exact_data_gpu import (LAM_CLOSURE, convert_V0, bkm_bond_cov,
                                    make_residual_tempsweep)

T0, NCORE, CSQ, B0REF = 1.0, 5, 0.5, 0.1


def timeit(fn, n=3):
    jax.block_until_ready(fn())
    t = time.perf_counter()
    for _ in range(n):
        jax.block_until_ready(fn())
    return (time.perf_counter() - t) / n


def peak_mem_gb():
    try:
        s = jax.devices()[0].memory_stats()
        return s.get("peak_bytes_in_use", 0) / 2**30
    except Exception:
        return float("nan")


def tridiag_from_kappa(kappa, N):
    """Dirichlet-terminated conductance Laplacian, as (dl, d, du)."""
    d = jnp.zeros(N).at[:N-1].add(kappa).at[1:].add(kappa)
    du = -kappa
    dl = -kappa
    # last row is the Dirichlet condition Phi[N-1] = 0, matching the residual
    d = d.at[N-1].set(1.0)
    du = du.at[N-2].set(0.0)
    return dl, d, du


def make_apply(dl, d, du):
    """M^{-1}x via the native tridiagonal solve (parallel on GPU)."""
    dl_ = jnp.concatenate([jnp.zeros(1), dl])
    du_ = jnp.concatenate([du, jnp.zeros(1)])

    @jit
    def apply(x):
        return jax.lax.linalg.tridiagonal_solve(
            dl_[None, :], d[None, :], du_[None, :], x[None, :, None]
        )[0, :, 0]
    return apply


def flat_kappa(N, t0=T0):
    n = jnp.arange(1, N + 1, dtype=jnp.float64)
    return (4.0 * jnp.pi * n**2)[:N-1] * t0**2


def current_kappa(Phi, beta0, N, t0=T0, csq=CSQ):
    n = jnp.arange(1, N + 1, dtype=jnp.float64)
    g = 4.0 * jnp.pi * n**2
    lapse = 1.0 + Phi / csq
    Nbar = 0.5 * (lapse[:-1] + lapse[1:])
    off = -t0 * jnp.abs(Nbar)
    z = jnp.zeros(N, dtype=jnp.float64)
    bg = bkm_bond_cov(z, -t0 * jnp.ones(N - 1, dtype=jnp.float64), beta0)
    cov = bkm_bond_cov(z, off, beta0)
    return g[:-1] * t0**2 * Nbar**2 * (cov / jnp.maximum(bg, 1e-30))


def krylov_steps(Jv, F, M, tol=1e-8, cap=60):
    """Smallest restart count reaching ||J dx + F|| / ||F|| < tol."""
    nrm = float(jnp.max(jnp.abs(F)))
    for k in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, cap):
        x, _ = jax.scipy.sparse.linalg.gmres(
            Jv, -F, M=M, tol=1e-12, atol=0.0, restart=k, maxiter=1)
        x = jax.block_until_ready(x)
        r = float(jnp.max(jnp.abs(Jv(x) + F))) / nrm
        if r < tol:
            return k, r
    return None, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="200,400,800")
    ap.add_argument("--bt0", type=float, default=20.0)
    ap.add_argument("--V0", type=float, default=0.027)
    args = ap.parse_args()

    print(f"device: {jax.devices()[0]}   bt0={args.bt0}  V0={args.V0}\n")
    hdr = (f"{'N':>6} {'resid':>9} {'1 jvp':>9} {'jacfwd':>11} {'k(flat)':>8} "
           f"{'k(kappa)':>9} {'NK time':>9} {'speedup':>8} {'peakGB':>8}")
    print(hdr)
    print("-" * len(hdr))

    for N in [int(x) for x in args.sizes.split(",")]:
        V0 = convert_V0(args.V0, N, T0, B0REF, CSQ, LAM_CLOSURE)
        resid = make_residual_tempsweep(N, T0, NCORE, CSQ, lam=LAM_CLOSURE)
        V, B = jnp.float64(V0), jnp.float64(args.bt0 / T0)

        # a physically representative iterate, not Phi = 0
        Phi = jnp.array(-2.0e-2 * np.exp(-np.arange(N) / 12.0))
        Phi = Phi.at[-1].set(0.0)

        f = jit(lambda p: resid(p, V, B))
        F = jax.block_until_ready(f(Phi))
        t_res = timeit(lambda: f(Phi))

        jvp1 = jit(lambda p, v: jax.jvp(lambda x: resid(x, V, B), (p,), (v,))[1])
        v0 = jnp.ones(N, dtype=jnp.float64)
        t_jvp = timeit(lambda: jvp1(Phi, v0))
        Jv = lambda x: jvp1(Phi, x)

        # baseline
        try:
            Jf = jit(jacfwd(resid, argnums=0))
            J = jax.block_until_ready(Jf(Phi, V, B))
            t_jac = timeit(lambda: Jf(Phi, V, B), n=2)
            dx_direct = jnp.linalg.solve(
                J + jnp.diag(1e-10 * jnp.maximum(jnp.abs(jnp.diag(J)), 1e-10)), -F)
            del J
        except Exception as e:
            t_jac, dx_direct = float("nan"), None
            print(f"{N:6d}  jacfwd FAILED: {type(e).__name__}")

        Mflat = make_apply(*tridiag_from_kappa(flat_kappa(N), N))
        Mkap = make_apply(*tridiag_from_kappa(current_kappa(Phi, B, N), N))
        k_flat, _ = krylov_steps(Jv, F, Mflat)
        k_kap, r_kap = krylov_steps(Jv, F, Mkap)

        kbest = min([k for k in (k_flat, k_kap) if k], default=None)
        Mbest = Mkap if (k_kap and (not k_flat or k_kap <= k_flat)) else Mflat
        if kbest:
            t_nk = timeit(lambda: jax.scipy.sparse.linalg.gmres(
                Jv, -F, M=Mbest, tol=1e-12, atol=0.0, restart=kbest, maxiter=1)[0])
        else:
            t_nk = float("nan")

        sp = t_jac / t_nk if (np.isfinite(t_jac) and np.isfinite(t_nk)) else float("nan")
        print(f"{N:6d} {t_res*1e3:8.2f}m {t_jvp*1e3:8.2f}m "
              f"{t_jac*1e3:10.1f}m {str(k_flat):>8} {str(k_kap):>9} "
              f"{t_nk*1e3:8.2f}m {sp:7.1f}x {peak_mem_gb():7.2f}")

        # AGREEMENT: the Krylov step must land where the direct step lands
        if dx_direct is not None and kbest:
            x, _ = jax.scipy.sparse.linalg.gmres(
                Jv, -F, M=Mbest, tol=1e-12, atol=0.0, restart=kbest, maxiter=1)
            rel = float(jnp.max(jnp.abs(x - dx_direct)) /
                        jnp.max(jnp.abs(dx_direct)))
            print(f"       Newton step agrees with the direct solve to "
                  f"{rel:.2e} (relative)")


if __name__ == "__main__":
    main()
