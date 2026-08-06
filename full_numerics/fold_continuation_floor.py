#!/usr/bin/env python3
"""Is the TK 'fold' at beta_0 t_0 = 5 a bifurcation or a continuation failure?

Runs the same mu-continuation as tov_tk_bt5_eigenvalue.py, but with a line-search
Newton, steps refined to 2e-7, and the solver's lapse floor N_floor as a control
parameter.  If the branch terminates at a genuine fold the stopping mu is
independent of N_floor; if it stops because the floor clamp activates, the stopping
mu moves with it.

Data -> fold_continuation_floor.npz
"""
import os
import sys
import time

import numpy as np

os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, jacfwd

beta0, m0, csq, N = 5.0, 3.0, 5.0, 200
t0 = 1.0
n_arr = jnp.arange(1, N + 1, dtype=jnp.float64)
g = 4.0 * jnp.pi * n_arr ** 2
diag0 = jnp.zeros(N, dtype=jnp.float64)


def _bkm_cov(diag_vals, off_vals):
    H = jnp.diag(diag_vals) + jnp.diag(off_vals, 1) + jnp.diag(off_vals, -1)
    evals, evecs = jnp.linalg.eigh(H)
    f = 1.0 / (jnp.exp(jnp.clip(beta0 * evals, -500.0, 500.0)) + 1.0)
    diff = beta0 * (evals[:, None] - evals[None, :])
    phi = jnp.where(jnp.abs(diff) > 1e-12,
                    jnp.expm1(diff) / jnp.where(jnp.abs(diff) > 1e-12, diff, 1.0), 1.0)
    K = f[:, None] * (1.0 - f[None, :]) * phi
    U_n, U_n1 = evecs[:-1, :], evecs[1:, :]
    return (jnp.sum((U_n1 ** 2) @ K * (U_n ** 2), axis=1) +
            jnp.sum((U_n ** 2) @ K * (U_n1 ** 2), axis=1) -
            2.0 * jnp.sum((U_n * U_n1) @ K * (U_n * U_n1), axis=1))


def _bond_correlations(diag_vals, off_vals):
    H = jnp.diag(diag_vals) + jnp.diag(off_vals, 1) + jnp.diag(off_vals, -1)
    evals, evecs = jnp.linalg.eigh(H)
    f = 1.0 / (jnp.exp(jnp.clip(beta0 * evals, -500.0, 500.0)) + 1.0)
    vf = evecs * f[None, :]
    return jnp.sum(evecs * vf, axis=1), jnp.sum(evecs[:-1] * vf[1:], axis=1)


cov_bg = jnp.maximum(_bkm_cov(diag0, -t0 * jnp.ones(N - 1)), 1e-30)


def make_residual(N_floor):
    def _energy(Phi, mu):
        lapse = jnp.maximum(1.0 + Phi / csq, N_floor)
        Nbar = 0.5 * (lapse[:-1] + lapse[1:])
        kappa = g[:-1] * t0 ** 2 * Nbar ** 2 * (_bkm_cov(diag0, -t0 * jnp.abs(Nbar)) / cov_bg)
        return 0.5 * jnp.sum(kappa * (Phi[:-1] - Phi[1:]) ** 2)

    ge = jax.grad(_energy, argnums=0)

    def _residual(Phi, mu):
        lapse = jnp.maximum(1.0 + Phi / csq, N_floor)
        Nb = jnp.abs(0.5 * (lapse[:-1] + lapse[1:]))
        V = jnp.maximum(m0 * (1.0 - lapse) - mu, 0.0)
        # Source convention (eq. sc-rho, sec:ex-observable): bond term NEGATIVE
        # against a positive V term, ONE power of the lapse (the smeared
        # correlator Gs already supplies it -- a second factor "would count the
        # lapse twice"), split half onto each endpoint of the bond.
        _, Gs_s = _bond_correlations(diag0, -t0 * Nb)
        rho_s = (jnp.zeros(N).at[:-1].add(-1.0 * t0 * Gs_s * g[:-1])
                              .at[1:].add(-1.0 * t0 * Gs_s * g[:-1]))
        Gd_t, Gs_t = _bond_correlations(V, -t0 * Nb)
        rho_t = (jnp.zeros(N).at[:-1].add(-1.0 * t0 * Gs_t * g[:-1])
                              .at[1:].add(-1.0 * t0 * Gs_t * g[:-1])
                 + V * Gd_t * g)
        return (ge(Phi, mu) - (beta0 / csq) * (rho_s - rho_t)).at[N - 1].set(Phi[N - 1])
    return _residual


def run(N_floor, Phi0, mu0, mu_cap=2.9999):
    res = jit(make_residual(N_floor))
    jac = jit(jacfwd(make_residual(N_floor), argnums=0))
    jax.block_until_ready(res(jnp.zeros(N), jnp.float64(0.1)))
    jax.block_until_ready(jac(jnp.zeros(N), jnp.float64(0.1)))

    def newton(P0, mu, tol=1e-9, it_max=50):
        P = jnp.array(P0)
        mj = jnp.float64(mu)
        for _ in range(it_max):
            F = np.array(res(P, mj))
            nr = float(np.max(np.abs(F)))
            if nr < tol:
                return np.array(P), True
            try:
                d = np.linalg.solve(np.array(jac(P, mj)), -F)
            except np.linalg.LinAlgError:
                return np.array(P), False
            al = 1.0
            for _ in range(12):
                Pt = P + al * jnp.array(d)
                if np.max(np.abs(np.array(res(Pt, mj)))) < nr:
                    P = Pt
                    break
                al *= 0.5
            else:
                return np.array(P), False
        return np.array(P), float(np.max(np.abs(np.array(res(P, mj))))) < tol

    def ok(P):
        return (float(np.max(np.abs(P))) > 0.01
                and float(np.min(1.0 + P / csq)) > 1.5 * N_floor)

    mu, P = mu0, np.array(Phi0)
    step = 0.005
    track = []
    while step > 2e-7 and mu < mu_cap:
        Pt, conv = newton(P, mu + step)
        if conv and ok(Pt):
            mu, P = mu + step, Pt
            lam = np.sort(np.abs(np.linalg.eigvals(np.array(jac(jnp.array(P), jnp.float64(mu)))[:-1, :-1])))
            track.append((mu, float(np.min(1.0 + P / csq)),
                          int(np.sum(np.maximum(m0 * (1.0 - np.maximum(1 + P / csq, N_floor)) - mu, 0.0) > 1e-10)),
                          float(lam[0])))
        else:
            step *= 0.5
    return mu, P, np.array(track)


def main():
    d = np.load("tov_tk_chemical_el.npz", allow_pickle=True)
    # Seed on the connected MULTI-SHELL star branch, selected by its own
    # properties rather than by a hardcoded mu window.  The old code took the
    # highest mu below mu/m0 = 0.66 on the assumption that the star branch ran
    # to 0.658 with n_eff = 10.  Under the corrected source it does not: the
    # multi-shell branch ends near mu/m0 ~ 0.23 and the region above it holds
    # n_eff = 1 solutions pinned at the lapse floor.  That window would now seed
    # on the pinned family -- exactly the failure the original comment warned
    # about -- and report its endpoint as a fold.
    cand = []
    for m in d["mu_values"]:
        mu = float(m)
        k = f"mu{mu:.2f}_Phi"
        if k not in d.files:
            continue
        P = np.array(d[k])
        if float(np.max(np.abs(P))) <= 0.01:
            continue
        lap = 1.0 + P / csq
        ne = int(np.sum(np.maximum(m0 * (1.0 - lap) - mu, 0.0) > 1e-10))
        mn = float(np.min(lap))
        # multi-shell and clear of the largest floor tested (5e-3) by 20x
        if ne >= 2 and mn > 0.1:
            cand.append((mu, ne, mn))
    if not cand:
        sys.exit("no multi-shell star-branch solution found to seed from")
    cand.sort()
    mu0, ne0, mn0 = cand[-1]
    Phi0 = np.array(d[f"mu{mu0:.2f}_Phi"])
    print(f"seed: mu/m0 = {mu0/m0:.4f}, n_eff = {ne0}, min N = {mn0:.4f}"
          f"   (highest-mu multi-shell solution of {len(cand)} candidates)\n",
          flush=True)

    out = {}
    for floor in [5e-3, 1e-3, 1e-4, 1e-5]:
        t = time.time()
        mu_end, P, track = run(floor, Phi0, mu0)
        out[floor] = (mu_end, track)
        lastN = float(np.min(1.0 + P / csq))
        print(f"  N_floor={floor:.0e}:  branch reaches mu/m0 = {mu_end/m0:.5f}, "
              f"min N = {lastN:.6f}, |lam_1| = {track[-1,3] if track.size else np.nan:.4g}"
              f"   ({time.time()-t:.0f}s, {len(track)} steps)", flush=True)

    fl = np.array(sorted(out))
    mue = np.array([out[f][0] / m0 for f in fl])
    _here = os.path.dirname(os.path.abspath(__file__))
    _out = os.path.normpath(os.path.join(_here, "..", "fermion", "numerical", "data", "fold_continuation_floor.npz"))
    np.savez(_out, N_floor=fl, mu_end_over_m0=mue,
             **{f"track_{i}": out[f][1] for i, f in enumerate(fl)})
    print("\nIf mu_end tracks N_floor the stop is the clamp, not a fold.")


if __name__ == "__main__":
    main()
