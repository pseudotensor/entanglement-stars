#!/usr/bin/env python3
"""Colored (banded) forward-mode Jacobian for the TK closure.

Everything else measured has now failed, and the failures are recorded so they
are not retried:

  seed batching (vmap over seeds)   3x SLOWER  -- jacfwd is already a wide
                                     kernel; a batch waits for its slowest
                                     member while the sequential path stops at
                                     the first seed that converges
  CPU instead of GPU                 Jacobian 856 ms vs 472 ms
  jacrev instead of jacfwd           478.9 ms vs 474.2 ms -- no difference
  batched line search                1.00x.  The 11.4-halvings figure that
                                     motivated it came from a broken probe with
                                     the active set frozen at the initial guess;
                                     in a real solve the line search is cheap.
  Jacobian reuse (chord/Shamanskii)  did not converge, AND 67 s vs 40 s

What survived is structural.  The Jacobian is numerically BANDED, because the
correlations that build it decay exponentially in shell separation:

    bandwidth  12 : max |J| outside = 5.9e-09  (relative)
    bandwidth  20 : max |J| outside = 4.5e-11
    bandwidth  40 : max |J| outside = 4.2e-15   <- machine precision

A banded Jacobian does not need N tangents.  Columns whose row supports do not
overlap can share one forward pass: with bandwidth b, columns j and j' are
independent when |j - j'| > 2b, so the columns split into p = 2b+1 groups by
j mod p, and p forward passes recover every entry exactly.  Within one group
consecutive columns differ by p > 2b, so at most one contributes to any row --
which is what makes the unscrambling exact rather than a least-squares fit.

At N = 200 that is 41 passes instead of 200 for b = 20.  The Jacobian is 51% of
the runtime.

This is safe for a reason worth stating: dropping entries below 1e-11 perturbs
the Newton DIRECTION, never the fixed point, because convergence is judged on
the full fp64 residual.  A wrong direction costs iterations; it cannot move the
answer.  The test below checks the answer anyway.
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
SCRIPT = os.path.join(HERE, "tov_tk_chemical_el.py")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax import jit  # noqa: E402


def load():
    src = open(SCRIPT).read()
    ns = {"__name__": "tk_probe", "__file__": SCRIPT}
    exec(compile(src[: src.index("mu_values = [round(")], SCRIPT, "exec"), ns)
    return ns


def make_banded_jac(resid_fn, N, b):
    """p = 2b+1 forward passes instead of N.  Exact for bandwidth <= b."""
    p = 2 * b + 1
    cols = jnp.arange(N)
    V = (cols[None, :] % p == jnp.arange(p)[:, None]).astype(jnp.float64)  # (p,N)
    rows = jnp.arange(N)
    band = jnp.abs(rows[:, None] - cols[None, :]) <= b       # (N,N)
    pick = (cols % p)[None, :]                               # (1,N) group of col j

    @jit
    def jac(P, mu, mk):
        f = lambda x: resid_fn(x, mu, mk)  # noqa: E731
        JV = jax.vmap(lambda v: jax.jvp(f, (P,), (v,))[1])(V)   # (p,N): JV[c,i]
        # J[i,j] = JV[j % p, i] inside the band, 0 outside
        J = JV[jnp.broadcast_to(pick, (N, N)), jnp.broadcast_to(rows[:, None], (N, N))]
        return jnp.where(band, J, 0.0)

    return jac, p


def main():
    ns = load()
    S, m0, csq, N_floor = ns["S"], ns["m0"], ns["csq"], ns["N_floor"]
    N, n_arr, nunk = S["N"], S["n_arr"], S["nunk"]
    resid_m, jac_full = S["_resid_m"], S["_jac_m"]

    mu = round(0.19100 * m0, 8)
    mj = jnp.float64(mu)
    seed = -3.0 * np.exp(-0.5 * (n_arr / 8.0) ** 2)
    seed[-1] = 0.0
    P0 = jnp.array(seed)
    lap0 = np.maximum(1.0 + np.array(P0) / csq, N_floor)
    mask0 = jnp.array((m0 * (1.0 - lap0) - mu > 0).astype(float))

    def t_of(fn, n=5):
        jax.block_until_ready(fn())
        t = time.perf_counter()
        for _ in range(n):
            jax.block_until_ready(fn())
        return (time.perf_counter() - t) / n

    print("=" * 70)
    print("A.  Accuracy and cost of the colored Jacobian")
    print("=" * 70)
    J_ref = np.array(jac_full(P0, mj, mask0))
    t_ref = t_of(lambda: jac_full(P0, mj, mask0))
    scale = np.abs(J_ref).max()
    print(f"  full jacfwd ({N} tangents)   {t_ref*1e3:8.2f} ms   (reference)")
    cands = []
    for b in (8, 12, 20, 30, 40):
        jb, p = make_banded_jac(resid_m, N, b)
        Jb = np.array(jb(P0, mj, mask0))
        err = np.abs(Jb - J_ref).max() / scale
        tb = t_of(lambda: jb(P0, mj, mask0))
        print(f"  banded b={b:3d} ({p:3d} tangents) {tb*1e3:8.2f} ms   "
              f"speedup {t_ref/tb:5.2f}x   max rel err {err:.2e}")
        cands.append((b, jb, tb, err))

    # ---------------- end-to-end, scored on the answer ---------------------
    print("\n" + "=" * 70)
    print("B.  End-to-end active-set solve (mu/m0 = 0.19100)")
    print("=" * 70)

    def solve(jac_fn, max_outer=25):
        P = np.append(np.asarray(seed, float), 0.0) if len(seed) == nunk \
            else np.asarray(seed, float).copy()
        P = jnp.array(P)
        lap = np.maximum(1.0 + np.array(P) / csq, N_floor)
        mask = (m0 * (1.0 - lap) - mu > 0).astype(float)
        seen = []
        nj = 0
        for _ in range(max_outer):
            mk = jnp.array(mask)
            for _k in range(60):
                F = np.array(resid_m(P, mj, mk))
                rn = float(np.max(np.abs(F)))
                if rn < 1e-10:
                    break
                J = np.array(jac_fn(P, mj, mk)); nj += 1
                try:
                    dP = np.linalg.solve(J, -F)
                except np.linalg.LinAlgError:
                    break
                al = 1.0
                for _ in range(14):
                    Pt = P + jnp.array(al * dP)
                    if float(np.max(np.abs(np.array(resid_m(Pt, mj, mk))))) < rn:
                        P = Pt
                        break
                    al *= 0.5
                else:
                    break
            lap = np.maximum(1.0 + np.array(P) / csq, N_floor)
            Vr = m0 * (1.0 - lap) - mu
            new = (Vr > 0).astype(float)
            if np.array_equal(new, mask):
                on_ok = (bool(np.all(Vr[mask > 0.5] > -1e-12))
                         if mask.any() else False)
                off_ok = bool(np.all(Vr[mask < 0.5] < 1e-12))
                return np.array(P), (rn if (on_ok and off_ok) else 1e30), nj
            key = tuple(new.astype(int))
            if key in seen:
                return np.array(P), 1e30, nj
            seen.append(key)
            mask = new
        return np.array(P), 1e30, nj

    t = time.perf_counter(); Pr, rr, njr = solve(jac_full); tr = time.perf_counter() - t
    print(f"  full jacfwd            {tr:7.2f}s  |R|={rr:.2e}  {njr:3d} Jacobians"
          f"   reference")
    for b, jb, _, _ in cands:
        t = time.perf_counter(); Pb, rb, njb = solve(jb); tb = time.perf_counter() - t
        if rb >= 1e-8:
            print(f"  banded b={b:3d}          {tb:7.2f}s  |R|={rb:.2e}  "
                  f"DID NOT CONVERGE -> unusable")
            continue
        d = float(np.max(np.abs(Pb - Pr)))
        verdict = (f"agrees to {d:.1e}   speedup {tr/tb:5.2f}x" if d < 1e-9
                   else f"WRONG: |dPhi| = {d:.2e}")
        print(f"  banded b={b:3d}          {tb:7.2f}s  |R|={rb:.2e}  {njb:3d} Jacobians"
              f"   {verdict}")


if __name__ == "__main__":
    main()
