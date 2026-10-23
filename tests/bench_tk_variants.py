#!/usr/bin/env python3
"""End-to-end TK solver variants: speed AND identical answer, or it doesn't count.

Round 1 (tests/bench_tk_solver.py) established, on one A6000 at N=200:
    eigh 200x200 fp64 GPU    10.7 ms   <- the residual is 3 of these and nothing else
    residual                 33.6 ms
    jacfwd fp64             474.2 ms   51% of an iteration
    line search             419.2 ms   45%, at 11.4 halvings mean
    batched residual B=14    86.8 ms   i.e. 5.4x cheaper than 14 sequential
    jacrev                  478.9 ms   no better than jacfwd

and killed two hypotheses by measurement: batching the SEED sweep (3x slower)
and moving to CPU (Jacobian 856 ms vs 472 ms).

Round 1's Jacobian-reuse test was invalid -- it froze the active set at the
initial guess, so the frozen problem had no solution and even the reuse=1
baseline "failed".  Redone here inside the real outer active-set loop.

This script measures:
  0. the Jacobian's off-diagonal decay.  The correlations decay exponentially,
     so if J is numerically banded, forward mode needs only 2b+1 structured
     tangents instead of N -- a far bigger prize than anything else here.
  1. baseline active-set solve (reference answer + time)
  2. batched line search
  3. Jacobian reuse, properly nested in the outer loop
  4. 2+3 together
Every variant is scored against the baseline Phi.  A variant that is faster and
lands anywhere else is reported as WRONG, not as a speedup.
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

ALPHAS = np.array([0.5 ** k for k in range(14)])


def load():
    src = open(SCRIPT).read()
    ns = {"__name__": "tk_probe", "__file__": SCRIPT}
    exec(compile(src[: src.index("mu_values = [round(")], SCRIPT, "exec"), ns)
    return ns


def main():
    ns = load()
    S, m0, csq, N_floor = ns["S"], ns["m0"], ns["csq"], ns["N_floor"]
    N, n_arr = S["N"], S["n_arr"]
    resid_m, jac_m = S["_resid_m"], S["_jac_m"]
    nunk = S["nunk"]

    mu = round(0.19100 * m0, 8)
    mj = jnp.float64(mu)
    seed = -3.0 * np.exp(-0.5 * (n_arr / 8.0) ** 2)
    seed[-1] = 0.0

    # ---- 0. is the Jacobian numerically banded? ---------------------------
    print("=" * 68)
    print("0.  Jacobian off-diagonal decay (is J effectively banded?)")
    print("=" * 68)
    P0 = jnp.array(seed)
    lap0 = np.maximum(1.0 + np.array(P0) / csq, N_floor)
    mask0 = jnp.array((m0 * (1.0 - lap0) - mu > 0).astype(float))
    J = np.array(jac_m(P0, mj, mask0))
    scale = np.abs(J).max()
    idx = np.abs(np.subtract.outer(np.arange(N), np.arange(N)))
    print(f"  max|J| = {scale:.3e}")
    for b in (1, 2, 3, 5, 8, 12, 20, 40, 80):
        out = np.abs(J[idx > b])
        print(f"    bandwidth {b:3d}: max |J| outside = {out.max()/scale:.3e} "
              f"(relative)")
    print("  A colored forward pass needs 2b+1 tangents where this is negligible;")
    print(f"  at N={N} that is the difference between {N} tangents and a handful.")

    # ---- shared pieces ----------------------------------------------------
    resid_v = jit(jax.vmap(lambda Pb, mu_, mk: resid_m(Pb, mu_, mk),
                           in_axes=(0, None, None)))

    def newton_frozen(P, mask, batched_ls, reuse, tol=1e-10, it=60):
        mk = jnp.array(mask)
        J_cache = None
        for k in range(it):
            F = np.array(resid_m(P, mj, mk))
            rn = float(np.max(np.abs(F)))
            if rn < tol:
                return P, rn
            if J_cache is None or k % reuse == 0:
                J_cache = np.array(jac_m(P, mj, mk))
            try:
                dP = np.linalg.solve(J_cache, -F)
            except np.linalg.LinAlgError:
                return P, rn
            if batched_ls:
                trials = jnp.array(np.array(P)[None, :] + ALPHAS[:, None] * dP)
                rt = np.array(jnp.max(jnp.abs(resid_v(trials, mj, mk)), axis=1))
                good = np.where(rt < rn)[0]
                if good.size == 0:
                    return P, rn
                P = trials[good[0]]          # first improving alpha: same rule
            else:
                al = 1.0
                for _ in range(14):
                    Pt = P + jnp.array(al * dP)
                    if float(np.max(np.abs(np.array(resid_m(Pt, mj, mk))))) < rn:
                        P = Pt
                        break
                    al *= 0.5
                else:
                    return P, rn
        return P, float(np.max(np.abs(np.array(resid_m(P, mj, mk)))))

    def active_set(seed_v, batched_ls=False, reuse=1, max_outer=25):
        P = np.asarray(seed_v, dtype=float).copy()
        if P.shape[0] == nunk:
            P = np.append(P, 0.0)
        P = jnp.array(P)
        lap = np.maximum(1.0 + np.array(P) / csq, N_floor)
        mask = (m0 * (1.0 - lap) - mu > 0).astype(float)
        seen = []
        for _ in range(max_outer):
            P, rn = newton_frozen(P, mask, batched_ls, reuse)
            lap = np.maximum(1.0 + np.array(P) / csq, N_floor)
            Vr = m0 * (1.0 - lap) - mu
            new = (Vr > 0).astype(float)
            if np.array_equal(new, mask):
                on_ok = (bool(np.all(Vr[mask > 0.5] > -1e-12))
                         if mask.any() else False)
                off_ok = bool(np.all(Vr[mask < 0.5] < 1e-12))
                return np.array(P), (rn if (on_ok and off_ok) else 1e30)
            key = tuple(new.astype(int))
            if key in seen:
                return np.array(P), 1e30
            seen.append(key)
            mask = new
        return np.array(P), 1e30

    print("\n" + "=" * 68)
    print("1-4.  End-to-end variants (mu/m0 = 0.19100), scored on the answer")
    print("=" * 68)
    variants = [
        ("baseline",                  dict(batched_ls=False, reuse=1)),
        ("batched line search",       dict(batched_ls=True,  reuse=1)),
        ("J reuse 3",                 dict(batched_ls=False, reuse=3)),
        ("J reuse 5",                 dict(batched_ls=False, reuse=5)),
        ("batched LS + J reuse 3",    dict(batched_ls=True,  reuse=3)),
        ("batched LS + J reuse 5",    dict(batched_ls=True,  reuse=5)),
    ]
    ref_P = ref_t = None
    for name, kw in variants:
        t = time.perf_counter()
        P, rn = active_set(seed, **kw)
        el = time.perf_counter() - t
        conv = rn < 1e-8
        if ref_P is None:
            ref_P, ref_t = P, el
            note = "reference"
        else:
            d = float(np.max(np.abs(P - ref_P))) if conv else float("nan")
            if not conv:
                note = "DID NOT CONVERGE -> unusable"
            elif d > 1e-9:
                note = f"WRONG: |dPhi| = {d:.2e} vs baseline"
            else:
                note = f"agrees to {d:.1e}   speedup {ref_t/el:.2f}x"
        print(f"  {name:<26s} {el:7.2f}s  |R|={rn:.2e}  {note}")


if __name__ == "__main__":
    main()
