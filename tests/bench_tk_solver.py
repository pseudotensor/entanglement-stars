#!/usr/bin/env python3
"""Full accounting of TK solver cost, and a measured test of each candidate fix.

Two hypotheses have already died by measurement and are recorded here so they
are not retried:

  - "batch the seed sweep with vmap"  -> 3x SLOWER.  jacfwd already pushes N
    tangents through eigh, so the Jacobian is already a wide kernel; there was
    no launch-overhead headroom, and a batch runs in lockstep to its slowest
    member while the sequential path stops at the first seed that converges.
  - "the small matrices belong on the CPU" -> also slower.  Jacobian 472 ms on
    GPU against 856 ms on CPU (and 894 ms with 8 threads).

Baseline profile, mu/m0 = 0.19100, N = 200, one A6000:
    jacobian jacfwd        472.5 ms   51%
    line-search residuals  419.2 ms   45%   (11.4 halvings mean, alpha=1 in 1/25)
    residual                32.2 ms    4%
    transfers + CPU solve    0.6 ms  ~0%    <- NOT the bottleneck

So the cost is (a) the Jacobian and (b) a line search that almost never accepts
a full step.  This script measures, for each candidate, both the speed AND the
answer, because a faster solver that lands somewhere else is worthless.
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
from jax import jit, jacfwd, jacrev  # noqa: E402


def load():
    src = open(SCRIPT).read()
    ns = {"__name__": "tk_probe", "__file__": SCRIPT}
    exec(compile(src[: src.index("mu_values = [round(")], SCRIPT, "exec"), ns)
    return ns


def t_of(fn, n=10):
    jax.block_until_ready(fn())
    t = time.perf_counter()
    for _ in range(n):
        jax.block_until_ready(fn())
    return (time.perf_counter() - t) / n


def section(s):
    print("\n" + "=" * 68 + f"\n{s}\n" + "=" * 68, flush=True)


def main():
    ns = load()
    S, m0, csq, N_floor = ns["S"], ns["m0"], ns["csq"], ns["N_floor"]
    beta0 = ns["beta0"]
    N = S["N"]
    n_arr = S["n_arr"]
    resid_m, jac_m = S["_resid_m"], S["_jac_m"]

    mu = round(0.19100 * m0, 8)
    mj = jnp.float64(mu)
    seed = -3.0 * np.exp(-0.5 * (n_arr / 8.0) ** 2)
    seed[-1] = 0.0
    P0 = jnp.array(seed, dtype=jnp.float64)
    lap = np.maximum(1.0 + np.array(P0) / csq, N_floor)
    mask = jnp.array((m0 * (1.0 - lap) - mu > 0).astype(float))

    print(f"backend={jax.default_backend()} devices={jax.devices()} N={N} "
          f"x64={jax.config.jax_enable_x64}")

    # ---------------------------------------------------------------- A. eigh
    section("A.  Is eigh itself the cost?  (fp64 vs fp32, dense vs structure)")
    d = jnp.array(np.random.RandomState(0).randn(N))
    o = jnp.array(np.random.RandomState(1).randn(N - 1))
    H64 = jnp.diag(d) + jnp.diag(o, 1) + jnp.diag(o, -1)
    H32 = H64.astype(jnp.float32)
    eigh64 = jit(jnp.linalg.eigh)
    eigh32 = jit(lambda A: jnp.linalg.eigh(A))
    t64 = t_of(lambda: eigh64(H64), 20)
    t32 = t_of(lambda: eigh32(H32), 20)
    H64_np = np.array(H64)
    tnp = t_of(lambda: np.linalg.eigh(H64_np), 5)
    print(f"  eigh {N}x{N} fp64 GPU     {t64*1e3:8.3f} ms")
    print(f"  eigh {N}x{N} fp32 GPU     {t32*1e3:8.3f} ms   ({t64/t32:.1f}x faster)")
    print(f"  eigh {N}x{N} fp64 numpy   {tnp*1e3:8.3f} ms")
    print("  (A6000 runs fp64 at 1/32 the fp32 rate; that ratio is the thing to"
          " watch)")

    # ------------------------------------------------- B. residual breakdown
    section("B.  Where does the 32 ms residual go?")
    t_res = t_of(lambda: resid_m(P0, mj, mask), 20)
    print(f"  full residual            {t_res*1e3:8.3f} ms")

    # ------------------------------------- C. can the line search be batched?
    section("C.  Line search: 11.4 halvings mean.  Batch the alphas?")
    resid_v = jit(jax.vmap(lambda P: ns["S"]["_resid_m"](P, mj, mask)))
    for B in (1, 4, 8, 14, 20):
        Pb = jnp.tile(P0, (B, 1))
        tb = t_of(lambda: resid_v(Pb), 10)
        print(f"  batched residual  B={B:3d}  {tb*1e3:8.3f} ms   "
              f"= {tb/t_res:5.2f}x one   -> per-alpha {tb/B*1e3:6.3f} ms")
    print("  If B=14 costs much less than 14x, the line search collapses to one"
          " call.")

    # ------------------------------------------------ D. Jacobian: cheaper?
    section("D.  Jacobian: jacfwd vs jacrev, and fp32")
    t_jf = t_of(lambda: jac_m(P0, mj, mask), 5)
    print(f"  jacfwd fp64              {t_jf*1e3:8.3f} ms")
    try:
        jr = jit(jacrev(lambda P, mu_, mk: ns["S"]["_resid_m"](P, mu_, mk),
                        argnums=0))
        t_jr = t_of(lambda: jr(P0, mj, mask), 5)
        print(f"  jacrev fp64              {t_jr*1e3:8.3f} ms   "
              f"({t_jf/t_jr:.2f}x vs jacfwd)")
    except Exception as e:  # noqa: BLE001
        print(f"  jacrev failed: {type(e).__name__}: {str(e)[:90]}")

    # ------------------------------------- E. how stale can the Jacobian be?
    section("E.  Jacobian reuse (chord/Shamanskii): does a stale J still work?")
    print("  Newton judges convergence on the fp64 RESIDUAL, so an approximate")
    print("  J changes only the rate, never the fixed point.  Measured below:")
    for reuse in (1, 3, 5, 10):
        P = P0
        nJ = nIt = 0
        J = None
        t = time.perf_counter()
        ok = False
        for it in range(120):
            F = np.array(resid_m(P, mj, mask))
            rn = float(np.max(np.abs(F)))
            if rn < 1e-10:
                ok = True
                break
            if J is None or it % reuse == 0:
                J = np.array(jac_m(P, mj, mask))
                nJ += 1
            try:
                dP = np.linalg.solve(J, -F)
            except np.linalg.LinAlgError:
                break
            al = 1.0
            for _ in range(14):
                Pt = P + jnp.array(al * dP)
                if float(np.max(np.abs(np.array(resid_m(Pt, mj, mask))))) < rn:
                    P = Pt
                    break
                al *= 0.5
            else:
                break
            nIt += 1
        el = time.perf_counter() - t
        print(f"  reuse={reuse:3d}: {'converged' if ok else 'FAILED   '}  "
              f"{nIt:3d} steps, {nJ:3d} Jacobians, {el:7.2f}s"
              + (f"  minPhi={float(jnp.min(P)):+.6f}" if ok else ""))


if __name__ == "__main__":
    main()
