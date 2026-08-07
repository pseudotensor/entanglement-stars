#!/usr/bin/env python3
"""Where does the TK Newton iteration actually spend its time?

Written after a batching attempt made things 3x SLOWER because the bottleneck
was guessed rather than measured.  So: measure, one component at a time, with
explicit block_until_ready so no cost hides behind JAX's async dispatch.

Reports, per Newton iteration of _newton_frozen:
  - residual evaluation (GPU)
  - Jacobian evaluation, jacfwd with N tangents (GPU)
  - np.linalg.solve of the NxN system (CPU, and OMP is pinned to 1 thread)
  - device->host transfers, which _newton_frozen does ~16x per iteration
  - how many line-search halvings are actually taken

and then the share of a real solve each accounts for.
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


def load():
    src = open(SCRIPT).read()
    ns = {"__name__": "tk_probe", "__file__": SCRIPT}
    exec(compile(src[: src.index("mu_values = [round(")], SCRIPT, "exec"), ns)
    return ns


def timeit(fn, n=20):
    fn()  # warm
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t) / n


def main():
    ns = load()
    S, m0, csq = ns["S"], ns["m0"], ns["csq"]
    N_floor = ns["N_floor"]
    N = S["N"]
    n_arr = S["n_arr"]
    print(f"\nbackend={jax.default_backend()}  devices={jax.devices()}  N={N}")
    print(f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")

    mu = round(0.19100 * m0, 8)
    seed = -3.0 * np.exp(-0.5 * (n_arr / 8.0) ** 2)
    seed[-1] = 0.0
    P = jnp.array(seed, dtype=jnp.float64)
    lap = np.maximum(1.0 + np.array(P) / csq, N_floor)
    mask = jnp.array((m0 * (1.0 - lap) - mu > 0).astype(float))
    mj = jnp.float64(mu)

    resid_m, jac_m = S["_resid_m"], S["_jac_m"]

    # --- component timings, each forced to completion -----------------------
    t_res = timeit(lambda: jax.block_until_ready(resid_m(P, mj, mask)))
    t_jac = timeit(lambda: jax.block_until_ready(jac_m(P, mj, mask)))

    J_dev = jax.block_until_ready(jac_m(P, mj, mask))
    F_dev = jax.block_until_ready(resid_m(P, mj, mask))
    J_np, F_np = np.array(J_dev), np.array(F_dev)

    t_pull_J = timeit(lambda: np.array(J_dev))          # device->host, N*N
    t_pull_F = timeit(lambda: np.array(F_dev))          # device->host, N
    t_cpu_solve = timeit(lambda: np.linalg.solve(J_np, -F_np))
    t_gpu_solve = timeit(lambda: jax.block_until_ready(
        jnp.linalg.solve(J_dev, -F_dev)))
    t_push = timeit(lambda: jax.block_until_ready(jnp.array(F_np)))

    print("\n--- per-call cost (ms) ---")
    for k, v in (("residual (GPU)", t_res), ("jacobian jacfwd (GPU)", t_jac),
                 ("pull J to host (NxN)", t_pull_J), ("pull F to host (N)", t_pull_F),
                 ("solve on CPU (numpy)", t_cpu_solve),
                 ("solve on GPU (jnp)", t_gpu_solve),
                 ("push vector to device", t_push)):
        print(f"  {k:<26s} {v*1e3:9.3f}")

    # --- how many halvings does the line search really take? ----------------
    halvings = []
    Pw = P
    for _ in range(25):
        F = np.array(resid_m(Pw, mj, mask))
        rn = float(np.max(np.abs(F)))
        if rn < 1e-10:
            break
        J = np.array(jac_m(Pw, mj, mask))
        try:
            dP = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            break
        al, k = 1.0, 0
        for k in range(14):
            Pt = Pw + jnp.array(al * dP)
            if float(np.max(np.abs(np.array(resid_m(Pt, mj, mask))))) < rn:
                Pw = Pt
                break
            al *= 0.5
        halvings.append(k)
    h = np.array(halvings) if halvings else np.array([0])
    print(f"\n--- line search over {len(h)} Newton steps ---")
    print(f"  halvings: mean {h.mean():.2f}  max {h.max()}  "
          f"accepted at alpha=1 in {(h == 0).sum()}/{len(h)} steps")

    # --- modelled iteration cost -------------------------------------------
    n_res_ls = h.mean() + 1
    per_iter = (t_res + t_jac + t_pull_J + t_pull_F + t_cpu_solve
                + n_res_ls * (t_push + t_res + t_pull_F))
    print("\n--- modelled cost of one Newton iteration (ms) ---")
    parts = {
        "jacobian": t_jac,
        "residual (convergence check)": t_res,
        "line-search residuals": n_res_ls * (t_push + t_res + t_pull_F),
        "device->host pulls": t_pull_J + t_pull_F,
        "CPU linear solve": t_cpu_solve,
    }
    for k, v in sorted(parts.items(), key=lambda x: -x[1]):
        print(f"  {k:<32s} {v*1e3:8.3f}   {100*v/per_iter:5.1f}%")
    print(f"  {'TOTAL':<32s} {per_iter*1e3:8.3f}")


if __name__ == "__main__":
    main()
