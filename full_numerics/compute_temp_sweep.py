#!/usr/bin/env python3
"""
Focused temperature sweep: solve exact two-state closure at varying β₀t₀.

Uses JAX/GPU with exact Jacobian (jacfwd) for robust convergence at low T.
Replaces Phase B2 of compute_exact_data_gpu.py with finer temperature steps,
more Newton iterations, and better continuation.

Reads:  fermion/numerical/data/exact_solutions.npz  (for base-temperature seeds)
Writes: fermion/numerical/data/exact_solutions.npz  (merges new temp sweep keys)

Run:  python3 full_numerics/compute_temp_sweep.py
"""

import os
import sys
import time
import numpy as np

os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, jacfwd, lax

print(f"JAX devices: {jax.devices()}")

# Add parent to path
BASEDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASEDIR)

from full_numerics.physics_twostate import TwoStateShellModel
from full_numerics.solve_twostate import solve_full as solve_full_analytic


# ═══════════════════════════════════════════════════════════════════
# JAX primitives (copied from compute_exact_data_gpu.py)
# ═══════════════════════════════════════════════════════════════════

def build_tridiag(diag, off):
    return jnp.diag(diag) + jnp.diag(off, 1) + jnp.diag(off, -1)


@jit
def bond_correlations(diag, off, beta0):
    H = build_tridiag(diag, off)
    evals, evecs = jnp.linalg.eigh(H)
    be = jnp.clip(beta0 * evals, -500.0, 500.0)
    f = 1.0 / (jnp.exp(be) + 1.0)
    vf = evecs * f[None, :]
    G_diag = jnp.sum(evecs * vf, axis=1)
    G_super = jnp.sum(evecs[:-1] * vf[1:], axis=1)
    return G_diag, G_super


@jit
def bkm_bond_cov(diag, off, beta0):
    H = build_tridiag(diag, off)
    evals, evecs = jnp.linalg.eigh(H)
    be = jnp.clip(beta0 * evals, -500.0, 500.0)
    f = 1.0 / (jnp.exp(be) + 1.0)
    diff = beta0 * (evals[:, None] - evals[None, :])
    phi = jnp.where(jnp.abs(diff) > 1e-12,
                     jnp.expm1(diff) / jnp.where(jnp.abs(diff) > 1e-12, diff, 1.0),
                     1.0)
    K = f[:, None] * (1.0 - f[None, :]) * phi
    U_n = evecs[:-1, :]
    U_n1 = evecs[1:, :]
    Un_sq = U_n**2
    Un1_sq = U_n1**2
    cross = U_n * U_n1
    t1 = jnp.sum((Un1_sq @ K) * Un_sq, axis=1)
    t2 = jnp.sum((Un_sq @ K) * Un1_sq, axis=1)
    t3 = jnp.sum((cross @ K) * cross, axis=1)
    return t1 + t2 - 2.0 * t3


def compute_kms_bg(N, t0, beta0):
    off = -t0 * jnp.ones(N - 1, dtype=jnp.float64)
    diag = jnp.zeros(N, dtype=jnp.float64)
    return bkm_bond_cov(diag, off, beta0)


@jit
def laplacian_action(kappa, Phi):
    N = Phi.shape[0]
    out = jnp.zeros(N, dtype=jnp.float64)
    out = out.at[:-1].add(kappa * (Phi[:-1] - Phi[1:]))
    out = out.at[1:].add(kappa * (Phi[1:] - Phi[:-1]))
    return out


# ═══════════════════════════════════════════════════════════════════
# Residual + Newton solver
# ═══════════════════════════════════════════════════════════════════

def make_residual_fixedpoint(N, t0, V0, n_core, beta0, cstar_sq):
    """JIT'd residual with full EL correction via jax.grad.

    The LHS is grad of the conductance energy E = (1/2) sum kappa_b (dPhi_b)^2,
    giving exact EL correction (no finite differences).
    """
    kms_bg = compute_kms_bg(N, t0, beta0)
    n_arr = jnp.arange(1, N + 1, dtype=jnp.float64)
    g = 4.0 * jnp.pi * n_arr**2
    diag0 = jnp.zeros(N, dtype=jnp.float64)
    core_mask = (jnp.arange(N) < n_core).astype(jnp.float64)

    def _energy(Phi):
        """Conductance energy functional."""
        lapse = 1.0 + Phi / cstar_sq
        Nbar = 0.5 * (lapse[:-1] + lapse[1:])
        # abs(Nbar): Newton-step safeguard; Nbar>0 on positive-lapse branch,
        # so abs() is a no-op for converged solutions.
        off = -t0 * jnp.abs(Nbar)
        cov = bkm_bond_cov(diag0, off, beta0)
        cov_ratio = cov / jnp.maximum(kms_bg, 1e-30)
        kappa = g[:-1] * t0**2 * Nbar**2 * cov_ratio
        dPhi = Phi[:-1] - Phi[1:]
        return 0.5 * jnp.sum(kappa * dPhi**2)

    _grad_energy = jax.grad(_energy)

    @jit
    def residual(Phi):
        # LHS = exact grad of energy (includes EL correction)
        lhs = _grad_energy(Phi)

        # Self-consistent source
        lapse = 1.0 + Phi / cstar_sq
        Nbar = 0.5 * (lapse[:-1] + lapse[1:])
        off = -t0 * jnp.abs(Nbar)

        _, Gs_sig = bond_correlations(diag0, off, beta0)
        rho_sig = jnp.zeros(N, dtype=jnp.float64)
        rho_sig = rho_sig.at[:-1].set(2.0 * t0 * jnp.abs(Nbar) * Gs_sig * g[:-1])

        diag_tgt = diag0 + V0 * core_mask
        Gd_tgt, Gs_tgt = bond_correlations(diag_tgt, off, beta0)
        rho_tgt = jnp.zeros(N, dtype=jnp.float64)
        rho_tgt = rho_tgt.at[:-1].set(2.0 * t0 * jnp.abs(Nbar) * Gs_tgt * g[:-1])
        rho_tgt = rho_tgt.at[:n_core].add(V0 * Gd_tgt[:n_core] * g[:n_core])

        pref = beta0 / cstar_sq
        F = lhs - pref * (rho_sig - rho_tgt)
        F = F.at[N - 1].set(Phi[N - 1])
        return F

    return residual, kms_bg


def make_newton_solver(residual_fn, N, max_iter=100, tol=1e-8):
    """JIT-compiled Newton solver with jacfwd and line search."""

    @jit
    def solve(Phi_init):
        jac_fn = jacfwd(residual_fn)

        def cond(state):
            _Phi, _res_norm, it, converged = state
            return (~converged) & (it < max_iter)

        def body(state):
            Phi, _res_norm, it, _converged = state
            F = residual_fn(Phi)
            norm_F = jnp.max(jnp.abs(F))
            conv_now = norm_F < tol

            J = jac_fn(Phi)
            reg = 1e-10 * jnp.maximum(jnp.abs(jnp.diag(J)), 1e-10)
            J_reg = J + jnp.diag(reg)
            dPhi = jnp.linalg.solve(J_reg, -F)

            # Line search
            def ls_cond(ls_state):
                alpha, _Phi_best, _improved, _norm_best = ls_state
                return alpha > 1.0 / 64.0

            def ls_body(ls_state):
                alpha, Phi_best, improved, norm_best = ls_state
                Phi_trial = Phi + alpha * dPhi
                F_trial = residual_fn(Phi_trial)
                norm_trial = jnp.max(jnp.abs(F_trial))
                better = norm_trial < norm_best
                Phi_best = jnp.where(better, Phi_trial, Phi_best)
                norm_best = jnp.where(better, norm_trial, norm_best)
                improved = improved | better
                alpha = jnp.where(better, 1.0 / 128.0, alpha * 0.5)
                return (alpha, Phi_best, improved, norm_best)

            ls_init = (1.0, Phi, jnp.bool_(False), norm_F)
            _, Phi_new, improved, norm_new = lax.while_loop(
                ls_cond, ls_body, ls_init)

            Phi_out = jnp.where(conv_now, Phi,
                        jnp.where(improved, Phi_new, Phi + 0.01 * dPhi))
            norm_out = jnp.where(conv_now, norm_F,
                        jnp.where(improved, norm_new, norm_F))
            return (Phi_out, norm_out, it + 1, conv_now)

        F0 = residual_fn(Phi_init)
        norm0 = jnp.max(jnp.abs(F0))
        init_state = (Phi_init, norm0, jnp.int32(0), jnp.bool_(False))
        Phi_final, final_res, iters, converged = lax.while_loop(
            cond, body, init_state)
        return Phi_final, converged, iters, final_res

    return solve


def extract_rs_np(Phi, N, n_core, cstar_sq, a=1.0):
    Phi = np.asarray(Phi)
    r = a * np.arange(1, N + 1, dtype=float)
    R = r[-1]
    i_lo = n_core + 3
    i_hi = min(N - 5, N // 2)
    if i_lo >= i_hi:
        return 0.0
    if not np.any(np.abs(Phi[i_lo:i_hi]) > 1e-15):
        return 0.0
    r_slice = r[i_lo:i_hi]
    GM_est = -Phi[i_lo:i_hi] * r_slice * R / (R - r_slice)
    GM = np.median(GM_est)
    return 2.0 * GM / cstar_sq


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    N = 200
    t0 = 1.0
    n_core = 5
    cstar_sq = 0.5
    a = 1.0
    r_arr = a * np.arange(1, N + 1)

    # Temperature grid: uniform 0.1 steps, fine 0.05 near the hard region
    bt0_coarse = np.arange(0.1, 1.51, 0.1)
    bt0_fine = np.arange(1.5, 3.01, 0.05)
    bt0_values = np.unique(np.round(np.concatenate([bt0_coarse, bt0_fine]), 4))

    V0_temp_list = [0.001, 0.005]

    print(f"Temperature sweep: {len(bt0_values)} points, "
          f"bt0 = {bt0_values[0]:.2f} to {bt0_values[-1]:.2f}")
    print(f"V0 values: {V0_temp_list}")

    # Load existing data for seeds
    DATADIR = os.path.join(BASEDIR, "fermion", "numerical", "data")
    npzpath = os.path.join(DATADIR, "exact_solutions.npz")
    print(f"Loading seeds from: {npzpath}")
    D = np.load(npzpath, allow_pickle=True)

    # Load base-temperature seeds (bt0=0.1, from Phase A)
    base_beta0 = float(D["beta0"])

    temp_sweep = {}

    for V0 in V0_temp_list:
        print(f"\n{'='*60}")
        print(f"V0 = {V0}  temperature sweep (GPU, jacfwd)")
        print(f"{'='*60}")

        # Get seed from existing solution at base temperature
        tag_base = f"sol_{V0:.4f}"
        if f"{tag_base}_Phi" in D.files:
            Phi_cont = np.array(D[f"{tag_base}_Phi"], dtype=np.float64)
            print(f"  Seed from existing solution at beta0={base_beta0}")
        else:
            # Solve fresh at base temperature
            print(f"  Solving fresh seed at beta0={base_beta0}...")
            resid_fn, _ = make_residual_fixedpoint(
                N, t0, V0, n_core, base_beta0, cstar_sq)
            model_an = TwoStateShellModel(
                N=N, t0=t0, V0=V0, n_core=n_core,
                beta0=base_beta0, cstar_sq=cstar_sq, mode="analytic")
            Phi_an, _, _ = solve_full_analytic(model_an, tol=1e-12, verbose=False)
            solver = make_newton_solver(resid_fn, N, max_iter=100, tol=1e-8)
            Phi_j, conv_j, nit_j, res_j = solver(jnp.array(Phi_an))
            Phi_cont = np.array(jax.block_until_ready(Phi_j))
            print(f"    conv={bool(conv_j)}, |F|={float(res_j):.2e}")

        results = []
        n_consecutive_fail = 0

        for i_bt, bt0 in enumerate(bt0_values):
            this_beta0 = bt0 / t0
            t_start = time.time()

            # Build residual + solver at this temperature
            resid_fn, _ = make_residual_fixedpoint(
                N, t0, V0, n_core, this_beta0, cstar_sq)
            solver = make_newton_solver(resid_fn, N, max_iter=100, tol=1e-8)

            # Solve from continuation seed
            Phi_j, conv_j, nit_j, res_j = solver(jnp.array(Phi_cont))
            Phi_ex = np.array(jax.block_until_ready(Phi_j))
            F_res = float(res_j)

            # If failed, try fresh analytic seed
            if F_res > 1e-6:
                model_an = TwoStateShellModel(
                    N=N, t0=t0, V0=V0, n_core=n_core,
                    beta0=this_beta0, cstar_sq=cstar_sq, mode="analytic")
                try:
                    Phi_an, _, _ = solve_full_analytic(
                        model_an, Phi_cont, tol=1e-12, verbose=False)
                    Phi_j2, conv_j2, nit_j2, res_j2 = solver(jnp.array(Phi_an))
                    Phi_ex2 = np.array(jax.block_until_ready(Phi_j2))
                    F_res2 = float(res_j2)
                    if F_res2 < F_res:
                        Phi_ex, F_res = Phi_ex2, F_res2
                        conv_j, nit_j = conv_j2, nit_j2
                except Exception:
                    pass

            lapse = 1.0 + Phi_ex / cstar_sq
            minN = float(lapse.min())
            rs = extract_rs_np(Phi_ex, N, n_core, cstar_sq)

            # Update continuation seed
            if F_res < 1e-6:
                Phi_cont = Phi_ex.copy()
                n_consecutive_fail = 0
            else:
                n_consecutive_fail += 1

            # Proxy at this temperature (analytic)
            model_an_p = TwoStateShellModel(
                N=N, t0=t0, V0=V0, n_core=n_core,
                beta0=this_beta0, cstar_sq=cstar_sq, mode="analytic")
            try:
                from full_numerics.solve_twostate import solve_proxy as _solve_proxy
                Phi_proxy, _ = _solve_proxy(model_an_p, Phi_cont, tol=1e-8)
                rs_proxy = extract_rs_np(Phi_proxy, N, n_core, cstar_sq)
            except Exception:
                rs_proxy = 0.0

            dt = time.time() - t_start
            status = "OK" if F_res < 1e-6 else "FAIL"
            print(f"  bt0={bt0:5.2f}: rs={rs:7.4f}  min(N)={minN:7.4f}  "
                  f"|F|={F_res:.2e}  nit={int(nit_j):3d}  "
                  f"proxy_rs={rs_proxy:.4f}  {status}  [{dt:.1f}s]")

            results.append({
                "bt0": bt0, "rs": rs, "conv": F_res < 1e-6,
                "F": F_res, "minN": minN, "Phi": Phi_ex.copy(),
                "rs_proxy": rs_proxy,
            })

        temp_sweep[V0] = results

    # ── Merge into existing npz ────────────────────────────────────
    print(f"\nMerging results into {npzpath}...")
    data = dict(D)

    data["temp_V0s"] = np.array(V0_temp_list)
    data["temp_bt0"] = bt0_values

    for V0 in V0_temp_list:
        tag = f"temp_{V0:.4f}"
        sw = temp_sweep[V0]
        data[f"{tag}_rs"] = np.array([e["rs"] for e in sw])
        data[f"{tag}_conv"] = np.array([e["conv"] for e in sw])
        data[f"{tag}_F"] = np.array([e["F"] for e in sw])
        data[f"{tag}_minN"] = np.array([e["minN"] for e in sw])
        data[f"{tag}_Phi"] = np.array([e["Phi"] for e in sw])
        data[f"{tag}_proxy_rs"] = np.array([e["rs_proxy"] for e in sw])

    np.savez_compressed(npzpath, **data)
    print(f"Saved: {npzpath}")

    # ── Summary ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for V0 in V0_temp_list:
        sw = temp_sweep[V0]
        n_ok = sum(1 for e in sw if e["conv"])
        n_tot = len(sw)
        print(f"  V0={V0}: {n_ok}/{n_tot} converged")
        if n_ok < n_tot:
            fails = [e["bt0"] for e in sw if not e["conv"]]
            print(f"    Failed at bt0: {fails}")

    print("\nDone.")


if __name__ == "__main__":
    main()
