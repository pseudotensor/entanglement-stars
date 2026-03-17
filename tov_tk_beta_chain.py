#!/usr/bin/env python3
"""Chain beta continuation: bt5→bt10→bt12→bt15, finding fold μ* at each.

Uses bt10 solutions (from tov_tk_bt10_mu_scan.npz) as seeds for bt12,
then bt12 solutions for bt15. Much faster than seeding from bt5 each time.
"""
import os, time
import numpy as np

os.environ["JAX_PLATFORM_NAME"] = "gpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, jacfwd

print(f"JAX devices: {jax.devices()}", flush=True)

m0 = 3.0; csq = 5.0; N = 200; t0 = 1.0; N_floor = 0.005
n_arr = jnp.arange(1, N+1, dtype=jnp.float64)
g = 4.0 * jnp.pi * n_arr**2
diag0 = jnp.zeros(N, dtype=jnp.float64)

def _bkm_cov(diag_vals, off_vals, beta0):
    H = jnp.diag(diag_vals) + jnp.diag(off_vals, 1) + jnp.diag(off_vals, -1)
    evals, evecs = jnp.linalg.eigh(H)
    be = jnp.clip(beta0 * evals, -500.0, 500.0)
    f = 1.0 / (jnp.exp(be) + 1.0)
    diff = beta0 * (evals[:, None] - evals[None, :])
    phi = jnp.where(jnp.abs(diff) > 1e-12,
                     jnp.expm1(diff) / jnp.where(jnp.abs(diff) > 1e-12, diff, 1.0), 1.0)
    K = f[:, None] * (1.0 - f[None, :]) * phi
    U_n = evecs[:-1, :]; U_n1 = evecs[1:, :]
    return (jnp.sum((U_n1**2) @ K * (U_n**2), axis=1) +
            jnp.sum((U_n**2) @ K * (U_n1**2), axis=1) -
            2.0 * jnp.sum((U_n * U_n1) @ K * (U_n * U_n1), axis=1))

def _bond_correlations(diag_vals, off_vals, beta0):
    H = jnp.diag(diag_vals) + jnp.diag(off_vals, 1) + jnp.diag(off_vals, -1)
    evals, evecs = jnp.linalg.eigh(H)
    be = jnp.clip(beta0 * evals, -500.0, 500.0)
    f = 1.0 / (jnp.exp(be) + 1.0)
    vf = evecs * f[None, :]
    return jnp.sum(evecs * vf, axis=1), jnp.sum(evecs[:-1] * vf[1:], axis=1)

def _energy(Phi, mu, beta0):
    off_bg = -t0 * jnp.ones(N-1, dtype=jnp.float64)
    cov_bg = jnp.maximum(_bkm_cov(diag0, off_bg, beta0), 1e-30)
    lapse = jnp.maximum(1.0 + Phi / csq, N_floor)
    Nbar = 0.5 * (lapse[:-1] + lapse[1:])
    off = -t0 * jnp.abs(Nbar)
    kappa = g[:-1] * t0**2 * Nbar**2 * (_bkm_cov(diag0, off, beta0) / cov_bg)
    dPhi = Phi[:-1] - Phi[1:]
    return 0.5 * jnp.sum(kappa * dPhi**2)

def _residual(Phi, mu, beta0):
    lhs = jax.grad(_energy, argnums=0)(Phi, mu, beta0)
    lapse = jnp.maximum(1.0 + Phi / csq, N_floor)
    Nbar = 0.5 * (lapse[:-1] + lapse[1:])
    Nbar_abs = jnp.abs(Nbar)
    off = -t0 * Nbar_abs
    V_defect = jnp.maximum(m0 * (1.0 - lapse) - mu, 0.0)
    Gd_s, Gs_s = _bond_correlations(diag0, off, beta0)
    rho_s = jnp.zeros(N, dtype=jnp.float64).at[:-1].add(2.0 * t0 * Nbar_abs * Gs_s * g[:-1])
    Gd_t, Gs_t = _bond_correlations(V_defect, off, beta0)
    rho_t = (jnp.zeros(N, dtype=jnp.float64).at[:-1].add(2.0 * t0 * Nbar_abs * Gs_t * g[:-1])
             + V_defect * Gd_t * g)
    F = lhs - (beta0 / csq) * (rho_s - rho_t)
    return F.at[N - 1].set(Phi[N - 1])

print("JIT compiling...", flush=True)
t_jit = time.time()
resid_jit = jit(_residual)
jac_jit = jit(jacfwd(_residual, argnums=0))
_d = jnp.zeros(N); _m = jnp.float64(0.1); _b = jnp.float64(5.0)
jax.block_until_ready(resid_jit(_d, _m, _b))
jax.block_until_ready(jac_jit(_d, _m, _b))
print(f"JIT done in {time.time() - t_jit:.1f}s", flush=True)


def newton_solve(Phi0, mu, beta0, max_iter=80, tol=1e-8):
    Phi = jnp.array(Phi0, dtype=jnp.float64)
    mu_j = jnp.float64(mu); b_j = jnp.float64(beta0)
    for it in range(max_iter):
        F = np.array(jax.block_until_ready(resid_jit(Phi, mu_j, b_j)))
        norm = float(np.max(np.abs(F)))
        if norm < tol:
            return np.array(Phi), norm, True, it
        J = np.array(jax.block_until_ready(jac_jit(Phi, mu_j, b_j)))
        try:
            dPhi = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            return np.array(Phi), norm, False, it
        alpha = 1.0
        for _ in range(12):
            Phi_trial = Phi + alpha * jnp.array(dPhi)
            F_trial = np.array(jax.block_until_ready(resid_jit(Phi_trial, mu_j, b_j)))
            if np.max(np.abs(F_trial)) < norm:
                Phi = Phi_trial; break
            alpha *= 0.5
        else:
            Phi = Phi + alpha * jnp.array(dPhi)
    F = np.array(jax.block_until_ready(resid_jit(Phi, mu_j, b_j)))
    norm = float(np.max(np.abs(F)))
    return np.array(Phi), norm, norm < tol, max_iter


def get_props(Phi, mu):
    lapse = np.maximum(1.0 + Phi / csq, N_floor)
    minN = float(np.min(lapse))
    V = np.maximum(m0 * (1.0 - lapse) - mu, 0.0)
    ne = int(np.sum(V > 1e-10))
    return ne, minN


def compute_eigenvalue(Phi, mu, beta0):
    J = np.array(jax.block_until_ready(jac_jit(jnp.array(Phi), jnp.float64(mu), jnp.float64(beta0))))
    J_int = J[:-1, :-1]
    eigs = np.sort(np.abs(np.linalg.eigvals(J_int)))
    return float(eigs[0])


def seed_and_scan(seeds, beta_target, label):
    """Seed solutions at beta_target from seeds dict, then μ-continue to fold.

    seeds: dict mu -> (Phi, n_eff, minN) at a nearby beta
    Returns: dict mu -> (Phi, n_eff, minN) at beta_target, plus fold info
    """
    print(f"\n{'='*60}")
    print(f"β₀t₀ = {beta_target} ({label})")
    print(f"{'='*60}")

    seed_mus = sorted(seeds.keys())
    solutions = {}

    # Step 1: Direct Newton seeding (only try 3 scales for speed)
    print(f"Seeding from {len(seed_mus)} solutions...", flush=True)
    for mu_val in seed_mus:
        Phi_seed, ne_seed, _ = seeds[mu_val]
        best = None
        for scale in [1.0, 0.95, 0.9]:
            Phi_sol, norm, ok, iters = newton_solve(Phi_seed * scale, mu_val, beta_target)
            if ok and np.max(np.abs(Phi_sol)) > 0.01:
                ne, minN = get_props(np.array(Phi_sol), mu_val)
                if minN > 0.01:
                    best = (np.array(Phi_sol), ne, minN)
                    break
        if best:
            solutions[mu_val] = best

    print(f"  Seeded {len(solutions)}/{len(seed_mus)} solutions", flush=True)
    if solutions:
        max_mu = max(solutions.keys())
        _, ne, minN = solutions[max_mu]
        print(f"  Highest: μ/m₀={max_mu/m0:.4f}, n_eff={ne}, min(N)={minN:.4f}", flush=True)

    if not solutions:
        print("  No seeds found!")
        return solutions, None

    # Step 2: μ-continuation from highest seed
    print("μ-continuation...", flush=True)
    sol_mus = sorted(solutions.keys())
    mu_cur = sol_mus[-1]
    Phi_cur = solutions[mu_cur][0].copy()
    ne_cur = solutions[mu_cur][1]

    # Tangent from last two solutions
    if len(sol_mus) >= 2:
        mu_prev = sol_mus[-2]
        dPhi_dmu = (Phi_cur - solutions[mu_prev][0]) / (mu_cur - mu_prev)
    else:
        dPhi_dmu = np.zeros(N)

    dmu = 0.005 * m0
    fails = 0

    while mu_cur < 3.0 * m0 and fails < 20:
        mu_next = mu_cur + dmu
        Phi_seed = Phi_cur + dPhi_dmu * dmu

        Phi_sol, norm, ok, iters = newton_solve(Phi_seed, mu_next, beta_target)
        if ok and np.max(np.abs(Phi_sol)) > 0.01:
            ne, minN = get_props(np.array(Phi_sol), mu_next)
            if minN < 0.01:
                ok = False

        if not ok:
            Phi_sol, norm, ok, iters = newton_solve(Phi_cur, mu_next, beta_target)
            if ok and np.max(np.abs(Phi_sol)) > 0.01:
                ne, minN = get_props(np.array(Phi_sol), mu_next)
                if minN < 0.01:
                    ok = False

        if not ok:
            for frac in [0.5, 0.25, 0.1]:
                sub_dmu = dmu * frac
                Phi_sub = Phi_cur + dPhi_dmu * sub_dmu
                Phi_sol, _, ok2, iters = newton_solve(Phi_sub, mu_cur + sub_dmu, beta_target)
                if ok2 and np.max(np.abs(Phi_sol)) > 0.01:
                    ne2, minN2 = get_props(np.array(Phi_sol), mu_cur + sub_dmu)
                    if minN2 > 0.01:
                        mu_next = mu_cur + sub_dmu
                        dmu = sub_dmu
                        ne, minN = ne2, minN2
                        ok = True
                        break

        if not ok:
            fails += 1
            dmu = max(dmu * 0.5, 0.0003 * m0)
            if fails % 5 == 0:
                print(f"  FAIL at μ/m₀={mu_next/m0:.4f} [{fails}]", flush=True)
            continue

        fails = 0
        solutions[mu_next] = (np.array(Phi_sol), ne, minN)
        dPhi_dmu = (np.array(Phi_sol) - Phi_cur) / (mu_next - mu_cur)
        Phi_cur = np.array(Phi_sol).copy()
        mu_cur = mu_next
        ne_cur = ne
        if dmu < 0.005 * m0:
            dmu = min(dmu * 2, 0.005 * m0)

    ne_final, minN_final = get_props(Phi_cur, mu_cur)
    print(f"  Fold: μ*/m₀ ≈ {mu_cur/m0:.4f}, n_eff={ne_final}, min(N)={minN_final:.4f}", flush=True)

    # Step 3: Eigenvalue at fold
    eig = compute_eigenvalue(Phi_cur, mu_cur, beta_target)
    print(f"  |λ₁| at fold: {eig:.4f}", flush=True)

    fold_info = {
        "mu_star": mu_cur, "ne_star": ne_final, "minN_star": minN_final,
        "eig1_star": eig
    }

    return solutions, fold_info


# ── Load bt5 seeds ──────────────────────────────────────────────────────
d5 = np.load("tov_tk_chemical_el.npz", allow_pickle=True)
bt5 = {}
for mu_val in d5["mu_values"]:
    key = f"mu{mu_val:.2f}"
    try:
        Phi = np.array(d5[f"{key}_Phi"])
        ne, minN = get_props(Phi, mu_val)
        if minN > 0.01 and np.max(np.abs(Phi)) > 0.01:
            bt5[mu_val] = (Phi, ne, minN)
    except KeyError:
        pass
print(f"bt5: {len(bt5)} solutions")

# ── Load bt10 seeds ─────────────────────────────────────────────────────
d10 = np.load("tov_tk_bt10_mu_scan.npz", allow_pickle=True)
bt10 = {}
# Reconstruct solutions by re-solving from the scan data
# The scan stored mu_values but not individual Phi arrays
# We need to re-seed from bt5 at bt10 — use the same approach
# Actually, let's just re-seed quickly
print("Re-seeding bt10 solutions from bt5 (fast)...", flush=True)
for mu_val in sorted(bt5.keys()):
    Phi5, ne5, _ = bt5[mu_val]
    for scale in [1.0, 0.9, 0.8]:
        Phi_sol, norm, ok, iters = newton_solve(Phi5 * scale, mu_val, 10.0)
        if ok and np.max(np.abs(Phi_sol)) > 0.01:
            ne, minN = get_props(np.array(Phi_sol), mu_val)
            if minN > 0.01:
                bt10[mu_val] = (np.array(Phi_sol), ne, minN)
                break
print(f"bt10: {len(bt10)} solutions")

# ── bt12 from bt10 ─────────────────────────────────────────────────────
bt12, fold12 = seed_and_scan(bt10, 12.0, "from bt10")

# ── bt15 from bt12 ─────────────────────────────────────────────────────
bt15, fold15 = seed_and_scan(bt12 if bt12 else bt10, 15.0, "from bt12" if bt12 else "from bt10")

# ── Summary ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("FOLD SUMMARY")
print(f"{'='*60}")
print(f"  β₀t₀=5:  μ*/m₀ ≈ 0.658, n_eff=10, min(N)≈0.055")
print(f"  β₀t₀=10: μ*/m₀ ≈ 0.702, n_eff=8,  min(N)≈0.173")
if fold12:
    print(f"  β₀t₀=12: μ*/m₀ ≈ {fold12['mu_star']/m0:.3f}, "
          f"n_eff={fold12['ne_star']}, min(N)≈{fold12['minN_star']:.3f}, |λ₁|={fold12['eig1_star']:.3f}")
if fold15:
    print(f"  β₀t₀=15: μ*/m₀ ≈ {fold15['mu_star']/m0:.3f}, "
          f"n_eff={fold15['ne_star']}, min(N)≈{fold15['minN_star']:.3f}, |λ₁|={fold15['eig1_star']:.3f}")

# Save
save_data = {"m0": m0, "csq": csq}
for bt_label, fold_data in [("bt12", fold12), ("bt15", fold15)]:
    if fold_data:
        for k, v in fold_data.items():
            save_data[f"{bt_label}_{k}"] = v

np.savez("tov_tk_beta_chain.npz", **save_data)
print("\nSaved tov_tk_beta_chain.npz")
