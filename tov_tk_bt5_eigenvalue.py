#!/usr/bin/env python3
"""β₀t₀=5 TK eigenvalue scan to characterize fold bifurcation.

Loads solutions from tov_tk_chemical_el.npz, computes Jacobian eigenvalues
for a sample of existing solutions, then continues with fine steps to
pin down the fold.

Saves results to tov_tk_bt5_eigenvalue.npz.
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

# ── Parameters ──────────────────────────────────────────────────────────
beta0 = 5.0; m0 = 3.0; csq = 5.0; N = 200
t0 = 1.0; N_floor = 0.005
n_arr = jnp.arange(1, N+1, dtype=jnp.float64)
g = 4.0 * jnp.pi * n_arr**2
diag0 = jnp.zeros(N, dtype=jnp.float64)

# ── Physics functions ───────────────────────────────────────────────────
def _bkm_cov(diag_vals, off_vals):
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

def _bond_correlations(diag_vals, off_vals):
    H = jnp.diag(diag_vals) + jnp.diag(off_vals, 1) + jnp.diag(off_vals, -1)
    evals, evecs = jnp.linalg.eigh(H)
    be = jnp.clip(beta0 * evals, -500.0, 500.0)
    f = 1.0 / (jnp.exp(be) + 1.0)
    vf = evecs * f[None, :]
    return jnp.sum(evecs * vf, axis=1), jnp.sum(evecs[:-1] * vf[1:], axis=1)

off_bg = -t0 * jnp.ones(N-1, dtype=jnp.float64)
cov_bg = jnp.maximum(_bkm_cov(diag0, off_bg), 1e-30)

def _energy(Phi, mu):
    lapse = jnp.maximum(1.0 + Phi / csq, N_floor)
    Nbar = 0.5 * (lapse[:-1] + lapse[1:])
    off = -t0 * jnp.abs(Nbar)
    kappa = g[:-1] * t0**2 * Nbar**2 * (_bkm_cov(diag0, off) / cov_bg)
    dPhi = Phi[:-1] - Phi[1:]
    return 0.5 * jnp.sum(kappa * dPhi**2)

_grad_energy = jax.grad(_energy, argnums=0)

def _residual(Phi, mu):
    lhs = _grad_energy(Phi, mu)
    lapse = jnp.maximum(1.0 + Phi / csq, N_floor)
    Nbar = 0.5 * (lapse[:-1] + lapse[1:])
    Nbar_abs = jnp.abs(Nbar)
    off = -t0 * Nbar_abs
    V_defect = jnp.maximum(m0 * (1.0 - lapse) - mu, 0.0)
    Gd_s, Gs_s = _bond_correlations(diag0, off)
    rho_s = jnp.zeros(N, dtype=jnp.float64).at[:-1].add(2.0 * t0 * Nbar_abs * Gs_s * g[:-1])
    Gd_t, Gs_t = _bond_correlations(V_defect, off)
    rho_t = (jnp.zeros(N, dtype=jnp.float64).at[:-1].add(2.0 * t0 * Nbar_abs * Gs_t * g[:-1])
             + V_defect * Gd_t * g)
    F = lhs - (beta0 / csq) * (rho_s - rho_t)
    return F.at[N - 1].set(Phi[N - 1])

# ── JIT compile ─────────────────────────────────────────────────────────
print("JIT compiling...", flush=True)
t_jit = time.time()
resid_jit = jit(_residual)
jac_jit = jit(jacfwd(_residual, argnums=0))
_d = jnp.zeros(N); _m = jnp.float64(0.1)
jax.block_until_ready(resid_jit(_d, _m))
jax.block_until_ready(jac_jit(_d, _m))
print(f"JIT done in {time.time() - t_jit:.1f}s", flush=True)


def newton_solve(Phi0, mu, max_iter=60, tol=1e-8):
    Phi = jnp.array(Phi0, dtype=jnp.float64)
    mu_j = jnp.float64(mu)
    for it in range(max_iter):
        F = np.array(jax.block_until_ready(resid_jit(Phi, mu_j)))
        norm = float(np.max(np.abs(F)))
        if norm < tol:
            return np.array(Phi), norm, True, it
        J = np.array(jax.block_until_ready(jac_jit(Phi, mu_j)))
        try:
            dPhi = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            return np.array(Phi), norm, False, it
        alpha = 1.0
        for _ in range(8):
            Phi_trial = Phi + alpha * jnp.array(dPhi)
            F_trial = np.array(jax.block_until_ready(resid_jit(Phi_trial, mu_j)))
            if np.max(np.abs(F_trial)) < norm:
                Phi = Phi_trial
                break
            alpha *= 0.5
        else:
            Phi = Phi + alpha * jnp.array(dPhi)
    F = np.array(jax.block_until_ready(resid_jit(Phi, mu_j)))
    norm = float(np.max(np.abs(F)))
    return np.array(Phi), norm, norm < tol, max_iter


def is_nontrivial(Phi):
    return float(np.max(np.abs(Phi))) > 0.01


def compute_eigenvalue(Phi, mu):
    J = np.array(jax.block_until_ready(jac_jit(jnp.array(Phi), jnp.float64(mu))))
    J_int = J[:-1, :-1]
    eigs = np.sort(np.abs(np.linalg.eigvals(J_int)))
    return float(eigs[0]), float(eigs[1])


def get_props(Phi, mu):
    lapse = np.maximum(1.0 + Phi / csq, N_floor)
    minN = float(np.min(lapse))
    V_defect = np.maximum(m0 * (1.0 - lapse) - mu, 0.0)
    ne = int(np.sum(V_defect > 1e-10))
    return ne, minN


# ── Load existing data ──────────────────────────────────────────────────
print("Loading existing data...", flush=True)
d = np.load("tov_tk_chemical_el.npz", allow_pickle=True)
mu_values = d["mu_values"]

# Get all nontrivial solutions with lapse data
mus_existing = []
Phis_existing = {}
for mu in mu_values:
    key = f"mu{mu:.2f}"
    try:
        Phi = np.array(d[f"{key}_Phi"])
        lapse = np.array(d[f"{key}_lapse"])
        if is_nontrivial(Phi):
            mus_existing.append(mu)
            Phis_existing[mu] = Phi
    except KeyError:
        pass

mus_existing.sort()
print(f"Loaded {len(mus_existing)} nontrivial solutions, "
      f"mu/m0: {mus_existing[0]/m0:.4f} to {mus_existing[-1]/m0:.4f}", flush=True)

# ── Eigenvalues for sampled existing solutions ──────────────────────────
print("\n=== Eigenvalues for existing solutions ===", flush=True)
# Sample: every 5th + last 15
sample_idx = list(range(0, len(mus_existing), 5)) + \
             list(range(max(0, len(mus_existing)-15), len(mus_existing)))
sample_idx = sorted(set(sample_idx))

all_data = []
for i in sample_idx:
    mu = mus_existing[i]
    Phi = Phis_existing[mu]
    ne, minN = get_props(Phi, mu)
    eig1, eig2 = compute_eigenvalue(Phi, mu)
    all_data.append((mu, ne, minN, eig1, eig2))
    print(f"  mu/m0={mu/m0:.4f}: n_eff={ne}, min(N)={minN:.4f}, "
          f"|λ₁|={eig1:.4f}, |λ₂|={eig2:.4f}", flush=True)

# ── Continue past last existing solution ────────────────────────────────
print("\n=== Continuation past existing data ===", flush=True)
mu_a = mus_existing[-2]
mu_b = mus_existing[-1]
Phi_a = Phis_existing[mu_a]
Phi_b = Phis_existing[mu_b]
dPhi_dmu = (Phi_b - Phi_a) / (mu_b - mu_a)

Phi_prev = Phi_b.copy()
mu_current = mu_b
dmu = 0.001 * m0  # very fine steps (0.003 in mu)

consecutive_fails = 0
t_start = time.time()

while mu_current < 2.5 * m0 and consecutive_fails < 15:
    mu_next = mu_current + dmu
    Phi_seed = Phi_prev + dPhi_dmu * dmu

    Phi_sol, norm, converged, iters = newton_solve(Phi_seed, mu_next)

    if converged and not is_nontrivial(Phi_sol):
        converged = False
        norm = -1.0

    if not converged:
        for frac in [0.5, 0.25, 0.1]:
            sub_dmu = dmu * frac
            mu_sub = mu_current + sub_dmu
            Phi_sub = Phi_prev + dPhi_dmu * sub_dmu
            Phi_sol, norm_s, converged, iters = newton_solve(Phi_sub, mu_sub)
            if converged and is_nontrivial(Phi_sol):
                mu_next = mu_sub
                dmu = sub_dmu
                norm = norm_s
                break
            converged = False

    if not converged:
        for scale in [1.01, 0.99, 1.03, 0.97, 1.05, 0.95]:
            for frac in [1.0, 0.5, 0.25]:
                mu_try = mu_current + dmu * frac
                Phi_sol, norm_s, conv, iters = newton_solve(Phi_prev * scale, mu_try)
                if conv and is_nontrivial(Phi_sol):
                    mu_next = mu_try
                    dmu = dmu * frac if frac < 1 else dmu
                    norm = norm_s
                    converged = True
                    break
            if converged:
                break

    if not converged:
        consecutive_fails += 1
        tag = "trivial" if norm < 0 else f"norm={norm:.2e}"
        print(f"  FAIL mu/m0={mu_next/m0:.4f} ({tag}) "
              f"[{consecutive_fails}/15]", flush=True)
        if consecutive_fails <= 5:
            dmu = max(dmu * 0.5, 0.0001 * m0)
        continue

    consecutive_fails = 0
    ne, minN = get_props(np.array(Phi_sol), mu_next)
    eig1, eig2 = compute_eigenvalue(np.array(Phi_sol), mu_next)

    all_data.append((mu_next, ne, minN, eig1, eig2))

    elapsed = time.time() - t_start
    print(f"  mu/m0={mu_next/m0:.4f}: n_eff={ne}, min(N)={minN:.4f}, "
          f"|λ₁|={eig1:.4f}, |λ₂|={eig2:.4f}, it={iters}, t={elapsed:.0f}s", flush=True)

    dPhi_dmu = (np.array(Phi_sol) - Phi_prev) / (mu_next - mu_current)
    Phi_prev = np.array(Phi_sol).copy()
    mu_current = mu_next

    if eig1 < 0.5:
        dmu = max(0.0001 * m0, dmu * 0.5)
    elif eig1 < 2.0:
        dmu = max(0.0003 * m0, min(dmu, 0.0005 * m0))
    elif consecutive_fails == 0 and dmu < 0.001 * m0:
        dmu = min(dmu * 1.3, 0.001 * m0)

    if eig1 < 0.05:
        print(f"\n*** FOLD DETECTED at mu/m0 ~ {mu_next/m0:.4f} ***", flush=True)
        break

# ── Save ────────────────────────────────────────────────────────────────
print(f"\nSaving {len(all_data)} data points...", flush=True)
np.savez("tov_tk_bt5_eigenvalue.npz",
    beta0=beta0, m0=m0, csq=csq, N=N,
    mu_values=np.array([d[0] for d in all_data]),
    n_eff_values=np.array([d[1] for d in all_data]),
    minN_values=np.array([d[2] for d in all_data]),
    eig1_values=np.array([d[3] for d in all_data]),
    eig2_values=np.array([d[4] for d in all_data]))
print("Saved tov_tk_bt5_eigenvalue.npz", flush=True)

# Summary
print(f"\n=== Summary ===")
mus = [d[0]/m0 for d in all_data]
eigs = [d[3] for d in all_data]
minNs = [d[2] for d in all_data]
print(f"mu/m0 range: {mus[0]:.4f} to {mus[-1]:.4f}")
print(f"min(N) range: {minNs[0]:.4f} to {minNs[-1]:.4f}")
print(f"|λ₁| range: {eigs[0]:.4f} to {eigs[-1]:.4f}")
print(f"Total time: {time.time() - t_start:.0f}s")
