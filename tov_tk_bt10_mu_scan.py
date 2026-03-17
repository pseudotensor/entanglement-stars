#!/usr/bin/env python3
"""μ-continuation at fixed β₀t₀=10 to find the true fold.

Step 1: Seed high-n_eff solutions at bt=10 from bt5 seeds (direct Newton).
Step 2: μ-continuation from highest-μ seed to find fold.
Step 3: Eigenvalue analysis near the fold.
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


BETA = 10.0

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
bt5_mus = sorted(bt5.keys())
print(f"bt5: {len(bt5)} non-floor solutions")

# ── Step 1: Seed at bt=10 from ALL bt5 solutions ────────────────────────
print(f"\n=== Seeding bt={BETA:.0f} solutions from bt5 ===", flush=True)
bt10_solutions = {}

for mu_val in bt5_mus:
    Phi5, ne5, _ = bt5[mu_val]
    best = None

    for scale in [1.0, 0.9, 0.8, 0.7, 1.1, 0.6, 0.5]:
        Phi_sol, norm, ok, iters = newton_solve(Phi5 * scale, mu_val, BETA)
        if ok and np.max(np.abs(Phi_sol)) > 0.01:
            ne, minN = get_props(np.array(Phi_sol), mu_val)
            if minN > 0.01:  # non-floor
                if best is None or ne > best[1]:
                    best = (np.array(Phi_sol), ne, minN, scale)
                if scale == 1.0:
                    break  # direct worked, good enough

    if best:
        Phi_sol, ne, minN, scale = best
        bt10_solutions[mu_val] = (Phi_sol, ne, minN)
        if ne != ne5 or mu_val == bt5_mus[-1]:
            print(f"  μ/m₀={mu_val/m0:.4f}: ne5={ne5}→ne10={ne}, min(N)={minN:.4f} (scale={scale})", flush=True)

print(f"\nFound {len(bt10_solutions)} non-floor bt10 solutions")

# ── Step 2: μ-continuation from highest-μ seed ──────────────────────────
print(f"\n=== μ-continuation at bt={BETA:.0f} ===", flush=True)

# Start from highest mu with a non-floor solution
bt10_mus = sorted(bt10_solutions.keys())
if not bt10_mus:
    print("No bt10 seeds found!")
    exit()

mu_start = bt10_mus[-1]
Phi_cur, ne_cur, minN_cur = bt10_solutions[mu_start]
print(f"Starting from μ/m₀={mu_start/m0:.4f}, n_eff={ne_cur}, min(N)={minN_cur:.4f}")

# Use two nearby solutions for tangent prediction
if len(bt10_mus) >= 2:
    mu_prev = bt10_mus[-2]
    Phi_prev = bt10_solutions[mu_prev][0]
    dPhi_dmu = (Phi_cur - Phi_prev) / (mu_start - mu_prev)
else:
    dPhi_dmu = np.zeros(N)

dmu = 0.005 * m0
mu_cur = mu_start
Phi_prev_sol = Phi_cur.copy()
fails = 0
all_data = []

# Also add existing solutions
for mu_val in bt10_mus:
    Phi, ne, minN = bt10_solutions[mu_val]
    all_data.append((mu_val, ne, minN, None))  # None = no eigenvalue yet

while mu_cur < 3.0 * m0 and fails < 20:
    mu_next = mu_cur + dmu

    # Tangent prediction
    Phi_seed = Phi_cur + dPhi_dmu * dmu

    Phi_sol, norm, ok, iters = newton_solve(Phi_seed, mu_next, BETA)
    if ok and np.max(np.abs(Phi_sol)) > 0.01:
        ne, minN = get_props(np.array(Phi_sol), mu_next)
        if minN < 0.01:
            ok = False  # reject floor-bound

    if not ok:
        # Try direct seed
        Phi_sol, norm, ok, iters = newton_solve(Phi_cur, mu_next, BETA)
        if ok and np.max(np.abs(Phi_sol)) > 0.01:
            ne, minN = get_props(np.array(Phi_sol), mu_next)
            if minN < 0.01:
                ok = False

    if not ok:
        # Try smaller steps
        for frac in [0.5, 0.25, 0.1]:
            sub_dmu = dmu * frac
            mu_sub = mu_cur + sub_dmu
            Phi_sub = Phi_cur + dPhi_dmu * sub_dmu
            Phi_sol, _, ok2, iters = newton_solve(Phi_sub, mu_sub, BETA)
            if ok2 and np.max(np.abs(Phi_sol)) > 0.01:
                ne2, minN2 = get_props(np.array(Phi_sol), mu_sub)
                if minN2 > 0.01:
                    mu_next = mu_sub
                    dmu = sub_dmu
                    ne, minN = ne2, minN2
                    ok = True
                    break

    if not ok:
        fails += 1
        dmu = max(dmu * 0.5, 0.0003 * m0)
        if fails % 3 == 0:
            print(f"  FAIL at μ/m₀={mu_next/m0:.4f} [{fails}]", flush=True)
        continue

    fails = 0
    all_data.append((mu_next, ne, minN, None))

    dPhi_dmu = (np.array(Phi_sol) - Phi_cur) / (mu_next - mu_cur)
    Phi_cur = np.array(Phi_sol).copy()
    mu_cur = mu_next

    if len(all_data) % 5 == 0 or ne != ne_cur:
        print(f"  μ/m₀={mu_cur/m0:.4f}: n_eff={ne}, min(N)={minN:.4f}, it={iters}", flush=True)

    ne_cur = ne

    # Adapt step
    if dmu < 0.005 * m0:
        dmu = min(dmu * 2, 0.005 * m0)

print(f"\nReached μ*/m₀={mu_cur/m0:.4f}, n_eff={ne_cur}, min(N)={minN:.4f}")
print(f"Total: {len(all_data)} solutions")

# ── Step 3: Eigenvalue analysis near fold ────────────────────────────────
print(f"\n=== Eigenvalue analysis ===", flush=True)

# Take last 15 solutions and sample earlier ones
n_total = len(all_data)
sample_idx = list(range(0, n_total-15, max(1, (n_total-15)//10))) + list(range(max(0, n_total-15), n_total))
sample_idx = sorted(set(sample_idx))

for i in sample_idx:
    mu_val, ne, minN, _ = all_data[i]
    # Find the solution (reconstruct from continuation)
    # We only have the last Phi_cur. Need to recompute for earlier points.
    # For now, just compute eigenvalue for solutions we can reconstruct.
    pass

# Actually, let's recompute eigenvalues by re-solving from nearby solutions
print("Re-solving for eigenvalues at sampled points...", flush=True)

# Start from the highest-mu bt10 seed and re-solve each sampled point
bt10_mus_sorted = sorted(bt10_solutions.keys())
# Build a list of all solution mu values
all_mus = sorted(set([d[0] for d in all_data]))

# Re-solve: for each sampled mu, find nearest known solution and use as seed
Phi_cache = dict(bt10_solutions)  # mu -> (Phi, ne, minN)

# First forward pass to cache solutions
Phi_chain = bt10_solutions[bt10_mus_sorted[0]][0].copy()
mu_chain = bt10_mus_sorted[0]

for mu_val in all_mus:
    if mu_val in Phi_cache:
        Phi_chain = Phi_cache[mu_val][0].copy()
        mu_chain = mu_val
        continue
    Phi_sol, _, ok, _ = newton_solve(Phi_chain, mu_val, BETA)
    if ok and np.max(np.abs(Phi_sol)) > 0.01:
        ne, minN = get_props(np.array(Phi_sol), mu_val)
        if minN > 0.01:
            Phi_cache[mu_val] = (np.array(Phi_sol), ne, minN)
            Phi_chain = np.array(Phi_sol).copy()
            mu_chain = mu_val

# Now compute eigenvalues at sampled points
eig_data = []
sample_mus = [all_mus[i] for i in sample_idx if i < len(all_mus)]

for mu_val in sample_mus:
    if mu_val not in Phi_cache:
        continue
    Phi, ne, minN = Phi_cache[mu_val]
    eig1 = compute_eigenvalue(Phi, mu_val, BETA)
    eig_data.append((mu_val, ne, minN, eig1))
    print(f"  μ/m₀={mu_val/m0:.4f}: n_eff={ne}, min(N)={minN:.4f}, |λ₁|={eig1:.4f}", flush=True)

# ── Save ────────────────────────────────────────────────────────────────
print("\nSaving...", flush=True)
np.savez("tov_tk_bt10_mu_scan.npz",
    beta0=BETA, m0=m0, csq=csq, N=N,
    mu_values=np.array([d[0] for d in all_data]),
    n_eff_values=np.array([d[1] for d in all_data]),
    minN_values=np.array([d[2] for d in all_data]),
    eig_mu=np.array([d[0] for d in eig_data]) if eig_data else np.array([]),
    eig_neff=np.array([d[1] for d in eig_data]) if eig_data else np.array([]),
    eig_minN=np.array([d[2] for d in eig_data]) if eig_data else np.array([]),
    eig1_values=np.array([d[3] for d in eig_data]) if eig_data else np.array([]),
)
print(f"Saved tov_tk_bt10_mu_scan.npz")
