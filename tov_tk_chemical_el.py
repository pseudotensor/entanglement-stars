#!/usr/bin/env python3
"""
TK star chemical potential scan — FULL EL with KMS/BKM conductances.

Uses jax.grad of the conductance energy functional for the exact EL LHS
(Laplacian + EL correction), and jax.jacfwd for exact Jacobians.
Host-side Newton loop avoids JIT compilation issues with lax.while_loop.

Continuation seeding: each mu uses the previous solution as initial guess.

Outputs figures to fermion/numerical/figures/ for the paper.
"""
import os, time
import numpy as np
os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, jacfwd

print(f"JAX devices: {jax.devices()}", flush=True)
t0 = 1.0; N_floor = 0.005


def make_solver(N, beta0, m0, csq):
    n_arr = jnp.arange(1, N+1, dtype=jnp.float64)
    g = 4.0 * jnp.pi * n_arr**2
    diag0 = jnp.zeros(N, dtype=jnp.float64)
    nunk = N - 1

    def _bond_correlations(diag_vals, off_vals):
        """Diagonal and off-diagonal correlations from eigh."""
        H = jnp.diag(diag_vals) + jnp.diag(off_vals, 1) + jnp.diag(off_vals, -1)
        evals, evecs = jnp.linalg.eigh(H)
        be = jnp.clip(beta0 * evals, -500.0, 500.0)
        f = 1.0 / (jnp.exp(be) + 1.0)
        vf = evecs * f[None, :]
        Gd = jnp.sum(evecs * vf, axis=1)
        Gs = jnp.sum(evecs[:-1] * vf[1:], axis=1)
        return Gd, Gs

    def _bkm_cov(diag_vals, off_vals):
        """BKM bond-current covariance for each bond."""
        H = jnp.diag(diag_vals) + jnp.diag(off_vals, 1) + jnp.diag(off_vals, -1)
        evals, evecs = jnp.linalg.eigh(H)
        be = jnp.clip(beta0 * evals, -500.0, 500.0)
        f = 1.0 / (jnp.exp(be) + 1.0)
        diff = beta0 * (evals[:, None] - evals[None, :])
        phi = jnp.where(jnp.abs(diff) > 1e-12,
                         jnp.expm1(diff) / jnp.where(jnp.abs(diff) > 1e-12, diff, 1.0),
                         1.0)
        K = f[:, None] * (1.0 - f[None, :]) * phi
        U_n = evecs[:-1, :]; U_n1 = evecs[1:, :]
        t1 = jnp.sum((U_n1**2) @ K * (U_n**2), axis=1)
        t2 = jnp.sum((U_n**2) @ K * (U_n1**2), axis=1)
        t3 = jnp.sum((U_n * U_n1) @ K * (U_n * U_n1), axis=1)
        return t1 + t2 - 2.0 * t3

    # Background BKM covariance (computed once)
    off_bg = -t0 * jnp.ones(N-1, dtype=jnp.float64)
    cov_bg = _bkm_cov(diag0, off_bg)
    cov_bg = jnp.maximum(cov_bg, 1e-30)

    def _conductances(Phi, mu):
        """BKM conductances kappa(Phi) = g t0^2 Nbar^2 cov/cov_bg."""
        lapse = jnp.maximum(1.0 + Phi / csq, N_floor)
        Nbar = 0.5 * (lapse[:-1] + lapse[1:])
        off = -t0 * jnp.abs(Nbar)
        cov = _bkm_cov(diag0, off)
        cov_ratio = cov / cov_bg
        return g[:-1] * t0**2 * Nbar**2 * cov_ratio

    def _energy(Phi, mu):
        """Conductance energy: (1/2) sum_b kappa_b(Phi) * (dPhi_b)^2."""
        kappa = _conductances(Phi, mu)
        dPhi = Phi[:-1] - Phi[1:]
        return 0.5 * jnp.sum(kappa * dPhi**2)

    _grad_energy = jax.grad(_energy, argnums=0)

    def _residual(Phi, mu):
        """JAX-traceable EL residual. mu is a traced scalar argument."""
        # LHS = grad of energy functional (Laplacian + EL correction, exact)
        lhs = _grad_energy(Phi, mu)

        # Self-consistent source
        lapse = jnp.maximum(1.0 + Phi / csq, N_floor)
        Nbar = 0.5 * (lapse[:-1] + lapse[1:])
        Nbar_abs = jnp.abs(Nbar)
        off = -t0 * Nbar_abs

        # Chemical potential threshold: V = max(m0*(1-N) - mu, 0)
        V_raw = m0 * (1.0 - lapse) - mu
        V_defect = jnp.maximum(V_raw, 0.0)

        # Sigma state (background Hamiltonian, same off-diag)
        Gd_s, Gs_s = _bond_correlations(diag0, off)
        rho_s = jnp.zeros(N, dtype=jnp.float64)
        rho_s = rho_s.at[:-1].add(2.0 * t0 * Nbar_abs * Gs_s * g[:-1])

        # Target state (with defect potential)
        Gd_t, Gs_t = _bond_correlations(V_defect, off)
        rho_t = jnp.zeros(N, dtype=jnp.float64)
        rho_t = rho_t.at[:-1].add(2.0 * t0 * Nbar_abs * Gs_t * g[:-1])
        rho_t = rho_t + V_defect * Gd_t * g

        pref = beta0 / csq
        F = lhs - pref * (rho_s - rho_t)
        # BC: Phi_N = 0 (last entry of full Phi vector)
        F = F.at[N - 1].set(Phi[N - 1])
        return F

    # JIT compile residual and Jacobian (mu is traced → compiles once)
    resid_jit = jit(_residual)
    jac_jit = jit(jacfwd(_residual, argnums=0))

    # Warm up JIT (one-time compilation cost)
    print("  JIT compiling residual...", flush=True)
    t_jit = time.time()
    _dummy = jax.block_until_ready(resid_jit(jnp.zeros(N), jnp.float64(0.1)))
    print(f"  Residual compiled in {time.time() - t_jit:.1f}s", flush=True)

    t_jit = time.time()
    _dummy = jax.block_until_ready(jac_jit(jnp.zeros(N), jnp.float64(0.1)))
    print(f"  Jacobian (jacfwd) compiled in {time.time() - t_jit:.1f}s", flush=True)

    def diagnostics(Phi_np, mu_val):
        """Compute all diagnostic quantities from a solution (numpy)."""
        Phi = np.array(Phi_np)
        lapse = np.maximum(1.0 + Phi / csq, N_floor)
        Nbar = 0.5 * (lapse[:-1] + lapse[1:])
        Nbar_abs = np.abs(Nbar)
        off = jnp.array(-t0 * Nbar_abs)

        V_raw = m0 * (1.0 - lapse) - mu_val
        V_defect = np.maximum(V_raw, 0.0)
        n_eff = int(np.sum(V_defect > 1e-10))

        Gd_s, Gs_s = _bond_correlations(diag0, off)
        Gd_s = np.array(Gd_s); Gs_s = np.array(Gs_s)
        Gd_t, Gs_t = _bond_correlations(jnp.array(V_defect), off)
        Gd_t = np.array(Gd_t); Gs_t = np.array(Gs_t)

        n_np = np.array(n_arr)
        g_np = np.array(g)
        rho_s = np.zeros(N); rho_s[:-1] = 2.0*t0*Nbar_abs*Gs_s*g_np[:-1]
        rho_t = np.zeros(N); rho_t[:-1] = 2.0*t0*Nbar_abs*Gs_t*g_np[:-1]
        rho_t += V_defect * Gd_t * g_np
        source = (beta0 / csq) * (rho_s - rho_t)

        cov = np.array(_bkm_cov(diag0, off))
        cov_bg_np = np.array(cov_bg)
        kappa_kms = g_np[:-1] * t0**2 * Nbar_abs**2 * (cov / cov_bg_np)

        return {"Phi": Phi, "lapse": lapse, "source": source,
                "V_defect": V_defect, "kappa_kms": kappa_kms,
                "n_eff": n_eff, "Nbar": Nbar_abs,
                "Gd_s": Gd_s, "Gd_t": Gd_t, "Gs_s": Gs_s, "Gs_t": Gs_t,
                "rho_s": rho_s, "rho_t": rho_t}

    def newton(mu_val, seed, max_iter=80, verbose=False):
        """Newton solve with jacfwd Jacobian. Full N-vector (Phi_N=0 via BC)."""
        Phi = jnp.array(seed, dtype=jnp.float64)
        if Phi.shape[0] == nunk:
            Phi = jnp.concatenate([Phi, jnp.array([0.0])])
        mu_j = jnp.float64(mu_val)

        best_Phi, best_rn = Phi, 1e30
        stagnant = 0

        for it in range(max_iter):
            F = jax.block_until_ready(resid_jit(Phi, mu_j))
            rn = float(jnp.max(jnp.abs(F)))
            if rn < best_rn:
                best_Phi = Phi; best_rn = rn
            if verbose and (it % 5 == 0 or rn < 1e-8):
                lapse_min = float(jnp.min(jnp.maximum(1.0 + Phi / csq, N_floor)))
                print(f"    it={it:3d} |R|={rn:.3e} minN={lapse_min:.4f}", flush=True)
            if rn < 1e-8:
                if verbose: print(f"    CONVERGED at it={it}")
                break

            J = jax.block_until_ready(jac_jit(Phi, mu_j))
            reg = 1e-10 * jnp.maximum(jnp.abs(jnp.diag(J)), 1e-10)
            J_reg = J + jnp.diag(reg)
            dPhi = jnp.linalg.solve(J_reg, -F)

            # Line search
            alpha = 1.0
            improved = False
            for _ in range(8):
                Phi_trial = Phi + alpha * dPhi
                lt = 1.0 + Phi_trial / csq
                if float(jnp.min(lt)) < N_floor:
                    alpha *= 0.5; continue
                F_trial = jax.block_until_ready(resid_jit(Phi_trial, mu_j))
                rn_trial = float(jnp.max(jnp.abs(F_trial)))
                if rn_trial < rn:
                    Phi = Phi_trial
                    improved = True
                    break
                alpha *= 0.5

            if not improved:
                Phi = Phi + 0.01 * dPhi
                stagnant += 1
                if stagnant >= 10:
                    break
            else:
                stagnant = 0

            if float(jnp.any(jnp.isnan(Phi))):
                break

        F_final = jax.block_until_ready(resid_jit(best_Phi, mu_j))
        final_rn = float(jnp.max(jnp.abs(F_final)))
        return np.array(best_Phi), final_rn

    return {"newton": newton, "diagnostics": diagnostics,
            "n_arr": np.array(n_arr), "g": np.array(g), "N": N, "nunk": nunk}


# Parameters
beta0 = 5.0; m0 = 3.0; csq = 5.0
N_val = 200

print(f"JAX devices: {jax.devices()}", flush=True)
print(f"Parameters: beta0={beta0}, m0={m0}, c*2={csq}, N={N_val}")
print(f"FULL EL solver with BKM conductances + jacfwd Jacobian\n")

S = make_solver(N_val, beta0, m0, csq)
n_arr = S["n_arr"]
g = S["g"]

# Scan mu with continuation seeding
mu_values = ([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
             + [round(1.0 + 0.05*i, 2) for i in range(1, 17)]   # 1.05..1.80 in 0.05
             + [round(1.80 + 0.01*i, 2) for i in range(1, 21)]  # 1.81..2.00 in 0.01
             + [round(2.0 + 0.05*i, 2) for i in range(1, 21)])  # 2.05..3.00 in 0.05
results = {}
prev_solutions = []  # list of (mu, Phi) for continuation seeding

for mu in mu_values:
    N_surface = 1.0 - mu / m0
    print(f"\n{'='*70}")
    print(f"  mu = {mu:.2f}  ->  N* = {N_surface:.4f}")
    print(f"{'='*70}", flush=True)
    t_start = time.time()

    best_Phi, best_rn = None, 1e30

    # Build seed list: scaled continuations from recent solutions, then Gaussians
    seeds = []
    for prev_mu, prev_Phi in reversed(prev_solutions[-3:]):
        # Scale previous solution by mu ratio (deeper well ~ proportional Phi)
        ratio = mu / prev_mu if prev_mu > 0 else 1.0
        for scale in [ratio, 1.0, 1.1*ratio, 0.9*ratio, 1.3*ratio, 1.5*ratio]:
            scaled = prev_Phi * scale
            scaled[-1] = 0.0
            if 1 + min(scaled)/csq >= N_floor:
                seeds.append((f"cont mu={prev_mu:.2f} x{scale:.2f}", scaled))
    for amp in [0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 18.0]:
        for w in [3.0, 5.0, 8.0, 15.0]:
            seed = -amp * np.exp(-0.5 * (n_arr / w)**2)
            seed[-1] = 0.0  # BC
            if 1 + min(seed)/csq < 0.01: continue
            seeds.append((f"gauss amp={amp},w={w}", seed))

    for label, seed in seeds:
        Phi, rn = S["newton"](mu, seed, max_iter=80, verbose=False)
        d = S["diagnostics"](Phi, mu)
        if abs(min(d["Phi"])) > 0.01 and rn < 1e-4 and d["n_eff"] > 0:
            if rn < best_rn:
                best_Phi, best_rn = Phi, rn
                best_d = d
                print(f"    [{label}] |R|={rn:.2e} minN={min(d['lapse']):.4f}", flush=True)
            if rn < 1e-8:
                break
        if best_rn < 1e-8:
            break

    dt = time.time() - t_start

    if best_Phi is not None and best_rn < 1e-4:
        d = best_d
        Phi = d["Phi"]; lapse = d["lapse"]
        n_eff = d["n_eff"]
        prev_solutions.append((mu, best_Phi.copy()))

        # w = N^2 fit: w = 1 - r_s/r  (Schwarzschild)
        w = lapse**2
        n_start = max(n_eff + 3, 5)
        n_end = min(N_val//2, n_eff + 40)
        if n_start < n_end:
            idx = np.arange(n_start-1, n_end)
            r_fit = n_arr[idx]
            basis_1r = 1.0/r_fit - 1.0/N_val
            target_w = -(w[idx] - 1.0)
            r_s_w = np.dot(basis_1r, target_w) / np.dot(basis_1r, basis_1r)
            w_pred = 1.0 - r_s_w * basis_1r
            rms_w = np.sqrt(np.mean((w[idx] - w_pred)**2))
            rms_w_norm = np.sqrt(np.mean((w[idx]-1.0)**2))
            w_rel_err = rms_w / rms_w_norm if rms_w_norm > 1e-30 else 0
        else:
            r_s_w = 0; w_rel_err = 0

        # Also do 1/r Phi-fit for comparison
        if n_start < n_end:
            basis_phi = 1.0/N_val - 1.0/r_fit
            A_fit = np.dot(basis_phi, Phi[idx]) / np.dot(basis_phi, basis_phi)
            Phi_pred = A_fit * basis_phi
            phi_rel_err = (np.sqrt(np.mean((Phi[idx]-Phi_pred)**2)) /
                           np.sqrt(np.mean(Phi[idx]**2))
                           if np.any(Phi[idx] != 0) else 0)
            r_s_phi = 2*abs(A_fit)/csq
        else:
            A_fit = 0; phi_rel_err = 0; r_s_phi = 0

        # w-flux constancy (conserved for EL)
        dw = w[:-1] - w[1:]
        J_w = g[:-1] * t0**2 * dw
        f_start = min(n_eff + 5, N_val - 2)
        f_end = min(N_val//2, n_eff + 40)
        if f_start < f_end:
            J_w_ext = J_w[f_start:f_end]
            J_w_mean = np.mean(J_w_ext)
            J_w_var = (np.max(np.abs(J_w_ext - J_w_mean)) / abs(J_w_mean)
                       if abs(J_w_mean) > 1e-30 and len(J_w_ext) > 1 else 0)
        else:
            J_w_var = 0

        # Compactness
        r_star = float(n_eff)
        compact_w = r_star / r_s_w if r_s_w > 0 else float('inf')
        compact_phi = r_star / r_s_phi if r_s_phi > 0 else float('inf')

        # Baryon number
        dN = d["Gd_t"] - d["Gd_s"]
        N_B = np.sum(g * np.maximum(dN, 0))

        results[mu] = {"d": d, "rn": best_rn, "n_eff": n_eff,
                        "A_fit": A_fit, "r_s_phi": r_s_phi,
                        "phi_rel_err": phi_rel_err,
                        "r_s_w": r_s_w, "w_rel_err": w_rel_err,
                        "compact_w": compact_w, "compact_phi": compact_phi,
                        "J_w_var": J_w_var, "N_B": N_B}

        print(f"\n  RESULTS (mu={mu:.2f}):")
        print(f"    Time:      {dt:.1f}s")
        print(f"    |R_EL|   = {best_rn:.2e}")
        print(f"    minPhi   = {min(Phi):.6f}")
        print(f"    minN     = {min(lapse):.6f}")
        print(f"    n_eff    = {n_eff}")
        print(f"    Phi-fit: A={A_fit:.4f}  r_s={r_s_phi:.3f}  r*/r_s={compact_phi:.2f}  err={phi_rel_err:.4f}")
        print(f"    w-fit:   r_s={r_s_w:.3f}  r*/r_s={compact_w:.2f}  err={w_rel_err:.4f}")
        print(f"    J_w var  = {J_w_var:.6f}")
    else:
        if best_Phi is not None:
            print(f"  Converged to trivial (minPhi={min(best_Phi):.4f}, |R|={best_rn:.2e})")
        else:
            print(f"  No solution found")
        print(f"  Time: {dt:.1f}s")


# Summary table
print(f"\n{'='*90}")
print(f"PAPER TABLE: Full EL TK star solutions")
print(f"N={N_val}, beta0={beta0}, m0={m0}, c*2={csq}")
print(f"{'='*90}")
print(f"  {'mu':>6s} {'N*':>8s} {'n_eff':>6s} {'minPhi':>10s} {'minN':>8s} "
      f"{'r_s(w)':>8s} {'r*/r_s':>8s} {'d_w':>8s} {'dJ_w':>8s} {'|R_EL|':>10s}")
for mu in sorted(results.keys()):
    r = results[mu]; d = r["d"]
    N_star = 1.0 - mu/m0
    print(f"  {mu:6.2f} {N_star:8.4f} {r['n_eff']:6d} {min(d['Phi']):10.4f} "
          f"{min(d['lapse']):8.4f} {r['r_s_w']:8.3f} {r['compact_w']:8.2f} "
          f"{r['w_rel_err']:8.5f} {r['J_w_var']:8.5f} {r['rn']:10.2e}")

# Comparison table
print(f"\n  Comparison: w-fit vs Phi-fit")
print(f"  {'mu':>6s} {'r_s(w)':>8s} {'r*/r_s(w)':>10s} {'d_w%':>8s} "
      f"{'r_s(Phi)':>8s} {'r*/r_s(Phi)':>10s} {'d_Phi%':>8s}")
for mu in sorted(results.keys()):
    r = results[mu]
    print(f"  {mu:6.2f} {r['r_s_w']:8.3f} {r['compact_w']:10.2f} {r['w_rel_err']*100:8.3f} "
          f"{r['r_s_phi']:8.3f} {r['compact_phi']:10.2f} {r['phi_rel_err']*100:8.3f}")


# Save data
np.savez("tov_tk_chemical_el.npz",
         beta0=beta0, m0=m0, csq=csq, N=N_val,
         mu_values=[mu for mu in sorted(results.keys())],
         **{f"mu{mu:.2f}_Phi": results[mu]["d"]["Phi"] for mu in results},
         **{f"mu{mu:.2f}_lapse": results[mu]["d"]["lapse"] for mu in results},
         **{f"mu{mu:.2f}_V_defect": results[mu]["d"]["V_defect"] for mu in results},
         **{f"mu{mu:.2f}_source": results[mu]["d"]["source"] for mu in results},
         **{f"mu{mu:.2f}_kappa_kms": results[mu]["d"]["kappa_kms"] for mu in results},
         **{f"mu{mu:.2f}_n_eff": results[mu]["n_eff"] for mu in results},
         **{f"mu{mu:.2f}_r_s_w": results[mu]["r_s_w"] for mu in results},
)
print("Saved tov_tk_chemical_el.npz")
print("\nDone.", flush=True)
