#!/usr/bin/env python3
"""Pseudo-arclength continuation around the fold bifurcation.

Parameterizes the solution branch by arclength s instead of V0.
The augmented system is:

  F(Phi, V0) = 0              (N equations)
  dot(Phi - Phi_prev, dPhi/ds) + (V0 - V0_prev) * (dV0/ds) - ds = 0  (1 equation)

where (dPhi/ds, dV0/ds) is the tangent predictor from the previous step.

If the fold is real, V0(s) will reach a maximum and turn back.
"""

import os
import numpy as np

os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, jacfwd

from compute_exact_data_gpu import make_residual_fixedpoint

N = 200; t0 = 1.0; n_core = 5; beta0 = 0.1; cstar_sq = 0.5

# Build residual and Jacobian
print("Building residual (grad-based EL)...")
resid_fn, kms_bg = make_residual_fixedpoint(N, t0, n_core, beta0, cstar_sq)
jac_Phi = jit(jacfwd(resid_fn, argnums=0))   # dF/dPhi: (N, N)
jac_V0  = jit(jacfwd(resid_fn, argnums=1))   # dF/dV0: (N,)
resid_jit = jit(resid_fn)

# Warm up
print("Warming up JIT...")
_Phi0 = jnp.zeros(N, dtype=jnp.float64)
_V0 = jnp.float64(0.001)
jax.block_until_ready(resid_jit(_Phi0, _V0))
jax.block_until_ready(jac_Phi(_Phi0, _V0))
jax.block_until_ready(jac_V0(_Phi0, _V0))
print("Ready.\n")

# Load two seed solutions near the fold from probe data
probe = np.load("probe_v0_critical.npz", allow_pickle=True)
V0s = probe["V0s"]
n_sol = len(V0s)
print(f"Loaded {n_sol} probe solutions, V0 range [{V0s[0]:.4f}, {V0s[-1]:.6f}]")

# Use solutions well before the fold to build tangent momentum
# Pick two solutions ~0.001 apart in V0
idx_a = max(0, n_sol - 8)  # ~V0=0.0676
idx_b = max(1, n_sol - 4)  # ~V0=0.0681
V0_a, Phi_a = float(V0s[idx_a]), probe[f"Phi_{idx_a:04d}"]
V0_b, Phi_b = float(V0s[idx_b]), probe[f"Phi_{idx_b:04d}"]
print(f"Starting from V0_a={V0_a:.6f}, V0_b={V0_b:.6f}")
print(f"  min(N)_a = {(1 + Phi_a/cstar_sq).min():.6f}")
print(f"  min(N)_b = {(1 + Phi_b/cstar_sq).min():.6f}")


def newton_corrector(Phi_pred, V0_pred, tangent_Phi, tangent_V0, ds,
                     Phi_prev, V0_prev, max_iter=30, tol=1e-8):
    """Newton corrector for the augmented system."""
    Phi = jnp.array(Phi_pred)
    V0 = float(V0_pred)

    for it in range(max_iter):
        V0j = jnp.float64(V0)
        F = np.array(jax.block_until_ready(resid_jit(Phi, V0j)))
        norm_F = np.max(np.abs(F))

        # Arclength constraint
        dPhi = np.array(Phi) - Phi_prev
        dV0 = V0 - V0_prev
        arc = float(np.dot(dPhi, tangent_Phi) + dV0 * tangent_V0 - ds)

        aug_norm = max(norm_F, abs(arc))
        if aug_norm < tol:
            return np.array(Phi), V0, True, it, aug_norm

        # Jacobian blocks
        J_Phi = np.array(jax.block_until_ready(jac_Phi(Phi, V0j)))  # (N, N)
        F_V0 = np.array(jax.block_until_ready(jac_V0(Phi, V0j)))    # (N,)

        # Augmented system: [J_Phi, F_V0; tangent_Phi^T, tangent_V0] [dPhi; dV0] = -[F; arc]
        # Bordering method: solve J_Phi * a = -F, J_Phi * b = -F_V0
        # then dV0 = -(arc + tangent_Phi @ a) / (tangent_V0 + tangent_Phi @ b)
        # then dPhi = a - dV0 * b

        reg = 1e-10 * np.maximum(np.abs(np.diag(J_Phi)), 1e-10)
        J_reg = J_Phi + np.diag(reg)

        a = np.linalg.solve(J_reg, -F)
        b = np.linalg.solve(J_reg, -F_V0)

        denom = tangent_V0 + np.dot(tangent_Phi, b)
        if abs(denom) < 1e-15:
            print(f"    WARNING: singular bordered system at it={it}")
            return np.array(Phi), V0, False, it, aug_norm

        delta_V0 = -(arc + np.dot(tangent_Phi, a)) / denom
        delta_Phi = a - delta_V0 * b

        # Damped update
        alpha = 1.0
        for _ in range(6):
            Phi_trial = Phi + alpha * jnp.array(delta_Phi)
            V0_trial = V0 + alpha * delta_V0
            F_trial = np.array(jax.block_until_ready(
                resid_jit(Phi_trial, jnp.float64(V0_trial))))
            dPhi_t = np.array(Phi_trial) - Phi_prev
            arc_trial = float(np.dot(dPhi_t, tangent_Phi)
                              + (V0_trial - V0_prev) * tangent_V0 - ds)
            trial_norm = max(np.max(np.abs(F_trial)), abs(arc_trial))
            if trial_norm < aug_norm:
                Phi = Phi_trial
                V0 = V0_trial
                break
            alpha *= 0.5
        else:
            # No improvement, take small step anyway
            Phi = Phi + 0.1 * jnp.array(delta_Phi)
            V0 = V0 + 0.1 * delta_V0

    return np.array(Phi), V0, False, max_iter, aug_norm


# Initial tangent from the two seed solutions
tangent_Phi = Phi_b - Phi_a
tangent_V0 = V0_b - V0_a
# Normalize
norm_t = np.sqrt(np.dot(tangent_Phi, tangent_Phi) + tangent_V0**2)
tangent_Phi /= norm_t
tangent_V0 /= norm_t

# Step size
ds0 = 0.01
ds = ds0
ds_min = 1e-5  # minimum step size floor

# History
history = [(V0_b, Phi_b.copy(), float((1 + Phi_b/cstar_sq).min()))]

Phi_prev = Phi_a.copy()
V0_prev = V0_a
Phi_curr = Phi_b.copy()
V0_curr = V0_b

print(f"\n{'step':>4s}  {'V0':>10s}  {'min(N)':>10s}  {'dV0/ds':>10s}  "
      f"{'|F|':>10s}  {'conv':>5s}  {'ds':>8s}")
print("-" * 75)
print(f"{'0':>4s}  {V0_curr:10.6f}  {(1+Phi_curr/cstar_sq).min():10.6f}  "
      f"{tangent_V0:10.6f}  {'---':>10s}  {'seed':>5s}  {'---':>8s}")

n_steps = 120
V0_max = -1e10
V0_max_step = -1

for step in range(1, n_steps + 1):
    # Predictor
    Phi_pred = Phi_curr + ds * tangent_Phi
    V0_pred = V0_curr + ds * tangent_V0

    # Corrector
    Phi_new, V0_new, conv, nit, aug_res = newton_corrector(
        Phi_pred, V0_pred, tangent_Phi, tangent_V0, ds,
        Phi_curr, V0_curr)

    min_N = float((1 + Phi_new / cstar_sq).min())

    if not conv:
        # Retry with smaller ds
        ds *= 0.5
        if ds < ds_min:
            ds = ds_min
            print(f"  Step {step}: failed (|aug|={aug_res:.2e}), at ds_min={ds_min:.1e}")
            # Try one more time at ds_min then give up
            Phi_pred2 = Phi_curr + ds * tangent_Phi
            V0_pred2 = V0_curr + ds * tangent_V0
            Phi_new2, V0_new2, conv2, nit, aug2 = newton_corrector(
                Phi_pred2, V0_pred2, tangent_Phi, tangent_V0, ds,
                Phi_curr, V0_curr, max_iter=50)
            if not conv2:
                print(f"  Step {step}: cannot continue, stopping.")
                break
            Phi_new, V0_new, conv, aug_res = Phi_new2, V0_new2, conv2, aug2
            min_N = float((1 + Phi_new / cstar_sq).min())
            # fall through to normal processing below
        else:
            print(f"  Step {step}: failed (|aug|={aug_res:.2e}), reducing ds to {ds:.5f}")
            continue

    # Update tangent from bordered system null vector
    # The tangent (dPhi/ds, dV0/ds) satisfies [J_Phi, F_V0] [dPhi/ds; dV0/ds] = 0
    # with normalization ||dPhi/ds||^2 + (dV0/ds)^2 = 1
    # Solve: J_Phi * t_Phi = -F_V0, then t_V0 = 1, normalize
    V0j_new = jnp.float64(V0_new)
    Phi_j_new = jnp.array(Phi_new)
    J_new = np.array(jax.block_until_ready(jac_Phi(Phi_j_new, V0j_new)))
    FV_new = np.array(jax.block_until_ready(jac_V0(Phi_j_new, V0j_new)))
    reg_t = 1e-10 * np.maximum(np.abs(np.diag(J_new)), 1e-10)
    try:
        t_Phi = np.linalg.solve(J_new + np.diag(reg_t), -FV_new)
        t_V0 = 1.0
        norm_t = np.sqrt(np.dot(t_Phi, t_Phi) + t_V0**2)
        t_Phi /= norm_t
        t_V0 /= norm_t
        # Ensure consistent orientation
        dot = np.dot(t_Phi, tangent_Phi) + t_V0 * tangent_V0
        if dot < 0:
            t_Phi = -t_Phi
            t_V0 = -t_V0
        tangent_Phi = t_Phi
        tangent_V0 = t_V0
    except np.linalg.LinAlgError:
        # Fallback to secant
        new_tangent_Phi = Phi_new - Phi_curr
        new_tangent_V0 = V0_new - V0_curr
        norm_t = np.sqrt(np.dot(new_tangent_Phi, new_tangent_Phi) + new_tangent_V0**2)
        if norm_t > 1e-15:
            tangent_Phi = new_tangent_Phi / norm_t
            tangent_V0 = new_tangent_V0 / norm_t

    # Track V0 maximum (fold point)
    if V0_new > V0_max:
        V0_max = V0_new
        V0_max_step = step

    # Detect fold: V0 started decreasing
    turned = V0_new < V0_curr and step > 2

    history.append((V0_new, Phi_new.copy(), min_N))

    marker = " <-- FOLD" if turned and step == V0_max_step + 1 else ""
    print(f"{step:4d}  {V0_new:10.6f}  {min_N:10.6f}  {tangent_V0:10.6f}  "
          f"{aug_res:10.3e}  {'Y' if conv else 'N':>5s}  {ds:8.5f}{marker}",
          flush=True)

    # Advance
    Phi_prev = Phi_curr.copy()
    V0_prev = V0_curr
    Phi_curr = Phi_new.copy()
    V0_curr = V0_new

    # Adaptive step size
    if nit < 5:
        ds = min(ds * 1.2, 0.02)
    elif nit > 15:
        ds *= 0.7

    # If we've gone well past the fold (V0 decreased significantly), we can stop
    if V0_max - V0_curr > 0.002:
        print(f"\n  V0 has decreased {V0_max - V0_curr:.4f} past the fold. Stopping.")
        break

# Summary
print("\n" + "=" * 70)
print("PSEUDO-ARCLENGTH CONTINUATION SUMMARY")
print("=" * 70)

V0_vals = [h[0] for h in history]
minN_vals = [h[2] for h in history]

V0_fold = max(V0_vals)
fold_idx = V0_vals.index(V0_fold)
print(f"V0_fold (maximum V0 on branch) = {V0_fold:.8f}")
print(f"min(N) at fold = {minN_vals[fold_idx]:.6f}")
print(f"Step at fold = {fold_idx}")

# Did V0 turn around?
if any(v < V0_fold - 1e-6 for v in V0_vals[fold_idx+1:]):
    print("\n*** V0 TURNED AROUND — fold bifurcation CONFIRMED ***")
    print("The branch continues past the fold onto the unstable side.")
    print("This is a TRUE saddle-node bifurcation, not a numerical artifact.")
else:
    print("\nV0 did not clearly turn around. May need more steps or smaller ds.")

# Save
np.savez("pseudoarc_fold.npz",
         V0s=np.array(V0_vals),
         min_Ns=np.array(minN_vals),
         V0_fold=V0_fold,
         min_N_fold=minN_vals[fold_idx])

print(f"\nData saved to pseudoarc_fold.npz")

# Print the branch
print(f"\n{'step':>4s}  {'V0':>10s}  {'min(N)':>10s}")
for i, (v0, _, mn) in enumerate(history):
    marker = " <-- fold" if i == fold_idx else ""
    print(f"{i:4d}  {v0:10.6f}  {mn:10.6f}{marker}")
