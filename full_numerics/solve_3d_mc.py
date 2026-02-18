#!/usr/bin/env python3
"""
3D cubic lattice: Monte Carlo + 1/r ansatz investigation.

Two questions:
1. If we FIX the shape to 1/r and vary amplitude, does the full residual have a zero?
2. Simple MC in Phi-space: does stochastic sampling find non-trivial solutions?
"""
import sys, os
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.chdir(_REPO)
import numpy as np
from scipy.optimize import least_squares, minimize_scalar
import time
import torch

from full_numerics.solve_3d import CubicLattice3D, ThreeDModel

DEVICE = torch.device('cuda:0')
torch.set_default_dtype(torch.float64)

R_MAX = 10
t0_val = 1.0; cstar_sq = 0.5; n_core = 3

print("=" * 90, flush=True)
print(f"3D INVESTIGATION: 1/r ANSATZ + MC (R_max={R_MAX})", flush=True)
print("=" * 90, flush=True)

lat = CubicLattice3D(R_MAX)
N_shell = lat.N_shell
r_shells = np.arange(1, N_shell + 1, dtype=float)

# ═══════════════════════════════════════════════════════════════════
# PART 1: 1/r ansatz — fix shape, scan amplitude
# Φ(r) = -A/r for r≥1, with Φ(R_max)=0 (shifted)
# ═══════════════════════════════════════════════════════════════════

def make_1_over_r(A, N_shell):
    """1/r potential: Φ_n = -A*(1/r_n - 1/R_max)"""
    r = np.arange(1, N_shell + 1, dtype=float)
    Phi = -A * (1.0/r - 1.0/r[-1])
    Phi[-1] = 0.0
    return Phi

for bt0 in [0.10, 0.50, 1.00, 2.11]:
    beta0 = bt0 / t0_val
    V0 = 0.003 / beta0

    print(f"\n{'─'*90}", flush=True)
    print(f"bt0={bt0} — 1/r ansatz amplitude scan", flush=True)
    print(f"{'─'*90}", flush=True)

    model = ThreeDModel(lat, t0=t0_val, V0=V0, n_core=n_core,
                        beta0=beta0, cstar_sq=cstar_sq)

    # Scan amplitudes
    print(f"{'A':>8s} {'|F|':>12s} {'F[0]':>12s} {'F[N/2]':>12s} {'min_N':>8s}", flush=True)
    for A in [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
        Phi = make_1_over_r(A, N_shell)
        # Check lapse stays positive
        lapse = 1.0 + Phi / cstar_sq
        if lapse.min() < 0.01:
            print(f"{A:8.3f} {'skip (lapse<0)':>12s}", flush=True)
            continue
        F = model.residual(Phi)
        res = np.max(np.abs(F[:-1]))
        print(f"{A:8.3f} {res:12.4e} {F[0]:12.4e} {F[N_shell//2]:12.4e} {lapse.min():8.4f}", flush=True)

    # Find optimal amplitude
    def res_vs_A(A):
        if A < 0: return 1e10
        Phi = make_1_over_r(A, N_shell)
        lapse = 1.0 + Phi / cstar_sq
        if lapse.min() < 0.01: return 1e10
        F = model.residual(Phi)
        return np.sum(F[:-1]**2)

    result = minimize_scalar(res_vs_A, bounds=(0, 0.5), method='bounded')
    A_opt = result.x
    Phi_opt = make_1_over_r(A_opt, N_shell)
    F_opt = model.residual(Phi_opt)
    res_opt = np.max(np.abs(F_opt[:-1]))
    lapse_opt = 1.0 + Phi_opt / cstar_sq
    print(f"  Optimal 1/r: A={A_opt:.6f}, |F|={res_opt:.4e}, min_N={lapse_opt.min():.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════
# PART 2: Proxy residual evaluated on 1/r
# If the proxy equation has a 1/r solution, it should show F≈0.
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*90}", flush=True)
print("PROXY residual on 1/r ansatz", flush=True)
print(f"{'='*90}", flush=True)

for bt0 in [0.10, 0.50, 1.00, 2.11]:
    beta0 = bt0 / t0_val
    V0 = 0.003 / beta0
    model = ThreeDModel(lat, t0=t0_val, V0=V0, n_core=n_core,
                        beta0=beta0, cstar_sq=cstar_sq)

    def proxy_res_vs_A(A):
        if A < 0: return 1e10
        Phi = make_1_over_r(A, N_shell)
        lapse = 1.0 + Phi / cstar_sq
        if lapse.min() < 0.01: return 1e10
        F = model.residual_proxy(Phi)
        return np.sum(F[:-1]**2)

    result = minimize_scalar(proxy_res_vs_A, bounds=(0, 0.5), method='bounded')
    A_opt = result.x
    Phi_opt = make_1_over_r(A_opt, N_shell)
    F_opt = model.residual_proxy(Phi_opt)
    res_opt = np.max(np.abs(F_opt[:-1]))
    lapse_opt = 1.0 + Phi_opt / cstar_sq
    print(f"  bt0={bt0}: proxy optimal A={A_opt:.6f}, |F|={res_opt:.4e}, min_N={lapse_opt.min():.6f}", flush=True)

# ═══════════════════════════════════════════════════════════════════
# PART 3: Simple MC in Phi-space at bt0=2.11
# Sample Phi ~ exp(-||F(Phi)||^2 / 2T_mc)
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*90}", flush=True)
print("MC sampling in Phi-space (bt0=2.11)", flush=True)
print(f"{'='*90}", flush=True)

bt0 = 2.11; beta0 = bt0 / t0_val; V0 = 0.003 / beta0
model = ThreeDModel(lat, t0=t0_val, V0=V0, n_core=n_core,
                    beta0=beta0, cstar_sq=cstar_sq)

# MC parameters
T_mc = 1e-6  # MC "temperature" for chi-squared
step_size = 0.001
n_mc = 500
n_burn = 100

np.random.seed(42)
Phi_curr = np.zeros(N_shell)
F_curr = model.residual(Phi_curr)
chi2_curr = np.sum(F_curr[:-1]**2)

best_phi = Phi_curr.copy()
best_chi2 = chi2_curr
accepted = 0

print(f"  Initial chi2 = {chi2_curr:.6e}", flush=True)
print(f"  MC: {n_mc} steps, step_size={step_size}, T_mc={T_mc:.1e}", flush=True)

t_s = time.time()
for step in range(n_mc):
    # Propose: random perturbation
    dPhi = np.random.randn(N_shell) * step_size
    dPhi[-1] = 0.0  # boundary condition

    Phi_prop = Phi_curr + dPhi
    # Clip to keep lapse > 0
    lapse = 1.0 + Phi_prop / cstar_sq
    if lapse.min() < 0.01:
        continue

    F_prop = model.residual(Phi_prop)
    chi2_prop = np.sum(F_prop[:-1]**2)

    # Metropolis acceptance
    dchi2 = chi2_prop - chi2_curr
    if dchi2 < 0 or np.random.rand() < np.exp(-dchi2 / (2 * T_mc)):
        Phi_curr = Phi_prop
        chi2_curr = chi2_prop
        F_curr = F_prop
        accepted += 1

        if chi2_curr < best_chi2:
            best_chi2 = chi2_curr
            best_phi = Phi_curr.copy()

    if (step + 1) % 50 == 0:
        lapse_b = 1.0 + best_phi / cstar_sq
        print(f"  step {step+1}: chi2_curr={chi2_curr:.4e}, best={best_chi2:.4e}, "
              f"acc={accepted}/{step+1}, min_N_best={lapse_b.min():.6f}", flush=True)

elapsed_mc = time.time() - t_s
lapse_best = 1.0 + best_phi / cstar_sq
F_best = model.residual(best_phi)
res_best = np.max(np.abs(F_best[:-1]))
print(f"\n  MC finished in {elapsed_mc:.0f}s", flush=True)
print(f"  Best chi2 = {best_chi2:.6e}, |F| = {res_best:.4e}", flush=True)
print(f"  Best lapse: {' '.join(f'{lapse_best[i]:.6f}' for i in range(N_shell))}", flush=True)

# Also try MC starting from 1/r seed
print(f"\n  MC from 1/r seed (A=0.1):", flush=True)
Phi_curr = make_1_over_r(0.1, N_shell)
F_curr = model.residual(Phi_curr)
chi2_curr = np.sum(F_curr[:-1]**2)
best_phi2 = Phi_curr.copy()
best_chi2_2 = chi2_curr
accepted = 0
step_size = 0.005

for step in range(n_mc):
    dPhi = np.random.randn(N_shell) * step_size
    dPhi[-1] = 0.0
    Phi_prop = Phi_curr + dPhi
    lapse = 1.0 + Phi_prop / cstar_sq
    if lapse.min() < 0.01:
        continue
    F_prop = model.residual(Phi_prop)
    chi2_prop = np.sum(F_prop[:-1]**2)
    dchi2 = chi2_prop - chi2_curr
    if dchi2 < 0 or np.random.rand() < np.exp(-dchi2 / (2 * T_mc)):
        Phi_curr = Phi_prop
        chi2_curr = chi2_prop
        accepted += 1
        if chi2_curr < best_chi2_2:
            best_chi2_2 = chi2_curr
            best_phi2 = Phi_curr.copy()
    if (step + 1) % 50 == 0:
        lapse_b = 1.0 + best_phi2 / cstar_sq
        print(f"  step {step+1}: chi2_curr={chi2_curr:.4e}, best={best_chi2_2:.4e}, "
              f"acc={accepted}/{step+1}, min_N_best={lapse_b.min():.6f}", flush=True)

lapse_best2 = 1.0 + best_phi2 / cstar_sq
print(f"  Best chi2 (from 1/r) = {best_chi2_2:.6e}", flush=True)
print(f"  Best lapse: {' '.join(f'{lapse_best2[i]:.6f}' for i in range(N_shell))}", flush=True)

print("\nDone.", flush=True)
