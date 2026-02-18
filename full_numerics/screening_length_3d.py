#!/usr/bin/env python3
"""
Compute the linear response ∂ρ_σ/∂Φ at Φ=0 in 3D.

The linearized closure equation at large r:
  κ₀ ∇²Φ = (β₀/c*²) × χ × Φ  +  source

where χ = ∂ρ_σ/∂Φ|_0 is the susceptibility.

If χ > 0: screening, Φ decays as exp(-r/λ) with λ = √(κ₀ c*² / (β₀ χ))
If χ = 0: no screening, Φ ∝ 1/r at large r
If χ < 0: anti-screening, Φ grows (unstable)

This determines whether 1/r tails can survive.
"""
import sys, os
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.chdir(_REPO)
import numpy as np
import time
import torch

from full_numerics.solve_3d import CubicLattice3D, ThreeDModel

DEVICE = torch.device('cuda:0')
torch.set_default_dtype(torch.float64)

R_MAX = 10
t0_val = 1.0; cstar_sq = 0.5; n_core = 3

print("=" * 90, flush=True)
print(f"LINEAR RESPONSE / SCREENING LENGTH IN 3D (R_max={R_MAX})", flush=True)
print("=" * 90, flush=True)

lat = CubicLattice3D(R_MAX)
N_shell = lat.N_shell

print(f"\n{'bt0':>6s} {'kappa_0':>10s} {'chi_avg':>12s} {'lambda_TF':>10s} "
      f"{'lambda/a':>10s} {'screened?':>10s}", flush=True)
print("─" * 70, flush=True)

for bt0 in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 1.00, 1.50, 2.00, 2.11]:
    beta0 = bt0 / t0_val
    V0 = 0.003 / beta0
    model = ThreeDModel(lat, t0=t0_val, V0=V0, n_core=n_core,
                        beta0=beta0, cstar_sq=cstar_sq)

    # Flat-space conductances
    kappa_0 = model._conductances_only(np.zeros(N_shell))

    # Numerical susceptibility: χ_n = (ρ_σ(Φ+dΦ) - ρ_σ(Φ-dΦ)) / (2 dΦ_n)
    # Apply uniform dΦ to all shells to get average χ
    eps = 1e-6
    Phi_plus = np.full(N_shell, eps); Phi_plus[-1] = 0
    Phi_minus = np.full(N_shell, -eps); Phi_minus[-1] = 0

    rho_plus = model._energy_profile_np(Phi_plus, smeared=True, add_V0=False)
    rho_minus = model._energy_profile_np(Phi_minus, smeared=True, add_V0=False)

    chi = (rho_plus - rho_minus) / (2.0 * eps)  # per-shell susceptibility
    chi_avg = np.mean(chi[2:-1])  # average over intermediate shells

    # Also compute shell-by-shell susceptibility
    chi_shell = np.zeros(N_shell)
    for n in range(N_shell - 1):
        dPhi = np.zeros(N_shell)
        dPhi[n] = eps
        rp = model._energy_profile_np(dPhi, smeared=True, add_V0=False)
        rm = model._energy_profile_np(-dPhi, smeared=True, add_V0=False)
        chi_shell[n] = (rp[n] - rm[n]) / (2.0 * eps)

    chi_diag = np.mean(chi_shell[2:-1])

    # Screening length: λ = √(κ₀ c*² / (β₀ |χ|))
    kappa_avg = np.mean(kappa_0[2:])
    if abs(chi_diag) > 1e-20:
        lambda_TF = np.sqrt(abs(kappa_avg * cstar_sq / (beta0 * abs(chi_diag))))
    else:
        lambda_TF = float('inf')

    screened = "YES" if chi_diag > 0 else ("ANTI" if chi_diag < 0 else "NO")
    print(f"{bt0:6.2f} {kappa_avg:10.4f} {chi_diag:12.4e} {lambda_TF:10.4f} "
          f"{lambda_TF:10.2f}a {screened:>10s}", flush=True)

    if bt0 in [0.10, 1.00, 2.11]:
        print(f"    chi per shell: {' '.join(f'{chi_shell[n]:.4e}' for n in range(N_shell))}", flush=True)
        print(f"    kappa:         {' '.join(f'{kappa_0[n]:.4f}' for n in range(N_shell-1))}", flush=True)

# ═══════════════════════════════════════════════════════════════════
# Direct check: does a 1/r tail survive the linear equation?
# At large r, F_n ≈ κ₀(Φ_{n-1} + Φ_{n+1} - 2Φ_n) - (β₀/c*²)χ_n Φ_n = 0
# Test: Φ_n = A/n (1/r in shell space)
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*90}", flush=True)
print("Linearized equation test: Φ_n = A/n", flush=True)
print(f"{'='*90}", flush=True)

for bt0 in [0.10, 1.00, 2.11]:
    beta0 = bt0 / t0_val
    V0 = 0.003 / beta0
    model = ThreeDModel(lat, t0=t0_val, V0=V0, n_core=n_core,
                        beta0=beta0, cstar_sq=cstar_sq)
    kappa_0 = model._conductances_only(np.zeros(N_shell))

    # Shell-diagonal susceptibility
    chi_shell = np.zeros(N_shell)
    eps = 1e-6
    for n in range(N_shell - 1):
        dPhi = np.zeros(N_shell)
        dPhi[n] = eps
        rp = model._energy_profile_np(dPhi, smeared=True, add_V0=False)
        rm = model._energy_profile_np(-dPhi, smeared=True, add_V0=False)
        chi_shell[n] = (rp[n] - rm[n]) / (2.0 * eps)

    # Check linearized residual for Φ_n = A/n
    A = 0.001
    r = np.arange(1, N_shell + 1, dtype=float)
    Phi_test = -A / r
    Phi_test[-1] = 0

    # Laplacian term: κ_n (Φ_n - Φ_{n+1}) + κ_{n-1} (Φ_n - Φ_{n-1})
    lap = np.zeros(N_shell)
    for n in range(N_shell):
        if n < N_shell - 1:
            lap[n] += kappa_0[n] * (Phi_test[n] - Phi_test[n+1])
        if n > 0:
            lap[n] += kappa_0[n-1] * (Phi_test[n] - Phi_test[n-1])

    # Screening term
    screen = (beta0 / cstar_sq) * chi_shell * Phi_test

    # Source (frozen, from rho_bg - rho_tgt)
    source = -(beta0 / cstar_sq) * (model.rho_bg - model.rho_tgt)

    print(f"\nbt0={bt0}:", flush=True)
    print(f"  {'n':>3s} {'Φ_n':>10s} {'Laplacian':>12s} {'Screen':>12s} {'Source':>12s} "
          f"{'L-S':>12s} {'L/(L-S)':>10s}", flush=True)
    for n in range(N_shell):
        L_minus_S = lap[n] - screen[n]
        ratio = lap[n] / L_minus_S if abs(L_minus_S) > 1e-20 else float('nan')
        print(f"  {n:3d} {Phi_test[n]:10.6f} {lap[n]:12.4e} {screen[n]:12.4e} "
              f"{source[n]:12.4e} {L_minus_S:12.4e} {ratio:10.4f}", flush=True)

print("\nDone.", flush=True)
