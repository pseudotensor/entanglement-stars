#!/usr/bin/env python3
"""
3D cubic lattice: LM method across all temperatures + continuation.

Key insight: trf found flat solutions at bt0≥0.50, but LM found a DEEP WELL
at bt0=0.10 (min_N=0.942). Is there a deep-well branch at higher bt0?

Strategy:
1. LM from zero at each bt0
2. LM from bt0=0.10 solution (T-continuation forward)
3. LM from proxy seed at each bt0
"""
import sys, os
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.chdir(_REPO)
import numpy as np
from scipy.optimize import least_squares
import time
import torch

from full_numerics.solve_3d import CubicLattice3D, ThreeDModel, print_result

DEVICE = torch.device('cuda:0')
torch.set_default_dtype(torch.float64)

R_MAX = 10
t0_val = 1.0; cstar_sq = 0.5; n_core = 3

print("=" * 90, flush=True)
print(f"3D LM SOLVER ACROSS TEMPERATURES (R_max={R_MAX})", flush=True)
print("=" * 90, flush=True)

lat = CubicLattice3D(R_MAX)
N_shell = lat.N_shell

def solve_lm(model, seed, label, max_nfev=2000):
    """Solve using LM method (unbounded)."""
    t_s = time.time()
    model.n_call = 0
    r = least_squares(model.residual, seed, method='lm',
                     ftol=1e-14, xtol=1e-14, max_nfev=max_nfev)
    F = model.residual(r.x)
    res = np.max(np.abs(F[:-1]))
    elapsed = time.time() - t_s
    print_result(label, r.x, res, model.n_call, elapsed, model)
    return r.x, res

def solve_proxy(model, max_nfev=2000):
    """Solve proxy with trf."""
    lb = np.full(N_shell, -cstar_sq * 5.0)
    ub = np.full(N_shell, cstar_sq * 0.5)
    lb[-1] = -1e-8; ub[-1] = 1e-8
    model.n_call = 0
    r = least_squares(model.residual_proxy, np.zeros(N_shell), method='trf',
                     bounds=(lb, ub), ftol=1e-14, xtol=1e-14, max_nfev=max_nfev)
    F = model.residual_proxy(r.x)
    res = np.max(np.abs(F[:-1]))
    return r.x, res

bt0_list = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00, 1.50, 2.00, 2.11]

# ═══════════════════════════════════════════════════════════════════
# Part 1: LM from zero and proxy at each bt0
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90, flush=True)
print("PART 1: LM from zero + proxy seed at each bt0", flush=True)
print("=" * 90, flush=True)

solutions = {}
for bt0 in bt0_list:
    beta0 = bt0 / t0_val
    V0 = 0.003 / beta0

    print(f"\n{'─'*90}", flush=True)
    print(f"bt0 = {bt0:.2f}, V0 = {V0:.6f}", flush=True)

    model = ThreeDModel(lat, t0=t0_val, V0=V0, n_core=n_core,
                        beta0=beta0, cstar_sq=cstar_sq)

    # Proxy
    Phi_p, res_p = solve_proxy(model)
    t_s = time.time()
    print_result("PROXY", Phi_p, res_p, 0, time.time()-t_s, model)

    # LM from zero
    Phi_lm0, res_lm0 = solve_lm(model, np.zeros(N_shell), "LM/0")

    # LM from proxy
    Phi_lmp, res_lmp = solve_lm(model, Phi_p, "LM/p")

    solutions[bt0] = {
        'proxy': Phi_p, 'lm0': Phi_lm0, 'lmp': Phi_lmp,
        'res_lm0': res_lm0, 'res_lmp': res_lmp,
    }

# ═══════════════════════════════════════════════════════════════════
# Part 2: Temperature continuation from bt0=0.10 deep well
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90, flush=True)
print("PART 2: T-continuation from bt0=0.10 deep well", flush=True)
print("=" * 90, flush=True)

# First get the bt0=0.10 deep well
Phi_deep = solutions[0.10]['lm0'].copy()
print(f"  Starting from bt0=0.10 deep well (min_N={1.0 + Phi_deep.min()/cstar_sq:.4f})", flush=True)

for bt0 in bt0_list[1:]:  # Skip bt0=0.10
    beta0 = bt0 / t0_val
    V0 = 0.003 / beta0

    print(f"\n  bt0 = {bt0:.2f}:", flush=True)
    model = ThreeDModel(lat, t0=t0_val, V0=V0, n_core=n_core,
                        beta0=beta0, cstar_sq=cstar_sq)

    Phi_cont, res_cont = solve_lm(model, Phi_deep, f"CONT/{bt0:.2f}")
    Phi_deep = Phi_cont.copy()  # Use as seed for next bt0

    solutions[bt0]['cont'] = Phi_cont
    solutions[bt0]['res_cont'] = res_cont

# ═══════════════════════════════════════════════════════════════════
# Summary table
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90, flush=True)
print("SUMMARY: min_N for each branch", flush=True)
print("=" * 90, flush=True)
print(f"{'bt0':>6s} {'min_N_proxy':>12s} {'min_N_LM0':>12s} {'min_N_LMp':>12s} "
      f"{'min_N_cont':>12s} {'|F|_LM0':>12s} {'|F|_cont':>12s}", flush=True)

for bt0 in bt0_list:
    d = solutions[bt0]
    def mn(Phi): return 1.0 + Phi.min() / cstar_sq

    cont_minN = mn(d['cont']) if 'cont' in d else float('nan')
    cont_res = d.get('res_cont', float('nan'))

    print(f"{bt0:6.2f} {mn(d['proxy']):12.6f} {mn(d['lm0']):12.6f} {mn(d['lmp']):12.6f} "
          f"{cont_minN:12.6f} {d['res_lm0']:12.2e} {cont_res:12.2e}", flush=True)

# Save
outfile = 'full_numerics/3d_lm_results_R10.npz'
np.savez(outfile, bt0_list=np.array(bt0_list),
         **{f'{bt0:.2f}_{k}': v for bt0, d in solutions.items()
            for k, v in d.items() if isinstance(v, np.ndarray)})
print(f"\nSaved to {outfile}", flush=True)
print("\nDone.", flush=True)
