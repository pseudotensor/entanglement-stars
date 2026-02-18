#!/usr/bin/env python3
"""
Deep investigation of bt0=0.10 in 3D — the one case where the full solver didn't converge.
Try multiple approaches: more nfev, LM method, continuation from proxy.
"""
import sys, os
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.chdir(_REPO)
import numpy as np
from scipy.optimize import least_squares, root
import time
import torch

# Import the 3D solver classes
from full_numerics.solve_3d import CubicLattice3D, ThreeDModel, solve, print_result

DEVICE = torch.device('cuda:0')
torch.set_default_dtype(torch.float64)

R_MAX = 10
t0 = 1.0; cstar_sq = 0.5; n_core = 3
bt0 = 0.10; beta0 = bt0 / t0; V0 = 0.003 / beta0

print("=" * 90, flush=True)
print(f"3D bt0=0.10 DEEP INVESTIGATION (R_max={R_MAX})", flush=True)
print(f"V0={V0:.6f}, β₀V₀=0.003", flush=True)
print("=" * 90, flush=True)

lat = CubicLattice3D(R_MAX)
model = ThreeDModel(lat, t0=t0, V0=V0, n_core=n_core, beta0=beta0, cstar_sq=cstar_sq)

N = model.N_shell
lb = np.full(N, -cstar_sq * 5.0)
ub = np.full(N, cstar_sq * 0.5)
lb[-1] = -1e-8; ub[-1] = 1e-8

# 1. Proxy solution (known to converge)
print("\n--- Proxy ---", flush=True)
t_s = time.time()
Phi_p, _, res_p, nfev_p = solve(model, mode='proxy')
print_result("PROXY", Phi_p, res_p, nfev_p, time.time()-t_s, model)

# 2. Full with many more iterations
print("\n--- Full, max_nfev=2000 ---", flush=True)
t_s = time.time()
model.n_call = 0
r = least_squares(model.residual, np.zeros(N), method='trf', bounds=(lb, ub),
                  ftol=1e-14, xtol=1e-14, max_nfev=2000)
F = model.residual(r.x)
res = np.max(np.abs(F[:-1]))
print_result("FULL/2k", r.x, res, model.n_call, time.time()-t_s, model)

# 3. Full with LM (unbounded, but clamp)
print("\n--- Full, LM method ---", flush=True)
t_s = time.time()
model.n_call = 0
r_lm = least_squares(model.residual, np.zeros(N), method='lm',
                     ftol=1e-14, xtol=1e-14, max_nfev=2000)
F_lm = model.residual(r_lm.x)
res_lm = np.max(np.abs(F_lm[:-1]))
print_result("FULL/LM", r_lm.x, res_lm, model.n_call, time.time()-t_s, model)

# 4. Full from proxy seed, LM
print("\n--- Full from proxy seed, LM ---", flush=True)
t_s = time.time()
model.n_call = 0
r_lm2 = least_squares(model.residual, Phi_p, method='lm',
                      ftol=1e-14, xtol=1e-14, max_nfev=2000)
F_lm2 = model.residual(r_lm2.x)
res_lm2 = np.max(np.abs(F_lm2[:-1]))
print_result("FULL/pLM", r_lm2.x, res_lm2, model.n_call, time.time()-t_s, model)

# 5. V0 continuation: start at large V0 (easy to solve), decrease
print("\n--- V0 continuation ---", flush=True)
V0_values = [0.30, 0.20, 0.10, 0.05, 0.03]
Phi_curr = np.zeros(N)
for V0_i in V0_values:
    t_s = time.time()
    model_i = ThreeDModel(lat, t0=t0, V0=V0_i, n_core=n_core, beta0=beta0, cstar_sq=cstar_sq)
    model_i.n_call = 0
    r_i = least_squares(model_i.residual, Phi_curr, method='trf', bounds=(lb, ub),
                       ftol=1e-14, xtol=1e-14, max_nfev=1000)
    F_i = model_i.residual(r_i.x)
    res_i = np.max(np.abs(F_i[:-1]))
    print_result(f"V0={V0_i:.2f}", r_i.x, res_i, model_i.n_call, time.time()-t_s, model_i)
    Phi_curr = r_i.x

print("\nDone.", flush=True)
