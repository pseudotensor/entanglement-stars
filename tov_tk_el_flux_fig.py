#!/usr/bin/env python3
"""
Generate Figure tov-flux from EL chemical potential scan data.

3-panel figure:
  (a) w-flux J_w = g·Δw in exterior (should be constant for EL)
  (b) Co-metric w = N² vs Schwarzschild 1 - r_s(1/r - 1/R)
  (c) Co-metric residual w - [1 - r_s(1/r - 1/R)]

Loads data from tov_tk_chemical_el.npz.
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Load EL data
d = np.load("tov_tk_chemical_el.npz", allow_pickle=True)
beta0 = float(d["beta0"])
m0 = float(d["m0"])
csq = float(d["csq"])
N = int(d["N"])
mu_values = d["mu_values"]

n_arr = np.arange(1, N + 1, dtype=np.float64)
g = 4.0 * np.pi * n_arr**2
t0 = 1.0

# Select μ values to plot (3 representative: low, mid, high compactness)
mu_show = [0.1, 0.5, 1.0]
if 0.7 in mu_values:
    mu_show = [0.1, 0.5, 0.7]
colors = {mu: c for mu, c in zip(mu_show, ['black', 'tab:red', 'tab:blue'])}
m0 = float(d["m0"])

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for mu in mu_show:
    key = f"mu{mu:.2f}"
    Phi = d[f"{key}_Phi"]
    lapse = d[f"{key}_lapse"]
    kappa_kms = d[f"{key}_kappa_kms"]
    V_defect = d[f"{key}_V_defect"]

    n_eff = int(np.sum(V_defect > 1e-10))
    c = colors[mu]

    # w = N²
    w = lapse**2

    # w-flux: J_w = g · Δw  (flat Dirichlet flux for w)
    dw = w[:-1] - w[1:]
    J_w = g[:-1] * t0**2 * dw

    # w-fit: w = 1 - r_s/r in exterior
    n_start = max(n_eff + 3, 5)
    n_end = min(N // 2, n_eff + 40)
    idx = np.arange(n_start - 1, n_end)
    r_fit = n_arr[idx]
    basis_1r = 1.0 / r_fit - 1.0 / N
    target_w = -(w[idx] - 1.0)
    r_s_w = np.dot(basis_1r, target_w) / np.dot(basis_1r, basis_1r)

    mu_rat = mu / m0
    # Panel (a): w-flux
    ext_start = max(n_eff, 2)
    axes[0].plot(n_arr[ext_start:-1], J_w[ext_start:], color=c, lw=1.5,
                 label=rf'$\mu/m_0={mu_rat:.2f}$ ($n_{{\rm eff}}={n_eff}$)')

    # Panel (b): w vs Schwarzschild
    n_ext = np.arange(max(n_eff - 1, 1), min(n_eff + 50, N))
    r_ext = n_arr[n_ext]
    w_schw = 1.0 - r_s_w * (1.0 / r_ext - 1.0 / N)
    axes[1].plot(r_ext, w[n_ext], color=c, lw=2,
                 label=rf'$\mu/m_0={mu_rat:.2f}$')
    axes[1].plot(r_ext, w_schw, color=c, ls='--', lw=1, alpha=0.5)

    # Panel (c): residual
    n_res = np.arange(max(n_eff + 2, 3), min(n_eff + 60, N))
    r_res = n_arr[n_res]
    w_schw_res = 1.0 - r_s_w * (1.0 / r_res - 1.0 / N)
    axes[2].plot(r_res, w[n_res] - w_schw_res, color=c, lw=1.5,
                 label=rf'$\mu/m_0={mu_rat:.2f}$')

# Panel formatting
axes[0].set_title(r'(a) $w$-flux $J_w = g\,\Delta w$ (exterior)')
axes[0].set_ylabel(r'$J_w$')
axes[0].set_xlabel('$n$ (shell)')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(0, 60)

axes[1].set_title(r'(b) Co-metric $w = N^2$ (solid) vs.\ $1 - r_s(1/r - 1/R)$ (dashed)')
axes[1].set_ylabel(r'$w = N^2$')
axes[1].set_xlabel('$n$ (shell)')
axes[1].axhline(1, color='gray', ls='--', lw=0.5)
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(0, 60)

axes[2].axhline(0, color='gray', ls='--', lw=0.5)
axes[2].set_title(r'(c) Co-metric residual $w - [1 - r_s(1/r - 1/R)]$')
axes[2].set_ylabel('Residual')
axes[2].set_xlabel('$n$ (shell)')
axes[2].legend(fontsize=8)
axes[2].grid(True, alpha=0.3)
axes[2].set_xlim(0, 100)

plt.suptitle(rf'Exterior diagnostics (full EL, $\beta_0={beta0:.0f}$, $N_s={N}$)',
             fontsize=14)
plt.tight_layout()
plt.savefig("fermion/numerical/figures/tov_tk_final_flux.png", dpi=150)
plt.close()
print("Saved fermion/numerical/figures/tov_tk_final_flux.png")

# Also print diagnostic table
print(f"\nFlux constancy diagnostics:")
print(f"  {'μ/m₀':>6s} {'n_eff':>6s} {'r_s(w)':>8s} {'r*/r_s':>8s} {'δJ_w':>10s}")
for mu in mu_values:
    key = f"mu{mu:.2f}"
    try:
        Phi = d[f"{key}_Phi"]
        lapse = d[f"{key}_lapse"]
        V_defect = d[f"{key}_V_defect"]
        n_eff = int(np.sum(V_defect > 1e-10))

        w = lapse**2
        dw = w[:-1] - w[1:]
        J_w = g[:-1] * t0**2 * dw

        n_start = max(n_eff + 3, 5)
        n_end = min(N // 2, n_eff + 40)
        idx = np.arange(n_start - 1, n_end)
        r_fit = n_arr[idx]
        basis_1r = 1.0 / r_fit - 1.0 / N
        target_w = -(w[idx] - 1.0)
        r_s_w = np.dot(basis_1r, target_w) / np.dot(basis_1r, basis_1r)

        f_start = min(n_eff + 5, N - 2)
        f_end = min(N // 2, n_eff + 40)
        J_w_ext = J_w[f_start:f_end]
        J_w_mean = np.mean(J_w_ext)
        J_w_var = (np.max(np.abs(J_w_ext - J_w_mean)) / abs(J_w_mean)
                   if abs(J_w_mean) > 1e-30 and len(J_w_ext) > 1 else 0)

        compact = float(n_eff) / r_s_w if r_s_w > 0 else float('inf')
        print(f"  {mu/m0:6.3f} {n_eff:6d} {r_s_w:8.3f} {compact:8.2f} {J_w_var:10.6f}")
    except KeyError:
        pass
