#!/usr/bin/env python3
"""
Quasi-static Page curve from self-consistent Gaussian states.

DATA script — saves to fermion/numerical/data/evaporation_selfconsistent.npz.
Figure script: evaporation_selfconsistent_fig.py

Two computations
----------------
Part A — V0 scan (quasi-static evaporation sequence):
  For each V0, solve the self-consistent closure and record rs(V0).
  The Bekenstein-Hawking entropy is:
      S_BH(V0) = alpha_S * 4*pi * rs(V0)^2 / a^2
  where alpha_S is determined in Part B.  This is the capacity S_max(t)
  for the kinematic Page curve bound:
      S_R(t) <= min{ S_max(0) - S_max(t),  S_max(t) }
  Mapping V0 -> rs -> M -> t via M(t) = M0*(1-t/t_evap)^{1/3} and
  rs = 2GM/c*^2 gives the full Page curve from closure dynamics alone.

Part B — area-law scan at fixed V0:
  At a single V0, vary the bipartition cut n_cut from 1..N_cut.
  The total MI I_total(n_cut) = 4*pi*n_cut^2 * I^(1)(A:B) is linear in
  n_cut^2 for large n_cut (exterior region, away from the core), verifying
  the area law S_ent ∝ A_H / a^2.  The saturation slope gives alpha_S.

Key point: both S_max and alpha_S are derived entirely from the
self-consistent thermal states sigma[Phi] produced by the closure.
No beam-splitter model or ad hoc circuit is needed.

Output
------
fermion/numerical/data/evaporation_selfconsistent.npz

Run
---
python3 full_numerics/evaporation_selfconsistent.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
from scipy.linalg import eigh_tridiagonal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from full_numerics.solve_twostate_exact import solve_exact

# ── Parameters ────────────────────────────────────────────────────────────────
N = 200
t0 = 1.0
n_core = 5
beta0 = 0.1
cstar_sq = 0.5
a = 1.0

# Part A: V0 scan (increasing, for warm-starts)
V0_scan = np.concatenate([
    np.linspace(0.002, 0.010, 5),
    np.linspace(0.010, 0.060, 18),
    np.linspace(0.060, 0.066, 5),
])
V0_scan = np.unique(V0_scan)

# Part B: area-law scan at fixed V0 near the fold
V0_area = 0.06
n_cut_max = 80    # scan cuts n_cut = 1..n_cut_max
n_cut_fit_min = 20  # use only exterior region (n >= n_cut_fit_min) for slope


# ── Helpers ───────────────────────────────────────────────────────────────────

def _h(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.clip(x, eps, 1.0 - eps)
    return -x * np.log(x) - (1.0 - x) * np.log(1.0 - x)


def smeared_corr_full(Phi: np.ndarray) -> tuple:
    """Full N×N correlation matrix for lapse-smeared background Hamiltonian.

    G = (exp(beta0 * h_smear[Phi]) + I)^{-1},
    h_smear: diag=0, off = -t0 * |Nbar|.

    This is sigma[Phi] — the reconstructed state in the closure language.
    Returns (G, f) where f are Fermi occupations.
    """
    lapse = 1.0 + Phi / cstar_sq
    Nbar = 0.5 * (lapse[:-1] + lapse[1:])
    diag = np.zeros(len(Phi))
    off = -t0 * np.abs(Nbar)
    evals, evecs = eigh_tridiagonal(diag, off)
    beta_e = np.clip(beta0 * evals, -500, 500)
    f = 1.0 / (np.exp(beta_e) + 1.0)
    G = (evecs * f[None, :]) @ evecs.T
    return G, f


def bipartition_MI(G: np.ndarray, f: np.ndarray, n_cut: int) -> float:
    """Single-channel I^(1)(A:B) for bipartition at n_cut."""
    nu_A = np.linalg.eigvalsh(G[:n_cut, :n_cut])
    nu_B = np.linalg.eigvalsh(G[n_cut:, n_cut:])
    S_A = float(np.sum(_h(nu_A)))
    S_B = float(np.sum(_h(nu_B)))
    S_AB = float(np.sum(_h(f)))
    return S_A + S_B - S_AB


def subsystem_MI(G: np.ndarray, idx_A: list, idx_B: list) -> float:
    """I(A:B) = S_A + S_B - S_AB for arbitrary (non-overlapping) index subsets."""
    idx_AB = sorted(set(idx_A) | set(idx_B))
    nu_A = np.linalg.eigvalsh(G[np.ix_(idx_A, idx_A)])
    nu_B = np.linalg.eigvalsh(G[np.ix_(idx_B, idx_B)])
    nu_AB = np.linalg.eigvalsh(G[np.ix_(idx_AB, idx_AB)])
    return float(np.sum(_h(nu_A)) + np.sum(_h(nu_B)) - np.sum(_h(nu_AB)))


# ── Part B: area-law scan ─────────────────────────────────────────────────────
# Run Part B first so alpha_S is available for Part A.
print("=" * 70)
print(f"Part B: area-law cut scan at V0={V0_area:.4f}")
print(f"  N={N}, beta0={beta0}, cuts 1..{n_cut_max}")
print("=" * 70)

res_area = solve_exact(N=N, t0=t0, V0=V0_area, n_core=n_core, beta0=beta0,
                       cstar_sq=cstar_sq, tol=1e-8, lapse_floor=0.01,
                       Phi_seed=None, verbose=False)

cuts = np.arange(1, n_cut_max + 1)
I1_cuts = np.zeros(len(cuts))
I_total_cuts = np.zeros(len(cuts))
g_cuts = 4.0 * np.pi * cuts.astype(float) ** 2
alpha_S_sat = np.nan
rs_area = np.nan

if res_area["conv_exact"]:
    Phi_area = res_area["Phi_exact"]
    rs_area = res_area["rs_exact"]
    G_area, f_area = smeared_corr_full(Phi_area)
    for j, nc in enumerate(cuts):
        I1_cuts[j] = bipartition_MI(G_area, f_area, int(nc))
        I_total_cuts[j] = g_cuts[j] * I1_cuts[j]

    # Extract alpha_S from the slope of I_total vs n_cut^2 in the exterior region.
    # In the exterior (n_cut >> r_s/a), the lapse -> 1 and correlations follow the
    # area law: I_total = 4*pi*alpha_S_sat * n_cut^2 + const.
    # Fit through the large-n_cut saturation regime (n_cut >= n_cut_fit_min).
    mask = cuts >= n_cut_fit_min
    n_sq_fit = cuts[mask] ** 2
    # Fit: I_total = slope * n_cut^2  (force through origin for pure area law)
    slope = (float(np.dot(n_sq_fit, I_total_cuts[mask]))
             / float(np.dot(n_sq_fit, n_sq_fit)))
    # I_total = 4*pi*n_cut^2 * I1  =>  slope_of_I_total = 4*pi * alpha_S_sat * 2
    # Actually: I_total = g_{n_cut} * I1 = 4*pi*n_cut^2 * I1
    # and S_BH = alpha_S * A_H / a^2 = alpha_S * 4*pi*rs^2
    # with alpha_S = (1/2) * I1_bond at the stretched horizon (order-1 factor).
    # Here we use: alpha_S_sat = slope / (4*pi) * (1/2) ...
    # Actually I_total / (4*pi*n_cut^2) = I1 = 2*alpha_S  =>  alpha_S = I1/2
    # And the slope we fit is I_total / n_cut^2 = 4*pi * I1 = 8*pi * alpha_S.
    alpha_S_sat = slope / (8.0 * np.pi)
    print(f"  rs_area={rs_area:.4f}")
    print(f"  Slope of I_total vs n_cut^2: {slope:.6f}")
    print(f"  alpha_S (saturated, exterior): {alpha_S_sat:.6f}")
    print(f"  I_total range: [{I_total_cuts.min():.5f}, {I_total_cuts.max():.4f}]")
else:
    print(f"  WARNING: Part B not converged at V0={V0_area:.4f}")

# ── Part A: V0 scan ───────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Part A: V0 scan (quasi-static evaporation sequence)")
print(f"  N={N}, beta0={beta0}, n_core={n_core}")
print(f"  {len(V0_scan)} V0 values; alpha_S_sat={alpha_S_sat:.6f}")
print("=" * 70)

scan_results: list[dict] = []
Phi_seed = None

t_start = time.time()
for i, V0 in enumerate(V0_scan):
    print(f"\n[{i+1:2d}/{len(V0_scan)}] V0={V0:.4f}", flush=True)
    res = solve_exact(N=N, t0=t0, V0=V0, n_core=n_core, beta0=beta0,
                      cstar_sq=cstar_sq, tol=1e-8, lapse_floor=0.01,
                      Phi_seed=Phi_seed, verbose=False)
    if not res["conv_exact"]:
        print(f"  WARNING: not converged, skipping")
        continue
    Phi = res["Phi_exact"]
    rs = res["rs_exact"]
    Phi_seed = Phi.copy()

    # S_BH = alpha_S_sat * 4*pi * rs^2 / a^2   (Bekenstein-Hawking from closure)
    S_BH = alpha_S_sat * 4.0 * np.pi * (rs / a) ** 2

    # Compute correlation matrix once for all MI quantities
    G, f = smeared_corr_full(Phi)

    # Per-channel alpha_S at core boundary for reference
    I1_core = bipartition_MI(G, f, n_core)
    alpha_S_core = I1_core / (2.0 * np.log(2.0))

    # AMPS-like correlations:
    # cap shell = shell closest to rs (boundary between star and radiation)
    n_cap = max(1, min(int(round(rs / a)), N - 2))
    idx_cap   = [n_cap]
    idx_star  = list(range(n_cap))           # inner shells (star)
    idx_outer = list(range(n_cap + 1, N))    # outer shells (radiation surrogate)

    I_cap_star  = subsystem_MI(G, idx_cap, idx_star)  if idx_star  else 0.0
    I_cap_outer = subsystem_MI(G, idx_cap, idx_outer) if idx_outer else 0.0

    print(f"  rs={rs:.4f}  n_cap={n_cap}  S_BH={S_BH:.5f}  "
          f"I_cap_star={I_cap_star:.6f}  I_cap_outer={I_cap_outer:.6f}  "
          f"min(N)={res['min_N_exact']:.5f}")

    scan_results.append({
        "V0": V0, "rs": rs, "min_N": res["min_N_exact"],
        "S_BH": S_BH, "I1_core": I1_core, "alpha_S_core": alpha_S_core,
        "n_cap": n_cap,
        "I_cap_star": I_cap_star,
        "I_cap_outer": I_cap_outer,
    })

dt = time.time() - t_start
print(f"\nPart A done: {len(scan_results)} solutions in {dt:.0f}s")

# ── Derived evaporation quantities ────────────────────────────────────────────

V0_arr = np.array([r["V0"] for r in scan_results])
rs_arr = np.array([r["rs"] for r in scan_results])
S_BH_arr = np.array([r["S_BH"] for r in scan_results])
alpha_S_core_arr = np.array([r["alpha_S_core"] for r in scan_results])
n_cap_arr = np.array([r["n_cap"] for r in scan_results])
I_cap_star_arr  = np.array([r["I_cap_star"]  for r in scan_results])
I_cap_outer_arr = np.array([r["I_cap_outer"] for r in scan_results])

# Evaporation time: t/t_evap = 1 - (M/M0)^3  where M ∝ rs
# t=0: heaviest star (largest rs, largest S_BH)
# t→1: fully evaporated
rs_max = rs_arr.max()
t_norm = 1.0 - (rs_arr / rs_max) ** 3

# S_max(t) = S_BH(t) from Bekenstein-Hawking formula
S_max = S_BH_arr
S_max_0 = S_max[np.argmax(rs_arr)]

# Kinematic Page curve bound (eq. ev-page-curve in paper)
S_page_early = S_max_0 - S_max   # early branch: emitted entropy
S_page_late = S_max               # late branch: remaining star capacity
S_page_bound = np.minimum(S_page_early, S_page_late)

# Page time: S_max_0 - S = S  =>  S = S_max_0/2  =>  rs = rs_max/sqrt(2)
# t_Page/t_evap = 1 - (1/sqrt(2))^3 = 1 - 2^{-3/2} ≈ 0.646
t_page = 1.0 - 2.0 ** (-1.5)
print(f"\nPage time: t_Page/t_evap = {t_page:.4f}")
print(f"S_max(0) = S_BH(rs_max={rs_max:.4f}) = {S_max_0:.5f}")
print(f"alpha_S_sat = {alpha_S_sat:.6f}")

# ── Save ──────────────────────────────────────────────────────────────────────

DATADIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "fermion", "numerical", "data",
)
os.makedirs(DATADIR, exist_ok=True)

outpath = os.path.join(DATADIR, "evaporation_selfconsistent.npz")
np.savez_compressed(outpath,
    # Parameters
    N=N, t0=t0, n_core=n_core, beta0=beta0, cstar_sq=cstar_sq, a=a,
    # Part A
    V0=V0_arr, rs=rs_arr,
    S_BH=S_BH_arr, alpha_S_core=alpha_S_core_arr,
    n_cap=n_cap_arr, I_cap_star=I_cap_star_arr, I_cap_outer=I_cap_outer_arr,
    t_norm=t_norm, S_max=S_max, S_max_0=S_max_0, rs_max=rs_max,
    S_page_bound=S_page_bound, S_page_early=S_page_early, S_page_late=S_page_late,
    t_page=t_page,
    # Part B
    alpha_S_sat=alpha_S_sat, V0_area=V0_area, rs_area=rs_area,
    cuts=cuts, I1_cuts=I1_cuts, I_total_cuts=I_total_cuts, g_cuts=g_cuts,
    n_cut_fit_min=n_cut_fit_min,
)
print(f"\nSaved: {outpath}")
print(f"  V0:       [{V0_arr.min():.4f}, {V0_arr.max():.4f}]")
print(f"  rs:       [{rs_arr.min():.4f}, {rs_arr.max():.4f}]")
print(f"  S_BH:     [{S_BH_arr.min():.5f}, {S_BH_arr.max():.5f}]")
print(f"  alpha_S:  {alpha_S_sat:.6f}")
print(f"  t_norm:   [{t_norm.min():.4f}, {t_norm.max():.4f}]")
print(f"Total: {time.time()-t_start:.0f}s")
