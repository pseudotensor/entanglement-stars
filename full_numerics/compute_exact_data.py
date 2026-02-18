#!/usr/bin/env python3
"""
Compute all exact two-state closure solutions and dump to .npz.

Runs the expensive Newton solvers (FD Jacobian, O(N^3) per step) and saves
comprehensive data so that plot_exact_figures.py can regenerate all 15
publication figures without re-solving.

Phases:
  A/A2/A3 — core/supercritical/proxy+analytic solutions
  B — floor-independence study
  B2 — temperature sweep (bt0-continuation)
  C — V0 sweep (exact + proxy + analytic)

Output: fermion/numerical/data/exact_solutions.npz

Run:  python3 full_numerics/compute_exact_data.py
"""

import numpy as np
import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from full_numerics.physics_twostate_exact import TwoStateExactModel
from full_numerics.physics_twostate import TwoStateShellModel
from full_numerics.solve_twostate_exact import solve_exact, newton_fd_exact
from full_numerics.solve_twostate import solve_full, solve_proxy
from full_numerics.solver import extract_rs


# ── Parameters ────────────────────────────────────────────────────
N = 200
t0 = 1.0
n_core = 5
beta0 = 0.1
cstar_sq = 0.5
a = 1.0

r_arr = a * np.arange(1, N + 1, dtype=float)
r_bond = 0.5 * (r_arr[:-1] + r_arr[1:])


# ── Helpers (copied from generate_exact_figures.py) ───────────────

def extract_kappa_H(lapse, r_arr, rs, cstar_sq, half_width=8):
    """Surface gravity from the derived Schwarzschild radius: kappa_H = c*^2/(2 r_s).

    Matches the paper's convention (eq. sc-surface-gravity) and
    plot_exact_figures.py.  The object is horizonless; r_s is an exterior
    fit parameter, so the polynomial-fit derivative is replaced by the
    analytic Schwarzschild identity |d(N^2)/dr|_{r=r_s} = 1/r_s.
    """
    if rs <= 0:
        return np.nan
    return cstar_sq / (2.0 * rs)


def solve_exact_floored(V0, lapse_floor=0.01, Phi_seed=None,
                        tol=1e-6, verbose=True):
    """Solve exact two-state with proper floor clamping."""
    model_an = TwoStateShellModel(N=N, t0=t0, V0=V0, n_core=n_core,
                                   beta0=beta0, cstar_sq=cstar_sq)
    Phi_an, conv_an, _ = solve_full(model_an, Phi_seed, tol=1e-12,
                                     lapse_floor=lapse_floor, verbose=False)

    model_ex = TwoStateExactModel(N=N, t0=t0, V0=V0, n_core=n_core,
                                   beta0=beta0, cstar_sq=cstar_sq)
    floor_val = -(1.0 - lapse_floor) * cstar_sq
    Phi = Phi_an.copy()

    for it in range(50):
        Phi = np.maximum(Phi, floor_val)
        clamped = Phi <= floor_val + 1e-13

        # Use fixedpoint residual for floored solver: the lapse floor
        # is the dominant approximation, so adding the EL correction to
        # a floor-regularized equation is not meaningful.
        F = model_ex.residual_fixedpoint(Phi)
        F[clamped] = 0.0
        F[-1] = Phi[-1]
        res = np.max(np.abs(F))

        if verbose and (it < 3 or it % 5 == 0 or res < tol):
            print(f"    Newton {it:3d}: |F|={res:.3e}, "
                  f"min(N)={(1+Phi/cstar_sq).min():.6f}, "
                  f"clamped={int(clamped.sum())}",
                  flush=True)

        if res < tol:
            break

        J = np.zeros((N, N))
        eps = 1e-7
        for j in range(N):
            if clamped[j]:
                J[j, j] = 1.0
                continue
            Phi_p = Phi.copy()
            step = eps * max(1.0, abs(Phi[j]))
            Phi_p[j] += step
            F_p = model_ex.residual_fixedpoint(Phi_p)
            F_p[clamped] = 0.0
            F_p[-1] = Phi_p[-1]
            J[:, j] = (F_p - F) / step

        for n in range(N):
            if clamped[n]:
                J[n, :] = 0.0
                J[n, n] = 1.0

        try:
            dPhi = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            if verbose:
                print(f"    Singular Jacobian at iter {it}")
            break

        alpha = 1.0
        for _ in range(30):
            Phi_trial = np.maximum(Phi + alpha * dPhi, floor_val)
            F_trial = model_ex.residual_fixedpoint(Phi_trial)
            F_trial[Phi_trial <= floor_val + 1e-13] = 0.0
            F_trial[-1] = Phi_trial[-1]
            if np.max(np.abs(F_trial)) < res * 1.1:
                break
            alpha *= 0.5
            if alpha < 1e-6:
                break

        Phi = np.maximum(Phi + alpha * dPhi, floor_val)

    lapse = 1.0 + Phi / cstar_sq
    rs = extract_rs(Phi, model_ex, vacuum=True)

    return {
        "Phi": Phi, "model": model_ex, "lapse": lapse, "rs": rs,
        "min_N": lapse.min(), "converged": res < tol, "residual": res,
        "Phi_analytic": Phi_an,
    }


def compute_embedding(Phi):
    """Compute R(rho) embedding from solution Phi."""
    lapse = 1.0 + Phi / cstar_sq
    r = a * np.arange(1, N + 1, dtype=float)
    Nbar = np.maximum(0.5 * (lapse[:-1] + lapse[1:]), 1e-15)
    dr = np.diff(r)
    d_rho = dr / (t0 * Nbar)
    rho = np.zeros(len(r))
    rho[1:] = np.cumsum(d_rho)
    return rho, r, lapse


# ═══════════════════════════════════════════════════════════════════
# PHASE A: Solve exact two-state for core V0 values
# ═══════════════════════════════════════════════════════════════════

V0_core = [0.01, 0.02, 0.03, 0.05, 0.057]
V0_extended = [0.005, 0.04, 0.055]
V0_supercrit = [0.07, 0.1]

print("=" * 70)
print("PHASE A: Solving exact two-state for core V0 values")
print("=" * 70)

solutions = {}
Phi_seed = None

t_start = time.time()
for V0 in sorted(V0_core + V0_extended):
    print(f"\n--- V0 = {V0:.4f} ---")
    res = solve_exact(
        N=N, t0=t0, V0=V0, n_core=n_core, beta0=beta0,
        cstar_sq=cstar_sq, tol=1e-6, lapse_floor=0.01,
        Phi_seed=Phi_seed, verbose=True)

    lapse_ex = 1.0 + res["Phi_exact"] / cstar_sq
    _, Nbar = res["model_exact"].lapse_nbar(res["Phi_exact"])
    kappa_ex = res["model_exact"].conductances_exact(res["Phi_exact"])
    kappa_flat = res["model_exact"].g[:-1] * t0**2

    solutions[V0] = {
        "Phi": res["Phi_exact"],
        "lapse": lapse_ex,
        "Nbar": Nbar,
        "kappa": kappa_ex,
        "kappa_flat": kappa_flat,
        "rs": res["rs_exact"],
        "min_N": res["min_N_exact"],
        "converged": res["conv_exact"],
        "residual": res["F_exact"],
    }
    print(f"  rs={res['rs_exact']:.4f}, min(N)={res['min_N_exact']:.6f}, "
          f"|F|={res['F_exact']:.2e}")

    if res["conv_exact"]:
        Phi_seed = res["Phi_exact"].copy()

dt_a = time.time() - t_start
print(f"\nPhase A done in {dt_a:.0f}s")

# Super-critical cases (with floor)
print("\n" + "=" * 70)
print("PHASE A2: Super-critical cases (with floor)")
print("=" * 70)

for V0 in V0_supercrit:
    floor = 1e-5
    print(f"\n--- V0 = {V0:.4f} (floor={floor}) ---")
    res = solve_exact_floored(V0, lapse_floor=floor, Phi_seed=None,
                               tol=1e-8, verbose=True)
    solutions[V0] = {
        "Phi": res["Phi"], "lapse": res["lapse"],
        "Nbar": 0.5 * (res["lapse"][:-1] + res["lapse"][1:]),
        "kappa": res["model"].conductances_exact(res["Phi"]),
        "kappa_flat": res["model"].g[:-1] * t0**2,
        "rs": res["rs"], "min_N": res["min_N"],
        "converged": res["converged"],
        "residual": res["residual"],
    }
    print(f"  rs={res['rs']:.4f}, min(N)={res['min_N']:.6f}")


# ═══════════════════════════════════════════════════════════════════
# PHASE A3: Proxy + analytic two-state for comparison overlays
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PHASE A3: Proxy + analytic two-state for comparison overlays")
print("=" * 70)

proxy_solutions = {}
analytic_solutions = {}
for V0 in sorted(set(V0_core + V0_extended)):
    print(f"  V0 = {V0:.4f} ... ", end="", flush=True)
    model_an = TwoStateShellModel(N=N, t0=t0, V0=V0, n_core=n_core,
                                   beta0=beta0, cstar_sq=cstar_sq)
    # Use exact solution as seed to avoid slow Picard for near-critical V0
    seed = solutions[V0]["Phi"] if V0 in solutions else None
    Phi_p, _ = solve_proxy(model_an, Phi_seed=seed, tol=1e-12, lapse_floor=0.01)
    Phi_a, _, _ = solve_full(model_an, Phi_seed=seed, tol=1e-12, lapse_floor=0.01)
    proxy_solutions[V0] = {
        "Phi": Phi_p, "lapse": 1 + Phi_p / cstar_sq,
        "rs": extract_rs(Phi_p, model_an, vacuum=True),
    }
    analytic_solutions[V0] = {
        "Phi": Phi_a, "lapse": 1 + Phi_a / cstar_sq,
        "rs": extract_rs(Phi_a, model_an, vacuum=True),
    }
    print(f"proxy rs={proxy_solutions[V0]['rs']:.4f}, "
          f"analytic rs={analytic_solutions[V0]['rs']:.4f}")

print("Phase A3 done.")


# ═══════════════════════════════════════════════════════════════════
# PHASE B: Floor-independence study (exact)
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PHASE B: Floor-independence study (exact solver)")
print("=" * 70)

floors = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00001]
V0_floor_test = [0.03, 0.05, 0.057, 0.07, 0.1]

floor_data = {}
proxy_floor_data = {}
V0_proxy_floor = [0.05, 0.057, 0.07, 0.085, 0.092, 0.1]

for V0 in V0_floor_test:
    print(f"\n  V0 = {V0:.3f}")
    floor_data[V0] = []
    Phi_seed_f = None

    for floor in floors:
        res = solve_exact_floored(V0, lapse_floor=floor,
                                   Phi_seed=Phi_seed_f,
                                   tol=1e-8, verbose=False)
        floor_data[V0].append({
            "floor": floor,
            "min_N": res["min_N"],
            "rs": res["rs"],
            "converged": res["converged"],
        })
        if res["converged"]:
            Phi_seed_f = res["Phi"].copy()

        print(f"    floor={floor:.1e}: min(N)={res['min_N']:.8f}, "
              f"rs={res['rs']:.4f}, conv={res['converged']}")

    minNs = [d["min_N"] for d in floor_data[V0]]
    spread = max(minNs) - min(minNs)
    follows = minNs[-1] < 2 * floors[-1]
    status = "FOLLOWS FLOOR" if follows else "FLOOR-INDEPENDENT"
    print(f"    -> {status} (spread={spread:.2e})")

# Proxy floor study
print("\n  Proxy floor study:")
for V0 in V0_proxy_floor:
    proxy_floor_data[V0] = []
    for floor in floors:
        model_pf = TwoStateShellModel(N=N, t0=t0, V0=V0, n_core=n_core,
                                       beta0=beta0, cstar_sq=cstar_sq)
        Phi_pf, _ = solve_proxy(model_pf, tol=1e-12, lapse_floor=floor)
        lapse_pf = 1 + Phi_pf / cstar_sq
        proxy_floor_data[V0].append({
            "floor": floor,
            "min_N": lapse_pf.min(),
        })
    minNs_p = [d["min_N"] for d in proxy_floor_data[V0]]
    follows_p = minNs_p[-1] < 2 * floors[-1]
    print(f"    V0={V0:.3f}: min(N) range [{min(minNs_p):.6f}, {max(minNs_p):.6f}], "
          f"{'FOLLOWS' if follows_p else 'INDEPENDENT'}")


# ═══════════════════════════════════════════════════════════════════
# PHASE C: V0 sweep for summary figure
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PHASE C: V0 sweep for summary (exact solver)")
print("=" * 70)

V0_sweep = np.array(sorted(set(
    list(np.linspace(0.005, 0.057, 12)) +
    list(V0_core) + list(V0_extended)
)))

sweep_results = []
Phi_seed_sw = None
for V0 in V0_sweep:
    if V0 in solutions:
        s = solutions[V0]
        sweep_results.append({
            "V0": V0, "rs": s["rs"], "min_N": s["min_N"],
        })
        Phi_seed_sw = s["Phi"].copy()
        continue

    res = solve_exact(
        N=N, t0=t0, V0=V0, n_core=n_core, beta0=beta0,
        cstar_sq=cstar_sq, tol=1e-6, lapse_floor=0.01,
        Phi_seed=Phi_seed_sw, verbose=False)
    sweep_results.append({
        "V0": V0,
        "rs": res["rs_exact"],
        "min_N": res["min_N_exact"],
    })
    if res["conv_exact"]:
        Phi_seed_sw = res["Phi_exact"].copy()
    print(f"  V0={V0:.4f}: rs={res['rs_exact']:.4f}, "
          f"min(N)={res['min_N_exact']:.6f}")

# Proxy + analytic V0 sweep (for continuous overlay curves in Figure 11)
print("\n  Proxy + analytic V0 sweep:")
sweep_proxy_rs = []
sweep_analytic_rs = []
sweep_proxy_minN = []
sweep_analytic_minN = []
for V0 in V0_sweep:
    model_an = TwoStateShellModel(N=N, t0=t0, V0=V0, n_core=n_core,
                                   beta0=beta0, cstar_sq=cstar_sq)
    seed = solutions[V0]["Phi"] if V0 in solutions else None
    Phi_p, _ = solve_proxy(model_an, Phi_seed=seed, tol=1e-12, lapse_floor=0.01)
    Phi_a, _, _ = solve_full(model_an, Phi_seed=seed, tol=1e-12, lapse_floor=0.01)
    sweep_proxy_rs.append(extract_rs(Phi_p, model_an, vacuum=True))
    sweep_analytic_rs.append(extract_rs(Phi_a, model_an, vacuum=True))
    sweep_proxy_minN.append((1 + Phi_p / cstar_sq).min())
    sweep_analytic_minN.append((1 + Phi_a / cstar_sq).min())
    print(f"  V0={V0:.4f}: proxy rs={sweep_proxy_rs[-1]:.4f}, "
          f"analytic rs={sweep_analytic_rs[-1]:.4f}")

sweep_proxy_rs = np.array(sweep_proxy_rs)
sweep_analytic_rs = np.array(sweep_analytic_rs)
sweep_proxy_minN = np.array(sweep_proxy_minN)
sweep_analytic_minN = np.array(sweep_analytic_minN)


# ═══════════════════════════════════════════════════════════════════
# PHASE B2: Temperature sweep (verify convergence at all β₀t₀)
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PHASE B2: Temperature sweep (exact solver, bt0-continuation)")
print("=" * 70)

bt0_values = np.arange(0.1, 2.61, 0.1)
V0_temp_list = [0.001, 0.005]
temp_sweep = {V0t: [] for V0t in V0_temp_list}
mask_far = (r_arr >= 60) & (r_arr <= 150)

t_b2 = time.time()
for V0_temp in V0_temp_list:
    print(f"\n  V0 = {V0_temp} temperature sweep")

    # Solve at bt0=0.1 for initial seed
    if V0_temp in solutions:
        Phi_cont = solutions[V0_temp]["Phi"].copy()
    else:
        res0 = solve_exact(N=N, t0=t0, V0=V0_temp, n_core=n_core,
                           beta0=0.1, cstar_sq=cstar_sq, tol=1e-8,
                           lapse_floor=0.01, verbose=False)
        Phi_cont = res0["Phi_exact"].copy()

    for bt0 in bt0_values:
        this_beta0 = bt0 / t0

        model_ex_T = TwoStateExactModel(
            N=N, t0=t0, V0=V0_temp, n_core=n_core,
            beta0=this_beta0, cstar_sq=cstar_sq)

        Phi_ex, conv_ex, nit_ex = newton_fd_exact(
            model_ex_T, Phi_cont.copy(), tol=1e-8, max_iter=50,
            lapse_guard=True, verbose=False,
            jacobian_func=model_ex_T.residual_fixedpoint)

        Fex = np.max(np.abs(model_ex_T.residual(Phi_ex)))
        lapse_ex = 1.0 + Phi_ex / cstar_sq
        minN_ex = lapse_ex.min()
        rs_ex = extract_rs(Phi_ex, model_ex_T, vacuum=True)

        dPhi_core = np.max(np.abs(np.diff(Phi_ex[:50])))
        Phi_scale = max(np.max(np.abs(Phi_ex)), 1e-15)
        stagger = dPhi_core / Phi_scale

        if conv_ex:
            Phi_cont = Phi_ex.copy()
        elif Fex < 1e-6:
            Phi_cont = Phi_ex.copy()

        R_outer = float(r_arr[-1])  # outer Dirichlet radius
        GM_est = -Phi_ex[mask_far] * r_arr[mask_far] * R_outer / (R_outer - r_arr[mask_far])
        GM_ex = float(np.mean(GM_est))
        GM_std_ex = float(np.std(GM_est) / GM_ex) if GM_ex > 0 else np.nan

        # Proxy at this temperature
        model_an_T = TwoStateShellModel(N=N, t0=t0, V0=V0_temp, n_core=n_core,
                                         beta0=this_beta0, cstar_sq=cstar_sq)
        Phi_p_T, _ = solve_proxy(model_an_T, tol=1e-12, lapse_floor=0.01)
        rs_proxy_T = extract_rs(Phi_p_T, model_an_T, vacuum=True)

        entry = {
            "bt0": bt0, "rs_exact": rs_ex, "conv_exact": conv_ex,
            "F_exact": Fex, "minN_exact": minN_ex,
            "stagger": stagger, "Phi_exact": Phi_ex,
            "GM_exact": GM_ex, "GM_std_exact": GM_std_ex,
            "rs_proxy": rs_proxy_T,
        }
        temp_sweep[V0_temp].append(entry)

        print(f"  bt0={bt0:.2f}: rs={rs_ex:.4f} "
              f"|F|={Fex:.2e} min(N)={minN_ex:.4f} "
              f"proxy_rs={rs_proxy_T:.4f} "
              f"{'OK' if conv_ex else 'NEAR' if Fex < 1e-6 else 'FAIL'}")

dt_b2 = time.time() - t_b2
print(f"\nPhase B2 done in {dt_b2:.0f}s")


# ═══════════════════════════════════════════════════════════════════
# Compute all derived quantities needed by plots
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("Computing derived quantities for plots...")
print("=" * 70)

# -- Embeddings (exact + proxy) --
V0_embed = [0.01, 0.03, 0.05, 0.057] + V0_supercrit
embed_rho = {}
embed_R = {}
proxy_embed_rho = {}
proxy_embed_R = {}
for V0 in V0_embed:
    rho, R, _ = compute_embedding(solutions[V0]["Phi"])
    embed_rho[V0] = rho
    embed_R[V0] = R
    print(f"  Embedding V0={V0}: rho_total={rho[-1]:.1f}")
    if V0 in proxy_solutions:
        rho_p, R_p, _ = compute_embedding(proxy_solutions[V0]["Phi"])
        proxy_embed_rho[V0] = rho_p
        proxy_embed_R[V0] = R_p
        print(f"    proxy embedding: rho_total={rho_p[-1]:.1f}")

# -- Thermodynamics --
V0_thermo = [V0 for V0 in sorted(solutions.keys())
             if 0.01 <= V0 <= 0.057 and V0 not in V0_supercrit]

thermo_data = []
for V0 in V0_thermo:
    s = solutions[V0]
    rs = s["rs"]
    if rs <= 0:
        continue

    kappa_H = extract_kappa_H(s["lapse"], r_arr, rs, cstar_sq)
    cstar = np.sqrt(cstar_sq)
    T_H = kappa_H / (2 * np.pi * cstar)

    kH_proxy = np.nan
    kH_analytic = np.nan
    if V0 in proxy_solutions:
        p = proxy_solutions[V0]
        rs_p = p["rs"]
        if rs_p > 0:
            kH_proxy = extract_kappa_H(p["lapse"], r_arr, rs_p, cstar_sq)
    if V0 in analytic_solutions:
        an = analytic_solutions[V0]
        rs_a = an["rs"]
        if rs_a > 0:
            kH_analytic = extract_kappa_H(an["lapse"], r_arr, rs_a, cstar_sq)

    thermo_data.append({
        "V0": V0, "rs": rs, "kappa_H": kappa_H, "T_H": T_H,
        "min_N": s["min_N"],
        "kH_proxy": kH_proxy, "kH_analytic": kH_analytic,
    })

# -- Cross-residuals for summary figure --
V0_comp = sorted(set(V0_core + V0_extended))
cross_proxy = []
cross_analytic = []
for V0 in V0_comp:
    if V0 in solutions:
        model_ex = TwoStateExactModel(N=N, t0=t0, V0=V0, n_core=n_core,
                                       beta0=beta0, cstar_sq=cstar_sq)
        F_p = model_ex.residual(proxy_solutions[V0]["Phi"])
        F_a = model_ex.residual(analytic_solutions[V0]["Phi"])
        cross_proxy.append(np.max(np.abs(F_p[:-1])))
        cross_analytic.append(np.max(np.abs(F_a[:-1])))
    else:
        cross_proxy.append(np.nan)
        cross_analytic.append(np.nan)

# -- T^QI profiles --
V0_tqi = [0.01, 0.02, 0.03, 0.05, 0.057]
tqi_sc_dict = {}
tqi_pr_dict = {}
for V0 in V0_tqi:
    s = solutions[V0]
    rs = s["rs"]
    if rs < 0.01:
        continue
    Nbar = s["Nbar"]
    Nbar_ext = np.concatenate([Nbar, [1.0]])
    la = beta0 * np.maximum(np.abs(Nbar_ext), 1e-10) * t0
    lt = np.log(np.maximum(la, 1e-30))
    tqi_sc = np.abs(Nbar_ext)**6 / np.maximum(lt**2, 1e-30)

    f_s = np.maximum(1.0 - rs / r_arr, 1e-10)
    la_s = beta0 * t0 * np.sqrt(np.maximum(f_s, 1e-10))
    lt_s = np.log(np.maximum(np.abs(la_s), 1e-30))
    tqi_pr = f_s**3 / np.maximum(lt_s**2, 1e-30)

    tqi_sc_dict[V0] = tqi_sc
    tqi_pr_dict[V0] = tqi_pr

# -- Verification data (V0=0.05) --
V0_verify = 0.05
s_v = solutions[V0_verify]
Phi_v = s_v["Phi"]
kappa_v = s_v["kappa"]

LHS_v = np.zeros(N)
LHS_v[:-1] += kappa_v * (Phi_v[:-1] - Phi_v[1:])
LHS_v[1:] += kappa_v * (Phi_v[1:] - Phi_v[:-1])

model_v = TwoStateExactModel(N=N, t0=t0, V0=V0_verify, n_core=n_core,
                               beta0=beta0, cstar_sq=cstar_sq)
F_v = model_v.residual(Phi_v)
RHS_v = LHS_v - F_v
RHS_v[-1] = LHS_v[-1]


# ═══════════════════════════════════════════════════════════════════
# Save everything to npz
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("Saving to npz...")
print("=" * 70)

DATADIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "fermion", "numerical", "data")
os.makedirs(DATADIR, exist_ok=True)

data = {}

# Parameters
data["N"] = N
data["t0"] = t0
data["n_core"] = n_core
data["beta0"] = beta0
data["cstar_sq"] = cstar_sq
data["a"] = a
data["r_arr"] = r_arr
data["r_bond"] = r_bond

# V0 lists
data["V0_core"] = np.array(V0_core)
data["V0_extended"] = np.array(V0_extended)
data["V0_supercrit"] = np.array(V0_supercrit)

# Per-V0 exact solutions
all_V0 = sorted(solutions.keys())
data["solution_V0s"] = np.array(all_V0)
for V0 in all_V0:
    s = solutions[V0]
    tag = f"sol_{V0:.4f}"
    data[f"{tag}_Phi"] = s["Phi"]
    data[f"{tag}_lapse"] = s["lapse"]
    data[f"{tag}_Nbar"] = s["Nbar"]
    data[f"{tag}_kappa"] = s["kappa"]
    data[f"{tag}_kappa_flat"] = s["kappa_flat"]
    data[f"{tag}_rs"] = s["rs"]
    data[f"{tag}_min_N"] = s["min_N"]
    data[f"{tag}_converged"] = s["converged"]
    data[f"{tag}_residual"] = s["residual"]

# Proxy + analytic solutions (with Nbar and kappa_ratio)
proxy_V0s = sorted(proxy_solutions.keys())
data["proxy_V0s"] = np.array(proxy_V0s)
for V0 in proxy_V0s:
    tag = f"proxy_{V0:.4f}"
    data[f"{tag}_Phi"] = proxy_solutions[V0]["Phi"]
    data[f"{tag}_lapse"] = proxy_solutions[V0]["lapse"]
    data[f"{tag}_rs"] = proxy_solutions[V0]["rs"]
    lapse_p = proxy_solutions[V0]["lapse"]
    Nbar_p = 0.5 * (lapse_p[:-1] + lapse_p[1:])
    data[f"{tag}_Nbar"] = Nbar_p
    data[f"{tag}_kappa_ratio"] = Nbar_p**2

analytic_V0s = sorted(analytic_solutions.keys())
data["analytic_V0s"] = np.array(analytic_V0s)
for V0 in analytic_V0s:
    tag = f"analytic_{V0:.4f}"
    data[f"{tag}_Phi"] = analytic_solutions[V0]["Phi"]
    data[f"{tag}_lapse"] = analytic_solutions[V0]["lapse"]
    data[f"{tag}_rs"] = analytic_solutions[V0]["rs"]
    lapse_a = analytic_solutions[V0]["lapse"]
    Nbar_a = 0.5 * (lapse_a[:-1] + lapse_a[1:])
    data[f"{tag}_Nbar"] = Nbar_a
    data[f"{tag}_kappa_ratio"] = Nbar_a**2

# Floor study — exact
data["floors"] = np.array(floors)
data["V0_floor_test"] = np.array(V0_floor_test)
for V0 in V0_floor_test:
    tag = f"floor_{V0:.4f}"
    data[f"{tag}_min_N"] = np.array([d["min_N"] for d in floor_data[V0]])
    data[f"{tag}_rs"] = np.array([d["rs"] for d in floor_data[V0]])
    data[f"{tag}_converged"] = np.array([d["converged"] for d in floor_data[V0]])

# Floor study — proxy
data["V0_proxy_floor"] = np.array(V0_proxy_floor)
for V0 in V0_proxy_floor:
    tag = f"pfloor_{V0:.4f}"
    data[f"{tag}_min_N"] = np.array([d["min_N"] for d in proxy_floor_data[V0]])

# Sweep (exact + proxy + analytic)
data["sweep_V0"] = np.array([d["V0"] for d in sweep_results])
data["sweep_rs"] = np.array([d["rs"] for d in sweep_results])
data["sweep_min_N"] = np.array([d["min_N"] for d in sweep_results])
data["sweep_proxy_rs"] = sweep_proxy_rs
data["sweep_analytic_rs"] = sweep_analytic_rs
data["sweep_proxy_minN"] = sweep_proxy_minN
data["sweep_analytic_minN"] = sweep_analytic_minN

# Embeddings (exact)
data["embed_V0s"] = np.array(V0_embed)
for V0 in V0_embed:
    tag = f"embed_{V0:.4f}"
    data[f"{tag}_rho"] = embed_rho[V0]
    data[f"{tag}_R"] = embed_R[V0]

# Embeddings (proxy)
for V0 in V0_embed:
    if V0 in proxy_embed_rho:
        ptag = f"pembed_{V0:.4f}"
        data[f"{ptag}_rho"] = proxy_embed_rho[V0]
        data[f"{ptag}_R"] = proxy_embed_R[V0]

# Thermodynamics
data["thermo_V0"] = np.array([d["V0"] for d in thermo_data])
data["thermo_rs"] = np.array([d["rs"] for d in thermo_data])
data["thermo_kappa_H"] = np.array([d["kappa_H"] for d in thermo_data])
data["thermo_T_H"] = np.array([d["T_H"] for d in thermo_data])
data["thermo_min_N"] = np.array([d["min_N"] for d in thermo_data])
data["thermo_kH_proxy"] = np.array([d["kH_proxy"] for d in thermo_data])
data["thermo_kH_analytic"] = np.array([d["kH_analytic"] for d in thermo_data])

# Cross-residuals
data["cross_V0"] = np.array(V0_comp)
data["cross_proxy"] = np.array(cross_proxy)
data["cross_analytic"] = np.array(cross_analytic)

# T^QI profiles
data["tqi_V0s"] = np.array(sorted(tqi_sc_dict.keys()))
for V0 in sorted(tqi_sc_dict.keys()):
    tag = f"tqi_{V0:.4f}"
    data[f"{tag}_sc"] = tqi_sc_dict[V0]
    data[f"{tag}_pr"] = tqi_pr_dict[V0]

# Temperature sweep (Phase B2)
data["temp_V0s"] = np.array(V0_temp_list)
data["temp_bt0"] = bt0_values
for V0_temp in V0_temp_list:
    tag = f"temp_{V0_temp:.4f}"
    sw = temp_sweep[V0_temp]
    data[f"{tag}_rs"] = np.array([e["rs_exact"] for e in sw])
    data[f"{tag}_conv"] = np.array([e["conv_exact"] for e in sw])
    data[f"{tag}_F"] = np.array([e["F_exact"] for e in sw])
    data[f"{tag}_minN"] = np.array([e["minN_exact"] for e in sw])
    data[f"{tag}_stagger"] = np.array([e["stagger"] for e in sw])
    data[f"{tag}_Phi"] = np.array([e["Phi_exact"] for e in sw])
    data[f"{tag}_GM"] = np.array([e["GM_exact"] for e in sw])
    data[f"{tag}_GM_std"] = np.array([e["GM_std_exact"] for e in sw])
    data[f"{tag}_proxy_rs"] = np.array([e["rs_proxy"] for e in sw])

# Verification (V0=0.05)
data["verify_V0"] = V0_verify
data["verify_LHS"] = LHS_v
data["verify_RHS"] = RHS_v

outpath = os.path.join(DATADIR, "exact_solutions.npz")
np.savez_compressed(outpath, **data)
print(f"Saved: {outpath}")
print(f"  Keys: {len(data)}")
print(f"  Size: {os.path.getsize(outpath) / 1024:.0f} KB")

# Summary table
print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)
print(f"{'V0':>8s} {'rs/a':>8s} {'min_N':>8s} {'kH':>10s} {'TH':>10s}")
print("-" * 50)
for d in thermo_data:
    print(f"{d['V0']:8.4f} {d['rs']:8.4f} {d['min_N']:8.4f} "
          f"{d['kappa_H']:10.4e} {d['T_H']:10.4e}")

print(f"\nTotal time: {time.time() - t_start:.0f}s")
print("Done.")
