#!/usr/bin/env python3
"""
Full EL convergence + parameter robustness for TK stars.

Study 1: N-convergence at beta0=3, mu=0.5, m0=3, csq=5
Study 2: Robustness across (beta0, m0, csq) at N=200, mu=0.5

Uses scipy eigh_tridiagonal (CPU) and hybrid analytical/FD Jacobian
for scalability to large N.
"""
import time
import numpy as np
from scipy.linalg import eigh_tridiagonal, solve_banded

t0 = 1.0
N_FLOOR = 0.005


def make_solver(N, beta0, m0):
    n_arr = np.arange(1, N + 1, dtype=np.float64)
    g = 4.0 * np.pi * n_arr**2
    nunk = N - 1

    def compute_state(diag, off):
        evals, evecs = eigh_tridiagonal(diag, off)
        be = np.clip(beta0 * evals, -500.0, 500.0)
        f = 1.0 / (np.exp(be) + 1.0)
        vf = evecs * f[None, :]
        return np.sum(evecs * vf, axis=1), np.sum(evecs[:-1] * vf[1:], axis=1)

    def residual(Phi_inner, csq, mu):
        Phi_full = np.append(Phi_inner, 0.0)
        lapse = np.maximum(1.0 + Phi_full / csq, N_FLOOR)
        Nbar = 0.5 * (lapse[:-1] + lapse[1:])
        Nbar_abs = np.abs(Nbar)

        V_raw = m0 * (1.0 - lapse) - mu
        V_defect = np.maximum(V_raw, 0.0)
        n_eff = int(np.sum(V_defect > 1e-10))

        off = -t0 * Nbar_abs
        Gd_s, Gs_s = compute_state(np.zeros(N), off)
        Gd_t, Gs_t = compute_state(V_defect, off)

        rho_s = np.zeros(N)
        rho_s[:-1] = 2.0 * t0 * Nbar_abs * Gs_s * g[:-1]
        rho_t = np.zeros(N)
        rho_t[:-1] = 2.0 * t0 * Nbar_abs * Gs_t * g[:-1]
        rho_t += V_defect * Gd_t * g
        source = (beta0 / csq) * (rho_s - rho_t)

        kappa = g[:-1] * t0**2 * Nbar_abs**2

        # L@Phi (matrix-free)
        dPhi = Phi_full[:-1] - Phi_full[1:]
        flux = kappa * dPhi
        LPhi = np.empty(nunk)
        LPhi[0] = flux[0]
        LPhi[1:] = flux[1:] - flux[:-1]

        # EL correction (vectorized)
        dPhi_sq = dPhi**2
        dkdPhi = g[:-1] * t0**2 * Nbar_abs / csq
        C = (dkdPhi * dPhi_sq).copy()
        C[1:] += dkdPhi[:-1] * dPhi_sq[:-1]
        C *= 0.5

        R = LPhi - source[:nunk] + C
        info = {"Phi": Phi_full, "lapse": lapse, "source": source,
                "V_defect": V_defect, "kappa": kappa, "n_eff": n_eff,
                "Nbar": Nbar_abs, "dPhi": dPhi,
                "Gd_s": Gd_s, "Gd_t": Gd_t}
        return R, info

    return {"residual": residual, "n_arr": n_arr, "g": g, "N": N, "nunk": nunk}


def hessian_diags(d, csq, g):
    """Tridiagonal diagonals of the Hessian of E_mis (analytical)."""
    p = 0.5 / csq
    u = d["Nbar"]
    f = d["dPhi"]
    gb = g[:-1] * t0**2
    pf2 = (p * f)**2
    upf = u * p * f
    u2 = u**2
    h00 = gb * (pf2 + 4 * upf + u2)
    h01 = gb * (pf2 - u2)
    h11 = gb * (pf2 - 4 * upf + u2)
    nunk = len(d["Phi"]) - 1
    main = h00.copy()
    main[1:] += h11[:-1]
    off = h01[:-1].copy()
    return main, off


def solve_tridiag(main, off, rhs):
    n = len(main)
    ab = np.zeros((3, n))
    ab[0, 1:] = off
    ab[1, :] = main
    ab[2, :-1] = off
    return solve_banded((1, 1), ab, rhs)


def newton(S, csq, mu, seed, tol=1e-8, max_iter=30, verbose=False,
           core_buf=10):
    nunk = S["nunk"]
    g = S["g"]
    x = seed[:nunk].copy()
    eps = max(1e-7 * csq, 1e-8)
    best_x, best_rn, best_d = x.copy(), 1e30, None
    stall_count = 0

    for it in range(max_iter):
        R, d = S["residual"](x, csq, mu)
        rn = np.linalg.norm(R)
        if rn < best_rn * 0.99:
            best_x, best_rn, best_d = x.copy(), rn, d
            stall_count = 0
        else:
            stall_count += 1
        if verbose:
            print(f"    it={it:2d} |R|={rn:.3e} minPhi={min(d['Phi']):.4f}"
                  f" n_eff={d['n_eff']}", flush=True)
        if rn < tol or stall_count >= 5:
            break

        n_eff = d["n_eff"]
        j_max = min(n_eff + core_buf, nunk)

        if j_max > nunk // 2:
            # Dense FD Jacobian
            J = np.zeros((nunk, nunk))
            for j in range(nunk):
                xp = x.copy()
                step = eps * max(1.0, abs(x[j]))
                xp[j] += step
                Rp, _ = S["residual"](xp, csq, mu)
                J[:, j] = (Rp - R) / step
            try:
                delta = np.linalg.solve(J, -R)
            except np.linalg.LinAlgError:
                break
        else:
            # Hybrid: analytical tridiagonal + FD for core columns
            main, off = hessian_diags(d, csq, g)

            # FD columns for core region
            FD = np.zeros((nunk, j_max))
            for k in range(j_max):
                xp = x.copy()
                step = eps * max(1.0, abs(x[k]))
                xp[k] += step
                Rp, _ = S["residual"](xp, csq, mu)
                FD[:, k] = (Rp - R) / step

            # Analytical T columns at core indices
            T_cols = np.zeros((nunk, j_max))
            for k in range(j_max):
                if k > 0:
                    T_cols[k - 1, k] = off[k - 1]
                T_cols[k, k] = main[k]
                if k < nunk - 1:
                    T_cols[k + 1, k] = off[k]

            dU = FD - T_cols  # low-rank correction

            # Woodbury: solve (T + dU @ E^T) delta = -R
            try:
                y = solve_tridiag(main, off, -R)
                Z = np.column_stack(
                    [solve_tridiag(main, off, dU[:, k])
                     for k in range(j_max)])
                M = np.eye(j_max) + Z[:j_max, :]
                w = np.linalg.solve(M, y[:j_max])
                delta = y - Z @ w
            except Exception:
                # Fallback: build dense Jacobian
                J = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)
                for k in range(j_max):
                    J[:, k] = FD[:, k]
                try:
                    delta = np.linalg.solve(J, -R)
                except np.linalg.LinAlgError:
                    break

        # Line search with lapse guard
        alpha = 1.0
        for ls in range(30):
            xt = x + alpha * delta
            if np.min(1.0 + xt / csq) < N_FLOOR:
                alpha *= 0.5
                continue
            Rt, _ = S["residual"](xt, csq, mu)
            if np.linalg.norm(Rt) < rn:
                break
            alpha *= 0.5
        else:
            alpha = 0.001
        x = x + alpha * delta

    if best_d is None:
        _, best_d = S["residual"](best_x, csq, mu)
    R_f, _ = S["residual"](best_x, csq, mu)
    return best_x, best_d, np.linalg.norm(R_f)


def interp_seed(S, ref_Phi):
    """Interpolate a reference Phi (any length) onto the current grid."""
    n_arr = S["n_arr"]
    nunk = S["nunk"]
    n_ref = np.arange(1, len(ref_Phi) + 1, dtype=np.float64)
    return np.interp(n_arr[:nunk], n_ref, ref_Phi[:len(ref_Phi)])


def find_nontrivial(S, csq, mu, ref_Phi=None, max_iter=80, verbose=False):
    """Try multiple seeds with full Newton; keep best nontrivial solution."""
    n_arr = S["n_arr"]
    nunk = S["nunk"]
    best = None  # (x, d, rn)

    seeds = []
    if ref_Phi is not None:
        seeds.append(interp_seed(S, ref_Phi))

    for amp in [0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0]:
        for w in [3.0, 5.0, 8.0, 15.0]:
            s = -amp * np.exp(-0.5 * (n_arr / w)**2)
            if 1 + min(s) / csq < 0.01:
                continue
            seeds.append(s[:nunk])

    for seed in seeds:
        x, d, rn = newton(S, csq, mu, seed, max_iter=max_iter,
                          verbose=False)
        if d is None or d["n_eff"] == 0 or abs(min(d["Phi"])) < 0.01:
            continue
        if best is None or rn < best[2]:
            best = (x, d, rn)
        if rn < 1e-8:
            break

    return best


def diagnostics(S, d, csq):
    n_arr = S["n_arr"]
    g = S["g"]
    N_val = S["N"]
    lapse = d["lapse"]
    n_eff = d["n_eff"]
    Phi = d["Phi"]
    w = lapse**2

    n_start = max(n_eff + 3, 5)
    n_end = min(N_val // 2, n_eff + 40)
    r_s = 0.0
    w_err = 0.0
    if n_start < n_end:
        idx = np.arange(n_start - 1, n_end)
        r_fit = n_arr[idx]
        basis = 1.0 / r_fit - 1.0 / N_val
        target = -(w[idx] - 1.0)
        r_s = np.dot(basis, target) / np.dot(basis, basis)
        w_pred = 1.0 - r_s * basis
        rms = np.sqrt(np.mean((w[idx] - w_pred)**2))
        rms_n = np.sqrt(np.mean((w[idx] - 1.0)**2))
        w_err = rms / rms_n if rms_n > 1e-30 else 0

    # w-flux variation
    dw = w[:-1] - w[1:]
    J_w = g[:-1] * t0**2 * dw
    f_start = min(n_eff + 5, N_val - 2)
    f_end = min(N_val // 2, n_eff + 40)
    J_w_var = 0.0
    if f_start < f_end:
        J_ext = J_w[f_start:f_end]
        J_mean = np.mean(J_ext)
        if abs(J_mean) > 1e-30 and len(J_ext) > 1:
            J_w_var = np.max(np.abs(J_ext - J_mean)) / abs(J_mean)

    return {"r_s": r_s, "w_err": w_err, "J_w_var": J_w_var,
            "min_Phi": float(min(Phi)), "min_N": float(min(lapse)),
            "n_eff": n_eff}


# ═══════════════════════════════════════════════════════
# CONVERGENCE STUDY
# ═══════════════════════════════════════════════════════
print("=" * 80)
print("CONVERGENCE: Full EL, chemical potential")
print("beta0=5, mu=0.5, m0=3, csq=5")
print("=" * 80, flush=True)

beta0_c = 5.0
m0_c = 3.0
csq_c = 5.0
mu_c = 0.5
N_values = [50, 100, 200, 400, 600, 1000, 1500, 2000, 3000]

# Load known N=200 solution as reference seed
try:
    ref = np.load("tov_tk_chemical_el.npz")
    ref_Phi = ref["mu0.50_Phi"]
    print(f"Loaded reference: N=200 mu=0.5 minPhi={min(ref_Phi):.4f}")
except FileNotFoundError:
    ref_Phi = None
    print("No reference solution found; using Gaussian seeds")

conv = []
prev_Phi = ref_Phi

for N_val in N_values:
    print(f"\nN = {N_val}", flush=True)
    t0_wall = time.time()
    S = make_solver(N_val, beta0_c, m0_c)
    seed = interp_seed(S, prev_Phi) if prev_Phi is not None else None
    if seed is not None:
        x, d, rn = newton(S, csq_c, mu_c, seed, verbose=True)
        if d is None or d["n_eff"] == 0 or abs(min(d["Phi"])) < 0.01:
            # Seed failed; try broad search
            result = find_nontrivial(S, csq_c, mu_c, ref_Phi=ref_Phi)
            if result:
                x, d, rn = result
            else:
                print(f"  => FAILED ({time.time()-t0_wall:.1f}s)")
                continue
    else:
        result = find_nontrivial(S, csq_c, mu_c)
        if result:
            x, d, rn = result
        else:
            print(f"  => FAILED ({time.time()-t0_wall:.1f}s)")
            continue

    dt = time.time() - t0_wall
    diag = diagnostics(S, d, csq_c)
    prev_Phi = d["Phi"]
    conv.append({"N": N_val, **diag, "rn": rn, "time": dt})
    print(f"  => minPhi={diag['min_Phi']:.6f}  r_s={diag['r_s']:.4f}"
          f"  dw={diag['w_err']:.5f}  dJw={diag['J_w_var']:.5f}"
          f"  |R|={rn:.2e}  ({dt:.1f}s)")

print(f"\n{'='*80}")
print("CONVERGENCE TABLE")
print(f"{'='*80}")
print(f"  {'N':>6s} {'minPhi':>10s} {'r_s':>8s} {'dw':>8s}"
      f" {'dJw':>8s} {'n_eff':>6s} {'|R_EL|':>10s} {'time':>8s}")
for r in conv:
    print(f"  {r['N']:6d} {r['min_Phi']:10.4f} {r['r_s']:8.4f}"
          f" {r['w_err']:8.5f} {r['J_w_var']:8.5f} {r['n_eff']:6d}"
          f" {r['rn']:10.2e} {r['time']:8.1f}s")


# ═══════════════════════════════════════════════════════
# PARAMETER ROBUSTNESS
# ═══════════════════════════════════════════════════════
print(f"\n\n{'='*80}")
print("ROBUSTNESS: Full EL, mu=0.5, N=200")
print(f"{'='*80}", flush=True)

param_sets = [
    {"beta0": 2.0, "m0": 3.0, "csq": 5.0, "label": "b=2"},
    {"beta0": 3.0, "m0": 3.0, "csq": 5.0, "label": "b=3 (base)"},
    {"beta0": 5.0, "m0": 3.0, "csq": 5.0, "label": "b=5"},
    {"beta0": 8.0, "m0": 3.0, "csq": 5.0, "label": "b=8"},
    {"beta0": 3.0, "m0": 1.5, "csq": 5.0, "label": "m=1.5"},
    {"beta0": 3.0, "m0": 5.0, "csq": 5.0, "label": "m=5"},
    {"beta0": 3.0, "m0": 3.0, "csq": 3.0, "label": "c=3"},
]

rob = []
for ps in param_sets:
    print(f"\n{ps['label']}...", flush=True)
    t0_wall = time.time()
    S = make_solver(200, ps["beta0"], ps["m0"])
    result = find_nontrivial(S, ps["csq"], mu_c, ref_Phi=ref_Phi,
                             verbose=False)
    dt = time.time() - t0_wall

    if result is not None:
        x, d, rn = result
        diag = diagnostics(S, d, ps["csq"])
        rob.append({"label": ps["label"], **diag, "rn": rn, "time": dt,
                     "beta0": ps["beta0"], "m0": ps["m0"], "csq": ps["csq"]})
        print(f"  => minPhi={diag['min_Phi']:.6f}  r_s={diag['r_s']:.4f}"
              f"  dw={diag['w_err']:.5f}  |R|={rn:.2e}  ({dt:.1f}s)")
    else:
        print(f"  => FAILED ({dt:.1f}s)")

print(f"\n{'='*80}")
print("ROBUSTNESS TABLE")
print(f"{'='*80}")
print(f"  {'params':>12s} {'minPhi':>10s} {'minN':>8s} {'r_s':>8s}"
      f" {'dw':>8s} {'dJw':>8s} {'|R_EL|':>10s}")
for r in rob:
    print(f"  {r['label']:>12s} {r['min_Phi']:10.4f} {r['min_N']:8.4f}"
          f" {r['r_s']:8.4f} {r['w_err']:8.5f} {r['J_w_var']:8.5f}"
          f" {r['rn']:10.2e}")

# Save
np.savez("tov_convergence_el.npz",
         conv_N=[r["N"] for r in conv],
         conv_minPhi=[r["min_Phi"] for r in conv],
         conv_rs=[r["r_s"] for r in conv],
         conv_werr=[r["w_err"] for r in conv],
         conv_Jwvar=[r["J_w_var"] for r in conv],
         conv_rn=[r["rn"] for r in conv],
         rob_labels=[r["label"] for r in rob],
         rob_minPhi=[r["min_Phi"] for r in rob],
         rob_rs=[r["r_s"] for r in rob],
         rob_werr=[r["w_err"] for r in rob],
         rob_rn=[r["rn"] for r in rob])
print("\nSaved tov_convergence_el.npz")
print("Done.")
