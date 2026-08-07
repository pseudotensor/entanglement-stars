#!/usr/bin/env python3
"""
TK star chemical potential scan — FULL EL with KMS/BKM conductances.

Uses jax.grad of the conductance energy functional for the exact EL LHS
(Laplacian + EL correction), and jax.jacfwd for exact Jacobians.
Host-side Newton loop avoids JIT compilation issues with lax.while_loop.

Continuation seeding: each mu uses the previous solution as initial guess.

Outputs data to tov_tk_chemical_el.npz (figures are made by the companion _fig.py
script, per the data/figure separation in CLAUDE.md).
"""

# Post-compaction re-read gate.  This script regenerates PUBLISHED data, so the
# protocol in .taskq/fermion_code_ledger.md requires the whole paper to have been
# re-read first.  That ordering was violated on 2026-08-06 and again on
# 2026-08-07; a rule that depends on being remembered has now failed twice, so it
# is enforced here instead.  No-ops on the compute host, which carries no .taskq.
import sys as _sys, os as _os
_gd = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), ".taskq")
_gd2 = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".taskq")
for _d in (_gd2, _gd):
    if _os.path.isfile(_os.path.join(_d, "gate_guard.py")):
        _sys.path.insert(0, _d)
        from gate_guard import require_reread as _rr
        _rr(__file__)
        break
import os, time
import numpy as np
os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, jacfwd

print(f"JAX devices: {jax.devices()}", flush=True)
t0 = 1.0; N_floor = 0.005


# The solver lives in tk_closure_solver.py so that tov_convergence_el.py runs
# the SAME one.  It used to have its own -- scipy tridiagonal eigensolver plus a
# finite-difference Jacobian -- which made the convergence study a test of a
# different code path than the one being published.
from tk_closure_solver import make_solver

# Parameters.
#
# m0/t0 CHANGED from 3.0 to 0.03 with the corrected closure prefactor.
# sec:ff-tov lists four dimensionless inputs -- beta0 t0, m0/t0, c_*^2/t0, mu/t0
# -- of which P is NOT one: P = c_*^2 t0^2/(lambda eps_b) = 36.825 is fixed by
# the theory, while m0/t0 is free.  The published m0 = 3.0 was chosen against a
# prefactor 37x too small (9x from the solver shorthand beta0/c_*^2, 4x from
# reading lambda as alpha_S rather than the per-channel alpha_S^bond), and it
# admits NO star at the correct one -- continuation in P terminates at 4-10% of
# it and 60+ randomized restarts per mu find nothing.
#
# Scanning m0 at the correct P, stars reappear immediately, with the core size
# trading against the mass:
#   m0 = 0.03 -> n_eff = 7-8     m0 = 0.10 -> n_eff = 3-4
#   m0 = 0.06 -> n_eff = 5       m0 = 0.20 -> n_eff = 2
# 0.03 is chosen as the best-resolved core, which puts n_eff in the same 4-10
# range sec:ff-tov reports and keeps the continuum reading of the interior as
# defensible as it was.  (m0 = 3/36.825 = 0.081 would hold the product P*m0
# fixed and reproduce the published n_eff = 4 most directly; it is the more
# conservative alternative if the core size is ever contested.)
beta0 = 5.0; m0 = 0.03; csq = 5.0
N_val = 200

print(f"JAX devices: {jax.devices()}", flush=True)
print(f"Parameters: beta0={beta0}, m0={m0}, c*2={csq}, N={N_val}")
print(f"FULL EL solver with BKM conductances + jacfwd Jacobian\n")

S = make_solver(N_val, beta0, m0, csq)
n_arr = S["n_arr"]
g = S["g"]

# Scan mu with continuation seeding
# The grid is specified in mu/m0, which is the physical variable, and scaled by
# m0.  The old grid was absolute mu = 0.05..3.0, i.e. mu/m0 = 0.017..1.0 at
# m0 = 3; at m0 = 0.03 those same absolute values are all far past the fold.
# Coarse to mu/m0 = 0.10, then fine through the fold, which the m0 scan places
# between 0.10 and 0.20.
_mu_over_m0 = ([0.005, 0.010, 0.015, 0.020, 0.030, 0.040, 0.050, 0.060,
                0.070, 0.080, 0.090, 0.100]
               + [round(0.100 + 0.005*i, 4) for i in range(1, 19)]   # 0.105..0.190
               # Refined through the fold.  The active-set solver puts the
               # saddle-node at mu*/m0 = 0.191087 (all three Crandall-Rabinowitz
               # conditions, |lam_1| = 2.8e-4 against a next eigenvalue of 10.3),
               # which is 0.0011 above the last coarse point -- well inside one
               # 0.005 step, so the coarse grid cannot resolve the approach.
               + [round(0.1900 + 0.0002*i, 5) for i in range(1, 6)]   # 0.1902..0.1910
               + [0.19105, 0.19108, 0.19109]
               + [round(0.200 + 0.020*i, 4) for i in range(1, 11)])  # 0.220..0.400
mu_values = [round(r * m0, 8) for r in _mu_over_m0]
results = {}
prev_solutions = []  # list of (mu, Phi) for continuation seeding

for mu in mu_values:
    N_surface = 1.0 - mu / m0
    print(f"\n{'='*70}")
    print(f"  mu = {mu:.6f}  ->  N* = {N_surface:.4f}")
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
                seeds.append((f"cont mu={prev_mu:.6f} x{scale:.2f}", scaled))
    for amp in [0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 18.0]:
        for w in [3.0, 5.0, 8.0, 15.0]:
            seed = -amp * np.exp(-0.5 * (n_arr / w)**2)
            seed[-1] = 0.0  # BC
            if 1 + min(seed)/csq < 0.01: continue
            seeds.append((f"gauss amp={amp},w={w}", seed))

    # Star-branch acceptance.  This was `abs(min(Phi)) > 0.01 and rn < 1e-4 and
    # n_eff > 0` -- no multi-shell test, no lapse test, and a residual four orders
    # looser than the 1e-8 the paper states.  With no branch discrimination the
    # floor-pinned n_eff=1 family simply wins on residual above mu ~ 0.75: the
    # committed npz holds 10 star points followed by 53 pinned ones, mu =
    # 0.80/0.90/1.00 have NO solution at all (the branch-switch gap), and from
    # mu = 1.05 continuation seeds off the pinned family, so the star branch is
    # never recovered.  Every downstream script reads this file and inherits it.
    ACCEPT_RN = 1e-8            # the paper's stated tolerance, not 1e-4
    MIN_LAPSE = 20.0 * N_floor  # same cut as the sibling scripts
    NEAR_MISS_CAP = 10          # consecutive non-improving near-misses; see below
    _near_miss = []
    _stall_best, _stall_count = np.inf, 0
    # The cap may not fire until the seed list has been sampled BROADLY.  The
    # list is ordered: 12-15 continuation seeds (scalings of the last three
    # solutions) come first, and near the fold those are near-duplicates -- on
    # the refined grid many pairs agree to under 1e-3 relative.  Ten
    # consecutive stalls among them is one seed tried ten times, not ten
    # independent probes agreeing, so the cap could fire before a single
    # Gaussian seed was ever tried.
    _n_cont = sum(1 for lb, _ in seeds if lb.startswith("cont"))
    _CAP_FLOOR = _n_cont + max(6, len(seeds) // 8)   # past the continuations
    for _seed_i, (label, seed) in enumerate(seeds, start=1):
        # Active set first: it is the one that can cross the free boundary.
        Phi, rn = S["active_set_newton"](mu, seed)
        rn_as = rn                       # keep the ACTIVE-SET residual separately
        if rn >= ACCEPT_RN:
            Phi, rn = S["newton"](mu, seed, max_iter=80, verbose=False)
        d = S["diagnostics"](Phi, mu)
        _is_star = (abs(min(d["Phi"])) > 0.01 and d["n_eff"] >= 2
                    and min(d["lapse"]) > MIN_LAPSE)
        if _is_star and rn >= ACCEPT_RN:
            _near_miss.append((rn, label, d["n_eff"], min(d["lapse"])))
            # No silent cap.  A mu PAST the fold has no solution, so the seed
            # list runs to exhaustion: 2627 s measured at the first such point,
            # against ~5 s for one that converges on its first seed, and eleven
            # of the 48 grid points sit past the fold.
            #
            # Giving up early is safe here for a specific reason, not a general
            # one: at that point all 26 star-shaped near-misses stalled at the
            # SAME |R| = 1.22e-4 with the same n_eff = 6 and minN = 0.3577.
            # They agree the branch has ended rather than disagreeing about
            # where it is, which is what a saddle-node looks like from the
            # solver's side.  So the cut is on a stalled BEST residual, never on
            # a seed count alone -- any seed that improves |R| resets it, and
            # the cut is printed with how many seeds went untried.
            #
            # Track the ACTIVE-SET residual, not the maskless newton() one.
            # `rn` here is always newton()'s, because reaching this branch
            # required the active-set solve to miss and be overwritten above.
            # This file's own docstring records that the maskless residual
            # STALLS wherever a shell crosses threshold, with a well-conditioned
            # Jacobian -- i.e. a solution exists and is simply unreachable that
            # way -- and that reading those stalls as branch ends is what
            # produced the earlier spurious folds.  Capping on it would
            # reinstate exactly that inference.
            _track = rn_as if np.isfinite(rn_as) and rn_as < 1e29 else rn
            if _track < 0.99 * _stall_best:
                _stall_best, _stall_count = _track, 0
            else:
                _stall_count += 1
            if _stall_count >= NEAR_MISS_CAP and _seed_i >= _CAP_FLOOR:
                print(f"    giving up: {_stall_count} consecutive star-shaped "
                      f"near-misses with no improvement on |R|={_stall_best:.2e}; "
                      f"{_seed_i} of {len(seeds)} seeds tried "
                      f"({_n_cont} continuation, "
                      f"{max(0, _seed_i - _n_cont)} independent)",
                      flush=True)
                break
        if _is_star and rn < ACCEPT_RN:
            if rn < best_rn:
                best_Phi, best_rn = Phi, rn
                best_d = d
                print(f"    [{label}] |R|={rn:.2e} minN={min(d['lapse']):.4f} "
                      f"n_eff={d['n_eff']}", flush=True)
            if rn < ACCEPT_RN:
                break
        if best_rn < ACCEPT_RN:
            break

    dt = time.time() - t_start

    # A star that only ALMOST converged is a diagnostic, not a silent drop: the
    # old code would have accepted it at 1e-4 and published it.
    if best_Phi is None and _near_miss:
        _near_miss.sort()
        print(f"  rejected {len(_near_miss)} star-shaped near-miss(es); best "
              f"|R|={_near_miss[0][0]:.2e} [{_near_miss[0][1]}] "
              f"n_eff={_near_miss[0][2]} minN={_near_miss[0][3]:.4f} "
              f"(tolerance {ACCEPT_RN:.0e})", flush=True)

    if best_Phi is not None and best_rn < ACCEPT_RN:
        d = best_d
        Phi = d["Phi"]; lapse = d["lapse"]
        n_eff = d["n_eff"]
        prev_solutions.append((mu, best_Phi.copy()))

        # w = N^2 fit: w = 1 - r_s/r  (Schwarzschild)
        w = lapse**2
        # Window r in [r_* + 3a, min(R/2, r_* + 40a)] with r_* = (n_eff+1) a, as
        # sec:tov-exterior states.  This was n_eff + 3 = r_* + 2a.
        n_start = max(n_eff + 4, 5)
        n_end = min(N_val//2, n_eff + 41)
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
        # Delta w_b := w_{n+1} - w_n, matching eq:tov-Jw (and tov_tk_el_flux_fig.py,
        # which was corrected to the paper's sign).  This script had the opposite
        # sign; only J_w_var below consumed it, and that is a ratio of magnitudes,
        # so nothing published moved -- but two scripts computing eq:tov-Jw with
        # opposite signs is a trap.
        dw = w[1:] - w[:-1]
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

        # Compactness.  The surface is the FIRST shell where V vanishes
        # (sec:ff-tov "The stellar surface is defined as the first shell where V
        # vanishes", and fold_transversality.py uses n_eff + 1).  This was
        # float(n_eff), one lattice spacing in, which made the printed
        # compactness ~25% low.  Shells are 1-based, so with V > 0 on 1..n_eff
        # the surface sits at n_eff + 1.
        r_star = float(n_eff + 1)
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

        print(f"\n  RESULTS (mu={mu:.6f}):")
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
    print(f"  {mu:9.6f} {N_star:8.4f} {r['n_eff']:6d} {min(d['Phi']):10.4f} "
          f"{min(d['lapse']):8.4f} {r['r_s_w']:8.3f} {r['compact_w']:8.2f} "
          f"{r['w_rel_err']:8.5f} {r['J_w_var']:8.5f} {r['rn']:10.2e}")

# Comparison table
print(f"\n  Comparison: w-fit vs Phi-fit")
print(f"  {'mu':>6s} {'r_s(w)':>8s} {'r*/r_s(w)':>10s} {'d_w%':>8s} "
      f"{'r_s(Phi)':>8s} {'r*/r_s(Phi)':>10s} {'d_Phi%':>8s}")
for mu in sorted(results.keys()):
    r = results[mu]
    print(f"  {mu:9.6f} {r['r_s_w']:8.3f} {r['compact_w']:10.2f} {r['w_rel_err']*100:8.3f} "
          f"{r['r_s_phi']:8.3f} {r['compact_phi']:10.2f} {r['phi_rel_err']*100:8.3f}")


# Save data
_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tov_tk_chemical_el.npz")
np.savez(_out,
         beta0=beta0, m0=m0, csq=csq, N=N_val,
         mu_values=[mu for mu in sorted(results.keys())],
         **{f"mu{mu:.6f}_Phi": results[mu]["d"]["Phi"] for mu in results},
         **{f"mu{mu:.6f}_lapse": results[mu]["d"]["lapse"] for mu in results},
         **{f"mu{mu:.6f}_V_defect": results[mu]["d"]["V_defect"] for mu in results},
         **{f"mu{mu:.6f}_source": results[mu]["d"]["source"] for mu in results},
         **{f"mu{mu:.6f}_kappa_kms": results[mu]["d"]["kappa_kms"] for mu in results},
         **{f"mu{mu:.6f}_n_eff": results[mu]["n_eff"] for mu in results},
         **{f"mu{mu:.6f}_r_s_w": results[mu]["r_s_w"] for mu in results},
         # The residual was never saved, so the paper's "converges to |F|_inf <
         # 1e-8" could not be checked from the data.  Save it, and the acceptance
         # thresholds that produced this file, so it can.
         **{f"mu{mu:.6f}_rn": results[mu]["rn"] for mu in results},
         **{f"mu{mu:.6f}_minN": float(np.min(results[mu]["d"]["lapse"])) for mu in results},
         **{f"mu{mu:.6f}_w_rel_err": results[mu]["w_rel_err"] for mu in results},
         **{f"mu{mu:.6f}_J_w_var": results[mu]["J_w_var"] for mu in results},
         accept_rn=1e-8, accept_min_lapse=20.0 * N_floor, accept_min_n_eff=2,
         N_floor=N_floor,
)
print(f"Saved {_out}")
print("\nDone.", flush=True)
