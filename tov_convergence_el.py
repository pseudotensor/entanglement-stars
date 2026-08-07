#!/usr/bin/env python3
"""Lattice-size convergence of the TK star, using the PRODUCTION solver.

This script used to carry its own independent solver -- scipy's tridiagonal
eigensolver with a finite-difference Jacobian and the truncated residual -- and
a convergence study that uses a different solver from the one whose results are
published is testing the wrong code path.  It now imports tk_closure_solver,
the same module tov_tk_chemical_el.py runs, so the only thing varying across
the study is N_s.

The two old solvers did agree where they overlapped (min Phi = -1.254655 at
N_s = 200, mu/m0 = 0.10, to all six digits), so the discretisation conclusion
stood.  What did NOT agree was r_s: the old study fitted the exterior over a
window starting at n_eff+3 against the main scan's n_eff+4, giving 1.5082
against 1.5057.  Both now call tk_closure_solver.exterior_fit.

Why the split existed, and why it no longer has to: the production solver
differentiates a dense N x N eigendecomposition, and forward mode with N
tangents is infeasible at N_s = 3000.  The colored Jacobian removes that --
its bandwidth is set by the correlation length, which is physical and does not
grow with N_s, so the cost is ~2b+1 forward passes at every lattice size.

Three defects in the old study, all diagnosed before it was retired:
  - it judged convergence on an ABSOLUTE L2 norm.  Both sides of the closure
    carry the shell degeneracy g_n ~ n^2, and an L2 norm adds another
    sqrt(N_s), so its residual rose from 9e-9 at N_s = 50 to 1.5e-4 at 3000
    while the SOLUTION was converging.  The criterion here is the
    dimensionless |R|_inf / scale.
  - it used the truncated residual V = max(m0(1-N)-mu, 0), which stalls
    wherever a shell crosses threshold against a well-conditioned Jacobian.
    That is why every robustness set sat at 4e-8 against a 1e-8 tolerance.
  - its finite-difference Jacobian used forward differences at a step ~30x the
    optimum.  Measured: fixing that moved the floor from 1.34e-9 to 1.29e-9,
    so it was NOT the binding constraint.  What binds is the conditioning of
    the linearised closure operator -- cond(J) = 2.5e4, 1.1e5, 3.5e5 at
    N_s = 50, 100, 200 -- which is a property of the equation rather than of
    the solver, and which the production solver shares.

The old parameter-robustness study (seven (beta0, m0, csq) combinations at
fixed N_s) is not reproduced here.  Its unique content was m0 and c_*^2
variation -- beta0 is covered better by the temperature chain in
tov_tk_bt10_mu_scan.py / tov_tk_beta_chain.py, which locate the FOLD at each
temperature rather than merely showing a solution exists -- and "a solution
exists at seven scattered points" is weak next to a fold that moves
monotonically.  If that dependence is wanted, the informative version is to
track the fold in m0 and c_*^2 with the same machinery.

Outputs tov_convergence_el.npz (figures are separate, per CLAUDE.md).
"""
import sys as _sys, os as _os
_gd = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), ".taskq")
_gd2 = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".taskq")
for _d in (_gd2, _gd):
    if _os.path.isfile(_os.path.join(_d, "gate_guard.py")):
        _sys.path.insert(0, _d)
        from gate_guard import require_reread as _rr
        _rr(__file__)
        break

import os
import time

import numpy as np

from tk_closure_solver import make_solver, exterior_fit

t0 = 1.0

# Same physical point as the main scan's mid-branch, so the two are directly
# comparable at N_s = 200.
beta0_c = 5.0
m0_c = 0.03
csq_c = 5.0
mu_c = round(0.10 * m0_c, 8)
N_values = [50, 100, 200, 400, 600, 1000, 1500, 2000, 3000]

# |R|_inf / scale.  This is a SANITY bound, not a target: the achievable
# residual is set by the conditioning of the linearised closure operator, which
# grows with N_s (cond(J) = 2.5e4, 1.1e5, 3.5e5 at N_s = 50, 100, 200), so any
# fixed threshold becomes unreachable at large enough N_s no matter how good
# the solver is.  Setting it to 1e-9 -- comparable to the main scan's 1e-8
# absolute, which is 4.1e-10 relative at N_s = 200 -- made N_s = 400 reject a
# converged star at 3.2e-9 and grind through fifteen fallback seeds for twenty
# minutes to find nothing better.  The study now reports the residual it
# achieves at each N_s and only rejects genuine failures.
TOL_REL = 1e-6


def interp_seed(Phi_ref, N_new):
    """Resample a solution onto another lattice at fixed r = n a."""
    n_old = np.arange(1, len(Phi_ref) + 1, dtype=float)
    n_new = np.arange(1, N_new + 1, dtype=float)
    out = np.interp(n_new, n_old, Phi_ref, left=Phi_ref[0], right=0.0)
    out[-1] = 0.0
    return out


def main():
    print("=" * 78)
    print("CONVERGENCE with lattice size -- production solver (tk_closure_solver)")
    print(f"beta0 t0={beta0_c}, mu={mu_c:.6f} (mu/m0=0.10), m0={m0_c}, csq={csq_c}")
    print(f"accept |R|_inf/scale < {TOL_REL:g}")
    print("=" * 78, flush=True)

    ref_Phi = None
    ref_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "tov_tk_chemical_el.npz")
    try:
        ref = np.load(ref_path, allow_pickle=True)
        ref_Phi = np.asarray(ref[f"mu{mu_c:.6f}_Phi"], float)
        print(f"seeded from the main scan at N=200: minPhi={ref_Phi.min():.6f}",
              flush=True)
    except (FileNotFoundError, KeyError):
        print("no reference solution; using Gaussian seeds", flush=True)

    rows = []
    prev = ref_Phi
    _band_cache = {}
    for N in N_values:
        t_w = time.time()
        # Calibrate the colored-Jacobian bandwidth ONCE, on the smallest
        # lattice, and reuse it.  Calibration forms the exact N-tangent
        # Jacobian, which needs 26 GiB at N = 600; the band itself is set by
        # the correlation length and does not grow with N_s (48 at 200, 45 at
        # 400), so measuring it small and reusing it is the intended use.
        S = make_solver(N, beta0_c, m0_c, csq_c, band=_band_cache.get("b"))
        if _band_cache.get("b") is None:
            _band_cache["b"] = S.get("band")
        n_arr, g = np.asarray(S["n_arr"]), np.asarray(S["g"])

        seeds = []
        if prev is not None:
            seeds.append(interp_seed(prev, N))
        for amp in (0.5, 1.0, 1.5, 2.0, 3.0):
            for wd in (4.0, 8.0, 15.0):
                sd = -amp * np.exp(-0.5 * (n_arr / wd) ** 2)
                sd[-1] = 0.0
                if 1.0 + sd.min() / csq_c > 0.02:
                    seeds.append(sd)

        best = None
        for sd in seeds:
            P, rn = S["active_set_newton"](mu_c, sd.copy())
            if rn >= 1e29:
                continue
            d = S["diagnostics"](P, mu_c)
            scale = max(float(np.max(np.abs(d["source"]))), 1e-300)
            rel = rn / scale
            star = (abs(float(np.min(d["Phi"]))) > 0.01 and d["n_eff"] >= 2
                    and float(np.min(d["lapse"])) > 20 * 0.005)
            if star and (best is None or rel < best[2]):
                best = (P, d, rel, rn)
                # The continuation seed is the physically right one and the
                # floor is arithmetic, not a matter of seeding, so stop here
                # rather than searching for a residual that cannot exist.
                if rel < TOL_REL:
                    break

        dt = time.time() - t_w
        if best is None or best[2] >= TOL_REL:
            got = "none" if best is None else f"{best[2]:.2e}"
            print(f"N={N:5d}  NOT CONVERGED (best |R|rel = {got})  ({dt:.1f}s)",
                  flush=True)
            continue
        P, d, rel, rn_abs = best
        fit = exterior_fit(np.asarray(d["Phi"]), np.asarray(d["lapse"]),
                           int(d["n_eff"]), N, n_arr, g, csq_c, t0)
        rows.append(dict(N=N, minPhi=float(np.min(d["Phi"])), n_eff=int(d["n_eff"]),
                         minN=float(np.min(d["lapse"])), rel=rel, rn=rn_abs, **fit))
        print(f"N={N:5d}  minPhi={rows[-1]['minPhi']:+.6f}  r_s={fit['r_s_w']:.4f}  "
              f"n_eff={rows[-1]['n_eff']}  werr={100*fit['w_rel_err']:.2f}%  "
              f"dJw={100*fit['J_w_var']:.2f}%  |R|rel={rel:.2e}  ({dt:.1f}s)",
              flush=True)
        prev = np.asarray(d["Phi"])

    if not rows:
        raise SystemExit("no converged lattice sizes")

    print("\n" + "=" * 78)
    print(f"{'N_s':>6} {'minPhi':>12} {'r_s':>9} {'n_eff':>6} {'werr%':>7} "
          f"{'dJw%':>7} {'|R|rel':>10}")
    for r in rows:
        print(f"{r['N']:6d} {r['minPhi']:12.6f} {r['r_s_w']:9.4f} {r['n_eff']:6d} "
              f"{100*r['w_rel_err']:7.2f} {100*r['J_w_var']:7.2f} {r['rel']:10.2e}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "tov_convergence_el.npz")
    np.savez(out,
             conv_N=np.array([r["N"] for r in rows]),
             conv_minPhi=np.array([r["minPhi"] for r in rows]),
             conv_rs=np.array([r["r_s_w"] for r in rows]),
             conv_neff=np.array([r["n_eff"] for r in rows]),
             conv_minN=np.array([r["minN"] for r in rows]),
             conv_werr=np.array([r["w_rel_err"] for r in rows]),
             conv_Jwvar=np.array([r["J_w_var"] for r in rows]),
             conv_rel=np.array([r["rel"] for r in rows]),
             conv_rn=np.array([r["rn"] for r in rows]),
             beta0=beta0_c, m0=m0_c, csq=csq_c, mu=mu_c, tol_rel=TOL_REL)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
