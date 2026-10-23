#!/usr/bin/env python3
"""Standalone temperature sweep at ONE V0 -- a probe, not a production rerun.

Why this exists.  The production temperature sweep (compute_exact_data_gpu.py,
PHASE B2) runs V0 = 0.001 and 0.005 in shorthand units.  Those values were
chosen when the closure was sourced with the lattice-unit shorthand
beta0/c_*^2, which GROWS with beta0, so a fixed V0 ramped the bare defect and
the branch reached a fold at beta0 t0 ~ 1.65.  The closure is now sourced with
the derived prefactor P = c_*^2 t0^2/(lambda eps_b), which FALLS with beta0, so
the same V0 gives a source that weakens on cooling: the whole sweep sits at
min(N) >= 0.982, the flat-conductance proxy reproduces it to 0.44%, and nothing
interesting happens.  The grid was chosen for the old normalisation.

This script re-runs the sweep at a V0 of your choosing, into its OWN npz.  It
does not touch fermion/numerical/data/exact_solutions.npz and it does not
re-run any other phase.

Parallelism.  The V0 chains are independent: each has its own seed, its own
beta0 continuation and its own output.  They share only the compiled solver,
which is stateless.  So one process per V0, one GPU each, is exact -- there is
no cross-talk to get wrong.  (What is NOT parallel is the beta0 loop inside a
chain: each solve seeds from the previous one, and that continuation is what
lets the near-critical points converge at all.)

    python3 temp_sweep_probe.py --V0 0.066 --out /path/sweep_0.066.npz

V0 is given in SHORTHAND units (0.005 .. 0.068, the units section 11c quotes)
and converted internally, once, at the reference beta0 -- which is what holds
the physical defect fixed while beta0 varies.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax                                     # noqa: E402
import jax.numpy as jnp                        # noqa: E402
from jax import jit                            # noqa: E402

from compute_exact_data_gpu import (           # noqa: E402
    LAM_CLOSURE, convert_V0, closure_prefactor, bkm_bond_cov,
    tempsweep_background,
    make_residual_tempsweep, make_newton_solver_3arg,
    extract_rs_np, solve_analytic_proxy,
    solve_proxy_V0_continuation, solve_fixedpoint_from_proxy,
    FixedpointSolver,
)
from v0_units import flat_bond_energy           # noqa: E402

# Production parameters, identical to compute_exact_data_gpu.main().
N, T0, N_CORE, BETA0_REF, CSTAR_SQ, A = 200, 1.0, 5, 0.1, 0.5, 1.0



def flat_proxy_seed(N, t0, V0, n_core, beta0, cstar_sq):
    """Flat-conductance proxy solution: ONE banded solve, O(N), no eigh.

    Used to seed the Newton-Krylov sweep.  The seed used to come from
    solve_proxy_V0_continuation -> make_newton_solver, which forms its Jacobian
    with jacfwd and therefore needs 104*N^3 bytes: at N=800 that is 22.9 GiB and
    the run died in the seed before the fast solver ever started.

    The flat proxy is LINEAR in Phi (kappa = g t0^2 does not depend on Phi), so
    there is no Newton loop here at all -- and it reproduces the full solve to
    0.3-1.4% in weak field, which is a better starting point than the solver
    needs.
    """
    from scipy.linalg import solve_banded
    from v0_units import flat_bond_energy, LAM_CLOSURE as _LAM
    n = np.arange(1, N + 1, dtype=float)
    g = 4.0 * np.pi * n**2
    kap = g[:N - 1] * t0**2
    P = cstar_sq * t0**2 / (_LAM * flat_bond_energy(N, t0, beta0))
    rhs = np.zeros(N)
    rhs[:n_core] = -(P * V0 / 2.0) * g[:n_core]
    d = np.zeros(N); d[:N - 1] += kap; d[1:] += kap
    du = -kap.copy(); dl = -kap.copy()
    d[N - 1] = 1.0; du[N - 2] = 0.0; rhs[N - 1] = 0.0
    ab = np.zeros((3, N))
    ab[0, 1:] = du
    ab[1] = d
    ab[2, :-1] = dl
    ab[2, N - 2] = 0.0
    return solve_banded((1, 1), ab, rhs)


def make_newton_krylov_3arg(residual_fn, N, t0, cstar_sq, max_iter=60,
                            tol=1e-8, gmres_restart=2, bg_fn=None):
    """Matrix-free Newton for residual_fn(Phi, V0, beta0).

    Same outer loop as make_newton_solver_3arg -- only the LINEAR SOLVE differs.
    jacfwd forms J with N tangents through two dense eigendecompositions:
    O(N^4) work and, measured, 104*N^3 bytes (it asked for 49.71 GiB at N=800
    against 104*800^3 = 49.6 GiB), which caps N near 780 on a 49 GB card.
    Here J is never formed; GMRES needs only J.v, a single jvp.

    Preconditioner: the tridiagonal conductance Laplacian at the CURRENT
    iterate, kappa = g t0^2 Nbar^2 (BKM ratio) -- the closure operator with the
    EL and source-derivative terms dropped.  Applied with a DENSE solve; see
    the note in M() for why the native tridiagonal routine cannot be used.

    TWO THINGS THAT HAD TO BE RIGHT, both of which were wrong first time:

      * The residual pins the outer shell, F[N-1] = Phi[N-1], so the Jacobian's
        last row is e_{N-1}.  The preconditioner must match: that means zeroing
        the SUBdiagonal entry of row N-1, not the superdiagonal of row N-2.
        Getting it wrong left a singular-ish operator and the solve produced
        NaN within a few temperatures.

      * The Newton step is compiled ONCE, with Phi and the preconditioner bands
        passed as arguments.  Building the jvp and preconditioner closures
        inside the loop retraces GMRES on every iteration, which cost 11 GB and
        most of the runtime.
    """
    resid_jit = jit(residual_fn)
    n_arr = jnp.arange(1, N + 1, dtype=jnp.float64)
    g = 4.0 * jnp.pi * n_arr**2
    _z = jnp.zeros(N, dtype=jnp.float64)
    _flat_off = -t0 * jnp.ones(N - 1, dtype=jnp.float64)
    _restart = int(min(gmres_restart, N))

    @jit
    def _bands(Phi, beta0):
        lapse = 1.0 + Phi / cstar_sq
        Nbar = 0.5 * (lapse[:-1] + lapse[1:])
        off = -t0 * jnp.abs(Nbar)
        bg = bkm_bond_cov(_z, _flat_off, beta0)
        cov = bkm_bond_cov(_z, off, beta0)
        kap = g[:-1] * t0**2 * Nbar**2 * (cov / jnp.maximum(bg, 1e-30))
        d = jnp.zeros(N).at[:N - 1].add(kap).at[1:].add(kap)
        dl = jnp.concatenate([jnp.zeros(1), -kap])   # dl[i] = sub entry of row i
        du = jnp.concatenate([-kap, jnp.zeros(1)])   # du[i] = super entry of row i
        # Dirichlet row N-1 must be exactly e_{N-1}, matching the residual.
        d = d.at[N - 1].set(1.0)
        dl = dl.at[N - 1].set(0.0)
        du = du.at[N - 1].set(0.0)
        d = jnp.where(jnp.abs(d) < 1e-30, 1e-30, d)
        return dl, d, du

    @jit
    def _step(Phi, F, dl, d, du, V0, b0, bg):
        def Jv(v):
            return jax.jvp(lambda x: residual_fn(x, V0, b0, *bg),
                           (Phi,), (v,))[1]

        def M(x):
            # DENSE solve of the tridiagonal preconditioner, not
            # lax.linalg.tridiagonal_solve.  That routine loses accuracy as the
            # lattice grows: against a dense solve of the SAME matrix its
            # relative error is 1.1e-14 at N=800 but 2.8e-02 at N=1600, and a
            # preconditioner that wrong makes GMRES worse than none at all
            # (true residual 66x the starting value at k=2, and the N=1600
            # sweep simply failed to converge).  With the dense solve, k=2
            # recovers 5.5e-07 at N=1600 -- identical to N=400.
            #
            # The cost is irrelevant in context: a 1600x1600 solve is ~1 ms
            # against a jvp of ~200 ms, and it runs k=2 times per Newton step.
            # O(N^3) with a tiny constant beats O(N) with a wrong answer.
            Mm = jnp.diag(d) + jnp.diag(du[:-1], 1) + jnp.diag(dl[1:], -1)
            return jnp.linalg.solve(Mm, x)

        # k=2, maxiter=1.  Tuned END TO END, not on the linear residual:
        #   k=1: 1076 ms/pt, 2.2 Newton its  (too inexact -- costs an iteration)
        #   k=2:  754 ms/pt, 1.4 its         <- optimum
        #   k=3:  820 ms/pt, 1.4 its
        #   k=4:  896 ms/pt, 1.4 its         (pure waste beyond k=2)
        # k=2 already drives the Newton count to its floor; r_s moves by 1.3e-7
        # between k=2 and k=4, five orders below the physics.  Unpreconditioned,
        # GMRES is still at 0.96 after 32 steps -- the tridiagonal
        # preconditioner is carrying the entire solve.
        dx, _ = jax.scipy.sparse.linalg.gmres(
            Jv, -F, M=M, tol=1e-12, atol=0.0, restart=_restart, maxiter=1)
        return dx

    def solve(Phi_init, V0, beta0):
        Phi = jnp.array(Phi_init)
        V0_jax = jnp.float64(V0)
        b0_jax = jnp.float64(beta0)
        # beta0 is concrete here, so the beta0-only background is built ONCE
        # per temperature and reused by every Newton iteration and every jvp.
        bg = tuple(bg_fn(b0_jax)) if bg_fn else ()
        stagnant = 0

        for it in range(max_iter):
            F = jax.block_until_ready(resid_jit(Phi, V0_jax, b0_jax, *bg))
            norm_F = float(jnp.max(jnp.abs(F)))
            if norm_F < tol:
                return Phi, True, it, norm_F

            dl, d, du = _bands(Phi, b0_jax)
            dPhi = jax.block_until_ready(
                _step(Phi, F, dl, d, du, V0_jax, b0_jax, bg))
            if not bool(jnp.all(jnp.isfinite(dPhi))):
                break

            alpha = 1.0
            best_Phi, best_norm, improved = Phi, norm_F, False
            for _ in range(6):
                Phi_trial = Phi + alpha * dPhi
                nt = float(jnp.max(jnp.abs(jax.block_until_ready(
                    resid_jit(Phi_trial, V0_jax, b0_jax, *bg)))))
                if nt < best_norm:
                    best_Phi, best_norm, improved = Phi_trial, nt, True
                    break
                alpha *= 0.5

            if improved:
                # STAGNATION STOP.  The line search accepts any improvement at
                # all, so once the residual reaches the floor set by the
                # correlator's own precision the solve crawls -- sub-percent
                # gains for all 50 iterations, which is why large-N points
                # cost 47 s instead of 6 s.  The floor rises with N (|F| lands
                # near 1e-8 at N=800 against a 1e-8 target), so demanding the
                # fixed tolerance there buys nothing but iterations.  Stop when
                # three successive steps fail to cut the residual by 20%, and
                # report the residual actually achieved.
                if best_norm > 0.8 * norm_F:
                    stagnant += 1
                    if stagnant >= 3:
                        Phi = best_Phi
                        F = resid_jit(Phi, V0_jax, b0_jax, *bg)
                        nrm = float(jnp.max(jnp.abs(jax.block_until_ready(F))))
                        return Phi, nrm < tol, it, nrm
                else:
                    stagnant = 0
                Phi = best_Phi
            else:
                Phi = Phi + 0.01 * dPhi
                stagnant += 1
                if stagnant >= 5:
                    break
            if best_norm > 1e6 or float(jnp.any(jnp.isnan(Phi))):
                break

        F = resid_jit(Phi, V0_jax, b0_jax, *bg)
        norm_F = float(jnp.max(jnp.abs(jax.block_until_ready(F))))
        return Phi, norm_F < tol, max_iter, norm_F

    return solve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--V0", type=float, required=True,
                    help="V0 in SHORTHAND units (converted internally)")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--bt0-min", type=float, default=0.1)
    ap.add_argument("--bt0-max", type=float, default=3.0)
    ap.add_argument("--bt0-step", type=float, default=0.05)
    ap.add_argument("--N", type=int, default=None,
                    help="lattice size (default 200, the production value). "
                         "The cold end is limited by FINITE SIZE, not by the "
                         "sweep: the Fermi-level spacing is 2*pi*t0/N, and once "
                         "the thermal window 1/beta0 holds only a couple of "
                         "levels the solution develops O(1%) jitter in beta0 "
                         "while still converging to |F| ~ 1e-9.  At N=200 that "
                         "sets in near beta0 t0 = 16; the limit scales as N.")
    ap.add_argument("--solver", choices=("jacfwd", "nk"), default="jacfwd",
                    help="jacfwd = form J with N tangents (current); "
                         "nk = matrix-free Newton-Krylov")
    ap.add_argument("--reverse", action="store_true",
                    help="sweep from the COLD end toward hot. Required when V0 "
                         "is supercritical at bt0-min: V0*(bt0) RISES with "
                         "beta0 t0, so a large V0 has no solution at the hot "
                         "end to seed from, and the branch terminates as it is "
                         "heated.")
    args = ap.parse_args()

    global N
    if args.N:
        N = int(args.N)
    r_arr = A * np.arange(1, N + 1, dtype=float)
    V0_sh = float(args.V0)
    V0 = convert_V0(V0_sh, N, T0, BETA0_REF, CSTAR_SQ, LAM_CLOSURE)
    P_ref = closure_prefactor(N, T0, BETA0_REF, CSTAR_SQ, LAM_CLOSURE)
    print(f"V0(shorthand) = {V0_sh:g}   ->   V0(derived) = {V0:.10e}", flush=True)
    print(f"P at reference beta0 t0 = {BETA0_REF}: {P_ref:.4f}", flush=True)

    bt0_values = np.round(
        np.arange(args.bt0_min, args.bt0_max + 1e-9, args.bt0_step), 4)
    if args.reverse:
        bt0_values = bt0_values[::-1]
    seed_bt0 = float(bt0_values[0])

    t0_build = time.time()
    if args.solver == "nk":
        # hoist_bg=True: the two beta0-only constants are supplied once per
        # temperature by the solver, where beta0 is a concrete float, instead of
        # being recomputed inside every residual.  They must NOT be cached
        # inside the residual itself -- beta0 is a tracer there.
        resid = make_residual_tempsweep(N, T0, N_CORE, CSTAR_SQ,
                                        lam=LAM_CLOSURE, hoist_bg=True)
        solver = make_newton_krylov_3arg(resid, N, T0, CSTAR_SQ,
                                         max_iter=50, tol=1e-8,
                                         bg_fn=lambda b: tempsweep_background(
                                             N, T0, b))
    else:
        resid = make_residual_tempsweep(N, T0, N_CORE, CSTAR_SQ, lam=LAM_CLOSURE)
        solver = make_newton_solver_3arg(resid, N, max_iter=50, tol=1e-8)
    print(f"solver: {args.solver}", flush=True)
    _P0 = jnp.zeros(N, dtype=jnp.float64)
    _warm = ((jnp.float64(V0), jnp.float64(BETA0_REF))
             + (tuple(tempsweep_background(N, T0, jnp.float64(BETA0_REF)))
                if args.solver == "nk" else ()))
    jax.block_until_ready(resid(_P0, *_warm))
    print(f"solver compiled in {time.time() - t0_build:.1f}s", flush=True)

    # Seed at the hot end, exactly as PHASE B2 does when the point is not
    # already in `solutions`: proxy continuation in V0, then the full solve.
    if args.solver == "nk":
        Phi_p = flat_proxy_seed(N, T0, V0, N_CORE, seed_bt0 / T0, CSTAR_SQ)
    else:
        Phi_p, _, _, _, _, _ = solve_proxy_V0_continuation(
            N, T0, V0, N_CORE, seed_bt0 / T0, CSTAR_SQ, n_steps=12, tol=1e-8,
            verbose=False)
    if args.solver == "nk":
        # The seed MUST use the same solver as the sweep.  It used to call
        # FixedpointSolver, which is jacfwd-based and needs 104*N^3 bytes: at
        # N=800 that is 49.7 GiB, so every large-N run died in the seed before
        # the fast solver ever ran, with the GPU idle while XLA compiled the
        # doomed Jacobian.
        Phi_ex, seed_conv, _, seed_res = solver(Phi_p, V0, seed_bt0 / T0)
        if not seed_conv:
            print(f"  WARNING: seed did not converge, |F|={seed_res:.2e}",
                  flush=True)
    else:
        fp = FixedpointSolver(N, T0, N_CORE, seed_bt0 / T0, CSTAR_SQ,
                              lam=LAM_CLOSURE, max_iter=120, tol=1e-6)
        fp.warmup(N)
        Phi_ex, _, _, _ = solve_fixedpoint_from_proxy(Phi_p, V0, fp, CSTAR_SQ,
                                                      verbose=False)
    Phi_cont = np.array(Phi_ex)
    print(f"seed at bt0={seed_bt0}: "
          f"min(N)={float((1 + Phi_cont/CSTAR_SQ).min()):.4f}", flush=True)

    rows = []
    fold_reached = False
    for bt0 in bt0_values:
        if fold_reached:
            break
        this_beta0 = bt0 / T0
        Phi_j, conv_j, _, res_j = solver(
            jnp.array(Phi_cont), jnp.float64(V0), jnp.float64(this_beta0))
        Phi = np.array(jax.block_until_ready(Phi_j))
        conv = bool(conv_j)
        F = float(res_j)
        minN = float((1.0 + Phi / CSTAR_SQ).min())

        # Fold is declared on NEGATIVE LAPSE ONLY -- non-convergence alone is
        # consistent with a solver limit rather than the end of the branch.
        if minN < 0:
            print(f"  bt0={bt0:.2f}: negative lapse, min(N)={minN:.4f} "
                  f"-- branch ends", flush=True)
            fold_reached = True
            continue
        if not conv and F > 1e-6:
            # NOT called a fold.  Non-convergence alone is consistent with a
            # solver limit; the PHASE A2 convention is that only a floor-
            # independence study plus a vanishing Jacobian eigenvalue
            # establishes a fold.  Recorded as the last point reached.
            print(f"  bt0={bt0:.2f}: no convergence (|F|={F:.1e}, "
                  f"min(N)={minN:.4f}) -- branch endpoint, NOT a verified fold",
                  flush=True)
            fold_reached = True
            continue

        rs = extract_rs_np(Phi, N, N_CORE, CSTAR_SQ)
        if conv or F < 1e-6:
            Phi_cont = Phi.copy()

        Phi_p_T = solve_analytic_proxy(N, T0, V0, N_CORE, this_beta0, CSTAR_SQ)
        rs_p = extract_rs_np(Phi_p_T, N, N_CORE, CSTAR_SQ)
        eps_b = float(flat_bond_energy(N, T0, this_beta0))

        rows.append(dict(bt0=float(bt0), rs=float(rs), rs_proxy=float(rs_p),
                         minN=minN, F=F, conv=conv, eps_b=eps_b,
                         Phi=Phi, Phi_proxy=np.asarray(Phi_p_T, float)))
        print(f"  bt0={bt0:5.2f}  rs={rs:9.5f}  proxy={rs_p:9.5f}  "
              f"dev={100*abs(rs-rs_p)/max(rs,1e-30):6.2f}%  "
              f"min(N)={minN:.4f}  |F|={F:.1e}  rs*eps_b={rs*eps_b:.6f}",
              flush=True)

    if not rows:
        raise SystemExit("no converged points")

    out = dict(
        V0_shorthand=V0_sh, V0_derived=V0, N=N, t0=T0, n_core=N_CORE,
        cstar_sq=CSTAR_SQ, beta0_ref=BETA0_REF, r_arr=r_arr,
        bt0=np.array([r["bt0"] for r in rows]),
        rs=np.array([r["rs"] for r in rows]),
        rs_proxy=np.array([r["rs_proxy"] for r in rows]),
        minN=np.array([r["minN"] for r in rows]),
        F=np.array([r["F"] for r in rows]),
        conv=np.array([r["conv"] for r in rows]),
        eps_b=np.array([r["eps_b"] for r in rows]),
        Phi=np.array([r["Phi"] for r in rows]),
        Phi_proxy=np.array([r["Phi_proxy"] for r in rows]),
        fold_reached=fold_reached,
    )
    np.savez(args.out, **out)
    print(f"Saved {args.out}  ({len(rows)} points, fold_reached={fold_reached})",
          flush=True)


if __name__ == "__main__":
    main()
