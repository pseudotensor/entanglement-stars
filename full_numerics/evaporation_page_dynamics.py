#!/usr/bin/env python3
"""
Self-consistent dynamical Page curve: unitary Gaussian evolution of sigma[Phi].

DATA script -- saves fermion/numerical/data/evaporation_page_dynamics.npz
Figure script: evaporation_page_dynamics_fig.py

What this does (and does not) assume
------------------------------------
Assumed (parameterized, not derived here):
  * the quasi-static mass-loss law M(t) = M0 (1 - t/t_evap)^{1/3}, which fixes
    only the map from the closure's V0 ladder to a time coordinate;
  * a total evaporation duration T_evap in lattice units hbar/t0, which sets
    how fast the Hamiltonian is dragged.  Scanned, not fitted.  (The emission
    rate itself -- the greybody/transmission problem -- is deliberately not
    touched here.)

Measured (no bound imposed):
  * Phi(V0), r_s(V0) from the closure (solve_exact);
  * the Gaussian state sigma[Phi] = (exp(beta0 h_smear[Phi]) + I)^{-1};
  * its exact unitary evolution G -> U G U^dagger under the closure's own
    lapse-smeared Hamiltonian h_smear[Phi(t)] as the star loses mass;
  * from the evolved G(t): the bipartition entropies S(A), S(B), the mutual
    information I(A:B) at EVERY radial cut, the area coefficient alpha_S(t)
    at the moving cap, the AMPS mutual informations, and the radiation
    entropy of the canonically purified state at every cut.

Nothing here applies min{S_max(0)-S_max(t), S_max(t)}: the Page-type
structure, if present, has to come out of the measured curves.

Output
------
fermion/numerical/data/evaporation_page_dynamics.npz

Run
---
Production (reproduces the committed .npz; ~50 min for Stage 1 at N=200):
    python3 full_numerics/evaporation_page_dynamics.py
Cross-check at the production core size (r_s/a <= 2.1, cap pinned at n=1-2):
    python3 full_numerics/evaporation_page_dynamics.py \
        --n-core 5 --V0-max 0.066 --tag nc5
Stage 2 alone, reusing a stored closure ladder:
    python3 full_numerics/evaporation_page_dynamics.py \
        --ladder-from fermion/numerical/data/evaporation_page_dynamics.npz
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from scipy.linalg import eigh_tridiagonal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from full_numerics.solve_twostate_exact import solve_exact


# ── Gaussian helpers ──────────────────────────────────────────────────────────

def _h(x, eps=1e-12):
    """Binary entropy h(x) = -x ln x - (1-x) ln(1-x), elementwise."""
    x = np.clip(np.real(x), eps, 1.0 - eps)
    return -x * np.log(x) - (1.0 - x) * np.log(1.0 - x)


def entropy_block(G, idx):
    """von Neumann entropy (nats) of a number-conserving Gaussian block."""
    if len(idx) == 0:
        return 0.0
    sub = G[np.ix_(idx, idx)]
    sub = 0.5 * (sub + sub.conj().T)
    return float(np.sum(_h(np.linalg.eigvalsh(sub))))


def mutual_info(G, a, b):
    ab = sorted(set(a) | set(b))
    return entropy_block(G, a) + entropy_block(G, b) - entropy_block(G, ab)


def smeared_h(Phi, t0, cstar_sq):
    """Closure's lapse-smeared single-particle Hamiltonian (tridiagonal).

    diag = 0, off = -t0 * |Nbar| with Nbar the bond-averaged lapse.
    Same object as smeared_corr_full() in evaporation_selfconsistent.py.
    """
    lapse = 1.0 + Phi / cstar_sq
    Nbar = 0.5 * (lapse[:-1] + lapse[1:])
    return np.zeros(len(Phi)), -t0 * np.abs(Nbar)


def thermal_G(diag, off, beta0):
    """G = (exp(beta0 h) + I)^{-1} for tridiagonal h."""
    evals, evecs = eigh_tridiagonal(diag, off)
    f = 1.0 / (np.exp(np.clip(beta0 * evals, -500, 500)) + 1.0)
    return (evecs * f[None, :]) @ evecs.T, evals, evecs


def propagate_interp(G, Phi_a, Phi_b, dt, n_sub, t0, cstar_sq):
    """Unitary Gaussian evolution across one closure rung.

    Phi is linearly interpolated from Phi_a to Phi_b and evaluated at each
    sub-step midpoint; each sub-step propagator is exact because h is
    diagonalised, so the only error is the staircase approximation to a
    continuous Phi(t), which falls as 1/n_sub and is independent of dt.

    Holding h frozen across a whole rung (n_sub = 1) is NOT the slow limit:
    it is a sudden quench followed by dephasing, and the dephasing is what
    then looks like rate dependence.  n_sub must be converged before any
    rate dependence can be believed.
    """
    if dt <= 0.0:
        return G
    ds = dt / max(1, n_sub)
    for j in range(max(1, n_sub)):
        frac = (j + 0.5) / max(1, n_sub)
        Phi_m = (1.0 - frac) * Phi_a + frac * Phi_b
        dm, om = smeared_h(Phi_m, t0, cstar_sq)
        ev, V = eigh_tridiagonal(dm, om)
        ph = np.exp(1j * ev * ds)      # G_ij = <c_i^dag c_j> obeys +i[h, G]
        U = (V * ph[None, :]) @ V.T
        G = U.conj().T @ G @ U
    return G


def canonical_purification(G):
    """Pure 2N x 2N correlation matrix purifying the Gaussian state G.

    G_tot = [[G, X], [X^dagger, I - G]] with X = sqrt(G (I-G)) (matrix
    function, basis independent).  G_tot is a projector, i.e. the doubled
    state is pure, and the ancilla labelled n is the partner of site n.
    """
    # np.real(G) here discarded the coherence the evolution generates.  The
    # imaginary part of a Hermitian G is antisymmetric and does NOT drop out of
    # the spectrum, so every entropy built on it was systematically wrong -- it
    # is what made the reported unitarity 7e-6 instead of the true 2e-15.
    G = 0.5 * (G + G.conj().T)
    w, V = np.linalg.eigh(G)
    w = np.clip(w, 0.0, 1.0)
    x = np.sqrt(w * (1.0 - w))
    X = (V * x[None, :]) @ V.conj().T
    n = G.shape[0]
    Gt = np.zeros((2 * n, 2 * n), dtype=np.complex128)
    Gt[:n, :n] = G
    Gt[:n, n:] = X
    Gt[n:, :n] = X.conj().T
    Gt[n:, n:] = np.eye(n) - G
    return Gt


# ── main ──────────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser()
    # Defaults reproduce the committed evaporation_page_dynamics.npz.
    # n_core=30 (not the production 5) because the realized Schwarzschild
    # radius at the fold grows as r_s*/a ~ 0.45 n_core: at n_core=5 the cap
    # sits within two shells of the origin and cannot be moved.
    p.add_argument("--N", type=int, default=200)
    p.add_argument("--n-core", dest="n_core", type=int, default=30)
    p.add_argument("--beta0", type=float, default=0.1)
    p.add_argument("--t0", type=float, default=1.0)
    p.add_argument("--cstar-sq", dest="cstar_sq", type=float, default=0.5)
    p.add_argument("--V0-max", dest="V0_max", type=float, default=0.00265)
    p.add_argument("--V0-min", dest="V0_min", type=float, default=3.0e-5,
                   help="bottom of the geometric tail; carries r_s -> O(a) so "
                        "the capacity actually reaches zero")
    p.add_argument("--K", type=int, default=26, help="steps in the t grid")
    p.add_argument("--K-tail", dest="K_tail", type=int, default=30,
                   help="rungs in the geometric tail")
    p.add_argument("--T-evap", dest="T_evap", type=float, nargs="+",
                   default=[1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0],
                   help="total evaporation durations in hbar/t0")
    p.add_argument("--cut-stride", dest="cut_stride", type=int, default=2)
    p.add_argument("--n-sub", dest="n_sub", type=int, default=256,
                   help="sub-steps per closure rung (see propagate_interp)")
    # Cross-check at the production core size:
    #   --n-core 5 --V0-max 0.066 --tag nc5
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--outdir", type=str, default=None)
    p.add_argument("--ladder-from", dest="ladder_from", type=str, default=None,
                   help="reuse the Stage-1 closure ladder (Phi, rs, V0) stored "
                        "in an earlier .npz instead of re-solving it")
    return p


def main():
    args = build_parser().parse_args()
    N, t0, beta0, cstar_sq = args.N, args.t0, args.beta0, args.cstar_sq
    n_core, a = args.n_core, 1.0
    K = args.K

    # ── Stage 1: quasi-static closure ladder ──────────────────────────────
    if args.ladder_from:
        old = np.load(args.ladder_from, allow_pickle=False)
        V0_arr = old["V0"]
        rs_arr = old["rs"]
        minN_arr = old["min_N"]
        Phi_arr = old["Phi"]
        Kc = len(V0_arr)
        print(f"Stage 1 reused from {args.ladder_from}: {Kc} rungs, "
              f"rs {rs_arr[0]:.3f} -> {rs_arr[-1]:.3f}", flush=True)
    else:
        # Two-part ladder.  A grid uniform in t/t_evap resolves the early
        # evaporation (M ∝ r_s ∝ V0 approximately ⇒ V0 ∝ (1-t)^{1/3}); a
        # geometric tail carries the star the rest of the way down, so the
        # capacity actually reaches zero instead of stopping while the star
        # is still there.
        t_grid = np.linspace(0.0, 0.95, K)
        V0_desc = args.V0_max * (1.0 - t_grid) ** (1.0 / 3.0)
        if args.V0_min is not None and args.V0_min < V0_desc[-1]:
            tail = np.geomspace(V0_desc[-1], args.V0_min, args.K_tail + 1)[1:]
            V0_desc = np.concatenate([V0_desc, tail])
        V0_asc = V0_desc[::-1].copy()      # ascending, for warm starts
        K = len(V0_asc)

        print("=" * 74)
        print(f"Stage 1: closure ladder  N={N} n_core={n_core} beta0={beta0} "
              f"c*^2={cstar_sq}  V0 in [{V0_asc[0]:.5f}, {V0_asc[-1]:.5f}]")
        print("=" * 74, flush=True)

        Phi_seed = None
        rows = []
        t_s1 = time.time()
        for i, V0 in enumerate(V0_asc):
            res = solve_exact(N=N, t0=t0, V0=float(V0), n_core=n_core,
                              beta0=beta0, cstar_sq=cstar_sq, tol=1e-7,
                              lapse_floor=0.01, Phi_seed=Phi_seed,
                              verbose=False)
            if not res["conv_exact"]:
                print(f"  [{i+1:2d}/{K}] V0={V0:.6f}  NOT CONVERGED "
                      f"(|F|={res['F_exact']:.2e}) -- skipped", flush=True)
                continue
            Phi_seed = res["Phi_exact"].copy()
            rows.append(dict(V0=float(V0), Phi=Phi_seed.copy(),
                             rs=float(res["rs_exact"]),
                             min_N=float(res["min_N_exact"]),
                             F=float(res["F_exact"])))
            print(f"  [{i+1:2d}/{K}] V0={V0:.6f}  rs={rows[-1]['rs']:8.4f}  "
                  f"min(N)={rows[-1]['min_N']:.5f}  |F|={rows[-1]['F']:.1e}",
                  flush=True)
        print(f"Stage 1 done: {len(rows)}/{K} converged "
              f"in {time.time()-t_s1:.0f}s")

        if len(rows) < 5:
            raise SystemExit("too few converged closures")

        rows = rows[::-1]                  # descending mass (t increasing)
        Kc = len(rows)
        V0_arr = np.array([r["V0"] for r in rows])
        rs_arr = np.array([r["rs"] for r in rows])
        minN_arr = np.array([r["min_N"] for r in rows])
        Phi_arr = np.array([r["Phi"] for r in rows])     # (Kc, N)

    rs0 = rs_arr[0]
    t_norm = 1.0 - (rs_arr / rs0) ** 3                    # M ∝ r_s
    n_cap = np.clip(np.rint(rs_arr / a).astype(int), 1, N - 2)
    g_cap = 4.0 * np.pi * (rs_arr / a) ** 2               # A_s/a^2 (continuum)

    # ── Stage 2: exact unitary Gaussian evolution ─────────────────────────
    cuts = np.arange(1, N, args.cut_stride)
    n_cuts = len(cuts)
    n_runs = len(args.T_evap) + 1        # +1 for the instantaneous-equilibrium ref

    S_in = np.zeros((n_runs, Kc, n_cuts))
    S_out = np.zeros((n_runs, Kc, n_cuts))
    I_cut = np.zeros((n_runs, Kc, n_cuts))
    SR_pure = np.zeros((n_runs, Kc, n_cuts))   # purified: radiation = outer block
    SS_pure = np.zeros((n_runs, Kc, n_cuts))   # purified: star = inner block
    S_tot = np.zeros((n_runs, Kc))
    I_cap_in = np.zeros((n_runs, Kc))
    I_cap_out = np.zeros((n_runs, Kc))
    # Monogamy observables on the EVOLVED state, so the register panel can be
    # computed from the unitary evolution rather than from a static ladder.
    # b = the cap shell, S = everything inside it, E = shells the cap has
    # vacated since the start, far = the rest of the exterior.
    I_b_S = np.zeros((n_runs, Kc))
    I_b_E = np.zeros((n_runs, Kc))
    I_b_far = np.zeros((n_runs, Kc))
    I_b_ext = np.zeros((n_runs, Kc))
    S_b = np.zeros((n_runs, Kc))
    mono_closure = np.zeros((n_runs, Kc))
    adiab = np.zeros((n_runs, Kc))             # ||G - G_eq||_F / ||G_eq||_F
    E_kin = np.zeros((n_runs, Kc))             # <H_smear> of the evolved state

    labels = [f"T_evap={T:g}" for T in args.T_evap] + ["instantaneous equilibrium"]

    # Flat-space (Phi=0) reference at the same cuts and temperature.  Dividing
    # by this separates the gravitational suppression of alpha_S from the
    # cut-position dependence any finite chain has.
    G_flat, f_flat, _ = thermal_G(np.zeros(N), -t0 * np.ones(N - 1), beta0)
    S_flat_tot = float(np.sum(_h(np.linalg.eigvalsh(0.5 * (G_flat + G_flat.T)))))
    I_cut_flat = np.zeros(n_cuts)
    for j, nc in enumerate(cuts):
        I_cut_flat[j] = (entropy_block(G_flat, list(range(int(nc))))
                         + entropy_block(G_flat, list(range(int(nc), N)))
                         - S_flat_tot)

    for ir in range(n_runs):
        equilibrium = (ir == len(args.T_evap))
        T_evap = None if equilibrium else args.T_evap[ir]
        print("=" * 74)
        print(f"Stage 2 [{ir+1}/{n_runs}]: {labels[ir]}")
        print("=" * 74, flush=True)
        t_s2 = time.time()

        diag0, off0 = smeared_h(Phi_arr[0], t0, cstar_sq)
        G, _, _ = thermal_G(diag0, off0, beta0)
        G = G.astype(np.complex128)

        for k in range(Kc):
            diag_k, off_k = smeared_h(Phi_arr[k], t0, cstar_sq)
            G_eq, ev_k, _ = thermal_G(diag_k, off_k, beta0)

            if equilibrium:
                G = G_eq.astype(np.complex128)
            elif k > 0:
                dt = (t_norm[k] - t_norm[k - 1]) * T_evap
                G = propagate_interp(G, Phi_arr[k - 1], Phi_arr[k], dt,
                                     args.n_sub, t0, cstar_sq)

            adiab[ir, k] = (np.linalg.norm(G - G_eq)
                            / max(np.linalg.norm(G_eq), 1e-30))
            hk = np.zeros((N, N))
            hk[np.arange(N - 1), np.arange(1, N)] = off_k
            hk[np.arange(1, N), np.arange(N - 1)] = off_k
            E_kin[ir, k] = float(np.real(np.trace(hk @ G)))

            Gr = 0.5 * (G + G.conj().T)
            w_all = np.linalg.eigvalsh(Gr)
            S_tot[ir, k] = float(np.sum(_h(w_all)))

            # --- monogamy on the evolved state -------------------------
            nb = int(n_cap[k])
            if 0 < nb < N - 1:
                b = [nb]
                interior = list(range(nb))
                n_shed = int(n_cap[0]) - nb
                early = list(range(nb + 1, min(nb + 1 + max(n_shed, 0), N)))
                far = list(range(nb + 1 + max(n_shed, 0), N))
                ext = list(range(nb + 1, N))
                S_b[ir, k] = entropy_block(Gr, b)
                I_b_S[ir, k] = mutual_info(Gr, b, interior)
                I_b_E[ir, k] = mutual_info(Gr, b, early) if early else 0.0
                I_b_far[ir, k] = mutual_info(Gr, b, far) if far else 0.0
                I_b_ext[ir, k] = mutual_info(Gr, b, ext)
                # The evolved global state is PURE only in the equilibrium run
                # if the purification is carried along; here the chain state is
                # mixed, so this residue is the purifier's share, exactly as in
                # the static ladder.  Recorded for the comparison, not as a
                # physical deficit.
                mono_closure[ir, k] = (2 * S_b[ir, k] - I_b_S[ir, k]
                                       - I_b_ext[ir, k])

            for j, nc in enumerate(cuts):
                A = list(range(int(nc)))
                B = list(range(int(nc), N))
                sa = entropy_block(Gr, A)
                sb = entropy_block(Gr, B)
                S_in[ir, k, j] = sa
                S_out[ir, k, j] = sb
                I_cut[ir, k, j] = sa + sb - S_tot[ir, k]

            # AMPS-like: cap shell vs interior / exterior, from the evolved state
            nc = int(n_cap[k])
            idx_cap = [nc]
            idx_in = list(range(nc))
            idx_out = list(range(nc + 1, N))
            I_cap_in[ir, k] = mutual_info(Gr, idx_cap, idx_in) if idx_in else 0.0
            I_cap_out[ir, k] = mutual_info(Gr, idx_cap, idx_out) if idx_out else 0.0

            # Canonically purified state: radiation = outer block (sites and
            # their partners), star = inner block.  Gauge note: the split of
            # the ancillas is a choice; "comoving" (partner n follows site n)
            # is used here and is the only local one.
            Gt = canonical_purification(Gr)
            for j, nc2 in enumerate(cuts):
                nc2 = int(nc2)
                S_idx = list(range(nc2)) + list(range(N, N + nc2))
                R_idx = list(range(nc2, N)) + list(range(N + nc2, 2 * N))
                SS_pure[ir, k, j] = entropy_block(Gt, S_idx)
                SR_pure[ir, k, j] = entropy_block(Gt, R_idx)

            if k % 5 == 0 or k == Kc - 1:
                jcap = int(np.argmin(np.abs(cuts - n_cap[k])))
                print(f"  k={k:3d} t={t_norm[k]:.4f} rs={rs_arr[k]:7.4f} "
                      f"n_cap={n_cap[k]:3d}  I_cap={I_cut[ir,k,jcap]:.6f}  "
                      f"S_tot={S_tot[ir,k]:.4f}  adiab={adiab[ir,k]:.2e}",
                      flush=True)
        print(f"  run done in {time.time()-t_s2:.0f}s", flush=True)

    # ── Save ──────────────────────────────────────────────────────────────
    outdir = args.outdir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "fermion", "numerical", "data")
    os.makedirs(outdir, exist_ok=True)
    name = "evaporation_page_dynamics" + (f"_{args.tag}" if args.tag else "")
    outpath = os.path.join(outdir, name + ".npz")
    np.savez_compressed(
        outpath,
        N=N, t0=t0, n_core=n_core, beta0=beta0, cstar_sq=cstar_sq, a=a,
        V0=V0_arr, rs=rs_arr, min_N=minN_arr, Phi=Phi_arr,
        t_norm=t_norm, n_cap=n_cap, g_cap=g_cap, rs0=rs0,
        cuts=cuts, T_evap=np.array(args.T_evap, dtype=float),
        labels=np.array(labels),
        S_in=S_in, S_out=S_out, I_cut=I_cut,
        SR_pure=SR_pure, SS_pure=SS_pure,
        S_tot=S_tot, I_cap_in=I_cap_in, I_cap_out=I_cap_out,
        adiab=adiab, E_kin=E_kin,
        I_b_S=I_b_S, I_b_E=I_b_E, I_b_far=I_b_far, I_b_ext=I_b_ext,
        S_b=S_b, mono_closure=mono_closure,
        I_cut_flat=I_cut_flat, S_flat_tot=S_flat_tot,
    )
    print(f"\nSaved {outpath}")


if __name__ == "__main__":
    main()
