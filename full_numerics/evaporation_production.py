#!/usr/bin/env python3
"""
Particle production and the emitted-radiation register in a self-consistent
evaporating star.

DATA script -- saves fermion/numerical/data/evaporation_production.npz
Figure script: evaporation_production_fig.py

Branch
------
The default ladder is a closure family at beta0*t0 = 0.1 with r_s/a = 15.2
down to 3.1, which is NOT the Hartle-Hawking branch beta0*t0 = 2 pi r_s/a
(eq:tov-solar-beta) -- on that branch beta0*t0 = 0.1 would mean r_s/a = 0.016,
an object smaller than one lattice site.  Nothing here should be read as an
emission rate for a physical star.  What it does test is rate-dependence: a
number-conserving quadratic drive makes pairs only non-adiabatically, and the
total it makes falls as the drive is slowed.  That is a statement about the
drive, not about the rung temperature.

Mechanism
---------
The closure gives the *base* state at each rung of a quasi-steady evolution;
it is exactly steady only if nothing changes.  Evolving under H(t) with the
hopping t0*Nbar(t) sliding as the star evaporates, the state does not track
the instantaneous eigenmodes -- it lags, and that lag is the production.

Reference states
----------------
Two references must be kept apart:

  sigma_eq(t)  = (exp(beta0 h(t)) + I)^{-1}
      instantaneous Gibbs state.  A closed unitary system does NOT relax to
      this, so G(t) - sigma_eq(t) does not vanish adiabatically and is not a
      measure of production.

  sigma_ad(t)  = V(t) diag(f_alpha(0)) V(t)^T
      the adiabatically transported state: the instantaneous eigenbasis V(t)
      carrying the INITIAL occupations.  The adiabatic theorem says
      G(t) -> sigma_ad(t) as the drive is slowed, so

          Delta(t) = G(t) - sigma_ad(t)

      is the produced population and must vanish in the strict adiabatic
      limit.  Its positive eigenvalues count produced particles, its negative
      eigenvalues count produced holes.

The tridiagonal h(t) has a simple, ordered spectrum with no crossings, so the
adiabatic map between rungs is index preserving.

Radiation register
------------------
A detector region D = shells [n_det, N) sits far outside the cap.  The
produced content that has reached D is Delta_D = G_D - sigma_ad,D, and the
entropy the radiation carries is the excess

    dS_D(t) = S(rho_D(t)) - S(rho_D^ad(t)),

with dS_S(t) the same for the complement S = [0, n_det).  Because unitary
evolution conserves the global entropy and adiabatic transport preserves the
occupations, S(G) - S(sigma_ad) = 0 identically over the whole chain; the
excess is purely a redistribution, which is exactly the emission.

Discretisation
--------------
Each rung of the closure ladder is subdivided into n_sub sub-steps with Phi
linearly interpolated and evaluated at the sub-step midpoint; each sub-step
propagator is exact (h is diagonalised), so the only error is the staircase
approximation to a continuous Phi(t).  That error is independent of the drive
rate and falls as 1/n_sub, so it is what has to be converged before any
rate dependence can be believed -- see --n-sub and --converge.

Output
------
fermion/numerical/data/evaporation_production.npz

Run
---
    python3 full_numerics/evaporation_production.py \
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


def _h(x, eps=1e-12):
    x = np.clip(np.real(x), eps, 1.0 - eps)
    return -x * np.log(x) - (1.0 - x) * np.log(1.0 - x)


def entropy_block(G, lo, hi):
    if hi <= lo:
        return 0.0
    sub = G[lo:hi, lo:hi]
    sub = 0.5 * (sub + sub.conj().T)
    return float(np.sum(_h(np.linalg.eigvalsh(sub))))


def smeared_h(Phi, t0, cstar_sq):
    """Closure's lapse-smeared single-particle Hamiltonian (tridiagonal)."""
    lapse = 1.0 + Phi / cstar_sq
    Nbar = 0.5 * (lapse[:-1] + lapse[1:])
    return np.zeros(len(Phi)), -t0 * np.abs(Nbar)


def fermi(eps, beta0):
    return 1.0 / (np.exp(np.clip(beta0 * eps, -500, 500)) + 1.0)


def rel_entropy(G, S, eps=1e-12):
    """S(rho_G || rho_S) for number-conserving Gaussian states."""
    def _terms(M):
        w, V = np.linalg.eigh(0.5 * (M + M.conj().T))
        w = np.clip(np.real(w), eps, 1.0 - eps)
        return V, np.log(w), np.log(1.0 - w)
    Gc = np.real(G)
    wg = np.clip(np.linalg.eigvalsh(0.5 * (Gc + Gc.T)), eps, 1.0 - eps)
    tr_rho_lnrho = float(np.sum(wg * np.log(wg) + (1 - wg) * np.log(1 - wg)))
    Vs, ls, l1s = _terms(S)
    lnS = (Vs * ls[None, :]) @ Vs.T
    ln1S = (Vs * l1s[None, :]) @ Vs.T
    tr_rho_lnsig = float(np.real(np.trace(Gc @ lnS))
                         + np.real(np.trace((np.eye(len(Gc)) - Gc) @ ln1S)))
    return tr_rho_lnrho - tr_rho_lnsig


def produced_modes(G, sig_ad, n_det, m_max):
    """Orthonormal basis of the m_max most strongly produced detector modes.

    Delta = G_D - sigma_ad,D; its eigenvectors with the largest |eigenvalue|
    are the modes whose occupation has been changed most by the drive.  The
    subspace is selected ONCE (at the end of the run, where production is
    largest) and then used at every time, so the register is a fixed set of
    modes filling up rather than a re-chosen one.
    """
    N = G.shape[0]
    D = G[n_det:, n_det:] - sig_ad[n_det:, n_det:]
    w, U = np.linalg.eigh(0.5 * (D + D.T))
    order = np.argsort(-np.abs(w))[:int(min(m_max, N - n_det))]
    P = np.zeros((N, len(order)))
    P[n_det:, :] = U[:, order]
    return P


def register_mi(G, P, m_list, S_tot):
    """I(P_m : complement) for nested prefixes of the fixed register basis."""
    N = G.shape[0]
    out = np.zeros(len(m_list))
    for i, m in enumerate(m_list):
        m = int(min(m, P.shape[1]))
        Pm = P[:, :m]
        W, _, _ = np.linalg.svd(Pm, full_matrices=True)
        Q = W[:, m:]
        GP = Pm.T @ G @ Pm
        GQ = Q.T @ G @ Q
        S_P = float(np.sum(_h(np.linalg.eigvalsh(0.5 * (GP + GP.T)))))
        S_Q = float(np.sum(_h(np.linalg.eigvalsh(0.5 * (GQ + GQ.T)))))
        out[i] = S_P + S_Q - S_tot
    return out


def _unused_mode_register(G, sig_ad, n_det, m_list, S_tot):
    """Mutual information between the produced-mode register and the rest.

    The produced content in the detector region is Delta = G_D - sigma_ad,D.
    Its eigenvectors with the largest |eigenvalue| are the produced modes.
    Embedding the top-m of them into the full chain gives an orthonormal
    subspace P; for a number-conserving Gaussian state any such splitting of
    the single-particle space factorises Fock space, so

        I(P : P^perp) = S_P + S_{P^perp} - S_tot

    is well defined and, unlike a spatial excess entropy, is not confounded
    by the near-maximal thermal background at beta0*t0 = 0.1.
    """
    N = G.shape[0]
    D = G[n_det:, n_det:] - sig_ad[n_det:, n_det:]
    w, U = np.linalg.eigh(0.5 * (D + D.T))
    order = np.argsort(-np.abs(w))
    out = np.zeros(len(m_list))
    for i, m in enumerate(m_list):
        m = int(min(m, N - n_det))
        P = np.zeros((N, m))
        P[n_det:, :] = U[:, order[:m]]
        # complement basis
        Q, _ = np.linalg.qr(np.eye(N) - P @ P.T)
        Q = Q[:, :N - m]
        GP = P.T @ G @ P
        GQ = Q.T @ G @ Q
        S_P = float(np.sum(_h(np.linalg.eigvalsh(0.5 * (GP + GP.conj().T)))))
        S_Q = float(np.sum(_h(np.linalg.eigvalsh(0.5 * (GQ + GQ.conj().T)))))
        out[i] = S_P + S_Q - S_tot
    return out


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--ladder-from", dest="ladder_from", required=True)
    p.add_argument("--t0", type=float, default=1.0)
    p.add_argument("--beta0", type=float, default=0.1)
    p.add_argument("--cstar-sq", dest="cstar_sq", type=float, default=0.5)
    p.add_argument("--T-evap", dest="T_evap", type=float, nargs="+",
                   default=[1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0,
                            3000.0, 10000.0])
    p.add_argument("--n-sub", dest="n_sub", type=int, default=256,
                   help="sub-steps per closure rung")
    p.add_argument("--converge", type=int, default=0,
                   help="if >0, also run at this n_sub for a convergence check")
    p.add_argument("--detectors", type=int, nargs="+",
                   default=[30, 45, 60, 100, 140])
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--outdir", type=str, default=None)
    return p


def run_rate(Phi_arr, t_norm, T_evap, n_sub, detectors, t0, beta0, cstar_sq,
             verbose=True):
    """Evolve through the ladder at one drive rate; return measurements."""
    Kc, N = Phi_arr.shape
    nd = len(detectors)

    d0, o0 = smeared_h(Phi_arr[0], t0, cstar_sq)
    eps0, V0 = eigh_tridiagonal(d0, o0)
    f_ad = fermi(eps0, beta0)                    # adiabatic reference occupations
    G = ((V0 * f_ad[None, :]) @ V0.T).astype(np.complex128)
    G0_diag = np.real(np.diag(G)).copy()

    N_prod_ad = np.zeros(Kc)
    N_prod_eq = np.zeros(Kc)
    S_glob = np.zeros(Kc)
    dS_D = np.zeros((Kc, nd))
    dS_S = np.zeros((Kc, nd))
    n_part = np.zeros((Kc, nd))
    n_hole = np.zeros((Kc, nd))
    dN_D = np.zeros((Kc, nd))
    dens_exc = np.zeros((Kc, N))
    m_list = [1, 2, 4, 8, 16, 32]
    S_rel = np.zeros((Kc, nd))
    I_reg = np.zeros((Kc, len(m_list)))     # register mutual information

    def measure(k, G, eps_k, V_k):
        # occupations in the instantaneous eigenbasis
        n_alpha = np.real(np.einsum("ia,ij,ja->a", V_k, G, V_k))
        dn_ad = n_alpha - f_ad
        dn_eq = n_alpha - fermi(eps_k, beta0)
        N_prod_ad[k] = 0.5 * float(np.sum(np.abs(dn_ad)))
        N_prod_eq[k] = 0.5 * float(np.sum(np.abs(dn_eq)))
        sig_ad = (V_k * f_ad[None, :]) @ V_k.T
        dens_exc[k] = np.real(np.diag(G)) - np.diag(sig_ad)
        Gr = np.real(G)
        w = np.linalg.eigvalsh(0.5 * (Gr + Gr.T))
        S_glob[k] = float(np.sum(_h(w)))
        for j, n_det in enumerate(detectors):
            n_det = int(n_det)
            D = Gr[n_det:, n_det:] - sig_ad[n_det:, n_det:]
            p = np.linalg.eigvalsh(0.5 * (D + D.T))
            n_part[k, j] = float(np.sum(p[p > 0]))
            n_hole[k, j] = float(-np.sum(p[p < 0]))
            dS_D[k, j] = (entropy_block(Gr, n_det, N)
                          - entropy_block(sig_ad, n_det, N))
            dS_S[k, j] = (entropy_block(Gr, 0, n_det)
                          - entropy_block(sig_ad, 0, n_det))
            dN_D[k, j] = float(np.sum(np.real(np.diag(G))[n_det:]
                                      - G0_diag[n_det:]))
            S_rel[k, j] = rel_entropy(Gr[n_det:, n_det:],
                                      sig_ad[n_det:, n_det:])
        G_hist[k] = Gr
        sig_hist[k] = sig_ad

    G_hist = np.zeros((Kc, N, N))
    sig_hist = np.zeros((Kc, N, N))

    measure(0, G, eps0, V0)

    t_start = time.time()
    for k in range(1, Kc):
        dt_rung = (t_norm[k] - t_norm[k - 1]) * T_evap
        ds = dt_rung / n_sub
        for j in range(n_sub):
            # midpoint interpolation of Phi across the rung
            frac = (j + 0.5) / n_sub
            Phi_m = (1.0 - frac) * Phi_arr[k - 1] + frac * Phi_arr[k]
            dm, om = smeared_h(Phi_m, t0, cstar_sq)
            ev, V = eigh_tridiagonal(dm, om)
            ph = np.exp(1j * ev * ds)            # G_ij = <c_i^dag c_j>: +i[h,G]
            U = (V * ph[None, :]) @ V.T
            G = U.conj().T @ G @ U
        dk, ok = smeared_h(Phi_arr[k], t0, cstar_sq)
        eps_k, V_k = eigh_tridiagonal(dk, ok)
        measure(k, G, eps_k, V_k)
        if verbose and (k % 10 == 0 or k == Kc - 1):
            print(f"    k={k:3d} t={t_norm[k]:.4f}  N_prod_ad={N_prod_ad[k]:.5e}"
                  f"  N_prod_eq={N_prod_eq[k]:.5e}  dS_D[0]={dS_D[k,0]:+.4e}"
                  f"  ({time.time()-t_start:.0f}s)", flush=True)

    # Fixed register: the modes most strongly produced by the END of the run.
    # I_reg is reported relative to its t=0 value, which is the generic
    # area-law mutual information any m-mode subspace has with the rest.
    P = produced_modes(G_hist[-1], sig_hist[-1], int(detectors[0]),
                       max(m_list))
    for k in range(Kc):
        I_reg[k] = register_mi(G_hist[k], P, m_list, S_glob[k])
    I_reg -= I_reg[0][None, :]

    return dict(N_prod_ad=N_prod_ad, N_prod_eq=N_prod_eq, S_glob=S_glob,
                dS_D=dS_D, dS_S=dS_S, n_part=n_part, n_hole=n_hole,
                dN_D=dN_D, dens_exc=dens_exc, S_rel=S_rel, I_reg=I_reg,
                m_list=np.array(m_list))


def main():
    args = build_parser().parse_args()
    old = np.load(args.ladder_from, allow_pickle=False)
    Phi_arr = old["Phi"]
    rs = old["rs"]
    V0_arr = old["V0"]
    minN = old["min_N"]
    t_norm = old["t_norm"]
    n_cap = old["n_cap"]
    g_cap = old["g_cap"]
    cuts = old["cuts"]
    I_cut_eq = old["I_cut"][-1]            # instantaneous-equilibrium alpha_S
    Kc, N = Phi_arr.shape
    print(f"ladder: {Kc} rungs, N={N}, rs {rs[0]:.3f} -> {rs[-1]:.3f}",
          flush=True)

    rates = list(args.T_evap)
    out = {k: [] for k in ("N_prod_ad", "N_prod_eq", "S_glob", "dS_D", "dS_S",
                           "n_part", "n_hole", "dN_D", "dens_exc",
                           "S_rel", "I_reg")}
    m_list = None
    for T in rates:
        print("=" * 74)
        print(f"drive: t_evap = {T:g} hbar/t0   (n_sub={args.n_sub})")
        print("=" * 74, flush=True)
        r = run_rate(Phi_arr, t_norm, T, args.n_sub, args.detectors,
                     args.t0, args.beta0, args.cstar_sq)
        for k in out:
            out[k].append(r[k])
        m_list = r["m_list"]
        print(f"  N_prod_ad final = {r['N_prod_ad'][-1]:.6e}   "
              f"max = {r['N_prod_ad'].max():.6e}   "
              f"I_reg(m=8) max = {r['I_reg'][:, 3].max():.6e}", flush=True)

    conv = {}
    if args.converge > 0:
        T = rates[-1]
        print("=" * 74)
        print(f"convergence check: t_evap={T:g}, n_sub={args.converge}")
        print("=" * 74, flush=True)
        r2 = run_rate(Phi_arr, t_norm, T, args.converge, args.detectors,
                      args.t0, args.beta0, args.cstar_sq)
        conv = dict(conv_n_sub=args.converge, conv_T_evap=T,
                    conv_N_prod_ad=r2["N_prod_ad"], conv_dS_D=r2["dS_D"])
        print(f"  n_sub={args.n_sub}: {out['N_prod_ad'][-1][-1]:.6e}   "
              f"n_sub={args.converge}: {r2['N_prod_ad'][-1]:.6e}", flush=True)

    outdir = args.outdir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "fermion", "numerical", "data")
    os.makedirs(outdir, exist_ok=True)
    name = "evaporation_production" + (f"_{args.tag}" if args.tag else "")
    path = os.path.join(outdir, name + ".npz")
    np.savez_compressed(
        path,
        N=N, t0=args.t0, beta0=args.beta0, cstar_sq=args.cstar_sq,
        n_sub=args.n_sub, T_evap=np.array(rates, dtype=float),
        detectors=np.array(args.detectors, dtype=int),
        t_norm=t_norm, rs=rs, V0=V0_arr, min_N=minN, n_cap=n_cap,
        g_cap=g_cap, cuts=cuts, I_cut_eq=I_cut_eq, m_list=m_list,
        **{k: np.array(v) for k, v in out.items()},
        **conv,
    )
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
