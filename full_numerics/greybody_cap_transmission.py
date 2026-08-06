"""Transmission through the extended low-lapse cap, on the model's own chain.

A single weak bond of hopping gamma t_0 would trap, and that is the model this
script rules out: the realized cap is ~r_s/a shells deep, so the one-bond formula
does not apply.  This script computes the transmission directly, by Landauer-Buettiker
NEGF on the near-horizon chain the closure actually produces --- the exact discrete
vacuum profile w_n = 1 - (r_s/a) psi_1(n) (eq:tov-w-trigamma) with the constant-lapse
conical interior attached below the cap --- and forms the thermally weighted
transmission that enters the energy current,

    <tau> := int_0^inf dE E tau(E) n_F(E/T_H) / (pi^2 T_H^2/12),
    T_H = 1/beta_0,  beta_0 t_0 = 2 pi r_s/a   (eq:tov-solar-beta).

The per-channel energy current is then J_E = <tau> (pi/12 hbar) T_H^2 (eq:ev-JE).
What this script supplies is the radial factor <tau>, one of the two entering the
bracket C of eq:ev-bracket; <tau> = 0.999 says the cap is ballistic and carries no
power of M, so C is set by the angular barrier alone (eq:ev-dirac-radial).  The
channel count is NOT A_s/a^2: the tangential bond carries alpha^perp, which
anticommutes with the radial alpha^r, so angular channels are gapped rather than
degenerate with the radial band, and C is O(1) and mass-independent.

Also runs, for calibration, the single-weak-bond chain, where the NEGF must return
the analytic tau_F = 4 gamma^2/(1+gamma^2)^2 quoted in sec:ev-quasistatic.

Data -> greybody_cap_transmission.npz ; no figure (see CLAUDE.md convention).
"""
import math

import os
import numpy as np
from scipy.special import polygamma

t0 = 1.0


# --------------------------------------------------------------------------
#  NEGF transmission for a tridiagonal chain with matched semi-infinite leads
# --------------------------------------------------------------------------

def surface_gf(E, t, eta=1e-12):
    """Retarded surface Green's function of a semi-infinite chain, hopping -t."""
    E = complex(E, eta)
    band = 2.0 * t
    root = np.lib.scimath.sqrt(E * E - band * band)
    g = (E - root) / (2.0 * t * t)
    if g.imag > 0:
        g = (E + root) / (2.0 * t * t)
    return complex(g)


def transmission(E, t_bonds, eps=None, tL=None, tR=None, eta=1e-12):
    """Landauer transmission at energy E through a chain with the given bonds."""
    n = t_bonds.size + 1
    eps = np.zeros(n) if eps is None else eps
    tL = float(t_bonds[0]) if tL is None else float(tL)
    tR = float(t_bonds[-1]) if tR is None else float(tR)
    SigL = t_bonds[0] ** 2 * surface_gf(E, tL, eta)
    SigR = t_bonds[-1] ** 2 * surface_gf(E, tR, eta)
    GamL, GamR = -2.0 * SigL.imag, -2.0 * SigR.imag
    if GamL <= 0.0 or GamR <= 0.0:
        return 0.0
    diag = np.full(n, complex(E, eta)) - eps
    diag[0] -= SigL
    diag[-1] -= SigR
    off = -t_bonds.astype(np.complex128)   # H_{n,n+1} = -t
    # Thomas solve for column 0 of (E - H - Sigma)^{-1}
    b, d = diag.copy(), np.zeros(n, dtype=np.complex128)
    d[0] = 1.0
    for i in range(1, n):
        m = off[i - 1] / b[i - 1]
        b[i] -= m * off[i - 1]
        d[i] -= m * d[i - 1]
    x = d[-1] / b[-1]
    T = GamL * GamR * abs(x) ** 2
    return float(min(max(T, 0.0), 1.0)) if np.isfinite(T) else 0.0


# --------------------------------------------------------------------------
#  Chains
# --------------------------------------------------------------------------

def cap_chain(rs_over_a, n_out, n_in):
    """Lapse profile: constant-lapse interior + exact discrete vacuum exterior."""
    n0 = 1
    while 1.0 - rs_over_a * polygamma(1, n0) <= 0.0:
        n0 += 1
    n = np.arange(n0, n0 + n_out, dtype=float)
    N_ext = np.sqrt(1.0 - rs_over_a * polygamma(1, n))
    r = np.concatenate([np.arange(1, n_in + 1, dtype=float) * n0 / (n_in + 1.0), n])
    N = np.concatenate([np.full(n_in, N_ext[0]), N_ext])
    return N, r


def weak_bond_chain(gamma, n_side=200):
    N = np.ones(2 * n_side + 1)
    bonds = t0 * 0.5 * (N[:-1] + N[1:])
    bonds[n_side] = gamma * t0
    return bonds


# --------------------------------------------------------------------------
#  Thermal weighting
# --------------------------------------------------------------------------

def thermal_average(tau_fn, T, n_quad=240, e_max_over_T=30.0):
    """<tau> = int E tau n_F dE / int E n_F dE, the last = pi^2 T^2/12."""
    x, wq = np.polynomial.legendre.leggauss(n_quad)
    e = 0.5 * e_max_over_T * T * (x + 1.0)
    w = 0.5 * e_max_over_T * T * wq
    nf = 1.0 / (np.exp(np.clip(e / T, -700, 700)) + 1.0)
    num = np.sum(w * e * nf * np.array([tau_fn(ei) for ei in e]))
    den = np.sum(w * e * nf)
    return num / den, den / T**2


def main():
    print("calibration: single weak bond, NEGF vs analytic 4 g^2/(1+g^2)^2")
    for g in [0.5, 0.2, 0.05, 0.01]:
        b = weak_bond_chain(g)
        t_negf = transmission(0.0, b)
        print(f"  gamma={g:6.3f}  NEGF tau(0)={t_negf:.6f}  "
              f"analytic={4*g**2/(1+g**2)**2:.6f}")

    print("\ncap transmission on the model's own near-horizon chain")
    print("  N_cap = lapse at the innermost positive-lapse shell")
    print("  <tau> = thermally weighted transmission at T_H = 1/beta_0\n")
    rows = []
    for rs in [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 50000]:
        n_out = max(4000, int(20 * rs ** 0.5))
        n_in = max(2000, int(10 * rs ** 0.5))
        N, _ = cap_chain(rs, n_out, n_in)
        bonds = t0 * 0.5 * (N[:-1] + N[1:])
        T_H = 1.0 / (2.0 * np.pi * rs)              # units t_0 = 1
        tau0 = transmission(0.0, bonds)
        tavg, _ = thermal_average(lambda E: transmission(E, bonds), T_H)
        N_cap = N[n_in]
        # one-bond model values for comparison
        xi = 1.0                                     # one-bond foil only: sets
        # gamma = xi/(2 r_s) with xi taken as one lattice spacing.  This is NOT a
        # claim that the correlation length is ~a here -- the cap is cold, where xi
        # is large; the column exists only to show how far the one-bond model is off.
        g_prop = xi / (2.0 * rs)
        rows.append((rs, N_cap, T_H, tau0, tavg,
                     4 * g_prop**2 / (1 + g_prop**2)**2,
                     4 * N_cap**2 / (1 + N_cap**2)**2))
        print(f"  rs/a={rs:6d}  N_cap={N_cap:.4e}  T_H/t0={T_H:.3e}  "
              f"tau(0)={tau0:.5f}  <tau>={tavg:.5f}   "
              f"[1-bond: xi/2rs -> {4*g_prop**2:.2e},  N_cap -> {4*N_cap**2:.2e}]",
              flush=True)

    r = np.array(rows, dtype=float)
    _here = os.path.dirname(os.path.abspath(__file__))
    _out = os.path.normpath(os.path.join(_here, "..", "fermion", "numerical", "data", "greybody_cap_transmission.npz"))
    np.savez(_out, rs_over_a=r[:, 0], N_cap=r[:, 1],
             T_H=r[:, 2], tau_zero=r[:, 3], tau_thermal=r[:, 4],
             tau_onebond_proper=r[:, 5], tau_onebond_Ncap=r[:, 6])

    print("\nscaling of <tau> with r_s/a (a power p means L propto M^{p-2}):")
    lr, lt = np.log(r[:, 0]), np.log(r[:, 4])
    for i in range(1, len(lr)):
        print(f"  r_s/a {r[i-1,0]:6.0f} -> {r[i,0]:6.0f}:  "
              f"d log<tau>/d log r_s = {(lt[i]-lt[i-1])/(lr[i]-lr[i-1]):+.4f}")
    print(f"\noverall fit over the full range: {np.polyfit(lr, lt, 1)[0]:+.4f}")


if __name__ == "__main__":
    main()
