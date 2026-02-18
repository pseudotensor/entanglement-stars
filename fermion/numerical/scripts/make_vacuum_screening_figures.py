#!/usr/bin/env python3
"""Generate v33 figures illustrating asymptotically-vacuum Newtonian tails vs in-medium Yukawa screening.

This script is intentionally self-contained (numpy + matplotlib only) and produces:

  fermion/numerical/figures/vacuum_screening_profile.pdf

The figure has two panels:
  (Left) an illustrative radius-dependent response coefficient m_xi(r)
  (Right) the corresponding exterior potential compared to (i) Newtonian 1/r and
         (ii) a uniform-medium Yukawa tail.

Modeling choices (transparent and adjustable):
- We solve the homogeneous radial equation outside a core radius r_min:

    d/dr ( r^2 dPhi/dr ) - r^2 m_xi(r)^2 Phi = 0,

  with boundary conditions Phi(r_min) = -GM/r_min (matching a point mass inside r_min)
  and Phi(r_max)=0.

- "Uniform medium" means m_xi(r) = m0 = const, giving Yukawa-like suppression.
- "Asymptotically vacuum" means m_xi(r) is localized near the would-be horizon/core
  and decays to zero, so the far-field equation is Poisson and Phi ~ 1/r.

Response-layer thickness:
- We parameterize the response layer by an *invariant proper* microscopic thickness rho_uv
  (think: UV cutoff, e.g. lattice spacing a).
- In Schwarzschild areal coordinate r, a fixed proper thickness maps to a coordinate thickness

    delta_r ~ rho_uv^2 / (4 r_s),

  which depends on r_s because Schwarzschild r is ill-conditioned at the horizon.
  This is a coordinate artifact; rho_uv is the UV input.

Run from the project root:

  python3 fermion/numerical/scripts/make_vacuum_screening_figures.py

Optional args:

  python3 fermion/numerical/scripts/make_vacuum_screening_figures.py \
    --rs 1.0 --rho_uv 0.25 --m0 1.0 --rmax 80 --N 2500

"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def thomas_solve(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Solve tridiagonal system with Thomas algorithm.

    a: subdiagonal (n-1)
    b: diagonal (n)
    c: superdiagonal (n-1)
    d: rhs (n)
    """
    n = len(b)
    cp = np.empty(n - 1, dtype=float)
    dp = np.empty(n, dtype=float)

    # Forward sweep
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n - 1):
        denom = b[i] - a[i - 1] * cp[i - 1]
        cp[i] = c[i] / denom
        dp[i] = (d[i] - a[i - 1] * dp[i - 1]) / denom
    dp[-1] = (d[-1] - a[-1] * dp[-2]) / (b[-1] - a[-1] * cp[-2])

    # Back substitution
    x = np.empty(n, dtype=float)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - (cp[i] * x[i + 1] if i < n - 1 else 0.0)
    return x


def solve_phi(r: np.ndarray, m: np.ndarray, phi0: float, phiN: float) -> np.ndarray:
    """Solve the BVP for Phi(r) on a fixed radial grid.

    Discretizes:
      (r^2 Phi')' - r^2 m(r)^2 Phi = 0
    with Dirichlet BCs Phi(r[0])=phi0, Phi(r[-1])=phiN.

    Uses a second-order finite-difference scheme on nonuniform grids.
    """
    N = len(r)
    if len(m) != N:
        raise ValueError("m must have same length as r")
    if N < 3:
        raise ValueError("Need at least 3 grid points")

    # Interior unknowns: i = 1..N-2
    n_int = N - 2
    a = np.zeros(n_int - 1, dtype=float)  # sub
    b = np.zeros(n_int, dtype=float)      # diag
    c = np.zeros(n_int - 1, dtype=float)  # super
    d = np.zeros(n_int, dtype=float)      # rhs

    for j in range(n_int):
        i = j + 1
        rm = r[i - 1]
        r0 = r[i]
        rp = r[i + 1]

        hm = r0 - rm
        hp = rp - r0

        # Discretize flux form with variable spacing:
        # (r^2 Phi')' ≈ ( (r_{i+1/2}^2 (Phi_{i+1}-Phi_i)/hp) - (r_{i-1/2}^2 (Phi_i-Phi_{i-1})/hm) ) / ((hm+hp)/2)
        r_ph = 0.5 * (r0 + rp)
        r_mh = 0.5 * (rm + r0)

        A = r_mh**2 / hm
        C = r_ph**2 / hp
        denom = 0.5 * (hm + hp)

        # Coefficients in: -(A*(Phi_i-Phi_{i-1}) - C*(Phi_{i+1}-Phi_i))/denom + r0^2 m_i^2 Phi_i = 0
        # => (A/denom) Phi_{i-1} - ((A+C)/denom + r0^2 m_i^2) Phi_i + (C/denom) Phi_{i+1} = 0
        # Negate to get positive diagonal:
        # => (-A/denom) Phi_{i-1} + ((A+C)/denom + r0^2 m_i^2) Phi_i + (-C/denom) Phi_{i+1} = 0
        ai = -A / denom
        bi = (A + C) / denom + (r0**2) * (m[i]**2)
        ci = -C / denom

        b[j] = bi

        if j > 0:
            a[j - 1] = ai
        else:
            # boundary contribution from Phi_0
            d[j] -= ai * phi0

        if j < n_int - 1:
            c[j] = ci
        else:
            # boundary contribution from Phi_N
            d[j] -= ci * phiN

    phi_int = thomas_solve(a, b, c, d)

    phi = np.empty(N, dtype=float)
    phi[0] = phi0
    phi[-1] = phiN
    phi[1:-1] = phi_int
    return phi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rs", type=float, default=1.0, help="Schwarzschild radius scale (dimensionless)")
    ap.add_argument("--m0", type=float, default=1.0, help="Peak response coefficient m0 in the layer")
    ap.add_argument("--rmax", type=float, default=500.0, help="Outer radius for the BVP")
    ap.add_argument("--N", type=int, default=2500, help="Number of radial grid points")
    ap.add_argument("--eps", type=float, default=1e-4, help="Inner offset: r_min = r_s*(1+eps)")
    ap.add_argument("--layer_radius", type=float, default=1.2, help="Outer edge of response layer in units of r_s")
    ap.add_argument("--decay_width", type=float, default=0.5, help="Gaussian decay half-width in units of r_s")
    ap.add_argument("--out", type=str, default=str(Path("fermion/numerical/figures/vacuum_screening_profile.pdf")),
                    help="Output PDF path")
    args = ap.parse_args()

    rs = float(args.rs)
    m0 = float(args.m0)

    rmin = rs * (1.0 + float(args.eps))
    rmax = float(args.rmax)
    if rmax <= rmin:
        raise ValueError("rmax must be > rmin")

    # Response layer extends from r_s to layer_radius * r_s, then decays
    # as a Gaussian with half-width decay_width * r_s.  These are illustrative
    # choices; the physical point is that any localized m_xi(r) -> 0 gives
    # Newtonian 1/r.
    r_layer = float(args.layer_radius) * rs
    sigma = float(args.decay_width) * rs

    # Use geometric spacing to resolve the near-horizon layer while covering large radii.
    r = np.geomspace(rmin, rmax, int(args.N))

    # Localized response profile: plateau near horizon, Gaussian decay outside.
    m_loc = np.where(r <= r_layer, m0, m0 * np.exp(-0.5 * ((r - r_layer) / sigma) ** 2))

    # For comparison: uniform medium response
    m_uni = np.full_like(r, m0)

    # Boundary conditions representing a point mass/source inside rmin.
    # Set G*M = 1 in units, so Phi ~ -1/r.
    GM = 1.0
    phi0 = -GM / r[0]

    # Localized case: outer BC = 0 so the solver finds M_eff naturally.
    phi_loc = solve_phi(r, m_loc, phi0=phi0, phiN=0.0)

    # Newtonian reference matched at r[0]
    phi_newt = phi0 * (r[0] / r)

    # Yukawa reference matched at r[0]
    xiY = 1.0 / m0
    phi_yuk = phi0 * (r[0] / r) * np.exp(-(r - r[0]) / xiY)

    # Diagnostics: far-field mass renormalization (should be ~1 when m is localized)
    # M_eff(r) = -r^2 Phi'(r) / G. Use a gradient estimate.
    dphi_dr = np.gradient(phi_loc, r)
    Meff = - (r ** 2) * dphi_dr / 1.0
    Meff_far = float(np.median(Meff[-200:]))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))

    # Left: m(r) on linear scale to show clear localization
    ax[0].plot(r / rs, m_loc / m0, label=r"localized $m_\xi(r)$")
    ax[0].plot(r / rs, m_uni / m0, linestyle="--", label=r"uniform $m_\xi=m_0$")
    ax[0].set_xscale("log")
    ax[0].set_xlabel(r"$r/r_s$")
    ax[0].set_ylabel(r"$m_\xi/m_0$")
    ax[0].set_ylim(-0.05, 1.15)
    ax[0].set_title(r"Response coefficient")
    ax[0].legend(fontsize=8)

    # Right: |Phi| on log-log to show 1/r vs screened vs Yukawa
    ax[1].plot(r / rs, np.abs(phi_newt), linewidth=1.0, linestyle=":", color="gray",
               label=r"Newtonian $1/r$", zorder=0)
    ax[1].plot(r / rs, np.abs(phi_loc), linewidth=2.0,
               label=r"localized response", zorder=2)
    ax[1].plot(r / rs, np.abs(phi_yuk), linestyle="--",
               label=r"uniform Yukawa", zorder=1)

    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlim(r[0] / rs, 40.0)
    ax[1].set_ylim(1e-4, 2.0)
    ax[1].set_xlabel(r"$r/r_s$")
    ax[1].set_ylabel(r"$|\Phi|$")
    ax[1].set_title(r"Exterior potential")
    ax[1].legend(fontsize=8)

    ax[0].set_xlim(ax[1].get_xlim())

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
