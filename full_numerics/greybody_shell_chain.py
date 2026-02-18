"""Greybody factors from the shell-chain scattering problem.

Default mode (transfer-matrix): solves the Regge-Wheeler equation in tortoise
coordinates via transfer matrices, giving the continuum-limit greybody factors
that the shell chain approaches for rs >> a.

Optional --lattice mode: computes transmission directly on the paper's
tight-binding chain using Landauer-Buettiker NEGF.

Outputs:
  fermion/numerical/figures/greybody_shell_chain.pdf
  fermion/numerical/data/greybody_shell_chain.npz

Run:
  python3 full_numerics/greybody_shell_chain.py
  python3 full_numerics/greybody_shell_chain.py --lattice --rs 10

"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
#  Transfer-matrix helpers (tortoise-coordinate Regge-Wheeler calculation)
# ---------------------------------------------------------------------------

def tortoise_coord(r: float | np.ndarray, rs: float) -> float | np.ndarray:
    """Tortoise coordinate r* = r + rs * ln(r/rs - 1)."""
    return r + rs * np.log(np.asarray(r, dtype=float) / rs - 1.0)


def invert_tortoise(rstar: float, rs: float) -> float:
    """Invert r*(r) = rstar for r > rs via Newton iteration."""
    # Initial guess
    if rstar > 2.0 * rs:
        r = rstar
    else:
        r = rs * (1.0 + math.exp((rstar - rs) / rs))
    for _ in range(80):
        f = r + rs * math.log(r / rs - 1.0) - rstar
        fp = r / (r - rs)
        dr = f / fp
        r -= dr
        if abs(dr) < 1e-12 * rs:
            break
    return r


def regge_wheeler_V(r: np.ndarray, rs: float, ell: int, spin: int = 0) -> np.ndarray:
    """Regge-Wheeler effective potential V = f(r)[l(l+1)/r^2 + (1-s^2) rs/r^3]."""
    r = np.asarray(r, dtype=float)
    f = 1.0 - rs / r
    return f * (ell * (ell + 1) / r**2 + (1 - spin * spin) * rs / r**3)


def transfer_matrix_transmission(omega: float, V_arr: np.ndarray, dr_star: float) -> float:
    """Transmission through V(r*) via transfer matrices.

    Solves  d^2 psi/dr*^2 + (omega^2 - V) psi = 0.
    Returns Gamma(omega) in [0, 1].
    """
    if omega <= 0.0:
        return 0.0
    omega2 = omega * omega
    M = np.eye(2)
    for V in V_arr:
        q2 = omega2 - V
        dx = dr_star
        if q2 > 0:
            q = math.sqrt(q2)
            c, s = math.cos(q * dx), math.sin(q * dx)
            Mn = np.array([[c, s / q], [-q * s, c]])
        elif q2 < 0:
            kappa = math.sqrt(-q2)
            ch, sh = math.cosh(kappa * dx), math.sinh(kappa * dx)
            Mn = np.array([[ch, sh / kappa], [kappa * sh, ch]])
        else:
            Mn = np.array([[1.0, dx], [0.0, 1.0]])
        M = Mn @ M
    # Asymptotic wavenumber (V -> 0 at both ends)
    k = omega
    denom = k * M[0, 0] + k * M[1, 1] + 1j * (k * k * M[0, 1] - M[1, 0])
    T = 4.0 * k * k / (abs(denom) ** 2)
    if not math.isfinite(T):
        return 0.0
    return float(max(0.0, min(1.0, T)))


# ---------------------------------------------------------------------------
#  Lattice NEGF helpers (tight-binding shell chain)
# ---------------------------------------------------------------------------

def surface_gf_1d(E: complex, t: float) -> complex:
    """Retarded surface Green's function for a semi-infinite 1D chain.

    Lead Hamiltonian: onsite 0, nearest-neighbor hopping -t.
    Dispersion: E = -2 t cos k.
    """
    if t <= 0:
        raise ValueError("t must be positive")
    E = complex(E)
    band = 2.0 * t
    if abs(E.real) <= band and abs(E.imag) < 1e-9:
        return (E - 1j * math.sqrt(max(0.0, band * band - E.real ** 2))) / (2.0 * t * t)
    root = np.lib.scimath.sqrt(E * E - band * band)
    g = (E - root) / (2.0 * t * t)
    if g.imag > 0:
        g = (E + root) / (2.0 * t * t)
    return complex(g)


def thomas_solve(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Solve tridiagonal Ax=d with subdiag a, diag b, superdiag c (complex)."""
    n = b.size
    ac, bc = a.astype(np.complex128).copy(), b.astype(np.complex128).copy()
    cc, dc = c.astype(np.complex128).copy(), d.astype(np.complex128).copy()
    for i in range(1, n):
        m = ac[i - 1] / bc[i - 1]
        bc[i] -= m * cc[i - 1]
        dc[i] -= m * dc[i - 1]
    x = np.zeros(n, dtype=np.complex128)
    x[-1] = dc[-1] / bc[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (dc[i] - cc[i] * x[i + 1]) / bc[i]
    return x


@dataclass(frozen=True)
class Profile:
    r: np.ndarray
    N: np.ndarray
    Nbar: np.ndarray


def make_schwarzschild_profile(rs: float, r_max: float, a: float, xi: float) -> Profile:
    """Schwarzschild-like lapse capped to N_sh = xi/(2 rs)."""
    n_sites = int(math.ceil(r_max / a))
    r = a * (np.arange(n_sites, dtype=float) + 1.0)
    N_sh = max(1e-8, xi / (2.0 * rs))
    N = np.empty_like(r)
    inside = r <= rs
    N[inside] = N_sh
    N[~inside] = np.sqrt(np.maximum(N_sh * N_sh, 1.0 - rs / r[~inside]))
    Nbar = 0.5 * (N[:-1] + N[1:])
    return Profile(r=r, N=N, Nbar=Nbar)


def transmission_tridiag(E: float, t_bonds: np.ndarray, eps: np.ndarray,
                         tL: float, tR: float, eta: float = 1e-9) -> float:
    """Landauer transmission for a tridiagonal chain with leads tL, tR."""
    n = eps.size
    if t_bonds.size != n - 1:
        raise ValueError("t_bonds must have length N-1")
    gL = surface_gf_1d(E + 1j * eta, tL)
    gR = surface_gf_1d(E + 1j * eta, tR)
    tLc, tRc = float(t_bonds[0]), float(t_bonds[-1])
    SigmaL = tLc * tLc * gL
    SigmaR = tRc * tRc * gR
    GammaL = float(-2.0 * SigmaL.imag)
    GammaR = float(-2.0 * SigmaR.imag)
    if GammaL <= 0.0 or GammaR <= 0.0:
        return 0.0
    diag = (E + 1j * eta) - eps.astype(np.complex128)
    diag[0] -= SigmaL
    diag[-1] -= SigmaR
    sup = t_bonds.astype(np.complex128)
    sub = t_bonds.astype(np.complex128)
    rhs = np.zeros(n, dtype=np.complex128)
    rhs[0] = 1.0
    col0 = thomas_solve(sub, diag, sup, rhs)
    T = GammaL * GammaR * abs(col0[-1]) ** 2
    if not np.isfinite(T):
        return 0.0
    return float(max(0.0, min(1.0, T)))


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Greybody factors from shell-chain / Regge-Wheeler scattering")
    ap.add_argument("--lattice", action="store_true", default=False,
                    help="use lattice NEGF on the tight-binding chain instead of "
                         "the default Regge-Wheeler transfer-matrix calculation")

    # Parameters common to both modes
    ap.add_argument("--rs", type=float, default=50.0, help="Schwarzschild radius r_s (in units of a)")
    ap.add_argument("--a", type=float, default=1.0, help="lattice spacing / tortoise grid spacing")
    ap.add_argument("--cstar", type=float, default=1.0, help="emergent light speed c*")
    ap.add_argument("--rmax_factor", type=float, default=20.0, help="r_max = rmax_factor * r_s")
    ap.add_argument("--lmax", type=int, default=6, help="maximum angular momentum")
    ap.add_argument("--nE", type=int, default=300, help="number of frequency/energy points")
    ap.add_argument("--emax_over_TH", type=float, default=10.0, help="omega_max / T_H")

    # Lattice-mode parameters
    ap.add_argument("--t0", type=float, default=1.0, help="asymptotic hopping (lattice mode)")
    ap.add_argument("--xi", type=float, default=1.0, help="correlation length (lattice mode)")
    ap.add_argument("--include_interior", action="store_true", default=False,
                    help="include interior r<=rs in lattice scattering region")
    ap.add_argument("--profile", choices=["analytic", "exact_solutions", "npz"],
                    default="analytic", help="lapse profile source (lattice mode)")
    ap.add_argument("--exact_path", type=str,
                    default=os.path.join("fermion", "numerical", "data", "exact_solutions.npz"))
    ap.add_argument("--V0", type=float, default=0.1000)
    ap.add_argument("--npz_path", type=str, default=None)
    args = ap.parse_args()

    rs = float(args.rs)
    a = float(args.a)
    cstar = float(args.cstar)
    T_H = cstar / (4.0 * math.pi * rs)
    l_values = list(range(0, int(args.lmax) + 1))

    out_data = os.path.join("fermion", "numerical", "data", "greybody_shell_chain.npz")
    out_fig = os.path.join("fermion", "numerical", "figures", "greybody_shell_chain.pdf")
    os.makedirs(os.path.dirname(out_data), exist_ok=True)
    os.makedirs(os.path.dirname(out_fig), exist_ok=True)

    if not args.lattice:
        # ---------------------------------------------------------------
        #  Default: Regge-Wheeler transfer-matrix in tortoise coordinates
        # ---------------------------------------------------------------
        r_inner = rs + a
        r_outer = float(args.rmax_factor) * rs
        rstar_min = float(tortoise_coord(r_inner, rs))
        rstar_max = float(tortoise_coord(r_outer, rs))

        n_sites = max(300, int((rstar_max - rstar_min) / a))
        rstar_grid = np.linspace(rstar_min, rstar_max, n_sites)
        dr_star = float(rstar_grid[1] - rstar_grid[0])

        r_grid = np.array([invert_tortoise(float(rs_val), rs) for rs_val in rstar_grid])

        omega_grid = np.linspace(T_H * 0.05, args.emax_over_TH * T_H, int(args.nE))

        Gamma = {}
        for ell in l_values:
            V_RW = regge_wheeler_V(r_grid, rs, ell, spin=0)
            Tvals = np.array([
                transfer_matrix_transmission(float(omega), V_RW, dr_star)
                for omega in omega_grid
            ])
            Gamma[ell] = Tvals

        np.savez(out_data, mode="tortoise", rs=rs, a=a, cstar=cstar, T_H=T_H,
                 rstar_grid=rstar_grid, r_grid=r_grid, omega_grid=omega_grid,
                 **{f"Gamma_l{ell}": Gamma[ell] for ell in l_values})

        fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.0))
        x = omega_grid / T_H
        for ell in l_values:
            ax.plot(x, Gamma[ell], linewidth=1.4, label=fr"$\ell={ell}$")
        ax.set_xlabel(r"$\omega/T_H$")
        ax.set_ylabel(r"Greybody factor $\Gamma_{\ell}(\omega)$")
        ax.set_title("Shell-chain greybody factors (Regge-Wheeler)")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(frameon=False, fontsize=8, ncol=2)
        ax.grid(True, alpha=0.2)

    else:
        # ---------------------------------------------------------------
        #  Lattice NEGF mode (exterior-only by default)
        # ---------------------------------------------------------------
        t0 = float(args.t0)
        xi = float(args.xi)

        if args.profile == "analytic":
            r_max = float(args.rmax_factor) * rs
            prof = make_schwarzschild_profile(rs=rs, r_max=r_max, a=a, xi=xi)
            N_sh = max(1e-8, xi / (2.0 * rs))
        elif args.profile == "exact_solutions":
            data = np.load(args.exact_path)
            prefix = f"sol_{args.V0:0.4f}"
            if f"{prefix}_lapse" not in data or f"{prefix}_Nbar" not in data:
                raise SystemExit(f"could not find keys for V0={args.V0} in {args.exact_path}")
            rs = float(data[f"{prefix}_rs"]) * a
            prof = Profile(r=a * data["r_arr"].astype(float),
                           N=data[f"{prefix}_lapse"].astype(float),
                           Nbar=data[f"{prefix}_Nbar"].astype(float))
            N_sh = float(data.get(f"{prefix}_min_N", np.min(prof.N)))
        elif args.profile == "npz":
            if args.npz_path is None:
                raise SystemExit("npz mode requires --npz_path")
            data = np.load(args.npz_path)
            prof = Profile(r=a * data["r"].astype(float),
                           N=data["N"].astype(float),
                           Nbar=data["Nbar"].astype(float))
            if "rs" in data:
                rs = float(data["rs"]) * a
            N_sh = float(np.min(prof.N))
        else:
            raise SystemExit(f"unknown profile {args.profile}")

        # Restrict to exterior.
        if not args.include_interior:
            mask = prof.r > rs
            r_ext, N_ext = prof.r[mask], prof.N[mask]
            if r_ext.size < 3:
                raise SystemExit("too few exterior sites")
            prof = Profile(r=r_ext, N=N_ext, Nbar=0.5 * (N_ext[:-1] + N_ext[1:]))

        t_bonds = t0 * prof.Nbar
        tL, tR = float(t_bonds[0]), float(t_bonds[-1])
        T_H = cstar / (4.0 * math.pi * rs)

        emax = min(args.emax_over_TH * T_H, 1.99 * 2.0 * tL)
        if emax <= 0:
            raise SystemExit("emax <= 0")
        E_grid = np.linspace(0.0, emax, int(args.nE))

        Gamma = {}
        for ell in l_values:
            V = prof.N ** 2 * ell * (ell + 1.0) / prof.r ** 2
            Tvals = np.array([transmission_tridiag(float(E), t_bonds, V, tL=tL, tR=tR)
                              for E in E_grid])
            Gamma[ell] = Tvals

        np.savez(out_data, mode="lattice", profile=args.profile, rs=rs, a=a,
                 t0=t0, cstar=cstar, xi=xi, N_sh=N_sh, T_H=T_H,
                 r=prof.r, N=prof.N, Nbar=prof.Nbar, E_grid=E_grid,
                 **{f"Gamma_l{ell}": Gamma[ell] for ell in l_values})

        fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.0))
        x = E_grid / T_H
        for ell in l_values:
            ax.plot(x, Gamma[ell], linewidth=1.4, label=fr"$\ell={ell}$")
        ax.set_xlabel(r"$E/T_H$")
        ax.set_ylabel(r"Transmission $\Gamma_{\ell}(E)$")
        ax.set_title("Greybody factors from lattice shell-chain scattering")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(frameon=False, fontsize=8, ncol=2)
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_fig)
    print(f"wrote {out_fig}")
    print(f"wrote {out_data}")


if __name__ == "__main__":
    main()
