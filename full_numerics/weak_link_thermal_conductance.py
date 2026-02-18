"""Weak-link thermal conductance from exact Gaussian dynamics.

Purpose
-------
The evaporation section of the paper uses the Landauer/Pendry 1D energy-current
formula per channel and multiplies it by an effective transmission probability
\tau across the near-horizon "cap" (a weak link created by lapse-smearing of the
hopping).

A key modelling step is the scaling \tau \propto N_{\mathrm{sh}}^2 for small
stretched-horizon lapse N_{\mathrm{sh}}.  This script validates (and calibrates)
that scaling *inside the same microscopic free-fermion model*, by measuring the
steady energy current through a single weak bond.

Model
-----
- A 1D tight-binding chain with hopping t0 everywhere except a single bond with
  hopping t_w = gamma * t0.
- Initial state: product of thermal states of the decoupled left and right
  halves at temperatures T_L and T_R.
- Quench at t=0: couple the halves by switching on the weak bond.
- Evolve unitarily (Gaussian, number-conserving) under the full quadratic
  Hamiltonian.

We measure the energy current into the right half via

  J_E(t) = d/dt <H_R> = -i Tr( [H_R, H] G(t) )

and define an effective transmission

  tau_eff(gamma) = <J_E>_plateau / ( (pi/12) (T_L^2 - T_R^2) ),

where the denominator is the universal ballistic 1D result per channel.

Outputs
-------
- fermion/numerical/data/weak_link_conductance.npz
- fermion/numerical/figures/weak_link_conductance.pdf

Run
---
  python3 full_numerics/weak_link_thermal_conductance.py

The defaults are chosen to run quickly (seconds) while keeping the temperature
low enough that the universal formula is accurate.

"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt


def fermi_occupation(eps: np.ndarray, beta: float, mu: float = 0.0) -> np.ndarray:
    """Stable Fermi-Dirac occupation for possibly large beta."""
    x = beta * (eps - mu)
    # Avoid overflow.
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(x))


def chain_hamiltonian(L: int, t0: float = 1.0) -> np.ndarray:
    """Single-particle Hamiltonian h for open 1D chain: H = c^\dagger h c."""
    h = np.zeros((L, L), dtype=np.float64)
    for i in range(L - 1):
        h[i, i + 1] = -t0
        h[i + 1, i] = -t0
    return h


def thermal_correlation(h: np.ndarray, T: float, mu: float = 0.0) -> np.ndarray:
    """Correlation matrix G = <c^\dagger c> for quadratic Hamiltonian at temperature T."""
    if T <= 0.0:
        beta = 1e6
    else:
        beta = 1.0 / T
    eps, v = np.linalg.eigh(h)
    occ = fermi_occupation(eps, beta=beta, mu=mu)
    # G = v occ v^T
    return (v * occ) @ v.T


@dataclass(frozen=True)
class CurrentResult:
    gammas: np.ndarray
    tau_eff: np.ndarray
    tau_scatt: np.ndarray
    c_small: float


def energy_current_plateau(
    L_left: int,
    L_right: int,
    t0: float,
    gamma: float,
    T_left: float,
    T_right: float,
    mu: float,
    t_max: float,
    dt: float,
    t_plateau_min: float,
    t_plateau_max: float,
) -> float:
    """Compute plateau-averaged energy current into the right half."""

    L = L_left + L_right

    # Initial state: decoupled halves.
    hL = chain_hamiltonian(L_left, t0=t0)
    hR_half = chain_hamiltonian(L_right, t0=t0)
    GL = thermal_correlation(hL, T=T_left, mu=mu)
    GR = thermal_correlation(hR_half, T=T_right, mu=mu)

    G0 = np.zeros((L, L), dtype=np.float64)
    G0[:L_left, :L_left] = GL
    G0[L_left:, L_left:] = GR

    # Full Hamiltonian after quench: uniform hopping except weak link.
    h = chain_hamiltonian(L, t0=t0)
    h[L_left - 1, L_left] = -t0 * gamma
    h[L_left, L_left - 1] = -t0 * gamma

    # Right-half Hamiltonian H_R (internal bonds only; no weak link).
    hR = np.zeros((L, L), dtype=np.float64)
    hR[L_left:, L_left:] = hR_half

    # Commutator C = [hR, h] (single-particle).
    C = hR @ h - h @ hR  # real skew-symmetric

    # Diagonalize h once.
    eps, V = np.linalg.eigh(h)

    # Precompute in eigenbasis.
    A = V.T @ C @ V
    B = V.T @ G0 @ V

    # Matrix of eigenvalue differences.
    de = eps[:, None] - eps[None, :]

    # Elementwise product M_{ab} = A_{ab} B_{ba}.
    M = A * B.T

    # Time grid.
    times = np.arange(0.0, t_max + 1e-12, dt)
    mask = (times >= t_plateau_min) & (times <= t_plateau_max)
    if not np.any(mask):
        raise ValueError("Empty plateau window; adjust t_plateau_min/max.")

    # Compute current at each time using the eigenbasis phase sum.
    J = np.zeros_like(times, dtype=np.float64)
    for i, t in enumerate(times):
        phase = np.exp(1j * de * t)
        tr = np.sum(M * phase)  # complex
        J[i] = float((-1j * tr).real)

    return float(np.mean(J[mask]))


def run_sweep(args: argparse.Namespace) -> CurrentResult:
    gammas = np.array(args.gammas, dtype=np.float64)

    J_ball = (math.pi / 12.0) * (args.T_left**2 - args.T_right**2)
    if J_ball <= 0:
        raise ValueError("Need T_left > T_right >= 0.")

    tau_eff = []
    for g in gammas:
        Jp = energy_current_plateau(
            L_left=args.L_left,
            L_right=args.L_right,
            t0=args.t0,
            gamma=float(g),
            T_left=args.T_left,
            T_right=args.T_right,
            mu=args.mu,
            t_max=args.t_max,
            dt=args.dt,
            t_plateau_min=args.t_plateau_min,
            t_plateau_max=args.t_plateau_max,
        )
        tau_eff.append(Jp / J_ball)

    tau_eff = np.array(tau_eff, dtype=np.float64)

    # Analytic scattering transmission at the Fermi point (k = pi/2):
    # tau_F = 4 gamma^2 / (1 + gamma^2)^2.
    tau_scatt = 4.0 * gammas**2 / (1.0 + gammas**2) ** 2

    # Small-gamma coefficient c_small from least squares on gammas<=g_fit.
    fit_mask = gammas <= args.g_fit_max
    if np.sum(fit_mask) < 2:
        fit_mask = gammas <= 0.3
    c_small = float(np.mean(tau_eff[fit_mask] / (gammas[fit_mask] ** 2 + 1e-30)))

    return CurrentResult(gammas=gammas, tau_eff=tau_eff, tau_scatt=tau_scatt, c_small=c_small)


def save_outputs(res: CurrentResult, args: argparse.Namespace) -> None:
    out_data = os.path.join("fermion", "numerical", "data", "weak_link_conductance.npz")
    os.makedirs(os.path.dirname(out_data), exist_ok=True)
    np.savez(
        out_data,
        gammas=res.gammas,
        tau_eff=res.tau_eff,
        tau_scatt=res.tau_scatt,
        c_small=res.c_small,
        params=dict(
            L_left=args.L_left,
            L_right=args.L_right,
            t0=args.t0,
            T_left=args.T_left,
            T_right=args.T_right,
            mu=args.mu,
            t_max=args.t_max,
            dt=args.dt,
            t_plateau_min=args.t_plateau_min,
            t_plateau_max=args.t_plateau_max,
            g_fit_max=args.g_fit_max,
        ),
    )

    fig = plt.figure(figsize=(6.4, 4.2))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(res.gammas, res.tau_eff, marker="o", linewidth=1.5, label=r"$\tau_{\mathrm{eff}}$ (Gaussian quench)")
    ax.plot(res.gammas, res.tau_scatt, linestyle="--", linewidth=1.5, label=r"$\tau_{\mathrm{F}}=4\gamma^2/(1+\gamma^2)^2$")

    # Small-gamma fit line.
    g_line = np.linspace(0.0, float(np.max(res.gammas)), 200)
    ax.plot(g_line, res.c_small * g_line**2, linestyle=":", linewidth=1.2, label=rf"small-$\gamma$ fit: $\tau\approx {res.c_small:.2f}\,\gamma^2$")

    ax.set_xlabel(r"Weak-link ratio $\gamma=t_w/t_0$")
    ax.set_ylabel(r"Effective transmission $\tau$")
    ax.set_title("Weak-link thermal conductance from Gaussian dynamics")
    ax.set_ylim(bottom=0.0, top=1.05)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()

    out_fig = os.path.join("fermion", "numerical", "figures", "weak_link_conductance.pdf")
    os.makedirs(os.path.dirname(out_fig), exist_ok=True)
    fig.savefig(out_fig)

    print(f"wrote {out_data}")
    print(f"wrote {out_fig}")
    print(f"small-gamma fit coefficient: c_small = {res.c_small:.4f}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Measure weak-link thermal conductance from Gaussian dynamics")
    p.add_argument("--L-left", dest="L_left", type=int, default=60)
    p.add_argument("--L-right", dest="L_right", type=int, default=60)
    p.add_argument("--t0", type=float, default=1.0)
    p.add_argument("--T-left", dest="T_left", type=float, default=0.08)
    p.add_argument("--T-right", dest="T_right", type=float, default=0.0)
    p.add_argument("--mu", type=float, default=0.0)
    p.add_argument("--t-max", dest="t_max", type=float, default=20.0)
    p.add_argument("--dt", type=float, default=0.2)
    p.add_argument("--t-plateau-min", dest="t_plateau_min", type=float, default=6.0)
    p.add_argument("--t-plateau-max", dest="t_plateau_max", type=float, default=14.0)
    p.add_argument("--g-fit-max", dest="g_fit_max", type=float, default=0.30)
    p.add_argument(
        "--gammas",
        type=float,
        nargs="+",
        default=[0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.50, 0.70, 1.00],
        help="weak-link ratios gamma=t_w/t0",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()
    res = run_sweep(args)
    save_outputs(res, args)


if __name__ == "__main__":
    main()
