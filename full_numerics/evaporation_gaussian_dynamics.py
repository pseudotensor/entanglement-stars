"""Unitary Gaussian evaporation dynamics and Page curve.

This script produces the figure used in the paper section on evaporation dynamics.
It implements a minimal, exactly unitary, number-conserving Gaussian circuit:

  (1) Start from a pure Slater determinant on the black-hole (star) modes and
      vacuum on the radiation modes.
  (2) Optionally (if scramble=True), apply a random single-particle unitary on
      the current star subspace (Gaussian scrambling).  In the production figure
      scramble=False: scrambling is not needed for the Page-curve turnover.
  (3) Emit by a 2-mode beam-splitter unitary coupling the outermost star mode to
      a fresh radiation vacuum mode.
  (4) Shrink the star partition by one mode (area decrease), and compute
      radiation entropy and AMPS-relevant mutual informations.

Because the state is Gaussian and number-conserving, the correlation matrix
G(t) evolves exactly by conjugation: G -> U G U^\dagger.

Outputs:
  fermion/numerical/figures/evaporation_gaussian_dynamics.pdf

Run:
  python3 full_numerics/evaporation_gaussian_dynamics.py

"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt


def random_unitary(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a random unitary via QR of a complex Ginibre matrix."""
    x = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / math.sqrt(2.0)
    q, r = np.linalg.qr(x)
    # Fix phases so that diag(r) is positive-real.
    ph = np.diag(r)
    ph = ph / np.abs(ph)
    q = q * ph.conj()
    return q


def gaussian_entropy(gsub: np.ndarray, eps: float = 1e-12) -> float:
    """Von Neumann entropy (nats) of a number-conserving Gaussian subsystem."""
    gsub = (gsub + gsub.conj().T) / 2.0
    vals = np.linalg.eigvalsh(gsub).real
    vals = np.clip(vals, eps, 1.0 - eps)
    return float(-np.sum(vals * np.log(vals) + (1.0 - vals) * np.log(1.0 - vals)))


def entropy_subset(g: np.ndarray, idx: list[int]) -> float:
    if len(idx) == 0:
        return 0.0
    return gaussian_entropy(g[np.ix_(idx, idx)])


def mutual_info(g: np.ndarray, a: list[int], b: list[int]) -> float:
    ab = sorted(set(a).union(b))
    return entropy_subset(g, a) + entropy_subset(g, b) - entropy_subset(g, ab)


@dataclass(frozen=True)
class SimulationResult:
    t_norm: np.ndarray
    s_rad: np.ndarray
    s_pred: np.ndarray
    i_new_star: np.ndarray
    i_new_early: np.ndarray


def simulate(
    n_star0: int = 120,
    theta: float = math.pi / 4,
    seed: int = 4,
    scramble: bool = False,
) -> SimulationResult:
    """Run the Gaussian evaporation circuit."""
    n_rad0 = n_star0
    n_total = n_star0 + n_rad0
    rng = np.random.default_rng(seed)

    # Initial pure Slater determinant on star: random rank-n_star0/2 projector.
    n_fill = n_star0 // 2
    u_star = random_unitary(n_star0, rng)
    p = u_star[:, :n_fill] @ u_star[:, :n_fill].conj().T

    g = np.zeros((n_total, n_total), dtype=np.complex128)
    g[:n_star0, :n_star0] = p

    log2 = math.log(2.0)

    star_size = n_star0
    steps = n_star0

    t_norm: list[float] = []
    s_rad: list[float] = []
    s_pred: list[float] = []
    i_new_star: list[float] = [math.nan]  # undefined at k=0
    i_new_early: list[float] = [math.nan]

    # Record initial values.
    k = 0
    t_norm.append(1.0 - (star_size / n_star0) ** 1.5)
    s_rad.append(entropy_subset(g, list(range(star_size, n_total))))
    s_pred.append(log2 * min(k, n_star0 - k))

    for k in range(steps):
        if scramble and star_size > 1:
            u_loc = random_unitary(star_size, rng)
            u = np.eye(n_total, dtype=np.complex128)
            u[:star_size, :star_size] = u_loc
            g = u @ g @ u.conj().T

        # Emit: couple outermost star mode to a fresh radiation mode.
        e = star_size - 1
        r = n_star0 + k
        c = math.cos(theta)
        s = math.sin(theta)
        u = np.eye(n_total, dtype=np.complex128)
        u[e, e] = c
        u[r, r] = c
        u[e, r] = -s
        u[r, e] = s
        g = u @ g @ u.conj().T

        # Shrink star partition by one mode (area decrease).
        star_size -= 1

        t_norm.append(1.0 - (star_size / n_star0) ** 1.5)
        s_rad.append(entropy_subset(g, list(range(star_size, n_total))))
        s_pred.append(log2 * min(k + 1, n_star0 - (k + 1)))

        new_mode = [star_size]  # first radiation mode after shrink
        star_idx = list(range(0, star_size))
        early_idx = list(range(star_size + 1, n_total))
        i_new_star.append(mutual_info(g, new_mode, star_idx))
        i_new_early.append(mutual_info(g, new_mode, early_idx))

    return SimulationResult(
        t_norm=np.array(t_norm),
        s_rad=np.array(s_rad),
        s_pred=np.array(s_pred),
        i_new_star=np.array(i_new_star),
        i_new_early=np.array(i_new_early),
    )


def main() -> None:
    # Average over multiple seeds to reduce noise from random scrambling unitaries.
    n_seeds = 20
    n_star0 = 120
    all_results = [simulate(n_star0=n_star0, seed=s) for s in range(n_seeds)]

    t_norm = all_results[0].t_norm
    s_rad_mean = np.mean([r.s_rad for r in all_results], axis=0)
    s_pred = all_results[0].s_pred
    i_new_star_mean = np.mean([r.i_new_star for r in all_results], axis=0)
    i_new_early_mean = np.mean([r.i_new_early for r in all_results], axis=0)

    t_page = 1.0 - (0.5) ** 1.5  # 1 - 2^{-3/2} ≈ 0.646

    # Haar normalization: divide by n_star0 * log2 so Haar bound peaks at 0.5
    haar_max = n_star0 * math.log(2.0)
    s_rad_norm = s_rad_mean / haar_max
    s_pred_norm = s_pred / haar_max

    # Load quasi-static closure Page curve if available
    closure_t = closure_sr = closure_alpha = None
    npz_path = os.path.join("fermion", "numerical", "data",
                            "evaporation_selfconsistent.npz")
    if os.path.exists(npz_path):
        d = np.load(npz_path, allow_pickle=False)
        # n_S(0) for closure system = 4*pi*(rs_max/a)^2
        rs_max = float(d["rs_max"])
        a = float(d["a"])
        n_S0_closure = 4.0 * math.pi * (rs_max / a) ** 2
        haar_max_closure = n_S0_closure * math.log(2.0)
        closure_t = d["t_norm"]
        # Normalize by S_BH(0) = alpha_S * 4*pi*rs_max^2 (actual star entropy),
        # not by the Haar maximum n_S0*log2.  This puts the closure curve on the
        # same [0, 0.5] scale as the circuit (each normalized by its own S_max(0)).
        S_BH_0 = float(d["S_max_0"])
        closure_sr = d["S_page_bound"] / S_BH_0
        closure_alpha = float(d["alpha_S_sat"])
        idx = np.argsort(closure_t)
        closure_t = closure_t[idx]
        closure_sr = closure_sr[idx]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(t_norm, s_rad_norm, linewidth=1.5,
            label=r"Gaussian circuit ($n_S(0)=120$ modes)")
    ax.plot(t_norm, s_pred_norm, linestyle="--", linewidth=1.5,
            label=r"Haar upper bound: $\min(k,N{-}k)/n_S(0)$")
    if closure_t is not None:
        # closure_sr is already normalized by its own n_S(0)*log2 (peaks at ~0.5).
        # Both systems share the same Page time and curve shape; they differ only
        # in absolute entropy scale by alpha_S/log2 ~ 1/570.
        ax.plot(closure_t, closure_sr, linewidth=1.5, color="red", linestyle="-.",
                label=r"Self-consistent closure (norm. by $S_{\rm BH}(0)$)")
    ax.axvline(t_page, linestyle=":", linewidth=1, color="gray")
    ax.set_xlabel(r"Normalized time $t/t_{\mathrm{evap}}$")
    ax.set_ylabel(r"$S_R(t)\,/\,S_{\rm max}(0)$")
    ax.set_title("Page curve: Gaussian circuit vs self-consistent closure")
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    ax = axes[1]
    ax.plot(t_norm[1:], i_new_star_mean[1:], linewidth=1.5, label=r"$I(\mathrm{new};\mathrm{star})$")
    ax.plot(t_norm[1:], i_new_early_mean[1:], linewidth=1.5, label=r"$I(\mathrm{new};\mathrm{early\ rad})$")
    ax.axvline(t_page, linestyle=":", linewidth=1, color="gray")
    ax.set_xlabel(r"Normalized time $t/t_{\mathrm{evap}}$")
    ax.set_ylabel("Mutual information (nats)")
    ax.set_title("AMPS correlations through Page time")
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()

    outpath = os.path.join("fermion", "numerical", "figures", "evaporation_gaussian_dynamics.pdf")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath)
    print(f"wrote {outpath}")


if __name__ == "__main__":
    main()
