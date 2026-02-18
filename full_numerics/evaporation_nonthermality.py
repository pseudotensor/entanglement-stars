"""Exact nonthermality and correlation diagnostics for unitary Gaussian evaporation.

This is a companion to full_numerics/evaporation_gaussian_dynamics.py.

In a strictly unitary evaporation of a pure state, the radiation cannot remain
exactly thermal and uncorrelated at late times: correlations must build up to
purify the total state (Page curve). Because the circuit is Gaussian and
number-conserving, all reduced states are exactly determined by correlation
matrices, so these deviations are computable without any semiclassical
assumptions.

Diagnostics computed as functions of time:
  - Radiation entropy S(R).
  - Total correlation (multi-information)
        I_tot(R) = sum_i S(i) - S(R)
    which equals relative entropy S_rel( rho_R || ⊗_i rho_i ) and vanishes iff
    rho_R factorizes across modes.
  - Early–late mutual information inside the radiation.
  - Modular-Hamiltonian nonlocality: off-diagonal Frobenius fraction of the
    single-particle modular Hamiltonian
        K_R = log[(I-G_R) G_R^{-1}].

Outputs:
  fermion/numerical/figures/evaporation_nonthermality.pdf
  fermion/numerical/data/evaporation_nonthermality.npz

Run:
  python3 full_numerics/evaporation_nonthermality.py

"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt


def random_unitary(n: int, rng: np.random.Generator) -> np.ndarray:
    """Random unitary via QR of complex Ginibre."""
    x = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / math.sqrt(2.0)
    q, r = np.linalg.qr(x)
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


def single_mode_entropy(p: float, eps: float = 1e-12) -> float:
    p = float(np.clip(p, eps, 1.0 - eps))
    return float(-(p * math.log(p) + (1.0 - p) * math.log(1.0 - p)))


def entropy_subset(g: np.ndarray, idx: list[int]) -> float:
    if len(idx) == 0:
        return 0.0
    return gaussian_entropy(g[np.ix_(idx, idx)])


@dataclass(frozen=True)
class Result:
    t_norm: np.ndarray
    s_rad: np.ndarray
    itot: np.ndarray
    i_early_late: np.ndarray
    offdiag_frac: np.ndarray
    entE_snap: dict


def simulate_metrics(n_star0: int = 120, theta: float = math.pi / 4, seed: int = 0, scramble: bool = False) -> Result:
    """Run the Gaussian evaporation circuit and compute nonthermality metrics."""
    n_rad0 = n_star0
    n_total = n_star0 + n_rad0
    rng = np.random.default_rng(seed)

    # Initial pure Slater determinant on star: random rank-n_star0/2 projector.
    n_fill = n_star0 // 2
    u_star = random_unitary(n_star0, rng)
    p = u_star[:, :n_fill] @ u_star[:, :n_fill].conj().T

    g = np.zeros((n_total, n_total), dtype=np.complex128)
    g[:n_star0, :n_star0] = p

    star_size = n_star0
    steps = n_star0

    # Times at which we snapshot the entanglement spectrum.
    t_page = 1.0 - (0.5) ** 1.5
    snap_times = [0.2, t_page, 0.9]
    snap_idx: dict[float, int] = {}

    # Precompute t_norm at each step to find nearest indices.
    t_norm_all = [1.0 - ((n_star0 - k) / n_star0) ** 1.5 for k in range(steps + 1)]
    for tt in snap_times:
        snap_idx[tt] = int(np.argmin(np.abs(np.array(t_norm_all) - tt)))

    t_norm: list[float] = []
    s_rad: list[float] = []
    itot: list[float] = []
    i_early_late: list[float] = []
    offdiag_frac: list[float] = []
    entE_snap: dict[float, np.ndarray] = {}

    def record(k: int, star_size_local: int) -> None:
        rad_idx = list(range(star_size_local, n_total))
        gr = g[np.ix_(rad_idx, rad_idx)] if rad_idx else np.zeros((0, 0), dtype=np.complex128)

        t_norm.append(t_norm_all[k])
        sR = gaussian_entropy(gr) if gr.size else 0.0
        s_rad.append(sR)

        # Total correlation: sum single-mode entropies - joint entropy.
        s_single = 0.0
        for j in range(len(rad_idx)):
            pjj = float(np.real(gr[j, j]))
            s_single += single_mode_entropy(pjj)
        itot.append(max(0.0, s_single - sR))

        # Early-late mutual information inside radiation.
        nR = len(rad_idx)
        if nR <= 1:
            i_early_late.append(0.0)
        else:
            cut = nR // 2
            early = rad_idx[:cut]
            late = rad_idx[cut:]
            i_early_late.append(entropy_subset(g, early) + entropy_subset(g, late) - sR)

        # Modular Hamiltonian off-diagonal fraction.
        if nR <= 1:
            offdiag_frac.append(0.0)
        else:
            # Diagonalize G_R.
            grh = (gr + gr.conj().T) / 2.0
            vals, vecs = np.linalg.eigh(grh)
            vals = np.clip(vals.real, 1e-12, 1.0 - 1e-12)
            entE = np.log((1.0 - vals) / vals)  # entanglement energies
            K = (vecs * entE) @ vecs.conj().T
            K = (K + K.conj().T) / 2.0
            Kdiag = np.diag(np.diag(K))
            num = float(np.linalg.norm(K - Kdiag))
            den = float(np.linalg.norm(K))
            offdiag_frac.append(0.0 if den == 0 else num / den)

            # Snapshot spectrum (sorted).
            for tt, kk in snap_idx.items():
                if k == kk and tt not in entE_snap:
                    entE_snap[tt] = np.sort(entE)

    # Record initial state.
    record(k=0, star_size_local=star_size)

    for k in range(steps):
        if scramble and star_size > 1:
            u_loc = random_unitary(star_size, rng)
            u = np.eye(n_total, dtype=np.complex128)
            u[:star_size, :star_size] = u_loc
            g[:] = u @ g @ u.conj().T

        # Emit: couple outermost star mode to a fresh radiation vacuum mode.
        e = star_size - 1
        r = n_star0 + k
        c = math.cos(theta)
        s = math.sin(theta)
        u = np.eye(n_total, dtype=np.complex128)
        u[e, e] = c
        u[r, r] = c
        u[e, r] = -s
        u[r, e] = s
        g[:] = u @ g @ u.conj().T

        # Shrink star partition.
        star_size -= 1
        record(k=k + 1, star_size_local=star_size)

    return Result(
        t_norm=np.array(t_norm),
        s_rad=np.array(s_rad),
        itot=np.array(itot),
        i_early_late=np.array(i_early_late),
        offdiag_frac=np.array(offdiag_frac),
        entE_snap={k: v for k, v in entE_snap.items()},
    )


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--n_star0", type=int, default=80, help="initial star size (modes)")
    ap.add_argument("--n_seeds", type=int, default=8, help="number of random seeds to average")
    ap.add_argument("--theta", type=float, default=math.pi / 4, help="beam-splitter angle")
    ap.add_argument("--scramble", action="store_true", help="scramble interior between emissions")
    args = ap.parse_args()

    n_star0 = int(args.n_star0)
    n_seeds = int(args.n_seeds)
    theta = float(args.theta)
    scramble = bool(args.scramble)

    all_res = [simulate_metrics(n_star0=n_star0, theta=theta, seed=s, scramble=scramble) for s in range(n_seeds)]

    t = all_res[0].t_norm
    s_rad = np.mean([r.s_rad for r in all_res], axis=0)
    itot = np.mean([r.itot for r in all_res], axis=0)
    iel = np.mean([r.i_early_late for r in all_res], axis=0)
    off = np.mean([r.offdiag_frac for r in all_res], axis=0)

    # Use a representative run for the entanglement-energy snapshots.
    snap = all_res[0].entE_snap

    t_page = 1.0 - (0.5) ** 1.5

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))

    ax = axes[0]
    ax.plot(t, itot, linewidth=1.5, label=r"$\mathcal{I}_{\mathrm{tot}}(R)$")
    ax.plot(t, iel, linewidth=1.5, label=r"$I(R_{\mathrm{early}}:R_{\mathrm{late}})$")
    ax.axvline(t_page, linestyle=":", linewidth=1, color="gray")
    ax.set_xlabel(r"Normalized time $t/t_{\mathrm{evap}}$")
    ax.set_ylabel("Information (nats)")
    ax.set_title("Radiation correlations")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ax.plot(t, off, linewidth=1.5, label=r"off-diag fraction of $K_R$")
    ax.axvline(t_page, linestyle=":", linewidth=1, color="gray")
    ax.set_xlabel(r"Normalized time $t/t_{\mathrm{evap}}$")
    ax.set_ylabel("Fraction")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Modular nonlocality")

    # Inset: entanglement-energy snapshots.
    inset = ax.inset_axes([0.52, 0.12, 0.45, 0.45])
    for tt in sorted(snap.keys()):
        ee = snap[tt]
        inset.plot(np.arange(len(ee)), ee, linewidth=1.0, label=fr"$t={tt:.2f}$")
    inset.set_title(r"$\epsilon_j=\log[(1-\nu_j)/\nu_j]$", fontsize=8)
    inset.set_xlabel("mode index", fontsize=7)
    inset.set_ylabel(r"$\epsilon_j$", fontsize=7)
    inset.tick_params(axis='both', which='major', labelsize=7)
    inset.legend(frameon=False, fontsize=6)

    ax.legend(frameon=False, fontsize=8, loc="lower left")

    fig.tight_layout()

    out_fig = os.path.join("fermion", "numerical", "figures", "evaporation_nonthermality.pdf")
    out_data = os.path.join("fermion", "numerical", "data", "evaporation_nonthermality.npz")
    os.makedirs(os.path.dirname(out_fig), exist_ok=True)
    os.makedirs(os.path.dirname(out_data), exist_ok=True)

    fig.savefig(out_fig)

    np.savez(
        out_data,
        n_star0=n_star0,
        n_seeds=n_seeds,
        theta=theta,
        scramble=scramble,
        t_norm=t,
        s_rad=s_rad,
        itot=itot,
        i_early_late=iel,
        offdiag_frac=off,
        t_page=t_page,
        **{f"entE_t{tt:.2f}": snap[tt] for tt in snap.keys()},
    )

    print(f"wrote {out_fig}")
    print(f"wrote {out_data}")


if __name__ == "__main__":
    main()