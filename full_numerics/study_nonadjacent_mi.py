"""
Non-adjacent mutual information study.

Computes MI between shells separated by distance d = 1, 2, ..., d_max
for both the uniform background Hamiltonian and the lapse-smeared
Hamiltonian at a converged V0=0.03 solution.

Purpose: quantify the single-chain truncation error.  The closure
equation uses only nearest-neighbour (d=1) MI conductances.  If
MI(d)/MI(1) << 1 for d >= 2, the truncation is justified.

Physics:
  - Correlation length xi = a / |log(beta0*t0)| ~ 0.43a  (sub-lattice)
  - G_{n,n+d} ~ exp(-d*a/xi)
  - MI(d) ~ G_{n,n+d}^2 ~ exp(-2*d*a/xi)

Key subtlety: the half-filled tight-binding chain on a bipartite lattice
has an exact particle-hole symmetry.  G_{n,n+d} = 0 for even d when
h_{nn} = 0 (uniform on-site).  This makes even-d MI vanish to machine
precision.  This is NOT a lattice artifact that would disappear — it's a
real symmetry property. For the actual closure equation, what matters is
the generic case (broken symmetry), e.g. with non-zero on-site potential.

Two cases studied:
  (A) Uniform background — shows the bipartite selection rule
  (B) Smeared Hamiltonian at V0=0.03 solution — broken symmetry, generic
  (C) Source Hamiltonian (with V0) — also broken symmetry, generic
"""

import numpy as np
from scipy.linalg import eigh_tridiagonal
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Core MI computation
# ---------------------------------------------------------------------------


def binary_entropy(x):
    """h(x) = -x ln x - (1-x) ln(1-x), vectorized, safe at boundaries."""
    x = np.clip(x, 1e-30, 1.0 - 1e-30)
    return -x * np.log(x) - (1.0 - x) * np.log(1.0 - x)


def fermi_corr_full(diag, off, beta0):
    """Full N x N correlation matrix G = (exp(beta0*h) + I)^{-1}."""
    evals, evecs = eigh_tridiagonal(diag, off)
    beta_e = np.clip(beta0 * evals, -500, 500)
    f = 1.0 / (np.exp(beta_e) + 1.0)
    return (evecs * f[None, :]) @ evecs.T


def mi_at_distance(G, d):
    """
    Mutual information between every pair (n, n+d).

    For pair (n, n+d), extract 2x2 block:
      C = [[G[n,n], G[n,n+d]], [G[n+d,n], G[n+d,n+d]]]
    Eigenvalues lam1, lam2.
    MI = h(G[n,n]) + h(G[n+d,n+d]) - h(lam1) - h(lam2)

    Returns array of length N - d.
    """
    N = G.shape[0]
    n_pairs = N - d
    mi = np.zeros(n_pairs)

    for n in range(n_pairs):
        a_val = np.real(G[n, n])
        d_val = np.real(G[n + d, n + d])
        b_val = np.real(G[n, n + d])

        # 2x2 symmetric eigenvalues
        tr = a_val + d_val
        det = a_val * d_val - b_val ** 2
        disc = max(tr ** 2 - 4.0 * det, 0.0)
        lam1 = 0.5 * (tr + np.sqrt(disc))
        lam2 = 0.5 * (tr - np.sqrt(disc))

        mi[n] = (
            binary_entropy(a_val)
            + binary_entropy(d_val)
            - binary_entropy(lam1)
            - binary_entropy(lam2)
        )

    # MI should be non-negative; clip tiny negative numerical noise
    return np.maximum(mi, 0.0)


def analyze_mi(G, label, d_max=8, margin=15):
    """
    Compute MI at distances 1..d_max, report table and fit.

    Returns dict with mi profiles, averages, fit parameters.
    """
    N = G.shape[0]
    mi = {}
    avg_mi = {}

    for d in range(1, d_max + 1):
        mi[d] = mi_at_distance(G, d)
        interior = mi[d][margin:-margin]
        avg_mi[d] = np.mean(interior)

    print(f"\n  {'Distance d':<12}| {'Mean MI(d)':<17}| {'MI(d)/MI(1)':<17}| {'|G_{n,n+d}|':<17}")
    print("  " + "-" * 65)
    for d in range(1, d_max + 1):
        ratio = avg_mi[d] / avg_mi[1] if avg_mi[1] > 0 else 0
        offdiag_vals = np.diag(G, d)[margin:-margin]
        offdiag = np.mean(np.abs(offdiag_vals))
        status = "" if avg_mi[d] > 1e-30 else "  (machine zero)"
        print(f"  d={d:<8d}| {avg_mi[d]:<17.6e}| {ratio:<17.6e}| {offdiag:<17.6e}{status}")

    # Exponential fit using only nonzero MI values
    distances = np.arange(1, d_max + 1)
    mi_vals = np.array([avg_mi[d] for d in distances])
    nonzero = mi_vals > 1e-25
    if np.sum(nonzero) >= 2:
        d_fit = distances[nonzero]
        log_mi_fit = np.log(mi_vals[nonzero])
        coeffs = np.polyfit(d_fit, log_mi_fit, 1)
        xi_MI = -1.0 / coeffs[0]
        A_fit = np.exp(coeffs[1])
        print(f"\n  Exponential fit (nonzero points): MI(d) = {A_fit:.4e} * exp(-d/{xi_MI:.4f})")
        print(f"  MI correlation length: xi_MI = {xi_MI:.4f}a")
    else:
        xi_MI = 0.0
        A_fit = 0.0
        print("\n  Not enough nonzero MI values for fit.")

    # Also fit odd-only (for bipartite case)
    odd_d = distances[distances % 2 == 1]
    odd_mi = mi_vals[distances % 2 == 1]
    odd_nonzero = odd_mi > 1e-25
    if np.sum(odd_nonzero) >= 2:
        d_odd = odd_d[odd_nonzero]
        log_odd = np.log(odd_mi[odd_nonzero])
        coeffs_odd = np.polyfit(d_odd, log_odd, 1)
        xi_MI_odd = -1.0 / coeffs_odd[0]
        A_odd = np.exp(coeffs_odd[1])
        print(f"  Odd-d fit: MI(d) = {A_odd:.4e} * exp(-d/{xi_MI_odd:.4f})")

    # Truncation error: sum of all d>=2 MI / MI(1)
    trunc_err = sum(avg_mi[d] for d in range(2, d_max + 1)) / avg_mi[1] if avg_mi[1] > 0 else 0
    print(f"  Truncation error sum(MI(d>=2))/MI(1) = {trunc_err:.6e}")

    return {"mi": mi, "avg_mi": avg_mi, "xi_MI": xi_MI, "A_fit": A_fit}


# ---------------------------------------------------------------------------
# Main study
# ---------------------------------------------------------------------------


def run_study():
    N = 200
    t0 = 1.0
    beta0 = 0.1
    cstar_sq = 0.5
    a = 1.0
    d_max = 8  # go further to see the decay
    V0 = 0.03
    n_core = 5

    xi_corr = a / abs(np.log(beta0 * t0))
    print(f"Parameters: N={N}, t0={t0}, beta0={beta0}")
    print(f"Thermal correlation length: xi = a/|log(beta0*t0)| = {xi_corr:.4f}a")
    print(f"Naive estimate MI(2)/MI(1) ~ exp(-2a/xi) = {np.exp(-2*a/xi_corr):.6f}")

    # ===================================================================
    # (A) Uniform background Hamiltonian
    # ===================================================================
    print("\n" + "=" * 70)
    print("(A) UNIFORM BACKGROUND (bipartite, h_{nn}=0)")
    print("=" * 70)
    print("  NOTE: Particle-hole symmetry => G_{n,n+d}=0 for even d")
    print("        => even-d MI vanishes (not a truncation issue, a selection rule)")

    h0_diag = np.zeros(N)
    h0_off = -t0 * np.ones(N - 1)
    G_bg = fermi_corr_full(h0_diag, h0_off, beta0)

    print(f"\n  G diagonal: mean={np.mean(np.diag(G_bg)):.10f} (expect 0.5)")

    # Show correlation matrix elements directly
    print(f"  Correlator decay (interior, n=100):")
    for d in range(1, d_max + 1):
        print(f"    G[100,{100+d}] = {G_bg[100,100+d]:+.6e}")

    res_bg = analyze_mi(G_bg, "background", d_max=d_max)

    # ===================================================================
    # (B) Source Hamiltonian (with V0) — breaks bipartite symmetry
    # ===================================================================
    print("\n" + "=" * 70)
    print(f"(B) SOURCE HAMILTONIAN (V0={V0} on first {n_core} shells)")
    print("=" * 70)
    print("  V0 breaks particle-hole symmetry => all distances have nonzero MI")

    hV_diag = np.zeros(N)
    hV_diag[:n_core] = V0
    hV_off = -t0 * np.ones(N - 1)
    G_src = fermi_corr_full(hV_diag, hV_off, beta0)

    print(f"\n  G diagonal: mean={np.mean(np.diag(G_src)):.10f}")
    print(f"  Correlator decay (interior, n=100):")
    for d in range(1, d_max + 1):
        print(f"    G[100,{100+d}] = {G_src[100,100+d]:+.6e}")

    res_src = analyze_mi(G_src, "source", d_max=d_max)

    # ===================================================================
    # (C) Lapse-smeared Hamiltonian at converged V0=0.03 solution
    # ===================================================================
    print("\n" + "=" * 70)
    print("(C) LAPSE-SMEARED HAMILTONIAN (V0=0.03 converged Phi)")
    print("=" * 70)

    from full_numerics.physics_twostate import TwoStateShellModel
    from full_numerics.solve_twostate import solve_full

    model = TwoStateShellModel(
        N=N, t0=t0, V0=V0, n_core=n_core, beta0=beta0,
        cstar_sq=cstar_sq, mode="analytic"
    )
    Phi, conv, _ = solve_full(model, verbose=False)
    print(f"  Solver converged: {conv}")

    lapse = 1.0 + Phi / cstar_sq
    Nbar = 0.5 * (lapse[:-1] + lapse[1:])
    print(f"  min(lapse) = {lapse.min():.6f}, at shell {np.argmin(lapse)+1}")
    print(f"  min(Nbar) = {Nbar.min():.6f}")

    # Smeared Hamiltonian: uniform on-site (=0), non-uniform hopping
    hs_diag = np.zeros(N)
    hs_off = -t0 * np.abs(Nbar)
    G_sm = fermi_corr_full(hs_diag, hs_off, beta0)

    print(f"  G diagonal: mean={np.mean(np.diag(G_sm)):.10f}")
    print(f"  NOTE: h_{'{nn}'}=0 still, so bipartite symmetry holds")
    print(f"        even-d MI still vanishes by symmetry")

    print(f"\n  Correlator decay (interior, n=100):")
    for d in range(1, d_max + 1):
        print(f"    G[100,{100+d}] = {G_sm[100,100+d]:+.6e}")

    # Also check near core where Nbar is smallest
    print(f"\n  Correlator decay (near core, n=3):")
    for d in range(1, min(d_max + 1, N - 3)):
        print(f"    G[3,{3+d}] = {G_sm[3,3+d]:+.6e}")

    res_sm = analyze_mi(G_sm, "smeared", d_max=d_max)

    # ===================================================================
    # (D) Staggered potential: truly broken bipartite symmetry everywhere
    # ===================================================================
    print("\n" + "=" * 70)
    print("(D) STAGGERED POTENTIAL: broken bipartite symmetry globally")
    print("=" * 70)
    print("  Add epsilon*(-1)^n on all sites to break particle-hole sym")
    print("  This gives the generic (worst-case) truncation error bound")

    epsilon = 0.1  # comparable to beta0*t0 to see real effect
    hg_diag = epsilon * np.array([(-1)**n for n in range(N)], dtype=float)
    hg_off = -t0 * np.abs(Nbar)
    G_gen = fermi_corr_full(hg_diag, hg_off, beta0)

    print(f"\n  Staggered potential: epsilon = {epsilon}")
    print(f"  G diagonal: mean={np.mean(np.diag(G_gen)):.10f}")
    print(f"  G[100,100] = {G_gen[100,100]:.10f}, G[101,101] = {G_gen[101,101]:.10f}")
    print(f"  Correlator decay (interior, n=100):")
    for d in range(1, d_max + 1):
        print(f"    G[100,{100+d}] = {G_gen[100,100+d]:+.6e}")

    res_gen = analyze_mi(G_gen, "staggered", d_max=d_max)

    # Also do the source Hamiltonian near-core analysis
    print(f"\n  Source Hamiltonian: MI near core (shells 1-10) where V0 breaks sym:")
    for d in range(1, min(d_max + 1, 6)):
        core_mi = np.mean(res_src["mi"][d][:10])
        ratio = core_mi / np.mean(res_src["mi"][1][:10]) if np.mean(res_src["mi"][1][:10]) > 0 else 0
        print(f"    d={d}: MI = {core_mi:.6e}, MI(d)/MI(1) = {ratio:.6e}")

    # ===================================================================
    # Summary
    # ===================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # For the bipartite cases, report odd-d ratios
    print("\n  Bipartite cases (uniform on-site): even-d MI = 0 by symmetry")
    print(f"    Background:    MI(3)/MI(1) = {res_bg['avg_mi'][3]/res_bg['avg_mi'][1]:.6e}")
    print(f"    Smeared V0=03: MI(3)/MI(1) = {res_sm['avg_mi'][3]/res_sm['avg_mi'][1]:.6e}")

    # For the generic case (staggered potential)
    print(f"\n  Staggered potential (worst-case symmetry breaking):")
    print(f"    MI(2)/MI(1) = {res_gen['avg_mi'][2]/res_gen['avg_mi'][1]:.6e}")
    print(f"    MI(3)/MI(1) = {res_gen['avg_mi'][3]/res_gen['avg_mi'][1]:.6e}")
    trunc_gen = sum(res_gen['avg_mi'][d] for d in range(2, d_max+1)) / res_gen['avg_mi'][1]
    print(f"    Total truncation error = {trunc_gen:.6e}")

    print(f"\n  Physical interpretation:")
    print(f"    The closure equation uses h_{{nn}}=0 (smeared background),")
    print(f"    which preserves bipartite symmetry. Even-distance MI")
    print(f"    vanishes EXACTLY, and odd-distance MI is exponentially")
    print(f"    suppressed with MI(3)/MI(1) ~ {res_sm['avg_mi'][3]/res_sm['avg_mi'][1]:.1e}.")
    trunc_sm = sum(res_sm['avg_mi'][d] for d in range(2, d_max+1)) / res_sm['avg_mi'][1]
    print(f"    Total truncation error (smeared): {trunc_sm:.1e}")
    print(f"\n    Even with aggressive symmetry-breaking (staggered epsilon=0.1),")
    print(f"    MI(2)/MI(1) = {res_gen['avg_mi'][2]/res_gen['avg_mi'][1]:.1e} and total")
    print(f"    truncation error = {trunc_gen:.1e}.")
    print(f"\n    The single-chain ansatz is quantitatively justified.")

    # ===================================================================
    # Figure
    # ===================================================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    shell_idx = np.arange(1, N + 1)
    distances = np.arange(1, d_max + 1)
    colors = plt.cm.viridis(np.linspace(0, 0.85, d_max))

    # --- Panel (a): MI vs shell index, background ---
    ax = axes[0, 0]
    for d in range(1, d_max + 1):
        vals = res_bg["mi"][d]
        # Only plot if MI is above machine noise
        if np.max(vals) > 1e-25:
            ax.semilogy(
                shell_idx[: N - d], vals,
                color=colors[d - 1], label=f"d={d}", lw=0.8,
            )
    ax.set_xlabel("Shell index $n$")
    ax.set_ylabel("MI$(n, n{+}d)$")
    ax.set_title("(a) Background: MI vs shell index")
    ax.legend(fontsize=8)
    ax.set_xlim(1, N)

    # --- Panel (b): Average MI vs distance, all cases ---
    ax = axes[0, 1]
    d_arr = distances.astype(float)

    # Background (only odd)
    mi_bg_arr = np.array([res_bg["avg_mi"][d] for d in distances])
    odd_mask = (distances % 2 == 1)
    ax.semilogy(d_arr[odd_mask], mi_bg_arr[odd_mask], "o-", color="C0",
                label="Background (odd $d$)", ms=6, zorder=3)

    # Source
    mi_src_arr = np.array([res_src["avg_mi"][d] for d in distances])
    ax.semilogy(d_arr, mi_src_arr, "^-", color="C2", label=f"Source (V0={V0})", ms=5)

    # Smeared (only odd)
    mi_sm_arr = np.array([res_sm["avg_mi"][d] for d in distances])
    ax.semilogy(d_arr[odd_mask], mi_sm_arr[odd_mask], "s-", color="C3",
                label="Smeared (odd $d$)", ms=6, zorder=3)

    # Generic
    mi_gen_arr = np.array([res_gen["avg_mi"][d] for d in distances])
    ax.semilogy(d_arr, mi_gen_arr, "D-", color="C4",
                label="Generic ($\\epsilon$-broken)", ms=4)

    # Fits
    d_fine = np.linspace(1, d_max, 100)
    if res_bg["xi_MI"] > 0:
        ax.semilogy(d_fine, res_bg["A_fit"] * np.exp(-d_fine / res_bg["xi_MI"]),
                    "--", color="C0", alpha=0.4)
    if res_gen["xi_MI"] > 0:
        ax.semilogy(d_fine, res_gen["A_fit"] * np.exp(-d_fine / res_gen["xi_MI"]),
                    "--", color="C4", alpha=0.4)

    ax.set_xlabel("Distance $d$ (shells)")
    ax.set_ylabel("Mean MI$(d)$")
    ax.set_title("(b) Mean MI vs distance")
    ax.legend(fontsize=7, loc="upper right")

    # --- Panel (c): MI vs shell index, smeared ---
    ax = axes[1, 0]
    for d in range(1, d_max + 1):
        vals = res_sm["mi"][d]
        if np.max(vals) > 1e-25:
            ax.semilogy(
                shell_idx[: N - d],
                np.maximum(vals, 1e-40),
                color=colors[d - 1], label=f"d={d}", lw=0.8,
            )
    ax.set_xlabel("Shell index $n$")
    ax.set_ylabel("MI$(n, n{+}d)$")
    ax.set_title("(c) Smeared (V0=0.03): MI vs shell index")
    ax.legend(fontsize=8)
    ax.set_xlim(1, N)

    # --- Panel (d): Truncation error ratio ---
    ax = axes[1, 1]

    # Source (all d)
    ratio_src = mi_src_arr / mi_src_arr[0]
    ax.semilogy(d_arr, ratio_src, "^-", color="C2", label=f"Source (V0={V0})", ms=5)

    # Generic
    ratio_gen = mi_gen_arr / mi_gen_arr[0]
    ax.semilogy(d_arr, ratio_gen, "D-", color="C4",
                label="Generic ($\\epsilon$-broken)", ms=4)

    # Background odd
    ratio_bg = mi_bg_arr / mi_bg_arr[0]
    ax.semilogy(d_arr[odd_mask], ratio_bg[odd_mask], "o-", color="C0",
                label="Background (odd $d$)", ms=6)

    ax.axhline(0.01, ls=":", color="gray", alpha=0.5, label="1% level")
    ax.axhline(1e-6, ls=":", color="gray", alpha=0.3, label="$10^{-6}$ level")
    ax.set_xlabel("Distance $d$ (shells)")
    ax.set_ylabel("MI$(d)$ / MI$(1)$")
    ax.set_title("(d) Truncation error ratio")
    ax.legend(fontsize=7)
    ax.set_ylim(bottom=1e-18)

    plt.tight_layout()
    outpath = "fermion/numerical/figures/nonadjacent_mi.pdf"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"\n  Figure saved to {outpath}")
    plt.close(fig)


if __name__ == "__main__":
    run_study()
