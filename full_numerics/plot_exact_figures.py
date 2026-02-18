#!/usr/bin/env python3
"""
Generate all 15 publication figures from pre-computed exact_solutions.npz.

Figures 1-12: core analysis (potential, cometric, PN, floor, interior,
              embedding, thermo, conductance, lapse, TQI, summary, verify)
Figure 13: temperature sweep (with proxy rs overlay)
Figure 14: combined cometric + conductance + PN residual (3-panel)
Figure 15: exact vs analytic comparison (with proxy overlay)

Reads: fermion/numerical/data/exact_solutions.npz
Writes: fermion/numerical/figures/*.pdf

Run:  python3 full_numerics/plot_exact_figures.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os, sys
from scipy.linalg import solve_banded, eigh_tridiagonal

def compute_proxy_Phi(NN, t0, V0, n_core, this_beta0, cstar_sq):
    """Flat-conductance proxy: single tridiagonal solve."""
    n = np.arange(1, NN + 1, dtype=float)
    g = 4.0 * np.pi * n**2
    kappa_flat = g[:NN - 1] * t0**2
    # rhs = (beta0/c*^2)*(rho_bg - rho_tgt) = -(beta0*V0/(2*c*^2))*g*1_core
    rhs = np.zeros(NN)
    rhs[:n_core] = -(this_beta0 * V0 / (2.0 * cstar_sq)) * g[:n_core]
    # Build tridiagonal Laplacian
    sub_L = np.zeros(NN - 1)
    diag_L = np.zeros(NN)
    sup_L = np.zeros(NN - 1)
    for nn in range(NN - 1):
        diag_L[nn] += kappa_flat[nn]
        sup_L[nn] -= kappa_flat[nn]
        diag_L[nn + 1] += kappa_flat[nn]
        sub_L[nn] -= kappa_flat[nn]
    # BC: Phi_N = 0
    diag_L[NN - 1] = 1.0
    sub_L[NN - 2] = 0.0
    rhs[NN - 1] = 0.0
    ab = np.zeros((3, NN))
    ab[0, 1:] = sup_L
    ab[1, :] = diag_L
    ab[2, :-1] = sub_L
    return solve_banded((1, 1), ab, rhs)


def compute_alpha_S(NN, t0, this_beta0, n_cut=None):
    """Per-mode entanglement entropy alpha_S for the uniform flat chain.

    alpha_S = I_1ch / 2, where I_1ch is the bipartition mutual information
    of a single radial channel (1D tight-binding chain at half filling).
    See Remark in 10e_thermodynamics.tex, eq. (sc-alphaS-MI).
    """
    if n_cut is None:
        n_cut = NN // 2

    def binary_entropy(x, eps=1e-14):
        x = np.clip(x, eps, 1.0 - eps)
        return -x * np.log(x) - (1.0 - x) * np.log(1.0 - x)

    diag = np.zeros(NN)
    off = -t0 * np.ones(NN - 1)
    evals, evecs = eigh_tridiagonal(diag, off)
    beta_e = np.clip(this_beta0 * evals, -500, 500)
    f = 1.0 / (np.exp(beta_e) + 1.0)
    G = (evecs * f[None, :]) @ evecs.T

    G_A = G[:n_cut, :n_cut]
    G_B = G[n_cut:, n_cut:]
    nu_A = np.linalg.eigvalsh(G_A)
    nu_B = np.linalg.eigvalsh(G_B)
    S_A = np.sum(binary_entropy(nu_A))
    S_B = np.sum(binary_entropy(nu_B))
    S_AB = np.sum(binary_entropy(f))
    I_1ch = S_A + S_B - S_AB

    return I_1ch / 2.0


# ── Style ─────────────────────────────────────────────────────────
rcParams.update({
    "text.usetex": False, "font.family": "serif", "font.size": 11,
    "axes.labelsize": 13, "axes.titlesize": 13, "legend.fontsize": 9,
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "lines.linewidth": 1.5,
})

BASEDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(BASEDIR, "fermion", "numerical", "figures")
DATADIR = os.path.join(BASEDIR, "fermion", "numerical", "data")
os.makedirs(FIGDIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────
npzpath = os.path.join(DATADIR, "exact_solutions.npz")
print(f"Loading: {npzpath}")
D = np.load(npzpath, allow_pickle=True)

# Parameters
N = int(D["N"])
t0 = float(D["t0"])
n_core = int(D["n_core"])
beta0 = float(D["beta0"])
cstar_sq = float(D["cstar_sq"])
a = float(D["a"])
r_arr = D["r_arr"]
r_bond = D["r_bond"]

# V0 lists
V0_core = list(D["V0_core"])
V0_extended = list(D["V0_extended"])
V0_supercrit = list(D["V0_supercrit"])
solution_V0s = list(D["solution_V0s"])


# ── Helpers to read per-V0 data from npz ──────────────────────────

def get_sol(V0):
    """Get exact solution data for a V0 value."""
    tag = f"sol_{V0:.4f}"
    return {
        "Phi": D[f"{tag}_Phi"],
        "lapse": D[f"{tag}_lapse"],
        "Nbar": D[f"{tag}_Nbar"],
        "kappa": D[f"{tag}_kappa"],
        "kappa_flat": D[f"{tag}_kappa_flat"],
        "rs": float(D[f"{tag}_rs"]),
        "min_N": float(D[f"{tag}_min_N"]),
        "converged": bool(D[f"{tag}_converged"]),
        "residual": float(D[f"{tag}_residual"]),
    }


def get_proxy(V0):
    """Get proxy solution data for a V0 value."""
    tag = f"proxy_{V0:.4f}"
    d = {
        "Phi": D[f"{tag}_Phi"],
        "lapse": D[f"{tag}_lapse"],
        "rs": float(D[f"{tag}_rs"]),
    }
    nbar_key = f"{tag}_Nbar"
    if nbar_key in D.files:
        d["Nbar"] = D[nbar_key]
        d["kappa_ratio"] = D[f"{tag}_kappa_ratio"]
    else:
        d["Nbar"] = 0.5 * (d["lapse"][:-1] + d["lapse"][1:])
        d["kappa_ratio"] = d["Nbar"]**2
    return d


def get_analytic(V0):
    """Get analytic solution data for a V0 value."""
    tag = f"analytic_{V0:.4f}"
    d = {
        "Phi": D[f"{tag}_Phi"],
        "lapse": D[f"{tag}_lapse"],
        "rs": float(D[f"{tag}_rs"]),
    }
    nbar_key = f"{tag}_Nbar"
    if nbar_key in D.files:
        d["Nbar"] = D[nbar_key]
        d["kappa_ratio"] = D[f"{tag}_kappa_ratio"]
    else:
        d["Nbar"] = 0.5 * (d["lapse"][:-1] + d["lapse"][1:])
        d["kappa_ratio"] = d["Nbar"]**2
    return d


def get_embed(V0):
    """Get embedding data for a V0 value."""
    tag = f"embed_{V0:.4f}"
    return {
        "rho": D[f"{tag}_rho"],
        "R": D[f"{tag}_R"],
    }


def schwarzschild_embedding(rs, r_max, n_points=5000):
    """Schwarzschild R(rho) from horizon outward."""
    r = np.linspace(rs * 1.0001, r_max, n_points)
    v = np.sqrt(r / rs - 1.0)
    rho = rs * (v * np.sqrt(v**2 + 1) + np.arcsinh(v))
    return rho, r


proxy_V0s = list(D["proxy_V0s"])
analytic_V0s = list(D["analytic_V0s"])

print(f"  {len(solution_V0s)} exact solutions, "
      f"{len(proxy_V0s)} proxy, {len(analytic_V0s)} analytic")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 1: Potential + Lapse profiles
# ═══════════════════════════════════════════════════════════════════

print("\n--- Figure 1: Potential + Lapse ---")

V0_pot = [0.005, 0.01, 0.02, 0.05, 0.057]
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(V0_pot)))

for V0, c in zip(V0_pot, colors):
    s = get_sol(V0)
    label = f"$V_0={V0}$"

    axes[0].plot(r_arr, s["Phi"] / cstar_sq, "-", color=c, lw=2,
                 label=label)
    if V0 in proxy_V0s:
        p = get_proxy(V0)
        axes[0].plot(r_arr, p["Phi"] / cstar_sq,
                     "--", color=c, lw=1, alpha=0.5)

    rs = s["rs"]
    if rs > 0.01:
        r_ref = np.linspace(max(1, rs / 2), 200, 500)
        axes[0].plot(r_ref, -rs / (2 * r_ref), ":", color=c, lw=1.2, alpha=0.6)

    axes[1].plot(r_arr, s["lapse"], "-", color=c, lw=1.5, label=label)
    if V0 in proxy_V0s:
        p = get_proxy(V0)
        axes[1].plot(r_arr, p["lapse"],
                     "--", color=c, lw=1, alpha=0.5)

    if rs > 0.01:
        r_an = np.linspace(max(1, rs * 1.01), 200, 500)
        axes[1].plot(r_an, np.sqrt(1.0 - rs / r_an), ":", color=c,
                     lw=1.2, alpha=0.6)

axes[0].plot([], [], "k-", lw=2, label="Full (solid)")
axes[0].plot([], [], "k--", lw=1.2, alpha=0.5, label="Proxy (dashed)")
axes[0].plot([], [], "k:", lw=1.2, alpha=0.6, label="Newtonian $-GM/r$ (dotted)")

axes[0].set_xlabel("$r / a$")
axes[0].set_ylabel(r"$\Phi(r) / c_*^2$")
axes[0].set_xlim(0, 80)
axes[0].set_title("Full (solid) vs proxy (dashed)")
axes[0].legend(fontsize=7, loc="lower right", ncol=2)

axes[1].plot([], [], "k-", lw=2, label="Full (solid)")
axes[1].plot([], [], "k--", lw=1.2, alpha=0.5, label="Proxy (dashed)")
axes[1].plot([], [], "k:", lw=1.2, alpha=0.6, label=r"$\sqrt{1-r_s/r}$ (dotted)")
axes[1].set_xlabel("$r / a$")
axes[1].set_ylabel("Lapse $N(r)$")
axes[1].set_xlim(0, 40)
axes[1].set_title("Lapse profile")
axes[1].legend(fontsize=7, ncol=2)
axes[1].axhline(1, color="gray", lw=0.5, ls=":")

fig.tight_layout()
path = os.path.join(FIGDIR, "twostate_potential.pdf")
fig.savefig(path)
plt.close(fig)
print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 2: Co-metric h^rr vs Schwarzschild
# ═══════════════════════════════════════════════════════════════════

print("\n--- Figure 2: Co-metric comparison ---")

V0_cometric = [0.01, 0.02, 0.03, 0.05, 0.057]
fig, ax = plt.subplots(figsize=(9, 6))
colors_cm = plt.cm.viridis(np.linspace(0.15, 0.85, len(V0_cometric)))

x_ref = np.linspace(1.01, 50, 500)
ax.plot(x_ref, 1.0 - 1.0 / x_ref, "k--", lw=2, label="Schwarzschild $1-r_s/r$")

for V0, c in zip(V0_cometric, colors_cm):
    s = get_sol(V0)
    rs = s["rs"]
    if rs < 0.01:
        continue
    Nbar = s["Nbar"]
    hrr = s["kappa"] / s["kappa_flat"]

    x = r_bond / rs
    mask = Nbar > 0.02
    ax.plot(x[mask], hrr[mask], "-", color=c, lw=1.5,
            label=f"$V_0={V0}$ ($r_s={rs:.2f}a$)")

    if V0 in proxy_V0s:
        p = get_proxy(V0)
        hrr_p = p["kappa_ratio"]
        x_p = r_bond / p["rs"] if p["rs"] > 0.01 else r_bond / rs
        mask_p = p["Nbar"] > 0.02
        ax.plot(x_p[mask_p], hrr_p[mask_p], "-.", color=c, lw=1, alpha=0.5)

    if V0 in analytic_V0s:
        an = get_analytic(V0)
        hrr_a = an["Nbar"]**2
        x_a = r_bond / an["rs"] if an["rs"] > 0.01 else r_bond / rs
        mask_a = an["Nbar"] > 0.02
        ax.plot(x_a[mask_a], hrr_a[mask_a], ":", color=c, lw=1, alpha=0.5)

ax.plot([], [], "k-", lw=1.5, label="Full")
ax.plot([], [], "k-.", lw=1, alpha=0.5, label="Proxy")
ax.plot([], [], "k:", lw=1, alpha=0.5, label="Analytic")
ax.set_xlabel("$r / r_s$")
ax.set_ylabel(r"$h^{rr} / h_0 := \kappa/\kappa^{\mathrm{flat}}$")
ax.set_xlim(1, 50)
ax.set_ylim(0, 1.05)
ax.set_title("Emergent co-metric vs Schwarzschild")
ax.legend(fontsize=8)

fig.tight_layout()
path = os.path.join(FIGDIR, "cometric_comparison.pdf")
fig.savefig(path)
plt.close(fig)
print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 3: PN residual
# ═══════════════════════════════════════════════════════════════════

print("\n--- Figure 3: PN residual ---")

fig, ax = plt.subplots(figsize=(9, 6))

for V0, c in zip(V0_cometric, colors_cm):
    s = get_sol(V0)
    rs = s["rs"]
    if rs < 0.01:
        continue
    Nbar = s["Nbar"]
    hrr = s["kappa"] / s["kappa_flat"]

    x = r_bond / rs
    hrr_schw = 1.0 - rs / r_bond
    residual = hrr - hrr_schw

    mask = (Nbar > 0.02) & (x > 1.5) & (x < 50)
    ax.plot(x[mask], residual[mask], "-", color=c, lw=1.5,
            label=f"$V_0={V0}$")

    if V0 in analytic_V0s:
        an = get_analytic(V0)
        rs_a = an["rs"]
        if rs_a > 0.01:
            hrr_a = an["Nbar"]**2
            x_a = r_bond / rs_a
            hrr_schw_a = 1.0 - rs_a / r_bond
            res_a = hrr_a - hrr_schw_a
            mask_a = (an["Nbar"] > 0.02) & (x_a > 1.5) & (x_a < 50)
            ax.plot(x_a[mask_a], res_a[mask_a], "--", color=c, lw=1, alpha=0.5)

ax.plot([], [], "k-", lw=1.5, label="Full")
ax.plot([], [], "k--", lw=1, alpha=0.5, label="Analytic")
ax.axhline(0, color="k", lw=1.5, ls=":", alpha=0.4,
           label="Exact Schwarzschild (zero residual)")

ax.set_xlabel("$r / r_s$")
ax.set_ylabel(r"$h^{rr} - h^{rr}_{\mathrm{Schw}}$")
ax.set_xlim(1.5, 50)
ax.set_title(r"Post-Newtonian residual $h^{rr} - (1-r_s/r)$")
ax.legend(fontsize=8)

fig.tight_layout()
path = os.path.join(FIGDIR, "pn_comparison.pdf")
fig.savefig(path)
plt.close(fig)
print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 4: Floor-independence study
# ═══════════════════════════════════════════════════════════════════

print("\n--- Figure 4: Floor independence ---")

floors = D["floors"]
V0_floor_test = list(D["V0_floor_test"])
V0_proxy_floor = list(D["V0_proxy_floor"])

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
floor_ref = np.array(floors)

# (a) Proxy equation
colors_pf = plt.cm.viridis(np.linspace(0.15, 0.85, len(V0_proxy_floor)))
for V0, c in zip(V0_proxy_floor, colors_pf):
    tag = f"pfloor_{V0:.4f}"
    mN = D[f"{tag}_min_N"]
    axes[0].semilogx(floors, mN, "o-", color=c, ms=4, lw=1.5,
                     label=f"$V_0={V0}$")

axes[0].semilogx(floor_ref, floor_ref, "k:", lw=0.8, alpha=0.4)
axes[0].set_xlabel("Lapse floor $N_{\\mathrm{floor}}$")
axes[0].set_ylabel("$\\min_n N_n$")
axes[0].set_title("(a) Proxy equation")
axes[0].legend(fontsize=8)

# (b) Exact two-state equation
V0_floor_all = sorted(V0_floor_test)
colors_ef = plt.cm.viridis(np.linspace(0.15, 0.85, len(V0_floor_all)))
for V0, c in zip(V0_floor_all, colors_ef):
    tag = f"floor_{V0:.4f}"
    mN = D[f"{tag}_min_N"]
    axes[1].semilogx(floors, mN, "o-", color=c, ms=4, lw=1.5,
                     label=f"$V_0={V0}$")

axes[1].semilogx(floor_ref, floor_ref, "k:", lw=0.8, alpha=0.4,
                 label="$\\min N = $ floor")
axes[1].set_xlabel("Lapse floor $N_{\\mathrm{floor}}$")
axes[1].set_ylabel("$\\min_n N_n$")
axes[1].set_title("(b) Full two-state equation")
axes[1].legend(fontsize=8)

fig.tight_layout()
path = os.path.join(FIGDIR, "twostate_floor_comparison.pdf")
fig.savefig(path)
plt.close(fig)
print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 5: Interior structure
# ═══════════════════════════════════════════════════════════════════

print("\n--- Figure 5: Interior structure ---")

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

V0_int = [0.02, 0.05, 0.057]
colors_int = plt.cm.viridis(np.linspace(0.2, 0.8, len(V0_int)))

# (a) Lapse profile (core zoom)
for V0, c in zip(V0_int, colors_int):
    s = get_sol(V0)
    axes[0, 0].plot(r_arr[:30], s["lapse"][:30], "-", color=c, lw=2,
                    label=f"$V_0={V0}$")
    if V0 in analytic_V0s:
        an = get_analytic(V0)
        axes[0, 0].plot(r_arr[:30], an["lapse"][:30], ":", color=c, lw=1, alpha=0.5)
axes[0, 0].plot([], [], "k:", lw=1, alpha=0.5, label="Analytic")
axes[0, 0].axhline(0, color="red", lw=0.5, ls="--", alpha=0.4)
axes[0, 0].set_xlabel("$r/a$")
axes[0, 0].set_ylabel("Lapse $N(r)$")
axes[0, 0].set_title("(a) Lapse (sub-critical)")
axes[0, 0].legend()

# (b) Conductance ratio with analytic (1-rs/2r)^2 overlay
for V0, c in zip(V0_int, colors_int):
    s = get_sol(V0)
    ratio = s["kappa"] / s["kappa_flat"]
    axes[0, 1].plot(np.arange(1, N)[:30], ratio[:30], "-", color=c, lw=2,
                    label=f"$V_0={V0}$")
    rs = s["rs"]
    if rs > 0.01:
        an_kappa = np.maximum(1.0 - rs / r_bond, 0.0)
        axes[0, 1].plot(np.arange(1, N)[:30], an_kappa[:30], "--", color=c,
                        lw=1, alpha=0.5)
    if V0 in analytic_V0s:
        an = get_analytic(V0)
        axes[0, 1].plot(np.arange(1, N)[:30], an["kappa_ratio"][:30], ":", color=c,
                        lw=1, alpha=0.5)
axes[0, 1].plot([], [], "k--", lw=1, alpha=0.5, label=r"$1-r_s/r$")
axes[0, 1].plot([], [], "k:", lw=1, alpha=0.5, label="Analytic")
axes[0, 1].axhline(1.0, color="gray", lw=0.5, ls=":")
axes[0, 1].set_xlabel("Shell $n$")
axes[0, 1].set_ylabel(r"$\kappa_n / \kappa_n^{\mathrm{flat}}$")
axes[0, 1].set_title("(b) Conductance suppression")
axes[0, 1].legend(fontsize=8)

# (c) Super-critical: lapse hits floor
V0_super = [0.07, 0.1]
colors_sup2 = plt.cm.Reds(np.linspace(0.4, 0.8, len(V0_super)))
for V0, c in zip(V0_super, colors_sup2):
    s = get_sol(V0)
    axes[1, 0].plot(r_arr[:30], s["lapse"][:30], "-", color=c, lw=2,
                    label=f"$V_0={V0}$")
axes[1, 0].axhline(0, color="red", lw=0.5, ls="--", alpha=0.4)
axes[1, 0].set_xlabel("$r/a$")
axes[1, 0].set_ylabel("Lapse $N(r)$")
axes[1, 0].set_title("(c) Super-critical: lapse collapse")
axes[1, 0].legend(fontsize=8)

# (d) Proxy (dashed) vs exact (solid) comparison
for V0, c in zip(V0_int, colors_int):
    s = get_sol(V0)
    p = get_proxy(V0)
    axes[1, 1].plot(r_arr[:30], s["lapse"][:30], "-", color=c, lw=2,
                    label=f"full $V_0={V0}$")
    axes[1, 1].plot(r_arr[:30], p["lapse"][:30], "--", color=c, lw=1.2,
                    alpha=0.6)
axes[1, 1].axhline(0, color="red", lw=0.5, ls="--", alpha=0.4)
axes[1, 1].set_xlabel("$r/a$")
axes[1, 1].set_ylabel("Lapse $N(r)$")
axes[1, 1].set_title("(d) Proxy (dashed) vs full (solid)")
axes[1, 1].legend(fontsize=8)

fig.tight_layout()
path = os.path.join(FIGDIR, "twostate_interior.pdf")
fig.savefig(path)
plt.close(fig)
print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 6: Embedding diagrams
# ═══════════════════════════════════════════════════════════════════

print("\n--- Figure 6: Embedding diagram ---")

fig, ax = plt.subplots(1, 1, figsize=(6.5, 6))

# Schwarzschild reference (rs = a = 1)
rs_ref = a
rho_schw, R_schw = schwarzschild_embedding(rs_ref, 30.0)
rho_lim = 12

# Key V0 values: sub-critical, near-critical, super-critical (floor)
embed_V0s = [0.01, 0.057, 0.07]
embed_colors = ['#1976D2', '#D32F2F', '#7B1FA2']
embed_lws = [2.0, 2.0, 1.5]

for V0, color, lw in zip(embed_V0s, embed_colors, embed_lws):
    ed = get_embed(V0)
    s = get_sol(V0)
    mask = ed['rho'] <= rho_lim
    Nmin = s["Nbar"].min()
    floor_tag = ', floor' if V0 > 0.06 else ''
    ax.plot(ed['rho'][mask], ed['R'][mask], color=color, lw=lw,
            label=f'$V_0={V0}$ ($\\bar{{N}}_{{\\min}}={Nmin:.2f}${floor_tag})')

# Schwarzschild reference
mask_s = rho_schw <= rho_lim
ax.plot(rho_schw[mask_s], R_schw[mask_s], 'k--', lw=2.0,
        label='Schwarzschild exterior ($dR/d\\rho=0$)')

# Classical GR singularity reference
rho_cl = np.linspace(0, 6, 50)
R_cl = 0.5 * rho_cl
ax.plot(rho_cl, R_cl, color='gray', ls='--', lw=1.0, alpha=0.6,
        label=r"Classical GR ($R\to 0$)")

# Minimal sphere marker
ax.plot([0], [a], 'ko', ms=4, zorder=5)

# Cone slope guide line for near-critical V0=0.057
Nbar_057 = get_sol(0.057)["Nbar"]
N057 = Nbar_057.min()
rg = np.linspace(0, 7, 50)
ax.plot(rg, a + N057 * rg, ':', color=embed_colors[1], lw=0.8, alpha=0.5)

# Annotations
ax.annotate(f'$dR/d\\rho = \\bar{{N}}_{{\\min}} = {N057:.2f}$',
            xy=(4.5, a + N057 * 4.5), xytext=(6.5, 2.2),
            fontsize=8, color=embed_colors[1],
            arrowprops=dict(arrowstyle='->', color=embed_colors[1], lw=0.8))

ax.annotate('$dR/d\\rho = 0$\n(horizon)',
            xy=(0.2, rs_ref), xytext=(3.5, 1.2),
            fontsize=8, color='black',
            arrowprops=dict(arrowstyle='->', color='black', lw=0.8))

ax.annotate('Classical singularity', xy=(0.1, 0.05), xytext=(2.5, 0.4),
            fontsize=8, color='gray',
            arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

# R_min line
ax.axhline(a, color='gray', lw=0.5, ls=':', alpha=0.4)
ax.text(rho_lim - 0.2, a + 0.15, '$R_{\\min}=a$', fontsize=7,
        color='gray', ha='right')

# Sub-critical / near-critical labels along curves
ax.text(4.5, 5.3, 'sub-critical', fontsize=8, color=embed_colors[0],
        ha='center', va='bottom', rotation=44,
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1))
ax.text(9, 8.3, 'near-critical', fontsize=8, color=embed_colors[1],
        ha='center', va='bottom', rotation=42,
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1))

ax.set_xlabel(r'Proper distance $\rho\,/\,a$')
ax.set_ylabel(r'Area radius $R\,/\,a$')
ax.set_xlim(0, rho_lim)
ax.set_ylim(0, rho_lim)
ax.set_aspect('equal')
ax.legend(loc='upper left', fontsize=7.5, framealpha=0.9)
ax.set_title('Embedding: cone angle vs horizon', fontsize=11)

plt.tight_layout()
path = os.path.join(FIGDIR, "twostate_embedding.pdf")
fig.savefig(path)
plt.close(fig)
print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 7: Thermodynamics (kappa_H, T_H)
# ═══════════════════════════════════════════════════════════════════

print("\n--- Figure 7: Thermodynamics ---")


def newtonian_kappa_H(rs, cstar_sq):
    """Weak-field surface gravity: kappa_H = c*^2 / (2 r_s).

    For sub-lattice r_s (r_s < a), the lapse gradient at r=r_s cannot
    be resolved on the lattice.  The Newtonian formula is the correct
    leading-order result; corrections are O(r_s/a).
    """
    if rs <= 0:
        return np.nan
    return cstar_sq / (2.0 * rs)


# Compute kappa_H = c*^2/(2*rs) for each closure, only for converged solutions
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# Use ALL converged sub-critical solutions (not just the pre-computed thermo subset)
all_sol_V0s = sorted(D["solution_V0s"])
V0_supercrit = list(D["V0_supercrit"])
conv_V0s = []
kH_exact_list = []
kH_analytic_list = []
kH_proxy_list = []

for V0 in all_sol_V0s:
    if V0 in V0_supercrit:
        continue  # skip super-critical
    tag = f"sol_{V0:.4f}"
    if not bool(D[f"{tag}_converged"]):
        continue  # skip non-converged solutions
    rs_val = float(D[f"{tag}_rs"])
    if rs_val <= 0:
        continue  # need a valid rs for kappa_H
    conv_V0s.append(V0)
    rs_ex = float(D[f"{tag}_rs"])
    kH_exact_list.append(newtonian_kappa_H(rs_ex, cstar_sq))

    # Proxy kappa_H (uses proxy's own rs)
    ptag = f"proxy_{V0:.4f}"
    if ptag + "_rs" in D.files:
        rs_p = float(D[f"{ptag}_rs"])
        kH_proxy_list.append(newtonian_kappa_H(rs_p, cstar_sq))
    else:
        kH_proxy_list.append(np.nan)

    # Analytic kappa_H (uses analytic's own rs)
    atag = f"analytic_{V0:.4f}"
    if atag + "_rs" in D.files:
        rs_a = float(D[f"{atag}_rs"])
        kH_analytic_list.append(newtonian_kappa_H(rs_a, cstar_sq))
    else:
        kH_analytic_list.append(np.nan)

conv_V0s = np.array(conv_V0s)
kH_exact = np.array(kH_exact_list)
kH_analytic_arr = np.array(kH_analytic_list)
kH_proxy_arr = np.array(kH_proxy_list)
cstar = np.sqrt(cstar_sq)
TH_exact = kH_exact / (2 * np.pi * cstar)
TH_analytic_arr = kH_analytic_arr / (2 * np.pi * cstar)
TH_proxy_arr = kH_proxy_arr / (2 * np.pi * cstar)

print(f"  Using {len(conv_V0s)} converged V0 values: {list(conv_V0s)}")

axes[0].semilogy(conv_V0s, kH_exact, "o-", color="C0", ms=5, lw=1.5,
                 label=r"Full $\kappa_H = c_*^2/(2r_s)$")
axes[0].semilogy(conv_V0s, kH_analytic_arr, "s--", color="C1", ms=4, lw=1.2,
                 label="Analytic two-state")
axes[0].semilogy(conv_V0s, kH_proxy_arr, "^:", color="C2", ms=4, lw=1.2,
                 label="Proxy")
axes[0].set_xlabel("$V_0 / t_0$")
axes[0].set_ylabel(r"Surface gravity $\kappa_H = c_*^2/(2r_s)$")
axes[0].set_title("(a) Surface gravity")
axes[0].legend(fontsize=8)

axes[1].semilogy(conv_V0s, TH_exact, "o-", color="C0", ms=5, lw=1.5,
                 label="Full")
axes[1].semilogy(conv_V0s, TH_analytic_arr, "s--", color="C1", ms=4, lw=1.2,
                 label="Analytic two-state")
axes[1].semilogy(conv_V0s, TH_proxy_arr, "^:", color="C2", ms=4, lw=1.2,
                 label="Proxy")
axes[1].set_xlabel("$V_0 / t_0$")
axes[1].set_ylabel(r"$T_H = \hbar\kappa_H / (2\pi c_*)$")
axes[1].set_title(r"(b) Hawking temperature $T_H = \hbar\kappa_H/(2\pi c_*)$")
axes[1].legend(fontsize=8)

fig.tight_layout()
path = os.path.join(FIGDIR, "twostate_thermodynamics.pdf")
fig.savefig(path)
plt.close(fig)
print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 8: Standalone conductance profile
# ═══════════════════════════════════════════════════════════════════

print("\n--- Figure 8: Conductance profile ---")

V0_cond = [0.01, 0.02, 0.03, 0.05, 0.057]
fig, ax = plt.subplots(figsize=(9, 6))
colors_cond = plt.cm.viridis(np.linspace(0.15, 0.85, len(V0_cond)))

for V0, c in zip(V0_cond, colors_cond):
    s = get_sol(V0)
    ratio = s["kappa"] / s["kappa_flat"]
    ax.plot(r_bond, ratio, "-", color=c, lw=1.5,
            label=f"$V_0={V0}$")
    rs = s["rs"]
    if rs > 0.01:
        an_kappa = np.maximum(1.0 - rs / r_bond, 0.0)
        ax.plot(r_bond, an_kappa, "--", color=c, lw=1, alpha=0.5)

    if V0 in proxy_V0s:
        p = get_proxy(V0)
        ax.plot(r_bond, p["kappa_ratio"], "-.", color=c, lw=0.8, alpha=0.4)
    if V0 in analytic_V0s:
        an = get_analytic(V0)
        ax.plot(r_bond, an["kappa_ratio"], ":", color=c, lw=0.8, alpha=0.4)

ax.plot([], [], "k-", lw=1.5, label="Full")
ax.plot([], [], "k--", lw=1, alpha=0.5, label=r"$1-r_s/r$")
ax.plot([], [], "k-.", lw=0.8, alpha=0.4, label="Proxy $\\bar{N}^2$")
ax.plot([], [], "k:", lw=0.8, alpha=0.4, label="Analytic $\\bar{N}^2$")
ax.axhline(1, color="gray", lw=0.5, ls=":")
ax.axhline(0, color="red", lw=0.5, ls="--", alpha=0.3)
ax.set_xlabel(r"$r / a$")
ax.set_ylabel(r"$\kappa_n[\Phi]/\kappa_n^{\mathrm{flat}}$")
ax.set_title("Self-consistent conductance suppression")
ax.set_xlim(0, 100)
ax.set_ylim(-0.05, 1.1)
ax.legend(fontsize=8)

fig.tight_layout()
path = os.path.join(FIGDIR, "conductance_profile.pdf")
fig.savefig(path)
plt.close(fig)
print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 9: Standalone lapse profile
# ═══════════════════════════════════════════════════════════════════

print("\n--- Figure 9: Lapse profile ---")

V0_lapse = [0.01, 0.02, 0.03, 0.05, 0.057]
fig, ax = plt.subplots(figsize=(9, 6))
colors_lapse = plt.cm.viridis(np.linspace(0.15, 0.85, len(V0_lapse)))

for V0, c in zip(V0_lapse, colors_lapse):
    s = get_sol(V0)
    ax.plot(r_arr, s["lapse"], "-", color=c, lw=1.5,
            label=f"$V_0={V0}$")
    rs = s["rs"]
    if rs > 0.01:
        f_an = np.maximum(1.0 - rs / r_arr, 0.0)
        ax.plot(r_arr, np.sqrt(f_an), "--", color=c,
                lw=1, alpha=0.5)

    if V0 in proxy_V0s:
        p = get_proxy(V0)
        ax.plot(r_arr, p["lapse"], "-.", color=c, lw=0.8, alpha=0.4)
    if V0 in analytic_V0s:
        an = get_analytic(V0)
        ax.plot(r_arr, an["lapse"], ":", color=c, lw=0.8, alpha=0.4)

ax.plot([], [], "k-", lw=1.5, label="Full")
ax.plot([], [], "k--", lw=1, alpha=0.5, label=r"$\sqrt{1-r_s/r}$")
ax.plot([], [], "k-.", lw=0.8, alpha=0.4, label="Proxy")
ax.plot([], [], "k:", lw=0.8, alpha=0.4, label="Analytic")
ax.axhline(1, color="gray", lw=0.5, ls=":")
ax.axhline(0, color="red", lw=0.8, ls="--", alpha=0.5, label="Horizon $N=0$")
ax.set_xlabel(r"$r / a$")
ax.set_ylabel(r"$N(r)=1+\Phi/c_*^2$")
ax.set_title("Lapse function profile")
ax.set_xlim(0, 100)
ax.set_ylim(-0.05, 1.08)
ax.legend(fontsize=8)

fig.tight_layout()
path = os.path.join(FIGDIR, "lapse_profile.pdf")
fig.savefig(path)
plt.close(fig)
print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 10: T^QI profile (self-consistent vs prescribed)
# ═══════════════════════════════════════════════════════════════════

print("\n--- Figure 10: T^QI profile ---")

V0_tqi = list(D["tqi_V0s"])
fig, ax = plt.subplots(figsize=(9, 6))
colors_tqi = plt.cm.viridis(np.linspace(0.15, 0.85, len(V0_tqi)))

for V0, c in zip(V0_tqi, colors_tqi):
    # Recompute T^QI with correct q = beta0*t0*Nbar/4 (paper's MI tail parameter)
    s = get_sol(V0)
    rs = s["rs"]
    if rs < 0.01:
        continue
    Nbar = s["Nbar"]
    Nbar_ext = np.concatenate([Nbar, [1.0]])
    la = beta0 * np.maximum(np.abs(Nbar_ext), 1e-10) * t0 / 4.0
    lt = np.log(np.maximum(la, 1e-30))
    tqi_sc = np.abs(Nbar_ext)**6 / np.maximum(lt**2, 1e-30)

    f_s = np.maximum(1.0 - rs / r_arr, 1e-10)
    la_s = beta0 * t0 * np.sqrt(np.maximum(f_s, 1e-10)) / 4.0
    lt_s = np.log(np.maximum(np.abs(la_s), 1e-30))
    tqi_pr = f_s**3 / np.maximum(lt_s**2, 1e-30)

    ns = tqi_sc[-1] if abs(tqi_sc[-1]) > 1e-30 else 1.0
    np_ = tqi_pr[-1] if abs(tqi_pr[-1]) > 1e-30 else 1.0

    ax.plot(r_arr, tqi_sc / ns, "-", color=c, lw=1.5,
            label=f"$V_0={V0}$")
    ax.plot(r_arr, tqi_pr / np_, "--", color=c, alpha=0.5, lw=1.0)

ax.plot([], [], "k--", lw=1.5, alpha=0.5,
        label=r"Prescribed Schwarzschild $T^{\mathrm{QI}}_{\mathrm{pr}}$")
ax.set_xlabel(r"$r / a$")
ax.set_ylabel(r"$T^{\mathrm{QI}}/T^{\mathrm{QI}}_\infty$")
ax.set_title(r"Self-consistent vs. prescribed $T^{\mathrm{QI}}$")
ax.set_xlim(0, 100)
ax.set_ylim(-0.05, 1.15)
ax.legend(fontsize=8)

fig.tight_layout()
path = os.path.join(FIGDIR, "tqi_profile.pdf")
fig.savefig(path)
plt.close(fig)
print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 11: Summary comparison (min N, rs, cross-residual vs V0)
# ═══════════════════════════════════════════════════════════════════

print("\n--- Figure 11: Summary comparison ---")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Exact sweep data — filter to converged solutions only
V0_sw_all = D["sweep_V0"]
minN_sw_all = D["sweep_min_N"]
rs_sw_all = D["sweep_rs"]

# Trust stored convergence flags from the solver; do not apply hard-coded V0 cutoffs.
conv_mask = np.ones(len(V0_sw_all), dtype=bool)
for i, V0 in enumerate(V0_sw_all):
    tag = f"sol_{V0:.4f}"
    if (tag + "_converged") in D.files:
        conv_mask[i] = bool(D[tag + "_converged"])

V0_sw = V0_sw_all[conv_mask]
minN_sw = minN_sw_all[conv_mask]
rs_sw = rs_sw_all[conv_mask]

# Proxy and analytic sweep data (continuous curves from V0 sweep)
sweep_proxy_rs = D["sweep_proxy_rs"] if "sweep_proxy_rs" in D.files else None
sweep_analytic_rs = D["sweep_analytic_rs"] if "sweep_analytic_rs" in D.files else None
sweep_proxy_minN = D["sweep_proxy_minN"] if "sweep_proxy_minN" in D.files else None
sweep_analytic_minN = D["sweep_analytic_minN"] if "sweep_analytic_minN" in D.files else None

# Fallback: discrete markers from per-V0 proxy/analytic solutions
V0_comp = sorted(set(V0_core + V0_extended))
minN_proxy = [get_proxy(V0)["lapse"].min() for V0 in V0_comp]
rs_proxy_list = [get_proxy(V0)["rs"] for V0 in V0_comp]
minN_analytic = [get_analytic(V0)["lapse"].min() for V0 in V0_comp]
rs_analytic_list = [get_analytic(V0)["rs"] for V0 in V0_comp]

# Cross-residuals — filter to converged V0 only
cross_V0_all = D["cross_V0"]
cross_proxy_all = D["cross_proxy"]
cross_analytic_all = D["cross_analytic"]
cross_ok = np.array([
    bool(D[f"sol_{V0:.4f}_converged"]) if f"sol_{V0:.4f}_converged" in D.files
    else True
    for V0 in cross_V0_all
])
cross_V0 = cross_V0_all[cross_ok]
cross_proxy = cross_proxy_all[cross_ok]
cross_analytic = cross_analytic_all[cross_ok]

# (a) min(N) vs V0
axes[0].plot(V0_sw, minN_sw, "o-", ms=3, color="C0", lw=1.5, label="Full")
if sweep_analytic_minN is not None:
    axes[0].plot(V0_sw_all, sweep_analytic_minN, "--", color="C1", lw=1.2,
                 alpha=0.7, label="Analytic two-state")
else:
    axes[0].plot(V0_comp, minN_analytic, "s--", ms=4, color="C1", lw=1.2,
                 alpha=0.7, label="Analytic two-state")
if sweep_proxy_minN is not None:
    axes[0].plot(V0_sw_all, sweep_proxy_minN, ":", color="C2", lw=1.2,
                 alpha=0.7, label="Proxy")
else:
    axes[0].plot(V0_comp, minN_proxy, "^:", ms=4, color="C2", lw=1.2,
                 alpha=0.7, label="Proxy")
axes[0].set_xlabel("$V_0 / t_0$")
axes[0].set_ylabel("$\\min_n N_n$")
axes[0].set_title("(a) Minimum lapse")
axes[0].legend(fontsize=8)
axes[0].axhline(0, color="red", lw=0.5, ls="--", alpha=0.4)

# (b) rs vs V0
axes[1].plot(V0_sw, rs_sw, "o-", ms=3, color="C0", lw=1.5, label="Full")
if sweep_analytic_rs is not None:
    axes[1].plot(V0_sw_all, sweep_analytic_rs, "--", color="C1", lw=1.2,
                 alpha=0.7, label="Analytic two-state")
else:
    axes[1].plot(V0_comp, rs_analytic_list, "s--", ms=4, color="C1", lw=1.2,
                 alpha=0.7, label="Analytic two-state")
if sweep_proxy_rs is not None:
    axes[1].plot(V0_sw_all, sweep_proxy_rs, ":", color="C2", lw=1.2,
                 alpha=0.7, label="Proxy")
else:
    axes[1].plot(V0_comp, rs_proxy_list, "^:", ms=4, color="C2", lw=1.2,
                 alpha=0.7, label="Proxy")
# Linear prediction: fit slope from weak-field converged data
if len(V0_sw) >= 2:
    n_fit = min(4, len(V0_sw))
    slope = np.polyfit(V0_sw[:n_fit], rs_sw[:n_fit], 1)[0]
    V0_lin = np.linspace(0, max(V0_comp[-1], V0_sw[-1]), 100)
    rs_lin = slope * V0_lin
    axes[1].plot(V0_lin, rs_lin, "k:", lw=1, alpha=0.4, label="Linear prediction")
axes[1].set_xlabel("$V_0 / t_0$")
axes[1].set_ylabel("$r_s / a$")
axes[1].set_title("(b) Schwarzschild radius")
axes[1].legend(fontsize=8)

# (c) Cross-residual
axes[2].semilogy(cross_V0, cross_analytic, "s--", ms=4, color="C1", lw=1.2,
                 label=r"$|F_{\mathrm{full}}(\Phi_{\mathrm{analytic}})|$")
axes[2].semilogy(cross_V0, cross_proxy, "^:", ms=4, color="C2", lw=1.2,
                 label=r"$|F_{\mathrm{full}}(\Phi_{\mathrm{proxy}})|$")
axes[2].set_xlabel("$V_0 / t_0$")
axes[2].set_ylabel(r"$|F|_\infty$")
axes[2].set_title("(c) Cross-residual in full equation")
axes[2].legend(fontsize=8)

fig.tight_layout()
path = os.path.join(FIGDIR, "twostate_comparison.pdf")
fig.savefig(path)
plt.close(fig)
print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 12: Self-consistency verification
# ═══════════════════════════════════════════════════════════════════

print("\n--- Figure 12: Self-consistency verification ---")

V0_verify = float(D["verify_V0"])
LHS_v = D["verify_LHS"]
RHS_v = D["verify_RHS"]
s_v = get_sol(V0_verify)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

n_sites = np.arange(1, N + 1)
n_plot = 40

# (a) LHS and RHS
axes[0].plot(n_sites[:n_plot], LHS_v[:n_plot], "-", color="C0", lw=2,
             label="LHS: $L_{\\kappa[\\Phi]}\\,\\Phi$")
axes[0].plot(n_sites[:n_plot], RHS_v[:n_plot], "--", color="C3", lw=2,
             label=r"RHS: $(\beta_0/c_*^2)\,\Delta\rho[\Phi]$")
axes[0].set_xlabel("Shell index $n$")
axes[0].set_ylabel("Equation value")
axes[0].set_title(f"(a) Full closure equation, $V_0 = {V0_verify}$")
axes[0].legend(fontsize=9)

# (b) Conductance ratio: exact vs proxy vs analytic
kappa_v = s_v["kappa"]
kappa_flat_v = s_v["kappa_flat"]
ratio_exact = kappa_v / kappa_flat_v

axes[1].plot(n_sites[:-1][:n_plot], ratio_exact[:n_plot], "-", color="C0",
             lw=2, label="Full MI-based")

if V0_verify in proxy_V0s:
    p_v = get_proxy(V0_verify)
    lapse_p = p_v["lapse"]
    Nbar_p = 0.5 * (lapse_p[:-1] + lapse_p[1:])
    ratio_proxy = Nbar_p**2
    axes[1].plot(n_sites[:-1][:n_plot], ratio_proxy[:n_plot], "--", color="C1",
                 lw=1.5, alpha=0.7, label="Proxy $\\bar{N}^2$")

if V0_verify in analytic_V0s:
    a_v = get_analytic(V0_verify)
    lapse_a = a_v["lapse"]
    Nbar_a = 0.5 * (lapse_a[:-1] + lapse_a[1:])
    ratio_an = Nbar_a**2
    axes[1].plot(n_sites[:-1][:n_plot], ratio_an[:n_plot], ":", color="C2",
                 lw=1.5, alpha=0.7, label="Analytic $\\bar{N}^2$")

rs_v = s_v["rs"]
an_wf = np.maximum(1.0 - rs_v / r_bond, 0.0)
axes[1].plot(n_sites[:-1][:n_plot], an_wf[:n_plot], "k:", lw=1, alpha=0.4,
             label=r"$1-r_s/r$")

axes[1].axhline(1.0, color="gray", lw=0.5, ls=":")
axes[1].set_xlabel("Shell index $n$")
axes[1].set_ylabel(r"$\kappa_n[\Phi] / \kappa_n^{\,\mathrm{flat}}$")
axes[1].set_title(f"(b) Conductance ratio, $V_0 = {V0_verify}$")
axes[1].legend(fontsize=8)
axes[1].set_ylim(0.9 * ratio_exact[:n_plot].min(), 1.02)

fig.tight_layout()
path = os.path.join(FIGDIR, "twostate_verification.pdf")
fig.savefig(path)
plt.close(fig)
print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 13: Temperature sweep
# ═══════════════════════════════════════════════════════════════════

print("\n--- Figure 13: Temperature sweep ---")

if "temp_V0s" in D.files:
    temp_V0s = list(D["temp_V0s"])
    bt0_arr_all = D["temp_bt0"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    V0_main = temp_V0s[0]  # 0.001
    tag_main = f"temp_{V0_main:.4f}"
    rs_main = D[f"{tag_main}_rs"]
    F_main = D[f"{tag_main}_F"]
    near_main = F_main < 1e-6

    # (a) Schwarzschild radius vs bt0 — both V0 values + proxy overlay
    ax = axes[0, 0]
    for iv, V0_temp in enumerate(temp_V0s):
        tag_t = f"temp_{V0_temp:.4f}"
        rs_t = D[f"{tag_t}_rs"]
        F_t = D[f"{tag_t}_F"]
        near_t = F_t < 1e-6
        c_ex = f"C{2*iv}"
        ax.plot(bt0_arr_all[near_t], rs_t[near_t], "o-", color=c_ex, ms=4,
                label=f"Full $V_0={V0_temp}$")
        if np.any(~near_t):
            ax.plot(bt0_arr_all[~near_t], rs_t[~near_t], "x", color="red", ms=6, mew=2)
        # Proxy rs overlay
        prs_key = f"{tag_t}_proxy_rs"
        if prs_key in D.files:
            prs = D[prs_key]
            ax.plot(bt0_arr_all, prs, "--", color=c_ex, lw=1, alpha=0.5,
                    label=f"Proxy $V_0={V0_temp}$")
    ax.set_xlabel(r"$\beta_0 t_0$")
    ax.set_ylabel(r"$r_s / a$")
    ax.set_title("(a) Schwarzschild radius vs temperature")
    ax.legend(fontsize=8)

    # (b) Per-mode entanglement entropy alpha_S vs bt0
    ax = axes[0, 1]
    print("    Computing alpha_S vs bt0 ...")
    aS_arr = np.array([compute_alpha_S(N, t0, bt / t0) for bt in bt0_arr_all])
    ax.plot(bt0_arr_all, aS_arr, "o-", color="C0", ms=4)
    ax.axhline(0.25, ls=":", color="gray", lw=1, label=r"$\alpha_S = 1/4$")
    # Mark the crossover bt0 where alpha_S = 1/4
    crossings = np.where(np.diff(np.sign(aS_arr - 0.25)))[0]
    if len(crossings) > 0:
        idx_c = crossings[0]
        bt0_cross = bt0_arr_all[idx_c] + (0.25 - aS_arr[idx_c]) / (aS_arr[idx_c+1] - aS_arr[idx_c]) * (bt0_arr_all[idx_c+1] - bt0_arr_all[idx_c])
        ax.axvline(bt0_cross, ls="--", color="C3", lw=0.8, alpha=0.7)
        ax.text(bt0_cross + 0.05, 0.05, rf"$\beta_0 t_0 \approx {bt0_cross:.2f}$",
                fontsize=8, color="C3")
    ax.set_xlabel(r"$\beta_0 t_0$")
    ax.set_ylabel(r"$\alpha_S$ (nats per mode)")
    ax.set_title(r"(b) Per-mode entanglement entropy $\alpha_S$")
    ax.legend(fontsize=8)
    ax.set_ylim(0, None)

    # (c) Phi profiles at selected temperatures (V0_main) + proxy overlay
    ax = axes[1, 0]
    Phi_2d = D[f"{tag_main}_Phi"]
    bt0_show = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5]
    colors_T = plt.cm.coolwarm(np.linspace(0.0, 1.0, len(bt0_show)))
    for i, bt0_target in enumerate(bt0_show):
        idx = np.argmin(np.abs(bt0_arr_all - bt0_target))
        if F_main[idx] < 1e-6:
            ax.plot(r_arr, Phi_2d[idx] / cstar_sq, "-", color=colors_T[i],
                    lw=1.5, label=rf"$\beta_0 t_0 = {bt0_arr_all[idx]:.1f}$")
            # Proxy overlay (dashed)
            bt0_val = bt0_arr_all[idx]
            Phi_proxy = compute_proxy_Phi(N, t0, V0_main, n_core, bt0_val / t0, cstar_sq)
            ax.plot(r_arr, Phi_proxy / cstar_sq, "--", color=colors_T[i],
                    lw=0.8, alpha=0.6)
    ax.plot([], [], "k-", lw=2, label="Full (solid)")
    ax.plot([], [], "k--", lw=1.2, alpha=0.6, label="Proxy (dashed)")
    ax.set_xlabel("$r / a$")
    ax.set_ylabel(r"$\Phi(r) / c_*^2$")
    ax.set_title(f"(c) Potential profiles, $V_0 = {V0_main}$")
    ax.set_xlim(0, 80)
    ax.legend(fontsize=7, ncol=2)

    # (d) -Phi*r test: plateau = 1/r (Newtonian) + proxy overlay
    ax = axes[1, 1]
    for i, bt0_target in enumerate(bt0_show):
        idx = np.argmin(np.abs(bt0_arr_all - bt0_target))
        if F_main[idx] < 1e-6:
            R_outer = float(r_arr[-1])
            GM_est = -Phi_2d[idx] * r_arr * R_outer / np.maximum(R_outer - r_arr, 1e-12)
            ax.plot(r_arr, GM_est, "-", color=colors_T[i], lw=1.5,
                    label=rf"$\beta_0 t_0 = {bt0_arr_all[idx]:.1f}$")
            # Proxy overlay (dashed)
            bt0_val = bt0_arr_all[idx]
            Phi_proxy = compute_proxy_Phi(N, t0, V0_main, n_core, bt0_val / t0, cstar_sq)
            GM_est_p = -Phi_proxy * r_arr * R_outer / np.maximum(R_outer - r_arr, 1e-12)
            ax.plot(r_arr, GM_est_p, "--", color=colors_T[i], lw=0.8, alpha=0.6)
    ax.plot([], [], "k-", lw=2, label="Full (solid)")
    ax.plot([], [], "k--", lw=1.2, alpha=0.6, label="Proxy (dashed)")
    ax.set_xlabel("$r / a$")
    ax.set_ylabel(r"$GM_{\mathrm{est}}(r)$")
    ax.set_title(r"(d) BC-corrected $1/r$ test (no Yukawa)")
    ax.set_xlim(0, 180)
    ax.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    path = os.path.join(FIGDIR, "twostate_temperature_sweep.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")
else:
    print("  SKIPPED (no temp sweep data in npz)")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 14: Combined co-metric / conductance / PN residual
# ═══════════════════════════════════════════════════════════════════

print("\n--- Figure 14: Co-metric + PN residual (2-panel) ---")

V0_fig14 = [0.01, 0.02, 0.03, 0.05, 0.057]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
cm14 = plt.cm.viridis(np.linspace(0.1, 0.9, len(V0_fig14)))

# (a) Co-metric h^rr vs r/rs
ax = axes[0]
x_ref = np.linspace(1.5, 120, 500)
ax.plot(x_ref, 1.0 - 1.0 / x_ref, "k--", lw=1.5, alpha=0.6,
        label=r"$1-r_s/r$")
for V0, c in zip(V0_fig14, cm14):
    s = get_sol(V0)
    rs = s["rs"]
    if rs < 1e-6:
        continue
    Nbar = s["Nbar"]
    hrr = s["kappa"] / s["kappa_flat"]
    x = r_bond / rs
    m = (x > 1.0) & (x < 120) & (Nbar > 0.02)
    if np.any(m):
        ax.plot(x[m], hrr[m], "-", color=c, lw=1.2, label=f"$V_0={V0}$")
    if V0 in analytic_V0s:
        an = get_analytic(V0)
        rs_a = an["rs"]
        if rs_a > 1e-6:
            hrr_a = an["Nbar"]**2
            x_a = r_bond / rs_a
            m_a = (x_a > 1.0) & (x_a < 120) & (an["Nbar"] > 0.02)
            if np.any(m_a):
                ax.plot(x_a[m_a], hrr_a[m_a], ":", color=c, lw=1.2, alpha=0.7)
ax.plot([], [], "k-", lw=2, label="Full (solid)")
ax.plot([], [], "k:", lw=1.2, alpha=0.7, label="Analytic (dotted)")
ax.set_xlabel(r"$r / r_s$")
ax.set_ylabel(r"$h^{rr}/h_0 := \kappa/\kappa^{\mathrm{flat}}$")
ax.set_title("(a) Emergent co-metric", fontsize=10)
ax.set_xlim(0.8, 50)
ax.set_ylim(-0.05, 1.1)
ax.legend(fontsize=6, loc="lower right")

# (b) PN residual
ax = axes[1]
has_data = False
for V0, c in zip(V0_fig14, cm14):
    s = get_sol(V0)
    rs = s["rs"]
    if rs < 0.05:
        continue
    has_data = True
    Nbar = s["Nbar"]
    x = r_bond / rs
    residual = s["kappa"] / s["kappa_flat"] - (1.0 - 1.0 / x)
    m = (x > 2.0) & (x < 100) & (Nbar > 0.02)
    if np.any(m):
        ax.plot(x[m], residual[m], "-", color=c, lw=1.2, label=f"$V_0={V0}$")
    if V0 in analytic_V0s:
        an = get_analytic(V0)
        rs_a = an["rs"]
        if rs_a > 0.05:
            x_a = r_bond / rs_a
            res_a = an["Nbar"]**2 - (1.0 - 1.0 / x_a)
            m_a = (x_a > 2.0) & (x_a < 100) & (an["Nbar"] > 0.02)
            if np.any(m_a):
                ax.plot(x_a[m_a], res_a[m_a], ":", color=c, lw=1.2, alpha=0.7)
if has_data:
    ax.axhline(0, color="k", lw=1, alpha=0.4, ls="--",
               label="Exact Schwarzschild")
ax.plot([], [], "k-", lw=1.2, label="Full")
ax.plot([], [], "k:", lw=1.2, alpha=0.7, label="Analytic")
ax.set_xlabel(r"$r / r_s$")
ax.set_ylabel(r"$h^{rr} - h^{rr}_{\mathrm{Schw}}$")
ax.set_title("(b) Post-Newtonian residual", fontsize=10)
ax.set_xlim(1, 60)
ax.legend(fontsize=6)
ax.axhline(0, color="gray", lw=0.5, ls=":")

fig.tight_layout()
path = os.path.join(FIGDIR, "cometric_conductance_pn.pdf")
fig.savefig(path)
plt.close(fig)
print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 15: Exact vs analytic comparison (with proxy overlay)
# ═══════════════════════════════════════════════════════════════════

print("\n--- Figure 15: Exact vs analytic comparison ---")

V0_fig15 = [0.01, 0.02, 0.03, 0.05]
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
colors_15 = plt.cm.viridis(np.linspace(0.2, 0.8, len(V0_fig15)))

# (a) Phi/c*^2 profiles
ax = axes[0, 0]
for i, V0 in enumerate(V0_fig15):
    s = get_sol(V0)
    if V0 in analytic_V0s:
        an = get_analytic(V0)
        ax.plot(r_arr, an["Phi"] / cstar_sq, "--", color=colors_15[i], alpha=0.7,
                label=f"analytic $V_0={V0}$")
    ax.plot(r_arr, s["Phi"] / cstar_sq, "-", color=colors_15[i],
            label=f"full $V_0={V0}$")
    if V0 in proxy_V0s:
        p = get_proxy(V0)
        ax.plot(r_arr, p["Phi"] / cstar_sq, "-.", color=colors_15[i],
                lw=0.8, alpha=0.5)
ax.plot([], [], "k-", lw=1.5, label="Full")
ax.plot([], [], "k--", alpha=0.7, label="Analytic")
ax.plot([], [], "k-.", lw=0.8, alpha=0.5, label="Proxy")
ax.set_xlabel("$r/a$")
ax.set_ylabel("$\\Phi / c_*^2$")
ax.set_title("(a) Gravitational potential")
ax.set_xlim(0, 40)
ax.legend(fontsize=7, ncol=2)
ax.axhline(0, color="gray", lw=0.5)

# (b) Lapse N(r) profiles
ax = axes[0, 1]
for i, V0 in enumerate(V0_fig15):
    s = get_sol(V0)
    if V0 in analytic_V0s:
        an = get_analytic(V0)
        ax.plot(r_arr, an["lapse"], "--", color=colors_15[i], alpha=0.7)
    ax.plot(r_arr, s["lapse"], "-", color=colors_15[i],
            label=f"$V_0={V0}$")
    if V0 in proxy_V0s:
        p = get_proxy(V0)
        ax.plot(r_arr, p["lapse"], "-.", color=colors_15[i],
                lw=0.8, alpha=0.5)
ax.plot([], [], "k-", lw=1.5, label="Full")
ax.plot([], [], "k--", alpha=0.7, label="Analytic")
ax.plot([], [], "k-.", lw=0.8, alpha=0.5, label="Proxy")
ax.set_xlabel("$r/a$")
ax.set_ylabel("$N(r)$")
ax.set_title("(b) Lapse profiles")
ax.set_xlim(0, 40)
ax.axhline(1, color="gray", lw=0.5)
ax.legend(fontsize=8)

# (c) Fractional difference (exact - analytic) / |analytic|
ax = axes[1, 0]
for i, V0 in enumerate(V0_fig15):
    s = get_sol(V0)
    if V0 in analytic_V0s:
        an = get_analytic(V0)
        phi_an = an["Phi"]
        phi_ex = s["Phi"]
        mask = np.abs(phi_an) > 1e-6 * np.max(np.abs(phi_an))
        frac_diff = np.full_like(phi_an, np.nan)
        frac_diff[mask] = (phi_ex[mask] - phi_an[mask]) / np.abs(phi_an[mask])
        ax.plot(r_arr, frac_diff, "-", color=colors_15[i],
                label=f"$V_0={V0}$")
        if V0 in proxy_V0s:
            p = get_proxy(V0)
            phi_p = p["Phi"]
            frac_p = np.full_like(phi_an, np.nan)
            frac_p[mask] = (phi_p[mask] - phi_an[mask]) / np.abs(phi_an[mask])
            ax.plot(r_arr, frac_p, "-.", color=colors_15[i],
                    lw=0.8, alpha=0.5)
ax.plot([], [], "k-", lw=1.5, label="Full$-$Analytic")
ax.plot([], [], "k-.", lw=0.8, alpha=0.5, label="Proxy$-$Analytic")
ax.set_xlabel("$r/a$")
ax.set_ylabel("$(\\Phi - \\Phi_{\\mathrm{an}}) / |\\Phi_{\\mathrm{an}}|$")
ax.set_title("(c) Fractional difference")
ax.set_xlim(0, 40)
ax.axhline(0, color="gray", lw=0.5)
ax.legend(fontsize=8)

# (d) Conductance ratio exact/analytic Nbar^2
ax = axes[1, 1]
for i, V0 in enumerate(V0_fig15):
    s = get_sol(V0)
    kappa_ex = s["kappa"] / s["kappa_flat"]
    ax.plot(r_bond, kappa_ex, "-", color=colors_15[i],
            label=f"$V_0={V0}$")
    if V0 in proxy_V0s:
        p = get_proxy(V0)
        ax.plot(r_bond, p["kappa_ratio"], "-.", color=colors_15[i],
                lw=0.8, alpha=0.5)
ax.plot([], [], "k-", lw=1.5, label="Full MI")
ax.plot([], [], "k-.", lw=0.8, alpha=0.5, label="Proxy $\\bar{N}^2$")
ax.set_xlabel("$r/a$")
ax.set_ylabel("$\\kappa / \\kappa^{\\mathrm{flat}}$")
ax.set_title("(d) Conductance ratio")
ax.set_xlim(0, 40)
ax.axhline(1, color="gray", lw=0.5)
ax.legend(fontsize=8)

fig.tight_layout()
path = os.path.join(FIGDIR, "twostate_exact_comparison.pdf")
fig.savefig(path)
plt.close(fig)
print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)
V0s_th = D["thermo_V0"]
print(f"{'V0':>8s} {'rs/a':>8s} {'min_N':>8s} {'kH':>10s} {'TH':>10s}")
print("-" * 50)
for i in range(len(V0s_th)):
    print(f"{V0s_th[i]:8.4f} {D['thermo_rs'][i]:8.4f} {D['thermo_min_N'][i]:8.4f} "
          f"{D['thermo_kappa_H'][i]:10.4e} {D['thermo_T_H'][i]:10.4e}")

print(f"\nAll figures saved to: {FIGDIR}")
print("Done.")
