#!/usr/bin/env python3
"""
FIGURE script for the Page curve.  Loads
  fermion/numerical/data/greybody_entropy_ratio.npz   (R and its anchors)
  fermion/numerical/data/amps_cold_branch.npz         (cap correlations)

Panels
------
(a) The Page curve for the mass-loss law of sec:ev-massloss.  The angular
    Clifford sector gaps every channel lambda > 0, giving the massless Dirac
    barrier of the exterior and hence dM/dt ~ -M^{-2}.  The early branch is
    the entropy the radiation carries, R [S_ent(0) - S_ent(t)], not the
    entropy the star loses.
(b) t_Page/t_evap = 1 - [R/(1+R)]^{3/2}, with the measure's exact
    Gamma in {0,1} limits, the Page 1976 anchor, and the model's value.
(c) Monogamy with an accumulating emission register: I(b : star) against
    I(b : early radiation), b being the newly exposed cap mode and the
    register the shells the cap has vacated.  They cross, as purity and
    monogamy require, but at t/t_evap = 0.093 rather than at the Page time --
    the trade is over once the register is a few correlation lengths deep.

Output: fermion/numerical/figures/evaporation_page_dynamics.pdf
Run: python3 full_numerics/evaporation_page_dynamics_fig.py
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATADIR = os.path.join(ROOT, "fermion", "numerical", "data")
FIGDIR = os.path.join(ROOT, "fermion", "numerical", "figures")
os.makedirs(FIGDIR, exist_ok=True)

g = np.load(os.path.join(DATADIR, "greybody_entropy_ratio.npz"))
R_dirac = float(g["R_dirac_fermi"])          # the model's own barrier -> k = 2
R_page = float(g["A2_page"])


def t_page(R, k):
    return 1.0 - (R / (1.0 + R)) ** (0.5 * (k + 1.0))


print(f"anchors: flat 1D {float(g['A1_flat_fermi']):.5f} (exact 2), "
      f"geom. optics {float(g['A1_go_cont_fermi']):.5f} (exact {4/3:.5f})")
print(f"Page 1976 anchor: {R_page:.5f} (literature 1.4847)")
print(f"massless Dirac (k=2): R = {R_dirac:.5f}, "
      f"t_Page = {t_page(R_dirac, 2):.5f}")

PKG = ((2, R_dirac, "tab:blue", "massless Dirac barrier"),)

fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9))

# -- (a) the Page curve ----------------------------------------------------
ax = axes[0]
t = np.linspace(0.0, 1.0, 801)
for k, R, col, lab in PKG:
    late = (1.0 - t) ** (2.0 / (k + 1.0))
    ax.plot(t, np.minimum(R * (1.0 - late), late), color=col, lw=2.0,
            label=rf"$k={k}$, $R={R:.3f}$ ({lab})")
    ax.plot(t, late, color=col, lw=0.9, ls="--", alpha=0.6)
    tp = t_page(R, k)
    ax.axvline(tp, color=col, ls=":", lw=1.0)
    ax.annotate(rf"${tp:.3f}$", xy=(tp + 0.015, 0.88),
                fontsize=8, color=col)
ax.plot([], [], color="0.5", lw=0.9, ls="--",
        label=r"late branch $S_{\rm ent}(t)$")
ax.set_xlabel(r"$t/t_{\rm evap}$")
ax.set_ylabel(r"$S_R(t)\,/\,S_{\rm ent}(0)$")
ax.set_title("(a) Page curve with entropy production", fontsize=10)
ax.legend(fontsize=7.5, frameon=False, loc="lower center")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)

# -- (b) t_Page vs R -------------------------------------------------------
ax = axes[1]
Rg = np.linspace(1.0, 2.15, 400)
for k, _, col, lab in PKG:
    ax.plot(Rg, t_page(Rg, k), color=col, lw=1.8, label=rf"$k={k}$ ({lab})")
marks = [(1.0, 2, "reversible emission", "tab:red", "o"),
         (4.0 / 3.0, 2, r"$3{+}1$D blackbody", "tab:green", "s"),
         (R_page, 2, "Page 1976, $\\gamma+$graviton", "tab:olive", "^"),
         (R_dirac, 2, "massless Dirac, this model", "tab:blue", "*")]
for R, k, lab, col, mk in marks:
    ax.plot([R], [t_page(R, k)], mk, color=col, ms=10 if mk == "*" else 5.5,
            label=rf"$R={R:.3f}$: $t_{{\rm Page}}={t_page(R,k):.3f}$"
                  rf"  {lab}")
ax.set_xlabel(r"entropy-to-energy emission ratio $R=T_H J_S/J_E$")
ax.set_ylabel(r"$t_{\rm Page}/t_{\rm evap}$")
ax.set_title(r"(b) $t_{\rm Page}=1-[R/(1{+}R)]^{3/2}$", fontsize=10)
ax.legend(fontsize=6.2, frameon=False, loc="upper right")

# -- (c) monogamy against an accumulating emission register ----------------
ax = axes[2]
q = np.load(os.path.join(DATADIR, "amps_register.npz"))
# same ladder, renormalised to the k = 2 lifetime: t/t_evap = 1 - (rs/rs0)^3
tt = 1.0 - (q["rs"] / float(q["rs0"])) ** 3
ax.plot(tt, q["I_S"], "-o", color="k", ms=3.5, lw=1.6,
        label=r"$I(b:\mathrm{star})$")
ax.plot(tt, q["I_early"], "-s", color="tab:red", ms=3.5, lw=1.6,
        label=r"$I(b:\mathrm{early\ radiation})$")
ax.plot(tt, q["I_far"], "-^", color="tab:blue", ms=3.5, lw=1.6,
        label=r"$I(b:\mathrm{not\ yet\ vacated})$")
ax.axhline(2 * float(q["S_cap"][0]), color="0.6", lw=0.9, ls="--")
ax.annotate(r"$2S(b)$", xy=(0.62, 2 * float(q["S_cap"][0]) - 0.06),
            fontsize=7.5, color="0.4")
d = q["I_early"] - q["I_S"]
i = int(np.where(np.diff(np.sign(d)))[0][0])
f = -d[i] / (d[i + 1] - d[i])
t_x = tt[i] + f * (tt[i + 1] - tt[i])
ax.axvline(t_x, color="tab:red", ls=":", lw=1.0)
ax.annotate(rf"trade at ${t_x:.3f}$", xy=(t_x + 0.03, 0.30), fontsize=7.5,
            color="tab:red")
ax.axvline(t_page(R_dirac, 2), color="0.4", ls=":", lw=1.2)
ax.annotate(rf"$t_{{\rm Page}}={t_page(R_dirac,2):.3f}$",
            xy=(t_page(R_dirac, 2) + 0.03, 0.16), fontsize=7.5, color="0.3")
ax.set_xlabel(r"$t/t_{\rm evap}$")
ax.set_ylabel("mutual information (nats)")
ax.set_title("(c) monogamy with an emission register", fontsize=10)
ax.legend(fontsize=7.0, frameon=False, loc="upper right")
ax.set_xlim(-0.02, 0.95)
ax.set_ylim(0, 1.95)
print(f"AMPS register (beta0t0 = {q['beta0t0'][0]:.0f}..{q['beta0t0'][-1]:.0f},"
      f" r_s/a = {q['rs'][0]:.1f}..{q['rs'][-1]:.1f}): crossing at "
      f"t/t_evap = {t_x:.4f}, m_shed = "
      f"{q['m_shed'][i] + f*(q['m_shed'][i+1]-q['m_shed'][i]):.2f}; "
      f"monogamy deficit {q['deficit'].min():.4f}-{q['deficit'].max():.4f} "
      f"of 2S(b) = {2*q['S_cap'][0]:.4f}")

fig.tight_layout()
out = os.path.join(FIGDIR, "evaporation_page_dynamics.pdf")
fig.savefig(out, bbox_inches="tight")
print(f"wrote {out}")
