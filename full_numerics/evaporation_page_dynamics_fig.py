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
    monogamy require, but at t/t_evap = 0.253 rather than at the Page time --
    the trade is over once the register is a few correlation lengths deep.
    (0.253 is the k = 2 value, i.e. on the mass-loss law eq:ev-Mt this model's
    own channel count gives, which is the variable t_Page is expressed in too.
    This said 0.093, the k = 0 value, while the code below has always used
    k = 2 -- so the figure was right and the docstring was not.)

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

# Panel (a) carries the SAME reference cases panel (b) marks, so the model's own
# curve can be judged against them instead of standing alone.  Without the R = 1
# case in particular there is no baseline: it is the unshifted Page curve, where
# the radiation carries exactly the entropy the star loses, and its peak at 0.646
# is what 0.512 should be read against.
PKG = ((2, 1.0,       "0.55",      "reversible, $R=1$ (unshifted)"),
       (2, 4.0/3.0,   "tab:green", "$3{+}1$D blackbody, $R=4/3$"),
       (2, R_page,    "tab:olive", "Page 1976, $\\gamma+$graviton"),
       (2, R_dirac,   "tab:blue",  "massless Dirac, this model"))

_iy = [0]
fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9))

# -- (a) the Page curve ----------------------------------------------------
ax = axes[0]
t = np.linspace(0.0, 1.0, 801)
for k, R, col, lab in PKG:
    late = (1.0 - t) ** (2.0 / (k + 1.0))
    ax.plot(t, np.minimum(R * (1.0 - late), late), color=col, lw=2.0,
            label=rf"$k={k}$, $R={R:.3f}$ ({lab})")
    if R == R_dirac:
        ax.plot(t, late, color=col, lw=0.9, ls="--", alpha=0.6)
    # No vertical markers or inline numbers here: four of them collided with each
    # other and with the legend.  The Page times are quoted to three digits in
    # the caption, and panel (b) locates them against R.
    _iy[0] += 1
ax.plot([], [], color="0.5", lw=0.9, ls="--",
        label=r"late branch $S_{\rm ent}(t)$")
ax.set_xlabel(r"$t/t_{\rm evap}$")
ax.set_ylabel(r"$S_R(t)\,/\,S_{\rm ent}(0)$")
ax.set_title("(a) Page curve with entropy production", fontsize=10)
ax.legend(fontsize=6.6, frameon=False, loc="upper right")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)

# Panel (b) -- t_Page against R -- was removed: it plotted a closed-form
# expression with four marked points, and the four values are quoted in the
# caption.  R itself appears in panel (a)'s legend.

# -- (c) monogamy against an accumulating emission register ----------------
ax = axes[1]
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
_bud = 2 * float(q["S_cap"][0])
ax.annotate(rf"$2S(b)={_bud:.3f}$: monogamy budget (exact on a pure state)",
            xy=(0.02, _bud + 0.025), fontsize=7.0, color="0.35")
ax.annotate("closes exactly on the purified state", xy=(0.30, _bud - 0.085),
            fontsize=6.8, color="0.45")
# The shortfall below that line is what b shares with the thermal purifier
# rather than with either side -- see the caption.
# The monogamy budget is shared with the WHOLE exterior, so the sum that must
# reach 2S(b) on a pure state is I(b:S) + I(b:E) + I(b:not-yet-vacated).  An
# earlier version of this line omitted the third term and so appeared to show a
# large deficit; the deficit is small.
# Monogamy, evaluated on the PURIFIED state.  I(b:S) + I(b:exterior) with the
# exterior taken to include the thermofield-double purifier, so b, S and the
# exterior exhaust a pure state and the identity 2S(b) is exact rather than
# approached.  Plotting the chain alone leaves a 2.5-6.3% residue that is the
# purifier omitted, not a physical deficit.
_sum = q["I_S_pure"] + q["I_ext_pure"]
ax.plot(tt, _sum, "-", color="0.45", lw=1.4, ls=(0, (5, 2)),
        label=r"$I(b{:}S)+I(b{:}\mathrm{ext})$, purified")
d = q["I_early"] - q["I_S"]
i = int(np.where(np.diff(np.sign(d)))[0][0])
f = -d[i] / (d[i + 1] - d[i])
t_x = tt[i] + f * (tt[i + 1] - tt[i])
ax.axvline(t_x, color="tab:red", ls=":", lw=1.0)
ax.annotate(rf"trade at ${t_x:.3f}$", xy=(t_x + 0.02, 0.86),
            fontsize=7.5, color="tab:red")
ax.axvline(t_page(R_dirac, 2), color="0.4", ls=":", lw=1.2)
ax.axvspan(t_x, t_page(R_dirac, 2), color="0.85", alpha=0.45, zorder=0)
ax.annotate(rf"$t_{{\rm Page}}={t_page(R_dirac,2):.3f}$",
            xy=(t_page(R_dirac, 2) + 0.03, 0.16), fontsize=7.5, color="0.3")
ax.set_xlabel(r"$t/t_{\rm evap}$")
ax.set_ylabel("mutual information (nats)")
ax.set_title("(b) monogamy with an emission register", fontsize=10)
# Upper right but BELOW the 2S(b) line and its annotation: the band between the
# red curve's saturation (~0.71) and the "closes exactly" text (~1.30) is the
# only clear space, and the legend collided with the curves everywhere else.
ax.legend(fontsize=6.2, frameon=False, loc="upper right",
          bbox_to_anchor=(0.995, 0.86))
ax.set_xlim(-0.02, 0.95)
_top = 2 * float(q["S_cap"][0]) * 1.06
ax.set_ylim(0, _top)          # was a hard 1.95, which left the data in the bottom third
print(f"AMPS register (beta0t0 = {q['beta0t0'][0]:.0f}..{q['beta0t0'][-1]:.0f},"
      f" r_s/a = {q['rs'][0]:.1f}..{q['rs'][-1]:.1f}): crossing at "
      f"t/t_evap = {t_x:.4f}, m_shed = "
      f"{q['m_shed'][i] + f*(q['m_shed'][i+1]-q['m_shed'][i]):.2f}; "
      f"monogamy on the purified state closes to "
      f"{abs(q['closure']).max():.1e} of 2S(b) = {2*q['S_cap'][0]:.4f} "
      f"(chain alone leaves {q['deficit'].min():.4f}-{q['deficit'].max():.4f}, "
      f"which is the purifier omitted)")

fig.tight_layout()
out = os.path.join(FIGDIR, "evaporation_page_dynamics.pdf")
fig.savefig(out, bbox_inches="tight")
print(f"wrote {out}")
