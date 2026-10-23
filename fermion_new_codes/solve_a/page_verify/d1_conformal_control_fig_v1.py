#!/usr/bin/env python3
"""d1_conformal_control_fig_v1.py (solve lane, PAGE VERIFY D1): figure from d1_conformal_control_v1.npz.
Per cut (2, 16, 24, 30, 48 r_s): the control's entropy growth per copy and its printed excess per copy against the
record clock, the finite-chain (circle) form of the formula, the sheet's line formula and its truncated curve; the
window where the continuum formula is found to apply (Lam < 3 at that cut) shaded, the band-edge regime (3 <= Lam < 7.7)
hatched lightly; the production kappa = 1 chain's growth per copy for scale. Colors black / red / blue only.
Usage: python d1_conformal_control_fig_v1.py [--npz d1_conformal_control_v1.npz] [--out d1_conformal_control_v1.png]
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--npz", default="d1_conformal_control_v1.npz")
ap.add_argument("--out", default="d1_conformal_control_v1.png")
args = ap.parse_args()
z = np.load(args.npz)
t = z["clocks"]
cuts = ["2", "16", "24", "30", "48"]
fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
axes = axes.ravel()
for i, cut in enumerate(cuts):
    ax = axes[i]
    tag = "R%s" % cut
    Lam = z["Lam_%s" % tag]
    growth = z["growth_%s" % tag]; excess = z["excess_%s" % tag]
    circ = z["dS_circ_%s" % tag]; wall = z["dS_wall_%s" % tag]; trunc = z["trunc_pi_%s" % tag]; k1 = z["k1_growth_%s" % tag]
    # regimes
    tI = t[(Lam < 3.0)]; tII = t[(Lam >= 3.0) & (Lam < 7.7)]
    if tI.size:
        ax.axvspan(tI.min(), tI.max(), color="0.85", lw=0, label="formula applies (Lam < 3)" if i == 0 else None)
    if tII.size:
        ax.axvspan(tII.min(), tII.max(), facecolor="none", edgecolor="0.6", hatch="//", lw=0, label="band-edge regime (3 <= Lam < 7.7)" if i == 0 else None)
    ax.plot(t, circ, color="red", lw=2.0, label="formula, finite chain (circle form)")
    ax.plot(t, wall, color="blue", lw=1.4, ls="--", label="sheet's line formula dS_wall")
    ax.plot(t, trunc, color="blue", lw=1.2, ls=":", label="sheet's truncated curve (c = pi)")
    ax.plot(t, growth, color="black", lw=1.8, marker="o", ms=3, label="control: growth [S(t) - S(0)]/2 per copy")
    ax.plot(t, excess, color="black", lw=0.9, ls="-", alpha=0.45, marker="s", ms=2.5, label="control: printed excess over held vacuum, per copy")
    ax.plot(t, k1, color="0.4", lw=1.2, ls="-.", label="production kappa = 1: growth per copy")
    ax.set_title("cut %s r_s" % cut)
    ax.set_ylabel("nats per copy")
    ax.grid(alpha=0.3)
    ymax = max(np.nanmax(growth), np.nanmax(excess), 1.0) * 1.8
    ax.set_ylim(min(-0.15, np.nanmin(excess) - 0.05), min(ymax, 5.0))
    ax.set_xlim(0, 100)
    ax.set_xlabel("record clock t (r_s)")
# the sixth panel: the difference growth - circle at every cut against the cut's own blueshift
ax = axes[5]
for cut, col, ls in zip(cuts, ["black", "red", "blue", "0.5", "0.75"], ["-", "-", "-", "--", ":"]):
    tag = "R%s" % cut
    Lam = z["Lam_%s" % tag]; d = z["d_circ_%s" % tag]
    m = Lam > 0.05
    ax.plot(Lam[m], d[m], color=col, ls=ls, lw=1.5, marker="o", ms=3, label="cut %s r_s" % cut)
ax.axhline(0, color="black", lw=0.8)
ax.axhspan(-0.05, 0.05, color="0.9", lw=0)
ax.axvline(3.0, color="0.5", lw=0.8); ax.axvline(7.7, color="0.5", lw=0.8)
ax.set_xlim(0, 30)
ax.set_ylim(-1.0, 0.2)
ax.set_xlabel("blueshift Lam = -ln p' at the cut (e-foldings)")
ax.set_ylabel("growth - formula (circle), nats per copy")
ax.set_title("departure from the formula against the blueshift")
ax.legend(fontsize=8, loc="lower left")
ax.grid(alpha=0.3)
axes[0].legend(fontsize=7.5, loc="upper left")
fig.suptitle("D1 conformal control: kappa = 0 chain on B2's fields (v6s_B2_kz) against the exact 1+1 Gaussian formula on the traced ray map, per copy")
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(args.out, dpi=140)
print("wrote", args.out)
