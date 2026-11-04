#!/usr/bin/env python
"""figure_item5_collapse_v3.py (busi lane, inbox #351): Fig. 1 (Sec. IV, collapse) drawn to reviewer A's plan
(deliveries/lead_gaps/ReviewA_figures_v1.md, section 2). Loads only fig_item5_collapse_v3_data.npz (figures_data_v1.py) and draws;
nothing is computed beyond the sign level 0 of the stored N - |v| (contourf, panel b; the where-mask, panel a).
Size: the present item5_collapse_v2.pdf's box, 720 x 288 pt (10 x 4 in), placed at \\textwidth (about 7 in), so every font, line
width and marker size below is the plan's printed size times 10/7.
Colours black, red, blue; grey fills and reference lines. No grid, no symlog, no run names.
Usage: python figure_item5_collapse_v3.py DATA_NPZ OUT_STEM   (writes OUT_STEM.pdf and OUT_STEM.png)
"""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

S = 10.0 / 7.0                      # print scale (10 in figure placed at about 7 in)
LW_BOLD, LW, LW_FAINT, MS = 1.8 * S, 1.2 * S, 0.7 * S, 3.0 * S
FILL = "0.85"
REF = "0.55"


def letter(ax, s, box=False):
    kw = dict(bbox=dict(facecolor="white", edgecolor="none", pad=1.0)) if box else {}
    ax.text(0.025, 0.975, s, transform=ax.transAxes, ha="left", va="top", fontweight="bold", zorder=10, **kw)


def main():
    p_npz, stem = sys.argv[1:3]
    Z = np.load(p_npz)
    plt.rcParams.update({"font.size": 7.5 * S, "axes.labelsize": 7.5 * S, "xtick.labelsize": 7 * S, "ytick.labelsize": 7 * S,
                         "legend.fontsize": 6.8 * S, "axes.linewidth": 0.6 * S, "xtick.major.width": 0.6 * S,
                         "ytick.major.width": 0.6 * S, "xtick.major.size": 3 * S, "ytick.major.size": 3 * S,
                         "mathtext.fontset": "dejavusans", "pdf.fonttype": 42})
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(10.0, 4.0))
    fs_lab = 6.8 * S

    # (a) profiles before release and at 15 r_s
    r = Z["a_r_over_rs"]
    z0, z1 = Z["a_zeros_T15_over_rs"]
    a1.axvspan(z0, z1, color=FILL, lw=0, zorder=0)
    a1.text(0.38, 0.88, "trapped", ha="center", va="center", fontsize=fs_lab)
    a1.plot(r, Z["a_N_T0"], "-", color="black", lw=LW)
    a1.plot(r, Z["a_absv_T0"], "--", color="black", lw=LW)
    a1.plot(r, Z["a_N_T15"], "-", color="red", lw=LW)
    a1.plot(r, Z["a_absv_T15"], "--", color="red", lw=LW)
    a1.plot([float(Z["a_Rt_T15_over_rs"])], [float(Z["a_circle_N"])], "o", color="red", ms=MS * 1.4, zorder=5)
    xe = r[-1] - 0.03
    for y, dy, txt, col in ((Z["a_N_T0"][-1], 0.015, r"$N$, before release", "black"),
                            (Z["a_absv_T0"][-1], 0.03, r"$|v|$, before release", "black"),
                            (Z["a_N_T15"][-1], -0.022, r"$N$ at 15 $r_s$", "red"),
                            (Z["a_absv_T15"][-1], -0.03, r"$|v|$ at 15 $r_s$", "red")):
        a1.text(xe, y + dy, txt, color=col, ha="right", va="top" if dy < 0 else "bottom", fontsize=fs_lab)
    a1.set_xlim(0, 2); a1.set_ylim(0, 1.05)
    a1.set_xlabel(r"radius $r/r_s$")
    a1.set_ylabel(r"clock rate $N$ and inflow speed $|v|$")
    letter(a1, "(a)")

    # (b) the trapped region in time and radius
    T = Z["b_T_over_rs"]; rr = Z["b_r_over_rs"]; F = Z["b_N_minus_absv"]
    a2.contourf(T, rr, F.T, levels=[min(F.min(), -1.0) - 1.0, 0.0], colors=[FILL], zorder=0)
    Tr = Z["b_T_readout"]
    a2.plot(Tr, Z["b_Rt_over_rs"], "-", color="black", lw=LW)
    a2.plot(Tr, Z["b_2gE_over_rs"], ":", color="black", lw=LW)
    a2.plot(Tr, Z["b_Rc_over_rs"], "-", color="red", lw=LW)
    tf = float(Z["ev_formation"]); tc = float(Z["ev_closes"]); to = float(Z["ev_reopens"])
    a2.axvline(tf, color=REF, ls="--", lw=LW_FAINT * 1.2, zorder=1)
    a2.text(tf + 0.3, 1.20, "surface forms", ha="left", va="top", fontsize=fs_lab, color="0.3")
    tg = 0.5 * (tc + to)
    a2.annotate("closes %.1f, reopens %.1f" % (tc, to), xy=(tg, 0.42), xytext=(18.0, 1.10), fontsize=fs_lab,
                ha="left", va="center", arrowprops=dict(arrowstyle="->", lw=0.6 * S, color="black", shrinkA=1, shrinkB=1))
    ok = np.isfinite(Z["b_Rt_over_rs"])
    for y, dy, txt, col in ((Z["b_Rt_over_rs"][ok][-1], -0.03, "outer edge of\ntrapped region", "black"),
                            (Z["b_2gE_over_rs"][-1], -0.02, "$2GE$\n(whole object)", "black"),
                            (1.30, 0.0, "mean radius of the matter", "red")):
        a2.text(39.6, y + dy, txt, color=col, ha="right", va="top" if dy < 0 else "bottom", fontsize=6.2 * S, zorder=6)
    a2.set_xlim(10, 40); a2.set_ylim(0, 1.5)
    a2.set_xlabel(r"time at infinity $T/r_s$")
    a2.set_ylabel(r"radius $r/r_s$")
    letter(a2, "(b)", box=True)

    # (c) depth of trapping across the averaging width
    a3.axhline(0.0, color=REF, lw=LW_FAINT, zorder=0)
    a3.text(39.5, -0.45, "trapped where negative", ha="right", va="top", fontsize=fs_lab, color="0.3")
    a3.plot(Z["c_T_s2"], Z["c_minNv_s2"], "-", color="blue", lw=LW, label="averaging width 2 spacings")
    a3.plot(Z["c_T_s4"], Z["c_minNv_s4"], "-", color="black", lw=LW_BOLD, label="4 spacings (as (a),(b))")
    a3.plot(Z["c_T_s8"], Z["c_minNv_s8"], "-", color="red", lw=LW, label="8 spacings")
    a3.set_xlim(10, 40); a3.set_ylim(-0.8, 0.5)
    a3.set_xlabel(r"time at infinity $T/r_s$")
    a3.set_ylabel(r"depth of trapping, min over $r$ of $N-|v|$")
    a3.legend(loc="upper right", frameon=False, handlelength=1.4, borderaxespad=0.3, fontsize=6.0 * S)
    letter(a3, "(c)")

    fig.tight_layout(pad=0.4, w_pad=1.2)
    fig.savefig(stem + ".pdf")
    fig.savefig(stem + ".png", dpi=200)
    print("wrote %s.pdf and %s.png" % (stem, stem))


if __name__ == "__main__":
    main()
