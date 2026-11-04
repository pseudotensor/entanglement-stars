#!/usr/bin/env python
"""page_v5_fig.py: copy of busi/code/figures/figure_page_v5.py (md5 6a8e6283fa35b6dd3ff23c6959e0190a); only change:
panel (a) legend (three cuts) moved to the lower right, the thermal-budget line labelled at its upper end,
and the A/4G note moved to the upper left (owner, 2026-11-03).
Usage here: python page_v5_fig.py page_v5_data.npz page_v5

figure_page_v5.py (busi lane, inbox #351): Fig. 2 (Sec. V, radiation and entropy) drawn to reviewer A's plan
(deliveries/lead_gaps/ReviewA_figures_v1.md, section 3). Loads only fig_page_v5_data.npz (figures_data_v1.py) and draws; the
+-5 r_s mean of panel (b) is stored there. The thermal-budget line and the annotations carry the plan's printed numbers.
Size: the present page_v4.pdf's box, 504 x 176.4 pt (7.0 x 2.45 in), placed at \\textwidth: sizes are the plan's printed sizes.
Colours black, red, blue; grey reference lines. No grid, no symlog, no run names.
Usage: python figure_page_v5.py DATA_NPZ OUT_STEM   (writes OUT_STEM.pdf and OUT_STEM.png)
"""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LW_BOLD, LW, LW_FAINT, MS = 1.8, 1.2, 0.7, 3.0
REF = "0.55"
CUTS = (("2", "black"), ("16", "red"), ("30", "blue"))
T_FORM, BUDGET_100 = 12.125, 10.4          # the plan's printed endpoints of the thermal budget line
HOLD = 100.0


def letter(ax, s, x=0.03, y=0.97):
    ax.text(x, y, s, transform=ax.transAxes, ha="left", va="top", fontweight="bold")


def main():
    p_npz, stem = sys.argv[1:3]
    Z = np.load(p_npz)
    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                         "legend.fontsize": 5.8, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                         "xtick.minor.width": 0.45, "ytick.minor.width": 0.45, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                         "mathtext.fontset": "dejavusans", "pdf.fonttype": 42})
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(7.0, 2.45))
    fs = 5.8

    # (a) entanglement excess beyond the cut, 0-100
    t = Z["a_t_over_rs"]; s = t <= 100.0 + 1e-9
    for c, col in CUTS:
        a1.plot(t[s], Z["a_S_cut" + c][s], "-o", color=col, lw=LW, ms=MS, mew=0, label=r"cut at %s $r_s$" % c)
        for x in Z["a_light_cut" + c]:
            a1.plot([x, x], [11.4, 12.0], "-", color=col, lw=LW)
    a1.plot([T_FORM, 100.0], [0.0, BUDGET_100], "--", color=REF, lw=LW_FAINT)
    # the thermal budget is labelled at the line's upper end (a fourth legend entry would not clear the blue jump at 58)
    a1.text(99, 10.85, r"thermal budget at $T_H$", ha="right", va="bottom", fontsize=fs, color="0.3")
    a1.set_xlim(0, 100); a1.set_ylim(-1, 12)
    a1.set_xlabel(r"time at infinity $T/r_s$")
    a1.set_ylabel("excess entropy beyond the cut (nats)")
    # legend at the lower right, under the curves (owner: the upper-left box overlapped the black and grey curves)
    a1.legend(loc="lower right", frameon=False, handlelength=0.9, handletextpad=0.3, borderaxespad=0.15, labelspacing=0.25)
    # the A/4G number is stated in the caption (owner, 2026-11-04)
    letter(a1, "(a)", x=0.19, y=0.995)

    # (b) collar flux over Hawking flux, log axis
    tb = Z["b_t_over_rs"]; r = Z["b_ratio_signed"]; m = Z["b_ratio_mean_pm5"]
    ok = np.isfinite(r) & (r != 0) & (tb >= 10 - 1e-9)
    pos = ok & (r > 0); ngt = ok & (r < 0)
    a2.plot(tb[pos], r[pos], "o", color="black", ms=MS, mew=0, alpha=0.45)
    a2.plot(tb[ngt], -r[ngt], "o", color="black", ms=MS, mfc="none", mew=0.6, alpha=0.6)
    mm = np.where(np.isfinite(m) & (m > 0) & (tb >= 10 - 1e-9), m, np.nan)
    a2.plot(tb, mm, "-", color="black", lw=LW_BOLD)
    a2.axhline(1.0, color=REF, ls="--", lw=LW_FAINT)
    a2.text(99, 0.8, "Hawking flux", ha="right", va="top", fontsize=fs, color="0.3")
    j = np.nanargmax(np.where((tb >= 40) & (tb <= 48), np.abs(r), np.nan))
    a2.annotate("formation's light\narrives", xy=(tb[j], abs(r[j])), xytext=(56, 300), fontsize=fs, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", lw=0.6, color="black", shrinkA=1, shrinkB=2))
    a2.set_yscale("log")
    a2.set_xlim(10, 100); a2.set_ylim(1e-2, 1e3)
    a2.set_xlabel(r"time at infinity $T/r_s$")
    a2.set_ylabel(r"collar flux / Hawking flux, $L_{\rm col}/L_H$")
    letter(a2, "(b)")

    # (c) no turnover on held geometry, 0-500
    tc = Z["c_t_over_rs"]
    for c, col in CUTS:
        a3.plot(tc, Z["c_excess16_cut" + c], "-o", color=col, lw=LW, ms=MS, mew=0, label=r"cut at %s $r_s$" % c)
    a3.axvline(HOLD, color=REF, ls="--", lw=LW_FAINT)
    a3.text(HOLD + 8, 4.8, "fields frozen\nfrom here " + "→", ha="left", va="center", fontsize=fs, color="0.3")
    a3.text(0.97, 0.03, r"thermal lifetime $3.9\times10^5\,r_s$", transform=a3.transAxes, ha="right", va="bottom", fontsize=fs)
    a3.set_xlim(0, 500); a3.set_ylim(0, 30)
    a3.set_xlabel(r"time at infinity $T/r_s$")
    a3.set_ylabel("excess, lowest channel (nats)")
    a3.legend(loc="upper left", frameon=False, handlelength=1.8, borderaxespad=0.2, bbox_to_anchor=(0.06, 0.93))
    letter(a3, "(c)")

    fig.tight_layout(pad=0.3, w_pad=0.6)
    fig.savefig(stem + ".pdf")
    fig.savefig(stem + ".png", dpi=300)
    print("wrote %s.pdf and %s.png" % (stem, stem))


if __name__ == "__main__":
    main()
