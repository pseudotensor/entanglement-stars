#!/usr/bin/env python3
"""a4_beta_eff_fig_v1.py (solve, page_verify A4): the figure from A4_beta_eff_v1.npz only (no solver, no stored-run reads).
Panel a: beta_eff per window against the window's midpoint on the collar clock (6 r_s windows from 46 r_s, the burst 40-46, the
formation sub-window 36-40): cut-based at the 16 r_s cut (unlagged; it is the collar) and at the 2 r_s cut (lagged by the ray map),
K = 18 against the K = 18 collar flow and kappa 1 alone; occupation-based (the 24 r_s shells, windows shifted by the 16 -> 24 lag);
reference lines 4/3, 2, 0.2 and the estimator's thermal calibration on the same modes (dashed).
Panel b: beta_occ per |e|/T_H band over 46-100 r_s on the shells' clock against the thermal fill's per-band value.
Usage: a4_beta_eff_fig_v1.py A4_beta_eff_v1.npz OUT.png
"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    z = np.load(sys.argv[1], allow_pickle=False)
    rc = json.loads(str(z["rows_cut_json"])); ro = json.loads(str(z["rows_occ_json"]))
    sel_w = lambda w: w.startswith("late6_") or w in ("burst", "formation_c")
    mid = lambda r: 0.5 * (r["A"] + r["B"])
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.6, 1]})
    styles = [("K18", "16", "black", "o", "-", "cut 16 r_s, K = 18 / L_18"),
              ("K18", "2", "red", "s", "-", "cut 2 r_s (lagged), K = 18 / L_18"),
              ("k1", "16", "black", "o", ":", "cut 16 r_s, kappa 1 / L_1"),
              ("k1", "2", "red", "s", ":", "cut 2 r_s (lagged), kappa 1 / L_1")]
    for st, cut, col, mk, ls, lab in styles:
        R = sorted([r for r in rc if r["set"] == st and r["cut"] == cut and sel_w(r["window"]) and r["beta"] is not None], key=mid)
        ax.plot([mid(r) for r in R], [r["beta"] for r in R], color=col, marker=mk, ls=ls, ms=5, lw=1.4 if ls == "-" else 1.0,
                mfc=col if ls == "-" else "white", label=lab)
    for st, col, ls, lab in (("K18", "blue", "-", "occupations (24 r_s shells), K = 18"), ("k1", "blue", ":", "occupations, kappa 1")):
        R = sorted([r for r in ro if r["set"] == st and r["shifted_by"] > 0 and sel_w(r["window"]) and r["beta_occ"] is not None],
                   key=lambda r: 0.5 * (r["A_collar"] + r["B_collar"]))
        ax.plot([0.5 * (r["A_collar"] + r["B_collar"]) for r in R], [r["beta_occ"] for r in R], color=col, marker="^", ls=ls, ms=5,
                mfc=col if ls == "-" else "white", label=lab)
    for y, lab in ((2.0, "2 (one transmitted channel)"), (4.0 / 3.0, "4/3 (black body)"), (0.2, "0.2 (driven estimate)")):
        ax.axhline(y, color="gray", lw=0.9, ls="--"); ax.text(99.5, y + 0.03, lab, ha="right", va="bottom", fontsize=8, color="dimgray")
    calK = float(np.nanmedian(z["cal_K18"][z["cal_t"] >= 40])); cal1 = float(np.nanmedian(z["cal_k1"][z["cal_t"] >= 40]))
    ax.axhline(calK, color="blue", lw=0.8, ls="-.", alpha=0.7); ax.text(72, calK + 0.03, "thermal fill on the shells' modes, K = 18: %.2f" % calK, fontsize=8, color="blue")
    ax.axhline(cal1, color="blue", lw=0.8, ls=(0, (1, 3)), alpha=0.7); ax.text(72, cal1 - 0.09, "same, kappa 1: %.2f" % cal1, fontsize=8, color="blue")
    ax.axvspan(40, 46, color="tab:orange", alpha=0.12, lw=0); ax.text(43, 2.18, "burst", ha="center", fontsize=8)
    ax.set_xlim(36, 100); ax.set_ylim(-0.1, 2.3)
    ax.set_xlabel("window midpoint, collar (16 r_s) record clock T / r_s")
    ax.set_ylabel(r"$\beta_{\rm eff} = T_H\,\Delta S / \Delta E$")
    ax.set_title("a. entropy per emitted energy, per window (6 r_s windows from 46)", fontsize=10)
    ax.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(0.0, 0.66), ncol=1, framealpha=0.9)
    # panel b
    bands = [str(b) for b in z["occ_bands"]]
    r = [r for r in ro if r["set"] == "K18" and r["window"] == "late_46_100" and r["shifted_by"] == 0][0]
    r1 = [r for r in ro if r["set"] == "k1" and r["window"] == "late_46_100" and r["shifted_by"] == 0][0]
    xb = np.arange(len(bands))
    cb = np.nanmedian(z["cal_K18_band"][z["cal_t"] >= 40], axis=0); cb1 = np.nanmedian(z["cal_k1_band"][z["cal_t"] >= 40], axis=0)
    eb = np.array(r["dE_band"]); frac = eb / eb.sum()
    ok = np.abs(frac) > 1e-3
    bx.plot(xb[ok], np.array(r["beta_band"], float)[ok], "o-", color="black", label="observed, K = 18 (46-100 r_s)")
    bx.plot(xb[ok], np.array(r1["beta_band"], float)[ok], "o:", color="red", mfc="white", label="observed, kappa 1")
    bx.plot(xb, cb, "^--", color="blue", label="thermal fill at T_H, K = 18")
    bx.plot(xb, cb1, "^:", color="blue", mfc="white", label="thermal fill, kappa 1")
    for i in range(len(bands)):
        bx.text(xb[i], -0.25, "%.0f%%" % (100 * frac[i]) if abs(frac[i]) >= 1e-3 else "<0.1%", ha="center", fontsize=7.5, color="dimgray")
    bx.set_xticks(xb); bx.set_xticklabels(bands, fontsize=8)
    bx.set_xlabel(r"band of $|e|/T_H^{\rm ref}$ (share of the window's energy below)")
    bx.set_ylabel(r"$T_H\,\Delta S_{\rm band}/\Delta E_{\rm band}$")
    bx.set_ylim(-0.35, 4.0); bx.axhline(0, color="gray", lw=0.6)
    bx.set_title("b. occupation entropy per energy by band, 24 r_s shells", fontsize=10)
    bx.legend(fontsize=7.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(sys.argv[2], dpi=150)
    print("wrote", sys.argv[2])


if __name__ == "__main__":
    main()
