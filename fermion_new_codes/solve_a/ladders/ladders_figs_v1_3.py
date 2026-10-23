#!/usr/bin/env python
"""ladders_figs_v1.py (composition builder, the solve lane): the figures of MASS_ladder_v1 and SCALING_ladder_v1, from the producer's
ladders_readout_v1_3.npz and ladders_readout_v1_3.json only (no run file is read; no number is computed beyond evaluating the stored fit
coefficients on a plotting grid). Colours: black, red, blue and tab10; never yellow or viridis.
Usage: python ladders_figs_v1.py --readout-dir DIR --out-dir DIR"""
import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COL = {"M3b_s4": "tab:red", "P64_s4": "black", "M3a_s4": "tab:blue", "M3c_s4_nk": "tab:green", "M3d_s4": "tab:purple",
       "S4_16_s4": "tab:blue", "RS32_s4": "tab:green", "S4_128_s4": "tab:red", "S4_128_s8": "tab:brown", "S4_128_s16": "tab:purple",
       "P64_s2": "tab:gray", "P64_s8": "tab:cyan", "S4_R2_m0p2": "tab:orange", "S4_R2_m0p1": "tab:pink"}


def series(Z, nm, key):
    T = Z["%s__T" % nm]; y = Z["%s__%s" % (nm, key)]; c = Z["%s__counts" % nm].astype(bool)
    return T, y, c


def plot_history(ax, Z, D, names, key, scale_rs=False, ylabel=""):
    for nm in names:
        T, y, c = series(Z, nm, key)
        rs = D["members"][nm]["meta"]["rs"]
        yy = y / rs if scale_rs else y
        cert = Z["%s__certified" % nm].astype(bool)
        ok = c & cert
        late = c & ~cert
        ax.plot(np.where(ok, T, np.nan), np.where(ok, yy, np.nan), "-", color=COL.get(nm, "black"), lw=1.2, label=nm)
        if np.any(late):
            ax.plot(np.where(late, T, np.nan), np.where(late, yy, np.nan), ":", color=COL.get(nm, "black"), lw=0.9)
        if np.any(~c):
            ax.plot(T[~c], yy[~c], "x", color=COL.get(nm, "black"), ms=3)
    ax.set_xlabel("T/r_s"); ax.set_ylabel(ylabel)


def form_points(ax, D, names, xf, key):
    for nm in names:
        M = D["members"][nm]; fo = M["formation"]
        x = xf(nm)
        if fo is None:
            if key == "T_f" and M["convergence"]["last"] >= 10:
                ax.plot([x], [M["convergence"]["last"]], "^", color=COL.get(nm, "black"), mfc="none", ms=8)
                ax.annotate("%s: none to %g" % (nm, M["convergence"]["last"]), (x, M["convergence"]["last"]), fontsize=7,
                            xytext=(-60, -12), textcoords="offset points")
            continue
        y = fo["T"] if key == "T_f" else fo[key]
        ax.plot([x], [y], "o" if fo["certified"] else "s", color=COL.get(nm, "black"), mfc=COL.get(nm, "black") if fo["certified"] else "none", ms=7)
        ax.annotate(nm, (x, y), fontsize=7, xytext=(4, 3), textcoords="offset points")


def mass_fig(Z, D, out):
    names = ["M3b_s4", "P64_s4", "M3a_s4", "M3c_s4_nk", "M3d_s4"]
    fig, axs = plt.subplots(2, 2, figsize=(11, 8.5))
    plot_history(axs[0, 0], Z, D, names, "minNv", ylabel="min(N - |v|)")
    axs[0, 0].axhline(0, color="black", lw=0.5); axs[0, 0].legend(fontsize=7)
    axs[0, 0].set_title("(a) depth: solid certified, dotted after the certified prefix, x non-counting", fontsize=9)
    plot_history(axs[0, 1], Z, D, names, "Rt", scale_rs=True, ylabel="R_t/r_s")
    axs[0, 1].set_title("(b) outer root R_t/r_s", fontsize=9)
    F = D["fits"].get("mass_diag_iso")
    xf = lambda nm: D["members"][nm]["counts"]["x_iso"]
    for ax, key, lab in ((axs[1, 0], "T_f", "T_f/r_s"), (axs[1, 1], "R_t_over_rs", "R_t/r_s at formation")):
        form_points(ax, D, names, xf, key)
        if F:
            xs = np.linspace(0, max(xf(nm) for nm in names) * 1.05, 100)
            c = F[key]["coef"]
            ax.plot(xs, c[0] + c[1] * xs, "--", color="black", lw=1,
                    label="2-coefficient diagnostic, 3 members, dof %d (no law)" % F["dof"])
            ax.legend(fontsize=7)
        ax.set_xlabel("x = (p_F/m)^2, isotropic release form"); ax.set_ylabel(lab)
    axs[1, 0].set_title("(c) formation clock; open triangle: no zero to that record (a lower bound)", fontsize=9)
    axs[1, 1].set_title("(d) formation radius; filled: certified", fontsize=9)
    fig.suptitle("MASS_ladder_v1.3 (sigma/a 4, tau 8): first saved trapping records; open markers not certified", fontsize=10)
    fig.tight_layout()
    fig.savefig(out + ".png", dpi=150); fig.savefig(out + ".pdf")
    plt.close(fig)


def scaling_fig(Z, D, out):
    size = ["S4_16_s4", "RS32_s4", "P64_s4", "S4_128_s4"]
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.5))
    plot_history(axs[0, 0], Z, D, size, "minNv", ylabel="min(N - |v|)")
    axs[0, 0].axhline(0, color="black", lw=0.5); axs[0, 0].legend(fontsize=7)
    axs[0, 0].set_title("(a) depth, size ladder at sigma/a 4", fontsize=9)
    plot_history(axs[0, 1], Z, D, size + ["S4_128_s8", "S4_128_s16", "P64_s2", "P64_s8"], "Rt", scale_rs=True, ylabel="R_t/r_s")
    axs[0, 1].legend(fontsize=7); axs[0, 1].set_title("(b) outer root, size members and width points", fontsize=9)
    xf = lambda nm: D["members"][nm]["meta"]["rs"]
    form_points(axs[0, 2], D, size, xf, "T_f"); axs[0, 2].set_xscale("log"); axs[0, 2].set_xlabel("r_s/a"); axs[0, 2].set_ylabel("T_f/r_s")
    axs[0, 2].set_title("(c) formation clock; open square: uncertified; triangle: none to that record", fontsize=8)
    for ax in (axs[0, 2], axs[1, 0]):
        ax.set_xticks([16, 32, 64, 128]); ax.set_xticklabels(["16", "32", "64", "128"]); ax.minorticks_off()
    axs[0, 0].set_title("(a) depth: solid certified, dotted after the certified prefix, x non-counting", fontsize=8)
    form_points(axs[1, 0], D, size, xf, "R_t_over_rs"); axs[1, 0].set_xscale("log"); axs[1, 0].set_xlabel("r_s/a"); axs[1, 0].set_ylabel("R_t/r_s at formation")
    axs[1, 0].set_title("(d) formation radius against r_s/a", fontsize=9)
    axs[1, 0].set_xticks([16, 32, 64, 128]); axs[1, 0].set_xticklabels(["16", "32", "64", "128"]); axs[1, 0].minorticks_off()
    W = ["P64_s2", "P64_s4", "P64_s8", "S4_128_s4", "S4_128_s8", "S4_128_s16"]
    F = D["fits"].get("width_cross_certified")
    sx = lambda nm: D["members"][nm]["meta"]["sigma"] / D["members"][nm]["meta"]["rs"]
    for ax, key, lab in ((axs[1, 1], "T_f", "T_f/r_s"), (axs[1, 2], "R_t_over_rs", "R_t/r_s at formation")):
        form_points(ax, D, W, sx, key)
        if F:
            c = F[key]["coef"]
            for rs, colr in ((64.0, "black"), (128.0, "tab:red")):
                s = np.linspace(1 / 40.0, 0.14, 50)
                z = math.log(1 / rs); h = np.log(s * rs)
                ax.plot(s, c[0] + c[1] * z + c[2] * h + c[3] * z * h, "--", color=colr, lw=1,
                        label="1, z, h, zh through the certified points at r_s/a %g (dof %d)" % (rs, F["dof"]))
            ax.legend(fontsize=7)
        ax.set_xscale("log"); ax.set_xlabel("sigma/r_s"); ax.set_ylabel(lab)
        ax.set_xticks([1 / 32.0, 1 / 16.0, 1 / 8.0]); ax.set_xticklabels(["1/32", "1/16", "1/8"]); ax.minorticks_off()
    axs[1, 1].set_title("(e) width cross: formation clock", fontsize=9)
    axs[1, 2].set_title("(f) width cross: formation radius", fontsize=9)
    fig.suptitle("SCALING_ladder_v1.3 (m a 0.2): first saved trapping records; open markers not certified", fontsize=10)
    fig.tight_layout()
    fig.savefig(out + ".png", dpi=150); fig.savefig(out + ".pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readout-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    Z = np.load(os.path.join(a.readout_dir, "ladders_readout_v1_3.npz"))
    D = json.load(open(os.path.join(a.readout_dir, "ladders_readout_v1_3.json")))
    mass_fig(Z, D, os.path.join(a.out_dir, "MASS_ladder_v1.3_fig"))
    scaling_fig(Z, D, os.path.join(a.out_dir, "SCALING_ladder_v1.3_fig"))
    print("wrote figures in", a.out_dir)


if __name__ == "__main__":
    main()
