#!/usr/bin/env python
"""w6_fig_v1_2.py: the figure of W6_FIRSTLAW_rungs_v1.2 from w6_readout_v1_2.npz and .json only (colours black, red, blue; never yellow or
viridis). Version 1's panels with the reading's M1 (lines broken at the root-less gaps, where T_a of a pair is not T_b of the one before) and
R1 (panel (b): the representation defect of busi's v4c producer, the K_1 momentum weight, solid, against v4b's K_0 weight, dotted; engine route).
Usage: python w6_fig_v1_2.py --readout-dir DIR --out FILE_BASE"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COL = {2.0: "black", 4.0: "red", 8.0: "blue"}


def broken(x, y, breaks):
    """insert NaN before each pair index in breaks, so a line is not drawn across a root-less gap"""
    x = np.asarray(x, float); y = np.asarray(y, float)
    for i in sorted(breaks, reverse=True):
        x = np.insert(x, i, np.nan); y = np.insert(y, i, np.nan)
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readout-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Z = np.load(os.path.join(a.readout_dir, "w6_readout_v1_2.npz"))
    D = json.load(open(os.path.join(a.readout_dir, "w6_readout_v1_2.json")))
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    for k in D["order"]:
        r = D["rungs"][k]
        if r["route"] != "engine":
            continue
        c = COL.get(r["sigma"], "tab:gray")
        T = 0.5 * (Z[k + "__T_a"] + Z[k + "__T_b"]); br = list(Z[k + "__gap_breaks"])
        lab = "sigma/a %g" % r["sigma"]
        axs[0, 0].plot(T, Z[k + "__rate"], "o", ms=2, color=c, label="Delta M/dt, " + lab)
        axs[0, 0].plot(*broken(T, Z[k + "__rows_mid"], br), "-", lw=0.8, color=c, label="rows_mid, " + lab)
        axs[1, 0].plot(*broken(T, Z[k + "__clock_residual_mid"], br), "-", lw=0.8, color=c, label=lab)
        axs[1, 1].plot(*broken(T, Z[k + "__gate"], br), "-", lw=0.8, color=c, label=lab)
        axs[0, 1].semilogy(*broken(T, np.abs(Z[k + "__rep_defect_v4b"]) + 1e-300, br), ":", lw=0.8, color=c, label=lab + ", K_0 weight (v4b)")
        axs[0, 1].semilogy(*broken(T, np.abs(Z[k + "__rep_defect"]) + 1e-300, br), "-", lw=0.8, color=c, label=lab + ", K_1 weight (v4c)")
    axs[0, 0].set_title("(a) the records' rate and the rows' rate at the outer root", fontsize=9); axs[0, 0].set_ylabel("rate (energy per lattice time)")
    axs[0, 1].set_title("(b) |Delta M - rep|, momentum weight K_1 (solid, v4c) or K_0 (dotted, v4b)", fontsize=9)
    axs[1, 0].set_title("(c) the full-clock residual at the root (midpoint)", fontsize=9); axs[1, 0].axhline(0, color="black", lw=0.5)
    axs[1, 1].set_title("(d) the gate sigma/R at the outer root", fontsize=9); axs[1, 1].axhline(0.1, color="black", lw=0.5, ls="--")
    for ax in axs.ravel():
        ax.set_xlabel("T/r_s (pair midpoint; lines broken at the root-less gaps)"); ax.legend(fontsize=6)
    fig.suptitle("W6_FIRSTLAW_rungs_v1.2: (0.2, 64) at sigma/a 2, 4, 8 (engine kernel route)", fontsize=10)
    fig.tight_layout(); fig.savefig(a.out + ".png", dpi=150); fig.savefig(a.out + ".pdf"); plt.close(fig)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
