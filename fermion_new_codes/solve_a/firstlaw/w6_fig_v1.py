#!/usr/bin/env python
"""w6_fig_v1.py: the figure of W6_FIRSTLAW_rungs_v1 from w6_readout_v1.npz and .json only (colours black, red, blue; never yellow or viridis).
Panels: (a) the records' rate Delta M/dt and the rows' rate rows_mid at the outer root against T/r_s; (b) the representation defect
|Delta M - rep|; (c) the full-clock residual; (d) the gate sigma/R. Usage: python w6_fig_v1.py --readout-dir DIR --out FILE_BASE"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COL = {2.0: "black", 4.0: "red", 8.0: "blue"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readout-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Z = np.load(os.path.join(a.readout_dir, "w6_readout_v1.npz"))
    D = json.load(open(os.path.join(a.readout_dir, "w6_readout_v1.json")))
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    for k in D["order"]:
        r = D["rungs"][k]; c = COL.get(r["sigma"], "tab:gray"); ls = "-" if r["kernel"] == "engine" else ":"
        T = 0.5 * (Z[k + "__T_a"] + Z[k + "__T_b"])
        lab = "sigma/a %g (%s kernel)" % (r["sigma"], r["kernel"])
        if r["kernel"] == "engine" or not any(D["rungs"][q]["sigma"] == r["sigma"] and D["rungs"][q]["kernel"] == "engine" for q in D["order"]):
            axs[0, 0].plot(T, Z[k + "__rate"], "o", ms=2, color=c, label="Delta M/dt, " + lab)
            axs[0, 0].plot(T, Z[k + "__rows_mid"], "-", lw=0.8, color=c, label="rows_mid, " + lab)
            axs[1, 0].plot(T, Z[k + "__clock_residual_mid"], "-", lw=0.8, color=c, label=lab)
            axs[1, 1].plot(T, Z[k + "__gate"], "-", lw=0.8, color=c, label=lab)
        axs[0, 1].semilogy(T, np.abs(Z[k + "__rep_defect"]) + 1e-300, ls, lw=0.8, color=c, label=lab)
    axs[0, 0].set_title("(a) the records' rate and the rows' rate at the outer root", fontsize=9); axs[0, 0].set_ylabel("rate (energy per lattice time)")
    axs[0, 1].set_title("(b) the representation defect |Delta M - rep| (dotted: analytic kernel)", fontsize=9)
    axs[1, 0].set_title("(c) the full-clock residual at the root (midpoint)", fontsize=9); axs[1, 0].axhline(0, color="black", lw=0.5)
    axs[1, 1].set_title("(d) the gate sigma/R at the outer root", fontsize=9); axs[1, 1].axhline(0.1, color="black", lw=0.5, ls="--")
    for ax in axs.ravel():
        ax.set_xlabel("T/r_s (pair midpoint)"); ax.legend(fontsize=6)
    fig.suptitle("W6_FIRSTLAW_rungs_v1: (0.2, 64) at sigma/a 2, 4, 8", fontsize=10)
    fig.tight_layout(); fig.savefig(a.out + ".png", dpi=150); fig.savefig(a.out + ".pdf"); plt.close(fig)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
