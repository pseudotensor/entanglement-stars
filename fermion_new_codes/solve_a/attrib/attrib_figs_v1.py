#!/usr/bin/env python
"""attrib_figs_v1.py: the figures of W5_W4_formation_attribution_v1 and W3_completions_v1, from the producers' npz and json only.
Colours black, red, blue and tab10; never yellow or viridis. Usage: python attrib_figs_v1.py --attrib-dir DIR --w3-dir DIR --out-dir DIR"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

C = {"P64_s4": "black", "P64_s2": "tab:gray", "P64_s8": "tab:cyan", "W5_k10_tau8": "tab:blue", "W5_k14_tau8": "tab:green", "W5_k20_tau8": "tab:red",
     "W5_k10_tau4": "tab:blue", "W5_k14_tau4": "tab:green", "W5_k20_tau4": "tab:red", "W4b_pairsoff": "tab:purple", "W2_dg": "tab:brown",
     "P64_s4_tau4": "tab:pink", "P64_s4_tau2": "tab:olive"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attrib-dir", required=True)
    ap.add_argument("--w3-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    Z = np.load(os.path.join(a.attrib_dir, "attrib_readout_v1.npz"))
    D = json.load(open(os.path.join(a.attrib_dir, "attrib_readout_v1.json")))
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2))
    ax = axs[0]
    names = D["order"]
    for i, nm in enumerate(names):
        fo = D["histories"][nm]["formation"]
        if fo:
            ax.plot([fo["T"]], [i], "o", color=C.get(nm, "black"), mfc=C.get(nm, "black") if fo["certified"] else "none")
    for lab, ls in (("s2_density_continuum_rest_lapse-on_adaptive", ":"), ("s4_density_continuum_rest_lapse-on_adaptive", "--"),
                    ("s8_density_continuum_rest_lapse-on_adaptive", "-.")):
        w = D["w4a"].get(lab, {})
        if w.get("fold_T") is not None:
            ax.axvline(w["fold_T"], color="black", ls=ls, lw=0.8, label="dust comparator fold, %s" % lab.split("_")[0])
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7); ax.set_xlabel("formation T/r_s (first record holding a zero)")
    ax.legend(fontsize=7); ax.set_title("(a) formation clocks; vertical lines: the dust comparator's fold", fontsize=9)
    ax = axs[1]
    for nm in ("P64_s4", "W5_k10_tau8", "W5_k14_tau8", "W5_k20_tau8", "W5_k20_tau4", "W4b_pairsoff", "W2_dg", "P64_s4_tau4"):
        T = Z["%s__T" % nm]; y = Z["%s__minNv" % nm]; m = (T > 10.5) & (T < 14.01)
        ax.plot(T[m], y[m], "-" if "tau4" not in nm else "--", color=C.get(nm, "black"), lw=1, label=nm)
    k = "w4a_s4_density_continuum_rest_lapse-on_adaptive"
    if "%s__T" % k in Z:
        T = Z["%s__T" % k]; y = Z["%s__depth" % k]; m = (T > 10.5) & (T < 14.01)
        ax.plot(T[m], y[m], "o", color="black", ms=3, mfc="none", label="dust comparator s4 (records)")
    ax.axhline(0, color="black", lw=0.5); ax.set_xlabel("T/r_s"); ax.set_ylabel("min(N - |v|)"); ax.legend(fontsize=7)
    ax.set_title("(b) the depth through formation", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, "W5_W4_formation_attribution_v1_fig.png"), dpi=150)
    fig.savefig(os.path.join(a.out_dir, "W5_W4_formation_attribution_v1_fig.pdf")); plt.close(fig)

    Z = np.load(os.path.join(a.w3_dir, "w3_readout_v1.npz"))
    W = json.load(open(os.path.join(a.w3_dir, "w3_readout_v1.json")))
    cc = {"own": "black", "frozen-sources": "tab:red", "paired-avg": "tab:blue", "point-core": "tab:green"}
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    for nm in W["order"]:
        T = Z["%s__T" % nm]; m = T >= 10.0
        axs[0].plot(T[m], Z["%s__EE0" % nm][m], color=cc[nm], lw=1, label=nm)
        axs[1].plot(T[m], Z["%s__Rt" % nm][m], color=cc[nm], lw=1, label=nm)
        axs[2].plot(T[m], Z["%s__minNv" % nm][m], color=cc[nm], lw=1, label=nm)
    axs[0].set_ylabel("E/E0"); axs[1].set_ylabel("R_t (a)"); axs[2].set_ylabel("min(N - |v|)"); axs[2].axhline(0, color="black", lw=0.5)
    for ax in axs:
        ax.set_xlabel("T/r_s"); ax.legend(fontsize=7)
    fig.suptitle("W3_completions_v1: the record's own history and the completions from 11.125", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, "W3_completions_v1_fig.png"), dpi=150)
    fig.savefig(os.path.join(a.out_dir, "W3_completions_v1_fig.pdf")); plt.close(fig)
    print("figures written")


if __name__ == "__main__":
    main()
