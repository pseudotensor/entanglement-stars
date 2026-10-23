#!/usr/bin/env python
"""engine_item5_figure_v1.py (builder analyst; inbox #9 item 4): the item 5 figure drawn from engine_item5_figure_data_v1.py's npz only (no
solver, no record read here). Left: the production's lapse N (solid) and flow |v| (dashed) against r/r_s at the three clock times, the
trapping zeros (N = |v|: R_t and, where it exists, r_in) marked by circles. Right: the first trapping surface's radius R_t/r_s against the
clock at infinity T/r_s for the stepping ladder's members (tau 8, 4, 2) over the reproduced clock range only, each member's 2 g E/r_s as a
dotted line in its colour. Colours: black, red, blue (no yellow, no viridis). Args: --npz --out (path stem; writes .png and .pdf)."""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COL = ["black", "red", "blue"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Z = np.load(a.npz, allow_pickle=True)
    rs = float(Z["rs"])
    t0, t1 = [float(x) for x in Z["t_range"]]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.0))
    for i in range(int(Z["n_profiles"])):
        R = np.asarray(Z["profile_R_%d" % i]) / rs
        N = np.asarray(Z["profile_N_%d" % i]); v = np.abs(np.asarray(Z["profile_v_%d" % i]))
        t = float(Z["profile_T_%d" % i])
        ax1.plot(R, N, "-", color=COL[i], lw=1.4, label=r"$N$, $T/r_s$ = %.3f" % t)
        ax1.plot(R, v, "--", color=COL[i], lw=1.4, label=r"$|v|$, $T/r_s$ = %.3f" % t)
        for key in ("profile_Rt_%d" % i, "profile_rin_%d" % i):
            x = float(Z[key])
            if np.isfinite(x):
                y = float(np.interp(x / rs, R, N))
                ax1.plot([x / rs], [y], "o", mfc="none", mec=COL[i], ms=7)
    ax1.set_xlabel(r"$r/r_s$"); ax1.set_ylabel(r"$N$ and $|v|$")
    ax1.set_xlim(0, float(Z["rmax_over_rs"])); ax1.legend(fontsize=7, frameon=False, ncol=1)
    for j in range(int(Z["n_members"])):
        Tm = np.asarray(Z["member_T_%d" % j]); Rt = np.asarray(Z["member_Rt_%d" % j]) / rs; E2 = np.asarray(Z["member_2gE_%d" % j]) / rs
        lab = str(Z["member_label_%d" % j]); tau = float(Z["member_tau_%d" % j])
        ax2.plot(Tm, Rt, "-", color=COL[j % 3], lw=1.4, label=r"$R_t$, %s ($\tau$ = %g)" % (lab, tau))
        ax2.plot(Tm, E2, ":", color=COL[j % 3], lw=1.2, label=r"$2 g E$, %s" % lab)
    ax2.set_xlim(t0, t1)
    ax2.set_xlabel(r"$T/r_s$ (the clock at infinity)"); ax2.set_ylabel(r"$R_t/r_s$ and $2 g E/r_s$")
    ax2.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(a.out + ".png", dpi=200)
    fig.savefig(a.out + ".pdf")
    print("wrote %s.png and %s.pdf" % (a.out, a.out))


if __name__ == "__main__":
    main()
