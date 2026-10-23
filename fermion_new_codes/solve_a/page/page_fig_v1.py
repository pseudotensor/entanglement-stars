#!/usr/bin/env python
"""page_fig_v1.py (lead reviewer A, GAP03 integration): the figure from <tag>_curve.npz written by page_recursion_v1.py (data and figure
scripts separate; the figure never re-runs the solver). Left: S_esc(t_infinity) total (black), the direct route where available (blue
dashed), the stage transitions as ticks, the levels eta A/a^2 and A/(4G) at the initial charge (grey lines with labels; the comparison the
owner asks for), the continuum stationary-channel comparator S_dot_cont t (thin red, labelled as a comparator, not an input). Right: the
collected flux L(t) (black) and the collar current (blue) against the continuum thermal luminosity of the current charge (thin red).
Usage: python page_fig_v1.py <tag>_curve.npz out.pdf [--log-left]"""
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz"); ap.add_argument("out")
    ap.add_argument("--log-left", action="store_true")
    a = ap.parse_args()
    d = np.load(a.npz, allow_pickle=True)
    rs1 = float(d["rs1"]); g = float(d["g"]); eta = float(d["eta"])
    t = d["t"] / rs1; tinf = d["t_inf"] / rs1
    S = d["S_esc_total"]; Sd = d["S_direct_total"]; stage = d["stage"]
    L = d["L_collected"]; Lc = d["L_collar"]; Lcont = d["L_cont"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(tinf, S, color="black", lw=1.6, label=r"$S_{\rm esc}(t_\infty)$ (memory route, $\mathrm{tr}\,s(M)$)")
    ok = np.isfinite(Sd)
    if ok.any():
        ax1.plot(tinf[ok], Sd[ok], color="blue", lw=1.0, ls="--", label="direct route (stage covariance)")
    TH = float(d["TH"])
    Sdot = 1.6391 * 0.0052373 / rs1 ** 2 / TH
    ax1.plot(tinf, Sdot * d["t"], color="red", lw=0.8, ls=":", label=r"comparator $b\,L_{\rm cont}\,t/T_H$ (stationary channel; not an input)")
    for k in np.unique(stage)[1:]:
        i = int(np.argmax(stage == k))
        ax1.axvline(tinf[i], color="grey", lw=0.5, alpha=0.6)
    for lev, lab in ((float(d["eta_A_over_a2_initial"]), r"$\eta A/a^2$, $\eta=%.4f$" % eta), (float(d["A_over_4G_initial"]), r"$A/(4G)$, $G/a^2=%g$" % g)):
        if a.log_left or lev < 3 * max(S.max(), 1e-9):
            ax1.axhline(lev, color="grey", lw=0.8, ls="-.")
            ax1.text(tinf[-1], lev, lab, ha="right", va="bottom", fontsize=8, color="grey")
    if a.log_left:
        ax1.set_yscale("log")
    ax1.set_xlabel(r"$t_\infty / r_{s,1}$ (retarded clock at infinity; 0 at the end of the formation)")
    ax1.set_ylabel("entropy of the escaping sector [nats]")
    ax1.set_title(r"$r_s/a=%g$, $G/a^2=%g$, $m/T_H=0$ (collapse: 160); exact exterior sequence" % (rs1, g), fontsize=9)
    ax1.legend(fontsize=7, loc="upper left")
    ax2.plot(t, L, color="black", lw=1.2, label="collected flux (Killing energy per unit time)")
    ax2.plot(t, Lc, color="blue", lw=0.8, label=r"collar current $\mathrm{tr}[\Delta J_{\rm out}]$")
    ax2.plot(t, Lcont, color="red", lw=0.8, ls=":", label=r"continuum thermal $0.0052373/r_s^2$ (comparator)")
    ax2.set_yscale("symlog", linthresh=max(1e-12, 1e-3 * np.nanmax(np.abs(L)) if np.isfinite(L).any() else 1e-12))
    ax2.set_xlabel(r"$t / r_{s,1}$")
    ax2.set_ylabel("flux [lattice units]")
    ax2.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(a.out, dpi=200)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
