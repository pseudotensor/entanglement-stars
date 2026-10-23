#!/usr/bin/env python
"""C32_pathfinder_fig_v1.1.py (solve, composition 1, v1.1): figure step after the reviewer A correctness reading
(ReviewA_reading_COMP1_C32_v1.md, section 4 targets 4, 5, 7). Loads C32_pathfinder_compose_v1.npz only (unchanged); writes png and pdf.
Changes from v1: panel (b) no longer plots the difference of the integrated light collar flux (quadrature noise at the 3.6e-5 level,
wrong sign against the traces; it may not print as a flux difference); it plots the trace route instead: the relative difference
(percent) of the light sourcing operator trace beyond 16 and 24 r_s after the pulse's passage (from 52 at 16 r_s, 62 at 24 r_s), the inside trace at
16 r_s before the arrival, and the records' relative difference of E_inv_light_sourcing from T = 0. Panel (d): bars relabelled as Richardson estimates
on d (not bounds), the cut-local closure difference shown as the scale, and the trace route -(d H_out(T) - d H_out(40)) beside the
integral. Colours black, red, blue."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RS = 32.0


def main():
    p, out = sys.argv[1], sys.argv[2]
    Z = np.load(p)
    g = lambda k: Z[k]
    fig, ax = plt.subplots(3, 2, figsize=(11, 11), constrained_layout=True)

    def cur(a, F, title):
        T = g(F + "__r16__T") / RS
        a.plot(T, g(F + "__r16__C__I_C"), "k-", lw=1.2, label="member (coupling on)")
        a.plot(T, g(F + "__r16__K__I_C"), "r--", lw=1.2, label="control (coupling off)")
        a.axhline(0, color="0.6", lw=0.5); a.set_title(title, fontsize=10); a.set_xlabel("T / r_s"); a.set_ylabel("I_C (inward +)"); a.legend(fontsize=8)

    cur(ax[0, 0], "light_sourcing", "(a) light sourcing sum (kappa 1, 2): cut current at 16 r_s")

    # (b) trace route, relative differences in percent
    a = ax[0, 1]
    for c, col, lab, t0 in (("r16", "k", "16 r_s", 52.0), ("r24", "b", "24 r_s", 62.0)):
        T = g("light_sourcing__%s__T" % c) / RS
        K = g("light_sourcing__%s__K__H_out" % c); d = g("light_sourcing__%s__d__H_out" % c)
        m = (K > 1e-3) & (T >= t0 - 1e-9)
        a.plot(T[m], 100 * d[m] / K[m], "o-", ms=2.5, color=col, lw=1.0, label="trace beyond " + lab + " (from %g)" % t0)
    T = g("light_sourcing__r16__T") / RS
    K = g("light_sourcing__r16__K__H_in"); d = g("light_sourcing__r16__d__H_in")
    m = (K > 1e-3) & (T <= 46 + 1e-9)
    a.plot(T[m], 100 * d[m] / K[m], "s", ms=3, color="k", mfc="none", label="trace inside 16 r_s (40 to 46)")
    if "rec__T" in Z.files:
        Tr = g("rec__T")
        a.plot(Tr, 100 * g("rec__rel.light.E_inv_light_sourcing"), "r-", lw=0.8, label="records: E_inv_light_sourcing")
    a.axhline(0, color="0.6", lw=0.5)
    a.set_title("(b) light sourcing: relative d (percent), operator traces and records", fontsize=10)
    a.set_xlabel("T / r_s"); a.set_ylabel("(member - control) / control, percent"); a.legend(fontsize=8)

    cur(ax[1, 0], "heavy", "(c) heavy field (10 channels, 16 kappa): cut current at 16 r_s")

    # (d) heavy: integral and trace route
    a = ax[1, 1]
    for c, col, lab in (("r16", "k", "16 r_s"), ("r24", "b", "24 r_s")):
        T = g("heavy__%s__T" % c) / RS; d = g("heavy__%s__d__cum_int_I_C" % c)
        a.plot(T, d, "-", color=col, lw=1.0, label="d int I_C, " + lab)
        sc = g("heavy__%s__sigcorr__cum_int_I_C" % c); mc = np.isfinite(sc)
        a.errorbar(T[mc], d[mc], yerr=sc[mc], fmt="o", ms=2.5, color=col, capsize=3, lw=1.0)
        dh = g("heavy__%s__d__H_out" % c)
        a.plot(T, -(dh - dh[0]), "s", ms=3, color=col, mfc="none", label="-(d H_out - d H_out(40)), " + lab)
        a.plot(T, g("heavy__%s__d__closure_cutlocal" % c), ":", color=col, lw=1.0, label="d cut-local closure (scale), " + lab)
    a.axhline(0, color="0.6", lw=0.5)
    a.set_title("(d) heavy: d int I_C from 40 and the trace route\n(bars: Richardson estimate on d, not a bound)", fontsize=9)
    a.set_xlabel("T / r_s"); a.set_ylabel("member - control"); a.legend(fontsize=7)

    a = ax[2, 0]
    for c, col, lab in (("r16", "k", "16 r_s"), ("r24", "b", "24 r_s")):
        T = g("light_sourcing__%s__T" % c) / RS
        a.plot(T, g("light_sourcing__%s__d__H_out" % c), "-", color=col, label="d H_out, " + lab)
        a.plot(T, g("light_sourcing__%s__d__H_in" % c), ":", color=col, label="d H_in, " + lab)
    a.axhline(0, color="0.6", lw=0.5); a.set_title("(e) light sourcing: d of the operator traces H_out and H_in", fontsize=10); a.set_xlabel("T / r_s"); a.legend(fontsize=8)

    a = ax[2, 1]
    if "rec__T" in Z.files:
        Tr = g("rec__T"); m = (Tr >= 39 - 1e-9)
        a.plot(Tr[m], g("rec__d.E_over_E0")[m], "k-", lw=1.0, label="d E/E0")
        a.legend(loc="upper left", fontsize=8)
    a.axhline(0, color="0.6", lw=0.5); a.set_title("(f) records (every 1/16 r_s): d E/E0, 39 to 80", fontsize=10); a.set_xlabel("T / r_s")
    fig.suptitle("r_s/a 32 pathfinder s4c80 against its control s4offc80, T/r_s 40 to 80 (C_readout_v3b ledger), v1.1", fontsize=11)
    fig.savefig(out + ".png", dpi=130); fig.savefig(out + ".pdf")
    print("written", out + ".png", out + ".pdf")


if __name__ == "__main__":
    main()
