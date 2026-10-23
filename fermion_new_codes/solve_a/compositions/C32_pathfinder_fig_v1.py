#!/usr/bin/env python
"""C32_pathfinder_fig_v1.py (solve, composition 1): figure step. Loads C32_pathfinder_compose_v1.npz only; writes the png and pdf.
Panels: (a) light sourcing sum's cut current I_C at 16 r_s, member and control; (b) d = member - control of its integral from 40 at
16 and 24 r_s with sigma (Richardson) bars; (c) heavy field's I_C at 16 r_s; (d) heavy d(int I_C) at 16 and 24 r_s; (e) d(H_out)
of the light sourcing sum at 16 and 24 r_s; (f) the records' d(E/E0) and the relative d of E_inv_light_sourcing over 39 to 80
(no dense root R_t in either run from 40 on). Bars in (b), (d): Richardson on the difference series; dotted: d of the cut-local closure. Colours black, red, blue."""
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

    def dint(a, F, title):
        for c, col, lab in (("r16", "k", "16 r_s"), ("r24", "b", "24 r_s")):
            T = g(F + "__%s__T" % c) / RS; d = g(F + "__%s__d__cum_int_I_C" % c); s = g(F + "__%s__sig__cum_int_I_C" % c)
            a.plot(T, d, "-", color=col, lw=1.0, label="d int I_C, " + lab)
            sc = g(F + "__%s__sigcorr__cum_int_I_C" % c); mc = np.isfinite(sc)
            a.errorbar(T[mc], d[mc], yerr=sc[mc], fmt="o", ms=2.5, color=col, capsize=3, lw=1.0)
            a.plot(T, g(F + "__%s__d__closure_cutlocal" % c), ":", color=col, lw=1.0, label="d cut-local closure, " + lab)
        a.axhline(0, color="0.6", lw=0.5); a.set_title(title, fontsize=10); a.set_xlabel("T / r_s"); a.set_ylabel("member - control"); a.legend(fontsize=8)

    cur(ax[0, 0], "light_sourcing", "(a) light sourcing sum (kappa 1, 2): cut current at 16 r_s")
    dint(ax[0, 1], "light_sourcing", "(b) light sourcing: d of the integral from 40 (bars: Richardson on d)")
    cur(ax[1, 0], "heavy", "(c) heavy field (10 channels, 16 kappa): cut current at 16 r_s")
    dint(ax[1, 1], "heavy", "(d) heavy: d of the integral from 40 (bars: Richardson on d)")
    a = ax[2, 0]
    for c, col, lab in (("r16", "k", "16 r_s"), ("r24", "b", "24 r_s")):
        T = g("light_sourcing__%s__T" % c) / RS
        a.plot(T, g("light_sourcing__%s__d__H_out" % c), "-", color=col, label="d H_out, " + lab)
        a.plot(T, g("light_sourcing__%s__d__H_in" % c), ":", color=col, label="d H_in, " + lab)
    a.axhline(0, color="0.6", lw=0.5); a.set_title("(e) light sourcing: d H_out and d H_in", fontsize=10); a.set_xlabel("T / r_s"); a.legend(fontsize=8)
    a = ax[2, 1]
    Tr = None
    for k in ("rec__T",):
        if k in Z.files:
            Tr = g(k); break
    if Tr is not None:
        m = (Tr >= 39 - 1e-9)
        a.plot(Tr[m], g("rec__d.E_over_E0")[m], "k-", lw=1.0, label="d E/E0")
        a2 = a.twinx(); a2.plot(Tr[m], g("rec__rel.light.E_inv_light_sourcing")[m], "r-", lw=1.0, label="rel d E_inv_light_sourcing"); a2.set_ylabel("(C - K)/K of E_inv_light_sourcing", color="r")
        a.legend(loc="upper left", fontsize=8); a2.legend(loc="lower left", fontsize=8)
    a.axhline(0, color="0.6", lw=0.5); a.set_title("(f) records (every 1/16 r_s): d E/E0; relative d of the light sourcing E_inv", fontsize=10); a.set_xlabel("T / r_s")
    fig.suptitle("r_s/a 32 pathfinder s4c80 against its control s4offc80, T/r_s 40 to 80 (C_readout_v3b ledger)", fontsize=11)
    fig.savefig(out + ".png", dpi=130); fig.savefig(out + ".pdf")
    print("written", out + ".png", out + ".pdf")


if __name__ == "__main__":
    main()
