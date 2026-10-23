#!/usr/bin/env python
"""C32_pathfinder_tables_v1.py (solve, composition 1): prints the NOTES tables (markdown) from C32_pathfinder_compose_v1.npz only.
For each field group and cut, at the table clocks: member (C), control (K), d = C - K and sigma (the compose script's formulas) for H_in,
H_out, I_C, the integral of I_C from 40; d of the evolution term, work, ref and the closure beside; then the whole-window extrema of d/sigma."""
import sys
import numpy as np

RS = 32.0
CLK = [40, 44, 48, 50, 52, 56, 60, 64, 68, 72, 76, 80]


def f(x, n=4):
    return "-" if x is None or not np.isfinite(x) else ("%.*g" % (n, x))


def main():
    Z = np.load(sys.argv[1])
    groups = sys.argv[2].split(",") if len(sys.argv) > 2 else ["light_sourcing", "light_all", "heavy"]
    for F in groups:
        for c, lab in (("r16", "16 r_s (R = 512 a)"), ("r24", "24 r_s (R = 768 a)"), ("end", "chain end (site 2038)")):
            p = "%s__%s__" % (F, c)
            if p + "T" not in Z.files:
                continue
            T = Z[p + "T"] / RS
            g = lambda q: Z[p + q] if (p + q) in Z.files else np.full(len(T), np.nan)
            print("\n#### %s at %s\n" % (F, lab))
            print("| T/r_s | H_in C | H_in K | d H_in | sig | H_out C | H_out K | d H_out | I_C C | I_C K | d I_C | sig | int I_C C | int I_C K | d int | sig (Richardson) | sigcorr | d closure cut-local | d evol | d work | d ref | d closure |")
            print("|" + "---|" * 22)
            for t in CLK:
                i = np.argmin(abs(T - t))
                if abs(T[i] - t) > 1e-6:
                    continue
                row = [t, g("C__H_in")[i], g("K__H_in")[i], g("d__H_in")[i], g("sig__H_in")[i], g("C__H_out")[i], g("K__H_out")[i], g("d__H_out")[i],
                       g("C__I_C")[i], g("K__I_C")[i], g("d__I_C")[i], g("sig__I_C")[i], g("C__cum_int_I_C")[i], g("K__cum_int_I_C")[i], g("d__cum_int_I_C")[i],
                       g("sig__cum_int_I_C")[i], g("sigcorr__cum_int_I_C")[i], g("d__closure_cutlocal")[i], g("d__cum_evol")[i], g("d__cum_work")[i], g("d__cum_ref")[i], g("d__cum_closure_C")[i]]
                print("| " + " | ".join([str(t)] + [f(x, 7 if k in (0, 1, 4, 5) else 4) for k, x in enumerate(row[1:])]) + " |")
            # extrema over the window of |d|/sigma
            out = []
            for q, sk in (("H_in", "sig__H_in"), ("H_out", "sig__H_out"), ("I_C", "sig__I_C"), ("cum_int_I_C", "sig__cum_int_I_C"), ("cum_int_I_C (sigcorr)", "sigcorr__cum_int_I_C")):
                d, s = g("d__" + q.split()[0]), g(sk)
                m = np.isfinite(s) & (s > 0)
                if m.any():
                    z = np.abs(d[m]) / s[m]; j = np.argmax(z)
                    out.append("%s: max |d|/sig %s at T/r_s %g (d %s, sig %s; %d of %d clocks with |d| > 3 sig)" % (q, f(z[j], 3), T[m][j], f(d[m][j]), f(s[m][j]), int((z > 3).sum()), int(m.sum())))
            d = g("d__cum_evol"); j = np.nanargmax(np.abs(d)); out.append("cum_evol: max |d| %s at T/r_s %g" % (f(d[j]), T[j]))
            print("\nWindow 40 to 80 (every matched slice clock): " + "; ".join(out) + ".")


if __name__ == "__main__":
    main()
