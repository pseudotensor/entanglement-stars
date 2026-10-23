#!/usr/bin/env python
"""cvread_onset_v1.py (busi lane, inbox #424, CVREAD read (i) follow-up): where the X - P centre difference starts. Per record from T 3.0 to
4.0 (and at the asked clocks): N_nodes(0) on both sides and X - P; each side's converged flag and kept residual and where its own
|N - N_applied| peaks; and the X - P source differences (eps_tot, J_tot, eps_pairs, J_pairs) by radius band. Read-only.
  python cvread_onset_v1.py X_DIR/X_TAG P_DIR/P_TAG --rs 128 --kits KITS_CODE_DIR"""
import argparse
import sys

import numpy as np

BANDS = ((0, 16), (16, 64), (64, 128), (128, 1024), (1024, 2048))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("X"); ap.add_argument("P"); ap.add_argument("--rs", type=float, required=True); ap.add_argument("--kits", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.kits)
    import gate_common_v1 as GC
    rs = a.rs
    JX, FX = GC.load_run(*GC.split_spec(a.X)); JP, FP = GC.load_run(*GC.split_spec(a.P))
    nX = int(JX["args"]["nsites"]); R = np.asarray(FX["R"], float)
    RX, RP = GC.recs_by_T(JX, rs), GC.recs_by_T(JP, rs); IX, IP = GC.rows_by_T(FX, rs), GC.rows_by_T(FP, rs)
    print("-- T | N(0) X, P, X-P | X: converged, kept residual, own |N-N_applied| max at r | P: the same")
    for t in sorted(set(IX) & set(IP)):
        if not (3.0 <= t <= 4.0 or t in (1.375, 1.4375, 8.0, 11.5)):
            continue
        ix, ip = IX[t], IP[t]
        nx = float(np.asarray(FX["N_nodes"][ix], float)[0]); npp = float(np.asarray(FP["N_nodes"][ip], float)[0])
        own = []
        for F, i, J in ((FX, ix, RX[t]), (FP, ip, RP[t])):
            d = np.abs(np.asarray(F["N"][i], float) - np.asarray(F["N_applied"][i], float))
            own.append("%s %.2e |N-Na| %.2e at %.0f a" % (J.get("converged"), (J.get("solve_v32") or {}).get("kept_residual") or float("nan"), d.max(), float(F["R"][int(d.argmax())])))
        print(" %.4f | %.6f %.6f %+.2e | X %s | P %s" % (t, nx, npp, nx - npp, own[0], own[1]))
    print("-- X - P sources by radius band (a): max |d| in %s" % ", ".join("[%d,%d]" % b for b in BANDS))
    for t in (1.375, 3.25, 3.3125, 3.375, 3.4375, 3.5, 8.0, 11.5):
        ix, ip = IX[t], IP[t]
        for k in ("eps_pairs", "J_pairs", "eps_tot", "J_tot"):
            d = np.abs(np.asarray(FX[k][ix], float) - np.asarray(FP[k][ip], float)[:nX])
            print(" %.4f %-9s %s" % (t, k, "  ".join("%.2e" % d[(R > lo) & (R <= hi)].max() if lo else "%.2e" % d[R <= hi].max() for lo, hi in BANDS)))


if __name__ == "__main__":
    main()
