#!/usr/bin/env python
"""bprime_cardhours_v1.py (busi; the B' card-hours of PAGE_Bprime_design_v1.md sec. 6, emitted by script, never retyped).
Inputs: run_bprime_cost_v2.json (per size: amplitudes per column, Chebyshev order per step dt, CPU apply per column) and the measured window
counts at s = 16 (run_bprime_cost_v1 log: F1 supply 60, (0, 3 T0) 92, exit 60; passed as arguments). Assumptions (printed with the table):
  * H100 cost per apply = --ns-per-amp ns x amplitudes per column (default 1.0: GAP10 gap10_core_v2, the same 13-point four-component stencil
    in unfused torch, measured 177 ms per application at 4 x 4 x 225^3 amplitudes on the H100; GPU_opt_judgment_v1 sec. 1.1); a fused port is
    --fused-speedup times faster (6 to 12 in that note; not built or verified for pt3b_cube.Cube);
  * window counts scale by --count-ratio per doubling of s (4: the area law of the fixed omega/T0 window; the supply-limited branch is lower);
  * clocks: reviewer B T/L = 1e2 and 1e3; reviewer A T >= Delta t0(chi) ~ 1e3 L at 16, 2e3 L at 32 (scaled as 1/chi: 4e3 L at 64)."""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost2", required=True)
    ap.add_argument("--n-supply16", type=int, default=60)
    ap.add_argument("--n-exit16", type=int, default=60)
    ap.add_argument("--n-supply3T16", type=int, default=92)
    ap.add_argument("--ns-per-amp", type=float, default=1.0)
    ap.add_argument("--fused-speedup", type=float, default=8.0)
    ap.add_argument("--count-ratio", type=float, default=4.0)
    ap.add_argument("--bright-columns", type=int, default=32)
    a = ap.parse_args()
    C = json.load(open(a.cost2))
    print("| s | amplitudes/column | GB/column | applies per a | columns (supply + exit) | clock T/L | H100 h unfused | H100 h fused (/%g) | CPU h (box, 6 threads) |" % a.fused_speedup)
    print("|---|---|---|---|---|---|---|---|---|")
    rows = []
    for s_, row in sorted(C["sizes"].items(), key=lambda kv: float(kv[0])):
        s = float(s_)
        k = (s / 16.0)
        import math
        f = a.count_ratio ** math.log2(k)
        cols = int(round((a.n_supply16 + a.n_exit16) * f))
        amps = row["amplitudes"]; apa = row["cheb_order_per_step"] / C["dt"]
        reviewa_T = 1e3 * k
        for TL, lab in ((1e2, "1e2 (reviewer B)"), (1e3, "1e3 (reviewer B)"), (reviewa_T, "%g (reviewer A Delta t0)" % reviewa_T)):
            for ncol, clab in ((cols, "%d" % cols), (a.bright_columns, "%d (bright subset)" % a.bright_columns)):
                if ncol == a.bright_columns and TL == 1e2:
                    continue
                sec = ncol * TL * s * apa * amps * a.ns_per_amp * 1e-9
                cpu = ncol * TL * s * row["cpu_seconds_per_column_per_a"]
                rows.append((s, TL, ncol, sec / 3600))
                print("| %d | %.3g | %.3f | %.2f | %s | %s | %.3g | %.3g | %.3g |" % (s, amps, row["bytes_per_column"] / 1e9, apa, clab, lab, sec / 3600,
                                                                                 sec / 3600 / a.fused_speedup, cpu / 3600))
    print("\nassumptions: %g ns per amplitude per apply on the H100 (unfused), fused speedup %g, window count ratio %g per doubling, measured at s 16 F1: "
          "supply %d, exit %d, (0, 3 T0) %d" % (a.ns_per_amp, a.fused_speedup, a.count_ratio, a.n_supply16, a.n_exit16, a.n_supply3T16))
    print("scaling of the full arm at fixed T/chi: columns s^2 x clock (a) s^2 x box s^3 = s^7 (reviewer B's fixed T/L: s^6)")


if __name__ == "__main__":
    main()
