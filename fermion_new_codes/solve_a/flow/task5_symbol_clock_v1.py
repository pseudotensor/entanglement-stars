#!/usr/bin/env python3
"""Task 5 E4, the symbol's clock alone (builder analyst; a diagnostic for the lead's ruling on E4, labelled): flow_chart_v6.py's
symbol_series (the point emissions on the dust law from R_0 = 4 r_s, the outgoing stationary root, the arrival at R_obs = 8 r_s) at r_s = 128,
1024, 8192 (R_m = r_s/4) for k0 a in {pi/32, pi/16, pi/8}, massless, with the angular term of j = 1/2 (kappa_j = -1) and without it
(kappa_j = 0, the planar symbol); per series the local slopes d ln(1+z)/d(kappa t_arr) and d ln(dt_arr/dt_e)/d(kappa t_arr) and the largest
1 + z with an arrival.  Args: --out-json --out-md."""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import flow_chart_v6 as V6  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    a = ap.parse_args()
    out = {"label": V6.LABEL + "; the symbol only", "series": []}
    L = ["# Task 5 E4: the symbol's point-emission clock (task5_symbol_clock_v1.py over flow_chart_v6.symbol_series)", "",
         "| r_s | kappa_j | k0 a | saturation | largest 1+z with an arrival | slope d ln(1+z)/d(kappa t) at 1+z = 3, 10, 30, 100 (nearest) | "
         "slope d ln(dt_arr/dt_e)/d(kappa t) at the same points |", "|---|---|---|---|---|---|---|"]
    for rs in (128.0, 1024.0, 8192.0):
        for kj in (-1.0, 0.0):
            for k0 in (math.pi / 32, math.pi / 16, math.pi / 8):
                s = V6.symbol_series(k0, 0.0, kj, rs, rs / 4, 8 * rs, 4 * rs, 1 / (2 * rs), n=121)
                rows = [q for q in s["rows"] if q.get("t_arr") is not None]
                big = max((q["one_plus_z_point"] for q in rows), default=None)
                pick = []
                for z in (3.0, 10.0, 30.0, 100.0):
                    cand = [q for q in rows if q.get("local_slope_ln1pz") is not None]
                    if not cand or big is None or z > big:
                        pick.append(None)
                        continue
                    pick.append(min(cand, key=lambda q: abs(math.log(q["one_plus_z_point"] / z))))
                out["series"].append({"rs": rs, "kappa_j": kj, "k0": k0, "saturation": s["saturation"], "largest_1pz_arrival": big, "rows": s["rows"]})
                L.append("| %g | %g | %.5g | %.4g | %s | %s | %s |" % (
                    rs, kj, k0, s["saturation"], "%.4g" % big if big else "-",
                    ", ".join("%.3g (%.3g)" % (q["local_slope_ln1pz"], q["one_plus_z_point"]) if q else "-" for q in pick),
                    ", ".join("%.3g" % q["local_slope_ln_dtarr_dte"] if q and q.get("local_slope_ln_dtarr_dte") is not None else "-" for q in pick)))
                print(L[-1], flush=True)
    L += ["", "- Every number above is emitted by the script; the slope's argument in parentheses is the 1 + z of the row used."]
    json.dump(out, open(a.out_json, "w"), indent=1, default=float)
    open(a.out_md, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
