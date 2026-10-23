#!/usr/bin/env python
"""obs_J6_table_v2.py (builder analyst; engine spec 22.1, BUILDER: "the quadrant quadrature at nk 48, 96, 192 on the stored 48-mass table at
hold 0 and flat, the Richardson sequence, the excess per nk"): J6's stored per-channel entropies (obs_J6_gpu_v1's 48 grid masses) splined in
log m_c exactly as the run did (CubicSpline), the massless sea's S_A per area assembled by flow_chart_v4's quadrant quadrature with the
factor 2 at nk = 12, 24 (the run's two, reproduced), 48, 96 and 192; the quadrature masses below the grid's smallest mass (at nk 48 and up)
evaluated by the linear continuation in log m_c from the spline's end slope (the 1D log law of the light channels; their weight share
printed); per nk the flat and the hold-0 values, the excess, the differences from the record's 0.786658 and from the converged flat
0.7865834, and Richardson in nk^-2 on successive pairs. Args: --json --md."""
import argparse
import json
import math
import os
import sys

import numpy as np
from scipy.interpolate import CubicSpline

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "flow", "code"))
sys.path.insert(0, os.path.join(HERE, "..", "flow"))
import flow_chart_v4 as FC  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--md", required=True)
    ap.add_argument("--nk", default="12,24,48,96,192")
    a = ap.parse_args()
    J = json.load(open(a.json))
    lg = np.log([r["mc"] for r in J["grid"]])
    cols = {"flat": [r["flat"][0] for r in J["grid"]], "0.0": [r["formed"]["0.0"][0] for r in J["grid"]]}
    spl = {k: CubicSpline(lg, v) for k, v in cols.items()}
    lo, hi = lg[0], lg[-1]

    def S(k, m):
        x = math.log(m)
        if x < lo:
            return float(spl[k](lo)) + float(spl[k](lo, 1)) * (x - lo)
        if x > hi:
            return float(spl[k](hi)) + float(spl[k](hi, 1)) * (x - hi)
        return float(spl[k](x))
    L = ["## J6's quadrant quadrature refined (obs_J6_table_v2.py; the stored 48-mass table of %s)" % os.path.basename(a.json), "",
         "| nk | flat | formed, hold 0 | excess (formed/flat - 1) | flat - 0.7865834 | formed - 0.786658 | weight below the grid | Richardson flat | Richardson formed |",
         "|---|---|---|---|---|---|---|---|---|"]
    prev = None
    for nk in [int(x) for x in a.nk.split(",")]:
        q = [(float(np.sqrt(np.sin(ky) ** 2 + np.sin(kz) ** 2)), w) for ky, kz, w in FC.quadrant_quadrature(nk)]
        Sf = sum(2 * w * S("flat", m) for m, w in q); S0 = sum(2 * w * S("0.0", m) for m, w in q)
        wb = sum(w for m, w in q if math.log(m) < lo) / sum(w for m, w in q)
        rf = r0 = None
        if prev is not None:
            r_ = (nk / prev[0]) ** 2
            rf = (r_ * Sf - prev[1]) / (r_ - 1.0); r0 = (r_ * S0 - prev[2]) / (r_ - 1.0)
        L.append("| %d | %.8f | %.8f | %+.3e | %+.3e | %+.3e | %.2e | %s | %s |" % (nk, Sf, S0, S0 / Sf - 1.0, Sf - 0.7865834, S0 - 0.786658, wb,
                                                                          ("%.8f" % rf) if rf is not None else "-", ("%.8f" % r0) if r0 is not None else "-"))
        prev = (nk, Sf, S0)
    L += ["", "- the Richardson columns combine each nk with the previous one assuming an nk^-2 error; the masses below the grid's smallest (%.4f) use the linear continuation in log m_c (their weight share in the column beside)." % math.exp(lo),
          "- every number above is printed by the script from the run's json."]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
