#!/usr/bin/env python
"""Strand S S3 diagnostic (builder analyst, labelled): is the inner form isotropic?  For one normal, the massless ladder on --nrays rays of
the corner cell's half (phi from -pi/4 to 3pi/4), --npts gaps per ray in [--glo, --ghi], each fitted by version 3's inner form (alpha = 2/3
fixed; beta, gamma g^2 log g, delta g^2, eps g^4 log g, zeta g^4); per ray the coefficients and, for the graded sum, the predicted inner
contribution's gamma-weighted factor against phi = 0.  Uses strandS_sphere_v5.py's functions unchanged.  Args: --normal --nrays --npts --glo
--ghi --workers --out-json --out-md."""
import argparse
import json
import math
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import strandS_sphere_v5 as SV  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal", required=True)
    ap.add_argument("--nrays", type=int, default=8)
    ap.add_argument("--npts", type=int, default=16)
    ap.add_argument("--glo", type=float, default=0.015)
    ap.add_argument("--ghi", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    a = ap.parse_args()
    p = tuple(int(v) for v in a.normal.split(","))
    SV.init(p, 24.0, 480)
    phis = [-math.pi / 4 + (i + 0.5) * math.pi / a.nrays for i in range(a.nrays)]
    gl = np.geomspace(a.glo, a.ghi, a.npts)
    pool = Pool(a.workers, initializer=SV.init, initargs=(p, 24.0, 480))
    lad = pool.map(SV.job_ladder, [(phi, g) for phi in phis for g in gl], chunksize=1)
    pool.close()
    rows = []
    for phi in phis:
        rr = [r for r in lad if r[0] == phi and r[2] is not None]
        c, res = SV.fit_inner([r[2] for r in rr], [r[4] for r in rr])
        rows.append({"phi": phi, "coef": c, "max_resid": res, "points": len(rr), "L_max": max(r[3] for r in rr)})
    g0 = rows[min(range(len(rows)), key=lambda i: abs(rows[i]["phi"]))]["coef"][1]
    L = ["# Strand S S3 diagnostic: the inner form by ray (strandS_sphere_ladders_v1.py; normal %s, |p| %.4g; %d rays, %d gaps in [%g, %g])" % (
        p, SV.G["norm"], a.nrays, a.npts, a.glo, a.ghi), "", "| phi | beta | gamma (g^2 log g) | delta (g^2) | eps | zeta | max residual | gamma/gamma(phi nearest 0) |",
         "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        c = r["coef"]
        L.append("| %.4f | %.9f | %.6f | %.6f | %.4g | %.4g | %.2e | %.4f |" % (r["phi"], c[0], c[1], c[2], c[3], c[4], r["max_resid"], c[1] / g0))
    gm = float(np.mean([r["coef"][1] for r in rows]))
    L += ["", "- The ray mean of gamma %.6f against the phi-nearest-0 value %.6f (ratio %.4f); every number above is emitted by the script." % (gm, g0, gm / g0)]
    json.dump({"normal": list(p), "rows": rows, "ladder": [list(x) for x in lad]}, open(a.out_json, "w"), indent=1, default=float)
    open(a.out_md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
