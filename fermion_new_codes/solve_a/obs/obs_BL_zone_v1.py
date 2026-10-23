#!/usr/bin/env python3
"""obs_BL_zone_v1.py (builder analyst; SPEC_final4b_engine.md sec. 13.2's box paragraph and H2 of SPEC_final4b_observables.md sec. 9.3: "the
box's B_L against the zone integral"): P31's zone integral B_L = (1/2) int_BZ d^3p/(2 pi)^3 Q_1^2 c_1^2 (m^2 + d_2^2 + d_3^2)/eps^5 for the free
sine Dirac sea, with the reading (BUILDER, labelled; P31's note gives the formula without definitions): d_i = sin p_i, c_i = cos p_i,
Q_i = sin(2 p_i)/2 (the two-step current's symbol), eps = sqrt(m^2 + sum d_i^2), a = 1; the midpoint rule on an n^3 grid of the zone (the
integrand is periodic; at m = 0 it is integrable at the 8 valleys and the grid offset avoids them), n doubled as the control; printed beside
P31's lattice numbers (B_L = 0.01900638 at a m = 0.25 and 0.00804776 at a m = 1). The variants that differ only by a factor (the spinor's
2, the 1/2) are printed beside, so a mismatch by a simple factor is visible as such. Args: --m (list) --n (list) --out-json --md."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import argparse
import json
import math

import numpy as np

P31 = {0.25: 0.01900638, 1.0: 0.00804776}


def zone(m, n, chunk=64):
    p = (np.arange(n) + 0.5) * (2 * math.pi / n) - math.pi
    s, c = np.sin(p), np.cos(p)
    q = np.sin(2 * p) / 2
    tot = 0.0
    for i0 in range(0, n, chunk):
        s1 = s[i0:i0 + chunk, None, None]; c1 = c[i0:i0 + chunk, None, None]; q1 = q[i0:i0 + chunk, None, None]
        s2 = s[None, :, None]; s3 = s[None, None, :]
        eps2 = m * m + s1 ** 2 + s2 ** 2 + s3 ** 2
        f = q1 ** 2 * c1 ** 2 * (m * m + s2 ** 2 + s3 ** 2) / eps2 ** 2.5
        tot += float(f.sum())
    return 0.5 * tot / n ** 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=float, nargs="+", default=[0.25, 1.0, 0.1, 0.2, 0.0])
    ap.add_argument("--n", type=int, nargs="+", default=[96, 192])
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    rows = []
    for m in a.m:
        vals = {n: zone(m, n) for n in a.n}
        rows.append({"m": m, "B_L": {str(n): v for n, v in vals.items()}, "P31": P31.get(m)})
        print(m, vals, P31.get(m), flush=True)
    L = ["## P31's zone integral B_L on the sine Dirac sea (obs_BL_zone_v1.py; d = sin p, c = cos p, Q = sin 2p/2, a = 1)", "",
         "| a m | B_L per grid n | P31's number | ratio (finest grid)/P31 |", "|---|---|---|---|"]
    for r in rows:
        nmax = str(max(a.n))
        L.append("| %g | %s | %s | %s |" % (r["m"], ", ".join("%s: %.8f" % (n, v) for n, v in r["B_L"].items()), ("%.8f" % r["P31"]) if r["P31"] else "-",
                 ("%.6f" % (r["B_L"][nmax] / r["P31"])) if r["P31"] else "-"))
    L += ["", "- every number above is printed by the script; a ratio of a simple factor (2, 1/2, 4) names a convention difference, not a disagreement."]
    json.dump({"script": "obs_BL_zone_v1.py", "args": vars(a), "rows": rows}, open(a.out_json, "w"), indent=1)
    open(a.md, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
