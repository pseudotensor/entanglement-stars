#!/usr/bin/env python
"""Strand 6 (c1) (spec solve/specs/SPEC_strand6_validation.md (c1); builder analyst): P14's 4^3 unit test.

Periodic 4^3 cube (the 2d box B4, strand2d_lib_v1.lattice), a = 1, P = 1; the four packets of strand2a_lib.masses(m, H) at (m, H) = (1, 1),
i.e. squared masses (1, 2, 2, 3), orientations (+, -, -, +); f = cos(pi x_1/2).  The mixed Hessian of W_0 (strand2a_lib.W0) in (N, psi)
at the flat point along f, J_Npsi = d^2 W_0/(dlam_N dlam_psi) by the 3 x 3 stencil at h in {1e-3, 1e-4} and Richardson, divided by
sum_x f^2 (= 32), against P14's -0.1376054142 = -(3/2)(sqrt2 + 2 sqrt3 - sqrt5 + sqrt6 - 5) and the lead's telescoped form
-(3/2) sum_s eta_s [1/(e_{s,0} + e_{s,1}) + 2/(e_{s,1} + e_{s,2}) + 1/(e_{s,2} + e_{s,3})], e_{s,r} = sqrt(m_s^2 + r) (both evaluated here);
also NN and psi psi beside it; and the same at 2d's (m, H) = (0.1, 1) for the record.  Writes <out>/unittest.json."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(HERE, "..", "strand2", "frozen"), os.path.join(HERE, "..", "strand2"), os.path.join(HERE, "frozen"), HERE):
    sys.path.insert(0, _p)
import strand2a_lib as S2  # noqa: E402
import strand2d_lib_v1 as L2D  # noqa: E402

STENCIL = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    lat = L2D.lattice("B4")
    n = lat["nsite"]
    one = np.ones(n)
    f = np.cos(np.pi * lat["coord"][:, 0].astype(float) / 2)
    nf2 = float(f @ f)
    p14 = -1.5 * (np.sqrt(2) + 2 * np.sqrt(3) - np.sqrt(5) + np.sqrt(6) - 5)
    res = {"meta": {"sum_f2": nf2, "P14_closed_form": float(p14)}, "rows": []}
    for (m, H) in ((1.0, 1.0), (0.1, 1.0)):
        ms, etas = S2.masses(m, H)
        e = lambda s, r: np.sqrt(ms[s] ** 2 + r)
        tele = -1.5 * sum(etas[s] * (1 / (e(s, 0) + e(s, 1)) + 2 / (e(s, 1) + e(s, 2)) + 1 / (e(s, 2) + e(s, 3))) for s in range(4))
        jets = {}
        for h in (1e-3, 1e-4):
            vals = {}
            for ia, ib in STENCIL:
                vals[(ia, ib)] = S2.W0(lat, m, H, one + ib * h * f, one + ia * h * f)[0]
            jets[h] = {"NN": (vals[(1, 0)] - 2 * vals[(0, 0)] + vals[(-1, 0)]) / h ** 2,
                       "psipsi": (vals[(0, 1)] - 2 * vals[(0, 0)] + vals[(0, -1)]) / h ** 2,
                       "Npsi": (vals[(1, 1)] - vals[(1, -1)] - vals[(-1, 1)] + vals[(-1, -1)]) / (4 * h * h), "W_flat": vals[(0, 0)]}
        R = {k: (100 * jets[1e-4][k] - jets[1e-3][k]) / 99 for k in ("NN", "psipsi", "Npsi")}
        res["rows"].append({"m": m, "H": H, "squared_masses": [x * x for x in ms], "etas": etas, "jets": {str(h): v for h, v in jets.items()},
                            "richardson": R, "Npsi_per_sum_f2": R["Npsi"] / nf2, "telescoped_form": float(tele),
                            "minus_P14": R["Npsi"] / nf2 - p14 if (m, H) == (1.0, 1.0) else None,
                            "minus_telescoped": R["Npsi"] / nf2 - tele, "step_agreement_Npsi": abs(jets[1e-3]["Npsi"] - jets[1e-4]["Npsi"]) / abs(jets[1e-4]["Npsi"])})
        print(json.dumps(res["rows"][-1]["richardson"]), res["rows"][-1]["Npsi_per_sum_f2"], tele, flush=True)
    res["meta"]["seconds"] = time.time() - t0
    with open(os.path.join(a.out, "unittest.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
