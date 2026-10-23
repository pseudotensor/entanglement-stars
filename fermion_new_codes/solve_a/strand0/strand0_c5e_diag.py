#!/usr/bin/env python
"""Strand 0 builder diagnostic for control C5e (not in the spec's tables): ||H_exc - H_free^exc - V_*|| at
U = 0.001, 0.01, 0.1 on B0 and B3_e0.2 (m = 0.1, 0.2; N = 4, 5, 6), and the ratios of successive values, to check
that the remainder scales as U^2 (ratio 0.01) with the deviation shrinking as U decreases.  Writes <out>/c5e_diag.json."""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import argparse
import json
import sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "frozen"))
import strand0_lib as L  # noqa: E402
import native_sources as ns  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fock = ns.Fock(8)
    x = np.arange(4)
    bgs = {"B0": np.ones(4), "B3_e0.2": 1 + 0.2 * (-1.0) ** x}
    rows = []
    for name, psi in bgs.items():
        for m in (0.1, 0.2):
            law = L.Law(m, psi)
            vals = {}
            for U in (0.001, 0.01, 0.1):
                c = L.Completion(law, fock, U, sectors=range(2, 8), dress=(4, 5, 6))
                vals[U] = {N: float(np.linalg.norm(c.H_exc(N) - c.H_free(N) - c.Vstar[N])) for N in (4, 5, 6)}
            for N in (4, 5, 6):
                rows.append({"bg": name, "m": m, "N": N, "max_U_psi^-6_at_U=0.1": float(0.1 * np.max(psi ** -6)),
                             "R(0.001)": vals[0.001][N], "R(0.01)": vals[0.01][N], "R(0.1)": vals[0.1][N],
                             "ratio_0.01_over_0.1": vals[0.01][N] / vals[0.1][N],
                             "ratio_0.001_over_0.01": vals[0.001][N] / vals[0.01][N]})
                print(rows[-1], flush=True)
    with open(os.path.join(a.out, "c5e_diag.json"), "w") as fh:
        json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
