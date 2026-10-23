#!/usr/bin/env python
"""Strand 2b sec. 15 item 3, the cube part (builder analyst): the FD Jacobian of F (strand2b_c8_lib_v5.jacobian_fd) at the two cube
enclosure cases ((2,0), m 0.1, (0.2, 1, 1), k = 1 and 1e4) at steps h = 1e-6 and 1e-7: max |J(1e-6) - J(1e-7)| relative to max |J(1e-6)|,
per block (the C rows and the adjoint rows, psi and N columns).  Writes <out>/jac2step.json."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "frozen"))
import strand2a_lib as S2  # noqa: E402
import strand2b_c8_lib_v5 as C  # noqa: E402
import native_core as nc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    lat = S2.lattice("C8")
    K, _ = nc.K_matrix(lat["positions"].tolist())
    res = []
    for kk in (1, 10000):
        rec = json.load(open(os.path.join(a.records, "case_C8_b20_m0.1_gM0.2_gB1_L1_k%d.json" % kk)))
        bm = C.BlockModel(lat, rec["m"], rec["gM"], rec["gB"], float(rec["Lambda"]), (2, 0))
        sys_ = C.SystemC8(bm, K, rec["k"])
        sys_.d0_fixed = rec["d0"]
        z = np.concatenate([rec["psi"], rec["N"]])
        n = sys_.n
        J6 = C.jacobian_fd(sys_, z, h=1e-6)
        J7 = C.jacobian_fd(sys_, z, h=1e-7)
        blocks = {}
        for rn, rs in (("C_rows", slice(0, n)), ("adjoint_rows", slice(n, 2 * n))):
            for cn, cs in (("psi_cols", slice(0, n)), ("N_cols", slice(n, 2 * n))):
                d = np.abs(J6[rs, cs] - J7[rs, cs])
                blocks["%s_%s" % (rn, cn)] = {"max_abs_diff": float(d.max()), "max_abs_J": float(np.abs(J6[rs, cs]).max())}
        res.append({"k": rec["k"], "rel": float(np.abs(J6 - J7).max() / np.abs(J6).max()), "blocks": blocks})
        print(json.dumps(res[-1]), flush=True)
    with open(os.path.join(a.out, "jac2step.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
