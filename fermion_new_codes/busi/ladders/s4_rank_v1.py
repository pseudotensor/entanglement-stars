#!/usr/bin/env python
"""s4_rank_v1.py (busi lane, inbox #301, item S4 of deliveries/lead_gaps/ReviewA_reading_LADDERS_v1.md).

The rank of the design matrix [1, z, x^2, h, zh, j, j^2, zj] on ladder rows (r_s/a, sigma/a, tau/a), printed BEFORE any intercept or fit.
No fit is made. Variables exactly as the reading (reviewer A R7) and the review it cites (final4/REVIEWA_REVIEW_web40j_p1.md sec. 1.0 item 8,
and ReviewA_reading_web40j_C10_v1.md line 82): x = 64 a/r_s, z = x^(4/3), h = (sigma/4a)^2 x^2, j = (tau/8a) x.
For each row set: the singular values (raw columns, and columns scaled to unit norm), the numerical rank at relative tolerances 1e-10
and 1e-6 (singular value > tol * largest), the null vectors below each tolerance and the columns that carry them (|component| > 1e-8
after normalization), and column-pivoted QR's dependent columns. Row set 'review8' is a known-answer control (expected rank 7, null
(z - 1)(2/3 - 7h/24 + j), vector (-2/3, 2/3, 0, 7/24, -7/24, -1, 0, 1)).
"""
import json
import sys

import numpy as np
from scipy.linalg import qr

COLS = ["1", "z", "x^2", "h", "zh", "j", "j^2", "zj"]


def row(rs, sig, tau):
    x = 64.0 / rs
    z = x ** (4.0 / 3.0)
    h = (sig / 4.0) ** 2 * x ** 2
    j = (tau / 8.0) * x
    return [1.0, z, x * x, h, z * h, j, j * j, z * j]


def analyse(name, rows):
    X = np.array([row(*r[1:]) for r in rows])
    out = {"set": name, "rows": [[r[0]] + list(r[1:]) for r in rows], "n_rows": len(rows), "X": X.tolist()}
    for lab, M in (("raw", X), ("unit_columns", X / np.linalg.norm(X, axis=0))):
        U, s, Vt = np.linalg.svd(M)
        d = {"singular_values": s.tolist(), "s_min_over_s_max": float(s[-1] / s[0]) if len(s) else None}
        for tol in (1e-10, 1e-6):
            rank = int(np.sum(s > tol * s[0]))
            # null space: right singular vectors past the rank (including the 8 - n_rows ones when rows < 8)
            nulls = []
            for k in range(rank, Vt.shape[0]):
                v = Vt[k] / np.max(np.abs(Vt[k]))
                v = v * np.sign(v[np.argmax(np.abs(v))])
                nulls.append({"vector": [float(c) for c in v], "columns": [COLS[c] for c in range(8) if abs(v[c]) > 1e-8],
                              "sigma": float(s[k]) if k < len(s) else 0.0})
            d["rank_tol_%g" % tol] = rank
            d["null_tol_%g" % tol] = nulls
        Q, Rr, P = qr(M, pivoting=True)
        diag = np.abs(np.diag(Rr))
        d["qr_pivot_order"] = [COLS[c] for c in P]
        d["qr_abs_diag"] = diag.tolist()
        for tol in (1e-10, 1e-6):
            rk = int(np.sum(diag > tol * diag[0]))
            d["qr_dependent_cols_tol_%g" % tol] = [COLS[c] for c in P[rk:]]
        out[lab] = d
    if name == "review8":
        n = np.array([-2.0 / 3, 2.0 / 3, 0, 7.0 / 24, -7.0 / 24, -1, 0, 1])
        out["control_X_dot_null"] = (X @ n).tolist()
    return out


def main():
    cert = [("RS32_s4", 32, 4, 2), ("RS32_s4_tau1", 32, 4, 1), ("P64_s4", 64, 4, 8), ("P64_s8", 64, 8, 8),
            ("P64_s4_tau4", 64, 4, 4), ("P64_s4_tau2", 64, 4, 2), ("S4_128_s8", 128, 8, 8), ("S4_128_s16", 128, 16, 8)]
    unc = [("P64_s2 (30.3 only)", 64, 2, 8), ("RS32_s2 (30.3 only)", 32, 2, 2), ("S4_128_s4 (uncertified)", 128, 4, 8)]
    review8 = [("r16_4_8", 16, 4, 8), ("r32_4_2", 32, 4, 2), ("r64_4_8", 64, 4, 8), ("r64_2_8", 64, 2, 8), ("r64_8_8", 64, 8, 8),
               ("r64_4_4", 64, 4, 4), ("r64_4_2", 64, 4, 2), ("r128_16_8", 128, 16, 8)]
    sets = [("certified_formed", cert),
            ("certified_formed_plus_32_4_8_planned", cert + [("(32,4,8) planned", 32, 4, 8)]),
            ("every_formation_record (labelled)", cert + unc),
            ("review8", review8)]
    res = [analyse(n, r) for n, r in sets]
    json.dump(res, open(sys.argv[1], "w"), indent=1)
    for o in res:
        print("==", o["set"], "rows", o["n_rows"])
        for r in o["rows"]:
            print("   ", r)
        for lab in ("raw", "unit_columns"):
            d = o[lab]
            print("  [%s] singular values:" % lab, " ".join("%.6g" % s for s in d["singular_values"]))
            for tol in (1e-10, 1e-6):
                print("  [%s] rank at rel tol %g: %d of 8; QR dependent columns: %s" % (lab, tol, d["rank_tol_%g" % tol],
                                                                                    d["qr_dependent_cols_tol_%g" % tol]))
                for nv in d["null_tol_%g" % tol]:
                    print("      null (sigma %.3g):" % nv["sigma"], " ".join("%s:%.6g" % (c, v) for c, v in zip(COLS, nv["vector"])))
        if "control_X_dot_null" in o:
            print("  control X @ review null:", ["%.2e" % v for v in o["control_X_dot_null"]])


if __name__ == "__main__":
    main()
