#!/usr/bin/env python3
"""obs_ball_cut_ctrl_time_v1.py (builder analyst; S6 = J4 dense controls): times the lane's strand-3 dense ball code (balls.py of the
paper's orient package, md5 9c13956b9279935b3ac3d1b634b78d73, imported unmodified from the directory given by --orient) at each control
(R, L): one column of its dense build (Ball.matvec on a unit vector, averaged over --reps), the ball's site count and dimension D = 4 n, and
the projected dense build time D x t (the eigvalsh after it is not included).  Prints one line per control and writes --out-json."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import argparse
import json
import sys
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orient", required=True)
    ap.add_argument("--controls", nargs="+", default=["6.5:80", "4:64", "6:128"])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.orient)
    from balls import Ball  # noqa: E402  (the lane's code, unmodified)
    rows = []
    for c in a.controls:
        R, L = c.split(":")
        R, L = float(R), int(L)
        b = Ball(R, [0.0, 0.0, 0.0], L, 0.0)
        z = np.zeros(b.D)
        z[0] = 1.0
        b.matvec(z)
        t0 = time.time()
        for _ in range(a.reps):
            b.matvec(z)
        t = (time.time() - t0) / a.reps
        row = dict(R=R, L=L, sites=int(len(b.xyz)), D=int(b.D), t_matvec=t, projected_build_s=b.D * t)
        rows.append(row)
        print("control R %g L %d: sites %d D %d, matvec %.3f s, projected dense build %.0f s (%.2f h)" % (
            R, L, row["sites"], row["D"], t, row["projected_build_s"], row["projected_build_s"] / 3600), flush=True)
    with open(a.out_json, "w") as fh:
        json.dump(dict(script="obs_ball_cut_ctrl_time_v1.py", rows=rows), fh, indent=1)


if __name__ == "__main__":
    main()
