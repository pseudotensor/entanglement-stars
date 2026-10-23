#!/usr/bin/env python3
"""obs_box3d_dtladder_v1.py: the dt ladder of J4's smoke resolved in complex128 (builder analyst).  The smoke record (obs_box3d_v1.py,
smoke_L32.json) compared dt = 1 with dt = 0.5 in complex64, where the difference (max|dC| 4.5e-7) sits at the complex64 noise (complex64
against complex128 max|dC| 4.0e-7); here the same region quantity (the ball R < 5 at L = 32, r_s = 8, R_m = 2, R_cut = 2.5 r_s, m = 0,
hold 12/kappa) is computed in complex128 at dt = 1, 0.5 and 0.25 with obs_box3d_v2.py's routines unchanged (the Gram in complex128, the
entropy plain and particle-hole).  Label: the exact Painleve-Gullstrand profile (control), truncated on the periodic box."""
import argparse
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import obs_box3d_v2 as BOX  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--L", type=int, default=32)
    p.add_argument("--rs", type=float, default=8.0)
    p.add_argument("--Rm", type=float, default=2.0)
    p.add_argument("--rcut-over-rs", type=float, default=2.5)
    p.add_argument("--radius", type=float, default=5.0)
    p.add_argument("--hold-over-kappa", type=float, default=12.0)
    p.add_argument("--dts", type=float, nargs="+", default=[1.0, 0.5, 0.25])
    p.add_argument("--outdir", required=True)
    p.add_argument("--tag", required=True)
    a = p.parse_args()
    dev = torch.device("cuda:0")
    dt128 = torch.complex128
    kappa = 1.0 / (2 * a.rs)
    F, _, info = BOX.build_pg_box(a.L, a.rs, a.Rm, a.rcut_over_rs * a.rs, 0.0, dev, dt128)
    sea = BOX.FlatSea(a.L, 0.0, dev, dt128)
    sites, _ = BOX.region_indices(a.L, lambda R: R < a.radius)
    n = 4 * len(sites)
    h = a.hold_over_kappa / kappa
    out = {"label": BOX.LABEL_PG, "script": os.path.basename(__file__), "args": vars(a), "box": info, "modes": n, "started_utc": BOX.utc(), "runs": {}}
    Cs = {}
    for dt in a.dts:
        t0 = time.time()
        steps = BOX.history_ramp(2 * a.rs, dt, BOX.ramp_envelope(2 * a.rs))
        Zp = torch.empty((2 * a.L ** 3, n), dtype=dt128, device=dev)
        Zq = torch.empty((2 * a.L ** 3, n), dtype=dt128, device=dev)
        for c0 in range(0, n, 512):
            nb = min(512, n - c0)
            Y = BOX.backward_multi(F, BOX.onehot(a.L, sites, c0, nb, dt128, dev), steps, [h])[h]
            Zp[:, c0:c0 + nb], Zq[:, c0:c0 + nb] = sea.compress_both(Y)
        C = BOX.gram(Zp, dev, fp64=True)
        Cb = BOX.gram(Zq, dev, fp64=True)
        e = BOX.entropy_ph(C, Cb)
        Cs[dt] = C
        out["runs"]["%g" % dt] = {"S_ph": e["S"], "S_plain": e["S_plain"], "pair_defect_max": e["pair_defect_max"], "wall_s": time.time() - t0}
        del Zp, Zq, Cb
        BOX.log("dt %g: %s" % (dt, out["runs"]["%g" % dt]))
    ref = min(a.dts)
    for dt in a.dts:
        if dt != ref:
            out["runs"]["%g" % dt]["max_abs_dC_vs_dt%g" % ref] = float((Cs[dt] - Cs[ref]).abs().max())
            out["runs"]["%g" % dt]["dS_ph_vs_dt%g" % ref] = out["runs"]["%g" % dt]["S_ph"] - out["runs"]["%g" % ref]["S_ph"]
    if 1.0 in Cs and 0.5 in Cs and ref < 0.5:
        e1 = float((Cs[1.0] - Cs[ref]).abs().max())
        e05 = float((Cs[0.5] - Cs[ref]).abs().max())
        out["order_estimate_log2_err1_over_err05"] = float(np.log2(e1 / e05)) if e05 > 0 else None
    out["finished_utc"] = BOX.utc()
    BOX.jdump(out, os.path.join(a.outdir, "dtladder128_%s.json" % a.tag))
    BOX.log("done %s" % out)


if __name__ == "__main__":
    main()
