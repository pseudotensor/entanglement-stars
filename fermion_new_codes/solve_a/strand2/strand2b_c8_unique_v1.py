#!/usr/bin/env python
"""Strand 2b sec. 14 item 2 (builder analyst): the cube uniqueness rerun for the two enclosure cases ((2,0), m 0.1, (0.2, 1, 1),
k in {1, 1e4}): 16 starts with every component of z = (psi, N) on the 8 sites drawn in [0.8, 1.25] (seed 11), damped Newton with the
FD Jacobian of F (strand2b_c8_lib_v5.newton_fd: step halving while the residual grows or a component turns nonpositive; at most 40
iterations), the projector rank d_0 fixed at the case's flat-start value; per start the convergence, the iterations, the final
residual, the limit's distance to the continuation solution and its smallest component; the number of distinct positive solutions
(distance above 1e-8).  Starts run in parallel workers (one thread each).  Writes <out>/unique_<case>.json."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "frozen"))

_G = {}


def init(rec_path):
    import strand2a_lib as S2
    import strand2b_c8_lib_v5 as C
    import native_core as nc
    rec = json.load(open(rec_path))
    lat = S2.lattice("C8")
    K, _ = nc.K_matrix(lat["positions"].tolist())
    bm = C.BlockModel(lat, rec["m"], rec["gM"], rec["gB"], float(rec["Lambda"]), (2, 0))
    sys_ = C.SystemC8(bm, K, rec["k"])
    sys_.d0_fixed = rec["d0"]
    _G.update(C=C, sys=sys_, z=np.concatenate([np.array(rec["psi"]), np.array(rec["N"])]))


def run(args):
    trial, zs = args
    t0 = time.time()
    try:
        zz, inf = _G["C"].newton_fd(_G["sys"], np.array(zs), maxit=40)
        return {"trial": trial, "start": list(map(float, zs)), "ok": bool(inf["ok"]), "iterations": len(inf.get("residuals", [])) - 1,
                "final_residual": float(inf["residuals"][-1]) if inf.get("residuals") else None, "best": inf.get("best"),
                "stopped": inf.get("stopped", inf.get("why")), "limit": list(map(float, zz)),
                "distance_to_continuation": float(np.linalg.norm(zz - _G["z"])), "min_component": float(np.min(zz)),
                "seconds": time.time() - t0}
    except Exception as e:  # recorded, not fatal
        return {"trial": trial, "start": list(map(float, zs)), "ok": False, "error": repr(e), "seconds": time.time() - t0}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--record", required=True, help="the enclosure case JSON (its psi, N, d0, couplings, k)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--starts", type=int, default=16)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    rec = json.load(open(a.record))
    n2 = 2 * len(rec["psi"])
    rng = np.random.default_rng(11)
    starts = [(t, rng.uniform(0.8, 1.25, n2).tolist()) for t in range(a.starts)]
    with Pool(a.workers, initializer=init, initargs=(a.record,)) as pool:
        rows = pool.map(run, starts)
    z = np.concatenate([np.array(rec["psi"]), np.array(rec["N"])])
    sols = []
    for r in rows:
        if r.get("ok") and r.get("min_component", -1) > 0:
            lim = np.array(r["limit"])
            if not any(np.linalg.norm(lim - s) < 1e-8 for s in sols):
                sols.append(lim)
    out = {"record": os.path.basename(a.record), "k": rec["k"], "d0": rec["d0"], "starts": a.starts, "box": [0.8, 1.25], "seed": 11,
           "converged": sum(1 for r in rows if r.get("ok")), "converged_positive": sum(1 for r in rows if r.get("ok") and r.get("min_component", -1) > 0),
           "converged_nonpositive": sum(1 for r in rows if r.get("ok") and r.get("min_component", 1) <= 0),
           "distinct_positive": len(sols), "distinct_positive_distances_to_continuation": [float(np.linalg.norm(s - z)) for s in sols],
           "rows": rows, "seconds": time.time() - t0}
    tag = os.path.basename(a.record).replace("case_", "").replace(".json", "")
    with open(os.path.join(a.out, "unique_%s.json" % tag), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k_: v_ for k_, v_ in out.items() if k_ != "rows"}), flush=True)


if __name__ == "__main__":
    main()
