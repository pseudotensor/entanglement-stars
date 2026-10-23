#!/usr/bin/env python
"""Builder diagnostic for strand 2b R4 (not a spec job): at a given z = (psi, N) and lambda, compare the assembled Jacobian
(System.J) with a brute-force central-difference Jacobian of F (state re-solved at every point), column by column, and
report the block spectrum (lowest levels, d0, gap) and the smallest distance from E_0 to other sector levels."""
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
import strand2b_lib_v2 as B  # noqa: E402
import strand2b_row_R4_v2 as RR  # noqa: E402
import native_core as nc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, help="case JSON with the continuation path")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rec = json.load(open(a.case))
    blk = tuple(int(v) for v in rec["block"].split(","))
    Lam = np.inf if rec["Lambda"] == "inf" else float(rec["Lambda"])
    lat = S2.lattice("R4")
    K, _ = nc.K_matrix(nc.lattice("ring4")["positions"])
    fops = B.FockOps(8)
    nW = 4
    Nc = nW + blk[0] - blk[1]
    rp = B.RhoP(lat, rec["m"], rec["gM"], rec["gB"], Lam, Nc, fops)
    sys_ = B.System(rp, K, rec["k"], blk[0] + blk[1])
    n = 4
    sys_.d0_fixed = sys_.state(np.ones(n), np.ones(n))["d0_at_tolerance"]
    # re-run the continuation to the last accepted lambda to get z there
    z = np.ones(2 * n)
    out = {"path": []}
    lams = [p["lam"] for p in rec["continuation"]["path"]][1:]
    for lam in lams:
        z, info = sys_.newton(z, lam)
        st = sys_.state(z[:n], z[n:])
        b = st["b"]
        e, _ = B.block_states(st["H"], b["nexc"], sys_.kexc)
        w = np.linalg.eigvalsh(B.herm(st["H"]))
        out["path"].append({"lam": lam, "residuals": info["residuals"], "block_levels": e[:6].tolist(), "d0": st["d0"],
                            "gap": st["gap_block"], "sector_levels_near_E0": sorted(w[np.argsort(np.abs(w - e[0]))[:6]].tolist())})
    lam = lams[-1]
    Jm, st, mg = sys_.J(z, lam)
    hh = 1e-5
    Jfd = np.zeros_like(Jm)
    for c in range(2 * n):
        zp, zm = z.copy(), z.copy()
        zp[c] += hh
        zm[c] -= hh
        Jfd[:, c] = (sys_.F(zp, lam)[0] - sys_.F(zm, lam)[0]) / (2 * hh)
    out["lam"] = lam
    out["z"] = z.tolist()
    out["J_assembled"] = Jm.tolist()
    out["J_fd"] = Jfd.tolist()
    out["col_rel_diff"] = [float(np.linalg.norm(Jm[:, c] - Jfd[:, c]) / max(np.linalg.norm(Jfd[:, c]), 1e-300)) for c in range(2 * n)]
    out["min_level_distance"] = mg
    out["d0_fixed"] = sys_.d0_fixed
    with open(os.path.join(a.out, "debug_jac.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({"col_rel_diff": out["col_rel_diff"], "path_tail": out["path"][-1]}, indent=1))


if __name__ == "__main__":
    main()
