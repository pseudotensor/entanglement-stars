#!/usr/bin/env python
"""run_bprime_cost_v1.py (busi; B' cost inputs per ReviewB_pc13 sec. 1.8: operator-application throughput, the number of propagated columns
(the interior window counts), the interior eigensolve, the quasi-mode arm's sparse factorization). Box CPU only.
Per size s = L/a in --sizes and family in --families (F1 the c19 sigma4 r_s/a 64 record at T/r_s 30; F2 the frozen dust ball; A32 the set17
r_s/a 32 record at T/r_s 18), clock d = 0, ell/L = 1/8, retained radius = the averaged outer root:
  * the interior block's dimension, the window counts in (0, T0), (0, 3 T0) and the exit window, the structural zero modes excluded, the time;
  * the cube apply time per column at batch 1 and --batch (torch CPU, --threads), and the bytes per column;
  * for sizes in --lu-sizes: scipy splu of (h - i eta - sigma) on the full box (the shift-invert Arnoldi's factorization): time and fill."""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pt3_core as K3  # noqa: E402
import pt3b_cube as CU  # noqa: E402
import pt3b_kernels as KR  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="16,32")
    ap.add_argument("--families", default="F1,F2,A32")
    ap.add_argument("--lu-sizes", default="8,12")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--dense-max", type=int, default=6000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    profs = {"F1": lambda: CU.dense_record_profile(*CU.REC_F1), "F2": CU.f2_profile, "A32": CU.a32_profile}
    res = {"sizes": {}, "lu": {}, "threads": a.threads}
    cache = {}
    for s in [float(x) for x in a.sizes.split(",")]:
        L = s; H = int(round(2.5 * L)); T0 = 1 / (4 * math.pi * L)
        row = {"H": H, "amplitudes": 4 * (2 * H + 1) ** 3, "bytes_per_column": 16 * 4 * (2 * H + 1) ** 3, "families": {}}
        for fam in a.families.split(","):
            pr = cache.get(fam) or profs[fam](); cache[fam] = pr
            N, v, r, _ = CU.cube_fields(pr, L, H, 0.0, 1.0 / 8)
            ar = CU.averaged_root(N, v, r, H)["axis"]
            R_t = max(ar) if ar else max(pr["roots_over_L"]) * L
            Hs = int(math.ceil(R_t)) + 2
            # the sub-box with the averaging done on the full box: restrict the full box's fields
            off = H - Hs
            sl = slice(off, off + 2 * Hs + 1)
            Ns = N[sl, sl, sl]; vs = [c[sl, sl, sl] for c in v]; rsb = r[sl, sl, sl]
            hs = CU.cubic_csr(Ns, vs)
            ims = CU.masks(rsb, R_t)
            f = {"R_t": R_t, "R_t_over_L": R_t / L, "averaged_roots": ar, "interior_dim": int(ims.sum())}
            for lab, (lo, hi) in {"supply": (0.0, T0), "supply_3T0": (0.0, 3 * T0), "exit": (-T0, 0.0)}.items():
                zi = {}; t0 = time.time()
                try:
                    e, U, dI = KR.interior_window(hs, ims, lo, hi, dense_max=a.dense_max, info=zi)
                    f[lab] = {"count": int(len(e)), "zero_modes_excluded": zi.get("n_zero", 0), "seconds": time.time() - t0,
                              "e_over_T0": [float(x) for x in (e / T0)[:40]]}
                except Exception as ex:
                    f[lab] = {"error": repr(ex), "seconds": time.time() - t0}
            row["families"][fam] = f
            K3.log("s %g %s: R_t/L %.3f, interior dim %d, supply %s, 3T0 %s, exit %s" % (s, fam, R_t / L, f["interior_dim"], f["supply"].get("count"),
                                                                                       f["supply_3T0"].get("count"), f["exit"].get("count")))
        # throughput on the F1 fields
        N, v, r, _ = CU.cube_fields(cache["F1"] if "F1" in cache else CU.f2_profile(), L, H, 0.0, 1.0 / 8)
        cube = CU.Cube(N, v, CU.cap_reviewa(r, L))
        for b in (1, a.batch):
            X = torch.randn((b, 4) + N.shape, dtype=CU.CD)
            cube.apply(X)
            t0 = time.time(); nrep = 3
            for _ in range(nrep):
                cube.apply(X)
            row["apply_seconds_batch%d" % b] = (time.time() - t0) / nrep
            del X
        row["cheb_bound"] = cube.bound
        K3.log("s %g: apply %.3f s (batch 1), %.3f s (batch %d); bound %.3f; %.2f GB per column" % (s, row["apply_seconds_batch1"], row["apply_seconds_batch%d" % a.batch],
                                                                                                    a.batch, cube.bound, row["bytes_per_column"] / 1e9))
        res["sizes"][str(int(s))] = row
        K3.dump_json(res, os.path.join(a.out, "bprime_cost_v1.json"))
    for s in [float(x) for x in a.lu_sizes.split(",") if x]:
        from scipy.sparse.linalg import splu
        L = s; H = int(round(2.5 * L))
        N, v, r, _ = CU.cube_fields(cache.get("F1") or CU.f2_profile(), L, H, 0.0, 1.0 / 8)
        Hm = CU.cubic_csr(N, v, CU.cap_reviewa(r, L)).tocsc()
        t0 = time.time()
        try:
            lu = splu(Hm, permc_spec="COLAMD")
            res["lu"][str(int(s))] = {"dim": Hm.shape[0], "nnz": int(Hm.nnz), "fill_nnz": int(lu.L.nnz + lu.U.nnz), "seconds": time.time() - t0}
        except Exception as ex:
            res["lu"][str(int(s))] = {"dim": Hm.shape[0], "error": repr(ex), "seconds": time.time() - t0}
        K3.log("LU s %g: %s" % (s, res["lu"][str(int(s))]))
        K3.dump_json(res, os.path.join(a.out, "bprime_cost_v1.json"))


if __name__ == "__main__":
    main()
