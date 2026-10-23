#!/usr/bin/env python
"""h2_timing_v1.py (busi, analyst; PERF H2-A/H2-B): the H2 replay harness (perf_host/harness/newton_field_solve_replay_v1.py, md5 d136a85c)
with the field module named on the command line and time.perf_counter; no snapshot, no wrapper (the review's sec. 3: time without the gate
work). The options come from the environment (H2_PW_ONE, H2_LOCAL_SPLINE). If the module has the measurement attribute _H2_TM (only the
measurement copy engine_fields_v16b_tm.py, never shipped), its per-solve W-integration and residuals_at seconds are printed too.
usage: python h2_timing_v1.py SRC_DIR FIELDS_NPZ ELL NREC REPS [MODULE]"""
import importlib
import os
import sys
import time

import numpy as np

sys.path.insert(0, sys.argv[1])
FLD = importlib.import_module(sys.argv[6] if len(sys.argv) > 6 else "engine_fields_v16")
import engine_fields_v4 as F4, engine_fields_v12 as F12, engine_fields_v14 as F14   # noqa: E401,E402
import engine_fields_kcache_v1 as KC   # noqa: E402
KC.install([F4, F12, F14, FLD])
z = np.load(sys.argv[2], allow_pickle=False)
R = z["R"]; g = float(z["g"]); ell = float(sys.argv[3]); nrec = int(sys.argv[4]); reps = int(sys.argv[5])
idx = list(range(len(z["T"]) - nrec, len(z["T"])))
print("module %s file %s; H2_PW_ONE=%s H2_LOCAL_SPLINE=%s H2_RESIDUAL_PAIR=%s; OMP %s MKL %s OPENBLAS %s; pid %d" % (
    FLD.__name__, FLD.__file__, os.environ.get("H2_PW_ONE"), os.environ.get("H2_LOCAL_SPLINE"), os.environ.get("H2_RESIDUAL_PAIR"),
    os.environ.get("OMP_NUM_THREADS"), os.environ.get("MKL_NUM_THREADS"), os.environ.get("OPENBLAS_NUM_THREADS"), os.getpid()), flush=True)
tt = []; tw = []; tr = []
for rep in range(reps):
    for i in idx:
        tm = getattr(FLD, "_H2_TM", None)
        if tm is not None:
            tm.clear()
        t0 = time.perf_counter()
        Nn, vn, Wn, info = FLD.solve_constraints_ivp(z["eps_tot"][i], z["J_tot"][i], g, R, ell, N0=None, rtol=1e-12, atol=1e-16,
                                                     collar_policy="flag", support_rel=1e-14)
        if not info.get("failed"):
            Na, va, norms = FLD.site_fields_kernel_average(info, R, ell)
        tt.append(time.perf_counter() - t0)
        extra = ""
        if tm is not None:
            tw.append(tm.get("W", 0.0)); tr.append(tm.get("res", 0.0))
            extra = " W %.4f res %.4f (res64 %.4f res32 %.4f)" % (tw[-1], tr[-1], tm.get("res64", 0.0), tm.get("res32", 0.0))
        print("rep %d rec %d T %.4f solve %.4f s nfev %s failed %s%s" % (rep, i, z["T"][i], tt[-1], (info.get("ivp") or {}).get("nfev"),
                                                                       info.get("failed"), extra), flush=True)
tt = np.array(tt)
print("solves %d mean %.4f s median %.4f s total %.1f s; reps-after-first mean %.4f s; kcache %s" % (
    len(tt), tt.mean(), np.median(tt), tt.sum(), tt[len(idx):].mean() if reps > 1 else tt.mean(), KC.STATS))
if tw:
    print("W integration mean %.4f s, residuals_at (both) mean %.4f s per solve" % (np.mean(tw), np.mean(tr)))
