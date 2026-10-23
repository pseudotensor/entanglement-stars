#!/usr/bin/env python
"""h2_timing_interleaved_v1.py (busi, analyst; PERF H2-A/H2-B): the four arms off, A, B, A+B in ONE process, interleaved solve by solve
(the order of the arms rotates from record to record), so every arm sees the same box load and the same warm caches: the box was
oversubscribed (load 60 to 130 on 72 cores, other lanes' jobs) and the concurrent separate-process timing gave original 5.81 against off
6.45 s per solve for the same arithmetic. The flags are the module globals that the env vars set at import (_H2_PW_ONE, _H2_LOCAL_SPLINE;
read at call time). Runs on the measurement copy engine_fields_v16b_tm (v16b + perf_counter around the W integration and residuals_at), so
the W and residual seconds per arm come out as well. usage: python h2_timing_interleaved_v1.py SRC_TM FIELDS_NPZ ELL NREC REPS ROT0"""
import sys
import time

import numpy as np

sys.path.insert(0, sys.argv[1])
import engine_fields_v16b_tm as FLD   # noqa: E402
import engine_fields_v4 as F4, engine_fields_v12 as F12, engine_fields_v14 as F14   # noqa: E401,E402
import engine_fields_kcache_v1 as KC   # noqa: E402
KC.install([F4, F12, F14, FLD])
z = np.load(sys.argv[2], allow_pickle=False)
R = z["R"]; g = float(z["g"]); ell = float(sys.argv[3]); nrec = int(sys.argv[4]); reps = int(sys.argv[5]); rot = int(sys.argv[6])
idx = list(range(len(z["T"]) - nrec, len(z["T"])))
ARMS = [("off", False, False), ("A", True, False), ("B", False, True), ("AB", True, True)]
T = {a[0]: [] for a in ARMS}; TW = {a[0]: [] for a in ARMS}; TR = {a[0]: [] for a in ARMS}
k = rot
for rep in range(reps):
    for i in idx:
        for j in range(4):
            name, fa, fb = ARMS[(k + j) % 4]
            FLD._H2_PW_ONE, FLD._H2_LOCAL_SPLINE = fa, fb
            FLD._H2_TM.clear()
            t0 = time.perf_counter()
            Nn, vn, Wn, info = FLD.solve_constraints_ivp(z["eps_tot"][i], z["J_tot"][i], g, R, ell, N0=None, rtol=1e-12, atol=1e-16,
                                                         collar_policy="flag", support_rel=1e-14)
            if not info.get("failed"):
                FLD.site_fields_kernel_average(info, R, ell)
            T[name].append(time.perf_counter() - t0); TW[name].append(FLD._H2_TM["W"]); TR[name].append(FLD._H2_TM["res"])
        k += 1
        print("rep %d rec %d " % (rep, i) + " ".join("%s %.3f/%.3f" % (a[0], T[a[0]][-1], TW[a[0]][-1]) for a in ARMS), flush=True)
for a in ARMS:
    n = a[0]
    print("ARM %-3s solves %d mean %.4f s median %.4f s; W mean %.4f s; residuals_at mean %.4f s" % (n, len(T[n]), np.mean(T[n]), np.median(T[n]), np.mean(TW[n]), np.mean(TR[n])))
d = {a[0]: np.array(T[a[0]]) for a in ARMS}
for n in ("A", "B", "AB"):
    r = d[n] / d["off"]
    print("PAIRED %-3s/off: mean ratio %.4f, median ratio %.4f, mean saving %.4f s per solve" % (n, r.mean(), np.median(r), (d["off"] - d[n]).mean()))
