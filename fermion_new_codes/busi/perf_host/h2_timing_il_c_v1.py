#!/usr/bin/env python
"""h2_timing_il_c_v1.py (busi, analyst; PERF H2-C): Option C's increment over A+B, interleaved in one process like
h2_timing_interleaved_v1.py: arms AB (H2_PW_ONE, H2_LOCAL_SPLINE) and ABC (+ H2_RESIDUAL_PAIR) on the --with-c build engine_fields_v16bc
(module globals toggled; H2_CPU_QUOTA must be set). usage: python h2_timing_il_c_v1.py SRC FIELDS_NPZ ELL NREC REPS ROT0"""
import sys
import time

import numpy as np

sys.path.insert(0, sys.argv[1])
import engine_fields_v16bc as FLD   # noqa: E402
import engine_fields_v4 as F4, engine_fields_v12 as F12, engine_fields_v14 as F14   # noqa: E401,E402
import engine_fields_kcache_v1 as KC   # noqa: E402
KC.install([F4, F12, F14, FLD])
z = np.load(sys.argv[2], allow_pickle=False)
R = z["R"]; g = float(z["g"]); ell = float(sys.argv[3]); nrec = int(sys.argv[4]); reps = int(sys.argv[5]); rot = int(sys.argv[6])
idx = list(range(len(z["T"]) - nrec, len(z["T"])))
ARMS = [("AB", False), ("ABC", True)]
T = {a[0]: [] for a in ARMS}
k = rot
FLD._H2_PW_ONE = FLD._H2_LOCAL_SPLINE = True
for rep in range(reps):
    for i in idx:
        for j in range(2):
            name, fc = ARMS[(k + j) % 2]
            FLD._H2_RESIDUAL_PAIR = fc
            t0 = time.perf_counter()
            Nn, vn, Wn, info = FLD.solve_constraints_ivp(z["eps_tot"][i], z["J_tot"][i], g, R, ell, N0=None, rtol=1e-12, atol=1e-16,
                                                         collar_policy="flag", support_rel=1e-14)
            if not info.get("failed"):
                FLD.site_fields_kernel_average(info, R, ell)
            T[name].append(time.perf_counter() - t0)
        k += 1
        print("rep %d rec %d AB %.3f ABC %.3f" % (rep, i, T["AB"][-1], T["ABC"][-1]), flush=True)
a, c = np.array(T["AB"]), np.array(T["ABC"])
print("ARM AB  solves %d mean %.4f s median %.4f s" % (len(a), a.mean(), np.median(a)))
print("ARM ABC solves %d mean %.4f s median %.4f s" % (len(c), c.mean(), np.median(c)))
print("PAIRED ABC/AB: mean ratio %.4f, median ratio %.4f, mean saving %.4f s per solve" % ((c / a).mean(), np.median(c / a), (a - c).mean()))
