#!/usr/bin/env python
"""Builder check: strand2b_lib_v4 builds equal strand2b_lib_v3 builds (rho^P_x, n_exc) at random psi, all R4 blocks' sectors."""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "frozen"))
import strand2a_lib as S2
import strand2b_lib_v3 as B3
import strand2b_lib_v4 as B4
lat = S2.lattice("R4")
f3, f4 = B3.FockOps(8), B4.FockOps(8)
rng = np.random.default_rng(5)
worst = 0.0
for Nc in (3, 4, 5, 6):
    for (gM, gB, L) in ((0.2, 1.0, 1.0), (0.05, 0.0, np.inf)):
        r3 = B3.RhoP(lat, 0.1, gM, gB, L, Nc, f3)
        r4 = B4.RhoP(lat, 0.1, gM, gB, L, Nc, f4)
        psi = rng.uniform(0.8, 1.2, 4)
        a, b = r3.build(psi), r4.build(psi)
        worst = max(worst, float(np.max(np.abs(a["rho"] - b["rho"]))), float(np.max(np.abs(a["nexc"] - b["nexc"]))))
t0 = time.time()
r4 = B4.RhoP(lat, 0.1, 0.2, 1.0, 1.0, 5, f4)
for i in range(50):
    r4.build(np.ones(4) + 1e-3 * i)
print("max |v3 - v4| = %.3e; v4 build %.1f ms" % (worst, 1000 * (time.time() - t0) / 50))
