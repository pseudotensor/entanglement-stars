#!/usr/bin/env python
"""Builder profiling of strand2b_lib_v3.RhoP.build on R4 (20 builds at nearby psi), cProfile top entries."""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import cProfile, pstats, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "frozen"))
import strand2a_lib as S2
import strand2b_lib_v3 as B
lat = S2.lattice("R4")
fops = B.FockOps(8)
rp = B.RhoP(lat, 0.1, 0.2, 1.0, 1.0, 5, fops)
def run():
    for i in range(20):
        rp.build(np.array([1.0, 1.0, 1.0, 1.0]) + 1e-3 * i)
t0 = time.time()
cProfile.run("run()", "<internal>/prof_s2b")
print("20 builds: %.2f s" % (time.time() - t0))
pstats.Stats("<internal>/prof_s2b").sort_stats("cumulative").print_stats(18)
