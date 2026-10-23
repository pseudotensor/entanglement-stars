#!/usr/bin/env python
"""engine_valley_degeneracy_v1.py (builder analyst; for V3D's container decision, labelled): the lowest positive levels of the free
3-dimensional sine operator (N = 1, v = 0, m a = 0.3) on open lattice regions, grouped into degenerate sets (tolerance 1e-9): the open cube at
L = 8, 9, 10 (odd L: the open sine chain's exact zero mode in each direction), and lattice balls |x - c| <= R_wall (Dirichlet outside) centred
at a site (odd L) or at a cube-centre point (even L).  The question: are the 8 valleys degenerate (sets of 16 = 8 valleys x 2) on the
container that V3D uses?  Prints the sets; writes nothing else."""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import numpy as np
import scipy.linalg as sl
import engine_ops_v1 as OPS


def sets_of(e, n=80, tol=1e-9):
    ep = np.sort(e[e > 0])[:n]
    s = []
    for x in ep:
        if s and abs(x - s[-1][0]) < tol:
            s[-1][1] += 1
        else:
            s.append([float(x), 1])
    return s


for L in (8, 9, 10):
    h, Q = OPS.lattice3d_parts(L, 0.3)
    print("open cube L %d: %s" % (L, " ".join("%.5f(%d)" % (x, d) for x, d in sets_of(sl.eigvalsh(h.toarray()))[:10])), flush=True)
for L, Rw in ((13, 6.0), (14, 6.0), (13, 5.0), (12, 5.0)):
    h, Q = OPS.lattice3d_parts(L, 0.3)
    c = (L - 1) / 2.0
    g = np.indices((L, L, L)).reshape(3, -1).T
    r = np.sqrt(((g - c) ** 2).sum(1))
    sel = np.nonzero(r <= Rw)[0]
    modes = (4 * sel[:, None] + np.arange(4)[None, :]).ravel()
    e = sl.eigvalsh(h.toarray()[np.ix_(modes, modes)])
    print("ball L %d, centre at a %s, R_wall %g (%d sites): %s" % (L, "site" if L % 2 else "cube-centre point", Rw, len(sel),
          " ".join("%.5f(%d)" % (x, d) for x, d in sets_of(e)[:12])), flush=True)
