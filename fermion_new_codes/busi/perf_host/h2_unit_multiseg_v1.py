#!/usr/bin/env python
"""h2_unit_multiseg_v1.py (busi, analyst; PERF H2-A coverage): the stored R1 sources give one U/L segment per solve, so Option A's
multi-segment branch (searchsorted over the starts, the clip to [0, nseg-1], the per-segment clip) is not reached by the replay. Here a
solve's real Piecewise is re-cut into 7 segments over the same dense output (4 kept as the OdeSolution, 3 behind a plain Python function, as
a stretch's r(s) inversion is), and Piecewise.__call__ is compared off (v16's path) against H2_PW_ONE on, bytewise, at r0, every cut and
both adjacent floats, beyond the last end, negative radii, nan, +-inf and 5000 random radii; for 3 records.
usage: python h2_unit_multiseg_v1.py SRC_DIR FIELDS_NPZ ELL"""
import sys
import numpy as np
sys.path.insert(0, sys.argv[1])
import engine_fields_v16b as FLD, engine_fields_v4 as F4, engine_fields_v12 as F12, engine_fields_v14 as F14, engine_fields_kcache_v1 as KC  # noqa
KC.install([F4, F12, F14, FLD])
z = np.load(sys.argv[2], allow_pickle=False); R = z["R"]; g = float(z["g"]); ell = float(sys.argv[3])
rng = np.random.default_rng(7)
P = []
class RecP(FLD.Piecewise):
    def __init__(self, *a, **k):
        super().__init__(*a, **k); P.append(self)
FLD.Piecewise = RecP
key = lambda v: (np.asarray(v).dtype.str, np.asarray(v).shape, np.asarray(v).tobytes())
n = bad = 0
for i in (len(z["T"]) - 20, len(z["T"]) - 10, len(z["T"]) - 1):
    del P[:]
    FLD._H2_PW_ONE = False
    FLD.solve_constraints_ivp(z["eps_tot"][i], z["J_tot"][i], g, R, ell, N0=None, rtol=1e-12, atol=1e-16, collar_policy="flag", support_rel=1e-14)
    pw = P[0]; s0, s1, sol = pw.segs[0][:3]
    cuts = np.linspace(s0, s1, 8)
    segs = []
    for j in range(7):
        f = sol if j % 2 == 0 else (lambda r, _s=sol: _s(r))
        segs.append((float(cuts[j]), float(cuts[j + 1]), f, 1, 0))
    q = FLD.Piecewise.__mro__[1](pw.cc, pw.r0, segs)
    pts = np.concatenate([[pw.r0, 0.0, s1 + 1.0, 2 * s1], cuts])
    pts = np.concatenate([pts, np.nextafter(pts, -np.inf), np.nextafter(pts, np.inf)])
    pts = np.concatenate([pts, -pts, [-0.0, np.nan, np.inf, -np.inf], rng.uniform(-0.1 * s1, 1.2 * s1, 5000)])
    for x in pts:
        xa = np.array([x])
        with np.errstate(all="ignore"):
            FLD._H2_PW_ONE = False; off = q(xa)
            FLD._H2_PW_ONE = True; on = q(xa)
        FLD._H2_PW_ONE = False
        n += 1
        if [key(v) for v in off] != [key(v) for v in on]:
            bad += 1
            if bad <= 5:
                print("MISMATCH rec %d x %r" % (i, x))
    print("rec %d: 7 segments over [%.4f, %.4f], %d points" % (i, s0, s1, len(pts)), flush=True)
print("UNIT A multi-segment: %d checks, %d mismatches -> %s" % (n, bad, "PASS" if bad == 0 else "FAIL"))
raise SystemExit(0 if bad == 0 else 1)
