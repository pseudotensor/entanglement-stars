#!/usr/bin/env python
"""h2_unit_cov_v1.py (busi, analyst; PERF H2-A/H2-B): the coverage cases of the reviewer B perf_host_newton review sec. 3 ("Coverage
limitation") that a replay of stored sources may not reach, on engine_fields_v16b, bytewise (dtype, shape, bytes) option on against off.

Option B (_h2_scalar_spline): for each record's smoothed sources rho and J (SmoothedSources5 as solve_constraints_ivp builds them), the
wrapper against the original BSpline at every knot, both adjacent floats of every knot, the interval midpoints, the ends t[k], t[n] and
outside points, 0.0, -0.0, nan, +-inf, 4000 random points, as Python float and as np.float64, in ascending, descending, shuffled and
far-jump orders through ONE wrapper instance (so the last-interval cache and the local-spline cache are exercised in non-monotone order),
plus 0-d and length-one arrays (the pass-through path).
Option A (Piecewise / Nfin): every Piecewise the solve builds is recorded (a subclass that only stores the instance); then with the module
flag _H2_PW_ONE False and True the same length-one inputs are evaluated: r0 and its adjacent floats, every segment's start and end and their
adjacent floats, r_collar and its adjacent floats, 0.0, -0.0, negative radii, nan, +-inf, 2000 random radii; for pw(x) (U, L) and for
info["N_dense"] (the nested Nfin) and info["U_dense"].
usage: python h2_unit_cov_v1.py SRC_DIR FIELDS_NPZ ELL NREC"""
import os
import sys
import types

import numpy as np

sys.path.insert(0, sys.argv[1])
import engine_fields_v16b as FLD   # noqa: E402
import engine_fields_v4 as F4   # noqa: E402
import engine_fields_v12 as F12   # noqa: E402
import engine_fields_v14 as F14   # noqa: E402
import engine_fields_kcache_v1 as KC   # noqa: E402

KC.install([F4, F12, F14, FLD])
z = np.load(sys.argv[2], allow_pickle=False)
R = z["R"]; g = float(z["g"]); ell = float(sys.argv[3]); nrec = int(sys.argv[4])
idx = list(range(len(z["T"]) - nrec, len(z["T"])))
if os.environ.get("H2_GATE_RECS"):   # an explicit record list (the stretch records) in place of the last NREC
    idx = [int(v) for v in os.environ["H2_GATE_RECS"].split(",")]
rng = np.random.default_rng(20261020)


def key(v):
    a = np.asarray(v)
    return (type(v).__name__, a.dtype.str, a.shape, a.tobytes())


def adj(x):
    x = np.asarray(x, float)
    return np.concatenate([x, np.nextafter(x, -np.inf), np.nextafter(x, np.inf)])


PIECES = []


class RecPiecewise(FLD.Piecewise):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        PIECES.append(self)


FLD.Piecewise = RecPiecewise
nB = nBbad = nA = nAbad = 0
cov = {"segments": 0, "stretch_segments": 0, "neg_segments": 0, "gauge_intervals": 0, "zeros": 0, "near_tangencies": 0, "local_splines": 0}
for i in idx:
    # ---- option B
    r_nodes = FLD.coarse_grid(R, ell)
    src = FLD.SmoothedSources5(z["eps_tot"][i], z["J_tot"][i], R, ell, r_nodes[-1], 1e-14)
    for name in ("rho", "J"):
        sp = getattr(src, name)
        w = FLD._h2_scalar_spline(sp)
        if w is sp:
            raise SystemExit("record %d %s: the wrapper declined the spline (type %s, k %s)" % (i, name, type(sp), sp.k))
        t = sp.t; k = sp.k; n = len(t) - k - 1
        base = np.concatenate([adj(t), 0.5 * (t[1:] + t[:-1]), [t[k], t[n], t[0] - 1.0, t[-1] + 1.0, 0.0, -0.0, np.nan, np.inf, -np.inf],
                               rng.uniform(t[0] - 1.0, t[-1] + 1.0, 4000)])
        orders = [np.sort(base), np.sort(base)[::-1], rng.permutation(base)]
        srt = np.sort(base[np.isfinite(base)]); m = len(srt)
        orders.append(np.array([srt[j // 2] if j % 2 == 0 else srt[m - 1 - j // 2] for j in range(m)]))   # far jumps
        for xs in orders:
            for x in xs:
                for xv in (float(x), np.float64(x)):
                    nB += 1
                    if key(w(xv)) != key(sp(xv)):
                        nBbad += 1
                        if nBbad <= 5:
                            print("B MISMATCH rec %d %s x %r (%s)" % (i, name, xv, type(xv).__name__))
        for xv in (np.array(1.25 * ell), np.array([1.25 * ell]), np.array([0.5, 1.0, 3.0])):
            nB += 1
            if key(w(xv)) != key(sp(xv)):
                nBbad += 1; print("B MISMATCH (array pass-through) rec %d %s" % (i, name))
        cov["local_splines"] += len(w.__closure__[[c for c in w.__code__.co_freevars].index("cache")].cell_contents)
    # ---- option A
    del PIECES[:]
    FLD._H2_PW_ONE = False
    Nn, vn, Wn, info = FLD.solve_constraints_ivp(z["eps_tot"][i], z["J_tot"][i], g, R, ell, N0=None, rtol=1e-12, atol=1e-16,
                                                 collar_policy="flag", support_rel=1e-14)
    if info.get("failed"):
        print("record %d failed: %s" % (i, info["failed"])); continue
    gz = info.get("gauge") or {}
    cov["gauge_intervals"] += len(gz.get("intervals", [])); cov["zeros"] += len(gz.get("zeros", [])); cov["near_tangencies"] += int(gz.get("near_tangencies", 0))
    rc = float(info["r_collar"])
    for pw in PIECES:
        cov["segments"] += len(pw.segs)
        cov["stretch_segments"] += sum(1 for s in pw.segs if isinstance(s[2], types.FunctionType))   # the stretch's r(s) inversion is a plain function
        cov["neg_segments"] += sum(1 for s in pw.segs if len(s) > 3 and s[3] != 1)
        pts = [pw.r0, rc, 0.0]
        for s in pw.segs:
            pts += [float(s[0]), float(s[1])]
        pts = adj(np.array(pts))
        pts = np.concatenate([pts, -pts, [-0.0, np.nan, np.inf, -np.inf], rng.uniform(-0.1 * rc, 1.1 * rc, 2000)])
        fns = [("pw", pw), ("N_dense", info["N_dense"]), ("U_dense", info["U_dense"])]
        for x in pts:
            xa = np.array([x], dtype=float)
            for fname, f in fns:
                FLD._H2_PW_ONE = False
                with np.errstate(all="ignore"):
                    off = f(xa)
                FLD._H2_PW_ONE = True
                with np.errstate(all="ignore"):
                    on = f(xa)
                FLD._H2_PW_ONE = False
                ko = [key(v) for v in off] if isinstance(off, tuple) else [key(off)]
                kn = [key(v) for v in on] if isinstance(on, tuple) else [key(on)]
                nA += 1
                if ko != kn:
                    nAbad += 1
                    if nAbad <= 5:
                        print("A MISMATCH rec %d %s x %r" % (i, fname, x))
    print("rec %d: Piecewise instances %d, segments %s, B checks so far %d (%d bad), A checks so far %d (%d bad)"
          % (i, len(PIECES), [len(p.segs) for p in PIECES], nB, nBbad, nA, nAbad), flush=True)
print("coverage %s" % cov)
print("UNIT B: %d scalar/array checks, %d mismatches -> %s" % (nB, nBbad, "PASS" if nBbad == 0 else "FAIL"))
print("UNIT A: %d length-one checks, %d mismatches -> %s" % (nA, nAbad, "PASS" if nAbad == 0 else "FAIL"))
raise SystemExit(0 if (nBbad == 0 and nAbad == 0) else 1)
