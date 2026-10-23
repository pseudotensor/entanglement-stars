#!/usr/bin/env python
"""Strand 2b sec. 15 item 5 (spec solve/specs/SPEC_strand2b_row.md sec. 15; builder analyst): the cube's symmetry-broken solution classes.

(a) Continuation in k: the uniform continuation solution and the representatives of the two non-uniform classes of sec. 14 (c8unique1post)
at the enclosure case ((2,0), m 0.1, (g_M, g_B, Lambda) = (0.2, 1, 1), d_0 fixed at the record's), carried upward through
k = 1, 2, 5, 10, 30, 100, 1e3, 1e4 by the FD Newton of strand2b_c8_lib_v5 (damped, 40 iterations) from the previous k's limit; acceptance
at the noise floor (the best residual at most 10 times the largest change of |F| under ten random 1e-12 displacements at the limit, or
1e-9).  Per (class, k): converged, residual, noise floor, E (the block's lowest level), E_sel, the gap, sigma_min of the FD Jacobian, the
smallest component, and the distance to the uniform solution at the same k minimized over the cube's 48 symmetries; the class persists
(distance above 1e-6), merges (below) or is lost (no convergence).
(b) The near starts at k = 1 for (2,0) at (g_M, g_B) = (0.05, 0) and at g = 0 (Lambda = 1, d_0 fixed at the flat start's value): the
uniform continuation solution, then the 16 starts of sec. 14 (every component in [0.8, 1.25], seed 11), the limits at the noise floor
grouped into orbits of the 48 symmetries with E, E_sel and the distance to the uniform solution per orbit.  Starts in a pool of
single-thread workers.  Writes <out>/item5.json."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "frozen"))
import strand2a_lib as S2  # noqa: E402
import strand2b_c8_lib_v5 as C  # noqa: E402
import native_core as nc  # noqa: E402
from strand2b_c8_unique_post_v1 import symmetries, act  # noqa: E402

LAT = S2.lattice("C8")
K, _ = nc.K_matrix(LAT["positions"].tolist())
K = np.asarray(K, float)
PERMS = symmetries(LAT["positions"])


def make(m, gM, gB, Lam, k, d0=None):
    bm = C.BlockModel(LAT, m, gM, gB, Lam, (2, 0))
    s = C.SystemC8(bm, K, k)
    s.d0_fixed = d0
    return s


def noise(sys_, z):
    F0 = sys_.F(z)[0]
    rng = np.random.default_rng(5)
    return float(max(np.linalg.norm(sys_.F(z + 1e-12 * rng.normal(size=z.size))[0] - F0) for _ in range(10)))


def describe(sys_, z, zu=None):
    n = sys_.n
    Fz, st, src, t = sys_.F(z)
    k = sys_.k
    psi, N = z[:n], z[n:]
    J = C.jacobian_fd(sys_, z)
    sv = np.linalg.svd(J, compute_uv=False)
    out = {"residual": float(np.linalg.norm(Fz)), "E": st["E"], "gap": st["gap"], "d0_at_tolerance": st["d0_at_tolerance"],
           "E_sel": float(-k * (psi - 1) @ K @ (N * psi - 1) + np.dot(N, src)), "sigma_min_J": float(sv[-1]), "min_component": float(z.min()),
           "psi": psi.tolist(), "N": N.tolist()}
    if zu is not None:
        out["distance_to_uniform"] = float(min(np.linalg.norm(act(zu, p, n) - z) for p in PERMS))
    return out


def solve(sys_, z0):
    zz, inf = C.newton_fd(sys_, np.array(z0), maxit=40)
    nf = noise(sys_, zz) if np.all(zz > 0) else None
    best = inf.get("best", inf["residuals"][-1] if inf.get("residuals") else None)
    ok = bool(nf is not None and best is not None and best <= max(1e-9, 10 * nf) and np.all(zz > 0))
    return zz, ok, best, nf


_W = {}


def winit(args):
    m, gM, gB, Lam, k, d0 = args
    _W["sys"] = make(m, gM, gB, Lam, k, d0)


def wstart(zs):
    zz, ok, best, nf = solve(_W["sys"], zs)
    return {"limit": zz.tolist(), "ok": ok, "best": best, "noise": nf}


def part_b(m, gM, gB, Lam, workers):
    s0 = make(m, gM, gB, Lam, 1.0)
    zu, inf = C.continuation_fd(s0)
    d0 = s0.d0_fixed
    n = s0.n
    rec = {"g": [gM, gB, Lam], "d0": d0, "continuation_ok": bool(inf.get("ok")), "uniform": describe(s0, zu)}
    rng = np.random.default_rng(11)
    starts = [rng.uniform(0.8, 1.25, 2 * n) for _ in range(16)]
    with Pool(workers, initializer=winit, initargs=((m, gM, gB, Lam, 1.0, d0),)) as pool:
        rows = pool.map(wstart, starts)
    orbits = []
    for r in rows:
        if not r["ok"]:
            continue
        z = np.array(r["limit"])
        for o in orbits:
            if any(np.linalg.norm(act(o["rep"], p, n) - z) < 1e-6 for p in PERMS):
                o["count"] += 1
                break
        else:
            orbits.append({"rep": z, "count": 1})
    rec["starts"] = 16
    rec["converged_positive"] = sum(1 for r in rows if r["ok"])
    rec["orbits"] = []
    for o in orbits:
        d = describe(s0, o["rep"], zu)
        d["starts"] = o["count"]
        rec["orbits"].append(d)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--record", required=True)
    ap.add_argument("--post", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    rec = json.load(open(a.record))
    P = json.load(open(a.post))
    Lam = float(rec["Lambda"])
    z_u = np.concatenate([rec["psi"], rec["N"]])
    classes = {"uniform": z_u}
    for j, o in enumerate(P["orbits"]):
        if not o["is_continuation_solution"]:
            classes["class %d" % (j + 1)] = np.array(o["rep"])
    res = {"part_a": {name: [] for name in classes}, "part_b": []}
    cur = dict(classes)
    for k in (1.0, 2.0, 5.0, 10.0, 30.0, 100.0, 1e3, 1e4):
        s = make(rec["m"], rec["gM"], rec["gB"], Lam, k, rec["d0"])
        zu, oku, bu, nfu = solve(s, cur["uniform"])
        if oku:
            cur["uniform"] = zu
        du = describe(s, zu)
        du.update({"k": k, "converged": oku, "best": bu, "noise_floor": nfu})
        res["part_a"]["uniform"].append(du)
        for name in classes:
            if name == "uniform" or cur.get(name) is None:
                continue
            zz, ok, best, nf = solve(s, cur[name])
            d = describe(s, zz, cur["uniform"]) if np.all(zz > 0) else {"min_component": float(zz.min())}
            d.update({"k": k, "converged": ok, "best": best, "noise_floor": nf})
            d["status"] = ("lost" if not ok else ("merged" if d.get("distance_to_uniform", 1) < 1e-6 else "persists"))
            res["part_a"][name].append(d)
            if ok:
                cur[name] = zz
            print("k %g %s: %s, best %.2e, E %s, distance %s (%.0f s)" % (k, name, d["status"], best if best else -1, d.get("E"),
                                                                      d.get("distance_to_uniform"), time.time() - t0), flush=True)
            if d["status"] != "persists":
                cur[name] = None
        print("k %g uniform: converged %s E %.8f (%.0f s)" % (k, oku, du["E"], time.time() - t0), flush=True)
    for (gM, gB) in ((0.05, 0.0), (0.0, 0.0)):
        r = part_b(rec["m"], gM, gB, Lam, a.workers)
        res["part_b"].append(r)
        print("part b (%g, %g): d0 %s, uniform E %.6f; %d of 16 converged positive; %d orbits: %s (%.0f s)" % (
            gM, gB, r["d0"], r["uniform"]["E"], r["converged_positive"], len(r["orbits"]),
            [(o["starts"], round(o["E"], 6), round(o["distance_to_uniform"], 4)) for o in r["orbits"]], time.time() - t0), flush=True)
    res["seconds"] = time.time() - t0
    with open(os.path.join(a.out, "item5.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
