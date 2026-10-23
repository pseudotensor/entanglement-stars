#!/usr/bin/env python
"""Strand 2b sec. 14 item 2, post-processing (builder analyst): the limits of strand2b_c8_unique_v1.py's starts, grouped and checked.

D1 asks for every distinct positive solution.  The run accepted a limit at residual 1e-11, below the case's own evaluation noise
(the record's residual_noise_floor, the largest change of |F| under ten random 1e-12 displacements of the continuation solution).
This script takes every limit whose best residual is at or below that noise floor and whose components are positive, groups the
limits into orbits of the cube's 48 symmetries (signed permutations of the centred site positions acting on the site labels of
(psi, N)), and for one representative per orbit evaluates, with the same objects as the case driver strand2b_row_C8_v6.py: the
residual |F|, the noise floor (the same ten 1e-12 displacements, seed 5), the FD Jacobian's singular values (sigma_min, cond), the
state (E, gap, d0_at_tolerance, multiplet splitting), E_sel = -k (psi - 1) K (N psi - 1) + N . source, the smallest component, and
the driver's sampled enclosure (radii 1e-3, 1e-2, 1e-1; the 16 axes of z and 24 random unit directions, seed 3; q = sup ||1 - Y J(z
+ r v)||, Y = J(z)^-1; passed if |Y F| + r q < r; the positive domain and the smallest gap over the sampled ball); the same for the
continuation solution as the reference row.  Jacobians at the sampled points run in parallel workers (one thread each).
Writes <out>/post_<case>.json."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import itertools
import json
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "frozen"))

_G = {}


def build(rec):
    import strand2a_lib as S2
    import strand2b_c8_lib_v5 as C
    import native_core as nc
    lat = S2.lattice("C8")
    K, _ = nc.K_matrix(lat["positions"].tolist())
    bm = C.BlockModel(lat, rec["m"], rec["gM"], rec["gB"], float(rec["Lambda"]), tuple(int(x) for x in rec["block"].split(",")))
    sys_ = C.SystemC8(bm, K, rec["k"])
    sys_.d0_fixed = rec["d0"]
    return C, lat, K, sys_


def init(rec_path):
    rec = json.load(open(rec_path))
    C, lat, K, sys_ = build(rec)
    _G.update(C=C, sys=sys_)


def jac_and_gap(zp):
    zp = np.array(zp)
    n = _G["sys"].n
    Jp = _G["C"].jacobian_fd(_G["sys"], zp)
    gap = float(_G["sys"].state(zp[:n], zp[n:])["gap"])
    return Jp, gap


def symmetries(pos):
    c = pos - pos.mean(0)
    perms = []
    for P in itertools.permutations(range(3)):
        for sg in itertools.product((1, -1), repeat=3):
            R = np.zeros((3, 3))
            for i in range(3):
                R[i, P[i]] = sg[i]
            img = c @ R.T
            p = []
            for x in range(len(c)):
                d = np.linalg.norm(c - img[x], axis=1)
                y = int(np.argmin(d))
                if d[y] > 1e-9:
                    p = None
                    break
                p.append(y)
            if p is not None and len(set(p)) == len(c):
                perms.append(p)
    return perms


def act(z, p, n):
    # site x -> p[x]: the image configuration takes the value of site x at site p[x]
    psi, N = np.zeros(n), np.zeros(n)
    for x in range(n):
        psi[p[x]] = z[x]
        N[p[x]] = z[n + x]
    return np.concatenate([psi, N])


def checks(pool, sys_, C, K, z, radii=(1e-3, 1e-2, 1e-1), n_random=24):
    n = sys_.n
    psi, N = z[:n], z[n:]
    Fz, st, src, t = sys_.F(z)
    rngF = np.random.default_rng(5)
    noise = float(max(np.linalg.norm(sys_.F(z + 1e-12 * rngF.normal(size=z.size))[0] - Fz) for _ in range(10)))
    Jm = C.jacobian_fd(sys_, z)
    sv = np.linalg.svd(Jm, compute_uv=False)
    k = sys_.k
    out = {"psi": psi.tolist(), "N": N.tolist(), "residual": float(np.linalg.norm(Fz)), "noise_floor": noise,
           "sigma_min_J": float(sv[-1]), "cond_J": float(sv[0] / sv[-1]), "E": st["E"], "gap": st["gap"],
           "d0_at_tolerance": st["d0_at_tolerance"], "multiplet_splitting": st["multiplet_splitting"],
           "E_sel": float(-k * (psi - 1) @ K @ (N * psi - 1) + np.dot(N, src)), "min_component": float(z.min())}
    Y = np.linalg.inv(Jm)
    YF = float(np.linalg.norm(Y @ Fz))
    rngE = np.random.default_rng(3)
    dirs = [np.eye(2 * n)[i] for i in range(2 * n)]
    sph = [v / np.linalg.norm(v) for v in rngE.normal(size=(n_random, 2 * n))]
    enc = {"Y_F": YF, "norm_Y": float(np.linalg.norm(Y, 2)), "radii": {}}
    best = None
    for r in radii:
        pts = [z + r * v for v in dirs + sph]
        pts = [zp.tolist() for zp in pts if np.all(zp > 0)]
        res = pool.map(jac_and_gap, pts)
        qmax = max(float(np.linalg.norm(np.eye(2 * n) - Y @ Jp, 2)) for Jp, _ in res) if res else None
        gmin = min(g for _, g in res) if res else None
        ok_domain = bool(psi.min() - r > 0 and N.min() - r > 0)
        passed = bool(qmax is not None and YF + r * qmax < r)
        enc["radii"][str(r)] = {"q": qmax, "test_lhs": YF + r * qmax if qmax is not None else None, "passed": passed,
                                "positive_domain": ok_domain, "min_gap_sampled": gmin, "points": len(pts)}
        if passed and ok_domain:
            best = r
    enc["largest_passing_r_in_domain"] = best
    out["enclosure"] = enc
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--record", required=True)
    ap.add_argument("--unique", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--reference", action="store_true", help="also recompute the checks at the continuation solution")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    rec = json.load(open(a.record))
    U = json.load(open(a.unique))
    C, lat, K, sys_ = build(rec)
    n = sys_.n
    perms = symmetries(lat["positions"])
    z_cont = np.concatenate([np.array(rec["psi"]), np.array(rec["N"])])
    floor = float(rec["residual_noise_floor"])
    lim = []
    for r in U["rows"]:
        if r.get("best") is None or "limit" not in r:
            continue
        zz = np.array(r["limit"])
        if r["best"] <= floor and zz.min() > 0:
            lim.append((r["trial"], zz, r["best"]))
    orbits = []
    for trial, zz, b in lim:
        placed = False
        for o in orbits:
            if any(np.linalg.norm(act(o["rep"], p, n) - zz) < 1e-6 for p in perms):
                o["trials"].append(trial)
                o["best_residuals"].append(b)
                o["max_distance_to_rep_image"] = max(o["max_distance_to_rep_image"],
                                                     min(np.linalg.norm(act(o["rep"], p, n) - zz) for p in perms))
                placed = True
                break
        if not placed:
            orbits.append({"rep": zz, "trials": [trial], "best_residuals": [b], "max_distance_to_rep_image": 0.0})
    for o in orbits:
        o["orbit_size"] = len({tuple(np.round(act(o["rep"], p, n), 6)) for p in perms})
        o["stabilizer_order"] = len(perms) // o["orbit_size"]
        o["distance_to_continuation"] = float(min(np.linalg.norm(act(o["rep"], p, n) - z_cont) for p in perms))
        o["is_continuation_solution"] = bool(o["distance_to_continuation"] < 1e-6)
    out = {"record": os.path.basename(a.record), "unique": os.path.basename(a.unique), "k": rec["k"], "noise_floor_threshold": floor,
           "symmetries": len(perms), "limits_at_or_below_floor_positive": len(lim),
           "rows_not_at_floor": [{"trial": r["trial"], "best": r.get("best"), "final_residual": r.get("final_residual"), "error": r.get("error")}
                                 for r in U["rows"] if not (r.get("best") is not None and r["best"] <= floor and min(r.get("limit", [-1])) > 0)],
           "orbits": []}
    with Pool(a.workers, initializer=init, initargs=(a.record,)) as pool:
        for o in orbits:
            row = {k_: (v_.tolist() if isinstance(v_, np.ndarray) else v_) for k_, v_ in o.items()}
            if not o["is_continuation_solution"]:
                row["checks"] = checks(pool, sys_, C, K, o["rep"])
            out["orbits"].append(row)
            print(json.dumps({k_: v_ for k_, v_ in row.items() if k_ not in ("rep",)})[:1500], flush=True)
        if a.reference:
            out["continuation_checks"] = checks(pool, sys_, C, K, z_cont)
    out["seconds"] = time.time() - t0
    tag = os.path.basename(a.record).replace("case_", "").replace(".json", "")
    with open(os.path.join(a.out, "post_%s.json" % tag), "w") as fh:
        json.dump(out, fh, indent=1)
    print("seconds", out["seconds"], flush=True)


if __name__ == "__main__":
    main()
