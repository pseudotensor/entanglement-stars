#!/usr/bin/env python
"""Strand 3 addendum, parts C and D (spec solve/specs/SPEC_strand3_addendum.md; builder analyst).

C-masses:   from strand 3's masses.json, the lambda = 1 rows at a/l_P = 0.02 and 0.00594: the masses 4 q_s of the heavy packets
            (q_2 = q_3 and q_4), written as a job file for strand3_planar.py ((1,0,0), the (32,24) ladder) and strand3_batches.py
            (the mesh-2 sphere at the two 0.02 masses).
C-assemble: the four-sea totals with the light packet massless (eta(0) + 2 eta(4 q_2) + eta(4 q_4)), planar (32,24) and, at 0.02,
            sphere; R^static_local = total/(16 eta_req) beside the denominator-only R/16 of strand 3's scan.
D:          on the 4^3 periodic lattice of strand 3's part B (m = 0.1, both backgrounds), with R = {x_1 in {0, 1}}:
            ||P_R U (1 - P_R)|| (operator norm) for the polar unitary U; for the clock vertex X_x = psi_x E_0^(1/2) P_x E_0^(1/2) at
            sites x = (0,0,0) and (1,0,0) (the slab has width 2, so both are boundary sites) and at (2,0,0) (outside R):
            ||(1 - P_R) X_x (1 - P_R)||/||X_x|| and ||P_R X_x (1 - P_R)||/||X_x||; and the A1 covariance (projector on the negative
            modes of h_*) restricted to the image of the region's modes under V = diag(1, U^dag) (W = V[:, R], the spec's transported
            region) and, for comparison, under V^dag (W = V^dag[:, R], which is what part B's frame A2 restricts): restricted
            spectra (distance from {0, 1}) and entropies.
Writes <out>/masses_addC.json, <out>/addC.json, <out>/addD.json."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import glob
import json
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "orient"))
from lattice_entropy import h2  # noqa: E402


def key(q):
    return "%.12g" % q


def c_masses(a):
    M = json.load(open(a.masses))
    rows = []
    for r in M["rows"]:
        if r["lambda"] == 1.0 and any(abs(r["a_over_lP"] - v) < 1e-12 for v in (0.02, 0.00594)):
            rows.append({"a_over_lP": r["a_over_lP"], "q": r["q"], "eta_req": r["eta_req"],
                         "q4": [4 * r["q"][1], 4 * r["q"][3]]})
    jobs = [[[1, 0, 0], 32, 24, q] for r in rows for q in r["q4"]]
    sph = [q for r in rows if abs(r["a_over_lP"] - 0.02) < 1e-12 for q in r["q4"]]
    out = {"rows": rows, "planar_jobs": jobs, "sphere_masses": sph, "ball_jobs": []}
    with open(os.path.join(a.out, "masses_addC.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("masses_addC: planar %s sphere %s" % ([j[3] for j in jobs], sph))


def c_assemble(a):
    J = json.load(open(os.path.join(a.out, "masses_addC.json")))
    pl = {}
    for f in glob.glob(os.path.join(a.out, "planar", "planar_1_0_0_d32_nk24_q*.json")) + \
            glob.glob(os.path.join(a.s3run, "planar", "planar_1_0_0_d32_nk24_q0.json")):
        r = json.load(open(f))
        pl[key(r["mass"])] = r["eta"]
    sp = {}
    for f in glob.glob(os.path.join(a.out, "sphere", "sphere_q*.json")) + glob.glob(os.path.join(a.s3run, "sphere", "sphere_q0.json")):
        r = json.load(open(f))
        sp[key(r["mass"])] = r.get("eta_sphere", r.get("eta"))
    scan = json.load(open(os.path.join(a.s3run, "scan.json")))
    res = []
    for r in J["rows"]:
        q2, q4 = r["q4"]
        e0 = pl.get(key(0.0))
        rec = {"a_over_lP": r["a_over_lP"], "q4": r["q4"], "eta_req": r["eta_req"], "eta100_0": e0,
               "eta100_4q2": pl.get(key(q2)), "eta100_4q4": pl.get(key(q4))}
        if None not in (e0, rec["eta100_4q2"], rec["eta100_4q4"]):
            tot = e0 + 2 * rec["eta100_4q2"] + rec["eta100_4q4"]
            rec["total100_local_static"] = tot
            rec["R100_static_local"] = tot / (16 * r["eta_req"])
        if key(q2) in sp and key(q4) in sp and key(0.0) in sp:
            tots = sp[key(0.0)] + 2 * sp[key(q2)] + sp[key(q4)]
            rec["total_sphere_local_static"] = tots
            rec["Rsphere_static_local"] = tots / (16 * r["eta_req"])
        srow = [s for s in scan["rows"] if abs(s["a_over_lP"] - r["a_over_lP"]) < 1e-12 and s["lambda"] == 1.0]
        if srow:
            rec["scan_R100_static_denominator_only"] = {k: v for k, v in srow[0].items() if k.startswith("R100_static")}
        res.append(rec)
        print(json.dumps(rec), flush=True)
    with open(os.path.join(a.out, "addC.json"), "w") as fh:
        json.dump({"rows": res, "planar": pl, "sphere": sp}, fh, indent=1)


def part_d(a):
    import strand3_partB_v2 as PB
    D = PB.weyl()
    xyz = PB.coords()
    m = 0.1
    bgs = {"flat": (np.ones(PB.NSITE), np.ones(PB.NSITE)),
           "nonuniform": (1 + 0.2 * np.cos(2 * np.pi * xyz[:, 0] / 4) * np.cos(2 * np.pi * xyz[:, 1] / 4),
                          1 + 0.3 * np.cos(2 * np.pi * xyz[:, 2] / 4))}
    n = 2 * PB.NSITE
    inR = np.repeat(xyz[:, 0] <= 1, 2)
    PR = np.diag(inR.astype(float))
    Ridx = np.concatenate([np.where(inR)[0], n + np.where(inR)[0]])
    out = {}
    for name, (psi, N) in bgs.items():
        h, U, B0, B1, minAA = PB.law(m, psi, N, D)
        P = np.kron(np.diag(psi), np.eye(2))
        Pinv = np.kron(np.diag(1.0 / psi), np.eye(2))
        A = m * Pinv + 1j * Pinv @ Pinv @ Pinv @ D
        w0, V0 = np.linalg.eigh(PB.herm(A.conj().T @ A))
        E0h = (V0 * w0 ** 0.25) @ V0.conj().T
        rec = {"cross_boundary_U": float(np.linalg.norm(PR @ U @ (np.eye(n) - PR), 2)),
               "cross_boundary_U_rel": float(np.linalg.norm(PR @ U @ (np.eye(n) - PR), 2) / np.linalg.norm(U, 2))}
        for c in ((0, 0, 0), (1, 0, 0), (2, 0, 0)):
            x = PB.site(*c)
            Px = np.zeros((n, n))
            Px[2 * x:2 * x + 2, 2 * x:2 * x + 2] = np.eye(2)
            X = psi[x] * E0h @ Px @ E0h
            nx = np.linalg.norm(X, 2)
            rec["X_%d%d%d" % c] = {"outside_outside_rel": float(np.linalg.norm((np.eye(n) - PR) @ X @ (np.eye(n) - PR), 2) / nx),
                                   "inside_outside_rel": float(np.linalg.norm(PR @ X @ (np.eye(n) - PR), 2) / nx),
                                   "inside_inside_rel": float(np.linalg.norm(PR @ X @ PR, 2) / nx), "in_R": bool(c[0] <= 1)}
        wv, Vv = np.linalg.eigh(PB.herm(h))
        Vm = Vv[:, wv < 0]
        Pm = Vm @ Vm.conj().T
        Vt = np.block([[np.eye(n), np.zeros((n, n))], [np.zeros((n, n)), U.conj().T]])
        for lab, W in (("image_under_V", Vt[:, Ridx]), ("image_under_Vdag", Vt.conj().T[:, Ridx]), ("A1_sites", np.eye(2 * n)[:, Ridx])):
            C = W.conj().T @ Pm @ W
            ev = np.linalg.eigvalsh(PB.herm(C))
            rec["restricted_" + lab] = {"entropy": float(h2(ev).sum()), "max_distance_from_0_1": float(np.max(np.minimum(np.abs(ev), np.abs(1 - ev)))),
                                        "n_fractional_1e-6": int(np.sum(np.minimum(np.abs(ev), np.abs(1 - ev)) > 1e-6))}
        out[name] = rec
        print(name, json.dumps(rec), flush=True)
    with open(os.path.join(a.out, "addD.json"), "w") as fh:
        json.dump(out, fh, indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["C-masses", "C-assemble", "D"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--masses", default="")
    ap.add_argument("--s3run", default="", help="strand 3 run folder (planar/, sphere/, scan.json)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    {"C-masses": c_masses, "C-assemble": c_assemble, "D": part_d}[a.what](a)


if __name__ == "__main__":
    main()
