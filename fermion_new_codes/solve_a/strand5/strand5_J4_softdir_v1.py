#!/usr/bin/env python
"""Strand 5 J4, spec sec. 10(r)(a) (builder analyst): the soft full-node directions.  At the five S3b members of a family (stage 2 version 9's
five_members), with version 9's exact machinery (strand5_J4_stage2_v9: the state re-solved from the clock row at the member's n_c, the exact
gradient dM/dn_c = -lambda loc_n, the exact second variation): the full matter-node Hessian projected to fixed N_B in the volume metric (y = D v,
D = diag((4 pi r^2)^1/2), the complement of D 1 by a Householder reflection) is diagonalized with eigenvectors; along the lowest eigenvector (and
the second where the inertia is 2), v = D^-1 y with ||y|| = 1 (unit volume norm, sum 4 pi r^2 v = 0), the exact gradient is differenced
centrally, Rayleigh quotient q(h) = v . [g(n_c + h v) - g(n_c - h v)]/(2h), at h = eps ||n_c||_vol for eps = 1e-3, 5e-4, 2.5e-4 (each h capped
at half the largest step keeping n_c positive on every node, recorded); Richardson R1 = (4 q(h/2) - q(h))/3, R2 = (4 q(h/4) - q(h/2))/3, the
estimate (16 R2 - R1)/15 and the error |R2 - R1|; against the eigenvalue: the difference, and the ratio of the error to the eigenvalue's
distance from zero (accepted if below 1).  Writes <out>/softdir_<tag>.json."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import json
import sys
import time
from types import SimpleNamespace

import numpy as np
import scipy.linalg as sla

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import strand5_J4_stage2_v9 as V9  # noqa: E402

V8 = V9.V8
S = V9.S
EPS = (1e-3, 5e-4, 2.5e-4)


def full_node_vectors(ex, k=2):
    r, nc = ex.r, ex.nc
    idx = np.nonzero(nc > 0)[0]
    Jm = len(idx)
    Q = np.zeros((ex.J, Jm))
    Q[idx, np.arange(Jm)] = 1.0
    H, asym = ex.hessian(Q)
    dvec = np.sqrt(4 * np.pi * r[idx] ** 2)
    Hy = H / dvec[:, None] / dvec[None, :]
    u = dvec / np.linalg.norm(dvec)
    e1 = np.zeros(Jm)
    e1[0] = 1.0
    w = u - e1
    w /= np.linalg.norm(w)
    P = np.eye(Jm) - 2 * np.outer(w, w)
    Hp = (P.T @ Hy @ P)[1:, 1:]
    ev, V = sla.eigh(Hp, subset_by_index=[0, k - 1])
    neg = sla.eigh(Hp, eigvals_only=True, subset_by_value=[-np.inf, 0.0], check_finite=False)
    vecs = []
    for i in range(k):
        y = P[:, 1:] @ V[:, i]
        v = np.zeros(ex.J)
        v[idx] = y / dvec
        vecs.append(v)
    return ev, vecs, int(len(neg)), idx


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", required=True)
    ap.add_argument("--kernel", default=None)
    ap.add_argument("--packets", default=None)
    ap.add_argument("--stage2", required=True, help="stage 2 version 9's record of the family")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    R = json.load(open(a.family))
    M = R["meta"]
    S2 = json.load(open(a.stage2))
    five = S2["meta"]["five_members"]
    prof = np.load(a.family.replace(".json", "_profiles.npz"))
    args = SimpleNamespace(kernel=a.kernel, packets=a.packets, mlight=M["mlight"], aH=M["aH"], operator=M["operator"])
    stencil, kop, info = S.setup_kernel(args)
    gM, gB = M["couplings"]
    coup = (gM, gB, 1.0, 1.0)
    e1 = S.E1Table(M["mlight"], gM, gB, 1.0, 1.0)
    mem = [d for d in R["members"] if d.get("newton_ok")]
    res = {"meta": {"family": a.family, "stage2": a.stage2, "five_members": five, "eps": list(EPS), "k_op": kop, "version": 1}, "members": []}
    out = os.path.join(a.out, "softdir_" + os.path.basename(a.family))
    for i in five:
        d = mem[i]
        zc = d["z_c"]
        key = "z%.6f" % zc
        r, psi, N, n, qF = (prof[key + s] for s in ("_r", "_psi", "_N", "_n", "_qF"))
        J = len(r)
        star = S.Star(J, stencil, kop, M["mlight"], coup, e1)
        chi = N * psi
        x = np.concatenate([psi, chi, [d["A"], d["B"], d["mu_inf"]]])
        F = V8.Fam(star, M["mlight"], e1)
        kap_nodes = psi * psi * qF
        ex = V9.Exact(F, star, x, n, kop, kap_nodes)
        s2m = [m for m in S2["members"] if abs(m["z_c"] - zc) < 1e-12]
        inertia_s2 = s2m[0].get("full_node", {}).get("inertia") if s2m else None
        nvec = 2 if (inertia_s2 is not None and inertia_s2 >= 2) else 1
        ev, vecs, inertia, idx = full_node_vectors(ex, k=2)
        ncvol = float(np.sqrt(np.sum(4 * np.pi * r[idx] ** 2 * ex.nc[idx] ** 2)))
        rec = {"z_c": zc, "J": J, "matter_nodes": int(len(idx)), "inertia": inertia, "inertia_stage2": inertia_s2, "lowest2": ev.tolist(), "directions": []}
        for k in range(nvec):
            v = vecs[k]
            mk = np.abs(v) > 0
            hcap = 0.5 * float(np.min(ex.nc[mk] / np.abs(v[mk])))
            q, hs = [], []
            for eps in EPS:
                h = min(eps * ncvol, hcap * eps / EPS[0])
                g = []
                for sgn in (+1, -1):
                    ncp = ex.nc + sgn * h * v
                    xx = np.concatenate([ex.psi, ex.psi, [ex.A, ex.A, 0.0]])
                    e2 = V9.Exact(F, star, xx, ncp / ex.psi ** 6, kop, ex.kap)
                    g.append(e2.grad)
                q.append(float(v @ (g[0] - g[1]) / (2 * h)))
                hs.append(h)
            # Richardson for the actual step ratios (2 between successive steps unless capped)
            R1 = (4 * q[1] - q[0]) / 3
            R2 = (4 * q[2] - q[1]) / 3
            est = (16 * R2 - R1) / 15
            err = abs(R2 - R1)
            lam = float(ev[k])
            rec["directions"].append({"index": k, "eigenvalue": lam, "steps": hs, "capped": bool(hcap * 1.0 < EPS[0] * ncvol), "q": q, "R1": R1, "R2": R2,
                                      "estimate": est, "error": err, "estimate_minus_eigenvalue": est - lam,
                                      "error_over_distance_from_zero": err / abs(lam) if lam != 0 else None,
                                      "accepted": bool(err < abs(lam))})
            print("z_c %.4f dir %d: eigenvalue %.6e, q %s, estimate %.6e (minus eigenvalue %.2e), error %.2e, ratio %.2e (%.0f s)" % (
                zc, k, lam, ["%.6e" % y for y in q], est, est - lam, err, err / abs(lam) if lam else float("nan"), time.time() - t0), flush=True)
        res["members"].append(rec)
        json.dump(res, open(out, "w"), indent=1, default=float)
    res["meta"]["seconds"] = time.time() - t0
    json.dump(res, open(out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
