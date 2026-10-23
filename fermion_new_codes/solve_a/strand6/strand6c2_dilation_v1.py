#!/usr/bin/env python
"""Strand 6 (c2) (spec solve/specs/SPEC_strand6_validation.md (c2); builder analyst): the open-cube dilation identity on the 2b cube
solutions, P14's eq. 4 for the reduced action F = -k u^T K v + R + F_m.

The 2b system (strand2b_c8_lib_v5.SystemC8): the (2,0) block at (g_M, g_B, Lambda) = (0.2, 1, 1), m = 0.1, the installed exterior K
(native_core.K_matrix on the cube's positions), the projector rank fixed at the case's flat-start value; its rows F(z) = (C, G) are
C_x = src_x - k psi_x [K (psi - 1)]_x and G_y = k (K 1)_y + sum_x N_x t[y, x] - k N_y [K (psi - 1)]_y - k [K (N psi)]_y (lambda = b = 1),
the gradient of F = -k u^T K v + k 1^T K 1 + F_m, F_m = sum_x N_x src_x(psi) with t[y, x] = d src_x/d psi_y at the fixed Fock vector
(2b's sec. 10 route), u = psi - 1, v = N psi - 1, R = 0 (the installed row carries no vacuum remainder on the cube).
The generator: centred coordinates X = positions - centroid; zero-extended differences on the 2 x 2 x 2 cube (f = 0 off the cube), two
stencils: centred (grad_i f)(x) = (f(x + e_i) - f(x - e_i))/2 (the builder's primary reading) and forward (grad_i f)(x) = f(x + e_i) - f(x);
D = sum_i X_i grad_i - diag(sum_i X_i grad_i 1) (so D 1 = 0), T = D + I/2, Z = T^T K + K T; the tangent delta psi = T u + 1/2,
delta chi = T v + 1/2, delta N = (delta chi - N delta psi)/psi.
Pieces, each evaluated separately: LHS = (k/2) 1^T K (u + v); the anomaly -k u^T Z v; D_a F_m by 2b's fixed-Fock partials,
delta N . src + sum_{x,y} N_x t[y, x] delta psi_y; D_a F_m independently, by central differences of the matter energy E (the selected
level of H = sum_x N_x M_x(psi) in the moving block; d0 = 1 here) along the tangent (h = 1e-4, 5e-5, Richardson), whose difference from
the partials measures any first-order contribution of the (3,1)' component of the fixed state (spec 2d sec. 7(2) puts it at second order);
the residual contraction rho = C . delta N + G . delta psi.  The identity: LHS = -k u^T Z v + D_a F_m - rho (exact algebra of the rows);
closures with each D_a F_m.  Cases: the two enclosure records (k = 1 and 1e4, the continuation solutions), the two further k = 1 solution
classes of 2b sec. 14 (the representatives from c8unique1post), and the source-off vacuum (psi = N = 1, lambda = 0).
Writes <out>/dilation_c2.json."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import json
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(HERE, "..", "strand2", "frozen"), os.path.join(HERE, "..", "strand2"), os.path.join(HERE, "frozen"), HERE):
    sys.path.insert(0, _p)
import strand2a_lib as S2  # noqa: E402
import strand2b_c8_lib_v5 as C  # noqa: E402
import native_core as nc  # noqa: E402


def generators(pos):
    n = len(pos)
    X = pos - pos.mean(0)
    idx = {tuple(np.rint(p).astype(int)): i for i, p in enumerate(pos)}
    out = {}
    for name in ("centred", "forward"):
        G = []
        for i in range(3):
            M = np.zeros((n, n))
            for x in range(n):
                c = np.rint(pos[x]).astype(int)
                cp, cm = c.copy(), c.copy()
                cp[i] += 1
                cm[i] -= 1
                yp, ym = idx.get(tuple(cp)), idx.get(tuple(cm))
                if name == "centred":
                    if yp is not None:
                        M[x, yp] += 0.5
                    if ym is not None:
                        M[x, ym] -= 0.5
                else:
                    if yp is not None:
                        M[x, yp] += 1.0
                    M[x, x] -= 1.0
            G.append(M)
        D = sum(np.diag(X[:, i]) @ G[i] for i in range(3))
        D = D - np.diag(D @ np.ones(n))
        out[name] = D
    return out


def pieces(sys_, K, z, D, lam=1.0, hs=(1e-4, 5e-5)):
    n = sys_.n
    k = sys_.k
    psi, N = z[:n], z[n:]
    Fz, st, src, t = sys_.F(z, lam)
    Cr, Gr = Fz[:n], Fz[n:]
    u, v = psi - 1, N * psi - 1
    T = D + 0.5 * np.eye(n)
    Z = T.T @ K + K @ T
    dpsi = T @ u + 0.5
    dchi = T @ v + 0.5
    dN = (dchi - N * dpsi) / psi
    one = np.ones(n)
    lhs = 0.5 * k * one @ K @ (u + v)
    anomaly = -k * u @ Z @ v
    dFm_partials = lam * (dN @ src + N @ (t.T @ dpsi))   # sum_x N_x sum_y t[y, x] dpsi_y
    rho = Cr @ dN + Gr @ dpsi
    tau = np.concatenate([dpsi, dN])

    def Em(zz):
        s = sys_.state(zz[:n], zz[n:])
        return lam * (s["E"] if s["d0"] == 1 else s["E_multiplet_mean"])
    fd = []
    for h in hs:
        fd.append((Em(z + h * tau) - Em(z - h * tau)) / (2 * h))
    dFm_fd = (fd[1] * (hs[0] / hs[1]) ** 2 - fd[0]) / ((hs[0] / hs[1]) ** 2 - 1)
    E0 = Em(z)
    Fm_value = lam * float(N @ src)
    return {"LHS": float(lhs), "anomaly": float(anomaly), "DFm_partials": float(dFm_partials), "DFm_fd": float(dFm_fd),
            "DFm_fd_steps": [float(x) for x in fd], "residual_contraction": float(rho),
            "closure_partials": float(lhs - (anomaly + dFm_partials - rho)), "closure_fd": float(lhs - (anomaly + dFm_fd - rho)),
            "DFm_fd_minus_partials": float(dFm_fd - dFm_partials), "residual_norm": float(np.linalg.norm(Fz)),
            "E_level": float(E0), "F_m_value": Fm_value, "E_minus_F_m": float(E0 - Fm_value), "d0": int(st["d0"]),
            "d0_at_tolerance": int(st["d0_at_tolerance"]), "D_one_max": float(np.max(np.abs(D @ one))),
            "tangent_norm": float(np.linalg.norm(tau))}


def build(rec):
    lat = S2.lattice("C8")
    K, _ = nc.K_matrix(lat["positions"].tolist())
    K = np.asarray(K, float)
    bm = C.BlockModel(lat, rec["m"], rec["gM"], rec["gB"], float(rec["Lambda"]), tuple(int(x) for x in rec["block"].split(",")))
    sys_ = C.SystemC8(bm, K, rec["k"])
    sys_.d0_fixed = rec["d0"]
    return lat, K, sys_


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--records", required=True, help="directory with case_C8_b20_m0.1_gM0.2_gB1_L1_k{1,10000}.json")
    ap.add_argument("--post", required=True, help="c8unique1post/post_C8_b20_m0.1_gM0.2_gB1_L1_k1.json")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    res = {"meta": {}, "cases": []}
    cases = []
    for kk in (1, 10000):
        rec = json.load(open(os.path.join(a.records, "case_C8_b20_m0.1_gM0.2_gB1_L1_k%d.json" % kk)))
        cases.append(("continuation solution, k = %g" % rec["k"], rec, np.concatenate([rec["psi"], rec["N"]])))
    P = json.load(open(a.post))
    rec1 = json.load(open(os.path.join(a.records, "case_C8_b20_m0.1_gM0.2_gB1_L1_k1.json")))
    for j, o in enumerate(P["orbits"]):
        if o["is_continuation_solution"]:
            continue
        cases.append(("k = 1 solution class %d (2b sec. 14 orbit %d)" % (j + 1, j + 1), rec1, np.array(o["rep"])))
    lat0 = S2.lattice("C8")
    pos = np.asarray(lat0["positions"], float)
    gens = generators(pos)
    res["meta"]["positions"] = pos.tolist()
    res["meta"]["D"] = {k_: v_.tolist() for k_, v_ in gens.items()}
    for label, rec, z in cases:
        lat, K, sys_ = build(rec)
        row = {"case": label, "k": rec["k"], "psi": z[:8].tolist(), "N": z[8:].tolist()}
        for name, D in gens.items():
            row[name] = pieces(sys_, K, z, D)
        res["cases"].append(row)
        print(label, json.dumps({nm: {k_: row[nm][k_] for k_ in ("LHS", "anomaly", "DFm_partials", "DFm_fd", "residual_contraction",
                                                                   "closure_partials", "closure_fd")} for nm in gens}), flush=True)
    # the source-off vacuum
    lat, K, sys_ = build(rec1)
    z0 = np.ones(16)
    row = {"case": "source-off vacuum (psi = N = 1, lambda = 0), k = 1", "k": rec1["k"]}
    for name, D in gens.items():
        row[name] = pieces(sys_, K, z0, D, lam=0.0)
    res["cases"].append(row)
    print("vacuum", json.dumps({nm: {k_: row[nm][k_] for k_ in ("LHS", "anomaly", "DFm_partials", "DFm_fd", "residual_contraction",
                                                                  "closure_partials")} for nm in gens}), flush=True)
    res["meta"]["seconds"] = time.time() - t0
    with open(os.path.join(a.out, "dilation_c2.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
