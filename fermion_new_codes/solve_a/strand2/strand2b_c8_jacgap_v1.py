#!/usr/bin/env python
"""Strand 2b sec. 13 verification (builder analyst): does the pair term 2 Re <Z_1^(y)dag Phi| H_exc|_(3,1) |Z_1^(z)dag Phi> close the
recorded difference between the FD Jacobian of F and the analytic jacobian_c8 in the adjoint rows' psi columns?

At the cube (2,0) solution of the timing case (m 0.1, (g_M, g_B, Lambda) = (0.2, 1, 1), k 1; c8timing5's record, library v5): J_FD and
J_an at the solution (strand2b_c8_lib_v5), D = J_FD - J_an; with the gate's q-block route at the solution's psi (strand2a_lib.Light,
the seed's v_IJ of all classes, Dressed with the blocks (2,0), (2,1), (3,0)): the (2,0) state Phi (lowest level of H_exc|_(2,0) with
Br = sum_x N_x X_x in the |r> basis, X_x = psi_x E_0^(1/2) P_x E_0^(1/2)); z^(y)_km = <u_k| dP_+/d psi_y |v_m> (central differences
h = 1e-4 at the solution); w_y = Z_1^(y)dag Phi in the (3,1) q-block basis; u_y = e^(S_(3,1)) w_y (S_(3,1) sparse, expm_multiply);
P[y, z] = 2 Re <u_y| X |u_z>, X = sum over particle modes a, b of Br[a, b] F_a^dag G_p F_b + the hole terms with Br^T, G = (e^(-S))^2 on
the annihilated block (Dressed.H's dressed annihilators); against D[n + y, z] (lambda = 1).  Writes <out>/jacgap.json."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import glob
import json
import sys
import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "frozen"))
import strand2a_lib as S2  # noqa: E402
import strand2b_c8_lib_v5 as C  # noqa: E402
import native_core as nc  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--record", required=True, help="the c8timing5 case JSON")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    rec = json.load(open(a.record))
    m, gM, gB, Lam, k = rec["m"], rec["gM"], rec["gB"], float(rec["Lambda"]), rec["k"]
    lat = S2.lattice("C8")
    n = lat["nsite"]
    nW = n * lat["ncomp"]
    K, _ = nc.K_matrix(lat["positions"].tolist())
    psi, N = np.array(rec["psi"]), np.array(rec["N"])
    z = np.concatenate([psi, N])
    bm = C.BlockModel(lat, m, gM, gB, Lam, (2, 0))
    sys_ = C.SystemC8(bm, K, k)
    sys_.d0_fixed = rec["d0"]
    Jfd = C.jacobian_fd(sys_, z)
    Jan, _ = C.jacobian_c8(sys_, z)
    Dg = Jfd - Jan
    print("Jacobians done (%.0f s)" % (time.time() - t0), flush=True)
    # q-block route at the solution
    light = S2.Light(lat, m, psi, np.ones(n))
    terms = S2.seed_bilinears(lat, psi, gM, gB, Lam)
    T = S2.seed_tensor(terms, 2 * nW)
    v = S2.vIJ_from_WA(S2.two_two(S2.transform(T, light.Q), nW), nW)
    Dr = S2.Dressed(light, v, blocks=((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (3, 0), (2, 1)))
    qb = Dr.qb
    qb.block(3, 1)
    ncp = lat["ncomp"]
    Xs = []
    for x in range(n):
        Px = np.zeros((nW, nW))
        Px[x * ncp:(x + 1) * ncp, x * ncp:(x + 1) * ncp] = np.eye(ncp)
        Xs.append(light.rbasis(S2.herm(psi[x] * light.pk.E0h @ Px @ light.pk.E0h)))
    Br = np.einsum("x,xab->ab", N, np.array(Xs))
    H20 = Dr.H((2, 0), Br, dressed=True)
    w20, V20 = np.linalg.eigh(S2.herm(H20))
    Phi = V20[:, 0]
    ph = np.zeros((nW, nW), dtype=complex)
    for (p1, p2), c in zip(qb.block(2, 0), Phi):
        ph[p1, p2] = c
        ph[p2, p1] = -c
    S31 = csr_matrix(qb.monomial_op(Dr.coefS, (3, 1)))
    idx31 = qb.index[(3, 1)]
    us = []
    hh = 1e-4
    for y in range(n):
        pp, pm = psi.copy(), psi.copy()
        pp[y] += hh
        pm[y] -= hh
        vpp = S2.Light(lat, m, pp, np.ones(n)).vp
        vpm = S2.Light(lat, m, pm, np.ones(n)).vp
        dP = (vpp @ vpp.conj().T - vpm @ vpm.conj().T) / (2 * hh)
        zy = light.vp.conj().T @ dP @ light.vm
        w = np.zeros(qb.dim(3, 1), dtype=complex)
        for mm in range(nW):
            Z = zy[:, mm]
            Tt = Z[:, None, None] * ph[None, :, :] + Z[None, :, None] * ph.T[:, None, :] + Z[None, None, :] * ph[:, :, None]
            for aa in range(nW):
                for bb in range(aa + 1, nW):
                    for cc in range(bb + 1, nW):
                        w[idx31[(aa, bb, cc, nW + mm)]] = Tt[aa, bb, cc]
        us.append(expm_multiply(S31, w))
    print("vectors done (%.0f s)" % (time.time() - t0), flush=True)
    # X applied: particles a, b with Br[a, b]; holes with Br[b, a]
    G = {(2, 1): Dr.emS[(2, 1)] @ Dr.emS[(2, 1)], (3, 0): Dr.emS[(3, 0)] @ Dr.emS[(3, 0)]}
    ops = {}
    for j in range(2 * nW):
        r = qb.ann(j, (3, 1))
        ops[j] = None if r is None else csr_matrix(r[0])
    Fu = [{j: (ops[j] @ u if ops[j] is not None else None) for j in range(2 * nW)} for u in us]
    P = np.zeros((n, n))
    for y in range(n):
        for zz in range(n):
            val = 0.0
            for part, rs, sub in (("p", range(nW), (2, 1)), ("h", range(nW, 2 * nW), (3, 0))):
                for aa in rs:
                    fa = Fu[y][aa]
                    if fa is None:
                        continue
                    acc = np.zeros_like(fa)
                    for bb in rs:
                        fb = Fu[zz][bb]
                        if fb is None:
                            continue
                        wgt = Br[aa, bb] if part == "p" else Br[bb - nW, aa - nW]   # Dressed.H: Br[a, b] particles, Br[b, a] holes
                        if wgt != 0:
                            acc = acc + wgt * fb
                    val += np.vdot(fa, G[sub] @ acc)
            P[y, zz] = 2 * float(np.real(val))
    blk = Dg[n:, :n]
    res = {"record": os.path.basename(a.record), "d0": rec["d0"], "flat_ok": True,
           "D_adjoint_psi": blk.tolist(), "P_pair": P.tolist(), "max_abs_D": float(np.max(np.abs(blk))),
           "max_abs_D_minus_P": float(np.max(np.abs(blk - P))), "rel_closure": float(np.max(np.abs(blk - P)) / np.max(np.abs(blk))),
           "other_blocks_max_abs_D": {"C_psi": float(np.max(np.abs(Dg[:n, :n]))), "C_N": float(np.max(np.abs(Dg[:n, n:]))),
                                      "adj_N": float(np.max(np.abs(Dg[n:, n:])))},
           "max_abs_Jfd_adj_psi": float(np.max(np.abs(Jfd[n:, :n]))), "seconds": time.time() - t0}
    print(json.dumps({k_: v_ for k_, v_ in res.items() if k_ not in ("D_adjoint_psi", "P_pair")}), flush=True)
    with open(os.path.join(a.out, "jacgap.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
