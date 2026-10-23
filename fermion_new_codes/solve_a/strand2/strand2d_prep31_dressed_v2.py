#!/usr/bin/env python
"""Strand 2d sec. 7(2), stage 2 on C8 (builder analyst): the (3,1)' term of J^prep and the fixed-vacuum cost with the completion,
through the gate's q-block route (strand2a_lib.QBlocks and Dressed; S per block from the algebraic v_IJ of all three classes).

At the flat point (psi = N = 1) of the cube, coupling (g_M, g_B, Lambda): the dressed (2,0) multiplet Phi_0 (lowest d_0 levels of
Dressed.H((2,0)) by the 1e-9 tolerance); for each q in {0, pi} the ruler-perturbation amplitude z_km = <u_k| dP_+/dlam |v_m> (central
differences at h = 1e-4, as stage 1); the (3,1) vector w = Z_1^dag Phi_0 in the q-block basis (tuple (a, b, c, nW + m), amplitude
T_abc,m = z_am phi_bc + z_bm phi_ca + z_cm phi_ab), and the (1,1) vector w_vac = Z_1^dag Omega (amplitude z_km);
E_(3,1) = <w| H_exc|_(3,1) |w> averaged over the multiplet and E_(1,1) = <w_vac| H_exc|_(1,1) |w_vac>, dressed and undressed; with Br the
flat one-body matrix in the |r> basis (diagonal, the quasiparticle energies), H_exc|_blk = e^S [sum_a eps_a F_a^dag F_a] e^S with the
dressed annihilators of Dressed.H, evaluated as <u| X |u>, u = e^S w (S Hermitian), X applied through the annihilation matrices.
J^prep_(3,1)' = 2 E_(3,1), J^prep,vac = 2 E_(1,1).  The undressed values must equal stage 1's free evaluation (convention check).
Writes <out>/prep31_dressed_C8_gM<gM>_gB<gB>.json.
(version 2: the sparse annihilation matrices are converted once and cached; version 1 converted them at every call and was stopped)"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse
import json
import socket
import sys
import time

import numpy as np
import scipy.linalg as sla
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "frozen"))
import strand2a_lib as S2  # noqa: E402


def v_all(lat, light, psi, gM, gB, Lam, nW):
    if gM == 0 and gB == 0:
        return {}
    terms = S2.seed_bilinears(lat, psi, gM, gB, Lam)
    T = S2.seed_tensor(terms, 2 * nW)
    WA = S2.two_two(S2.transform(T, light.Q), nW)
    return S2.vIJ_from_WA(WA, nW)


_CSR = {}


def ann_csr(qb, j, blk):
    key = (j, blk)
    if key not in _CSR:
        r = qb.ann(j, blk)
        _CSR[key] = None if r is None else csr_matrix(r[0])
    return _CSR[key]


def expectation(qb, eS_w, emS, blk, eps, nW, dressed):
    """<u| sum_a eps_a F_a^dag F_a |u> on blk with u given (= e^S w when dressed, = w otherwise); F the (dressed) annihilators."""
    npp, nh = blk
    val = 0.0
    for part, rs, sub in (("p", range(nW), (npp - 1, nh)), ("h", range(nW, 2 * nW), (npp, nh - 1))):
        if sub[0] < 0 or sub[1] < 0:
            continue
        G = (emS[sub] @ emS[sub]) if dressed else None
        for j in rs:
            A = ann_csr(qb, j, blk)
            if A is None:
                continue
            y = A @ eS_w
            e = eps[j] if j < nW else eps[j - nW]
            val += float(np.real(np.vdot(y, G @ y) if dressed else np.vdot(y, y))) * e
    return val


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--m", type=float, default=0.1)
    ap.add_argument("--gM", type=float, required=True)
    ap.add_argument("--gB", type=float, required=True)
    ap.add_argument("--Lam", type=float, default=1.0)
    ap.add_argument("--tol", type=float, default=1e-9)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    lat = S2.lattice("C8")
    n = lat["nsite"]
    nW = n * lat["ncomp"]
    one = np.ones(n)
    light = S2.Light(lat, a.m, one, one)
    eps = light.eps
    v = v_all(lat, light, one, a.gM, a.gB, a.Lam, nW)
    blocks = ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (3, 0), (2, 1))   # dense e^{+-S} here; (3,1) only as a sparse S below
    Dr = S2.Dressed(light, v, blocks=blocks) if v else None
    qb = Dr.qb if Dr else S2.QBlocks(nW)
    for blk in blocks + ((3, 1),):
        qb.block(*blk)
    print("blocks built (%.0f s): dims %s" % (time.time() - t0, {b: qb.dim(*b) for b in blocks}), flush=True)
    Br = np.diag(eps).astype(complex)
    # flat (2,0) multiplet
    if Dr:
        H20 = Dr.H((2, 0), Br, dressed=True)
    else:
        H20 = S2.Dressed(light, {}, blocks=((2, 0), (1, 0), (0, 0))).H((2, 0), Br, dressed=False)
    w20, V20 = np.linalg.eigh(S2.herm(H20))
    d0 = int(np.sum(w20 - w20[0] < a.tol))
    C0 = V20[:, :d0]
    b20 = qb.block(2, 0)
    phis = []
    for col in range(d0):
        ph = np.zeros((nW, nW), dtype=complex)
        for (p1, p2), c in zip(b20, C0[:, col]):
            ph[p1, p2] = c
            ph[p2, p1] = -c
        phis.append(ph)
    idx31, idx11 = qb.index[(3, 1)], qb.index[(1, 1)]
    res = {"meta": {"host": socket.gethostname(), "gM": a.gM, "gB": a.gB, "Lambda": a.Lam, "m": a.m, "d0": d0,
                    "flat_levels": [float(x) for x in w20[:d0 + 3]]}, "rows": []}
    S31 = csr_matrix(qb.monomial_op(Dr.coefS, (3, 1))) if Dr else None
    if Dr:
        print("S on (3,1): nnz %d (%.0f s)" % (S31.nnz, time.time() - t0), flush=True)
    for q in (0.0, np.pi):
        x1 = lat["coord"][:, 0].astype(float)
        c = one if q == 0 else np.cos(q * x1)
        h = 1e-4
        Pp = S2.Light(lat, a.m, one + h * c, one).vp
        Pm = S2.Light(lat, a.m, one - h * c, one).vp
        dP = (Pp @ Pp.conj().T - Pm @ Pm.conj().T) / (2 * h)
        z = light.vp.conj().T @ dP @ light.vm
        # (1,1): w_vac
        wv = np.zeros(qb.dim(1, 1), dtype=complex)
        for k in range(nW):
            for mm in range(nW):
                wv[idx11[(k, nW + mm)]] = z[k, mm]
        out = {"q": q}
        for dressed in ((False, True) if Dr else (False,)):
            uv = expm_multiply(csr_matrix(Dr.S[(1, 1)]), wv) if dressed else wv
            out["E11_%s" % ("dressed" if dressed else "free")] = expectation(qb, uv, Dr.emS if Dr else None, (1, 1), eps, nW, dressed)
        # (3,1): w per member
        E31 = {"free": 0.0, "dressed": 0.0}
        norm2 = 0.0
        for ph in phis:
            w = np.zeros(qb.dim(3, 1), dtype=complex)
            for mm in range(nW):
                Z = z[:, mm]
                T = Z[:, None, None] * ph[None, :, :] + Z[None, :, None] * ph.T[:, None, :] + Z[None, None, :] * ph[:, :, None]
                for aa in range(nW):
                    for bb in range(aa + 1, nW):
                        for cc in range(bb + 1, nW):
                            w[idx31[(aa, bb, cc, nW + mm)]] = T[aa, bb, cc]
            norm2 += float(np.real(np.vdot(w, w)))
            E31["free"] += expectation(qb, w, None, (3, 1), eps, nW, False)
            if Dr:
                u = expm_multiply(S31, w)
                E31["dressed"] += expectation(qb, u, Dr.emS, (3, 1), eps, nW, True)
        out.update({"norm2_ZPhi": norm2 / d0, "E31_free": E31["free"] / d0, "J_prep_31_free": 2 * E31["free"] / d0,
                    "J_prep_vac_free": 2 * out["E11_free"]})
        if Dr:
            out.update({"E31_dressed": E31["dressed"] / d0, "J_prep_31_dressed": 2 * E31["dressed"] / d0,
                        "J_prep_vac_dressed": 2 * out["E11_dressed"],
                        "ratio_31_dressed_over_free": E31["dressed"] / E31["free"] if E31["free"] else None,
                        "ratio_vac_dressed_over_free": out["E11_dressed"] / out["E11_free"] if out["E11_free"] else None})
        res["rows"].append(out)
        print(json.dumps(out), "(%.0f s)" % (time.time() - t0), flush=True)
    res["meta"]["seconds"] = time.time() - t0
    with open(os.path.join(a.out, "prep31_dressed_C8_gM%g_gB%g.json" % (a.gM, a.gB)), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
