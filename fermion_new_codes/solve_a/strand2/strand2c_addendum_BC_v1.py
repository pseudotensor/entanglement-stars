#!/usr/bin/env python
"""Strand 2c addendum, parts B and C (spec solve/specs/SPEC_strand2c_addendum.md; builder analyst), with 2c's packet routine
(strand2c_mismatch_v2.packet_diag / w_field, GPU) and part A's k_a(0; aH) (kernel_L256.json).

B: the open box L = 16 (no wrapping links); psi = N = 1 on the boundary layer (any coordinate 0 or L - 1); I (ring4 or cube positions) at
   the box centre with the 2b solution's psi and N (the result's P2 cases: R4 (1,0) and (2,1) at m 0.1, (0.2, 1), Lambda 1, k 1 and 10;
   the cube cases when given); psi on the other sites the harmonic extension (Lap psi = 0 there, the 6-neighbour Laplacian with the
   interior and boundary values as Dirichlet data); N = 1 off I.  For H in {1, 0.5}: w_x on the open box; the heavy packets' smallest
   singular value of A (the one-body spectrum's minimum) against m_s; on I: w, the installed row -k_lat psi [K_I (psi - 1)] and the same
   with k_a(0; aH); Delta C and its norm ratio; the exterior profile of w by shells; sum_x w_x and the local row's Gauss charge
   -(1/4 pi) sum_x [+k_lat psi Lap(psi - 1)]_x (the open-box Laplacian).
C: the periodic box L = 16 with psi = 1 + 0.2 exp(-r^2/sigma^2) (r the minimum-image distance from the centre), sigma^2 in {4, 9}, N = 1,
   H in {1, 0.5}: the row with the lead's sign, +k_lat psi Lap(psi - 1) (and with k_a(0; aH)); ||Delta C||/||w|| over the box; the centre
   ratio w/installed; sigma H; the periodic image weight exp(-(L/2)^2/sigma^2).
Writes <out>/addBC.json."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import json
import socket
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "frozen"))
import strand2c_mismatch_v2 as M2  # noqa: E402
import native_core as nc  # noqa: E402

RING4, CUBE = M2.RING4, M2.CUBE


def weyl_open(L):
    n = L ** 3
    D = np.zeros((2 * n, 2 * n), dtype=complex)
    X = M2.coords(L)
    for j, s in enumerate(M2.SIG):
        for x in range(n):
            c = X[x].copy()
            c[j] += 1
            if c[j] >= L:
                continue
            y = M2.site(L, c)
            D[2 * x:2 * x + 2, 2 * y:2 * y + 2] += s / 2j
            D[2 * y:2 * y + 2, 2 * x:2 * x + 2] += -s / 2j
    return D


def lap_open(L, f):
    F = f.reshape(L, L, L, order="F")
    out = np.zeros_like(F)
    for ax in range(3):
        sl_p = [slice(None)] * 3
        sl_m = [slice(None)] * 3
        sl_p[ax] = slice(1, None)
        sl_m[ax] = slice(None, -1)
        d = F[tuple(sl_p)] - F[tuple(sl_m)]           # f(x + e) - f(x) on existing links
        out[tuple(sl_m)] += d
        out[tuple(sl_p)] -= d
    return out.reshape(-1, order="F")


def harmonic_extension(L, fixed_idx, fixed_val):
    """Solve Lap f = 0 on the free sites of the open box with f fixed on fixed_idx (the boundary layer and I)."""
    n = L ** 3
    free = np.ones(n, dtype=bool)
    free[fixed_idx] = False
    f = np.ones(n)
    f[fixed_idx] = fixed_val
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl
    X = M2.coords(L)
    rows, cols, vals = [], [], []
    for x in range(n):
        for j in range(3):
            for sgn in (1, -1):
                c = X[x].copy()
                c[j] += sgn
                if 0 <= c[j] < L:
                    y = M2.site(L, c)
                    rows += [x, x]
                    cols += [y, x]
                    vals += [1.0, -1.0]
    Lap = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    fi = np.where(free)[0]
    A = Lap[fi][:, fi]
    b = -Lap[fi][:, ~free] @ f[~free]
    f[fi] = spl.spsolve(A.tocsc(), b)
    return f


def k_a0(kernel_json, aH):
    K = json.load(open(kernel_json))
    R = sorted([r for r in K["equal"] if abs(r["aH"] - aH) < 1e-12], key=lambda r: r["q"])
    q = np.array([r["q"] for r in R])
    ka = np.array([r["k_a"] for r in R])
    sel = q <= aH / 4
    if sel.sum() >= 4:
        A = np.column_stack([np.ones(sel.sum()), q[sel] ** 2])
        return float(np.linalg.lstsq(A, ka[sel], rcond=None)[0][0]), "fit"
    return float(ka[0]), "smallest q"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--profiles", required=True, help="2b P2 profiles JSON (strand2c_p2_profiles_v1 output)")
    ap.add_argument("--kernel", required=True, help="part A kernel_L256.json")
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--L", type=int, default=16)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    L = a.L
    n = L ** 3
    X = M2.coords(L)
    c0 = L // 2
    res = {"meta": {"host": socket.gethostname(), "L": L, "gpu": a.gpu}, "B": [], "C": []}
    ka0 = {H: k_a0(a.kernel, H) for H in (1.0, 0.5)}
    res["meta"]["k_a0"] = {str(H): v for H, v in ka0.items()}
    # B
    Dopen = weyl_open(L)
    bnd = np.where(np.any((X == 0) | (X == L - 1), axis=1))[0]
    P2 = json.load(open(a.profiles))
    for pr in P2:
        if pr["name"] not in ("R4_b10_k1", "R4_b10_k10", "R4_b21_k1", "R4_b21_k10", "C8_b10_k1", "C8_b10_k10", "C8_b20_k1", "C8_b20_k10"):
            continue
        pos = RING4 if pr["lattice"] == "R4" else CUBE
        Iidx = [M2.site(L, (c0 + p[0], c0 + p[1], c0 + p[2])) for p in pos]
        fixed = np.concatenate([bnd, Iidx])
        fval = np.concatenate([np.ones(len(bnd)), pr["psi"]])
        psi = harmonic_extension(L, fixed, fval)
        Nf = np.ones(n)
        Nf[Iidx] = pr["N"]
        K, _ = nc.K_matrix(pos)
        for H in (1.0, 0.5):
            ms, etas = M2.masses(H)
            minsv = []
            for mm in ms:
                pd = M2.packet_diag(Dopen, np.ones(n), mm, a.gpu)
                minsv.append(float(np.sqrt(pd["minAA"])))
            wx, W0, tr, _ = M2.w_field(Dopen, psi, Nf, H, a.gpu, {}, pr["name"] + "_open")
            kl = 3 * float(sum(e * x * np.log(x) for e, x in zip(etas, np.array(ms) ** 2))) / np.pi ** 2
            pI = psi[Iidx]
            inst = -kl * pI * (K @ (pI - 1))
            inst_a = -ka0[H][0] * pI * (K @ (pI - 1))
            gauss = -float(np.sum(kl * psi * lap_open(L, psi - 1))) / (4 * np.pi)
            rec = {"profile": pr["name"], "H": H, "k_lat": kl, "k_a0": ka0[H][0], "minsv_flat_open_box": minsv, "masses": ms,
                   "w_I": wx[Iidx].tolist(), "installed_I": inst.tolist(), "installed_I_k_a0": inst_a.tolist(),
                   "DeltaC_I": (wx[Iidx] - inst).tolist(), "DeltaC_I_k_a0": (wx[Iidx] - inst_a).tolist(),
                   "rel_DeltaC": float(np.linalg.norm(wx[Iidx] - inst) / np.linalg.norm(wx[Iidx])),
                   "rel_DeltaC_k_a0": float(np.linalg.norm(wx[Iidx] - inst_a) / np.linalg.norm(wx[Iidx])),
                   "sum_w_box": float(np.sum(wx)), "sum_w_over_psi_box": float(np.sum(wx / psi)), "gauss_charge_local_row": gauss,
                   "W0": W0, "euler_minus_trace": abs(W0 - tr), "shells": M2.shells(L, Iidx, wx),
                   "psi_exterior_max_dev": float(np.max(np.abs(np.delete(psi, Iidx) - 1)))}
            res["B"].append(rec)
            print("B %s H %g: rel %.4f rel(k_a0) %.4f sum w %.4e (%.0f s)" % (pr["name"], H, rec["rel_DeltaC"], rec["rel_DeltaC_k_a0"],
                                                                            rec["sum_w_box"], time.time() - t0), flush=True)
    # C
    Dper = M2.weyl(L)
    xc = np.array([c0, c0, c0])
    dv = np.abs(X - xc)
    dv = np.minimum(dv, L - dv)
    r2 = (dv ** 2).sum(1)
    for s2 in (4.0, 9.0):
        psi = 1 + 0.2 * np.exp(-r2 / s2)
        lp = M2.lap(L, psi - 1)
        for H in (1.0, 0.5):
            ms, etas = M2.masses(H)
            kl = 3 * float(sum(e * x * np.log(x) for e, x in zip(etas, np.array(ms) ** 2))) / np.pi ** 2
            wx, W0, tr, _ = M2.w_field(Dper, psi, np.ones(n), H, a.gpu, {}, "bump%g" % s2)
            inst = kl * psi * lp
            inst_a = ka0[H][0] * psi * lp
            ic = M2.site(L, xc)
            rec = {"sigma2": s2, "H": H, "sigma_H": float(np.sqrt(s2) * H), "image_weight": float(np.exp(-(L / 2) ** 2 / s2)), "k_lat": kl,
                   "k_a0": ka0[H][0], "rel_DeltaC": float(np.linalg.norm(wx - inst) / np.linalg.norm(wx)),
                   "rel_DeltaC_k_a0": float(np.linalg.norm(wx - inst_a) / np.linalg.norm(wx)), "w_centre": float(wx[ic]),
                   "installed_centre": float(inst[ic]), "centre_ratio": float(wx[ic] / inst[ic]), "centre_ratio_k_a0": float(wx[ic] / inst_a[ic]),
                   "W0": W0, "euler_minus_trace": abs(W0 - tr)}
            res["C"].append(rec)
            print("C sigma^2 %g H %g: centre ratio %.4f (k_a0 %.4f) rel %.4f (%.0f s)" % (s2, H, rec["centre_ratio"], rec["centre_ratio_k_a0"],
                                                                                       rec["rel_DeltaC"], time.time() - t0), flush=True)
    res["meta"]["seconds"] = time.time() - t0
    with open(os.path.join(a.out, "addBC.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
