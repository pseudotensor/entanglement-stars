#!/usr/bin/env python
"""Strand 2b on C8 (spec SPEC_strand2b_row.md with sec. 10's ruling; builder analyst): blocks (1,0) and (2,0) of the light
packet on the open cube, the reduced system in z = (psi, N), partial derivatives at fixed Fock vector by Slater overlaps,
Newton with Broyden updates, continuation, the lock and its pieces, the option-(a) diagnostic (fixed coefficients in the
aligned moving basis) and the sea-rotation term; enclosure and uniqueness as optional stages.  One case per call.
(version 2: the finite-difference keys fd_h1e-03, dEdb_h1e-04 etc., as in the R4 driver v8; version 1 had a KeyError in R_inf)"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import socket
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "frozen"))
import strand2a_lib as S2  # noqa: E402
import strand2b_c8_lib as C  # noqa: E402
import native_core as nc  # noqa: E402


def solve_case(lat, K, blk, m, gM, gB, Lam, k, do_enclosure=False, do_unique=False, radii=(1e-4, 1e-3, 1e-2, 1e-1)):
    t0 = time.time()
    bm = C.BlockModel(lat, m, gM, gB, Lam, blk)
    sys_ = C.SystemC8(bm, K, k)
    n = sys_.n
    rec = {"block": "%d,%d" % blk, "m": m, "gM": gM, "gB": gB, "Lambda": "inf" if not np.isfinite(Lam) else Lam, "k": k}
    z, info = C.continuation_c8(sys_)
    rec["continuation"] = {"ok": info["ok"], "path": info["path"], **({"lost_at_lam": info["lost_at_lam"]} if not info["ok"] else {})}
    if not info["ok"]:
        rec["seconds"] = time.time() - t0
        return rec
    Fz, st, src, t = sys_.F(z)
    psi, N = z[:n], z[n:]
    Q = k * np.ones(n) @ K @ (psi - 1)
    A = float(np.ones(n) @ K @ (psi - 1) / (4 * np.pi))
    Bc = float(-np.ones(n) @ K @ (N * psi - 1) / (4 * np.pi))
    Jm, _ = C.jacobian_c8(sys_, z)
    sv = np.linalg.svd(Jm, compute_uv=False)
    rec.update({"psi": psi.tolist(), "N": N.tolist(), "E": st["E"], "d0": st["d0"], "gap_block": st["gap"],
                "residual": float(np.linalg.norm(Fz)), "Q": float(Q), "A": A, "B": Bc,
                "A_over_B_minus_1": (A / Bc - 1) if Bc != 0 else None,
                "E_sel": float(-k * (psi - 1) @ K @ (N * psi - 1) + np.dot(N, src)), "source": src.tolist(),
                "sigma_min_J": float(sv[-1]), "cond_J": float(sv[0] / sv[-1]), "t_fixed_fock": t.tolist()})
    t5 = sys_.d1(psi, st, 1e-5)
    rec["d1_step_agreement_rel"] = float(np.max(np.abs(t - t5)) / max(np.max(np.abs(t)), 1e-300))
    ta = sys_.d1_fixed_coeff(psi, st)
    rec["t_fixed_coeff_aligned"] = ta.tolist()
    rec["sea_rotation_term_max"] = float(np.max(np.abs(t - ta)))
    # lock
    ones = np.ones(n)
    Kone = K @ ones
    dQ = {}
    for hb in (1e-3, 1e-4):
        qs = []
        for sb in (+1, -1):
            bb = 1 + sb * hb
            zb, ib = C.newton_c8(sys_, z, b=bb, maxit=20)
            qs.append(k * ones @ K @ (zb[:n] - bb))
        dQ["fd_h%.0e" % hb] = float((qs[0] - qs[1]) / (2 * hb))
    dFdb = np.concatenate([k * psi * Kone, k * N * Kone])
    dz = -np.linalg.solve(Jm, dFdb)
    dQ["implicit"] = float(-k * ones @ Kone + k * Kone @ dz[:n])
    R = float(Q + dQ["implicit"])

    def psi_free(bb):
        p = psi.copy()
        for _ in range(50):
            g = src - k * p * (K @ (p - bb))
            Jf = -k * (np.diag(K @ (p - bb)) + np.diag(p) @ K)
            dp = np.linalg.solve(Jf, -g)
            p = p + dp
            if np.linalg.norm(dp) < 1e-15:
                break
        return p
    hb = 1e-4
    Qf = [k * ones @ K @ (psi_free(1 + s * hb) - (1 + s * hb)) for s in (+1, -1)]
    R_free = float(Q + (Qf[0] - Qf[1]) / (2 * hb))

    def Eblock(bb):
        b = bm.build(bb * ones)
        e = np.linalg.eigvalsh(C.herm(np.einsum("y,yab->ab", ones, b["M"])))
        return float(e[0])
    Ebd = {}
    for hb in (1e-3, 1e-4):
        Ebd["dEdb_h%.0e" % hb] = (Eblock(1 + hb) - Eblock(1 - hb)) / (2 * hb)
        Ebd["d2Edb2_h%.0e" % hb] = (Eblock(1 + hb) - 2 * Eblock(1.0) + Eblock(1 - hb)) / hb ** 2
    rec["lock"] = {"Q": float(Q), "dQdb": dQ, "R": R, "R_free": R_free, "R_mat": R - R_free, "R_inf": Ebd["dEdb_h1e-04"],
                   "E_block_derivs": Ebd, "E_block_flat": Eblock(1.0)}
    if do_unique:
        rng = np.random.default_rng(11)
        sols, neg = [], 0
        for trial in range(20):
            zs = rng.uniform(0.5, 2.0, 2 * n)
            zz, inf2 = C.newton_c8(sys_, zs, maxit=40)
            if inf2["ok"]:
                if np.any(zz <= 0):
                    neg += 1
                elif not any(np.linalg.norm(zz - s) < 1e-8 for s in sols):
                    sols.append(zz)
        rec["uniqueness"] = {"distinct_positive": len(sols), "negative_converged": neg,
                             "contains_continuation_solution": bool(any(np.linalg.norm(z - s) < 1e-8 for s in sols))}
    if do_enclosure:
        Y = np.linalg.inv(Jm)
        YF = float(np.linalg.norm(Y @ Fz))
        rngE = np.random.default_rng(3)
        dirs = [np.eye(2 * n)[i] * s for i in range(2 * n) for s in (1, -1)]
        sph = [v / np.linalg.norm(v) for v in rngE.normal(size=(50, 2 * n))]
        enc = {"Y_F": YF, "norm_Y": float(np.linalg.norm(Y, 2)), "radii": {}}
        for r in radii:
            qmax = 0.0
            for v in dirs + sph:
                zp = z + r * v
                if np.any(zp <= 0):
                    continue
                Jp, _ = C.jacobian_c8(sys_, zp)
                qmax = max(qmax, float(np.linalg.norm(np.eye(2 * n) - Y @ Jp, 2)))
            enc["radii"][str(r)] = {"q": qmax, "test_lhs": YF + r * qmax, "passed": bool(YF + r * qmax < r),
                                    "positive_domain": bool(psi.min() - r > 0 and N.min() - r > 0)}
        rec["enclosure"] = enc
    rec["seconds"] = time.time() - t0
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--block", required=True, choices=["1,0", "2,0"])
    ap.add_argument("--m", type=float, required=True)
    ap.add_argument("--gM", type=float, required=True)
    ap.add_argument("--gB", type=float, required=True)
    ap.add_argument("--Lam", default="1")
    ap.add_argument("--k", type=float, required=True)
    ap.add_argument("--enclosure", action="store_true")
    ap.add_argument("--unique", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    Lam = np.inf if a.Lam == "inf" else float(a.Lam)
    lat = S2.lattice("C8")
    K, _ = nc.K_matrix(lat["positions"].tolist())
    blk = tuple(int(v) for v in a.block.split(","))
    rec = solve_case(lat, K, blk, a.m, a.gM, a.gB, Lam, a.k, a.enclosure, a.unique)
    rec["host"] = socket.gethostname()
    tag = "C8_b%s_m%g_gM%g_gB%g_L%s_k%g" % (a.block.replace(",", ""), a.m, a.gM, a.gB, a.Lam, a.k)
    with open(os.path.join(a.out, "case_%s.json" % tag), "w") as fh:
        json.dump(rec, fh, indent=1, default=float)
    print("%s done in %.1f s, ok %s" % (tag, rec["seconds"], rec["continuation"]["ok"]), flush=True)


if __name__ == "__main__":
    main()
