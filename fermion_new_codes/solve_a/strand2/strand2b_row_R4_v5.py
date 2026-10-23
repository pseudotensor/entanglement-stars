#!/usr/bin/env python
"""Strand 2b on R4 (spec solve/specs/SPEC_strand2b_row.md with secs. 9-10; builder analyst).
Version 2: library v2 (the state's projector rank fixed at the flat start's multiplet degeneracy).
Version 3: library v3 (fourth-order stencil at h = 1e-3 for d rho^P/d psi; the spec's second-order check reported).
Version 4: library v4 (speed: BLAS contractions, the direct mixed stencil for second derivatives).
Version 5: library v5 (Newton best iterate and noise-floor stop).

The reduced static system in z = (psi, N) (the state rho = P_low/d_0 of H[N, psi] on the block, eliminated): C_x = 0 and the
adjoint row (3); the c-basis Fock sector of R4 holds the state as a fixed vector, so "at fixed state" derivatives are
central differences of rho^P_x[psi] in that fixed basis.  Per case: continuation, Newton, uniqueness from 20 random starts,
the sampled enclosure on the reduced map, the lock R = Q + dQ/db (central differences and implicit), R_free, R_mat, R_inf,
and control C13 (on (1,0) and (2,0): the Slater-overlap route of sec. 10 against the c-basis route).  One case per call:
--block, --m, --gM, --gB, --Lam, --k; writes <out>/case_<tag>.json.
"""
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
import strand2b_lib_v5 as B  # noqa: E402
import native_core as nc  # noqa: E402

BLOCKS = {"1,0": (1, 0), "2,0": (2, 0), "2,1": (2, 1), "1,1": (1, 1), "0,0": (0, 0), "0,1": (0, 1)}


def tagf(a):
    return ("%s" % a).replace("inf", "inf")


def slater_route(sys_, psi, Phi_c, Nc, blk, h):
    """C13: d/dpsi_y of <Phi| rho^P_x |Phi> at fixed Fock vector via Slater overlaps onto the moved block (sec. 10),
    against the c-basis central difference and the fixed-coefficient (aligned moving basis) diagnostic.
    Phi_c: the state as a c-basis sector vector (nondegenerate case) at psi."""
    rp = sys_.rp
    lat, nW = rp.lat, rp.nW
    npp, nh = blk

    def block_basis(ps):
        light = S2.Light(lat, rp.m, ps, np.ones(rp.n))
        qb = S2.QBlocks(nW)
        tuples = qb.block(npp, nh)
        # orbital matrices: particles in tuple order, then the sea orbitals in index order (holes remove sea orbitals)
        mats = []
        for t in tuples:
            parts = [light.vp[:, i] for i in t if i < nW]
            holes = [i - nW for i in t if i >= nW]
            sea = [light.vm[:, r] for r in range(nW) if r not in holes]
            mats.append(np.array(parts + sea).T)
        return light, tuples, np.array(mats)

    def block_matrix(ps, light, tuples, x):
        # rho^P_x[ps] compressed to the block, in the Slater basis of ps, built in the c-basis and mapped by the explicit vectors
        b = rp.build(ps)
        Wb = slater_vectors(light, tuples, Nc)
        return Wb.conj().T @ b["rho"][x] @ Wb, Wb

    def slater_vectors(light, tuples, N):
        # explicit c-basis vectors of q_I^dag |Omega> with the strand 0 sign convention (QFock chain on Omega)
        from strand0_lib import QFock
        import native_sources as ns

        class Ad:
            pass
        ad = Ad()
        ad.n, ad.vp, ad.vm = nW, light.vp, light.vm
        qf = QFock(ad, rp.f.fock)
        Om, N0 = qf.vacuum()
        cols = []
        for t in tuples:
            M, Nout = qf.chain([("ad", i) for i in t], N0)
            cols.append(M @ Om)
        return np.array(cols).T

    light0, tup0, mats0 = block_basis(psi)
    W0 = slater_vectors(light0, tup0, Nc)
    c0 = W0.conj().T @ Phi_c
    out = {"norm_c0": float(np.linalg.norm(c0))}
    der_c, der_o, der_a = [], [], []
    for y in range(rp.n):
        vals_c, vals_o, vals_a = [], [], []
        for sgn in (+1, -1):
            ps = np.array(psi, float)
            ps[y] += sgn * h
            lt, tp, mats = block_basis(ps)
            # overlaps O_{I'J} = det(V_I'^dag V_J); the sign convention of the explicit vectors is checked against them
            O = np.linalg.det(np.einsum("iab,jac->ijbc", mats.conj(), mats0))
            Wp = slater_vectors(lt, tp, Nc)
            Oc = Wp.conj().T @ W0
            out.setdefault("det_vs_vector_overlap_max", 0.0)
            out["det_vs_vector_overlap_max"] = max(out["det_vs_vector_overlap_max"], float(np.max(np.abs(O - Oc))))
            cp = Oc @ c0
            Mx = [Wp.conj().T @ rp.build(ps)["rho"][x] @ Wp for x in range(rp.n)]
            vals_o.append([float(np.real(cp.conj() @ Mx[x] @ cp)) for x in range(rp.n)])
            vals_c.append([float(np.real(Phi_c.conj() @ rp.build(ps)["rho"][x] @ Phi_c)) for x in range(rp.n)])
            # fixed coefficients in the moving basis, aligned to the basis at psi by the polar part of the overlap
            Uo, _, Vh = np.linalg.svd(Oc)
            Ualign = Uo @ Vh
            ca = Ualign @ c0
            vals_a.append([float(np.real(ca.conj() @ Mx[x] @ ca)) for x in range(rp.n)])
        der_c.append([(a - b) / (2 * h) for a, b in zip(*vals_c)])
        der_o.append([(a - b) / (2 * h) for a, b in zip(*vals_o)])
        der_a.append([(a - b) / (2 * h) for a, b in zip(*vals_a)])
    der_c, der_o, der_a = map(np.array, (der_c, der_o, der_a))
    out.update({"d_cbasis": der_c.tolist(), "d_overlap": der_o.tolist(), "d_fixed_coeff_aligned": der_a.tolist(),
                "C13_rel": float(np.max(np.abs(der_o - der_c)) / max(np.max(np.abs(der_c)), 1e-300)),
                "sea_rotation_term_max": float(np.max(np.abs(der_o - der_a)))})
    return out


def solve_case(lat, K, blk, m, gM, gB, Lam, k, fops, do_enclosure=True, do_unique=True, seed_u=11, seed_e=3):
    t0 = time.time()
    npp, nh = blk
    nW = lat["nsite"] * lat["ncomp"]
    Nc = nW + npp - nh
    kexc = npp + nh
    rp = B.RhoP(lat, m, gM, gB, Lam, Nc, fops)
    sys_ = B.System(rp, K, k, kexc)
    n = rp.n
    rec = {"block": "%d,%d" % blk, "m": m, "gM": gM, "gB": gB, "Lambda": "inf" if not np.isfinite(Lam) else Lam, "k": k,
           "Nc": Nc}
    z, info = sys_.continuation()
    rec["continuation"] = {"ok": info["ok"], "path": info["path"], **({"lost_at_lam": info["lost_at_lam"]} if not info["ok"] else {})}
    if not info["ok"]:
        rec["seconds"] = time.time() - t0
        return rec
    Fz, st, src, t = sys_.F(z)
    psi, N = z[:n], z[n:]
    Q = k * np.ones(n) @ K @ (psi - 1)
    A = float(np.ones(n) @ K @ (psi - 1) / (4 * np.pi))
    Bc = float(-np.ones(n) @ K @ (N * psi - 1) / (4 * np.pi))
    Esel = float(-k * (psi - 1) @ K @ (N * psi - 1) + np.dot(N, src))
    Jm, _, mingap = sys_.J(z)
    sv = np.linalg.svd(Jm, compute_uv=False)
    rec.update({"psi": psi.tolist(), "N": N.tolist(), "E": st["E"], "d0": st["d0"], "d0_at_tolerance": st["d0_at_tolerance"],
                "multiplet_splitting": st["multiplet_splitting"], "E_multiplet_mean": st["E_multiplet_mean"], "gap_block": st["gap_block"],
                "min_level_distance_sector": mingap, "residual": float(np.linalg.norm(Fz)), "Q": float(Q), "A": A, "B": Bc,
                "A_over_B_minus_1": (A / Bc - 1) if Bc != 0 else None, "E_sel": Esel, "source": src.tolist(),
                "sigma_min_J": float(sv[-1]), "cond_J": float(sv[0] / sv[-1])})
    # FD step agreement for d rho^P/d psi
    d1, d2 = sys_.drho2(psi, 1e-4), sys_.drho2(psi, 1e-5)
    rec["drho_step_agreement_rel"] = float(np.max(np.abs(d1 - d2)) / max(np.max(np.abs(d1)), 1e-300))
    d4 = sys_.drho(psi)
    rec["drho_4th_order_vs_2nd_1e-4_rel"] = float(np.max(np.abs(d4 - d1)) / max(np.max(np.abs(d4)), 1e-300))
    # uniqueness
    if do_unique:
        rng = np.random.default_rng(seed_u)
        sols = []
        neg = 0
        for trial in range(20):
            zs = rng.uniform(0.5, 2.0, 2 * n)
            zz, inf2 = sys_.newton(zs, maxit=40)
            if inf2["ok"]:
                if np.any(zz <= 0):
                    neg += 1
                    continue
                if not any(np.linalg.norm(zz - s) < 1e-8 for s in sols):
                    sols.append(zz)
        rec["uniqueness"] = {"distinct_positive": len(sols), "negative_converged": neg,
                             "solutions": [s.tolist() for s in sols], "contains_continuation_solution":
                             bool(any(np.linalg.norm(z - s) < 1e-8 for s in sols))}
    # enclosure (sampled, numerical)
    if do_enclosure:
        Y = np.linalg.inv(Jm)
        YF = float(np.linalg.norm(Y @ Fz))
        rngE = np.random.default_rng(seed_e)
        dirs = [np.eye(2 * n)[i] * s for i in range(2 * n) for s in (1, -1)]
        sph = rngE.normal(size=(50, 2 * n))
        sph = [v / np.linalg.norm(v) for v in sph]
        enc = {"Y_F": YF, "norm_Y": float(np.linalg.norm(Y, 2)), "radii": {}}
        best = None
        for r in (1e-4, 1e-3, 1e-2, 1e-1):
            qmax = 0.0
            ok_domain = (psi.min() - r > 0) and (N.min() - r > 0)
            for v in dirs + sph:
                zp = z + r * v
                if np.any(zp <= 0):
                    continue
                Jp, stp, _ = sys_.J(zp)
                qmax = max(qmax, float(np.linalg.norm(np.eye(2 * n) - Y @ Jp, 2)))
            passed = YF + r * qmax < r
            enc["radii"][str(r)] = {"q": qmax, "test_lhs": YF + r * qmax, "passed": bool(passed), "positive_domain": bool(ok_domain)}
            if passed and ok_domain:
                best = r
        # second-derivative estimate along 20 random directions
        rngH = np.random.default_rng(seed_e + 1)
        hh = 1e-3
        d2n = 0.0
        for _ in range(20):
            v = rngH.normal(size=2 * n)
            v /= np.linalg.norm(v)
            Jp, _, _ = sys_.J(z + hh * v)
            Jn, _, _ = sys_.J(z - hh * v)
            d2n = max(d2n, float(np.linalg.norm((Jp - Jn) / (2 * hh), 2)))
        enc["D2F_estimate"] = d2n
        enc["largest_passing_r_in_domain"] = best
        enc["gap_minus_r_hessian"] = {str(r): st["gap_block"] - r * d2n for r in (1e-4, 1e-3, 1e-2, 1e-1)}
        enc["note"] = "sampled sup (axis and 50 random sphere points), numerical, not interval-certified; reduced map in z = (psi, N)"
        rec["enclosure"] = enc
    # lock: R = Q + dQ/db at b = 1
    ones = np.ones(n)
    Kone = K @ ones
    dQ = {}
    for hb in (1e-3, 1e-4):
        qs = []
        for sb in (+1, -1):
            bb = 1 + sb * hb
            zb, ib = sys_.newton(np.concatenate([psi, N]), b=bb, maxit=20)
            qs.append(k * ones @ K @ (zb[:n] - bb))
        dQ["fd_%g" % hb] = float((qs[0] - qs[1]) / (2 * hb))
    dFdb = np.concatenate([k * psi * Kone, k * N * Kone])
    dz = -np.linalg.solve(Jm, dFdb)
    dQ["implicit"] = float(-k * ones @ Kone + k * Kone @ dz[:n])
    R = float(Q + dQ["implicit"])
    # R_free: matter source frozen at its solution value; psi(b) from s_x = k psi_x [K(psi - b)]_x alone
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
    # R_inf: dE_block(b)/db at b = 1, H_exc[N = 1, psi = b 1] on the block
    def Eblock(bb):
        bld = rp.build(bb * ones)
        H = np.einsum("y,yab->ab", ones, bld["rho"])
        e, _ = B.block_states(H, bld["nexc"], kexc)
        return float(e[0])
    Ebd = {}
    for hb in (1e-3, 1e-4):
        Ebd["dEdb_%g" % hb] = (Eblock(1 + hb) - Eblock(1 - hb)) / (2 * hb)
        Ebd["d2Edb2_%g" % hb] = (Eblock(1 + hb) - 2 * Eblock(1.0) + Eblock(1 - hb)) / hb ** 2
    rec["lock"] = {"Q": float(Q), "dQdb": dQ, "R": R, "R_free": R_free, "R_mat": R - R_free, "R_inf": Ebd["dEdb_1e-4"],
                   "E_block_derivs": Ebd, "E_block_flat": Eblock(1.0)}
    # C13 on (1,0) and (2,0) when nondegenerate
    if blk in ((1, 0), (2, 0)) and st["d0"] == 1:
        Phi = st["Pl"][:, 0]
        rec["C13"] = slater_route(sys_, psi, Phi, Nc, blk, 1e-4)
    rec["seconds"] = time.time() - t0
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--block", required=True, choices=list(BLOCKS))
    ap.add_argument("--m", type=float, required=True)
    ap.add_argument("--gM", type=float, required=True)
    ap.add_argument("--gB", type=float, required=True)
    ap.add_argument("--Lam", default="1")
    ap.add_argument("--k", type=float, required=True)
    ap.add_argument("--no-enclosure", action="store_true")
    ap.add_argument("--no-unique", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    Lam = np.inf if a.Lam == "inf" else float(a.Lam)
    lat = S2.lattice("R4")
    K, _ = nc.K_matrix(nc.lattice("ring4")["positions"])
    fops = B.FockOps(2 * lat["nsite"] * lat["ncomp"])
    rec = solve_case(lat, K, BLOCKS[a.block], a.m, a.gM, a.gB, Lam, a.k, fops, not a.no_enclosure, not a.no_unique)
    rec["host"] = socket.gethostname()
    tag = "R4_b%s_m%g_gM%g_gB%g_L%s_k%g" % (a.block.replace(",", ""), a.m, a.gM, a.gB, a.Lam, a.k)
    with open(os.path.join(a.out, "case_%s.json" % tag), "w") as fh:
        json.dump(rec, fh, indent=1, default=float)
    print("%s done in %.1f s, ok %s" % (tag, rec["seconds"], rec["continuation"]["ok"]), flush=True)


if __name__ == "__main__":
    main()
