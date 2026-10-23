#!/usr/bin/env python
"""Strand 2b on C8 (spec SPEC_strand2b_row.md with sec. 10's ruling; builder analyst): blocks (1,0) and (2,0) of the light
packet on the open cube, the reduced system in z = (psi, N), partial derivatives at fixed Fock vector by Slater overlaps,
Newton with Broyden updates, continuation, the lock and its pieces, the option-(a) diagnostic (fixed coefficients in the
aligned moving basis) and the sea-rotation term; enclosure and uniqueness as optional stages.  One case per call.
(version 2: the finite-difference keys fd_h1e-03, dEdb_h1e-04 etc., as in the R4 driver v8; version 1 had a KeyError in R_inf)
(version 3: library v3 (the common-sea Schur overlap; the projector rank fixed at the flat start, sec. 11.1); the record carries
d0_at_tolerance, the multiplet splitting and mean)
(version 4: library v4, the Kronecker build of the (2,0) block's v_IJ, 220 times faster than v3's seed tensor, same S and M_x to 1e-15)
(version 5: library v5: Newton, the lock's implicit derivative, the uniqueness starts and the enclosure all use the FD Jacobian of F (the
total derivative); the analytic jacobian_c8's difference from it at the solution is recorded per block; the residual's noise floor is
measured at the solution and a final residual above 1e-11 is flagged with the solution still reported, as in the R4 driver v8)
(version 6: the cube's reduced enclosure design of SPEC sec. 12(b): radii {1e-3, 1e-2, 1e-1}, the 16 axes of z plus 24 random unit
directions (seed 3), 120 Jacobians; 8 Newton starts for uniqueness; per radius the smallest sampled gap; the second-derivative estimate
along 20 random directions (seed 4) and gap - r ||D^2 F|| as in the R4 driver)
(version 7, SPEC sec. 15 item 6: library v6 (Newton never executes a rejected step and catches exceptions inside, with its stop
reason and last correction; the state's cluster gap e[d_0] - e[d_0 - 1] beside the block gap; the continuation records both gaps and
sigma_min of the FD Jacobian at every accepted step); every uniqueness start records its start and returned vectors, the stop reason,
and at the returned vector its own residual, ||J^-1 F||, the smallest component and the distance from the continuation solution; a scan
with no accepted start is worded "scan inconclusive: 0 accepted starts"; the enclosure counts per radius the points skipped as
non-positive and the failed evaluations (exceptions recorded), reports the smallest sampled cluster gap beside the block gap, and
passes only with zero failed evaluations; the warm b-solves record their status; R_free by the closed form 2 Q^2/(k kappa + 2 Q)
beside the differenced value with the uniformity measures; the gap - r ||D^2 F|| column dropped (withdrawn in sec. 15 item 3).)"""
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
import strand2b_c8_lib_v6 as C  # noqa: E402
import native_core as nc  # noqa: E402


def solve_case(lat, K, blk, m, gM, gB, Lam, k, do_enclosure=False, do_unique=False, radii=(1e-3, 1e-2, 1e-1), n_random=24,
               n_starts=8):
    t0 = time.time()
    bm = C.BlockModel(lat, m, gM, gB, Lam, blk)
    sys_ = C.SystemC8v6(bm, K, k)
    n = sys_.n
    rec = {"block": "%d,%d" % blk, "m": m, "gM": gM, "gB": gB, "Lambda": "inf" if not np.isfinite(Lam) else Lam, "k": k}
    z, info = C.continuation_fd(sys_)
    rec["continuation"] = dict(info)
    if "lost_at_lam" in info:
        rec["seconds"] = time.time() - t0
        return rec
    rec["final_residual_above_1e-11"] = not info["ok"]
    Fz, st, src, t = sys_.F(z)
    psi, N = z[:n], z[n:]
    Q = k * np.ones(n) @ K @ (psi - 1)
    A = float(np.ones(n) @ K @ (psi - 1) / (4 * np.pi))
    Bc = float(-np.ones(n) @ K @ (N * psi - 1) / (4 * np.pi))
    Jm = C.jacobian_fd(sys_, z)
    Ja, _ = C.jacobian_c8(sys_, z)
    rec["analytic_jacobian_vs_fd"] = {"%s_%s" % (rn, cn): {"max_abs_diff": float(np.max(np.abs(Ja[rs, cs] - Jm[rs, cs]))),
                                                          "max_abs_fd": float(np.max(np.abs(Jm[rs, cs])))}
                                      for rn, rs in (("C_rows", slice(0, n)), ("adjoint_rows", slice(n, 2 * n)))
                                      for cn, cs in (("psi_cols", slice(0, n)), ("N_cols", slice(n, 2 * n)))}
    rngF = np.random.default_rng(5)
    rec["residual_noise_floor"] = float(max(np.linalg.norm(sys_.F(z + 1e-12 * rngF.normal(size=z.size))[0] - Fz) for _ in range(10)))
    sv = np.linalg.svd(Jm, compute_uv=False)
    rec.update({"psi": psi.tolist(), "N": N.tolist(), "E": st["E"], "d0": st["d0"], "gap_block": st["gap"], "gap_cluster": st["gap_cluster"],
                "d0_at_tolerance": st["d0_at_tolerance"], "multiplet_splitting": st["multiplet_splitting"],
                "E_multiplet_mean": st["E_multiplet_mean"],
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
    bsolves = []
    for hb in (1e-3, 1e-4):
        qs = []
        for sb in (+1, -1):
            bb = 1 + sb * hb
            zb, ib = C.newton_fd(sys_, z, b=bb, maxit=20)
            qs.append(k * ones @ K @ (zb[:n] - bb))
            bsolves.append({"b": bb, "ok": bool(ib["ok"]), "stop": ib.get("why"), "best": ib.get("best"), "iterations": len(ib.get("residuals", [])),
                            "min_component": float(np.min(zb)), "distance_from_b1_solution": float(np.linalg.norm(zb - z))})
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
    kappa = float(ones @ K @ ones)
    R_free_cf = float(2 * Q * Q / (k * kappa + 2 * Q))
    rec["lock"] = {"Q": float(Q), "dQdb": dQ, "R": R, "R_free": R_free_cf, "R_free_closed_form": R_free_cf, "R_free_differenced": R_free,
                   "R_mat": R - R_free_cf, "R_mat_differenced": R - R_free, "R_inf": Ebd["dEdb_h1e-04"],
                   "E_block_derivs": Ebd, "E_block_flat": Eblock(1.0), "kappa": kappa, "warm_b_solves": bsolves,
                   "uniformity": {"psi_rel_spread": float(np.ptp(psi) / np.mean(psi)), "source_rel_spread": float(np.ptp(src) / max(abs(np.mean(src)), 1e-300)),
                                  "K_row_sum_spread": float(np.ptp(Kone))}}
    if do_unique:
        rng = np.random.default_rng(11)
        sols, neg, nconv, starts = [], 0, 0, []
        for trial in range(n_starts):
            zs = rng.uniform(0.5, 2.0, 2 * n)
            zz, inf2 = C.newton_fd(sys_, zs, maxit=40)
            srec = {"trial": trial, "start": zs.tolist(), "returned": zz.tolist(), "ok": bool(inf2["ok"]), "stop": inf2.get("why"),
                    "best": inf2.get("best"), "last_correction_computed": inf2.get("last_correction"),
                    "iterations": len(inf2.get("residuals", [])), "error": inf2.get("error")}
            srec.update(C.returned_vector_record(sys_, zz, z))
            starts.append(srec)
            if inf2["ok"]:
                if np.any(zz <= 0):
                    neg += 1
                else:
                    nconv += 1
                    if not any(np.linalg.norm(zz - s) < 1e-8 for s in sols):
                        sols.append(zz)
        stops = {}
        for srec in starts:
            stops[srec.get("stop")] = stops.get(srec.get("stop"), 0) + 1
        wording = ("scan inconclusive: 0 accepted starts" if nconv == 0 else
                   ("no second positive solution found in %d converged starts" % nconv if len(sols) <= 1 else
                    "%d distinct positive solutions in %d converged starts" % (len(sols), nconv)))
        rec["uniqueness"] = {"starts": n_starts, "distinct_positive": len(sols), "negative_converged": neg, "converged_positive": nconv,
                             "starts_record": starts, "stop_counts": stops, "wording": wording,
                             "solutions": [s_.tolist() for s_ in sols],
                             "contains_continuation_solution": bool(any(np.linalg.norm(z - s) < 1e-8 for s in sols))}
    if do_enclosure:
        Y = np.linalg.inv(Jm)
        YF = float(np.linalg.norm(Y @ Fz))
        rngE = np.random.default_rng(3)
        dirs = [np.eye(2 * n)[i] for i in range(2 * n)]
        sph = [v / np.linalg.norm(v) for v in rngE.normal(size=(n_random, 2 * n))]
        enc = {"Y_F": YF, "norm_Y": float(np.linalg.norm(Y, 2)), "radii": {}, "design": {"radii": list(radii), "axes": 2 * n,
                                                                                     "random": n_random, "seed": 3}}
        best = None
        for r in radii:
            qmax, gmin, gcmin = 0.0, np.inf, np.inf
            ok_domain = bool(psi.min() - r > 0 and N.min() - r > 0)
            nfail, nskip, errs = 0, 0, []
            for v in dirs + sph:
                zp = z + r * v
                if np.any(zp <= 0):
                    nskip += 1
                    continue
                try:
                    Jp = C.jacobian_fd(sys_, zp)
                    stp = sys_.state(zp[:n], zp[n:])
                except Exception as e:  # noqa: BLE001  recorded; a failed evaluation fails the radius
                    nfail += 1
                    if len(errs) < 5:
                        errs.append(repr(e))
                    continue
                qmax = max(qmax, float(np.linalg.norm(np.eye(2 * n) - Y @ Jp, 2)))
                gmin = min(gmin, float(stp["gap_block"]))
                gcmin = min(gcmin, float(stp["gap_cluster"]))
            passed = bool(YF + r * qmax < r and nfail == 0)
            enc["radii"][str(r)] = {"q": qmax, "test_lhs": YF + r * qmax, "passed": passed, "positive_domain": ok_domain,
                                    "min_gap_sampled": gmin, "min_block_gap_sampled": gmin, "min_cluster_gap_sampled": gcmin,
                                    "failed_points": nfail, "skipped_nonpositive": nskip, "sampled_points": len(dirs) + len(sph),
                                    "errors": errs, "contraction_test_alone": bool(YF + r * qmax < r)}
            if passed and ok_domain:
                best = r
        rngH = np.random.default_rng(4)
        d2n = 0.0
        for _ in range(20):
            v = rngH.normal(size=2 * n)
            v /= np.linalg.norm(v)
            d2n = max(d2n, float(np.linalg.norm((C.jacobian_fd(sys_, z + 1e-3 * v) - C.jacobian_fd(sys_, z - 1e-3 * v)) / 2e-3, 2)))
        enc["D2F_estimate"] = d2n
        enc["largest_passing_r_in_domain"] = best
        enc["note"] = "sampled sup (16 axes and 24 random sphere points), numerical, not interval-certified; reduced map in z = (psi, N)"
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
