#!/usr/bin/env python
"""test_heldfield_absorber_v1.py (solve builder 5; inbox #74 solve task 4, owner decision O3 in #75: the held-field controls' absorber
qualification gate before any c15/c30 rerun with an absorber). The held-field controls hold the light channel (m 0, kappa 1 and 2) on the
frozen T/r_s 15 (or 30) record of the production, on the 8192-site chain whose sites beyond the production's 2048 carry the declared exterior
N = 1, v = -sqrt(r_s/R) (A's extension). Measured here, record-only on the box CPU: the squared return of an absorber appended beyond site
8192 (engine_capture_v11's extend_exterior and outer_absorber: L sites, Gamma_out = gamma clip((R - R_N)/L, 0, 1)^power) for an outgoing
packet of the light channel at omega = 0.5, 1 and 3 T_H (T_H = 1/(4 pi r_s), r_s = 2 g E of the record), by test_capture_v11_absorber_v1's
difference method: the same packet propagated with the absorber and on the same chain extended by --ref-append sites without one (Hermitian,
Chebyshev; its far end returns nothing within the run, checked); in [1, R_max] the two differ only by what the absorber sent back.
Printed per kappa and omega: the largest squared return over the run as a fraction of the launched weight I and of the arrived outgoing
weight (the reflection the targets 7.4e-8 (c15) and 1.5e-8 (the 10.7/a reading) are stated against), the amplitude delta = sqrt of it and
its coherent bound 2 delta + delta^2, and PASS/FAIL against each target at 1 T_H (the ruled decision point; 0.5 and 3 T_H printed).
Not measured here (they need the hold's states, so they come with the rerun or the Hermitian tail): the energy sink Tr{H, W} Delta and the
band-top bound-sector occupation at the hold's start and end. Args: --fields --T-over-rs --nbase --omega-TH --kappas --variants
(append:gamma:power) --ref-append --R-launch --sigma-omega-rel --t-end-over-rs --json."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, json, math, sys, time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine_capture_v11 as C  # noqa: E402

VF, FC = C.VF, C.FC
TARGETS = {"c15": 7.4e-8, "10.7/a": 1.5e-8}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", required=True)
    ap.add_argument("--T-over-rs", type=float, default=15.0)
    ap.add_argument("--rs-nominal", type=float, default=64.0)
    ap.add_argument("--g", type=float, default=0.25)
    ap.add_argument("--nbase", type=int, default=8192)
    ap.add_argument("--omega-TH", type=float, nargs="+", default=[0.5, 1.0, 3.0])
    ap.add_argument("--kappas", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--variants", nargs="+", default=["512:0.05:3"])
    ap.add_argument("--ref-append", type=int, default=16384)
    ap.add_argument("--R-launch", type=float, default=5000.0)
    ap.add_argument("--sigma-omega-rel", type=float, default=0.5)
    ap.add_argument("--ws", type=float, default=0.01)
    ap.add_argument("--t-end-over-rs", type=float, default=260.0)
    ap.add_argument("--json", required=True)
    a = ap.parse_args()
    import engine_fields_v14 as F8
    Z = np.load(a.fields, allow_pickle=True)
    k_rec = int(np.argmin(np.abs(np.asarray(Z["T"], float) / a.rs_nominal - a.T_over_rs)))
    R0 = np.asarray(Z["R"], float); ell = int(Z["ell"])
    Ns, vs, Ws, info = F8.solve_constraints_ivp(np.asarray(Z["eps_tot"][k_rec], float), np.asarray(Z["J_tot"][k_rec], float), a.g, R0, ell, collar_policy="flag")
    E = float(info["E_collar"]); rs = 2 * a.g * E; TH = 1.0 / (4.0 * math.pi * rs)
    N0 = np.asarray(Z["N_applied"][k_rec], float); v0 = np.asarray(Z["v_applied"][k_rec], float)
    R, N, v = C.extend_exterior(R0, N0, v0, rs, a.nbase - len(R0))          # the hold's 8192-site chain (A's extension)
    RN = float(R[-1]); Rmax = RN - 16.0
    nT = int(round(a.t_end_over_rs))
    out = {"script": os.path.basename(__file__), "record_T_over_rs": float(Z["T"][k_rec] / a.rs_nominal), "E": E, "r_s": rs, "T_H": TH,
           "R_N": RN, "R_max": Rmax, "targets": TARGETS, "args": vars(a), "rows": []}
    print("record T/r_s %.4f: E %.6f, r_s %.4f, T_H %.4e; chain %d sites (R_N %g, R_max %g); reference %d appended sites without absorber; run %d r_s"
          % (out["record_T_over_rs"], E, rs, TH, len(R), RN, Rmax, a.ref_append, nT), flush=True)
    for kappa in a.kappas:
        hs, Qs = VF.chain_sparse(R, 0.0, float(kappa)); B = C.SB(C.clocked_sparse(hs, Qs, N, v))
        Rb, Nb, vb = C.extend_exterior(R, N, v, rs, a.ref_append)
        hsb, Qsb = VF.chain_sparse(Rb, 0.0, float(kappa)); Bb = C.SB(C.clocked_sparse(hsb, Qsb, Nb, vb))
        for x in a.omega_TH:
            om = x * TH; mc = kappa / a.R_launch
            t0 = time.time()
            k, omk, a2 = FC.branch_momenta(om, a.sigma_omega_rel * om, mc, float(np.interp(a.R_launch, R, v)), outgoing=True)
            psi0 = FC.build_packet(len(R), int(round(a.R_launch - R[0])), k, a2, mc, m=0.0, a=kappa / a.R_launch)
            psi0 = psi0 - C.cheb_neg_step(B, psi0, a.ws)
            I = float(np.vdot(psi0, psi0).real)
            pb = np.concatenate([psi0, np.zeros(2 * a.ref_append, complex)]); ref = []
            for j in range(nT):
                pb = C.cheb_expm(Bb, pb, rs); ref.append(pb.reshape(-1, 2)[Rb <= Rmax].copy())
            arrived = float(np.sum(np.abs(pb.reshape(-1, 2)[Rb > RN]) ** 2)) / I
            far_end = float(np.sum(np.abs(pb.reshape(-1, 2)[Rb > RN + a.ref_append - 512]) ** 2)) / I
            print("kappa %d, omega %.2f T_H (a omega %.3e): I %.6f; arrived beyond R_N by %d r_s %.4e of I; within 512 sites of the reference's end %.2e (%.0f s)"
                  % (kappa, x, om, I, nT, arrived, far_end, time.time() - t0), flush=True)
            for var in a.variants:
                n_app, gam, pw = int(var.split(":")[0]), float(var.split(":")[1]), float(var.split(":")[2])
                t1 = time.time()
                Rx, Nx, vx = C.extend_exterior(R, N, v, rs, n_app)
                gam_o = C.outer_absorber(Rx, RN, n_app, gam, pw)
                hsx, Qsx = VF.chain_sparse(Rx, 0.0, float(kappa))
                Ho = (C.clocked_sparse(hsx, Qsx, Nx, vx) - 1j * sp.diags(np.repeat(gam_o, 2))).tocsc()
                px = np.concatenate([psi0, np.zeros(2 * n_app, complex)]); cnt = Rx <= Rmax
                tr = []
                for j in range(nT):
                    px = expm_multiply(-1j * Ho * rs, px)
                    tr.append(float(np.sum(np.abs(px.reshape(-1, 2)[cnt] - ref[j]) ** 2)) / I)
                jm = int(np.argmax(tr)); ret = tr[jm]; rel = ret / arrived if arrived > 0 else float("nan")
                dl = math.sqrt(rel) if rel == rel else float("nan")
                row = {"kappa": kappa, "omega_TH": x, "a_omega": om, "variant": var, "I": I, "arrived": arrived, "far_end": far_end,
                       "return_max_over_I": ret, "return_max_t_over_rs": jm + 1, "return_end_over_I": tr[-1], "return_over_arrived": rel,
                       "amplitude_delta": dl, "coherent_bound_2d_plus_d2": 2 * dl + dl * dl,
                       "pass": {k_: bool(rel <= t_) for k_, t_ in TARGETS.items()}, "absorbed_end": 1.0 - float(np.vdot(px, px).real) / I,
                       "trace": tr, "wall_s": time.time() - t1}
                out["rows"].append(row)
                print("  %s: squared return %.3e of I (at %d r_s), %.3e of the arrived weight; delta %.3e, 2 delta + delta^2 %.3e; against 7.4e-8 %s, 1.5e-8 %s (%.0f s)"
                      % (var, ret, jm + 1, rel, dl, 2 * dl + dl * dl, "PASS" if row["pass"]["c15"] else "FAIL", "PASS" if row["pass"]["10.7/a"] else "FAIL", row["wall_s"]), flush=True)
                json.dump(out, open(a.json, "w"), indent=1)
    at1 = [r_ for r_ in out["rows"] if abs(r_["omega_TH"] - 1.0) < 1e-12 and r_["variant"] == a.variants[0]]
    out["verdict_at_1TH"] = {"variant": a.variants[0], "pass_c15": all(r_["pass"]["c15"] for r_ in at1), "pass_10.7/a": all(r_["pass"]["10.7/a"] for r_ in at1)}
    print("VERDICT at 1 T_H for %s: c15 target %s, 10.7/a target %s" % (a.variants[0], "PASS" if out["verdict_at_1TH"]["pass_c15"] else "FAIL",
                                                                     "PASS" if out["verdict_at_1TH"]["pass_10.7/a"] else "FAIL"), flush=True)
    json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
