#!/usr/bin/env python
"""test_capture_v11_absorber_v1.py (builder analyst; the lane's ruling on engine spec 38.5 (3), gate (a): "your outgoing-packet
reflection test at each probe frequency (omega/m 1.1, 1.25, 1.5, 2; kappa 1 and 60, and the list's largest kappa) returning below 1e-3
of the outgoing weight, net of the packet's own inward and not-yet-arrived parts"; record-only, box CPU). The return is measured by
difference, so the packet's own inward and not-yet-arrived parts cancel exactly: an outgoing packet of the probe's frequency (the
exterior's outgoing branch at --R-launch, flow_chart_v4.branch_momenta(outgoing=True), version 9's packet construction and
positive-frequency projection, both on the production's chain) is propagated (a) on engine_capture_v11's extended chain with its
absorber (v11's extend_exterior and outer_absorber, expm_multiply one r_s per step) and (b) on the same chain extended by --ref-append
sites of the declared exterior with no absorber (Chebyshev in the Hermitian h_m; its far end returns nothing to R_max within the run,
the clock printed); both start from the same state with zeros on the appended sites. In [1, R_max] the two differ only by what the
absorber sent back, so the return at T is ||P_[1, R_max] (psi_a(T) - psi_b(T))||^2 / I; printed at every r_s, its largest value over
the run, and that over the weight the reference has carried beyond R_N by the end (the arrived outgoing weight). The ruled 16-site port
(version 10's) is measured the same way on request (--with-v10-port). The largest kappa per probe is read from a version 9 JSON
(--v9-json: the channel list's largest kappa). Every number printed by the script. Args: --fields --slice --omega-over-m --kappas
--v9-json --variants (append:gamma:power) --ref-append --R-launch --t-end-over-rs --with-v10-port --json."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import json
import math
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine_capture_v11 as C  # noqa: E402

VF, FC = C.VF, C.FC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", required=True)
    ap.add_argument("--slice", type=int, default=302)
    ap.add_argument("--g", type=float, default=0.25)
    ap.add_argument("--m", type=float, default=0.2)
    ap.add_argument("--omega-over-m", type=float, required=True)
    ap.add_argument("--kappas", type=int, nargs="*", default=[1, 60])
    ap.add_argument("--v9-json", default=None, help="add the version 9 run's largest kappa for this probe")
    ap.add_argument("--variants", nargs="+", required=True, help="append:gamma:power")
    ap.add_argument("--ref-append", type=int, default=4096)
    ap.add_argument("--R-launch", type=float, default=1800.0)
    ap.add_argument("--t-end-over-rs", type=float, default=60.0)
    ap.add_argument("--with-v10-port", action="store_true")
    ap.add_argument("--sigma-omega-rel", type=float, default=0.1)
    ap.add_argument("--ws", type=float, default=0.01)
    ap.add_argument("--json", required=True)
    a = ap.parse_args()
    import engine_fields_v14 as F8
    Z = np.load(a.fields, allow_pickle=True)
    R = np.asarray(Z["R"], float); ell = int(Z["ell"])
    Ns, vs, Ws, info = F8.solve_constraints_ivp(np.asarray(Z["eps_tot"][a.slice], float), np.asarray(Z["J_tot"][a.slice], float), a.g, R, ell, collar_policy="flag")
    E = float(info["E_collar"]); rs = 2 * a.g * E
    N = np.asarray(Z["N_applied"][a.slice], float); v = np.asarray(Z["v_applied"][a.slice], float)
    RN = float(R[-1]); Rmax = RN - 16.0
    om = a.omega_over_m * a.m
    kappas = list(a.kappas)
    kmax_src = None
    if a.v9_json:
        d9 = json.load(open(a.v9_json))
        for r_ in d9["results"]:
            if r_["background"].startswith("solved") and r_["massive"] and abs(r_["massive"]["e"] - a.omega_over_m) < 1e-9:
                kmax_src = max(c["kappa"] for c in r_["channels"])
        if kmax_src is not None and kmax_src not in kappas:
            kappas.append(kmax_src)
    nT = int(round(a.t_end_over_rs))
    out = {"script": os.path.basename(__file__), "r_s": rs, "R_N": RN, "R_max": Rmax, "omega_over_m": a.omega_over_m, "kappas": kappas,
           "kappa_max_from_v9": kmax_src, "args": vars(a), "rows": []}
    print("r_s %.4f; R_N %g, R_max %g; omega/m %g; kappas %s (the version 9 list's largest %s); reference: %d appended sites without absorber, its far end at R %g"
          % (rs, RN, Rmax, a.omega_over_m, kappas, kmax_src, a.ref_append, RN + a.ref_append), flush=True)
    for kappa in kappas:
        hs, Qs = VF.chain_sparse(R, a.m, float(kappa))
        hm = C.clocked_sparse(hs, Qs, N, v)
        B = C.SB(hm)
        mc = math.sqrt(a.m ** 2 + (kappa / a.R_launch) ** 2)
        k, omk, a2 = FC.branch_momenta(om, a.sigma_omega_rel * om, mc, float(np.interp(a.R_launch, R, v)), outgoing=True)
        psi0 = FC.build_packet(len(R), int(round(a.R_launch - R[0])), k, a2, mc, m=a.m, a=kappa / a.R_launch)
        psi0 = psi0 - C.cheb_neg_step(B, psi0, a.ws)
        I = float(np.vdot(psi0, psi0).real)
        # the reference: the declared exterior appended without absorber (Hermitian; Chebyshev)
        Rb, Nb, vb = C.extend_exterior(R, N, v, rs, a.ref_append)
        hsb, Qsb = VF.chain_sparse(Rb, a.m, float(kappa)); Bb = C.SB(C.clocked_sparse(hsb, Qsb, Nb, vb))
        t0 = time.time()
        pb = np.concatenate([psi0, np.zeros(2 * a.ref_append, complex)])
        ref = []
        for j in range(nT):
            pb = C.cheb_expm(Bb, pb, rs)
            ref.append(pb.reshape(-1, 2).copy())
        cnt_b = Rb <= Rmax
        arrived = float(np.sum(np.abs(ref[-1][Rb > RN]) ** 2)) / I
        far_end = float(np.sum(np.abs(ref[-1][Rb > RN + a.ref_append - 256]) ** 2)) / I
        print("kappa %d: k %.4f to %.4f; I %.6f; the reference's weight beyond R_N at %d r_s %.4e (arrived outgoing weight), within 256 sites of its far end %.2e (%.0f s)"
              % (kappa, float(k.min()), float(k.max()), I, nT, arrived, far_end, time.time() - t0), flush=True)
        cases = [("v11 append %d, gamma %g, power %g" % (int(x.split(":")[0]), float(x.split(":")[1]), float(x.split(":")[2])),
                  int(x.split(":")[0]), float(x.split(":")[1]), float(x.split(":")[2])) for x in a.variants]
        if a.with_v10_port:
            cases.insert(0, ("version 10's ruled port (16 sites, gamma 0.2, ramp 8)", 0, 0.2, None))
        for lab, n_app, gam, pw in cases:
            t1 = time.time()
            if n_app == 0:
                Rx, Nx, vx = R, N, v
                gam_o = gam * np.clip((R - Rmax) / 8.0, 0.0, 1.0)
            else:
                Rx, Nx, vx = C.extend_exterior(R, N, v, rs, n_app)
                gam_o = C.outer_absorber(Rx, RN, n_app, gam, pw)
            hsx, Qsx = VF.chain_sparse(Rx, a.m, float(kappa))
            Ho = (C.clocked_sparse(hsx, Qsx, Nx, vx) - 1j * sp.diags(np.repeat(gam_o, 2))).tocsc()
            px = np.concatenate([psi0, np.zeros(2 * (len(Rx) - len(R)), complex)])
            cnt_x = Rx <= Rmax
            tr = []
            for j in range(nT):
                px = expm_multiply(-1j * Ho * rs, px)
                dx = px.reshape(-1, 2)[cnt_x] - ref[j][cnt_b]
                tr.append({"t_over_rs": j + 1, "return": float(np.sum(np.abs(dx) ** 2)) / I, "absorbed": 1.0 - float(np.vdot(px, px).real) / I})
            rmax_ = max(tr, key=lambda r_: r_["return"])
            row = {"kappa": kappa, "case": lab, "append": n_app, "gamma": gam, "power": pw, "I": I, "arrived": arrived, "return_max": rmax_["return"],
                   "return_max_t_over_rs": rmax_["t_over_rs"], "return_end": tr[-1]["return"], "return_max_over_arrived": rmax_["return"] / arrived if arrived > 0 else None,
                   "absorbed_end": tr[-1]["absorbed"], "wall_s": time.time() - t1, "trace": tr}
            out["rows"].append(row)
            print("kappa %d | %s: the return on [1, R_max] largest %.3e of I at %d r_s (%.3e of the arrived outgoing weight), %.3e at %d r_s; absorbed by the end %.4e (%.0f s)"
                  % (kappa, lab, rmax_["return"], rmax_["t_over_rs"], row["return_max_over_arrived"] if arrived > 0 else float("nan"), tr[-1]["return"], nT, tr[-1]["absorbed"], row["wall_s"]), flush=True)
            json.dump(out, open(a.json, "w"), indent=1)
    print("wrote %s" % a.json, flush=True)


if __name__ == "__main__":
    main()
