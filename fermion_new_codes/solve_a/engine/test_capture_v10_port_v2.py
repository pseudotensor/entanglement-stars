#!/usr/bin/env python
"""test_capture_v10_port_v2.py (builder analyst; engine spec 38.5 (3); record-only, a reading for the lane and the lead): version 1 of
this test found the ruled outer port (the inner port's construction, -i 0.2 clip((R - R_max)/8, 0, 1) on the last 16 sites) reflecting
about 20, 11, 3.6 and 0.4 percent of an arriving outgoing packet at omega/m 1.1, 1.25, 1.5 and 2 (net of the packet's own inward and
not-arrived parts). This version measures the same reflection for absorbers placed on the declared exterior appended beyond the chain
(N = 1, v = -sqrt(r_s/R) on R = R_N + 1 to R_N + --append, the form the J3' probes used beyond the chain), the counted region
[1, R_max = R_N - 16] unchanged: Gamma(R) = gamma clip((R - R_N)/L, 0, 1)^p on R > R_N (L = --append), for each --variants entry
gamma:p, and the ruled port on the unextended chain as the reference. Per probe and kappa: the weight on [1, R_max], the inward- and
outward-classified weights on [100, R_max) at every r_s (version 6's projections), and the reflection net of the packet's own inward
part and of the part not yet arrived: (inward(T) - inward(1 r_s))/(1 - inward(1 r_s) - outward(T)). Every number printed by the script.
Args: --fields --slice --omega-over-m --kappas --append --variants --t-end-over-rs --json."""
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
import engine_capture_v10 as C  # noqa: E402

VF, FC = C.VF, C.FC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", required=True)
    ap.add_argument("--slice", type=int, default=302)
    ap.add_argument("--g", type=float, default=0.25)
    ap.add_argument("--m", type=float, default=0.2)
    ap.add_argument("--omega-over-m", type=float, nargs="*", default=[1.1, 1.25, 1.5, 2.0])
    ap.add_argument("--kappas", type=int, nargs="*", default=[1, 60])
    ap.add_argument("--append", type=int, default=256)
    ap.add_argument("--variants", nargs="*", default=["0.05:2", "0.1:2", "0.1:3"])
    ap.add_argument("--R-launch", type=float, default=1800.0)
    ap.add_argument("--t-end-over-rs", type=float, default=48.0)
    ap.add_argument("--sigma-omega-rel", type=float, default=0.1)
    ap.add_argument("--ws", type=float, default=0.01)
    ap.add_argument("--json", required=True)
    a = ap.parse_args()
    import engine_fields_v14 as F8
    Z = np.load(a.fields, allow_pickle=True)
    R0_ = np.asarray(Z["R"], float); ell = int(Z["ell"])
    Ns, vs, Ws, info = F8.solve_constraints_ivp(np.asarray(Z["eps_tot"][a.slice], float), np.asarray(Z["J_tot"][a.slice], float), a.g, R0_, ell, collar_policy="flag")
    E = float(info["E_collar"]); rs = 2 * a.g * E
    N0 = np.asarray(Z["N_applied"][a.slice], float); v0 = np.asarray(Z["v_applied"][a.slice], float)
    RN = float(R0_[-1]); Rmax = RN - 16.0
    Rap = RN + np.arange(1, a.append + 1, dtype=float)
    Rx = np.concatenate([R0_, Rap]); Nx = np.concatenate([N0, np.ones(a.append)]); vx = np.concatenate([v0, -np.sqrt(rs / Rap)])
    out = {"script": os.path.basename(__file__), "r_s": rs, "R_N": RN, "R_max": Rmax, "append": a.append, "args": vars(a),
           "v_chain_end": float(v0[-1]), "v_declared_at_R_N": float(-math.sqrt(rs / RN)), "rows": []}
    print("r_s %.4f; R_N %g, R_max %g; v at the chain end %.8f against the declared -sqrt(r_s/R_N) %.8f; appended %d sites" % (
        rs, RN, Rmax, out["v_chain_end"], out["v_declared_at_R_N"], a.append), flush=True)
    nT = int(round(a.t_end_over_rs))
    for x in a.omega_over_m:
        om = x * a.m
        for kappa in a.kappas:
            cases = [("ruled port, chain unextended", R0_, N0, v0, 0.2 * np.clip((R0_ - Rmax) / 8.0, 0.0, 1.0))]
            for vv in a.variants:
                gm, pw = float(vv.split(":")[0]), float(vv.split(":")[1])
                cases.append(("appended %d, gamma %g, power %g" % (a.append, gm, pw), Rx, Nx, vx, gm * np.clip((Rx - RN) / a.append, 0.0, 1.0) ** pw))
            for lab, R, N, v, gam in cases:
                hs, Qs = VF.chain_sparse(R, a.m, float(kappa))
                hm = C.clocked_sparse(hs, Qs, N, v)
                Ho = (hm - 1j * sp.diags(np.repeat(gam, 2))).tocsc()
                B = C.SB(hm)
                mc = math.sqrt(a.m ** 2 + (kappa / a.R_launch) ** 2)
                k, omk, a2 = FC.branch_momenta(om, a.sigma_omega_rel * om, mc, float(np.interp(a.R_launch, R, v)), outgoing=True)
                psi = FC.build_packet(len(R), int(round(a.R_launch - R[0])), k, a2, mc, m=a.m, a=kappa / a.R_launch)
                psi = psi - C.cheb_neg_step(B, psi, a.ws)
                I = float(np.vdot(psi, psi).real)
                t0 = time.time(); tr = []
                for j in range(nT):
                    psi = expm_multiply(-1j * Ho * rs, psi)
                    p2 = np.abs(psi.reshape(-1, 2)) ** 2
                    tr.append({"t_over_rs": j + 1, "in_counted": float(p2[R <= Rmax].sum()) / I, "beyond_Rmax": float(p2[R > Rmax].sum()) / I,
                               "outward": C.outward_weight(psi, R, N, v, a.m, kappa, 100.0, Rmax, 64) / I,
                               "inward": C.inward_weight(psi, R, N, v, a.m, kappa, 100.0, Rmax, 64) / I})
                i1, iT, oT = tr[0]["inward"], tr[-1]["inward"], tr[-1]["outward"]
                refl = (iT - i1) / (1.0 - i1 - oT)
                row = {"omega_over_m": x, "kappa": kappa, "case": lab, "net_reflection": refl, "inward_1": i1, "inward_end": iT, "outward_end": oT,
                       "in_counted_end": tr[-1]["in_counted"], "beyond_Rmax_end": tr[-1]["beyond_Rmax"], "wall_s": time.time() - t0, "trace": tr}
                out["rows"].append(row)
                print("omega/m %g kappa %d | %s: net reflection %.3e (inward on [100, R_max) %.4e at 1 r_s, %.4e at %d r_s; outward there %.4e; weight on [1, R_max] %.4e, beyond R_max %.2e) (%.0f s)"
                      % (x, kappa, lab, refl, i1, iT, nT, oT, tr[-1]["in_counted"], tr[-1]["beyond_Rmax"], row["wall_s"]), flush=True)
                json.dump(out, open(a.json, "w"), indent=1)
    print("wrote %s" % a.json, flush=True)


if __name__ == "__main__":
    main()
