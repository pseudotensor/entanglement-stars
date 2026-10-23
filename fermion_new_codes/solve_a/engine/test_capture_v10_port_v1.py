#!/usr/bin/env python
"""test_capture_v10_port_v1.py (builder analyst; engine spec 38.5 (3), the brief's "build and test the port on the box CPU first"):
three record-only checks of engine_capture_v10's outer port on the production's slice (the applied site fields, as the capture runs):
(1) the propagators: per channel kappa, one monitor-grid step (1 r_s) of a launched capture packet by version 9's Chebyshev series in h_m
and by version 10's expm_multiply in h_m - i Gamma_out, the wall clock of each and the norm of their difference (the packet far from the
port, so the difference is the two propagators' own); (2) the port's reflection: an outgoing packet of the probe's frequency (the
exterior's outgoing branch at the launch radius, flow_chart_v4.branch_momenta(outgoing=True), version 9's packet construction and
positive-frequency projection) launched at --R-launch, propagated to --t-end-over-rs with the port (version 10's operator) and without it
(version 9's reflecting end): the weight left on [1, R_max] at the end over the launched weight is the port's reflection (band-averaged
over the packet), beside the reflecting end's; the weight on (R_max, R_N] printed; (3) the same reflection for the port variants given by
--variants (gamma:ramp pairs), a reading for the lead (the ruling fixes the inner port's construction). Every number printed by the script;
nothing is fitted. Args: --fields --slice --g --m --omega-over-m --kappas --R-launch --t-end-over-rs --variants --json."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
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
    ap.add_argument("--timing-kappas", type=int, nargs="*", default=[1, 50, 100, 200])
    ap.add_argument("--R-launch", type=float, default=1800.0)
    ap.add_argument("--t-end-over-rs", type=float, default=36.0)
    ap.add_argument("--gamma", type=float, default=0.2)
    ap.add_argument("--ramp", type=float, default=8.0)
    ap.add_argument("--variants", nargs="*", default=["0.05:16", "0.1:16"])
    ap.add_argument("--sigma-omega-rel", type=float, default=0.1)
    ap.add_argument("--ws", type=float, default=0.01)
    ap.add_argument("--json", required=True)
    a = ap.parse_args()
    import engine_fields_v14 as F8
    Z = np.load(a.fields, allow_pickle=True)
    idx = a.slice
    R = np.asarray(Z["R"], float); ell = int(Z["ell"])
    Ns, vs, Ws, info = F8.solve_constraints_ivp(np.asarray(Z["eps_tot"][idx], float), np.asarray(Z["J_tot"][idx], float), a.g, R, ell, collar_policy="flag")
    E = float(info["E_collar"]); rs = 2 * a.g * E
    N = np.asarray(Z["N_applied"][idx], float); v = np.asarray(Z["v_applied"][idx], float)
    Rmax = float(R[-1]) - 16.0
    out = {"script": os.path.basename(__file__), "slice": idx, "E": E, "r_s": rs, "R_max": Rmax, "R_N": float(R[-1]), "args": vars(a),
           "v_at_R_max": float(np.interp(Rmax, R, v)), "N_at_R_max": float(np.interp(Rmax, R, N)), "timing": [], "reflection": []}
    print("slice %d: E %.6f, r_s %.4f; R_max %g; N, v at R_max %.6f, %.6f" % (idx, E, rs, Rmax, out["N_at_R_max"], out["v_at_R_max"]), flush=True)

    def ops(kappa, gamma, ramp):
        hs, Qs = VF.chain_sparse(R, a.m, float(kappa))
        hm = C.clocked_sparse(hs, Qs, N, v)
        gam_o = gamma * np.clip((R - Rmax) / ramp, 0.0, 1.0)
        return hm, (hm - 1j * sp.diags(np.repeat(gam_o, 2))).tocsc()

    def packet(hm, kappa, om, R0, outgoing):
        mc = math.sqrt(a.m ** 2 + (kappa / R0) ** 2)
        vext = float(np.interp(R0, R, v))
        k, omk, a2 = FC.branch_momenta(om, a.sigma_omega_rel * om, mc, vext, outgoing=outgoing)
        psi0 = FC.build_packet(len(R), int(round(R0 - R[0])), k, a2, mc, m=a.m, a=kappa / R0)
        B = C.SB(hm)
        psi0 = psi0 - C.cheb_neg_step(B, psi0, a.ws)
        return B, psi0, float(k.min()), float(k.max())

    # (1) the propagators, one 1 r_s step of the capture packet (launched at 8 r_s, the ingoing branch) at omega/m 2
    om = 2.0 * a.m
    for kappa in a.timing_kappas:
        hm, Ho = ops(kappa, a.gamma, a.ramp)
        try:
            B, psi0, kmin, kmax = packet(hm, kappa, om, 8.0 * rs, outgoing=False)
        except ValueError as err:
            print("timing kappa %d: no packet (%s)" % (kappa, err), flush=True); continue
        t0 = time.time(); y1 = C.cheb_expm(B, psi0, rs); t1 = time.time()
        y2 = expm_multiply(-1j * Ho * rs, psi0); t2 = time.time()
        d = float(np.linalg.norm(y1 - y2)); nrm = float(np.linalg.norm(psi0))
        row = {"kappa": kappa, "omega_over_m": 2.0, "bound": B.bound, "cheb_s": t1 - t0, "expm_multiply_s": t2 - t1, "ratio": (t2 - t1) / (t1 - t0),
               "diff_norm_over_norm": d / nrm}
        out["timing"].append(row)
        print("timing kappa %d (omega/m 2, bound %.3f): one r_s step: Chebyshev %.3f s, expm_multiply %.3f s (ratio %.2f); |difference|/|psi| %.3e"
              % (kappa, B.bound, t1 - t0, t2 - t1, row["ratio"], d / nrm), flush=True)

    # (2), (3) the reflection of the outer port: an outgoing packet launched at R_launch
    variants = [(a.gamma, a.ramp, "the ruled port (the inner port's construction)")] + [(float(x.split(":")[0]), float(x.split(":")[1]), "variant") for x in a.variants]
    nT = int(round(a.t_end_over_rs))
    for x in a.omega_over_m:
        om = x * a.m
        for kappa in a.kappas:
            hm0 = None
            for (gamma, ramp, lab) in [(None, None, "the reflecting end (version 9)")] + variants:
                if gamma is None:
                    hs, Qs = VF.chain_sparse(R, a.m, float(kappa)); hm0 = C.clocked_sparse(hs, Qs, N, v); hm = hm0; Ho = None
                else:
                    hm, Ho = ops(kappa, gamma, ramp)
                try:
                    B, psi0, kmin, kmax = packet(hm, kappa, om, a.R_launch, outgoing=True)
                except ValueError as err:
                    print("omega/m %g kappa %d: no outgoing packet at R %g (%s)" % (x, kappa, a.R_launch, err), flush=True); break
                I = float(np.vdot(psi0, psi0).real)
                psi = psi0; t0 = time.time(); trace = []
                for j in range(nT):
                    psi = C.cheb_expm(B, psi, rs) if Ho is None else expm_multiply(-1j * Ho * rs, psi)
                    p2 = np.abs(psi.reshape(-1, 2)) ** 2
                    trace.append({"t_over_rs": j + 1, "in_counted": float(p2[R <= Rmax].sum()) / I, "in_port": float(p2[R > Rmax].sum()) / I,
                                  "outward_counted": C.outward_weight(psi, R, N, v, a.m, kappa, 100.0, Rmax, 64) / I,
                                  "inward_counted": C.inward_weight(psi, R, N, v, a.m, kappa, 100.0, Rmax, 64) / I})
                last = trace[-1]
                row = {"omega_over_m": x, "kappa": kappa, "port": lab, "gamma": gamma, "ramp": ramp, "k_range": [kmin, kmax], "I": I,
                       "end_in_counted": last["in_counted"], "end_in_port": last["in_port"], "end_inward_counted": last["inward_counted"],
                       "min_in_counted": min(r_["in_counted"] for r_ in trace), "wall_s": time.time() - t0, "trace": trace}
                out["reflection"].append(row)
                print("omega/m %g kappa %d %s (gamma %s, ramp %s): k %.4f to %.4f; at %d r_s: weight on [1, R_max] %.4e, in the port %.2e, inward-classified on [100, R_max) %.4e; the smallest weight on [1, R_max] over the run %.4e (%.0f s)"
                      % (x, kappa, lab, gamma, ramp, kmin, kmax, nT, last["in_counted"], last["in_port"], last["inward_counted"], row["min_in_counted"], row["wall_s"]), flush=True)
            json.dump(out, open(a.json, "w"), indent=1, default=float)
    json.dump(out, open(a.json, "w"), indent=1, default=float)
    print("wrote %s" % a.json, flush=True)


if __name__ == "__main__":
    main()
