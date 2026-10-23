#!/usr/bin/env python
"""test_capture_v10_propagators_v1.py (builder analyst; engine spec 38.5 (3); a record-only check before any pod run): version 9's
Chebyshev step (cheb_expm in h_m) and scipy's expm_multiply (in h_m, and in version 10's h_m - i Gamma_out) disagreed by 5e-06 to 4e-03
in norm on one 1 r_s step of a capture packet (test_capture_v10_port_v1's timing lines), so each is compared here with an exact
reference: the dense eigendecomposition of the Hermitian h_m, psi(tau) = V exp(-i E tau) V^dagger psi_0, and for the port operator the
dense scipy.linalg.expm of -i (h_m - i Gamma_out) tau applied to psi_0 (kappa in --dense-expm-kappas only). Per kappa: the spectral radius
of h_m against the Chebyshev bound, the relative error of each propagator, the weight beyond R_max = R_N - 16 in each and in the exact
state, and the launched packet's own weight beyond R_max. The packet is chan_init's (the exterior's ingoing branch at 8 r_s, the
positive-frequency projection), on the slice's applied site fields. Every number printed by the script. Args: --fields --slice --kappas
--omega-over-m --dense-expm-kappas --json."""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import scipy.linalg as sl
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
    ap.add_argument("--omega-over-m", type=float, default=2.0)
    ap.add_argument("--kappas", type=int, nargs="*", default=[1, 100, 200])
    ap.add_argument("--dense-expm-kappas", type=int, nargs="*", default=[1])
    ap.add_argument("--gamma", type=float, default=0.2)
    ap.add_argument("--ramp", type=float, default=8.0)
    ap.add_argument("--ws", type=float, default=0.01)
    ap.add_argument("--json", required=True)
    a = ap.parse_args()
    import engine_fields_v14 as F8
    Z = np.load(a.fields, allow_pickle=True)
    R = np.asarray(Z["R"], float); ell = int(Z["ell"])
    Ns, vs, Ws, info = F8.solve_constraints_ivp(np.asarray(Z["eps_tot"][a.slice], float), np.asarray(Z["J_tot"][a.slice], float), a.g, R, ell, collar_policy="flag")
    E = float(info["E_collar"]); rs = 2 * a.g * E
    N = np.asarray(Z["N_applied"][a.slice], float); v = np.asarray(Z["v_applied"][a.slice], float)
    Rmax = float(R[-1]) - 16.0; far = np.repeat(R > Rmax, 2)
    om = a.omega_over_m * a.m; R0 = 8.0 * rs
    out = {"script": os.path.basename(__file__), "r_s": rs, "omega": om, "rows": []}
    for kappa in a.kappas:
        hs, Qs = VF.chain_sparse(R, a.m, float(kappa))
        hm = C.clocked_sparse(hs, Qs, N, v)
        gam_o = a.gamma * np.clip((R - Rmax) / a.ramp, 0.0, 1.0)
        Ho = (hm - 1j * sp.diags(np.repeat(gam_o, 2))).tocsc()
        B = C.SB(hm)
        mc = math.sqrt(a.m ** 2 + (kappa / R0) ** 2)
        k, omk, a2 = FC.branch_momenta(om, 0.1 * om, mc, float(np.interp(R0, R, v)), outgoing=False)
        psi0 = FC.build_packet(len(R), int(round(R0 - R[0])), k, a2, mc, m=a.m, a=kappa / R0)
        psi0 = psi0 - C.cheb_neg_step(B, psi0, a.ws)
        n0 = float(np.linalg.norm(psi0))
        t0 = time.time()
        Hd = hm.toarray()
        herm = float(np.max(np.abs(Hd - Hd.conj().T)))
        ev, V = np.linalg.eigh(Hd)
        t1 = time.time()
        ex = V @ (np.exp(-1j * ev * rs) * (V.conj().T @ psi0))
        ch = C.cheb_expm(B, psi0, rs)
        em = expm_multiply(-1j * hm * rs, psi0)
        eo = expm_multiply(-1j * Ho * rs, psi0)
        row = {"kappa": kappa, "bound": B.bound, "spectral_radius": float(np.max(np.abs(ev))), "hermiticity_defect": herm, "eigh_s": t1 - t0,
               "norm_psi0": n0, "far_weight_psi0": float(np.sum(np.abs(psi0[far]) ** 2)) / n0 ** 2,
               "err_cheb": float(np.linalg.norm(ch - ex)) / n0, "err_expm_multiply_hm": float(np.linalg.norm(em - ex)) / n0,
               "port_minus_exact_hermitian": float(np.linalg.norm(eo - ex)) / n0,
               "far_weight_exact": float(np.sum(np.abs(ex[far]) ** 2)) / n0 ** 2, "far_weight_cheb": float(np.sum(np.abs(ch[far]) ** 2)) / n0 ** 2,
               "far_weight_expm_hm": float(np.sum(np.abs(em[far]) ** 2)) / n0 ** 2, "far_weight_port": float(np.sum(np.abs(eo[far]) ** 2)) / n0 ** 2,
               "norm_loss_port": 1.0 - float(np.linalg.norm(eo)) ** 2 / n0 ** 2}
        if kappa in a.dense_expm_kappas:
            t2 = time.time()
            Ud = sl.expm(-1j * Ho.toarray() * rs)
            exo = Ud @ psi0
            row.update({"dense_expm_s": time.time() - t2, "err_expm_multiply_port": float(np.linalg.norm(eo - exo)) / n0,
                        "port_exact_minus_hermitian_exact": float(np.linalg.norm(exo - ex)) / n0,
                        "norm_loss_port_exact": 1.0 - float(np.linalg.norm(exo)) ** 2 / n0 ** 2})
        out["rows"].append(row)
        print(json.dumps(row), flush=True)
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
