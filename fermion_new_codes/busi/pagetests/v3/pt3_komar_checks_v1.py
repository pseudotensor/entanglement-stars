#!/usr/bin/env python
"""pt3_komar_checks_v1.py (busi; reviewer A pst1 (a), inbox #442: three checks of the rows' collar-charge shift, not blocking stage 2, required before
stage 5). Box CPU. The record P64_s4_B2n4096 at T/r_s 100 with its sources zero-padded to 8192 sites (vacuum exterior beyond 4096), the massless
k = 1 probe, a particle packet at 6 T_H (sigma = 2 T_H) added as one copy (central difference +-1) to the sources and re-solved by the field
module of record (engine_fields_v16). Reads at R_col = 16 r_s:
  M_rows(R_col) = W(R_col)/(2g)  (the rows' read, lapse normalized at the outer boundary);
  M_ren(R_col)  = int_0^R_col 4 pi r^2 (N(r)/N(R_col))^2 rho dr  (the lapse renormalized at R_col: the constraint integrated from the centre);
  the direct trace Tr(dC h_{R_col}).
Checks: (1) R_pkt -> 2 R_pkt: the shift should fall about as 1/(R_pkt - r_s); (2) the interior content halved (sources x 1/2): the shift should
halve; (3) M_ren's shift should drop to the direct trace's level (~5e-9 omega). All pass -> a Komar-type effect: the coefficient
c = dM_rows / (omega E_I / (R_pkt - r_s)) recorded (E_I the interior content at R_col, r_s = 2 g E)."""
import math
import os
import sys

import numpy as np
from scipy.integrate import quad

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_pt3_stage1 as T  # noqa: E402
import pt3_core as K  # noqa: E402
import pt3_calib as CB  # noqa: E402

NEXT = 8192


def setup(scale):
    rc = T.record()
    H = rc["H"]; FM = rc["FM"]
    n0 = len(H["R_sites"])
    R = np.arange(1, NEXT + 1, dtype=float)
    eps = np.zeros(NEXT); J = np.zeros(NEXT)
    eps[:n0] = scale * H["eps_tot"]; J[:n0] = scale * H["J_tot"]
    _, _, _, info = FM.solve_constraints_ivp(eps, J, H["g"], R, H["ell"], collar_policy="flag", support_rel=1e-14)
    Na, va, _ = FM.site_fields_kernel_average(info, R, H["ell"])
    ops = K.ChainOps(NEXT, 1.0)
    h = ops.h(Na, va)
    e, U, _ = K.full_spectrum_from_sector(h)
    return {"H": H, "FM": FM, "R": R, "eps": eps, "J": J, "N": np.asarray(Na), "v": np.asarray(va), "ops": ops, "h": h, "Phi": U[:, e < 0],
            "E": float(info["E_collar"]), "info": info}


def reads(S, de, dJ, Rcol):
    H = S["H"]; g = H["g"]
    _, _, _, info = S["FM"].solve_constraints_ivp(S["eps"] + de, S["J"] + dJ, g, S["R"], H["ell"], collar_policy="flag", support_rel=1e-14)
    W, Nd, rho = info["W_dense"], info["N_dense"], info["rho_ell"]
    Nc = float(Nd(np.array([Rcol]))[0])
    mren = quad(lambda r: 4 * math.pi * r * r * (float(Nd(np.array([r]))[0]) / Nc) ** 2 * float(rho(r)), 0, Rcol, limit=1000, epsabs=1e-13, epsrel=1e-13)[0]
    return {"M_rows": float(W(np.array([Rcol]))[0] / (2 * g)), "M_ren": mren, "E": float(info["E_collar"])}


def shift(S, x0, TH, Rcol):
    import pt_lib_v1 as L1
    ops = S["ops"]
    Nf = lambda R_: np.interp(R_, ops.R, S["N"]); vf = lambda R_: np.interp(R_, ops.R, S["v"])
    psi, inf = L1.wkb_packet(Nf, vf, 1.0, 6 * TH, 2 * TH, x0, ops.n)
    x, kept = CB.project_particle(psi, S["Phi"])
    om = float(np.real(np.vdot(x, S["h"] @ x)))
    de = ops.dens(ops.h0, x[:, None]); dJ = ops.dens(ops.Q, x[:, None])
    p = reads(S, de, dJ, Rcol); m = reads(S, -de, -dJ, Rcol)
    d = {k: (p[k] - m[k]) / 2.0 for k in p}
    direct = float(np.real(np.vdot(x, ops.h_R(S["N"], S["v"], Rcol) @ x)))
    return {"x0": x0, "omega": om, "kept": kept, "lo": inf["lo"], "hi": inf["hi"], "dM_rows_over_omega": d["M_rows"] / om, "dM_ren_over_omega": d["M_ren"] / om,
            "dE_over_omega": d["E"] / om, "direct_over_omega": direct / om}


def main():
    out_dir = sys.argv[1]
    res = {}
    full = setup(1.0)
    H = full["H"]; g = H["g"]; rsn = H["rs_nom"]
    TH = T.T_H_rec(H)
    Rcol = 16 * rsn
    bg = reads(full, 0 * full["eps"], 0 * full["J"], Rcol)
    rs_full = 2 * g * full["E"]
    a = shift(full, 2400.0, TH, Rcol); b = shift(full, 4800.0, TH, Rcol)
    pred = (2400.0 - rs_full) / (4800.0 - rs_full)
    res["check1"] = {"R_pkt": a, "2R_pkt": b, "ratio": b["dM_rows_over_omega"] / a["dM_rows_over_omega"], "predicted_ratio": pred,
                     "pass": bool(abs(b["dM_rows_over_omega"] / a["dM_rows_over_omega"] / pred - 1) < 0.15)}
    K.log("check 1: shift at %g -> %g sites: %.5f -> %.5f (ratio %.4f, 1/(R - r_s) predicts %.4f)" % (2400, 4800, a["dM_rows_over_omega"], b["dM_rows_over_omega"],
                                                                                                    res["check1"]["ratio"], pred))
    half = setup(0.5)
    c = shift(half, 2400.0, TH, Rcol)
    bgh = reads(half, 0 * half["eps"], 0 * half["J"], Rcol)
    res["check2"] = {"full": a, "half": c, "ratio": c["dM_rows_over_omega"] / a["dM_rows_over_omega"], "E_I_full": bg["M_rows"], "E_I_half": bgh["M_rows"],
                     "content_ratio": bgh["M_rows"] / bg["M_rows"], "pass": bool(abs(c["dM_rows_over_omega"] / a["dM_rows_over_omega"] / (bgh["M_rows"] / bg["M_rows"]) - 1) < 0.15)}
    K.log("check 2: content halved (E_I %.4f -> %.4f): shift %.5f -> %.5f (ratio %.4f)" % (bg["M_rows"], bgh["M_rows"], a["dM_rows_over_omega"], c["dM_rows_over_omega"],
                                                                                         res["check2"]["ratio"]))
    res["check3"] = {"dM_ren_over_omega": [a["dM_ren_over_omega"], b["dM_ren_over_omega"], c["dM_ren_over_omega"]],
                     "direct_over_omega": [a["direct_over_omega"], b["direct_over_omega"], c["direct_over_omega"]],
                     "pass": bool(max(abs(a["dM_ren_over_omega"]), abs(b["dM_ren_over_omega"]), abs(c["dM_ren_over_omega"])) < 1e-6)}
    K.log("check 3: renormalized-lapse read shift / omega %s; direct trace %s" % (res["check3"]["dM_ren_over_omega"], res["check3"]["direct_over_omega"]))
    coef = [s_["dM_rows_over_omega"] / (E_I / (s_["x0"] - rs_)) for s_, E_I, rs_ in ((a, bg["M_rows"], rs_full), (b, bg["M_rows"], rs_full), (c, bgh["M_rows"], 2 * g * half["E"]))]
    res["coefficient_c"] = coef
    res["background"] = {"E_full": full["E"], "E_half": half["E"], "M_rows_Rcol_full": bg["M_rows"], "M_rows_Rcol_half": bgh["M_rows"], "r_s_full_sites": rs_full}
    res["all_pass"] = bool(res["check1"]["pass"] and res["check2"]["pass"] and res["check3"]["pass"])
    K.log("coefficient c = dM_rows / (omega E_I/(R_pkt - r_s)) (units: sites, g = %g): %s; all pass: %s" % (g, coef, res["all_pass"]))
    K.dump_json(res, os.path.join(out_dir, "komar_checks_v1.json"))


if __name__ == "__main__":
    main()
