#!/usr/bin/env python
"""pt3_findings_diag_v1.py (busi, PAGE v3 stage 1 findings for the lead/reviewer A, HOLD of stage 2). Box CPU, minutes. Two diagnostics:
(a) the rows' collar charge shift: the packet's sources split into its energy density alone (dJ = 0) and its momentum density alone (deps = 0),
    read through engine_fields_v16 at +-1 copy (central difference), M(R) = W(R)/(2g) at R_col = 16 r_s and at R = 8, 24 r_s, for particle packets at
    6 T_H centred at x0 = 2000, 2400, 2800 sites; the direct trace Tr(dC h_{R_col}) beside;
(b) the inverted content of every stored engine-sea channel at T/r_s 100 (the slice's e_k and mask_k: occupied with e > 0, empty with e < 0)."""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_pt3_stage1 as T  # noqa: E402
import pt3_core as K  # noqa: E402
import pt3_calib as CB  # noqa: E402


def main():
    out_dir = sys.argv[1]
    rc = T.record()
    H = rc["H"]; Z = rc["Z"]
    import pt_lib_v1 as L1
    ops = K.ChainOps(len(H["R_sites"]), 1.0)
    h_avg = ops.h(rc["N_avg"], rc["v_avg"])
    e, U, _ = K.full_spectrum_from_sector(h_avg)
    Phi = U[:, e < 0]
    TH = T.T_H_rec(H)
    rs = H["rs_nom"]
    Nf = lambda R: np.interp(R, ops.R, rc["N_avg"]); vf = lambda R: np.interp(R, ops.R, rc["v_avg"])
    FM = rc["FM"]
    reads = [8 * rs, 16 * rs, 24 * rs]
    res = {"a": [], "b": {}}

    def solveM(de, dJ):
        _, _, _, info = FM.solve_constraints_ivp(H["eps_tot"] + de, H["J_tot"] + dJ, H["g"], H["R_sites"], H["ell"], collar_policy="flag", support_rel=1e-14)
        W = info["W_dense"]
        return {"E": float(info["E_collar"]), **{"M%g" % (R / rs): float(W(np.array([R]))[0] / (2 * H["g"])) for R in reads},
                "N_centre": float(info["N_dense"](np.array([0.0]))[0])}
    for x0 in (2000.0, 2400.0, 2800.0):
        psi, inf = L1.wkb_packet(Nf, vf, 1.0, 6 * TH, 2 * TH, x0, ops.n)
        x, kept = CB.project_particle(psi, Phi)
        target = float(np.real(np.vdot(x, h_avg @ x)))
        d_eps = ops.dens(ops.h0, x[:, None]); d_J = ops.dens(ops.Q, x[:, None])
        row = {"x0": x0, "x0_rs": x0 / rs, "target": target, "lo": inf["lo"], "hi": inf["hi"],
               "direct_Tr_dC_h_Rcol": float(np.real(np.vdot(x, ops.h_R(rc["N_avg"], rc["v_avg"], 16 * rs) @ x)))}
        for lab, (de, dJ) in {"both": (d_eps, d_J), "eps_only": (d_eps, 0 * d_J), "J_only": (0 * d_eps, d_J)}.items():
            p = solveM(de, dJ); m = solveM(-de, -dJ)
            row[lab] = {k: (p[k] - m[k]) / 2 for k in p}
        row["dM_Rcol_over_omega"] = row["both"]["M16"] / target
        res["a"].append(row)
        K.log("x0 %.0f (%.1f r_s): target %.4e; dE %.4e; dM(8, 16, 24 r_s) %s; eps-only dM16 %.3e, J-only dM16 %.3e; dN_centre %.3e; direct %.1e" % (
            x0, x0 / rs, target, row["both"]["E"], [round(row["both"]["M%g" % r] / target, 4) for r in (8, 16, 24)], row["eps_only"]["M16"],
            row["J_only"]["M16"], row["both"]["N_centre"], row["direct_Tr_dC_h_Rcol"]))
    for k in [int(x) for x in Z["kappas"]]:
        ek = np.asarray(Z["e_%d" % k], float)
        mk = np.unpackbits(np.asarray(Z["mask_%d" % k]))[: len(ek)].astype(bool)
        res["b"][str(k)] = {"occupied_positive": int(np.sum(mk & (ek > 0))), "empty_negative": int(np.sum(~mk & (ek < 0))),
                            "min_abs_e_over_TH": float(np.min(np.abs(ek)) / TH), "n_within_10TH": int(np.sum(np.abs(ek) < 10 * TH))}
    K.log("engine sea channels at T/r_s 100: %s" % res["b"])
    K.dump_json(res, os.path.join(out_dir, "findings_diag_v1.json"))


if __name__ == "__main__" and len(sys.argv) == 2:
    main()


def paper_charge(out_dir):
    """(c) the paper's retained-charge form M'(r) = 4 pi r^2 (rho + s w J), w = |v| (eq dr-charge, reviewer B pc9 1.3), both signs s, with the solve's
    smoothed sources rho_ell, J_ell, against the rows' W/(2g): the background at the collar (which form gives E) and the packet's change at
    R_col = 16 r_s (central difference at +-1 copy), particle at 6 T_H, x0 = 2400."""
    from scipy.integrate import quad
    rc = T.record()
    H = rc["H"]
    import pt_lib_v1 as L1
    ops = K.ChainOps(len(H["R_sites"]), 1.0)
    h_avg = ops.h(rc["N_avg"], rc["v_avg"])
    e, U, _ = K.full_spectrum_from_sector(h_avg)
    Phi = U[:, e < 0]
    TH = T.T_H_rec(H); rs = H["rs_nom"]; g = H["g"]
    Nf = lambda R: np.interp(R, ops.R, rc["N_avg"]); vf = lambda R: np.interp(R, ops.R, rc["v_avg"])
    psi, _ = L1.wkb_packet(Nf, vf, 1.0, 6 * TH, 2 * TH, 2400.0, ops.n)
    x, _ = CB.project_particle(psi, Phi)
    target = float(np.real(np.vdot(x, h_avg @ x)))
    d_eps = ops.dens(ops.h0, x[:, None]); d_J = ops.dens(ops.Q, x[:, None])
    FM = rc["FM"]

    def forms(sc, R):
        _, _, _, info = FM.solve_constraints_ivp(H["eps_tot"] + sc * d_eps, H["J_tot"] + sc * d_J, g, H["R_sites"], H["ell"], collar_policy="flag", support_rel=1e-14)
        rho, J, v, Nd, W = info["rho_ell"], info["J_ell"], info["v_dense"], info["N_dense"], info["W_dense"]
        f = lambda r, s: 4 * np.pi * r * r * (float(rho(r)) + s * abs(float(v(np.array([r]))[0])) * float(J(r)))
        fN = lambda r: 4 * np.pi * r * r * float(Nd(np.array([r]))[0]) ** 2 * float(rho(r))
        o = {"W_over_2g": float(W(np.array([R]))[0] / (2 * g)), "rows_N2rho": quad(fN, 0, R, limit=800)[0],
             "plus_wJ": quad(lambda r: f(r, 1.0), 0, R, limit=800)[0], "minus_wJ": quad(lambda r: f(r, -1.0), 0, R, limit=800)[0], "E": float(info["E_collar"])}
        return o
    bg = forms(0.0, float(H["R_sites"][-1]))
    p = forms(1.0, 16 * rs); m = forms(-1.0, 16 * rs)
    d = {k: (p[k] - m[k]) / 2 / target for k in p}
    res = {"background_at_domain_edge": bg, "packet_change_at_Rcol_over_omega": d, "target": target}
    K.log("(c) background at the edge: %s; packet's change at R_col / omega: %s" % (bg, d))
    K.dump_json(res, os.path.join(out_dir, "findings_diag_v1_paper_charge.json"))


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "c":
    paper_charge(sys.argv[1])
