#!/usr/bin/env python
"""i1_assemble_v1.py (busi, foundations reviewb5 item 1 with item 9; agent A). READ script, box CPU, seconds to minutes.

Assembles the coupled-branch Jacobian I + K from the stored pieces and bounds sigma_min(I + K):
  shell space (the actual N3 loop):  K_sh = chi_sh J_solve      (i1_chi_v1 shell_chi; i1_solve_shell_v1 J_fd_h4)
  PM1 on T_d:                        K_d  = chi_d A_d^-1         (i1_chi_v1 d<d>_chi; i2_projected_v1 A_d<d>_b+0)
Sign: s = L(C - P(u(s))) gives (I + K) ds = L dC with K = (dL(P)/du)(du/ds).
For each: eigenvalues of K, sigma_min(I + K) in the stated norm (Euclidean on the retained coordinates), its enclosure for the float
matrix (verify_v1: rigorous), the error of the float matrix (chi: the rigorous per-entry bound of i1_chi_v1; J_solve / A_d: floating
estimates, the FD step pair / LV route and the table resolution), and over the neighbourhood |c|_2 <= eta of PM1:
  tau(eta) = eta sqrt(sum_j ||V_j||^2) >= ||h(u) - h(u_0)||,
  ||chi(u) - chi(u_0)||_F <= 2 tau/(pi (d - tau)^2) ||F_T||_2 ||F_U||_2          (rigorous, d the gap half-width lower bound),
  ||A(u) - A(u_0)||_2 <= eta L_A (1 + r)                                          (floating: L_A, r from i2_projected_v1),
  sigma_min(I + K(u)) >= sigma_lo(u_0) - ||K(u) - K(u_0)||, with the resolvent identities for A^-1.
Also the P'_u bounds (Heinz, resolvent integral) with the certified gap and norms. Output: <out>.json."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import verify_v1 as VF  # noqa: E402

D = "<internal>/items128"


def analyse_K(K, dK_F, label):
    n = K.shape[0]
    M = np.eye(n) + K
    ev = np.linalg.eigvals(K)
    enc = VF.sigma_min_enclosure(M)
    s = np.linalg.svd(K, compute_uv=False)
    out = {"label": label, "n": n, "K_norm2_float": float(s[0]), "K_spectral_radius": float(np.max(np.abs(ev))),
           "K_eigs_largest_abs": sorted([[float(e.real), float(e.imag)] for e in ev], key=lambda z: -abs(complex(*z)))[:6],
           "min_abs_eig_I_plus_K": float(np.min(np.abs(1 + ev))), "I_plus_K_sigma_enclosure_float_matrix": enc,
           "dK_F_bound_or_estimate": dK_F, "sigma_min_lower_incl_matrix_error": enc["lo"] - dK_F}
    return out


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "shell_d4_d3"
    chi_p = os.path.join(D, "i1_chi_v1_%s.npz" % tag)
    if not os.path.exists(chi_p):                    # preview on the partial file while the run continues
        chi_p = os.path.join(D, "i1_chi_v1_%s_partial.npz" % tag)
    C = np.load(chi_p)
    jp = os.path.join(D, "i1_chi_v1_%s.json" % tag)
    if os.path.exists(jp):
        cj = json.load(open(jp))
    else:                                            # the run's log carries the certificate lines verbatim
        import re
        L = open(os.path.join(D, "i1_chi_v1_%s.log" % tag)).read()
        m = re.search(r"delta_bwd ([0-9.e+-]+), gap half-width >= ([0-9.e+-]+)", L)
        nl = json.loads(re.search(r"norm upper bounds: (\{.*\})", L).group(1))
        dbw, dh_ = float(m.group(1)), float(m.group(2))
        mm = re.search(r"min\|e\| ([0-9.]+)", L)
        cj = {"eig_cert": {"gap_halfwidth_lower": dh_, "full_gap_lower": None, "delta_bwd_upper": dbw}, "norm_upper": nl, "n_occ": 4218,
              "shell": {"Pprime_over_heinz_max": (float(re.search(r"max ratio to Heinz ([0-9.]+)", L).group(1)) if "max ratio to Heinz" in L else None)},
              "from_log": True}
        # full gap: e_emp_min - e_occ_max >= 2 * (gap half-width) is NOT implied in general; use the stored eigenvalues
        e = np.load(os.path.join(os.path.dirname(D), "common", "pm1_background_v1.npz"))["e"]
        cj["eig_cert"]["full_gap_lower"] = float(e[e > 0].min() - e[e < 0].max()) - 2 * dbw
    cert = cj["eig_cert"]; nrm = cj["norm_upper"]
    dh = cert["gap_halfwidth_lower"]; Dl = cert["full_gap_lower"]
    res = {"version": "i1_assemble_v1", "chi": chi_p, "cert_from_log": bool(cj.get("from_log")), "gap_halfwidth_lower": dh, "full_gap_lower": Dl}
    # ------------------------------------------------------------ P'_u bounds
    h0, Q = nrm["h0"], nrm["Q"]
    res["Pprime_bounds"] = {
        "statement": "P(u) = 1_(-inf,0)(h[W^+ u]); h is affine in u, h'_u[du] = {W^+ dN, h_0}/2 + sum_i {W^+ dv_i, Q_i}/2, "
                     "||h'_u[du]|| <= ||h_0|| |W^+ dN|_inf + sum_i ||Q_i|| |W^+ dv_i|_inf",
        "h0_norm_upper": h0, "Q_norm_upper": Q,
        "first_derivative": "||P'_u[du]|| <= ||h'_u[du]|| / Delta (Heinz), Delta >= %.10f" % Dl,
        "first_derivative_per_unit_sup_N": h0 / Dl, "first_derivative_per_unit_sup_v_each": max(Q) / Dl,
        "second_derivative": "||P''[E, E]|| <= 2 ||E||^2 / (pi d^2), d >= %.10f (resolvent integral on Re z = 0)" % dh,
        "second_derivative_coeff": 2.0 / (np.pi * dh ** 2),
        "third_derivative_coeff": 1.5 / dh ** 3,
        "neighbourhood": "on {||h(u) - h(u_0)|| <= tau}, tau < d: no level crosses 0 (Weyl), occupation n_occ = %d fixed, P analytic; "
                         "||P'|| <= ||E||/(2 (d - tau)), ||P'_u - P'_u0|| <= 2 tau ||E|| / (pi (d - tau)^2)" % cj["n_occ"],
        "shell_Pprime_over_heinz_max": cj.get("shell", {}).get("Pprime_over_heinz_max")}
    # ------------------------------------------------------------ shell space
    if "shell_chi" in C.files:
        S = np.load(os.path.join(D, "i1_solve_shell_v1.npz"))
        Jh4, Jh, Jlv = S["J_fd_h4"], S["J_fd_h"], S["J_lv"]
        chi = C["shell_chi"]; err = C["shell_err"]
        K = chi @ Jh4
        dJ = np.maximum(np.abs(Jh - Jh4), np.abs(Jlv - Jh4))
        dK = float(np.linalg.norm(err) * np.linalg.norm(Jh4, 2) + VF.norm2_upper(chi) * np.linalg.norm(dJ))
        out = analyse_K(K, dK, "shell space (55 eps_R + 54 J_R), Euclidean on shell totals, J_solve = engine FD (h/4)")
        out["K_with_J_lv_rel_diff"] = float(np.linalg.norm(chi @ Jlv - K, 2) / max(np.linalg.norm(K, 2), 1e-300))
        out["J_solve_symmetry_defect_rel"] = float(np.linalg.norm(Jh4 - Jh4.T, 2) / np.linalg.norm(Jh4, 2))
        out["chi_symmetry_defect_rel"] = float(np.linalg.norm(chi - chi.T, 2) / np.linalg.norm(chi, 2))
        u = C["shell_u"]
        out["identity_uT_K_rel"] = float(np.abs(u @ K).max() / max(np.abs(u[:, None] * K).sum(axis=0).max(), 1e-300))
        nr = 55
        out["blocks_norm2"] = {"eps_from_eps": float(np.linalg.norm(K[:nr, :nr], 2)), "eps_from_J": float(np.linalg.norm(K[:nr, nr:], 2)),
                               "J_from_eps": float(np.linalg.norm(K[nr:, :nr], 2)), "J_from_J": float(np.linalg.norm(K[nr:, nr:], 2))}
        out["chi_blocks_norm2"] = {"E_N": float(np.linalg.norm(chi[:nr, :nr], 2)), "E_v": float(np.linalg.norm(chi[:nr, nr:], 2)),
                                   "J_N": float(np.linalg.norm(chi[nr:, :nr], 2)), "J_v": float(np.linalg.norm(chi[nr:, nr:], 2))}
        out["J_solve_blocks_norm2"] = {"N_eps": float(np.linalg.norm(Jh4[:nr, :nr], 2)), "N_J": float(np.linalg.norm(Jh4[:nr, nr:], 2)),
                                       "v_eps": float(np.linalg.norm(Jh4[nr:, :nr], 2)), "v_J": float(np.linalg.norm(Jh4[nr:, nr:], 2))}
        res["shell"] = out
        print(json.dumps(out, indent=1), flush=True)
    # ------------------------------------------------------------ PM1 T_d
    I2 = np.load(os.path.join(D, "i2_projected_v1.npz"))
    i2j = json.load(open(os.path.join(D, "i2_projected_v1.json")))
    for d in (4, 3, 2):
        key = "d%d_chi" % d
        if key not in C.files or ("A_d%d_b+0" % d) not in I2.files:
            continue
        chi = C[key]; err = C["d%d_err" % d]
        A = I2["A_d%d_b+0" % d]
        Ainv = np.linalg.inv(A)
        K = chi @ Ainv
        a3 = i2j["A3d"]["d%d_b+0" % d]
        a0 = a3["inv_norm_upper"]
        dA = a3["table_resolution_dA_F"]
        chin = VF.norm2_upper(chi)
        dAinv = a0 ** 2 * dA / max(1 - a0 * dA, 1e-300)
        dK = float(np.linalg.norm(err) * a0 + chin * dAinv)
        out = analyse_K(K, dK, "PM1 T_%d (Euclidean on the test values), u'_s = A_%d^-1 (compact bumps)" % (d, d))
        out["A_inv_norm_upper"] = a0; out["chi_norm2_upper"] = chin
        # the hbar identity row: Tr(hbar P'[V_j]) = 0
        out["identity_hbar_row_maxabs"] = float(np.abs(C["d%d_hbar_row" % d]).max()); out["identity_hbar_err_bound_max"] = float(C["d%d_hbar_err" % d].max())
        # neighbourhood. chi variation (rigorous given the stored float norms, with their rounding inflation):
        #   |chi_XY(u) - chi_XY(u_0)| <= 2 tau/(pi (d - tau)^2) [xi_X xi_Y + xi_X F_Y + F_X xi_Y],  xi = ||X_oe||_F at the moving P,
        #   xi(t) <= xi(0) + 2 tau F/(d - tau); xi(0)^2 = 2 ||P V (1 - P)||_F^2 (stored zf2), + the h vs h' and rounding terms.
        # A variation (floating): ||A(u) - A_0|| <= eta L_A (1 + r);  K(u) - K_0 = (chi_u - chi_0) A_u^-1 - K_0 dA A_u^-1.
        FT = C["d%d_FT" % d]; FU = C["d%d_FU" % d]; VU = C["d%d_U_norm_upper" % d]
        dbw = cj["eig_cert"].get("delta_bwd_upper", 1e-7)
        if ("d%d_zf2T" % d) in C.files:
            xT0 = np.sqrt(2 * C["d%d_zf2T" % d]) * (1 + 1e-6) + 2 * (dbw / (2 * (dh - dbw))) * FT + 1e-8 * FT
            xU0 = np.sqrt(2 * C["d%d_zf2U" % d]) * (1 + 1e-6) + 2 * (dbw / (2 * (dh - dbw))) * FU + 1e-8 * FU
            out["xi_over_F_median"] = {"T": float(np.median(xT0 / FT)), "U": float(np.median(xU0 / FU))}
        else:
            xT0, xU0 = FT.copy(), FU.copy()
            out["xi_over_F_median"] = "not stored (run 1): xi = F used"
        var = i2j.get("A_variation", {}).get("d%d" % d)
        if var is None:
            alt = os.path.join(D, "i2_projected_v1_d%dvar.json" % d)
            if os.path.exists(alt):
                var = json.load(open(alt)).get("A_variation", {}).get("d%d" % d)
        K0n = VF.norm2_upper(K) + dK
        out["K0_norm2_upper"] = K0n
        nb = []
        s0 = out["I_plus_K_sigma_enclosure_float_matrix"]["lo"] - dK
        for eta in (1e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3):
            tau = eta * float(np.sqrt((VU ** 2).sum()))
            row = {"eta": eta, "tau": tau}
            if tau >= dh:
                row["note"] = "tau >= d: outside the gapped neighbourhood"; nb.append(row); continue
            xT = xT0 + 2 * tau * FT / (dh - tau); xU = xU0 + 2 * tau * FU / (dh - tau)
            Bm = np.outer(xT, xU) + np.outer(xT, FU) + np.outer(FT, xU)
            dchi = 2 * tau / (np.pi * (dh - tau) ** 2) * float(np.linalg.norm(Bm))
            row["dchi_F_rigorous"] = dchi
            if var is not None:
                rem = max(v["half_step_ratio"] for v in var["remainder_check"].values())
                dAu = eta * var["L_A_first_order"] * (1 + max(0.0, 2.0 - rem))
                row["dA_estimate"] = dAu
                if a0 * dAu < 1:
                    Ainv_u = a0 / (1 - a0 * dAu)
                    dKu = dchi * Ainv_u + K0n * dAu * Ainv_u
                    row["dK_neighbourhood"] = dKu
                    row["dK_parts"] = {"chi": dchi * Ainv_u, "A": K0n * dAu * Ainv_u}
                    row["sigma_min_lower_on_neighbourhood"] = s0 - dKu
                else:
                    row["note"] = "a0 dA >= 1: the A-perturbation bound fails"
            nb.append(row)
        out["neighbourhood"] = nb
        res["d%d" % d] = out
        print(json.dumps(out, indent=1), flush=True)
    outp = os.path.join(D, "i1_assemble_v1_%s%s.json" % (tag, "_preview" if cj.get("from_log") else ""))
    json.dump(res, open(outp, "w"), indent=1)


if __name__ == "__main__":
    main()
