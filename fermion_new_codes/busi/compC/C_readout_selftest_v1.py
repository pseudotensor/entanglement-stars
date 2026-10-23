#!/usr/bin/env python
"""C_readout_selftest_v1.py (the busi lane, Computation C): a manufactured 5-record coupling-on json, its s11 npz, a 5-record light-follow
json and a 5-record 1-field json exercising every readout of C_readout_v1.py (2, 3, 4, 5, 7, the also-printed list, the not-built listing
with a --slice-dir). The manufactured data satisfy every identity the readout recomputes by the same arithmetic, so each identity's residual
must be <= 1e-12; the clocks are T/r_s 0, 8, 16, 24, 32 (tau 256, r_s 32) so that the table rows 0, formation (16), 16, 24, 32 exist and 20,
28, 36, 40 are reported absent; the last record carries a negative mass-function node and a manufactured stop key so those paths print.
Then keys are removed one at a time and the readout must stop with MissingKey naming the key. Exit 0 and PASS when everything holds."""
import copy
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import C_readout_v1 as RO  # noqa: E402

TOL = 1e-12
N = 5
TAU = 256.0
RS = 32.0
ELL = 4.0
SRC = ["1", "2"]
GRD = ["3"]
CH = SRC + GRD
W = {c: 16.0 * float(c) for c in CH}
COLLARS = ["8", "16", "24"]
BINS = ("(-inf,-1)", "(-1,0)", "(0,1)", "(1,inf)")
FIRST_RETURN = {"from_0_rs_to_end": 82.17, "from_0_rs_back_to_8_rs": 129.8, "from_0_rs_back_to_16_rs": 123.6, "from_0_rs_back_to_24_rs": 117.0,
                "from_2_rs_to_end": 79.15, "from_2_rs_back_to_8_rs": 126.8, "from_2_rs_back_to_16_rs": 120.5, "from_2_rs_back_to_24_rs": 114.0,
                "note": "manufactured"}


def base_args(**over):
    a = {"m": 0.2, "g": 0.25, "rs": RS, "nsites": 2048, "ell": 4, "prep_ell": 4, "tau": TAU, "T_over_rs": 32.0, "reference": "R2", "mix": "newton",
         "s11_collars": "8,16,24", "light_kappas": "1,2", "light_guard_kappas": "3", "light_coupling": "on", "light_transport": "banded",
         "slices_c128": "0,16,24", "slices_light_c128": "8", "slice_dir": "<internal>/states", "tag": "selftest", "out": "x", "gpu": 0}
    a.update(over)
    return a


def cert(k, c):
    return {"pieces": k + int(c), "cholesky_calls": 2 * k, "certified": True, "why": None, "d_min_certified": 1.3e-3 / float(c), "norm_dK_G": 0.01 * k,
            "smallest_piece": (None if k == 0 else 0.1), "d_min_effective": 3e-7}


def manufacture():
    T = [k * TAU for k in range(N)]
    Trs = [t / RS for t in T]
    E = [60.0 - 0.01 * k - 0.001 * k * k for k in range(N)]
    Ep = [e * (1.0 + 2e-5) for e in E]
    minN = [0.4, 0.2, -0.002, -0.01, -0.005]
    Hin = [54.0 - 0.3 * k for k in range(N)]
    # the s11 npz arrays (33 nodes at r = 0..32; the cell r <= ell; the last record with a negative node at r = 30)
    r_nodes = np.tile(np.arange(33, dtype=float), (N, 1))
    M = np.array([0.001 * (k + 1) * r_nodes[k] for k in range(N)])
    M[4, 30] = -0.001
    rho = np.array([1e-3 * (1 + k) * np.exp(-r_nodes[k] / 8.0) for k in range(N)])
    mask = r_nodes[0] > ELL
    # the formation (trapping zeros from record 2)
    tzC = {k: [18.7 + 0.1 * k, 21.5 + 0.2 * k] for k in range(2, N)}
    tzP = {k: [18.7 + 0.1 * k + 0.003, 21.5 + 0.2 * k - 0.004] for k in range(2, N)}
    # ---------------- the coupling-on run
    recsC = []
    I0h = [1e-7 * (1 + 0.1 * k) for k in range(N)]
    Ieh = [0.9e-7 * (1 + 0.1 * k) for k in range(N)]
    Hl, EinvL, I0l, Iel = [], [], [], []
    for k in range(N):
        chans = {}
        for c in CH:
            w = W[c]
            Es = -1294.0 - 2.0 * float(c) + 0.001 * k
            Er = Es - (1e-5 * k + 1e-13) / float(c)
            Ei = Es - Er
            ie = 1e-8 * (k + 1) * (1.0 if c != "2" else -0.5)
            eps = 1e-4 * k / float(c)
            ch = {"sourcing": c in SRC, "w": w, "E_inv": Ei, "E_sea": Es, "E_ref": Er, "E_inv_weighted": w * Ei, "I_E_cut_weighted": w * ie,
                  "H_in_weighted": w * (Ei * 0.99), "eps_sum": eps, "J_sum": 0.9e-4 * k / float(c), "eps_sum_weighted": w * eps,
                  "mu": 4e-4 * float(c), "gap": 1.4e-3 * float(c), "n_plus": 1024, "n_pocket": 0, "pockets_occupied_positive": 0, "holes_empty_negative": 0,
                  "fallback": False, "why": None, "cut_crossing": False, "mode": "mu S selector, certified walk (%d pieces)" % (k + int(c)), "reanchored": False,
                  "min_abs_e": 9.8e-4 * float(c), "threshold_radius": None, "certified": cert(k, c),
                  "nocc_deviation_bins": {b: (i - 1.5) * 1e-5 * k / float(c) for i, b in enumerate(BINS)},
                  "beyond": {cl: {"number": -1e-11 * k, "energy_density_route": 1e-8 * k / float(cl), "weighted_energy_density_route": w * 1e-8 * k / float(cl)} for cl in COLLARS}}
            chans[c] = ch
        E_inv_light = 0.0
        E_inv_s = 0.0
        I_all = 0.0
        I_s = 0.0
        H_l = 0.0
        eps_fed = 0.0
        eps_all = 0.0
        J_fed = 0.0
        for c in CH:
            ch = chans[c]
            E_inv_light += ch["E_inv_weighted"]
            I_all += ch["I_E_cut_weighted"]
            H_l += ch["H_in_weighted"]
            eps_all += ch["eps_sum_weighted"]
        for c in SRC:
            ch = chans[c]
            E_inv_s += ch["E_inv_weighted"]
            I_s += ch["I_E_cut_weighted"]
            eps_fed += ch["eps_sum_weighted"]
            J_fed += ch["w"] * ch["J_sum"]
        g_E = 0.0
        g_I = 0.0
        g_eps = 0.0
        for c in GRD:
            g_E += chans[c]["E_inv_weighted"]
            g_I += chans[c]["I_E_cut_weighted"]
            g_eps += chans[c]["eps_sum_weighted"]
        Hl.append(H_l)
        EinvL.append(E_inv_light)
        I0l.append(I_all)
        Iel.append(0.9 * I_all)
        rec = {"T": T[k], "T_over_rs": Trs[k], "E": E[k], "E_over_E0": E[k] / E[0], "E_dust_shells": 72.0 + k, "E_pairs_shells": -0.01 * k,
               "R_t": (tzC[k][1] if k in tzC else None), "r_in": (tzC[k][0] if k in tzC else None), "trapping_zeros": (tzC[k] if k in tzC else []),
               "min_N_minus_absv": minN[k], "r_min_N_minus_absv": 100.0 - 20 * k, "M1_I_E": I0h[k], "H_in": Hin[k], "continuation_label": "verified (Weyl)",
               "slice_saved": ("<internal>/orbitals_selftest_T%07.2f.npz" % Trs[k]) if Trs[k] in (0.0, 16.0, 24.0) else None,
               "solver": {"W_tail": {"E_collar": E[k]}}}
        s11 = {"T": T[k], "T_over_rs": Trs[k], "E": E[k], "M_min_outside_cell": float(M[k][mask].min()), "r_at_M_min": float(r_nodes[k][mask][int(np.argmin(M[k][mask]))]),
               "negative_points_outside_cell": int(np.count_nonzero(M[k][mask] < 0)), "compactness_max_2gM_over_r": 0.2 + 0.1 * k,
               "source_min_outside_cell": 1e-20 - 1e-14 * k, "r_at_source_min": 236.0, "rho_ell_min_outside_cell": float(rho[k][mask].min()),
               "ball_source_min_outside_cell": 1e-25, "pairs_source_min_outside_cell": -1e-7 * k, "light_source_min_outside_cell": -3e-8 * k,
               "light_source_integral": eps_fed, "pairs_source_integral": -0.01 * k, "ball_source_integral": 72.0 + k, "light_fed": True,
               "light_certified_on_record": True, "continuation_label": "verified (Weyl)", "N_centre": 0.7 - 0.1 * k}
        if k == N - 1:
            s11["stop"] = {"case": "(ii) manufactured for the self-test", "T_over_rs": Trs[k]}
        rec["s11"] = s11
        lv = {"coupling": "on", "fed": True, "transport": "banded", "label": "light reference: certified (mu S walk) on every record", "certified_on_record": True,
              "channels": chans, "E_inv_light": E_inv_light, "E_inv_light_sourcing": E_inv_s, "E_inv_light_over_E": E_inv_light / E[k],
              "E_inv_light_sourcing_over_E": E_inv_s / E[k], "E_inv_heavy": 54.0 - 0.3 * k, "I_E_light": I_all, "I_E_light_sourcing": I_s, "H_in_light": H_l,
              "eps_fed_sum": eps_fed, "J_fed_sum": J_fed, "eps_light_all_sum": eps_all,
              "truncation_guard": {"E_inv_guard_over_sourcing": (None if E_inv_s == 0 else g_E / E_inv_s), "I_E_guard_over_sourcing": (None if I_s == 0 else g_I / I_s),   # busi fix: the manufactured record 0 has 0 sourcing (the engine's real record 0 is tiny, not 0); null as for eps
                                   "eps_guard_over_sourcing": (None if eps_fed == 0 else g_eps / eps_fed)},
              "threshold_radius_massless": {c: None for c in CH}, "timing": {"walk_pieces": 3 * k, "walk_cholesky": 4 * k, "walk_fallbacks": 0, "cut_crossings": 0},
              "light_slice_saved": ("<internal>/light_selftest_T%07.2f.npz" % Trs[k]) if Trs[k] == 8.0 else None}
        if k >= 1:
            X1 = -1.2e6 - k
            X2 = X1 - 40.0 - 0.5 * k
            Y1 = X1 - 100.0
            Y2 = Y1 - 40.4 - 0.5 * k
            fw = (Y2 - Y1) - (X2 - X1)
            evt = TAU * (I0h[k - 1] + Ieh[k]) / 2.0
            eve = evt * 1.001
            dH = Hin[k] - Hin[k - 1]
            rr = dH - eve - fw
            rec["M1_parts"] = {"X1": X1, "X2": X2, "Y1": Y1, "Y2": Y2, "I_E_end_prev_interval": Ieh[k], "I_E_start": I0h[k], "cut": 2038, "I_E_mid": None}
            rec["M1_balance"] = {"dHin": dH, "evolution_exact": eve, "evolution_trapezoid": evt, "field_work": fw, "reference_reset": rr, "closure": 0.0, "cut_changed": False,
                                 "per_unit_time": {"dHin": dH / TAU, "evolution": eve / TAU, "field_work": fw / TAU, "reference_reset": rr / TAU}}
            rec["dE_dT"] = (E[k] - E[k - 1]) / TAU
            rec["dHin_dT"] = (Hin[k] - Hin[k - 1]) / TAU
            rec["balance_d_E_minus_Hin_dT"] = rec["dE_dT"] - rec["dHin_dT"]
            lX1 = -1.24e5 - 0.01 * k
            lX2 = lX1 - 0.0001 - 1e-5 * k
            lY1 = lX1 - 2.4
            lY2 = lY1 - 0.00012 - 1e-5 * k
            lfw = (lY2 - lY1) - (lX2 - lX1)
            levt = TAU * (I0l[k - 1] + Iel[k]) / 2.0
            leve = levt * 1.001
            ldH = Hl[k] - Hl[k - 1]
            lrr = ldH - leve - lfw
            lv["M1_parts"] = {"X1": lX1, "X2": lX2, "Y1": lY1, "Y2": lY2, "I_E_end_prev_interval": Iel[k], "I_E_start": I0l[k], "cut": 2038}
            lv["M1_balance"] = {"dHin": ldH, "evolution_exact": leve, "evolution_trapezoid": levt, "field_work": lfw, "reference_reset": lrr, "closure": 0.0,
                                "per_unit_time": {"dHin": ldH / TAU, "evolution": leve / TAU, "field_work": lfw / TAU, "reference_reset": lrr / TAU}}
            lv["dE_inv_light_dT"] = (EinvL[k] - EinvL[k - 1]) / TAU
        else:
            rec["M1_parts"] = {"X1": -1.2e6, "X2": -1.2e6 - 40.0, "Y1": 0.0, "Y2": 0.0, "I_E_end_prev_interval": 0.0, "I_E_start": I0h[0], "cut": 235, "I_E_mid": None}
            lv["M1_parts"] = {"X1": -1.24e5, "X2": -1.24e5, "Y1": 0.0, "Y2": 0.0, "I_E_end_prev_interval": 0.0, "I_E_start": I0l[0], "cut": 235}
        rec["light_v36"] = lv
        recsC.append(rec)
    headC = {"E0": E[0], "N_B": 320.0,
             "light_v36": {"channels": {c: {"kappa": int(c), "w": W[c], "sourcing": c in SRC, "n_sea": 2048, "n_plus": 1024, "mu0": 4e-4 * float(c), "gap_mu0": 1.4e-3 * float(c),
                                            "min_abs_e": 9.8e-4 * float(c), "selection_equals_negative_set": True, "n_sea_equals_nsites": True,
                                            "commutator_h_S_max_abs": 0.0, "h_m_real_part_max": 0.0} for c in CH},
                           "coupling": "on", "transport": "banded",
                           "subcritical_T0": {"min_N_minus_absv_sites": 0.4893, "min_N_minus_absv_dense": minN[0], "r_min": 104.3, "zeros": []},
                           "first_return": dict(FIRST_RETURN), "chain_end_over_rs": 64.0}}
    runC = {"script": "manufactured (self-test)", "args": base_args(), "head": headC, "records": recsC}
    # ---------------- the light-follow (on the production's clocks and values)
    recsF = []
    for k in range(N):
        chans = {}
        for c in CH:
            Es = -642.0 - float(c) + 0.0005 * k
            Er = Es - (1.003e-5 * k + 1e-13) / float(c)
            Ei = Es - Er
            ie = 1e-8 * (k + 1) * (0.98 if c != "2" else -0.45)   # busi fix: -0.49 with the weights 16 and 32 cancelled the sourcing current exactly (16 x 0.98 = 32 x 0.49), leaving its identity untested
            chans[c] = {"sourcing": c in SRC, "E_inv": Ei, "E_sea": Es, "E_ref": Er, "I_E_cut": ie, "H_in": Ei * 0.98, "cut": 1014, "eps_sum": 1.002e-4 * k / float(c),
                        "J_sum": 0.9e-4 * k / float(c) * 1.001, "mu": 8e-4 * float(c), "gap": 2.7e-3 * float(c), "n_plus": 512, "n_pocket": 0,
                        "pockets_occupied_positive": 0, "holes_empty_negative": 0, "fallback": False, "why": None, "cut_crossing": False,
                        "mode": "mu S selector, certified walk (%d pieces)" % k, "min_abs_e": 1.9e-3 * float(c), "certified": cert(k, c),
                        "nocc_deviation_bins": {b: (i - 1.5) * 1.004e-5 * k / float(c) for i, b in enumerate(BINS)},
                        "beyond": {cl: {"number": 1e-11 * k, "energy_density_route": 1.01e-8 * k / float(cl)} for cl in COLLARS}}
        E_s = 0.0
        E_all = 0.0
        I_s = 0.0
        I_all = 0.0
        eps_fed = 0.0
        J_fed = 0.0
        for c in SRC:
            E_s += W[c] * chans[c]["E_inv"]
            I_s += W[c] * chans[c]["I_E_cut"]
            eps_fed += W[c] * chans[c]["eps_sum"]
            J_fed += W[c] * chans[c]["J_sum"]
        for c in CH:
            E_all += W[c] * chans[c]["E_inv"]
            I_all += W[c] * chans[c]["I_E_cut"]
        g_E = 0.0
        g_I = 0.0
        for c in GRD:
            g_E += W[c] * chans[c]["E_inv"]
            g_I += W[c] * chans[c]["I_E_cut"]
        recsF.append({"T": T[k], "T_over_rs": Trs[k], "E": Ep[k], "R_t": (tzP[k][1] if k in tzP else None), "min_N_minus_absv": minN[k], "channels": chans,
                      "E_inv_light_sourcing": E_s, "E_inv_light_all": E_all, "E_rad0_over_E": E_s / Ep[k], "I_E_light_sourcing": I_s, "I_E_light_all": I_all,
                      "eps_fed_sum": eps_fed, "J_fed_sum": J_fed, "truncation_guard": {"E_inv_guard_over_sourcing": (None if E_s == 0 else g_E / E_s), "I_E_guard_over_sourcing": (None if I_s == 0 else g_I / I_s)},   # busi fix: null where the manufactured record 0's sourcing is 0
                      "timing": {"walk_pieces": k, "walk_cholesky": k, "walk_fallbacks": 0, "cut_crossings": 0}, "seconds": 1.0})
    headF = {"light_v36": {c: {"kappa": int(c), "w": W[c], "sourcing": c in SRC, "n_sea": 1024} for c in CH}, "subcritical_T0": {"min_N_minus_absv_sites": 0.48932}}
    runF = {"script": "manufactured (self-test, light-follow)", "args": base_args(nsites=1024, mix="plain", light_follow="/manufactured/production"), "followed": "/manufactured/production",
            "label": "manufactured", "head": headF, "records": recsF}
    # ---------------- the 1-field production
    recsP = []
    for k in range(N):
        r = {"T": T[k], "T_over_rs": Trs[k], "E": Ep[k], "E_over_E0": Ep[k] / Ep[0], "R_t": (tzP[k][1] if k in tzP else None), "r_in": (tzP[k][0] if k in tzP else None),
             "trapping_zeros": (tzP[k] if k in tzP else []), "min_N_minus_absv": minN[k], "r_min_N_minus_absv": 100.0 - 20 * k + 0.01, "H_in": Hin[k] + 0.001,
             "continuation_label": "verified (Weyl)"}
        if k >= 1:
            r["dE_dT"] = (Ep[k] - Ep[k - 1]) / TAU
        recsP.append(r)
    argsP = base_args(nsites=1024, mix="plain")
    for q in ("light_kappas", "light_guard_kappas", "light_coupling", "light_transport", "slices_light_c128"):
        argsP.pop(q, None)
    runP = {"script": "manufactured (self-test, 1-field)", "args": argsP, "head": {"E0": Ep[0]}, "records": recsP}
    npz = {"T": np.array(T), "r_nodes": r_nodes, "M": M, "rho_ell": rho, "W": 0.5 * M, "N": np.ones_like(M), "v": -0.1 * np.ones_like(M)}
    return runC, runF, runP, npz


def write_set(d, runC, runF, runP, npz, drop_npz_key=None):
    os.makedirs(d, exist_ok=True)
    paths = {"run": os.path.join(d, "run.json"), "follow": os.path.join(d, "follow.json"), "prod": os.path.join(d, "prod.json"), "s11": os.path.join(d, "run_s11.npz")}
    for p, obj in ((paths["run"], runC), (paths["follow"], runF), (paths["prod"], runP)):
        with open(p, "w") as f:
            json.dump(obj, f)
    arrays = {k: v for k, v in npz.items() if k != drop_npz_key}
    np.savez(paths["s11"], **arrays)
    return paths


def main():
    tmp = tempfile.mkdtemp(prefix="C_readout_selftest_")
    fails = []
    try:
        runC, runF, runP, npz = manufacture()
        paths = write_set(os.path.join(tmp, "good"), runC, runF, runP, npz)
        slice_dir = os.path.join(tmp, "slices")
        os.makedirs(slice_dir)
        open(os.path.join(slice_dir, "orbitals_selftest_T0016.00.npz"), "wb").close()
        out = RO.run_readout(paths["run"], paths["s11"], paths["follow"], paths["prod"], os.path.join(tmp, "out"), "selftest", slice_dir=slice_dir, quiet=True)
        # every identity to 1e-12 (the manufactured data follow the readout's own arithmetic)
        for name in out["identities_order"]:
            rec = out["identities"][name]
            if rec["max_abs_residual"] > TOL:
                fails.append("identity '%s' (%s): max |residual| %.3g > %.0e" % (name, rec["kind"], rec["max_abs_residual"], TOL))
            if rec["n"] == 0 and rec["kind"] in ("exact", "relation", "hypothesis"):
                fails.append("identity '%s' tested on no record" % name)
        # the structure: outputs, clocks, formation, rows, the stop key, the slice listing
        if not (os.path.isfile(out["written"]["json"]) and os.path.isfile(out["written"]["md"])):
            fails.append("outputs not written")
        with open(out["written"]["json"]) as f:
            back = json.load(f)
        if len(back["series"]["E"]) != N:
            fails.append("series length %d != %d" % (len(back["series"]["E"]), N))
        if not (out["formation"]["C"] and abs(out["formation"]["C"]["T_over_rs"] - 16.0) < 1e-12):
            fails.append("formation clock not 16: %r" % (out["formation"]["C"],))
        if out["formation"]["dT_over_rs"] != 0.0:
            fails.append("formation dT != 0: %r" % out["formation"]["dT_over_rs"])
        with open(out["written"]["md"]) as f:
            md = f.read()
        for want in ("formation (16)", "| 16 |", "| 24 |", "| 32 |", "20: absent", "28: absent", "36: absent", "40: absent", "T/r_s 16: orbitals_selftest_T0016.00.npz"):
            if want not in md:
                fails.append("md lacks '%s'" % want)
        r4 = out["readouts"]["4"]
        if not (r4["stop_records"] and r4["stop_records"][0]["T_over_rs"] == 32.0 and "stop" in r4["stop_records"][0]["keys"]):
            fails.append("the manufactured stop key on the last record was not reported: %r" % (r4["stop_records"],))
        if r4["records_with_negative_points_T_over_rs"] != [32.0]:
            fails.append("negative points not reported on the last record: %r" % (r4["records_with_negative_points_T_over_rs"],))
        if out["identities"]["C: dE_dT is a finite difference of E"]["best"].split()[0] != "backward":
            fails.append("dE_dT hypothesis did not pick the backward difference")
        if out["readouts"]["2"]["trapezoid_sign"] != {"heavy": 1.0, "light": 1.0}:
            fails.append("trapezoid sign hypothesis: %r" % (out["readouts"]["2"]["trapezoid_sign"],))
        if out["meta"]["matched_clocks"] != {"follow": N, "production": N, "follow_vs_production": N}:
            fails.append("matched clocks: %r" % (out["meta"]["matched_clocks"],))
        if abs(out["series"]["A"][2] - 4.0 * np.pi * runC["records"][2]["R_t"] ** 2) > 1e-12:
            fails.append("A = 4 pi R_t^2 wrong at the formation")
        print("readout on the manufactured data: %d identities, max |residual| %.3g, %d fails so far"
              % (len(out["identities_order"]), max(r["max_abs_residual"] for r in out["identities"].values()), len(fails)))

        # a missing key must stop the readout loudly, naming the key
        def rm(obj, path):
            o = copy.deepcopy(obj)
            cur = o
            for p in path[:-1]:
                cur = cur[p]
            del cur[path[-1]]
            return o
        cases = [
            ("run records[3].M1_I_E", lambda: (rm(runC, ("records", 3, "M1_I_E")), runF, runP, None), "M1_I_E"),
            ("run records[1].light_v36.E_inv_light_over_E", lambda: (rm(runC, ("records", 1, "light_v36", "E_inv_light_over_E")), runF, runP, None), "E_inv_light_over_E"),
            ("run records[2].light_v36.channels.2.certified.pieces", lambda: (rm(runC, ("records", 2, "light_v36", "channels", "2", "certified", "pieces")), runF, runP, None), "pieces"),
            ("run records[4].s11.M_min_outside_cell", lambda: (rm(runC, ("records", 4, "s11", "M_min_outside_cell")), runF, runP, None), "M_min_outside_cell"),
            ("run records[2].M1_balance (required from the second record on)", lambda: (rm(runC, ("records", 2, "M1_balance")), runF, runP, None), "M1_balance"),
            ("run records[3].light_v36.truncation_guard.eps_guard_over_sourcing", lambda: (rm(runC, ("records", 3, "light_v36", "truncation_guard", "eps_guard_over_sourcing")), runF, runP, None), "eps_guard_over_sourcing"),
            ("run head.light_v36.first_return", lambda: (rm(runC, ("head", "light_v36", "first_return")), runF, runP, None), "first_return"),
            ("run args.light_guard_kappas", lambda: (rm(runC, ("args", "light_guard_kappas")), runF, runP, None), "light_guard_kappas"),
            ("follow records[2].E_rad0_over_E", lambda: (runC, rm(runF, ("records", 2, "E_rad0_over_E")), runP, None), "E_rad0_over_E"),
            ("follow records[0].channels.1.I_E_cut", lambda: (runC, rm(runF, ("records", 0, "channels", "1", "I_E_cut")), runP, None), "I_E_cut"),
            ("production records[1].R_t", lambda: (runC, runF, rm(runP, ("records", 1, "R_t")), None), "R_t"),
            ("s11 npz without M", lambda: (runC, runF, runP, "M"), "M"),
        ]
        for i, (desc, make, key) in enumerate(cases):
            c, f_, p_, drop = make()
            pths = write_set(os.path.join(tmp, "case%02d" % i), c, f_, p_, npz, drop_npz_key=drop)
            try:
                RO.run_readout(pths["run"], pths["s11"], pths["follow"], pths["prod"], os.path.join(tmp, "out%02d" % i), "case", quiet=True)
                fails.append("missing %s: the readout did not stop" % desc)
            except RO.MissingKey as e:
                if ("'%s'" % key) not in str(e) and key not in str(e):
                    fails.append("missing %s: MissingKey raised but the message does not name '%s': %s" % (desc, key, e))
                else:
                    print("missing %s: stopped loudly: %s" % (desc, str(e)[:160]))
            except Exception as e:  # noqa: BLE001
                fails.append("missing %s: stopped with %s instead of MissingKey: %s" % (desc, type(e).__name__, e))
        # the allowed absence: record 0 without M1_balance etc. is the manufactured case itself (it ran above)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if fails:
        print("FAIL (%d):" % len(fails))
        for f in fails:
            print("  - " + f)
        return 1
    print("PASS: every identity holds to %.0e on the manufactured data and every removed key stops the readout naming it" % TOL)
    return 0


if __name__ == "__main__":
    sys.exit(main())
