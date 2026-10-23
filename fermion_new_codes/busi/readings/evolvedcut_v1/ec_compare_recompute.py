#!/usr/bin/env python
"""ec_compare_recompute.py (busi lane, cold check of EVOLVEDCUT_route_compare_v1): my from-slice recomputation (recompute.json,
ec_recompute.py) set against both routes' stored per-channel entropies at the 60 matched rows: complement (interior-block) and
exterior-Gram sides for C (polar), P_- (held) and P_0 (flat), the excess on both sides, the weighted sums, and the exterior
cluster diagnostics. Writes OUT/compare_recompute.json and OUT/compare_recompute_summary.txt."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ec_common as C  # noqa: E402

MATCHED = [(12, 58), (15, 51), (20, 38), (28, 37), (35, 37), (40, 36)]


def main():
    rc = C.jload(C.OUT + "/recompute.json")
    mine = {(r["T"], r["kappa"]): r for r in rc["rows"]}
    assert len(mine) == 60
    r2 = C.r2_rows_index(C.jload(C.R2_RESULTS))
    out = {"inputs": {"recompute": [C.OUT + "/recompute.json", C.md5(C.OUT + "/recompute.json")], "r2_results": [C.R2_RESULTS, C.md5(C.R2_RESULTS)]},
           "selftest": rc.get("selftest"), "clocks": rc.get("clocks"), "rows": [], "weighted": {}, "summary": {}}
    for T, n_in in MATCHED:
        d1 = C.jload(C.r1_path(T))
        out["inputs"]["r1_T%d" % T] = [C.r1_path(T), C.md5(C.r1_path(T))]
        ch1 = {int(c["kappa"]): c for c in d1["channels"]}
        Wm = {"excess_int": 0.0, "excess_ext": 0.0, "C_int": 0.0, "C_ext": 0.0, "held_int": 0.0, "held_ext": 0.0, "flat_int": 0.0, "flat_ext": 0.0, "excess_int_refined": 0.0}
        acc = {k: [] for k in Wm}
        for k in C.KAPPAS:
            m = mine[(T, k)]
            c1 = ch1[k]
            x1 = c1["cuts"]["root"]
            cut2 = C.r2_cut(r2[(T, k)], n_in)
            oc = cut2["orthonormal_control"]
            w = 16.0 * k
            row = {"T": T, "kappa": k, "n_in": n_in, "source_dtype": m["source_dtype"], "n_occ": m["n_occ"], "n_dust": m["n_dust"],
                   "mine": {"C_int": m["C"]["S_int"], "C_ext": m["C"]["S_ext"], "C_ext_minus_int": m["C"]["ext_minus_int"], "C_cluster": m["C"]["S_ext_cluster"],
                            "C_genuine_minus_int": m["C"]["S_ext_genuine_minus_int"], "C_cluster_max_dev": m["C"]["cluster_max_abs_dev"], "C_cluster_n_below_1": m["C"]["cluster_n_below_1"],
                            "C_int_near_boundary_S": m["C"]["int_S_near_boundary"], "C_mu_min": m["C"]["mu_sorted_smallest3"][0],
                            "Cr_int": m["C_refined"]["S_int"], "Cr_ext": m["C_refined"]["S_ext"], "Cr_ext_minus_int": m["C_refined"]["ext_minus_int"], "Cr_cluster": m["C_refined"]["S_ext_cluster"],
                            "held_int": m["held"]["S_int"], "held_ext": m["held"]["S_ext"], "held_ext_minus_int": m["held"]["ext_minus_int"], "held_cluster": m["held"]["S_ext_cluster"],
                            "flat_int": m["flat"]["S_int"], "flat_ext": m["flat"]["S_ext"], "flat_ext_minus_int": m["flat"]["ext_minus_int"],
                            "excess_int": m["excess_int"], "excess_ext": m["excess_ext"], "excess_int_refined": m["excess_int_refined"],
                            "orth_Q": m["orth_Q_frob"], "orth_Q2": m["orth_Q2_frob"], "S_global_Q": m["S_global_Q"], "gram_defect": m["gram_defect_frob"],
                            "held_n_neg": m["held_info"]["n_neg"], "held_gap": m["held_info"]["nearest_zero"], "held_eig_vs_stored": m["held_info"]["max_abs_eig_vs_stored_e_k"],
                            "held_orth": m["held_info"]["orth_defect_frob"], "flat_orth": m["flat_info"]["orth_defect_frob"], "n_pair": m["n_pair"], "n_pair_alt_layout": m["n_pair_alt_layout"]},
                   "r1": {"S_X": x1["S_X"], "S_Xc": x1["S_Xc"], "S_X_vac": x1["S_X_vac"], "S_Xc_vac": x1["S_Xc_vac"], "S_X_P0": x1["flat"]["S_X_P0"], "S_Xc_P0": x1["flat"]["S_Xc_P0"],
                          "excess": x1["excess"], "excess_c": x1["excess_c"], "n_pair": (c1.get("trace_norm") or {}).get("n_pair"), "loewdin_orthonormality": c1["loewdin_orthonormality"]},
                   "r2": {"pol_ext": oc["physical_exterior_entropy"], "pol_int": oc["physical_interior_entropy"], "held": cut2["held"]["entropy"], "flat": cut2["flat"]["entropy"],
                          "raw_ext": cut2["raw"].get("physical_exterior_entropy"), "raw_int": cut2["raw"].get("physical_interior_entropy"), "polar_residual": r2[(T, k)]["orthogonality"]["orthonormal_control_frobenius"]}}
            mi, a, b = row["mine"], row["r1"], row["r2"]
            row["d"] = {"C_int_minus_r1_Xc": mi["C_int"] - a["S_Xc"], "C_int_minus_r2_int": mi["C_int"] - b["pol_int"], "Cr_int_minus_r1_Xc": mi["Cr_int"] - a["S_Xc"],
                        "C_ext_minus_r1_X": mi["C_ext"] - a["S_X"], "C_ext_minus_r2_ext": mi["C_ext"] - b["pol_ext"], "Cr_ext_minus_r1_X": mi["Cr_ext"] - a["S_X"],
                        "C_int_minus_r1_X": mi["C_int"] - a["S_X"], "C_int_minus_r2_ext": mi["C_int"] - b["pol_ext"],
                        "held_int_minus_r1_Xc_vac": mi["held_int"] - a["S_Xc_vac"], "held_int_minus_r2_held": mi["held_int"] - b["held"], "held_ext_minus_r1_X_vac": mi["held_ext"] - a["S_X_vac"],
                        "flat_int_minus_r1_Xc_P0": mi["flat_int"] - a["S_Xc_P0"], "flat_int_minus_r2_flat": mi["flat_int"] - b["flat"], "flat_ext_minus_r1_X_P0": mi["flat_ext"] - a["S_X_P0"],
                        "excess_int_minus_r1_excess_c": mi["excess_int"] - a["excess_c"], "excess_int_minus_r2_int_excess": mi["excess_int"] - (b["pol_int"] - b["held"]),
                        "excess_ext_minus_r1_excess": mi["excess_ext"] - a["excess"], "excess_ext_minus_r2_pol_excess": mi["excess_ext"] - (b["pol_ext"] - b["held"]),
                        "excess_int_minus_r1_excess": mi["excess_int"] - a["excess"], "excess_int_minus_r2_pol_excess": mi["excess_int"] - (b["pol_ext"] - b["held"]),
                        "Cr_int_minus_C_int": mi["Cr_int"] - mi["C_int"], "n_pair_minus_r1": (mi["n_pair"] - a["n_pair"]) if a["n_pair"] is not None else None,
                        "C_int_minus_r2_raw_int": (mi["C_int"] - b["raw_int"]) if b["raw_int"] is not None else None}
            out["rows"].append(row)
            for key, val in (("excess_int", mi["excess_int"]), ("excess_ext", mi["excess_ext"]), ("C_int", mi["C_int"]), ("C_ext", mi["C_ext"]), ("held_int", mi["held_int"]),
                             ("held_ext", mi["held_ext"]), ("flat_int", mi["flat_int"]), ("flat_ext", mi["flat_ext"]), ("excess_int_refined", mi["excess_int_refined"])):
                acc[key].append(w * val)
        Wm = {k: math.fsum(v) for k, v in acc.items()}
        w1 = d1["weighted"]["root"]
        rows2 = [C.r2_cut(r2[(T, k)], n_in) for k in C.KAPPAS]
        w2_ext = math.fsum(16.0 * k * c["orthonormal_control"]["physical_exterior_entropy"] for k, c in zip(C.KAPPAS, rows2))
        w2_int = math.fsum(16.0 * k * c["orthonormal_control"]["physical_interior_entropy"] for k, c in zip(C.KAPPAS, rows2))
        w2_held = math.fsum(16.0 * k * c["held"]["entropy"] for k, c in zip(C.KAPPAS, rows2))
        w1_exc_c = math.fsum(16.0 * k * ch1[k]["cuts"]["root"]["excess_c"] for k in C.KAPPAS)
        out["weighted"][T] = {"n_in": n_in, "mine": Wm, "r1_excess_ext": w1["excess"], "r1_excess_comp": w1_exc_c, "r2_excess_ext": w2_ext - w2_held, "r2_excess_comp": w2_int - w2_held,
                              "mine_int_minus_r1_ext": Wm["excess_int"] - w1["excess"], "mine_int_minus_r1_comp": Wm["excess_int"] - w1_exc_c,
                              "mine_int_minus_r2_ext": Wm["excess_int"] - (w2_ext - w2_held), "mine_int_minus_r2_comp": Wm["excess_int"] - (w2_int - w2_held),
                              "mine_ext_minus_r1_ext": Wm["excess_ext"] - w1["excess"], "mine_ext_minus_r2_ext": Wm["excess_ext"] - (w2_ext - w2_held),
                              "mine_ext_minus_mine_int": Wm["excess_ext"] - Wm["excess_int"], "mine_refined_int_minus_int": Wm["excess_int_refined"] - Wm["excess_int"],
                              "mine_C_ext_minus_int": Wm["C_ext"] - Wm["C_int"], "mine_held_ext_minus_int": Wm["held_ext"] - Wm["held_int"]}
    R = out["rows"]
    S = out["summary"]
    for key in R[0]["d"]:
        vals = [r["d"][key] for r in R if r["d"][key] is not None]
        S["max_abs_" + key] = max(abs(v) for v in vals)
        S["range_" + key] = [min(vals), max(vals)]
        S["n_" + key] = len(vals)
    for key in ("C_ext_minus_int", "Cr_ext_minus_int", "held_ext_minus_int", "flat_ext_minus_int", "C_cluster", "Cr_cluster", "held_cluster", "C_genuine_minus_int", "C_cluster_max_dev",
                "C_int_near_boundary_S", "C_mu_min", "orth_Q", "orth_Q2", "S_global_Q", "gram_defect", "held_gap", "held_eig_vs_stored", "held_orth", "flat_orth", "n_pair"):
        vals = [r["mine"][key] for r in R]
        S["range_mine_" + key] = [min(vals), max(vals)]
    S["all_held_n_neg_2048"] = all(r["mine"]["held_n_neg"] == 2048 for r in R)
    S["n_pair_alt_layout_T12k1"] = mine[(12, 1)]["n_pair_alt_layout"]
    S["n_pair_T12k1"] = mine[(12, 1)]["n_pair"]
    S["r1_n_pair_T12k1"] = R[0]["r1"]["n_pair"]
    S["n_pair_minus_n_dust_range"] = [min(r["mine"]["n_pair"] - r["n_dust"] for r in R), max(r["mine"]["n_pair"] - r["n_dust"] for r in R)]
    S["weighted_mine_int_minus_r1_ext"] = {T: out["weighted"][T]["mine_int_minus_r1_ext"] for T, _ in MATCHED}
    S["weighted_mine_int_minus_r1_comp"] = {T: out["weighted"][T]["mine_int_minus_r1_comp"] for T, _ in MATCHED}
    S["weighted_mine_int_minus_r2_ext"] = {T: out["weighted"][T]["mine_int_minus_r2_ext"] for T, _ in MATCHED}
    S["weighted_mine_int_minus_r2_comp"] = {T: out["weighted"][T]["mine_int_minus_r2_comp"] for T, _ in MATCHED}
    S["weighted_mine_ext_minus_r1_ext"] = {T: out["weighted"][T]["mine_ext_minus_r1_ext"] for T, _ in MATCHED}
    S["weighted_mine_ext_minus_mine_int"] = {T: out["weighted"][T]["mine_ext_minus_mine_int"] for T, _ in MATCHED}
    S["weighted_mine_excess_int"] = {T: out["weighted"][T]["mine"]["excess_int"] for T, _ in MATCHED}
    S["weighted_mine_excess_ext"] = {T: out["weighted"][T]["mine"]["excess_ext"] for T, _ in MATCHED}
    C.jdump(out, C.OUT + "/compare_recompute.json")
    L = ["ec_compare_recompute.py: my from-slice recomputation against both routes at the 60 matched rows",
         "selftest: %s" % rc.get("selftest"),
         "clocks (applied root mine vs route 2; route 1 cut):"]
    for T, _ in MATCHED:
        ck = rc["clocks"][str(T)]
        L.append("  T%d: my applied outer root %s, route 2 %.6f, route 1 applied_outer %s; route 1 cut %.6f (n_in from it %d); n_in from my root %s; min(N-|v|) %.4f"
                 % (T, ("%.6f" % ck["applied_outer_mine"]) if ck["applied_outer_mine"] is not None else None, ck["r2_radius"], ck["r1_applied_outer"], ck["r1_R_cut"], ck["n_in_from_r1_R_cut"], ck["n_in_from_my_applied_root"], ck["min_N_minus_absv"]))
    L.append("per-channel differences, max |.| over 60 rows (range):")
    for key in R[0]["d"]:
        L.append("  %-34s %.3e   [%+.3e, %+.3e]  (n %d)" % (key, S["max_abs_" + key], S["range_" + key][0], S["range_" + key][1], S["n_" + key]))
    L.append("my own diagnostics (ranges over 60 rows):")
    for key in ("C_ext_minus_int", "Cr_ext_minus_int", "held_ext_minus_int", "flat_ext_minus_int", "C_cluster", "Cr_cluster", "held_cluster", "C_genuine_minus_int", "C_cluster_max_dev",
                "C_int_near_boundary_S", "C_mu_min", "orth_Q", "orth_Q2", "S_global_Q", "gram_defect", "held_gap", "held_eig_vs_stored", "held_orth", "flat_orth"):
        L.append("  %-24s [%+.3e, %+.3e]" % (key, S["range_mine_" + key][0], S["range_mine_" + key][1]))
    L.append("  held n_neg all 2048: %s; n_pair - n_dust range [%.4f, %.4f]; T12 k1 n_pair %.6f (route 1 %s), alternative block-layout reading %.3f" % (S["all_held_n_neg_2048"], S["n_pair_minus_n_dust_range"][0], S["n_pair_minus_n_dust_range"][1], S["n_pair_T12k1"], S["r1_n_pair_T12k1"], S["n_pair_alt_layout_T12k1"]))
    L.append("weighted (16 kappa) excess at the matched partitions:")
    for T, _ in MATCHED:
        w = out["weighted"][T]
        L.append("  T%d: mine int %.9f ext %.9f (ext-int %+.3e; refined int - int %+.2e) | r1 ext %.9f comp %.9f | r2 ext %.9f comp %.9f | mine int - r1 ext %+.3e, - r1 comp %+.3e, - r2 ext %+.3e, - r2 comp %+.3e | mine ext - r1 ext %+.3e"
                 % (T, w["mine"]["excess_int"], w["mine"]["excess_ext"], w["mine_ext_minus_mine_int"], w["mine_refined_int_minus_int"], w["r1_excess_ext"], w["r1_excess_comp"], w["r2_excess_ext"], w["r2_excess_comp"],
                    w["mine_int_minus_r1_ext"], w["mine_int_minus_r1_comp"], w["mine_int_minus_r2_ext"], w["mine_int_minus_r2_comp"], w["mine_ext_minus_r1_ext"]))
    txt = "\n".join(L) + "\n"
    with open(C.OUT + "/compare_recompute_summary.txt", "w") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
