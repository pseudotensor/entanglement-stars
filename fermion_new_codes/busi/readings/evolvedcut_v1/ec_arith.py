#!/usr/bin/env python
"""ec_arith.py (busi lane, cold check of EVOLVEDCUT_route_compare_v1 and EVOLVEDCUT_error_statement_v1): every number the
comparison and the error statement rest on, recomputed from the two routes' stored per-channel jsons with own code (pure json
arithmetic; the from-slice recomputation is ec_recompute.py, the stored-spectrum analysis ec_spectra.py).

Reads (read-only): route 1's evolvedcut_prod_T*.json (and the s8 member's for the 2.1e-10 allowance), route 2's results.json,
tables.json and matched_purity_diagnostics.json, and the two delivery markdowns copied to OUT/in/ for the cell-by-cell check.
Writes OUT/arith.json and OUT/arith_summary.txt."""
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ec_common as C  # noqa: E402

IN = C.OUT + "/in"
DELIV_MAIN = IN + "/EVOLVEDCUT_route_compare_v1.md"
DELIV_TABLES = IN + "/EVOLVEDCUT_route_compare_v1_tables.md"

# the error statement's printed numbers (solve_b/EVOLVEDCUT_error_statement_v1.md)
ERR_E2 = {12: 2.53911e-6, 15: 2.55370e-6, 20: 2.62237e-6, 28: 2.69705e-6, 35: 2.50281e-6, 40: 2.50157e-6}
ERR_GAP = {12: -2.57e-6, 15: -2.53e-6, 20: -2.65e-6, 28: -2.73e-6, 35: -2.49e-6, 40: -2.49e-6}
ERR_DELTA_RANGE = (2.1699e-9, 3.6495e-9)
# the purity-bound delivery's printed numbers (solve_b/EVOLVEDCUT_purity_bound_v1.md), context only
PB_DELTA = {12: 285.10112677308, 15: 109.35586072133, 20: 296.51013200938, 28: 297.71584732641, 35: 299.44353886812, 40: 310.88840041701}
PB_G = {12: 3.157690810e-8, 15: 2.016616839e-8, 20: 3.087574155e-8, 28: 3.091492189e-8, 35: 1.537948210e-8, 40: 1.649115047e-8}
PB_PRINTED_P = {12: 285.101126742, 15: 109.355860741, 20: 296.510131979, 28: 297.715847296, 35: 299.443538883, 40: 310.888400433}


def wsum(pairs):
    return math.fsum(w * v for w, v in pairs)


def main():
    out = {"inputs": {}, "route1": {}, "route2": {}, "matched": {}, "unmatched": {}, "statement": {}, "error_statement": {},
           "purity_bound_context": {}, "cells": {}}
    # ---------------- inputs and md5s
    r1 = {}
    for T in C.CLOCKS:
        p = C.r1_path(T)
        r1[T] = C.jload(p)
        out["inputs"]["r1_prod_T%d" % T] = [p, C.md5(p)]
    r1s8 = {}
    for T in C.S8_CLOCKS:
        p = C.r1_path(T, "s8")
        if os.path.exists(p):
            r1s8[T] = C.jload(p)
            out["inputs"]["r1_s8_T%d" % T] = [p, C.md5(p)]
    for name, p in (("r2_results", C.R2_RESULTS), ("r2_tables", C.R2_TABLES), ("r2_diag", C.R2_DIAG)):
        out["inputs"][name] = [p, C.md5(p)]
    for name, p in (("deliv_main", DELIV_MAIN), ("deliv_tables", DELIV_TABLES)):
        out["inputs"][name] = [p, C.md5(p) if os.path.exists(p) else None]
    r2 = C.jload(C.R2_RESULTS)
    r2t = C.jload(C.R2_TABLES)
    r2d = C.jload(C.R2_DIAG)
    r2i = C.r2_rows_index(r2)

    # ---------------- route 1 internal arithmetic (prod, all 10 clocks; s8 for the allowance)
    R1 = {}
    for T, d in r1.items():
        chans = {int(c["kappa"]): c for c in d["channels"]}
        assert sorted(chans) == C.KAPPAS
        rows = []
        for k in C.KAPPAS:
            c = chans[k]
            x = c["cuts"]["root"]
            assert float(c["weight"]) == 16.0 * k
            rows.append({"kappa": k, "w": 16.0 * k, "S_X": x["S_X"], "S_Xc": x["S_Xc"], "S_X_vac": x["S_X_vac"], "S_Xc_vac": x["S_Xc_vac"],
                         "S_X_P0": x["flat"]["S_X_P0"], "S_Xc_P0": x["flat"]["S_Xc_P0"], "excess": x["excess"], "excess_c": x["excess_c"],
                         "purity_gap": x["purity_gap"], "identity_defect": x["pure_cut_identity"]["defect"], "n_Xc": int(x["n_Xc"]),
                         "n_X": int(x["n_X"]), "eig_X_extremes": x["eig_X_extremes"], "loewdin_orthonormality": c["loewdin_orthonormality"],
                         "n_dust": int(c["n_dust"]), "N_occ": int(c["N_occ"]),
                         "chk_excess": x["S_X"] - x["S_X_vac"] - x["excess"], "chk_excess_c": x["S_Xc"] - x["S_Xc_vac"] - x["excess_c"],
                         "chk_purity_gap": x["S_X"] - x["S_Xc"] - x["purity_gap"],
                         "ext_minus_int_vac": x["S_X_vac"] - x["S_Xc_vac"], "ext_minus_int_P0": x["flat"]["S_X_P0"] - x["flat"]["S_Xc_P0"]})
        n_in = {r["n_Xc"] // 2 for r in rows}
        assert len(n_in) == 1
        n_in = n_in.pop()
        W = d["weighted"]["root"]
        my = {key: wsum((r["w"], r[key]) for r in rows) for key in ("S_X", "S_Xc", "S_X_vac", "S_Xc_vac", "S_X_P0", "S_Xc_P0", "excess", "excess_c")}
        rec = d["record"]
        R1[T] = {"dtype": d["dtype"], "n_in": n_in, "R_cut": d["root"]["R_cut"], "source": d["root"]["source"], "applied_outer": d["root"].get("applied_outer"),
                 "record_outer": d["root"].get("record_outer"), "record_two_gE": rec.get("two_gE"), "record_trapping_zeros": rec.get("trapping_zeros"),
                 "rows": rows, "weighted_mine": my,
                 "weighted_stored": {key: W.get(key) for key in ("S_X", "S_Xc", "S_X_vac", "S_X_P0", "excess", "max_abs_purity_gap", "max_abs_identity_defect", "n_pair")},
                 "weighted_mine_minus_stored": {key: my[key] - W[key] for key in ("S_X", "S_Xc", "S_X_vac", "S_X_P0", "excess")},
                 "max_abs_purity_gap": max(abs(r["purity_gap"]) for r in rows), "max_abs_identity_defect": max(abs(r["identity_defect"]) for r in rows),
                 "max_abs_chk": max(max(abs(r["chk_excess"]), abs(r["chk_excess_c"]), abs(r["chk_purity_gap"])) for r in rows),
                 "n_in_from_R_cut": int(math.floor(d["root"]["R_cut"]))}
    out["route1"]["clocks"] = R1
    pg_all = [abs(r["purity_gap"]) for T in R1 for r in R1[T]["rows"]]
    idf_all = [abs(r["identity_defect"]) for T in R1 for r in R1[T]["rows"]]
    out["route1"]["prod_max_abs_purity_gap_100rows"] = max(pg_all)
    out["route1"]["prod_max_abs_identity_defect_100rows"] = max(idf_all)
    out["route1"]["prod_range_ext_minus_int_C"] = [min(r["purity_gap"] for T in R1 for r in R1[T]["rows"]), max(r["purity_gap"] for T in R1 for r in R1[T]["rows"])]
    out["route1"]["prod_range_ext_minus_int_vac"] = [min(r["ext_minus_int_vac"] for T in R1 for r in R1[T]["rows"]), max(r["ext_minus_int_vac"] for T in R1 for r in R1[T]["rows"])]
    out["route1"]["prod_range_ext_minus_int_P0"] = [min(r["ext_minus_int_P0"] for T in R1 for r in R1[T]["rows"]), max(r["ext_minus_int_P0"] for T in R1 for r in R1[T]["rows"])]
    if r1s8:
        s8pg, s8id = [], []
        for T, d in r1s8.items():
            for c in d["channels"]:
                x = c["cuts"]["root"]
                s8pg.append(abs(x["purity_gap"]))
                s8id.append(abs(x["pure_cut_identity"]["defect"]))
        out["route1"]["s8_max_abs_purity_gap"] = max(s8pg)
        out["route1"]["s8_max_abs_identity_defect"] = max(s8id)
        out["route1"]["s8_rows"] = len(s8pg)
    out["route1"]["prod_excess_range"] = [min(R1[T]["weighted_mine"]["excess"] for T in R1), max(R1[T]["weighted_mine"]["excess"] for T in R1)]
    out["route1"]["prod_excess_argrange"] = [min(R1, key=lambda T: R1[T]["weighted_mine"]["excess"]), max(R1, key=lambda T: R1[T]["weighted_mine"]["excess"])]
    if r1s8:
        ex8 = {T: r1s8[T]["weighted"]["root"]["excess"] for T in r1s8}
        out["route1"]["s8_excess"] = ex8
        out["route1"]["s8_excess_range"] = [min(ex8.values()), max(ex8.values())]

    # ---------------- route 2 internal arithmetic: per-channel to totals, all 10 clocks x 3 shifts
    R2 = {}
    for T in C.CLOCKS:
        k1 = r2i[(T, 1)]
        prim = int(k1["interior_site_count"])
        shifts = {}
        for c in k1["cuts"]:
            shifts[int(c["site_shift"])] = int(c["interior_sites"])
        R2[T] = {"cut_label": k1["cut_label"], "radius": k1["radius_over_a"], "primary_n_in": prim, "partitions": shifts, "dtype": k1["source_dtype"],
                 "n_in_from_radius": int(math.floor(k1["radius_over_a"])), "by_shift": {}}
        for sh, n_in in shifts.items():
            rows = []
            raw_all = True
            for k in C.KAPPAS:
                row = r2i[(T, k)]
                assert int(row["weight"]) == 16 * k
                cut = C.r2_cut(row, n_in)
                assert cut is not None and int(cut["site_shift"]) == sh
                oc = cut["orthonormal_control"]
                rw = cut["raw"]
                raw_ok = bool(row["raw_global"]["valid"]) and rw.get("physical_exterior_entropy") is not None
                raw_all = raw_all and raw_ok
                rows.append({"kappa": k, "w": 16.0 * k, "pol_ext": oc["physical_exterior_entropy"], "pol_int": oc["physical_interior_entropy"],
                             "pol_ext_raw_field": oc["exterior"]["entropy"], "pol_int_raw_field": oc["interior"]["entropy"],
                             "pol_valid": bool(oc["exterior"]["valid"] and oc["interior"]["valid"]),
                             "raw_ext": rw.get("physical_exterior_entropy"), "raw_int": rw.get("physical_interior_entropy"), "raw_ok": raw_ok,
                             "raw_ext_field": rw["exterior"]["entropy"], "raw_int_field": rw["interior"]["entropy"],
                             "held": cut["held"]["entropy"], "held_valid": bool(cut["held"]["valid"]), "flat": cut["flat"]["entropy"], "flat_valid": bool(cut["flat"]["valid"]),
                             "pol_global_entropy": row["polar_global"]["entropy"], "raw_global_entropy": row["raw_global"]["entropy"],
                             "pol_ext_tol": oc["exterior"]["rounding_tolerance"], "pol_ext_clip": oc["exterior"]["clipping_displacement"],
                             "pol_ext_min": oc["exterior"]["minimum"], "pol_ext_max": oc["exterior"]["maximum"], "pol_ext_outside": oc["exterior"]["outside_count"],
                             "pol_int_min": oc["interior"]["minimum"], "pol_int_max": oc["interior"]["maximum"],
                             "polar_residual": row["orthogonality"]["orthonormal_control_frobenius"],
                             "gram_evec_orth": row["orthogonality"]["joint_eigenvector_orthogonality_frobenius"],
                             "gram_defect": row["orthogonality"]["joint_defect_frobenius"]})
                rows[-1]["delta"] = rows[-1]["pol_ext"] - rows[-1]["pol_int"]
                rows[-1]["pol_excess"] = rows[-1]["pol_ext"] - rows[-1]["held"]
                rows[-1]["pol_excess_c"] = rows[-1]["pol_int"] - rows[-1]["held"]
                rows[-1]["raw_delta"] = (rows[-1]["raw_ext"] - rows[-1]["raw_int"]) if raw_ok else None
            tot = {"orthonormal_control": wsum((r["w"], r["pol_ext"]) for r in rows), "held": wsum((r["w"], r["held"]) for r in rows),
                   "flat": wsum((r["w"], r["flat"]) for r in rows), "pol_int": wsum((r["w"], r["pol_int"]) for r in rows)}
            tot["orthonormal_control_excess"] = tot["orthonormal_control"] - tot["held"]
            tot["orthonormal_control_excess_from_channels"] = wsum((r["w"], r["pol_excess"]) for r in rows)
            tot["pol_int_excess"] = tot["pol_int"] - tot["held"]
            tot["held_minus_flat"] = tot["held"] - tot["flat"]
            tot["E2_emp"] = wsum((r["w"], abs(r["delta"])) for r in rows)
            tot["signed_delta"] = wsum((r["w"], r["delta"]) for r in rows)
            tot["weighted_pol_global_entropy"] = wsum((r["w"], r["pol_global_entropy"]) for r in rows)
            if raw_all:
                tot["raw"] = wsum((r["w"], r["raw_ext"]) for r in rows)
                tot["raw_excess"] = tot["raw"] - tot["held"]
            t2 = [x for x in r2t["rows"] if int(round(x["clock_rs"])) == T and int(x["site_shift"]) == sh]
            assert len(t2) == 1
            t2 = t2[0]["totals"]
            cmp = {key: tot[key] - t2[key] for key in ("orthonormal_control", "held", "flat", "orthonormal_control_excess", "held_minus_flat") if t2.get(key) is not None}
            if raw_all and t2.get("raw") is not None:
                cmp["raw"] = tot["raw"] - t2["raw"]
                cmp["raw_excess"] = tot["raw_excess"] - t2["raw_excess"]
            R2[T]["by_shift"][sh] = {"n_in": n_in, "rows": rows, "totals_mine": tot, "totals_stored": t2, "mine_minus_stored": cmp,
                                     "delta_min": min(r["delta"] for r in rows), "delta_max": max(r["delta"] for r in rows),
                                     "all_delta_positive": all(r["delta"] > 0 for r in rows), "raw_all": raw_all}
    out["route2"]["clocks"] = R2
    out["route2"]["max_abs_mine_minus_stored_totals"] = max(abs(v) for T in R2 for sh in R2[T]["by_shift"] for v in R2[T]["by_shift"][sh]["mine_minus_stored"].values())

    # ---------------- matching and the comparison's quantities
    M = {}
    U = {}
    for T in C.CLOCKS:
        n1 = R1[T]["n_in"]
        parts = R2[T]["partitions"]
        inv = {v: k for k, v in parts.items()}
        if n1 not in inv:
            U[T] = {"r1_n_in": n1, "r2_partitions": sorted(parts.values()), "r1_excess": R1[T]["weighted_mine"]["excess"],
                    "r2_primary_pol_excess": R2[T]["by_shift"][0]["totals_mine"]["orthonormal_control_excess"],
                    "r2_primary_raw_excess": R2[T]["by_shift"][0]["totals_mine"].get("raw_excess")}
            continue
        sh = inv[n1]
        a = R1[T]["rows"]
        b = R2[T]["by_shift"][sh]["rows"]
        rows = []
        for ra, rb in zip(a, b):
            assert ra["kappa"] == rb["kappa"]
            row = {"kappa": ra["kappa"], "w": ra["w"],
                   "r1_S_X": ra["S_X"], "r2_pol_ext": rb["pol_ext"],
                   "d_C": ra["S_X"] - rb["pol_ext"], "d_C_comp": ra["S_Xc"] - rb["pol_int"],
                   "d_held": ra["S_X_vac"] - rb["held"], "d_held_comp": ra["S_Xc_vac"] - rb["held"],
                   "d_flat": ra["S_X_P0"] - rb["flat"], "d_flat_comp": ra["S_Xc_P0"] - rb["flat"],
                   "d_excess": ra["excess"] - rb["pol_excess"], "d_excess_comp": ra["excess_c"] - rb["pol_excess_c"],
                   "d_C_raw": (ra["S_X"] - rb["raw_ext"]) if rb["raw_ok"] else None,
                   "d_excess_raw": (ra["excess"] - (rb["raw_ext"] - rb["held"])) if rb["raw_ok"] else None,
                   "r1_ext_int": ra["purity_gap"], "r2_pol_ext_int": rb["delta"], "r2_raw_ext_int": rb["raw_delta"],
                   "r1_ext_int_vac": ra["ext_minus_int_vac"], "r1_ext_int_P0": ra["ext_minus_int_P0"],
                   "r2_pol_global_entropy": rb["pol_global_entropy"], "r1_identity_defect": ra["identity_defect"]}
            for key in ("d_C", "d_C_comp", "d_held", "d_held_comp", "d_flat", "d_flat_comp", "d_excess", "d_excess_comp"):
                row[key + "_within"] = abs(row[key]) <= C.TOL_ID
            # the error statement's descriptive inequalities
            row["err_ineq_excess"] = abs(row["d_excess"]) <= abs(rb["delta"]) + C.TOL_ID
            row["err_ineq_C"] = abs(row["d_C"]) <= abs(rb["delta"]) + C.TOL_ID
            # where does route 1 minus route 2 sit relative to route 2's own asymmetry: d_C + delta = r1_S_X - r2_pol_int
            row["r1_S_X_minus_r2_pol_int"] = ra["S_X"] - rb["pol_int"]
            rows.append(row)
        w1 = R1[T]["weighted_mine"]
        t2 = R2[T]["by_shift"][sh]["totals_mine"]
        W = {"excess": [w1["excess"], t2["orthonormal_control_excess"], w1["excess"] - t2["orthonormal_control_excess"]],
             "excess_comp": [w1["excess_c"], t2["pol_int_excess"], w1["excess_c"] - t2["pol_int_excess"]],
             "C": [w1["S_X"], t2["orthonormal_control"], w1["S_X"] - t2["orthonormal_control"]],
             "C_comp": [w1["S_Xc"], t2["pol_int"], w1["S_Xc"] - t2["pol_int"]],
             "held": [w1["S_X_vac"], t2["held"], w1["S_X_vac"] - t2["held"]],
             "held_comp": [w1["S_Xc_vac"], t2["held"], w1["S_Xc_vac"] - t2["held"]],
             "flat": [w1["S_X_P0"], t2["flat"], w1["S_X_P0"] - t2["flat"]],
             "flat_comp": [w1["S_Xc_P0"], t2["flat"], w1["S_Xc_P0"] - t2["flat"]]}
        if R2[T]["by_shift"][sh]["raw_all"]:
            W["C_raw"] = [w1["S_X"], t2["raw"], w1["S_X"] - t2["raw"]]
            W["excess_raw"] = [w1["excess"], t2["raw_excess"], w1["excess"] - t2["raw_excess"]]
        # the comparison also stores route 1's own exterior-minus-complement weighted excess and route 2's E2
        W["r1_excess_ext_minus_comp"] = w1["excess"] - w1["excess_c"]
        W["r2_E2_emp"] = t2["E2_emp"]
        W["r2_signed_delta_weighted"] = t2["signed_delta"]
        W["r2_weighted_pol_global_entropy"] = t2["weighted_pol_global_entropy"]
        # the same weighted differences summed row by row (difference first, then weight): free of the cancellation of two ~1000-nat sums
        for key, rk in (("excess", "d_excess"), ("excess_comp", "d_excess_comp"), ("C", "d_C"), ("C_comp", "d_C_comp"), ("held", "d_held"),
                        ("held_comp", "d_held_comp"), ("flat", "d_flat"), ("flat_comp", "d_flat_comp")):
            W[key + "_diff_rowwise"] = wsum((r["w"], r[rk]) for r in rows)
            W[key + "_diff_sumorder_noise"] = W[key][2] - W[key + "_diff_rowwise"]
        W["excess_within_TOL_W"] = abs(W["excess"][2]) <= C.TOL_W
        W["excess_comp_within_TOL_W"] = abs(W["excess_comp"][2]) <= C.TOL_W
        W["factor_over_TOL_W"] = abs(W["excess"][2]) / C.TOL_W
        W["relative_excess_difference"] = W["excess"][2] / w1["excess"]
        W["err_ineq_weighted"] = abs(W["excess"][2]) <= t2["E2_emp"] + C.TOL_W
        W["err_ineq_weighted_printed"] = abs(W["excess"][2]) <= t2["E2_emp"] + 1.85e-7
        mx = {key: max(abs(r[key]) for r in rows) for key in ("d_C", "d_C_comp", "d_held", "d_held_comp", "d_flat", "d_flat_comp", "d_excess", "d_excess_comp", "r1_ext_int", "r2_pol_ext_int")}
        mx["d_C_raw"] = max((abs(r["d_C_raw"]) for r in rows if r["d_C_raw"] is not None), default=None)
        M[T] = {"shift": sh, "n_in": n1, "rows": rows, "weighted": W, "max_abs": mx,
                "range_d_C": [min(r["d_C"] for r in rows), max(r["d_C"] for r in rows)],
                "range_d_excess": [min(r["d_excess"] for r in rows), max(r["d_excess"] for r in rows)],
                "range_r2_pol_ext_int": [min(r["r2_pol_ext_int"] for r in rows), max(r["r2_pol_ext_int"] for r in rows)],
                "range_r1_ext_int": [min(r["r1_ext_int"] for r in rows), max(r["r1_ext_int"] for r in rows)],
                "range_d_held": [min(r["d_held"] for r in rows), max(r["d_held"] for r in rows)],
                "range_d_held_comp": [min(r["d_held_comp"] for r in rows), max(r["d_held_comp"] for r in rows)],
                "range_d_C_raw": ([min(r["d_C_raw"] for r in rows), max(r["d_C_raw"] for r in rows)] if R2[T]["by_shift"][sh]["raw_all"] else None),
                "all_d_C_negative": all(r["d_C"] < 0 for r in rows), "all_err_ineq": all(r["err_ineq_excess"] and r["err_ineq_C"] for r in rows),
                "n_within": {key: sum(1 for r in rows if r[key + "_within"]) for key in ("d_C", "d_C_comp", "d_held", "d_held_comp", "d_flat", "d_flat_comp", "d_excess", "d_excess_comp")}}
    out["matched"] = M
    out["unmatched"] = U
    allrows = [r for T in M for r in M[T]["rows"]]
    G = {"n_matched_rows": len(allrows)}
    for key in ("d_C", "d_C_comp", "d_held", "d_held_comp", "d_flat", "d_flat_comp", "d_excess", "d_excess_comp", "r1_ext_int", "r2_pol_ext_int"):
        G["max_abs_" + key] = max(abs(r[key]) for r in allrows)
        G["range_" + key] = [min(r[key] for r in allrows), max(r[key] for r in allrows)]
    G["max_abs_d_C_raw"] = max(abs(r["d_C_raw"]) for r in allrows if r["d_C_raw"] is not None)
    G["range_d_C_raw"] = [min(r["d_C_raw"] for r in allrows if r["d_C_raw"] is not None), max(r["d_C_raw"] for r in allrows if r["d_C_raw"] is not None)]
    G["n_rows_raw"] = sum(1 for r in allrows if r["d_C_raw"] is not None)
    G["all_d_C_negative"] = all(r["d_C"] < 0 for r in allrows)
    G["all_d_excess_negative"] = all(r["d_excess"] < 0 for r in allrows)
    G["all_r2_delta_positive"] = all(r["r2_pol_ext_int"] > 0 for r in allrows)
    G["all_r1_ext_int_positive"] = all(r["r1_ext_int"] > 0 for r in allrows)
    G["all_err_ineq_excess"] = all(r["err_ineq_excess"] for r in allrows)
    G["all_err_ineq_C"] = all(r["err_ineq_C"] for r in allrows)
    G["n_within"] = {key: sum(1 for r in allrows if r[key + "_within"]) for key in ("d_C", "d_C_comp", "d_held", "d_held_comp", "d_flat", "d_flat_comp", "d_excess", "d_excess_comp")}
    # route 1 minus route 2 on the exterior, relative to route 2's own exterior-minus-interior asymmetry
    G["range_r1_S_X_minus_r2_pol_int"] = [min(r["r1_S_X_minus_r2_pol_int"] for r in allrows), max(r["r1_S_X_minus_r2_pol_int"] for r in allrows)]
    G["range_d_C_plus_r2_delta"] = G["range_r1_S_X_minus_r2_pol_int"]
    G["range_d_C_minus_r1_ext_int_plus_r2_delta"] = [min(r["d_C"] - r["r1_ext_int"] + r["r2_pol_ext_int"] for r in allrows), max(r["d_C"] - r["r1_ext_int"] + r["r2_pol_ext_int"] for r in allrows)]
    # route 2's own global-entropy diagnostic of the pure polar state against its exterior-minus-interior asymmetry
    G["r2_pol_global_vs_delta_ratio_range"] = [min(r["r2_pol_global_entropy"] / r["r2_pol_ext_int"] for r in allrows), max(r["r2_pol_global_entropy"] / r["r2_pol_ext_int"] for r in allrows)]
    G["r2_pol_global_minus_delta_range"] = [min(r["r2_pol_global_entropy"] - r["r2_pol_ext_int"] for r in allrows), max(r["r2_pol_global_entropy"] - r["r2_pol_ext_int"] for r in allrows)]
    Wd = {T: M[T]["weighted"]["excess"][2] for T in M}
    G["weighted_excess_diff"] = Wd
    G["weighted_excess_diff_range"] = [min(Wd.values()), max(Wd.values())]
    G["factor_over_TOL_W_range"] = [min(M[T]["weighted"]["factor_over_TOL_W"] for T in M), max(M[T]["weighted"]["factor_over_TOL_W"] for T in M)]
    G["factor_over_1p85e-7_range"] = [min(abs(Wd[T]) / 1.85e-7 for T in M), max(abs(Wd[T]) / 1.85e-7 for T in M)]
    G["relative_range"] = [min(M[T]["weighted"]["relative_excess_difference"] for T in M), max(M[T]["weighted"]["relative_excess_difference"] for T in M)]
    G["weighted_excess_comp_diff"] = {T: M[T]["weighted"]["excess_comp"][2] for T in M}
    G["max_abs_weighted_excess_comp_diff"] = max(abs(M[T]["weighted"]["excess_comp"][2]) for T in M)
    G["weighted_held_diff"] = {T: M[T]["weighted"]["held"][2] for T in M}
    G["weighted_flat_diff"] = {T: M[T]["weighted"]["flat"][2] for T in M}
    G["weighted_C_diff"] = {T: M[T]["weighted"]["C"][2] for T in M}
    G["weighted_C_comp_diff"] = {T: M[T]["weighted"]["C_comp"][2] for T in M}
    G["weighted_held_comp_diff"] = {T: M[T]["weighted"]["held_comp"][2] for T in M}
    G["weighted_flat_comp_diff"] = {T: M[T]["weighted"]["flat_comp"][2] for T in M}
    G["TOL_W_exact"] = C.TOL_W
    G["endpoints"] = {"T15": M[15]["weighted"]["excess"][:2] if 15 in M else None, "T40": M[40]["weighted"]["excess"][:2] if 40 in M else None}
    out["statement"] = G

    # ---------------- the error statement
    E = {"clocks": {}}
    diag_rows = {(int(r["clock_rs"]), int(r["kappa"])): r for r in r2d["rows"]}
    diag_sum = {int(s["clock_rs"]): s for s in r2d["summaries"]}
    for T in M:
        sh = M[T]["shift"]
        t2 = R2[T]["by_shift"][sh]["totals_mine"]
        gap = M[T]["weighted"]["excess"][2]
        E["clocks"][T] = {"E2_mine": t2["E2_emp"], "E2_printed": ERR_E2[T], "E2_mine_minus_printed": t2["E2_emp"] - ERR_E2[T],
                          "E2_diag_json": diag_sum[T]["weighted_absolute_delta_E2_empirical"], "signed_delta_weighted": t2["signed_delta"],
                          "E2_equals_signed_delta": abs(t2["E2_emp"] - t2["signed_delta"]) < 1e-18,
                          "gap_mine": gap, "gap_printed": ERR_GAP[T], "gap_agrees_2dec": abs(gap - ERR_GAP[T]) <= 0.5e-8 + 1e-15,
                          "ineq_weighted": abs(gap) <= t2["E2_emp"] + C.TOL_W, "ineq_weighted_1p85": abs(gap) <= t2["E2_emp"] + 1.85e-7,
                          "abs_gap_minus_E2": abs(gap) - t2["E2_emp"], "E2_over_2": t2["E2_emp"] / 2.0,
                          "diag_delta_max_abs_mismatch": max(abs(diag_rows[(T, k)]["delta_exterior_minus_interior"] - R2[T]["by_shift"][sh]["rows"][k - 1]["delta"]) for k in C.KAPPAS)}
    E["delta_range_mine"] = G["range_r2_pol_ext_int"]
    E["delta_range_printed"] = ERR_DELTA_RANGE
    E["all_60_positive"] = G["all_r2_delta_positive"]
    E["ineq_all_rows_excess"] = G["all_err_ineq_excess"]
    E["ineq_all_rows_C"] = G["all_err_ineq_C"]
    E["quoted_complement_max_2p450e-11"] = G["max_abs_d_C_comp"]
    E["quoted_held_flat_ext_max_1p990e-10"] = max(G["max_abs_d_held"], G["max_abs_d_flat"])
    E["quoted_weighted_comp_excess_max_4p751e-9"] = G["max_abs_weighted_excess_comp_diff"]
    E["factor_14_range"] = G["factor_over_1p85e-7_range"]
    out["error_statement"] = E

    # ---------------- purity-bound context (own weighted complement-minus-held against route 1's printed excess)
    P = {}
    for T in M:
        sh = M[T]["shift"]
        t2 = R2[T]["by_shift"][sh]["totals_mine"]
        dh = t2["pol_int_excess"]
        p_full = R1[T]["weighted_mine"]["excess"]
        P[T] = {"delta_hat_mine": dh, "delta_hat_printed": PB_DELTA[T], "mine_minus_printed": dh - PB_DELTA[T],
                "G_with_printed_p": abs(dh - PB_PRINTED_P[T]) + 0.5e-9, "G_printed": PB_G[T],
                "gap_full_precision": dh - p_full, "r1_excess_comp_minus_r2_comp_excess": R1[T]["weighted_mine"]["excess_c"] - dh}
    out["purity_bound_context"] = P

    # ---------------- cell-by-cell check of the delivery's tables
    out["cells"] = check_cells(M, R1, R2, U, G)
    C.jdump(out, C.OUT + "/arith.json")
    write_summary(out)


NUM = re.compile(r"^[+-]?\d+(\.\d+)?([eE][+-]?\d+)?$")


def agree(ptxt, v):
    """printed text against my value, at the printed precision (half a unit in the last printed digit)."""
    s = ptxt.strip().replace("\\(", "").replace("\\)", "").replace("\\times10^{", "e").replace("}", "")
    s = s.replace("−", "-")
    if not NUM.match(s):
        return None
    p = float(s)
    if v is None:
        return {"printed": p, "mine": None, "ok": False}
    m = re.match(r"^[+-]?\d+(?:\.(\d+))?(?:[eE]([+-]?\d+))?$", s)
    dec = len(m.group(1)) if m.group(1) else 0
    ex = int(m.group(2)) if m.group(2) else 0
    tol = 0.5 * 10.0 ** (ex - dec) * (1 + 1e-9) + 1e-300
    return {"printed": p, "mine": v, "ok": abs(p - v) <= tol, "tol": tol}


def parse_tables(path):
    tables = []
    cur = None
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("|"):
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                if cur is None:
                    cur = {"header": cells, "rows": []}
                elif all(set(c) <= set("-: ") for c in cells):
                    continue
                else:
                    cur["rows"].append(cells)
            else:
                if cur is not None:
                    tables.append(cur)
                    cur = None
                if line.startswith("- max "):
                    tables.append({"maxline": line})
    if cur is not None:
        tables.append(cur)
    return tables


def check_cells(M, R1, R2, U, G):
    res = {"checked": 0, "mismatch": [], "unparsed": 0, "tables": []}
    if not os.path.exists(DELIV_TABLES):
        res["note"] = "tables file not copied"
        return res

    def chk(where, ptxt, v):
        a = agree(ptxt, v)
        if a is None:
            res["unparsed"] += 1
            return
        res["checked"] += 1
        if not a["ok"]:
            res["mismatch"].append({"where": where, **a})

    tabs = [t for t in parse_tables(DELIV_TABLES) if "header" in t]
    maxlines = [t["maxline"] for t in parse_tables(DELIV_TABLES) if "maxline" in t]
    assert len(tabs) == 4, len(tabs)
    # table 0: clock table
    for cells in tabs[0]["rows"]:
        T = int(cells[0])
        res["tables"].append(("clock", T))
        if T in M:
            W = M[T]["weighted"]
            chk(("clock", T, "r1 excess"), cells[6], W["excess"][0])
            chk(("clock", T, "r2 polar excess"), cells[7], W["excess"][1])
            chk(("clock", T, "difference"), cells[8], W["excess"][2])
            if (cells[9] == "NO") != (not W["excess_within_TOL_W"]):
                res["mismatch"].append({"where": ("clock", T, "within flag"), "printed": cells[9], "mine": W["excess_within_TOL_W"]})
            chk(("clock", T, "complement diff"), cells[10], W["excess_comp"][2])
            if (cells[11] == "yes") != W["excess_comp_within_TOL_W"]:
                res["mismatch"].append({"where": ("clock", T, "comp within flag"), "printed": cells[11], "mine": W["excess_comp_within_TOL_W"]})
            m = re.search(r"shift \+(\d) \((\d+) sites\)", cells[5])
            if m and (int(m.group(1)) != M[T]["shift"] or int(m.group(2)) != M[T]["n_in"]):
                res["mismatch"].append({"where": ("clock", T, "shift/sites"), "printed": cells[5], "mine": [M[T]["shift"], M[T]["n_in"]]})
        else:
            chk(("clock", T, "r1 excess"), cells[6], U[T]["r1_excess"])
        m = re.match(r"([\d.]+) .*; (\d+)$", cells[2])
        chk(("clock", T, "r1 R/a"), m.group(1), R1[T]["R_cut"])
        if int(m.group(2)) != R1[T]["n_in"]:
            res["mismatch"].append({"where": ("clock", T, "r1 sites"), "printed": m.group(2), "mine": R1[T]["n_in"]})
        m = re.match(r"([\d.]+) .*; (\d+)$", cells[3])
        chk(("clock", T, "r2 R/a"), m.group(1), R2[T]["radius"])
        if int(m.group(2)) != R2[T]["primary_n_in"]:
            res["mismatch"].append({"where": ("clock", T, "r2 sites"), "printed": m.group(2), "mine": R2[T]["primary_n_in"]})
        saved = sorted(int(x) for x in cells[4].split(","))
        if saved != sorted(R2[T]["partitions"].values()):
            res["mismatch"].append({"where": ("clock", T, "partitions saved"), "printed": cells[4], "mine": sorted(R2[T]["partitions"].values())})
        if cells[1] != R1[T]["dtype"]:
            res["mismatch"].append({"where": ("clock", T, "dtype"), "printed": cells[1], "mine": R1[T]["dtype"]})
    # table 1: side by side at shift 0
    for cells in tabs[1]["rows"]:
        T = int(cells[0])
        chk(("side", T, "r1 excess"), cells[1], R1[T]["weighted_mine"]["excess"])
        chk(("side", T, "r2 polar excess shift0"), cells[2], R2[T]["by_shift"][0]["totals_mine"]["orthonormal_control_excess"])
        if cells[3] != "unavailable":
            chk(("side", T, "r2 raw excess shift0"), cells[3], R2[T]["by_shift"][0]["totals_mine"].get("raw_excess"))
        elif R2[T]["by_shift"][0]["raw_all"]:
            res["mismatch"].append({"where": ("side", T, "raw availability"), "printed": "unavailable", "mine": "available"})
    # table 2: weighted components
    for cells in tabs[2]["rows"]:
        T = int(cells[0])
        W = M[T]["weighted"]
        for j, (key, idx) in enumerate([("C", 0), ("C", 1), ("C", 2), ("held", 0), ("held", 1), ("held", 2), ("flat", 0), ("flat", 1), ("flat", 2)]):
            chk(("wcomp", T, key, idx), cells[1 + j], W[key][idx])
        if cells[10].startswith("unavailable"):
            if "C_raw" in W:
                res["mismatch"].append({"where": ("wcomp", T, "raw availability"), "printed": cells[10], "mine": "available"})
        else:
            chk(("wcomp", T, "C_raw", 1), cells[10], W["C_raw"][1])
            chk(("wcomp", T, "C_raw", 2), cells[11], W["C_raw"][2])
    # table 3: per channel
    for cells in tabs[3]["rows"]:
        T, k = int(cells[0]), int(cells[1])
        r = M[T]["rows"][k - 1]
        assert r["kappa"] == k
        chk(("chan", T, k, "r1 S_X"), cells[2], r["r1_S_X"])
        chk(("chan", T, k, "r2 pol ext"), cells[3], r["r2_pol_ext"])
        chk(("chan", T, k, "d ext"), cells[4], r["d_C"])
        chk(("chan", T, k, "d comp"), cells[5], r["d_C_comp"])
        a, b = [x.strip() for x in cells[6].split(";")]
        chk(("chan", T, k, "d held ext"), a, r["d_held"])
        chk(("chan", T, k, "d held comp"), b, r["d_held_comp"])
        a, b = [x.strip() for x in cells[7].split(";")]
        chk(("chan", T, k, "d flat ext"), a, r["d_flat"])
        chk(("chan", T, k, "d flat comp"), b, r["d_flat_comp"])
        chk(("chan", T, k, "d excess ext"), cells[8], r["d_excess"])
        if cells[9] != "unavailable":
            chk(("chan", T, k, "d C vs raw"), cells[9], r["d_C_raw"])
        elif r["d_C_raw"] is not None:
            res["mismatch"].append({"where": ("chan", T, k, "raw availability"), "printed": "unavailable", "mine": r["d_C_raw"]})
        chk(("chan", T, k, "r1 ext-int"), cells[10], r["r1_ext_int"])
        chk(("chan", T, k, "r2 pol ext-int"), cells[11], r["r2_pol_ext_int"])
        if (cells[12] == "yes") != r["d_C_within"]:
            res["mismatch"].append({"where": ("chan", T, k, "ext within"), "printed": cells[12], "mine": r["d_C_within"]})
        if (cells[13] == "yes") != r["d_C_comp_within"]:
            res["mismatch"].append({"where": ("chan", T, k, "comp within"), "printed": cells[13], "mine": r["d_C_comp_within"]})
    # the maxima lines
    want = {"max |d C| =": G["max_abs_d_C"], "max |d C_complement| =": G["max_abs_d_C_comp"], "max |d held| =": G["max_abs_d_held"],
            "max |d held_complement| =": G["max_abs_d_held_comp"], "max |d flat| =": G["max_abs_d_flat"], "max |d flat_complement| =": G["max_abs_d_flat_comp"],
            "max |d excess_polar| =": G["max_abs_d_excess"], "max |r1 ext-int| =": G["max_abs_r1_ext_int"], "max |r2 polar ext-int| =": G["max_abs_r2_pol_ext_int"],
            "max |d C vs r2 raw| =": G["max_abs_d_C_raw"]}
    for line in maxlines:
        for key, v in want.items():
            if key in line:
                seg = line.split(key)[1].strip().split(";")[0].strip()
                chk(("max", key), seg, v)
    res["n_maxlines"] = len(maxlines)
    return res


def write_summary(out):
    G = out["statement"]
    M = out["matched"]
    L = []
    L.append("ec_arith.py: recomputation from the stored jsons (own code)")
    L.append("inputs:")
    for k, v in out["inputs"].items():
        L.append("  %s: %s  %s" % (k, v[1], v[0]))
    r1 = out["route1"]
    L.append("route 1 (prod, 100 channel rows): max |purity gap| %.4e, max |identity defect| %.4e; max |stored - recomputed| of excess/purity %.1e; weighted mine - stored max %.1e"
             % (r1["prod_max_abs_purity_gap_100rows"], r1["prod_max_abs_identity_defect_100rows"], max(r1["clocks"][T]["max_abs_chk"] for T in r1["clocks"]),
                max(abs(v) for T in r1["clocks"] for v in r1["clocks"][T]["weighted_mine_minus_stored"].values())))
    if "s8_max_abs_purity_gap" in r1:
        L.append("route 1 (s8, %d rows): max |purity gap| %.4e, max |identity defect| %.4e" % (r1["s8_rows"], r1["s8_max_abs_purity_gap"], r1["s8_max_abs_identity_defect"]))
    L.append("route 1 ext-int ranges over the 100 prod rows: C [%.3e, %.3e], held [%.3e, %.3e], flat [%.3e, %.3e]" % tuple(r1["prod_range_ext_minus_int_C"] + r1["prod_range_ext_minus_int_vac"] + r1["prod_range_ext_minus_int_P0"]))
    L.append("route 1 prod excess range %.6f (T%d) to %.6f (T%d)" % (r1["prod_excess_range"][0], r1["prod_excess_argrange"][0], r1["prod_excess_range"][1], r1["prod_excess_argrange"][1]))
    if "s8_excess_range" in r1:
        L.append("route 1 s8 excess range %.6f to %.6f: %s" % (r1["s8_excess_range"][0], r1["s8_excess_range"][1], {T: round(v, 4) for T, v in r1["s8_excess"].items()}))
    L.append("route 2: max |my totals - tables.json totals| over 10 clocks x 3 shifts: %.2e" % out["route2"]["max_abs_mine_minus_stored_totals"])
    for T in C.CLOCKS:
        R = out["route1"]["clocks"][T]
        Q = out["route2"]["clocks"][T]
        L.append("  T%d: r1 n_in %d (R_cut %.4f, %s; floor %d); r2 primary n_in %d (R %.4f, %s; floor %d), partitions %s; matched: %s"
                 % (T, R["n_in"], R["R_cut"], R["source"], R["n_in_from_R_cut"], Q["primary_n_in"], Q["radius"], Q["cut_label"], Q["n_in_from_radius"], Q["partitions"],
                    ("shift %+d" % M[T]["shift"]) if T in M else "none"))
    L.append("matched clocks: %s" % sorted(M))
    L.append("per-channel maxima over the %d matched rows:" % G["n_matched_rows"])
    for key in ("d_C", "d_C_comp", "d_held", "d_held_comp", "d_flat", "d_flat_comp", "d_excess", "d_excess_comp", "r1_ext_int", "r2_pol_ext_int", "d_C_raw"):
        L.append("  max |%s| = %.4e   range [%+.4e, %+.4e]" % (key, G["max_abs_" + key], G["range_" + key][0], G["range_" + key][1]))
    L.append("  n within 2.1e-10: %s" % G["n_within"])
    L.append("  all d_C negative: %s; all d_excess negative: %s; all r2 delta positive: %s; all r1 ext-int positive: %s" % (G["all_d_C_negative"], G["all_d_excess_negative"], G["all_r2_delta_positive"], G["all_r1_ext_int_positive"]))
    L.append("  r1 S_X minus r2 polar interior (= d_C + r2 delta): range [%+.3e, %+.3e]" % tuple(G["range_r1_S_X_minus_r2_pol_int"]))
    L.append("  d_C - r1 ext-int + r2 delta (= r1 S_Xc - r2 polar interior): range [%+.3e, %+.3e]" % tuple(G["range_d_C_minus_r1_ext_int_plus_r2_delta"]))
    L.append("  route 2's polar_global entropy / its ext-int delta: ratio range [%.3f, %.3f]; difference range [%+.2e, %+.2e]" % tuple(G["r2_pol_global_vs_delta_ratio_range"] + G["r2_pol_global_minus_delta_range"]))
    L.append("weighted (16 kappa) at the matched partitions:")
    for T in sorted(M):
        W = M[T]["weighted"]
        L.append("  T%d shift %+d n_in %d: excess r1 %.9f r2 %.9f d %+.4e (|d|/1.848e-7 = %.2f, rel %+.2e) | comp-side excess d %+.4e | C d %+.3e C_comp d %+.3e held d %+.3e held_comp d %+.3e flat d %+.3e flat_comp d %+.3e | E2 %.6e, weighted polar_global %.6e, r1 ext-comp excess diff %+.3e"
                 % (T, M[T]["shift"], M[T]["n_in"], W["excess"][0], W["excess"][1], W["excess"][2], W["factor_over_TOL_W"], W["relative_excess_difference"], W["excess_comp"][2],
                    W["C"][2], W["C_comp"][2], W["held"][2], W["held_comp"][2], W["flat"][2], W["flat_comp"][2], W["r2_E2_emp"], W["r2_weighted_pol_global_entropy"], W["r1_excess_ext_minus_comp"]))
        if "C_raw" in W:
            L.append("      raw: C d %+.3e, excess d %+.3e" % (W["C_raw"][2], W["excess_raw"][2]))
        L.append("      row-wise sums of the differences: excess %+.6e (sum-order noise %+.1e), comp excess %+.6e (noise %+.1e), C_comp %+.4e, held_comp %+.4e, flat_comp %+.4e, held %+.6e, flat %+.6e"
                 % (W["excess_diff_rowwise"], W["excess_diff_sumorder_noise"], W["excess_comp_diff_rowwise"], W["excess_comp_diff_sumorder_noise"], W["C_comp_diff_rowwise"],
                    W["held_comp_diff_rowwise"], W["flat_comp_diff_rowwise"], W["held_diff_rowwise"], W["flat_diff_rowwise"]))
    L.append("  weighted excess diff range [%+.4e, %+.4e]; factor over 1.848e-7: [%.2f, %.2f]; over 1.85e-7: [%.2f, %.2f]; relative [%+.2e, %+.2e]; max |comp-side weighted excess diff| %.3e"
             % (G["weighted_excess_diff_range"][0], G["weighted_excess_diff_range"][1], G["factor_over_TOL_W_range"][0], G["factor_over_TOL_W_range"][1], G["factor_over_1p85e-7_range"][0], G["factor_over_1p85e-7_range"][1], G["relative_range"][0], G["relative_range"][1], G["max_abs_weighted_excess_comp_diff"]))
    L.append("unmatched clocks (side by side, shift 0): %s" % {T: (round(U["r1_excess"], 6), round(U["r2_primary_pol_excess"], 6), (round(U["r2_primary_raw_excess"], 6) if U["r2_primary_raw_excess"] is not None else None)) for T, U in out["unmatched"].items()})
    E = out["error_statement"]
    L.append("error statement: delta range mine [%.5e, %.5e] printed %s; all positive %s; per-row inequalities (excess, C): %s %s" % (E["delta_range_mine"][0], E["delta_range_mine"][1], E["delta_range_printed"], E["all_60_positive"], E["ineq_all_rows_excess"], E["ineq_all_rows_C"]))
    for T in sorted(E["clocks"]):
        e = E["clocks"][T]
        L.append("  T%d: E2 mine %.6e printed %.6e (diff %+.1e; diag json %.6e; E2 == signed sum: %s); gap mine %+.4e printed %+.2e (ok %s); |gap| <= E2 + 1.848e-7: %s; |gap| - E2 = %+.3e"
                 % (T, e["E2_mine"], e["E2_printed"], e["E2_mine_minus_printed"], e["E2_diag_json"], e["E2_equals_signed_delta"], e["gap_mine"], e["gap_printed"], e["gap_agrees_2dec"], e["ineq_weighted"], e["abs_gap_minus_E2"]))
    L.append("  quoted maxima: complement %.4e (printed 2.450e-11), held/flat exterior %.4e (printed 1.990e-10), weighted complement excess %.4e (printed 4.751e-9); factor range over 1.85e-7 [%.2f, %.2f]"
             % (E["quoted_complement_max_2p450e-11"], E["quoted_held_flat_ext_max_1p990e-10"], E["quoted_weighted_comp_excess_max_4p751e-9"], E["factor_14_range"][0], E["factor_14_range"][1]))
    L.append("purity-bound context (own weighted complement minus held vs the printed Delta-hat and G):")
    for T in sorted(out["purity_bound_context"]):
        p = out["purity_bound_context"][T]
        L.append("  T%d: Delta-hat mine %.11f printed %.11f (diff %+.1e); G mine %.9e printed %.9e; full-precision gap (r2 comp excess - r1 ext excess) %+.4e; r1 comp excess - r2 comp excess %+.4e"
                 % (T, p["delta_hat_mine"], p["delta_hat_printed"], p["mine_minus_printed"], p["G_with_printed_p"], p["G_printed"], p["gap_full_precision"], p["r1_excess_comp_minus_r2_comp_excess"]))
    c = out["cells"]
    L.append("cell-by-cell against the delivery's tables: %d numeric cells checked at printed precision, %d mismatches, %d unparsed, %d maxima lines" % (c.get("checked", 0), len(c.get("mismatch", [])), c.get("unparsed", 0), c.get("n_maxlines", 0)))
    for m in c.get("mismatch", [])[:50]:
        L.append("  MISMATCH %s" % (m,))
    txt = "\n".join(L) + "\n"
    with open(C.OUT + "/arith_summary.txt", "w") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
