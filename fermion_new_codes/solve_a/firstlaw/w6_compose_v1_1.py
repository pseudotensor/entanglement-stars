#!/usr/bin/env python
"""w6_compose_v1_1.py (the solve lane): writes W6_FIRSTLAW_rungs_v1.1.md from w6_readout_v1_1.json, a notes json and a changes json (no run file
is opened; every number is the producer's). Version 1's composer with the cold reading's sections added (R1 the K_1-weight columns; R3 the
largest-term columns; M4 the inner-root block; the changes list; the numbers for the paper's R4 and R5). Usage: python w6_compose_v1_1.py
--readout JSON --notes JSON --changes JSON --out FILE"""
import argparse
import json
import math


def f(x, fmt="%.4g"):
    if x is None:
        return "-"
    try:
        if isinstance(x, float) and not math.isfinite(x):
            return "-"
        return fmt % x
    except TypeError:
        return str(x)


def e(x):
    return f(x, "%.2e")


def table(h, rows):
    return "\n".join(["| " + " | ".join(h) + " |", "|" + "---|" * len(h)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readout", required=True)
    ap.add_argument("--notes", required=True)
    ap.add_argument("--changes", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    D = json.load(open(a.readout)); N = json.load(open(a.notes)); CH = json.load(open(a.changes))
    R = D["rungs"]; order = D["order"]
    o = ["# W6_FIRSTLAW_rungs_v1.1: FIRSTLAW_records v2.3 (the exact fixed-radius mass-rate identity in 4 columns, the rows' rate, the full-clock "
         "residual and the gate at every dense outer root) on the averaging ladder at (0.2, 64), sigma/a 2, 4 and 8 at fixed preparation", ""]
    o.append("Version 1.1 of W6_FIRSTLAW_rungs_v1 (which stays as issued), with the corrections of its cold reading (%s); the list \"Changes from v1\" "
             "names each reading item and what changed. Version 1's text follows, corrected." % CH["reading"])
    o.append("")
    o.append("Composed by w6_readout_v1_1.py (version 1's w6_readout_v1.py unchanged, extended), w6_k1weight_v1_1.py (the K_1 weight) and w6_compose_v1_1.py from the outputs of the busi lane's firstlaw_records_v4b.py (unchanged), run on the "
             "outputs of firstlaw_records_v3b.py (unchanged) on each rung's stored records; box CPU, no engine rerun. The builder composes; no "
             "sentence here reads the physics. The definitions are RUN_firstlaw_records_v4b.md's: column 1 the flux through the old cut (the "
             "sharp-cut balance of the state's charge), column 2 the cut-assignment correction, column 3 the averaging commutator, column 4 the "
             "continued-reference term; rep their sum (the kernel representation of Delta M; in version 1.1 with the energy part of the site charge "
             "weighted by the cumulative weight of K_0 and the momentum part by that of K_1, the engine's kernels for rho and J); the representation defect Delta M - rep; the rows' "
             "rate 4 pi N R^2 T_ll at the root (rows_mid the mean of the a- and b-side rates at the fixed radius); the defect Delta M/dt - rows_mid; "
             "the full-clock residual s_t/N + (1 + s^2)(ln N)' - 4 pi G r (rho + p_r) at the root by the midpoint rule; the gate sigma/R with "
             "sigma = ell. Rows are labelled by the averaging scale and the kernel route of the identity (analytic: the shell Gaussian in closed "
             "form; engine: the engine's own kernel rebuilt from its SmoothedSources, the RUN note's fallback when the analytic kernel check is "
             "of order 1).")
    o.append("")
    o.append("## 0 Headline")
    o.append("")
    for line in N.get("headline", []):
        o.append("- " + line)
    o.append("")
    o.append("## Changes from v1")
    o.append("")
    for line in CH["changes"]:
        o.append("- " + line)
    o.append("")
    o.append("## 1 The rungs: inputs and certification")
    o.append("")
    rows = []
    for k in order:
        r = R[k]; c = r.get("certification") or {}
        rows.append([k, f(r["sigma"], "%g"), r["kernel"], r["records_json"], c.get("member", "-"),
                     "%s (%s)" % (f(c.get("last_converged"), "%.4f") if c.get("last_converged") is not None else "none",
                                  ("first non-counting %s: %s" % (f(c.get("first_noncounting"), "%.4f"), c.get("first_noncounting_gate"))) if c.get("first_noncounting") is not None else "every record counts"),
                     r.get("pairs_beyond_certified", "-"), r["outer"]["n_pairs"], "%s to %s" % (f(r["outer"].get("T_first"), "%.3f"), f(r["outer"].get("T_last"), "%.3f")),
                     r["inner"]["n_pairs"], f(r.get("seconds"), "%.1f")])
    o.append(table(["rung / kernel", "sigma/a", "kernel", "records json", "ladder member", "last converged record (MASS/SCALING v1.1 rule)",
                    "pairs after the certified prefix", "outer-root pairs", "T/r_s span", "inner-root pairs", "v4b seconds"], rows))
    o.append("")
    o.append("## 2 The identity's closure and its checks (outer roots)")
    o.append("")
    rows = []
    for k in order:
        s = R[k]["outer"]
        rows.append([k, e(s.get("closure_max")), "%s (%s; %s)" % (e(s.get("rep_defect_max")), e(s.get("rep_defect_rel_max")), e(s.get("rep_defect_rel_median"))), e(s.get("rep_defect_median")),
                     e(s.get("M_rep_minus_quad_max")), e(s.get("M_quad_minus_ode_max")), "%s; %s (median %s)" % (e(s.get("kernel_rel_rho_max")), e(s.get("kernel_rel_J_max")), e(s.get("kernel_rel_J_median"))),
                     e(s.get("exact_relation_max")),
                     ("%s (median %s; relative median %s); %s" % (e(R[k]["K1_outer"]["rep_defect_K1_max"]), e(R[k]["K1_outer"]["rep_defect_K1_median"]),
                                                                 e(R[k]["K1_outer"]["rep_defect_K1_rel_median"]), e(R[k]["K1_outer"]["M_rep_K1_minus_quad_max"]))) if "K1_outer" in R[k] else "-"])
    o.append(table(["rung / kernel", "max |sum4 - rep| (closure)", "version 1's K_0 momentum weight: max |Delta M - rep| (max relative to |Delta M|, over pairs whose Delta M is near 0; median relative)", "median |Delta M - rep|",
                    "max |M_rep - M_quad|", "max |M_quad - M_ode|", "kernel check rho; J (max relative)", "exact relation at fixed R (max)",
                    "K_1 momentum weight: max |Delta M - rep| (median; median relative); max |M_rep - M_quad|"], rows))
    o.append("")
    o.append("## 3 The 4 columns (sums over the outer-root pairs)")
    o.append("")
    o.append("### 3a With the engine's K_1 weight on the momentum part (R1; the columns read as the mass rate's decomposition)")
    o.append("")
    rows = []
    for k in order:
        if "K1_outer" not in R[k]:
            continue
        s = R[k]["K1_outer"]; v = R[k]["K1_validation"]
        rows.append([k, s["n_pairs"], f(s["sum_dM"]), f(s["sum_rep_K1"]), f(s["sum_col1"]), f(s["sum_col2_K1"]), f(s["sum_col3"]), f(s["sum_col4_K1"]),
                     "%s; %s; %s; %s" % (f(s["max_abs_col1"]), f(s["max_abs_col2_K1"]), f(s["max_abs_col3"]), f(s["max_abs_col4_K1"])),
                     "%s; %s" % (e(max(v.values())), e(R[k]["K1_quadrature_check"]))])
    o.append(table(["rung (engine route)", "pairs", "sum Delta M", "sum rep (K_1)", "sum col1", "sum col2 (K_1)", "sum col3", "sum col4", "max |col1|; |col2|; |col3|; |col4| per pair",
                    "validation: max |v1.1 with K_0 weight - v4b| over col2, col4, rep, M_rep; the weight quadrature's check on K_0"], rows))
    o.append("")
    tw = R[order[0]].get("K1_total_weight_at_sites") if "K1_total_weight_at_sites" in R[order[0]] else None
    o.append("The total weight of K_1 inside the grid, int 4 pi s^2 K_1,n ds, per site radius R_n (a), against exactly 1 for K_0: " + "; ".join(
        "sigma/a %g: %s" % (R[k]["sigma"], ", ".join("R_n %s %.3f" % (x, y) for x, y in R[k]["K1_total_weight_at_sites"].items()))
        for k in order if "K1_total_weight_at_sites" in R[k]) + ".")
    o.append("")
    o.append("### 3b With version 1's K_0 weight on the momentum part (not read: the representation defect of R1)")
    o.append("")
    rows = []
    for k in order:
        s = R[k]["outer"]
        rows.append([k, s["n_pairs"], f(s.get("sum_dM")), f(s.get("sum_rep")), "%s (flow-change part %s)" % (f(s.get("sum_col1_flux_old_cut")), f(s.get("sum_col1_flow_change_part"))),
                     f(s.get("sum_col2_cut_assignment")), f(s.get("sum_col3_averaging_commutator")), f(s.get("sum_col4_continued_reference")),
                     "%s; %s; %s; %s" % (f(s.get("max_abs_col1_flux_old_cut")), f(s.get("max_abs_col2_cut_assignment")), f(s.get("max_abs_col3_averaging_commutator")),
                                         f(s.get("max_abs_col4_continued_reference")))])
    o.append(table(["rung / kernel", "pairs", "sum Delta M", "sum rep", "sum col1 (flux through the old cut)", "sum col2 (cut assignment)",
                    "sum col3 (averaging commutator)", "sum col4 (continued reference)", "max |col1|; |col2|; |col3|; |col4| per pair"], rows))
    o.append("")
    o.append("## 4 The rows' rate against the records' rate at the root (the defect)")
    o.append("")
    rows = []
    for k in order:
        s = R[k]["outer"]
        rows.append([k, f(s.get("sum_rate")), f(s.get("sum_rows_mid")), f(s.get("sum_defect_mid")), f(s.get("median_abs_defect_over_rows"), "%.3f"),
                     f(s.get("same_sign_fraction"), "%.3f"), "%s; %s; %s of %s" % (s.get("rows_negative"), s.get("rate_negative"), s.get("Tll_a_negative"), s["n_pairs"]),
                     "%s; %s" % (f(s.get("sum_rows_a_nostress")), f(s.get("sum_rows_a_stress"))),
                     f(R[k]["R3_outer"]["median_abs_defect_over_largest"], "%.3f"), "%s; %s" % (f(R[k]["R3_outer"]["median_abs_rows_over_largest"], "%.3f"), f(R[k]["R3_outer"]["median_abs_rate_over_largest"], "%.3f")),
                     f(R[k]["R3_outer"]["frac_defect_gt_rows"], "%.2f"),
                     "%s; %s; %s" % (f(R[k]["R3_outer"]["median_2J_over_rho"], "%.2f"), f(R[k]["R3_outer"]["median_p_over_rho"], "%.2f"), f(R[k]["R3_outer"]["median_Tll_over_rho"], "%.2f"))])
    o.append(table(["rung / kernel", "sum Delta M/dt", "sum rows_mid", "sum defect", "median |defect|/|rows_mid|", "same-sign fraction",
                    "negative rows_mid; negative rate; negative T_ll(a)", "stress share sums: (rho - 2J) part; p_r part",
                    "median |defect|/L (L = 4 pi N R^2 max(|rho|, |p_r|, |2J|) at the root of a)", "median |rows_mid|/L; |Delta M/dt|/L", "fraction |defect| > |rows_mid|",
                    "median 2J/rho; p_r/rho; T_ll/rho at a"], rows))
    o.append("")
    o.append("## 5 The full-clock residual at the roots")
    o.append("")
    rows = []
    for k in order:
        s = R[k]["outer"]
        rows.append([k, e(s.get("clock_mid_median")), e(s.get("clock_mid_abs_median")), e(s.get("clock_mid_abs_max")), "%s (|.| max %s)" % (e(s.get("clock_a_median")), e(s.get("clock_a_abs_max"))),
                     "%s; %s; %s" % (e(s.get("clock_term_st_over_N_median")), e(s.get("clock_term_clock_median")), e(s.get("clock_term_matter_median"))),
                     "%s (quartiles %s to %s; 90th percentile %s)" % (f(R[k]["R3_outer"]["clock_over_largest_median"], "%.3f"), f(R[k]["R3_outer"]["clock_over_largest_q25"], "%.3f"),
                                                                      f(R[k]["R3_outer"]["clock_over_largest_q75"], "%.3f"), f(R[k]["R3_outer"]["clock_over_largest_q90"], "%.3f")),
                     "%s (median (sigma/R)^2 %s)" % (f(R[k]["R3_outer"]["clock_over_largest_over_gate2_median"], "%.3g"), f(R[k]["R3_outer"]["gate2_median"], "%.4f"))])
    o.append(table(["rung / kernel", "median (midpoint)", "|.| median", "|.| max", "endpoint median", "median terms: s_t/N; (1 + s^2)(ln N)'; -4 pi G r (rho + p_r)",
                    "|.| over the largest of its 3 terms, median", "per pair (|.|/largest term)/(sigma/R)^2, median"], rows))
    o.append("")
    o.append("## 6 The gate sigma/R per root")
    o.append("")
    rows = []
    for k in order:
        s = R[k]["outer"]; si = R[k]["outer"]
        rows.append([k, "%s to %s (median %s)" % (f(s.get("gate_min"), "%.4f"), f(s.get("gate_max"), "%.4f"), f(s.get("gate_median"), "%.4f")),
                     "%s of %s" % (s.get("gate_inside"), s["n_pairs"]), "%s to %s a" % (f(s.get("R_min"), "%.2f"), f(s.get("R_max"), "%.2f")),
                     "%s to %s at the records' inner zeros (%s of them, inside 0.1 at %s); %s paired (above 2 ell from the centre)" % (
                         f(si.get("gate_inner_min"), "%.3f"), f(si.get("gate_inner_max"), "%.3f"), si.get("gate_inner_n"), si.get("gate_inner_inside"), R[k]["inner"]["n_pairs"])])
    o.append(table(["rung / kernel", "outer root sigma/R", "inside 0.1", "outer root R", "inner root sigma/R"], rows))
    o.append("")
    o.append("## 6b The inner trapping root (the inner zero above 2 ell from the centre; M4, the block version 1 computed and did not print)")
    o.append("")
    rows = []
    for k in order:
        s = R[k]["inner"]; q = R[k]["R3_inner"]
        if not s.get("n_pairs"):
            continue
        rows.append([k, s["n_pairs"], "%s to %s a" % (f(q["R_min"], "%.2f"), f(q["R_max"], "%.2f")), f(q["gate_median"], "%.3f"),
                     "%s; %s; %s" % (f(s.get("sum_rate")), f(s.get("sum_rows_mid")), f(s.get("sum_defect_mid"))),
                     f(q["median_abs_defect_over_rows"], "%.3f"), f(q["median_abs_defect_over_largest"], "%.3f"), f(q["same_sign_fraction"], "%.3f"),
                     "%s; %s" % (e(s.get("clock_mid_abs_median")), f(q["clock_over_largest_median"], "%.3f")),
                     "%s of %s" % (q["Tll_a_positive"], q["n_pairs"]), "%s; %s" % (f(q["median_2J_over_rho"], "%.2f"), f(q["median_p_over_rho"], "%.2f")),
                     ("%s; %s" % (e(R[k]["K1_inner"]["rep_defect_K1_max"]), e(R[k]["K1_inner"]["M_rep_K1_minus_quad_max"]))) if R[k].get("K1_inner", {}).get("n_pairs") else "-"])
    o.append(table(["rung / kernel", "inner pairs", "inner root R", "gate median", "sum Delta M/dt; sum rows_mid; sum defect", "median |defect|/|rows_mid|",
                    "median |defect|/L", "same-sign fraction", "clock |.| median; over its largest term", "T_ll(a) positive", "median 2J/rho; p_r/rho",
                    "K_1 weight: max |Delta M - rep|; max |M_rep - M_quad|"], rows))
    o.append("")
    o.append("## 7 By phase (T_b in the stated windows; the same boundaries %s on every rung)" % ", ".join("%g" % x for x in D["phases"]))
    o.append("")
    rows = []
    for k in order:
        for p in R[k]["phases"]:
            if not p.get("n_pairs"):
                continue
            rows.append([k, "(%s, %s]" % (f(p["T_b_from"], "%g") if p["T_b_from"] > -1e8 else "-inf", f(p["T_b_to"], "%g") if p["T_b_to"] < 1e8 else "inf"), p["n_pairs"],
                         f(p.get("sum_dM")), f(p.get("sum_col1_flux_old_cut")), f(p.get("sum_col2_cut_assignment")), f(p.get("sum_col3_averaging_commutator")),
                         f(p.get("sum_col4_continued_reference")), e(p.get("rep_defect_max")), f(p.get("sum_rate")), f(p.get("sum_rows_mid")),
                         e(p.get("clock_mid_abs_median")), "%s to %s" % (f(p.get("gate_min"), "%.3f"), f(p.get("gate_max"), "%.3f"))])
    o.append(table(["rung / kernel", "T_b window", "pairs", "sum Delta M", "col1", "col2 (K_0 weight)", "col3", "col4", "max |Delta M - rep| (K_0 weight)", "sum Delta M/dt", "sum rows_mid",
                    "clock |.| median", "gate"], rows))
    o.append("")
    o.append("With the K_1 weight (engine route):")
    o.append("")
    rows = []
    for k in order:
        for p in R[k].get("K1_phases", []):
            if not p.get("n_pairs"):
                continue
            rows.append([k, "(%s, %s]" % (f(p["T_b_from"], "%g") if p["T_b_from"] > -1e8 else "-inf", f(p["T_b_to"], "%g") if p["T_b_to"] < 1e8 else "inf"), p["n_pairs"],
                         f(p["sum_dM"]), f(p["sum_col1"]), f(p["sum_col2_K1"]), f(p["sum_col3"]), f(p["sum_col4_K1"]), f(p["sum_rep_K1"]), e(p["rep_defect_K1_max"])])
    o.append(table(["rung", "T_b window", "pairs", "sum Delta M", "col1", "col2 (K_1)", "col3", "col4", "sum rep (K_1)", "max |Delta M - rep| (K_1)"], rows))
    o.append("")
    o.append("## 8 Labels and what stays open")
    o.append("")
    for line in N.get("notes", []):
        o.append("- " + line)
    o.append("")
    o.append("## 8b For the paper (the reading's R4 and R5; this lane does not edit the paper)")
    o.append("")
    for line in CH["paper"]:
        o.append("- " + line)
    o.append("")
    o.append("## 8c Where the data differ from the reading")
    o.append("")
    for line in CH["contradictions"]:
        o.append("- " + line)
    o.append("")
    o.append("## 9 SOURCES")
    o.append("")
    rows = [[k, R[k]["v4b_json"], R[k]["v4b_md5"], R[k].get("v3b_json"), R[k].get("v3b_md5")] for k in order]
    o.append(table(["rung / kernel", "v4b json", "md5", "v3b json", "md5"], rows))
    o.append("")
    for line in N.get("sources", []):
        o.append("- " + line)
    o.append("")
    open(a.out, "w").write("\n".join(o))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
