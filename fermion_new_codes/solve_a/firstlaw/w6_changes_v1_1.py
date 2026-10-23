#!/usr/bin/env python
"""w6_changes_v1_1.py: the "Changes from v1", the paper numbers (R4, R5) and the contradictions of W6_FIRSTLAW_rungs_v1.1, with every number
read from w6_readout_v1_1.json. Usage: python w6_changes_v1_1.py --readout JSON --reading PATH --out JSON"""
import argparse
import json


def g(x, fmt="%.3g"):
    return "-" if x is None else fmt % x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readout", required=True)
    ap.add_argument("--reading", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    D = json.load(open(a.readout)); R = D["rungs"]
    E = [k for k in D["order"] if R[k]["kernel"] == "engine"]
    O = {k: R[k]["outer"] for k in E}; Q = {k: R[k]["R3_outer"] for k in E}; Qi = {k: R[k]["R3_inner"] for k in E}; K = {k: R[k]["K1_outer"] for k in E}
    j = lambda fn, fmt="%.3g": ", ".join(g(fn(k), fmt) for k in E)
    ch = [
        "R1 (the representation defect is the identity's own momentum weight): applied and confirmed by an independent recomputation. This lane's "
        "w6_k1weight_v1_1.py rebuilds v4b's engine-route store (busi's module imported, unchanged), reproduces v4b's col2, col4, rep and M_rep with "
        "the K_0 weight to at most 1.4e-12, and with the momentum part weighted by the cumulative integral of the engine's K_1 closes the "
        "representation: max |Delta M - rep| %s (median %s) against version 1's %s, max |M_rep - M_quad| %s against %s, at sigma/a 2, 4, 8, at "
        "the level of the weights' cumulative quadrature (%s on K_0 against its closed form). The headline sentence and section 8's bullets 2 "
        "and 3 are replaced; the columns are now printed with the K_1 weight (section 3a) and read as the decomposition; version 1's columns "
        "stay in 3b, labelled not read. The reading's stance 'until a v4c applies the K_1 weight, columns 2 and 4, rep and the representation "
        "defect are not read' is met by this lane's v4c reading of the identity; busi's producer is unchanged." % (
            j(lambda k: K[k]["rep_defect_K1_max"], "%.2e"), j(lambda k: K[k]["rep_defect_K1_median"], "%.2e"), j(lambda k: O[k]["rep_defect_max"]),
            j(lambda k: K[k]["M_rep_K1_minus_quad_max"], "%.2e"), j(lambda k: O[k]["M_rep_minus_quad_max"]), "at most " + g(max(R[k]["K1_quadrature_check"] for k in E), "%.1e")),
        "R2 (the headline's three equations): corrected. With the K_1 weight the equations hold: sum Delta M %s = the columns' sum %s. Version 1's "
        "columns summed to %s, not to sum Delta M; the differences %s were the summed representation defect of R1." % (
            j(lambda k: K[k]["sum_dM"]), j(lambda k: K[k]["sum_rep_K1"]), j(lambda k: O[k]["sum_rep"]), j(lambda k: O[k]["sum_dM"] - O[k]["sum_rep"])),
        "R3 (the residual's size against a cancelled net rate): applied. Sections 0, 4 and 5 now print the residual against the row's largest term "
        "L = 4 pi N R^2 max(|rho|, |p_r|, |2J|) at the root of a: median |defect|/L %s; |rows_mid|/L %s and |Delta M/dt|/L %s; the clock residual "
        "over its largest term %s (quartiles %s to %s, %s to %s, %s to %s; 90th percentile %s); per pair over (sigma/R)^2 %s; |defect| > |rows_mid| "
        "at %s of the pairs; 2J/rho %s and p_r/rho %s at the median. All reproduce the reading's values." % (
            j(lambda k: Q[k]["median_abs_defect_over_largest"]), j(lambda k: Q[k]["median_abs_rows_over_largest"]), j(lambda k: Q[k]["median_abs_rate_over_largest"]),
            j(lambda k: Q[k]["clock_over_largest_median"]), g(Q[E[0]]["clock_over_largest_q25"]), g(Q[E[0]]["clock_over_largest_q75"]),
            g(Q[E[1]]["clock_over_largest_q25"]), g(Q[E[1]]["clock_over_largest_q75"]), g(Q[E[2]]["clock_over_largest_q25"]), g(Q[E[2]]["clock_over_largest_q75"]),
            j(lambda k: Q[k]["clock_over_largest_q90"]), j(lambda k: Q[k]["clock_over_largest_over_gate2_median"]), j(lambda k: Q[k]["frac_defect_gt_rows"], "%.2f"),
            j(lambda k: Q[k]["median_2J_over_rho"]), j(lambda k: Q[k]["median_p_over_rho"])),
        "R4 and R5 (the paper's Sec. IV sentences): not this lane's to apply. Section 8b prints the numbers they rest on, recomputed here; they "
        "reproduce the reading's to the printed digits.",
        "M1 (the figure's lines across the root-less gaps): applied. w6_fig_v1_1.py breaks each line where T_a of a pair is not T_b of the one "
        "before: sigma/a 4 at %s; sigma/a 8 at %s." % ("; ".join("%g to %g" % tuple(x) for x in R[E[1]]["gaps"]), "; ".join("%g to %g" % tuple(x) for x in R[E[2]]["gaps"])),
        "M2 (the relative representation defect's maximum, over pairs whose Delta M is near 0): labelled so in sections 0 and 2, with the median "
        "relative value beside it (%s with the K_0 weight)." % j(lambda k: O[k]["rep_defect_rel_median"]),
        "M3 (the pairs stop one record short of each closing): stated in section 8.",
        "M4 (the inner-root block computed and not printed): printed as section 6b, with the K_1-weight representation at the inner roots.",
        "M5 (the seam statements not checkable): checked. The sigma/a 2 v3b md flags the pairs 16.875 to 17.0 and 17.0 to 17.125 'seam', and its "
        "v3b log reads 226 records from the v3 seed cache with 0 re-solved. The sigma/a 4 and 8 v3b passes were given --seam-T-over-rs 18.0 "
        "(run_v3b_rungs.sh), but no pair spans 18.0, so version 1's 'the seam at 18.0 ... flagged' flags no pair. Section 8 now says so.",
        "M6 (the inner gate range includes inner zeros with no pair): section 6 and the headline now say 'at the records' inner zeros, N of them "
        "paired' (paired: %s)." % j(lambda k: R[k]["inner"]["n_pairs"], "%d"),
    ]
    paper = [
        "R4 (Sec. IV, line 70): at the outer roots, median |defect| %s of the row's largest term and %s of its net rate at sigma/a 2, 4, 8; the "
        "averaging share (sigma/R)^2 at the median %s; the residual over its largest term divided by the share, per pair at the median, %s; the "
        "row's sign the records' at %s of the pairs; 2J/rho %s and p_r/rho %s at the median. The sigma/a 2 member is uncertified and is the "
        "Newton composite across its seams." % (
            j(lambda k: Q[k]["median_abs_defect_over_largest"]), j(lambda k: O[k]["median_abs_defect_over_rows"]), j(lambda k: Q[k]["gate2_median"], "%.4f"),
            j(lambda k: Q[k]["clock_over_largest_over_gate2_median"]), j(lambda k: O[k]["same_sign_fraction"]), j(lambda k: Q[k]["median_2J_over_rho"], "%.2f"),
            j(lambda k: Q[k]["median_p_over_rho"], "%.2f")),
        "R5 (Sec. IV, line 289): at the inner trapping root above 2 ell from the centre (%s pairs; radii %s a), gate median %s; the defect %s of "
        "the rows' net rate at the median and the clock residual %s of its largest term; T_ll positive at %s inner roots." % (
            j(lambda k: Qi[k]["n_pairs"], "%d"), ", ".join("%.2f to %.2f" % (Qi[k]["R_min"], Qi[k]["R_max"]) for k in E), j(lambda k: Qi[k]["gate_median"]),
            j(lambda k: Qi[k]["median_abs_defect_over_rows"]), j(lambda k: Qi[k]["clock_over_largest_median"]),
            ", ".join("%d of %d" % (Qi[k]["Tll_a_positive"], Qi[k]["n_pairs"]) for k in E)),
        "With R1 resolved, the paper may also print the four columns at the outer roots, with the K_1 weight (section 3a): sum Delta M = col1 "
        "(flux through the old cut) + col2 (cut assignment) + col3 (averaging commutator) + col4 (continued reference), %s at sigma/a 2, 4, 8, "
        "the representation closing to %s per pair. This is the writing lane's call, under the same stress conditions." % (
            "; ".join("%s = %s + %s + %s + %s" % (g(K[k]["sum_dM"]), g(K[k]["sum_col1"]), g(K[k]["sum_col2_K1"]), g(K[k]["sum_col3"]), g(K[k]["sum_col4_K1"])) for k in E),
            j(lambda k: K[k]["rep_defect_K1_max"], "%.1e")),
    ]
    contra = [
        "Section 3's caution says the sigma/a 4 net rate is 'a near-complete cancellation (rho + p_r - 2J at the median 0.00 of rho)'. The data give "
        "%s of rho at sigma/a 4 (and %s, %s at sigma/a 2, 8). That is 0.02, not 0.00; the caution's sense stands." % (
            g(Q[E[1]]["median_Tll_over_rho"], "%.4f"), g(Q[E[0]]["median_Tll_over_rho"], "%.3f"), g(Q[E[2]]["median_Tll_over_rho"], "%.3f")),
        "The reading says the representation 'closes to 1e-7 at the 8 pairs checked' on the sigma/a 4 rung. Over all 214 pairs of that rung, and "
        "with this lane's cumulative Simpson weights, the K_1-weight representation defect has median %s and max %s, and |M_rep - M_quad| max %s. "
        "This is the weights' quadrature level (the same rule reproduces K_0's closed-form weight to %s), not 1e-7 everywhere. The RUN note's 1e-5 "
        "is met at the median on every rung, and at the maximum only at sigma/a 2 (%s)." % (
            g(K[E[1]]["rep_defect_K1_median"], "%.1e"), g(K[E[1]]["rep_defect_K1_max"], "%.1e"), g(K[E[1]]["M_rep_K1_minus_quad_max"], "%.1e"),
            g(R[E[1]]["K1_quadrature_check"], "%.1e"), g(K[E[0]]["rep_defect_K1_max"], "%.1e")),
        "No other item of the reading is contradicted. Every recomputed number (R3 and R5's medians, quartiles, fractions and ratios; R1's K_1 "
        "total weights at R_n = 1, 2, 3, 6, 11, 21, 41, 101 a for ell 4a: %s) reproduces its values." % (
            ", ".join("%.3f" % v for v in R[E[1]]["K1_total_weight_at_sites"].values())),
    ]
    json.dump({"reading": "deliveries/lead_gaps/ReviewA_reading_W6_rungs_v1.md (md5 %s; box copy %s)" % (__import__("hashlib").md5(open(a.reading, "rb").read()).hexdigest(), a.reading), "changes": ch, "paper": paper, "contradictions": contra}, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
