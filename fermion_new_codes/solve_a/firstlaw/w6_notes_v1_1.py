#!/usr/bin/env python
"""w6_notes_v1_1.py: the headline, labels, changes list and SOURCES lines of W6_FIRSTLAW_rungs_v1.1, generated from w6_readout_v1_1.json
(numbers read, none computed beyond selection and the sums the reading's R2 prints). Version 1's w6_notes_v1.py line 43 wrote
"sum Delta M = col1 + ... (the representation)", false as an equation (R2); the line is replaced here. Usage: python w6_notes_v1_1.py
--readout JSON --code-dir DIR --reading PATH --out JSON"""
import argparse
import hashlib
import json
import os


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def g(x, fmt="%.3g"):
    return "-" if x is None else fmt % x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readout", required=True)
    ap.add_argument("--code-dir", required=True)
    ap.add_argument("--reading", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    D = json.load(open(a.readout)); R = D["rungs"]
    eng = [k for k in D["order"] if R[k]["kernel"] == "engine"]
    H = []
    H.append("The 4-column identity closes on its own representation to roundoff on every rung (max |sum4 - rep| %s), and with the engine's momentum "
             "kernel applied to the momentum part of the site charge (R1 below) the representation reproduces the dense Delta M: max |Delta M - rep| "
             "%s (median %s) and max |M_rep - M_quad| %s, at the level of the cumulative quadrature of the weights (%s on K_0 against its closed form). "
             "Version 1's columns weighted the momentum part u(R_n) J_n with the scalar kernel K_0's cumulative weight (busi's firstlaw_records_v4b.py, "
             "cumulative_weight in pair_identity) where the engine smooths J with the l = 1 kernel K_1 (engine_fields_v4.kernels): that, not the records, "
             "is version 1's representation defect (max |Delta M - rep| %s, max |M_rep - M_quad| %s), the same on both kernel routes because the weights "
             "never changed; the engine route removes only the kernel check's J mismatch (%s with the analytic kernel, %s with the engine's)." % (
                 g(max(R[k]["outer"]["closure_max"] for k in eng)),
                 ", ".join("sigma/a %g %s" % (R[k]["sigma"], g(R[k]["K1_outer"]["rep_defect_K1_max"])) for k in eng),
                 ", ".join(g(R[k]["K1_outer"]["rep_defect_K1_median"]) for k in eng),
                 ", ".join(g(R[k]["K1_outer"]["M_rep_K1_minus_quad_max"]) for k in eng),
                 "at most " + g(max(R[k]["K1_quadrature_check"] for k in eng)),
                 ", ".join(g(R[k]["outer"]["rep_defect_max"]) for k in eng), ", ".join(g(R[k]["outer"]["M_rep_minus_quad_max"]) for k in eng),
                 ", ".join(g(R[k]["outer"].get("kernel_rel_J_max")) for k in D["order"] if R[k]["kernel"] == "analytic"),
                 ", ".join(g(R[k]["outer"].get("kernel_rel_J_max")) for k in eng)))
    for k in eng:
        s = R[k]["outer"]; K = R[k]["K1_outer"]; q = R[k]["R3_outer"]
        H.append("sigma/a %g: %d outer-root pairs (T/r_s %s to %s); sum Delta M %s = col1 %s + col2 %s + col3 %s + col4 %s with the K_1 weight (the "
                 "columns' sum %s; with version 1's K_0 weight the columns summed to %s, the difference %s the representation defect of R1); records' rate "
                 "summed %s against the rows' %s (defect %s); the defect at the median %s of the rows' net rate and %s of the row's largest term, the net "
                 "rate being a cancellation (2J/rho %s, p_r/rho %s at the median), %s times (sigma/R)^2 per pair; same sign %s; full-clock residual |.| "
                 "median %s (%s of its largest term), max %s; gate %s to %s at the outer root (inside 0.1 at %d of %d); at the records' inner zeros %s "
                 "to %s, %d of them paired." % (
                     R[k]["sigma"], s["n_pairs"], g(s["T_first"], "%.3f"), g(s["T_last"], "%.3f"), g(K["sum_dM"]), g(K["sum_col1"]), g(K["sum_col2_K1"]),
                     g(K["sum_col3"]), g(K["sum_col4_K1"]), g(K["sum_rep_K1"]), g(s["sum_rep"]), g(s["sum_dM"] - s["sum_rep"]),
                     g(s["sum_rate"]), g(s["sum_rows_mid"]), g(s["sum_defect_mid"]), g(s["median_abs_defect_over_rows"]), g(q["median_abs_defect_over_largest"], "%.3f"),
                     g(q["median_2J_over_rho"]), g(q["median_p_over_rho"]), g(q["clock_over_largest_over_gate2_median"]), g(s["same_sign_fraction"]),
                     g(s["clock_mid_abs_median"]), g(q["clock_over_largest_median"], "%.3f"), g(s["clock_mid_abs_max"]), g(s["gate_min"], "%.4f"), g(s["gate_max"], "%.4f"),
                     s["gate_inside"], s["n_pairs"], g(s.get("gate_inner_min"), "%.3f"), g(s.get("gate_inner_max"), "%.3f"), R[k]["inner"]["n_pairs"]))
    notes = [
        "The sigma/a 2 point is the composite history (Newton across its seams; its v3b md flags the pairs 16.875 to 17.0 and 17.0 to 17.125 at the "
        "seam 17.0) whose v3b pass read all 226 dense records from the v3 seed cache (its log: 0 re-solved). Under MASS_ladder_v1.1's rule it has "
        "no certified record (8 of its 321 records have a confirming residual above stop_fields 1e-6, 1.06e-6 to 2.99e-6, the first at T = 0), so "
        "every one of its pairs lies after its certified prefix. The sigma/a 4 and 8 rungs (rung 4 and rung 8 on v36d from T = 0, every record "
        "converged and counting) had no seed cache; their v3b pass was given the seam 18.0 (run_v3b_rungs.sh, --seam-T-over-rs 18.0, the "
        "continuation from the T0018.00 slice), and no pair spans it (no rooted records there), so no pair is flagged.",
        "The engine's momentum kernel is the l = 1 kernel K_1, not the shell Gaussian K_0, at every spacing; the analytic kernel check of J is "
        "therefore of order 1 on all three rungs and the engine kernel's route is the primary one. With version 1's K_0 weight on the momentum, the "
        "column sums change only in column 3 between the routes (column 3 alone uses the J kernel).",
        "With the K_1 weight (w6_k1weight_v1_1.py; busi's v4b unchanged and imported; validated by reproducing v4b's own columns to at most 1.4e-12 "
        "with the K_0 weight) the representation closes at the cumulative quadrature's level, and the four columns read as the mass rate's "
        "decomposition at the outer roots: column 1 and column 3 are unchanged, column 2 (the cut assignment) takes the momentum part that version "
        "1 misplaced, column 4 is unchanged because no momentum reference is stored (J_R2 absent: the momentum part of column 4 is not composed). "
        "This is a v4c reading of the identity by this lane's script; busi's producer itself is unchanged.",
        "The phases use the sigma/a 2 point's boundaries (T_b 14.75 and 17.5) on every rung for comparability; the rungs' own episodes differ "
        "(sigma/a 4: 12.125 to 17.375, then from 18.375; sigma/a 8: 12.75 to 16.375, 26.625 to 29.75, from 38.875), and the pairs stop one record "
        "short of each closing (last T_b 17.25 against the closing 17.375 at sigma/a 4; 16.25 and 29.625 against 16.375 and 29.75 at sigma/a 8), "
        "the closing records carrying no v3b pair: 214 and 61 pairs stand against 215 and 63 consecutive rooted records.",
        "The size of the residual: the rows' net rate at the root, 4 pi N R^2 (rho + p_r - 2J), is a cancellation of terms each of order rho (2J at "
        "1.38 to 1.49 of rho, p_r at 0.21 to 0.49 of rho, at the median; T_ll at -0.17, 0.02 and 0.11 of rho), so the residual is printed against "
        "both the net rate and the row's largest term L = 4 pi N R^2 max(|rho|, |p_r|, |2J|) at the root of a, and the clock residual against the "
        "largest of its three terms.",
        "The rows' rate carries the handoff's stress under FIRSTLAW_records v2.2's four open conditions (the stress module's version, the "
        "normalization by the ratio test alone, the companion row by the T key, the between-node remainder); a stress factor of order a few would "
        "move the net-rate ratios and the sign counts (T_ll negative at 162 of 224, 85 of 214, 0 of 61 outer roots) far more than the "
        "largest-term ratios.",
        "The stored errors: the records' kept residuals (below 1e-6 on every record of the 2 rungs; the composite's records below 9.9e-07 with the "
        "confirming residual above 1e-6 at 8 records), the rows' interpolation of the stress from the handoff nodes (v3b's stride-2 error printed "
        "in each v3b md), the record step (the records' rate is a finite difference over tau = 8 a against the rows' instantaneous rate; the a- and "
        "b-side rates are in the v4b json). One reference convention, one step, the three averaging scales at one preparation.",
        "Not composed: a face-flux array (none in the handoff), the momentum part of column 4 (no J_R2 stored), the sigma/a 2 plain rung (not on "
        "the box), the rows at the core's half-mass radius (not among W6's places).",
    ]
    code = {k: md5(os.path.join(a.code_dir, k)) for k in ("w6_readout_v1.py", "w6_readout_v1_1.py", "w6_k1weight_v1_1.py", "w6_compose_v1_1.py",
                                                          "w6_fig_v1_1.py", "w6_notes_v1_1.py")}
    src = ["busi's firstlaw_records_v4b.py and firstlaw_selftest_v4bw6.py at <internal>/w6w7 (md5 13666b78f03b8d554d1b5345444978f2 and "
           "b97bdcecc581cb5e708ecd7581ef9506), unchanged; firstlaw_records_v3b.py at <internal>/code; the run scripts "
           "<internal>/run_v3b_rungs.sh, run_v4b_engine_kernel.sh, run_v4b_rest.sh and their logs beside the output directories",
           "the records: <internal>/collapse_c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0_to40 (md5 of the json "
           "d7d69adcea7a43f8375ab4c7c5a80efc), <internal>/collapse_c32_rs64_A_R2_tau8_ell8_prep4_avg_plain_v36d_from0_to40 "
           "(d1ca037a77c198aa83b33055495eceff), <internal>/collapse_c32_rs64_A_R2_tau8_ell2_prep4_avg_nk40r2b_v36cc_v36d "
           "(0a005ac5f66bef02a0104e4278867765), each with its _fields.npz and _handoff.npz (S_r_ell, S_t_ell)",
           "the reading applied: deliveries/lead_gaps/ReviewA_reading_W6_rungs_v1.md (md5 %s; box copy %s)" % (md5(a.reading), a.reading),
           "the K_1-weight reading: w6_k1weight_v1_1.py (md5 %s), output %s (md5 %s)" % (code["w6_k1weight_v1_1.py"], D["k1_json"], D["k1_md5"]),
           "producer w6_readout_v1_1.py (md5 %s; it runs version 1's w6_readout_v1.py, md5 %s, unchanged, and extends it), composer w6_compose_v1_1.py (%s), "
           "figure w6_fig_v1_1.py (%s), these lines w6_notes_v1_1.py (%s), all in <internal>/firstlaw (box copies "
           "<internal>/qi2_w6_code); outputs <internal>/firstlaw and the box <internal>/w6_rungs_v1_1 "
           "with MANIFEST_w6_v1_1.md5" % (code["w6_readout_v1_1.py"], code["w6_readout_v1.py"], code["w6_compose_v1_1.py"], code["w6_fig_v1_1.py"], code["w6_notes_v1_1.py"])]
    json.dump({"headline": H, "notes": notes, "sources": src}, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
