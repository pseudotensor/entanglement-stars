#!/usr/bin/env python
"""w6_notes_v1.py: the headline, labels and SOURCES lines of W6_FIRSTLAW_rungs_v1, generated from w6_readout_v1.json (numbers read, none
computed beyond selection). Usage: python w6_notes_v1.py --readout JSON --code-dir DIR --out JSON"""
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
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    D = json.load(open(a.readout)); R = D["rungs"]
    eng = [k for k in D["order"] if R[k]["kernel"] == "engine"]
    H = []
    H.append("The 4-column identity closes on its own representation to roundoff on every rung (max |sum4 - rep| %s), but the representation does not "
             "reproduce the dense Delta M: max |Delta M - rep| %s on the engine-kernel rows (relative to |Delta M| up to %s), with max |M_rep - M_quad| %s "
             "while |M_quad - M_ode| stays at %s. The RUN note expects the representation defect below 1e-5; it is a failure flag of the identity's "
             "representation on the records at every averaging scale, not of the kernel: the engine kernel removes the analytic kernel's J mismatch "
             "(kernel check J %s with the analytic kernel, %s with the engine's) and leaves the defect unchanged." % (
                 g(max(R[k]["outer"]["closure_max"] for k in eng)), ", ".join("sigma/a %g %s" % (R[k]["sigma"], g(R[k]["outer"]["rep_defect_max"])) for k in eng),
                 g(max(R[k]["outer"]["rep_defect_rel_max"] for k in eng)), ", ".join("%s" % g(R[k]["outer"]["M_rep_minus_quad_max"]) for k in eng),
                 g(max(R[k]["outer"]["M_quad_minus_ode_max"] for k in eng)),
                 ", ".join(g(R[k]["outer"].get("kernel_rel_J_max")) for k in D["order"] if R[k]["kernel"] == "analytic"),
                 ", ".join(g(R[k]["outer"].get("kernel_rel_J_max")) for k in eng)))
    for k in eng:
        s = R[k]["outer"]
        H.append("sigma/a %g: %d outer-root pairs (T/r_s %s to %s); sum Delta M %s = col1 %s + col2 %s + col3 %s + col4 %s (the representation %s); records' "
                 "rate summed %s against the rows' %s (defect %s, median |defect|/|rows| %s, same sign %s); full-clock residual |.| median %s, max %s; gate "
                 "%s to %s at the outer root (inside 0.1 at %d of %d), inner %s to %s." % (
                     R[k]["sigma"], s["n_pairs"], g(s["T_first"], "%.3f"), g(s["T_last"], "%.3f"), g(s["sum_dM"]), g(s["sum_col1_flux_old_cut"]),
                     g(s["sum_col2_cut_assignment"]), g(s["sum_col3_averaging_commutator"]), g(s["sum_col4_continued_reference"]), g(s["sum_rep"]),
                     g(s["sum_rate"]), g(s["sum_rows_mid"]), g(s["sum_defect_mid"]), g(s["median_abs_defect_over_rows"]), g(s["same_sign_fraction"]),
                     g(s["clock_mid_abs_median"]), g(s["clock_mid_abs_max"]), g(s["gate_min"], "%.4f"), g(s["gate_max"], "%.4f"), s["gate_inside"], s["n_pairs"],
                     g(s.get("gate_inner_min"), "%.3f"), g(s.get("gate_inner_max"), "%.3f")))
    notes = [
        "The sigma/a 2 point is the composite history (Newton across its seams, seam 17.0 in its v3b pass) whose v3b pass used the v3 seed cache; "
        "under MASS_ladder_v1.1's rule it has no certified record (its T = 0 record's confirming residual 2.99e-06 exceeds stop_fields 1e-6), so "
        "every one of its pairs lies after its certified prefix. The sigma/a 4 and 8 rungs (rung 4 and rung 8 on v36d from T = 0, every record "
        "converged and counting) had no seed cache: their dense pass re-solved every rooted record and its neighbours fresh, with the seam at "
        "18.0 (the continuation from the T0018.00 slice) flagged.",
        "The analytic kernel check of J is of order 1 on all three rungs (the analytic shell Gaussian is not the engine's J kernel at this spacing); "
        "the RUN note's fallback, the engine kernel, was run on all three and its rows are the primary ones; both kernels' rows print. The column "
        "sums change only in column 3 (the averaging commutator) between the kernels.",
        "The representation defect (Delta M - rep) at O(1 to 20) and |M_rep - M_quad| at O(50) on every rung are above the RUN note's 1e-5 bound; "
        "the note's instruction is to post the line, not to read the columns as the mass rate's decomposition. The columns and their sums print as "
        "computed under that flag; the rows' rate, the defect against it, the full-clock residual and the gate do not depend on the representation "
        "and print as the records' readout.",
        "The phases use the sigma/a 2 point's boundaries (T_b 14.75 and 17.5) on every rung for comparability; the rungs' own episodes differ "
        "(sigma/a 4: 12.125 to 17.375, then from 18.375; sigma/a 8: 12.75 to 16.375, 26.625 to 29.75, from 38.875).",
        "The stored errors: the records' kept residuals (below 1e-6 on every record of the 2 rungs; the composite's records below 9.9e-07 with the "
        "confirming residual above 1e-6 at 8 records), the rows' interpolation of the stress from the handoff nodes (v3b's stride-2 error printed "
        "in each v3b md), the record step (the records' rate is a finite difference over tau = 8 a against the rows' instantaneous rate; the a- and "
        "b-side rates are in the v4b json). One reference convention, one step, the three averaging scales at one preparation.",
        "Not composed: a face-flux array (none in the handoff), the momentum part of column 4 (no J_R2 stored), the sigma/a 2 plain rung (not on "
        "the box).",
    ]
    code = {k: md5(os.path.join(a.code_dir, k)) for k in ("w6_readout_v1.py", "w6_compose_v1.py", "w6_fig_v1.py", "w6_notes_v1.py")}
    src = ["busi's firstlaw_records_v4b.py and firstlaw_selftest_v4bw6.py at <internal>/w6w7 (md5 13666b78f03b8d554d1b5345444978f2 and "
           "b97bdcecc581cb5e708ecd7581ef9506, equal to the busi worktree's), self-test 6 checks, 0 failed, run before the rungs; firstlaw_records_v3b.py at "
           "<internal>/code; the run scripts <internal>/run_v3b_rungs.sh, run_v4b_engine_kernel.sh, "
           "run_v4b_rest.sh and their logs beside the output directories",
           "the records: <internal>/collapse_c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0_to40 (md5 of the json "
           "d7d69adcea7a43f8375ab4c7c5a80efc), <internal>/collapse_c32_rs64_A_R2_tau8_ell8_prep4_avg_plain_v36d_from0_to40 "
           "(d1ca037a77c198aa83b33055495eceff), <internal>/collapse_c32_rs64_A_R2_tau8_ell2_prep4_avg_nk40r2b_v36cc_v36d "
           "(0a005ac5f66bef02a0104e4278867765), each with its _fields.npz and _handoff.npz (S_r_ell, S_t_ell)",
           "producer w6_readout_v1.py (md5 %s), composer w6_compose_v1.py (%s), figure w6_fig_v1.py (%s), these lines w6_notes_v1.py (%s), all in "
           "<internal>/firstlaw; outputs <internal>/firstlaw (w6_readout_v1.json, "
           ".npz, W6_FIRSTLAW_rungs_v1_fig.png) and the box <internal>/w6_rungs_v1 with MANIFEST_w6_readout_v1.md5" % (
               code["w6_readout_v1.py"], code["w6_compose_v1.py"], code["w6_fig_v1.py"], code["w6_notes_v1.py"])]
    json.dump({"headline": H, "notes": notes, "sources": src}, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
