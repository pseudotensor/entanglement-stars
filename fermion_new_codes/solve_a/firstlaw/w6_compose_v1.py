#!/usr/bin/env python
"""w6_compose_v1.py (composition builder, the solve lane): writes W6_FIRSTLAW_rungs_v1.md from w6_readout_v1.json and a notes json (no run
file is opened; every number is the producer's). Usage: python w6_compose_v1.py --readout JSON --notes JSON --out FILE"""
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
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    D = json.load(open(a.readout)); N = json.load(open(a.notes))
    R = D["rungs"]; order = D["order"]
    o = ["# W6_FIRSTLAW_rungs_v1: FIRSTLAW_records v2.3 (the exact fixed-radius mass-rate identity in 4 columns, the rows' rate, the full-clock "
         "residual and the gate at every dense outer root) on the averaging ladder at (0.2, 64), sigma/a 2, 4 and 8 at fixed preparation", ""]
    o.append("Composed by w6_readout_v1.py and w6_compose_v1.py from the outputs of the busi lane's firstlaw_records_v4b.py (unchanged), run on the "
             "outputs of firstlaw_records_v3b.py (unchanged) on each rung's stored records; box CPU, no engine rerun. The builder composes; no "
             "sentence here reads the physics. The definitions are RUN_firstlaw_records_v4b.md's: column 1 the flux through the old cut (the "
             "sharp-cut balance of the state's charge), column 2 the cut-assignment correction, column 3 the averaging commutator, column 4 the "
             "continued-reference term; rep their sum (the kernel representation of Delta M); the representation defect Delta M - rep; the rows' "
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
        rows.append([k, e(s.get("closure_max")), "%s (%s)" % (e(s.get("rep_defect_max")), e(s.get("rep_defect_rel_max"))), e(s.get("rep_defect_median")),
                     e(s.get("M_rep_minus_quad_max")), e(s.get("M_quad_minus_ode_max")), "%s; %s (median %s)" % (e(s.get("kernel_rel_rho_max")), e(s.get("kernel_rel_J_max")), e(s.get("kernel_rel_J_median"))),
                     e(s.get("exact_relation_max"))])
    o.append(table(["rung / kernel", "max |sum4 - rep| (closure)", "max |Delta M - rep| (relative to |Delta M|)", "median |Delta M - rep|",
                    "max |M_rep - M_quad|", "max |M_quad - M_ode|", "kernel check rho; J (max relative)", "exact relation at fixed R (max)"], rows))
    o.append("")
    o.append("## 3 The 4 columns (sums over the outer-root pairs)")
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
                     "%s; %s" % (f(s.get("sum_rows_a_nostress")), f(s.get("sum_rows_a_stress")))])
    o.append(table(["rung / kernel", "sum Delta M/dt", "sum rows_mid", "sum defect", "median |defect|/|rows_mid|", "same-sign fraction",
                    "negative rows_mid; negative rate; negative T_ll(a)", "stress share sums: (rho - 2J) part; p_r part"], rows))
    o.append("")
    o.append("## 5 The full-clock residual at the roots")
    o.append("")
    rows = []
    for k in order:
        s = R[k]["outer"]
        rows.append([k, e(s.get("clock_mid_median")), e(s.get("clock_mid_abs_median")), e(s.get("clock_mid_abs_max")), "%s (|.| max %s)" % (e(s.get("clock_a_median")), e(s.get("clock_a_abs_max"))),
                     "%s; %s; %s" % (e(s.get("clock_term_st_over_N_median")), e(s.get("clock_term_clock_median")), e(s.get("clock_term_matter_median")))])
    o.append(table(["rung / kernel", "median (midpoint)", "|.| median", "|.| max", "endpoint median", "median terms: s_t/N; (1 + s^2)(ln N)'; -4 pi G r (rho + p_r)"], rows))
    o.append("")
    o.append("## 6 The gate sigma/R per root")
    o.append("")
    rows = []
    for k in order:
        s = R[k]["outer"]; si = R[k]["outer"]
        rows.append([k, "%s to %s (median %s)" % (f(s.get("gate_min"), "%.4f"), f(s.get("gate_max"), "%.4f"), f(s.get("gate_median"), "%.4f")),
                     "%s of %s" % (s.get("gate_inside"), s["n_pairs"]), "%s to %s a" % (f(s.get("R_min"), "%.2f"), f(s.get("R_max"), "%.2f")),
                     "%s to %s; inside %s of %s" % (f(si.get("gate_inner_min"), "%.3f"), f(si.get("gate_inner_max"), "%.3f"), si.get("gate_inner_inside"), si.get("gate_inner_n"))])
    o.append(table(["rung / kernel", "outer root sigma/R", "inside 0.1", "outer root R", "inner root sigma/R"], rows))
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
    o.append(table(["rung / kernel", "T_b window", "pairs", "sum Delta M", "col1", "col2", "col3", "col4", "max |Delta M - rep|", "sum Delta M/dt", "sum rows_mid",
                    "clock |.| median", "gate"], rows))
    o.append("")
    o.append("## 8 Labels and what stays open")
    o.append("")
    for line in N.get("notes", []):
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
