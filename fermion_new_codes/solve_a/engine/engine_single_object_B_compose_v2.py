#!/usr/bin/env python
"""engine_single_object_B_compose_v2.py (builder analyst; version 2 = version 1 (called once, on the primary runs, into the scratchpad) with two
labels added: the launch-spectrum weights e < 0 and abs(e) < m overlap on -m < e < 0; B.5's readouts are per actual orbital (strand R SO4),
their sums are not the detector density, and negative-energy (sea) orbitals' weight beyond R_t is not particle escape; inbox #41's revised S4
part B, its compose): reads the outputs of
engine_single_object_B_v1.py (--mode oracle, frozen, history, orbitals) and engine_single_object_B5_prep_v1.py (never edits them) and writes
one md and one summary json: per run, launch record and channel, the launch, the readouts of B.4 per column, D's checks (the dense oracle,
the Killing-spectrum conservation on the frozen records, the unit norm per record and its accumulation, the frozen-record drain as the
positive control, the piecewise-constant against cubic field control), B.5's orbitals against the clock, and the sigma/a 8 member as B.6's
scale control. The column order of B v1's launch set: 2 depths (the annulus middle; 2 sigma inside the outer root) x 3 (ak_0, sigma) =
(2 pi/64, 16), (pi/4, 8), (pi/2, 4) x the 2 signs of k_0. Conventions printed with the tables. No mutual information (D15); the massive
channel only (D14). Every number in the md is printed by this script from the files named on its command line.
Args: --oracle --frozen --history (comma-separated lists allowed: the primary first) --run-names --orbitals --orbitals-prep --md --out-json."""
import argparse
import json
import math

LABEL = ("the single object's own cross-surface 2-point block and transfer on the solved history of the massive collapse, per radial channel; "
         "the interior beyond the hydrodynamic gate H sigma > 0.1 from T/r_s 11.75, the record's own limit; the massless channel, where present, "
         "a test channel under owner decision D14")
REGIONS = ("beyond_outer_root", "beyond_2GE", "beyond_16rs", "collar_16_24", "far_32_48", "beyond_Rcut")


def fmt(x, f="%.3e"):
    return (f % x) if isinstance(x, (int, float)) and not isinstance(x, bool) and x is not None and x == x else "-"


def col_name(c):
    l_ = c["launch"]
    return "%s; R_c %.2f; ak0 %+.4f (%s); sigma %.0f" % ("annulus middle" if l_["depth"].startswith("annulus") else "2 sigma inside R_t", l_["R_c"], l_["k0"],
                                                          l_["ak0_name"].split(" ")[0], l_["sigma"])


def last_pre_reflection(L, c_):
    pre = [s_ for s_ in L["series"] if not s_["reflection_reached"][c_]]
    return pre[-1] if pre else None


def launch_rows(L, mode, m):
    """per column: the launch, B.4's readouts, the positive control's accounting (frozen)."""
    rows = []
    for c_, c in enumerate(L["columns"]):
        sp = c["spectrum_launch"]
        lp = last_pre_reflection(L, c_)
        refl_end = bool(L["series"][-1]["reflection_reached"][c_])
        out_cut = lp["w"]["beyond_Rcut"][c_] if lp else None
        acc = (1.0 - out_cut - c["bound_overlap_launch"]) if (lp and mode == "frozen") else None
        ak0 = abs(c["launch"]["k0"])
        rows.append({"column": c_, "name": col_name(c), "omega_bar": c["omega_bar"], "spread": sp["spread"], "soft": c["soft"], "w_negative": sp["w_negative"],
                     "w_in_gap": sp["w_in_gap"], "w_above_m": sp["w_above_m"], "beyond16_final_before_reflection": c["beyond16_final_before_reflection"],
                     "arrival_after_launch_rs": c["arrival_after_launch_rs"], "pred_2rs_log": c["pred_2rs_log"],
                     "pred_kin_log": c["pred_kin_log"] if ak0 < 1.0 else None, "pred_kin_log_note": None if ak0 < 1.0 else "not applicable: ak0 >= 1",
                     "reflection_clock_after_launch_rs": c["reflection_clock_after_launch_rs"], "end_includes_reflection": refl_end,
                     "bound_overlap_launch": c["bound_overlap_launch"], "bound_overlap_end": c["bound_overlap_end"], "interior_end": c["interior_end"],
                     "max_radius_rs": c["max_radius_rs"], "end": c["end"], "beyond_Rcut_last_pre_reflection": out_cut,
                     "last_pre_reflection_after_launch_rs": lp["after_launch_rs"] if lp else None, "positive_control_unaccounted": acc,
                     "spectrum_end": c.get("spectrum_end")})
    return rows


def norm_checks(L):
    ser = L["series"]
    mx = max(s_["norm_minus_1_max"] for s_ in ser)
    first = next((s_["after_launch_rs"] for s_ in ser if s_["norm_minus_1_max"] > 1e-12), None)
    per = mx / len(ser) if ser else None
    return {"max": mx, "first_above_1e-12_after_launch_rs": first, "steps": len(ser), "max_over_steps": per}


def run_md(name, F, mode, m, L_out):
    L = []
    for La in F["launches"]:
        rows = launch_rows(La, mode, m)
        nc = norm_checks(La)
        rec = {"run": name, "mode": mode, "T_launch_over_rs": La["T_launch_over_rs"], "kappa": La["kappa"], "E": La["E"], "two_gE": La["two_gE"], "T_H": La["T_H"],
               "R_outer": La["R_outer"], "R_inner": La["R_inner"], "kappa_in": La["kappa_in"], "bound_states": La["bound_states"], "cheb_K_max": La["cheb_K_max"],
               "cheb_tail_acc": La["cheb_tail_acc"], "spectrum_conservation_max": La["spectrum_conservation_max"], "norm": nc, "cubic_control": La.get("cubic_control"),
               "T_end_over_rs": La.get("T_end_over_rs"), "columns": rows}
        L_out.append(rec)
        L += ["#### %s (%s): launch T/r_s %.3f, kappa %d" % (name, mode, La["T_launch_over_rs"], La["kappa"]), "",
              "- E %.6f, 2GE %.4f, T_H %.4e; the dense roots %s; outer %s, inner %s, kappa_in %s; bound states of the frozen h (weight >= 0.99 inside R_cut): %d; "
              "Chebyshev K max %d, accumulated tail %.1e; the Killing spectrum's largest change over the run %.1e (%s); the norms' largest defect %.2e over %d steps "
              "(first above 1e-12 at %s r_s after launch; the defect over the steps %.1e per step)%s." % (
                  La["E"], La["two_gE"], La["T_H"], ", ".join("%.4f (%+.5f)" % (q["R"], q["slope"]) for q in La["roots_dense"]), fmt(La["R_outer"], "%.4f"),
                  fmt(La["R_inner"], "%.4f"), fmt(La["kappa_in"], "%.5f"), La["bound_states"], La["cheb_K_max"], La["cheb_tail_acc"], La["spectrum_conservation_max"],
                  "the propagator check, target 1e-10" if mode == "frozen" else "the shift on the solved history",
                  nc["max"], nc["steps"], fmt(nc["first_above_1e-12_after_launch_rs"], "%.1f"), nc["max_over_steps"],
                  ("; to T/r_s %.3f" % La["T_end_over_rs"]) if La.get("T_end_over_rs") is not None else "")]
        cc = La.get("cubic_control")
        if cc:
            L.append("- The piecewise-constant against cubic field control on T/r_s %.3f to %.3f: the columns' largest element difference %.3e; the change in each readout: %s." % (
                cc["window_rs"][0], cc["window_rs"][1], cc["columns_diff_max"], "; ".join("%s %.3e" % (k_, v_) for k_, v_ in cc["region_diff"].items())))
        L += ["",
              "| column | omega_bar (spread) | soft (omega_bar < m) | launch weights: e < 0; abs(e) < m; e > m | beyond 16 r_s, last before the reflection | arrival (r_s after launch) | 2 r_s ln(1/(a omega_bar)) | kappa_in^-1 ln(1/(ak_0)) | reflection clock (r_s after launch; 208 beside) | bound overlap: launch; end | interior weight at the end | max radius (r_s) | at the end: %s%s |" % (
                  "; ".join(REGIONS), "; the end includes the chain-end reflection" if mode == "frozen" else ""),
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            L.append("| %d: %s | %+.4f (%.4f) | %s | %.2e; %.2e; %.2e | %s | %s | %s | %s | %.1f | %.4e; %.4e | %.4e | %.1f | %s%s |" % (
                r["column"], r["name"], r["omega_bar"], r["spread"], r["soft"], r["w_negative"], r["w_in_gap"], r["w_above_m"],
                fmt(r["beyond16_final_before_reflection"], "%.4e"), fmt(r["arrival_after_launch_rs"], "%.1f"), fmt(r["pred_2rs_log"], "%.2f"),
                fmt(r["pred_kin_log"], "%.2f") if r["pred_kin_log"] is not None else r["pred_kin_log_note"], r["reflection_clock_after_launch_rs"],
                r["bound_overlap_launch"], r["bound_overlap_end"], r["interior_end"], r["max_radius_rs"], "; ".join("%.3e" % r["end"].get(k_, float("nan")) for k_ in REGIONS),
                (" (yes)" if r["end_includes_reflection"] else " (no)") if mode == "frozen" else ""))
        L.append("")
        if mode == "frozen":
            L += ["The positive control (D: the frozen-record drain): per column, 1 - (the weight beyond R_cut at the last clock before the reflection) - (the overlap with the bound states); "
                  "a wavelength-r_s column (ak_0 = 2 pi/64) that neither leaves nor overlaps the bound states would show here.", "",
                  "| column | the last clock before the reflection (r_s after launch) | beyond R_cut there | bound overlap | unaccounted |", "|---|---|---|---|---|"]
            for r in rows:
                L.append("| %d: %s | %s | %s | %.4e | %s |" % (r["column"], r["name"], fmt(r["last_pre_reflection_after_launch_rs"], "%.1f"),
                                                             fmt(r["beyond_Rcut_last_pre_reflection"], "%.4e"), r["bound_overlap_launch"], fmt(r["positive_control_unaccounted"], "%+.3e")))
            L.append("")
        if mode == "history":
            L += ["The Killing spectrum of the columns at the end, on the launch record's frozen h and on the end record's: mean energy; weights e < 0; abs(e) < m; e > m.", "",
                  "| column | at launch | at the end, the launch record's h | at the end, the end record's h |", "|---|---|---|---|"]
            for c_, r in enumerate(rows):
                se = r["spectrum_end"] or {}; sl = La["columns"][c_]["spectrum_launch"]; sE = (La.get("spectrum_end_on_end_record") or [None] * 12)[c_] or {}
                f3 = lambda s: "%+.4f; %.2e; %.2e; %.2e" % (s["mean"], s["w_negative"], s["w_in_gap"], s["w_above_m"]) if s else "-"
                L.append("| %d: %s | %s | %s | %s |" % (r["column"], r["name"], f3(sl), f3(se), f3(sE)))
            L.append("")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", required=True); ap.add_argument("--frozen", required=True); ap.add_argument("--history", required=True)
    ap.add_argument("--run-names", default="sigma/a 4 (production),sigma/a 8 (B.6 scale control)")
    ap.add_argument("--orbitals", default=None); ap.add_argument("--orbitals-prep", default=None)
    ap.add_argument("--md", required=True); ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    O = json.load(open(a.oracle))
    names = a.run_names.split(",")
    fro = [json.load(open(p)) for p in a.frozen.split(",")]; his = [json.load(open(p)) for p in a.history.split(",")]
    L = ["## S4 (revised) part B: transfer by wavelength and depth, per launch record and radial channel", "",
         "Composed by engine_single_object_B_compose_v2.py from %s, %s, %s%s%s (engine_single_object_B_v1.py; inbox #41)." % (
             a.oracle, a.frozen, a.history, (", " + a.orbitals) if a.orbitals else "", (", " + a.orbitals_prep) if a.orbitals_prep else ""), "",
         "- %s." % LABEL, "- No mutual information (D15); the massive channel (ma %s) only; no massless channel (D14)." % fro[0].get("m"),
         "- Machinery: page_history_v6b (the record's applied fields on the engine's sites, the vacuum Painleve-Gullstrand exterior at the record's E "
         "beyond them, %s sites = 128 r_s), its Chebyshev action exp(-i tau h) X to the Bessel tail %s, page_ops_v6's operators; the frozen runs hold the "
         "launch record's h for 300 r_s in steps of 1 r_s; the history runs step record by record (piecewise-constant applied fields, the brief's convention)." % (
             fro[0].get("nsites"), fro[0].get("cheb_tol")),
         "- The launch columns: g(R) e^{i k_0 R} u_+(k_0), u_+ the positive-energy spinor of the local symbol at the launch site, both signs of k_0; the "
         "depths: the middle of the trapped annulus and 2 sigma inside the outer root (the launch record's dense roots). omega_bar: the column's mean "
         "Killing energy on the launch record's frozen h (soft: omega_bar < m).",
         "- The arrival: the first clock at which the weight beyond 16 r_s reaches half its value at the last clock before the chain-end reflection; the "
         "reflection clock: launch + (128 r_s - R_launch) + (128 - 24) r_s at unit speed, the Page builder's 208 r_s beside it; a frozen run's end (300 r_s) "
         "includes the reflection.",
         "- Bound-state content: the overlap with the eigenvectors of the frozen h with weight >= 0.99 inside R_cut 24 r_s; weight that never leaves is "
         "bound-state content, never trapping. For soft columns exit to infinity is barred by the mass threshold for the part below m; the launch "
         "spectrum prints the part above m.",
         "- The launch-spectrum weights e < 0 and abs(e) < m overlap on -m < e < 0; the partition is e < -m, abs(e) < m, e > m.",
         "- kappa_in^-1 ln(1/(ak_0)) is the brief's small-momentum comparison; strand R (SO11) gives the inner-root blueshift time "
         "kappa_in^-1 log(tan q_*/tan q_0) on its patch; for ak_0 >= 1 the small-momentum form is not applicable.", "",
         "### D: the dense oracle", "",
         "- %s; %d records from T/r_s %.3f, roots %s; the largest differences, Chebyshev against the dense propagator: %s; pass at 1e-10: %s." % (
             O["fixture"], O["records"], O["launch_T_over_rs"], ", ".join("%.3f" % r_ for r_ in O["roots"]),
             "; ".join("%s %.1e" % (k_, v_) for k_, v_ in O["max_abs_diff"].items()), O["pass_1e-10"]), ""]
    recs = []
    for nm, F, Hh in zip(names, fro, his):
        L += ["### %s: frozen records (B.1 (i)); %s; %s to %s" % (nm, F["record"], F.get("utc_start"), F.get("utc_end")), ""]
        L += run_md(nm, F, "frozen", F["m"], recs)
        L += ["### %s: the solved history (B.1 (ii)); %s; %s to %s" % (nm, Hh["record"], Hh.get("utc_start"), Hh.get("utc_end")), ""]
        L += run_md(nm, Hh, "history", Hh["m"], recs)
    orb = None
    if a.orbitals:
        orb = json.load(open(a.orbitals))
        prep = json.load(open(a.orbitals_prep)) if a.orbitals_prep else None
        L += ["### B.5: the state's own interior orbitals on the solved history (%s; %s to %s)" % (orb["orbitals_npz"], orb.get("utc_start"), orb.get("utc_end")), "",
              "The readouts are per actual orbital, ||Pi_D U phi||^2 (strand R SO4); their sums over the orbitals are not the detector density (the initial cross blocks are omitted), "
              "and the weight beyond R_t(t) of negative-energy (sea) orbitals is where that part of the occupied set's support went, not particle escape. "
              "The launch spectrum shows each set's negative-energy, gap and positive-energy weight (e < 0 and abs(e) < m overlap on -m < e < 0).", ""]
        if prep:
            L += ["The orbitals (engine_single_object_B5_prep_v1.py): the right singular vectors of Pi_X Phi_o with sigma^2 >= %s, X the sites inside the dense outer root at the target record; the columns Phi_o v_j (the actual orbitals with their tails)." % prep["lam_min"], "",
                  "| tag | kappa | provenance | target T/r_s | R_outer (X sites) | occupied (sea; dust) | stored C^2 - C (op) | orbitals | smallest lambda; the next below | orthonormality | inside-X weight minus lambda | propagation: Chebyshev K; tail; the occupied set's orthonormality after |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for it in prep["items"]:
                if it.get("R_outer") is None:
                    L.append("| %s | - | %s | %.3f | none | | | | | | | |" % (it["tag"], it.get("note"), it["T_target_over_rs"])); continue
                L.append("| %s | %d | %s | %.3f | %.4f (%d) | %d (%d; %d) | %.1e | %d | %s; %s | %s | %s | %d; %.1e; %.1e |" % (
                    it["tag"], it["kappa"], it["provenance"], it["T_target_over_rs"], it["R_outer"], it["iX"], it["n_occ"], it["n_sea"], it["n_dust"], it["stored_idempotency"],
                    it["n_orbitals"], fmt(it["lambda_min"], "%.6f"), fmt(it["lambda_next_below"], "%.6f"), fmt(it["orbitals_orthonormality"], "%.1e"),
                    fmt(it["inside_X_minus_lambda_max"], "%.1e"), it["cheb_K_max"], it["cheb_tail_acc"], it["occupied_orthonormality_after_propagation"]))
            L.append("")
        tau = 8.0
        if prep:
            taus = [st["tau"] for it in prep["items"] for st in (it.get("propagated_records") or [])]
            tau = taus[0] if taus else tau
        for La in orb["launches"]:
            t = La["spectrum_launch_total"]
            ser = La["series"]
            L += ["#### orbitals %s, kappa %d: %d orbitals from T/r_s %.3f (smallest lambda %s); %.0f s" % (La["tag"], La["kappa"], La["n_orbitals"], La["T_launch_over_rs"], fmt(La["lambda_min"], "%.6f"), La["seconds"]), "",
                  "- The launch spectrum on the launch record's frozen h (summed over the orbitals, of %d): weights e < 0 %.4f, abs(e) < m %.4f, e > m %.4f; the summed mean energy %+.4f." % (
                      t["n"], t["w_negative"], t["w_in_gap"], t["w_above_m"], t["mean_energy_sum"]),
                  "- Over the %d records to T/r_s %.3f: the largest summed weight beyond R_t(t) %s (T/r_s %s), beyond 2GE(t) %s (T/r_s %s); the largest single orbital beyond R_t(t) %s; the norms' largest defect %.1e; the outward current through R_t(t), summed over the orbitals and integrated over the clock (sum of current x the record step %.1f lattice units) %s." % (
                      len(ser), ser[-1]["t_over_rs"], fmt(max((s_["beyond_Rt_sum"] or 0.0) for s_ in ser)), fmt(max(ser, key=lambda s_: s_["beyond_Rt_sum"] or 0.0)["t_over_rs"], "%.3f"),
                      fmt(max(s_["beyond_2gE_sum"] for s_ in ser)), fmt(max(ser, key=lambda s_: s_["beyond_2gE_sum"])["t_over_rs"], "%.3f"),
                      fmt(max((s_["beyond_Rt_max"] or 0.0) for s_ in ser)), max(s_["norm_defect"] for s_ in ser),
                      tau, fmt(sum((s_["current_out_sum"] or 0.0) * tau for s_ in ser), "%+.4e")), "",
                  "| T/r_s | R_t | 2GE | beyond R_t: summed; largest single | beyond 2GE: summed; largest single | outward current through R_t (summed) |", "|---|---|---|---|---|---|"]
            for s_ in ser:
                if abs((s_["t_over_rs"] * 8.0) - round(s_["t_over_rs"] * 8.0)) < 1e-6 and (round(s_["t_over_rs"] * 8.0) % 4 == 0 or s_ is ser[-1]):
                    L.append("| %.3f | %s | %.4f | %s; %s | %.4e; %.4e | %s |" % (s_["t_over_rs"], fmt(s_["R_t"], "%.4f"), s_["two_gE"], fmt(s_["beyond_Rt_sum"], "%.4e"),
                                                                             fmt(s_["beyond_Rt_max"], "%.4e"), s_["beyond_2gE_sum"], s_["beyond_2gE_max"], fmt(s_["current_out_sum"], "%+.4e")))
            L.append("")
    L += ["- Every number above is printed by this script from the B files named at the top."]
    open(a.md, "w").write("\n".join(L) + "\n")
    summary = {"script": "engine_single_object_B_compose_v2.py", "label": LABEL, "no_mutual_information": "D15", "massless": "not run (D14)",
               "oracle": {"records": O["records"], "max_abs_diff": O["max_abs_diff"], "pass_1e-10": O["pass_1e-10"]}, "runs": names, "launches": recs,
               "orbitals": [{"tag": La["tag"], "kappa": La["kappa"], "n_orbitals": La["n_orbitals"], "spectrum_launch_total": La["spectrum_launch_total"],
                             "max_beyond_Rt_sum": max((s_["beyond_Rt_sum"] or 0.0) for s_ in La["series"]), "max_beyond_2gE_sum": max(s_["beyond_2gE_sum"] for s_ in La["series"]),
                             "end": La["series"][-1]} for La in (orb["launches"] if orb else [])]}
    json.dump(summary, open(a.out_json, "w"), indent=1, default=float)
    print("composed %s: %d launches (runs %s), oracle pass %s, orbitals %d" % (a.md, len(recs), names, O["pass_1e-10"], len(summary["orbitals"])), flush=True)


if __name__ == "__main__":
    main()
