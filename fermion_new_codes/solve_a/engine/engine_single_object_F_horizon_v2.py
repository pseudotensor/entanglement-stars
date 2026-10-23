#!/usr/bin/env python
"""engine_single_object_F_horizon_v2.py (builder analyst; version 2 = version 1 (called once, on the data so far, into the scratchpad) with
three additions: the history runs' window-limited arrival (the growth of the weight beyond 16 r_s over the last r_s before T/r_s 40), the
devices the B runs used, and the drain's accounting convention stated; inbox #41's revised S4, deliverable F): writes HORIZON_single_object_v1.md and its
summary json from the summaries and outputs of parts A, B, C and D (never edits them): "tables A per record and channel, B per launch with its
figure-ready columns, C, D; every number copied from the summaries, nothing retyped". Reads: A's compose summary
(engine_single_object_A_compose_v2.py) and A's family files (for the slabs and the vacuum control); B's outputs (engine_single_object_B_v1.py:
oracle, frozen and history per run, orbitals) with B's compose functions (engine_single_object_B_compose_v1.launch_rows, norm_checks; the same
code, imported); B.5's column file json (engine_single_object_B5_prep_v1.py); C's json (engine_s7_part3_relabel_v1.py). Every number in the md is
printed by this script from those files; each input's md5 is printed. No mutual information (D15); the massive channel only (D14).
Args: --a-summary --a-families --b-oracle --b-frozen --b-history --b-run-names --b-orbitals --b-prep --c-json --md --out-json."""
import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine_single_object_B_compose_v1 as BC   # noqa: E402

LABEL = BC.LABEL
REG = BC.REGIONS


def fmt(x, f="%.3e"):
    return BC.fmt(x, f)


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-summary", required=True); ap.add_argument("--a-families", required=True)
    ap.add_argument("--b-oracle", required=True); ap.add_argument("--b-frozen", required=True); ap.add_argument("--b-history", required=True)
    ap.add_argument("--b-run-names", required=True); ap.add_argument("--b-orbitals", default=None); ap.add_argument("--b-prep", default=None)
    ap.add_argument("--c-json", required=True); ap.add_argument("--md", required=True); ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    inputs = [a.a_summary] + a.a_families.split(",") + [a.b_oracle] + a.b_frozen.split(",") + a.b_history.split(",") + \
        ([a.b_orbitals] if a.b_orbitals else []) + ([a.b_prep] if a.b_prep else []) + [a.c_json]
    AS = json.load(open(a.a_summary)); AF = [json.load(open(p)) for p in a.a_families.split(",")]
    O = json.load(open(a.b_oracle)); names = a.b_run_names.split(",")
    BF = [json.load(open(p)) for p in a.b_frozen.split(",")]; BH = [json.load(open(p)) for p in a.b_history.split(",")]
    ORB = json.load(open(a.b_orbitals)) if a.b_orbitals else None; PREP = json.load(open(a.b_prep)) if a.b_prep else None
    C = json.load(open(a.c_json))
    now = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    S = {"script": "engine_single_object_F_horizon_v2.py", "composed_utc": now, "label": LABEL, "no_mutual_information": "D15", "massless": "not run (D14)",
         "inputs": {p: md5(p) for p in inputs}}
    L = ["# HORIZON_single_object_v1: the single object's own cross-surface block and transfer (S4 revised, inbox #41)", "",
         "**Label (every output):** %s." % LABEL, "",
         "Composed %s by engine_single_object_F_horizon_v2.py (the builder of the .lane-alt solve lane, analyst) from the files listed with their md5 at the end; every number below is printed by that script from those files. "
         "The owner's ruling (REVIEWA_RULING_oneobject_MI.md): one object; no second preparation, no copy doublet, no mutual information (D15). The massive channel (ma 0.2) only; no massless channel was run (D14). "
         "The independent analytic route is icloud's R1 side: %s (source %s; its section 7 compares %s)." % (now, AS["strand_R"]["scope"], AS["strand_R"]["source"], AS["strand_R"]["hreduce_compared_there"]), ""]
    # ---------------- A
    L += ["## A. Correlations across the root, per record and radial channel (post-processing of the stored states)", "",
          "Per radial channel kappa as stored; no 16 kappa weight. X: the sites inside the outer root; the identity checked on complete splits only (the outer split; the trapped annulus against its full complement); "
          "S_X agreement is an internal check, \"%s\". Band occupation: the weight of the interior components at abs(ak) >= pi/4 and in [pi/4, pi/2) (strand R's outward band), an occupation readout, never escape. "
          "The full tables (both roots, both slab geometries, the Frobenius norms, the quarter shares) are in A_single_object_v2.md." % AS["entropy_label"], ""]
    for root in ("dense", "applied"):
        L += ["### A, the %s root: items 1, 2, 4, 5" % root, "",
              "| family | T/r_s | kappa | R_outer (X sites) | n sigma > 1e-12 | sigma_max(K) | norm_F(K) | S_X agreement | SO1 residual | annulus: sigma_max; norm_F; SO1 residual | F (D = C - P) | F (C) | localized orbitals | occupation abs(ak) >= pi/4: localized; all; dust | occupation [pi/4, pi/2): localized; dust | stored C^2 - C |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in AS["rows"]:
            if r["root"] != root:
                continue
            an = r.get("annulus"); bo = r["band_occupation"]
            L.append("| %s | %.3f | %d | %.4f (%d) | %d | %.6e | %.6e | %.1e | %.1e | %s | %+.4e | %+.4e | %s | %s; %s; %s | %s; %s | %.1e |" % (
                r["family"], r["T_over_rs"], r["kappa"], r["R_outer"], r["iX"], r["n_above_1e-12"], r["sigma_max"], r["fro"], r["S_X_agreement"], r["SO1_residual"],
                ("%.4e; %.4e; %.1e" % (an["sigma_max"], an["fro"], an["SO1_residual"])) if an else "-", r["F_D"], r["F_C"], fmt(bo["localized_count"], "%d"),
                fmt(bo["localized_ge_pi4"], "%.4f"), fmt(bo["all_ge_pi4"], "%.4f"), fmt(bo["dust_ge_pi4"], "%.4f"), fmt(bo["localized_pi4_pi2"], "%.4f"), fmt(bo["dust_pi4_pi2"], "%.4f"),
                r["stored_purity"]))
        L.append("")
    L += ["### A, item 3: the 4-site slabs at d = 2, 4, 8 sites (P07 sec. 10's geometry, the dense root): sigma_max of the cross block, correlations only (strand R SO3)", "",
          "| family | T/r_s | kappa | C (d = 2; 4; 8) | P_- Killing vacuum | P continued reference | C - P_- (block difference) | C - P (block difference) |", "|---|---|---|---|---|---|---|---|"]
    for F in AF:
        for rr in F["records"]:
            for ch in (rr.get("channels") or []):
                s_ = ch["splits"].get("dense") or {}
                sl = s_.get("slabs")
                if not sl:
                    continue
                g = lambda key: "; ".join(fmt((sl.get("P07_%d" % d) or {}).get(key, {}).get("sigma_max") if (sl.get("P07_%d" % d) or {}).get(key) else None, "%.4e") for d in (2, 4, 8))
                L.append("| %s | %.3f | %d | %s | %s | %s | %s | %s |" % (F["family"], rr["T_over_rs"], ch["kappa"], g("C"), g("P_minus"), g("P"), g("C_minus_P_minus"), g("C_minus_P")))
    L.append("")
    # ---------------- B
    L += ["## B. Transfer by wavelength and depth, per launch (figure-ready columns)", "",
          "Machinery: page_history_v6b's Chebyshev propagation (the record's applied fields piecewise constant per record; the vacuum Painleve-Gullstrand exterior appended to 128 r_s, %s sites; the Bessel tail %s). "
          "Frozen: the launch record's h held 300 r_s (steps of 1 r_s; the end includes the chain-end reflection, the arrival does not). History: the solved history from the launch clock to T/r_s 40 "
          "(T/r_s 100 needs S1's engine continuation, which does not exist). Columns: 2 depths x 3 (ak_0, sigma) x 2 signs; omega_bar the mean Killing energy on the launch record's h; soft: omega_bar < m. "
          "Bound-state content: overlap with the frozen h's eigenvectors with weight >= 0.99 inside R_cut 24 r_s, never trapping. The sigma/a 8 member is a different object, printed beside, never averaged." % (
              BF[0].get("nsites"), BF[0].get("cheb_tol")), ""]
    blaunch = []
    for nm, Ff, Hh in zip(names, BF, BH):
        for mode, Fx in (("frozen", Ff), ("history", Hh)):
            for La in Fx["launches"]:
                rows = BC.launch_rows(La, mode, Fx["m"]); nc = BC.norm_checks(La)
                for c_, r in enumerate(rows):
                    l_ = La["columns"][c_]["launch"]
                    blaunch.append({"run": nm, "mode": mode, "record_file": Fx["record"], "T_launch_over_rs": La["T_launch_over_rs"], "kappa": La["kappa"], "R_outer": La["R_outer"],
                                    "R_inner": La["R_inner"], "kappa_in": La["kappa_in"], "two_gE": La["two_gE"], "T_H": La["T_H"], "column": c_,
                                    "depth": "annulus middle" if l_["depth"].startswith("annulus") else "2 sigma inside R_t", "R_c": l_["R_c"], "ak0": l_["k0"], "sigma": l_["sigma"],
                                    **{k_: r[k_] for k_ in ("omega_bar", "spread", "soft", "w_negative", "w_in_gap", "w_above_m", "beyond16_final_before_reflection", "arrival_after_launch_rs",
                                                            "pred_2rs_log", "pred_kin_log", "bound_overlap_launch", "bound_overlap_end", "interior_end", "max_radius_rs",
                                                            "beyond_Rcut_last_pre_reflection", "positive_control_unaccounted", "end_includes_reflection")},
                                    **{"end_" + k_: r["end"].get(k_) for k_ in REG}, "norm_defect_max": nc["max"], "spectrum_change_max": La["spectrum_conservation_max"]})
                L += ["### B, %s, %s: launch T/r_s %.3f, kappa %d (R_t %.4f, inner %s, kappa_in %s; 2GE %.4f; %d bound states)" % (
                    nm, mode, La["T_launch_over_rs"], La["kappa"], La["R_outer"], fmt(La["R_inner"], "%.4f"), fmt(La["kappa_in"], "%.5f"), La["two_gE"], La["bound_states"]), "",
                    "| column | depth | R_c | ak_0 | sigma | omega_bar | soft | weight e > m at launch | beyond 16 r_s before the reflection | arrival (r_s after launch) | 2 r_s ln(1/(a omega_bar)) | kappa_in^-1 ln(1/(ak_0)) | bound overlap | interior at the end | end: beyond R_t | end: beyond 2GE | end: collar 16-24 | end: far 32-48 | end: beyond R_cut | max radius (r_s) |",
                    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
                for c_, r in enumerate(rows):
                    l_ = La["columns"][c_]["launch"]
                    L.append("| %d | %s | %.2f | %+.4f | %.0f | %+.4f | %s | %.3e | %s | %s | %s | %s | %.4e | %.4e | %.4e | %.4e | %.4e | %.4e | %.4e | %.1f |" % (
                        c_, "annulus middle" if l_["depth"].startswith("annulus") else "2 sigma inside R_t", l_["R_c"], l_["k0"], l_["sigma"], r["omega_bar"], r["soft"], r["w_above_m"],
                        fmt(r["beyond16_final_before_reflection"], "%.4e"), fmt(r["arrival_after_launch_rs"], "%.1f"), fmt(r["pred_2rs_log"], "%.2f"),
                        fmt(r["pred_kin_log"], "%.2f") if r["pred_kin_log"] is not None else "n/a (ak_0 >= 1)", r["bound_overlap_launch"], r["interior_end"],
                        r["end"]["beyond_outer_root"], r["end"]["beyond_2GE"], r["end"]["collar_16_24"], r["end"]["far_32_48"], r["end"]["beyond_Rcut"], r["max_radius_rs"]))
                L.append("")
    S["B_columns"] = blaunch
    # figure-ready series (downsampled): frozen every 5 r_s, history every 0.5 r_s
    ser_out = []
    for nm, Ff, Hh in zip(names, BF, BH):
        for mode, Fx in (("frozen", Ff), ("history", Hh)):
            for La in Fx["launches"]:
                pts = []
                for s_ in La["series"]:
                    al = s_["after_launch_rs"]
                    keep = (abs(al / 5.0 - round(al / 5.0)) < 1e-9) if mode == "frozen" else (abs(al * 2.0 - round(al * 2.0)) < 1e-9)
                    if keep or s_ is La["series"][-1]:
                        pts.append({"t_over_rs": s_["t_over_rs"], "after_launch_rs": al, "R_t": s_["R_t"], "two_gE": s_["two_gE"], **{k_: s_["w"].get(k_) for k_ in REG},
                                    "reflection_reached": s_["reflection_reached"]})
                ser_out.append({"run": nm, "mode": mode, "T_launch_over_rs": La["T_launch_over_rs"], "kappa": La["kappa"], "points": pts})
    S["B_series_figure_ready"] = ser_out
    if ORB:
        L += ["### B.5: the state's own interior orbitals on the solved history (to T/r_s 40)", ""]
        if PREP:
            L += ["| tag | kappa | provenance | target T/r_s | R_outer (X sites) | orbitals (lambda >= %s) | smallest lambda; the next below | orthonormality | inside-X weight minus lambda |" % PREP["lam_min"],
                  "|---|---|---|---|---|---|---|---|---|"]
            for it in PREP["items"]:
                if it.get("R_outer") is None:
                    continue
                L.append("| %s | %d | %s | %.3f | %.4f (%d) | %d | %s; %s | %s | %s |" % (it["tag"], it["kappa"], it["provenance"], it["T_target_over_rs"], it["R_outer"], it["iX"], it["n_orbitals"],
                                                                                fmt(it["lambda_min"], "%.6f"), fmt(it["lambda_next_below"], "%.6f"), fmt(it["orbitals_orthonormality"], "%.1e"),
                                                                                fmt(it["inside_X_minus_lambda_max"], "%.1e")))
            L.append("")
        L += ["| tag | kappa | orbitals | launch spectrum (summed): e < 0; abs(e) < m; e > m | at T/r_s 40: beyond R_t (summed; largest single) | at T/r_s 40: beyond 2GE (summed) | largest beyond R_t over the clock (summed; at T/r_s) | outward current through R_t at T/r_s 40 (summed) | norms' largest defect |",
              "|---|---|---|---|---|---|---|---|---|"]
        S["B5"] = []
        for La in ORB["launches"]:
            t = La["spectrum_launch_total"]; ser = La["series"]; e_ = ser[-1]
            mx = max(ser, key=lambda s_: s_["beyond_Rt_sum"] or 0.0)
            L.append("| %s | %d | %d | %.4f; %.4f; %.4f | %s; %s | %.4e | %s (%.3f) | %s | %.1e |" % (
                La["tag"], La["kappa"], La["n_orbitals"], t["w_negative"], t["w_in_gap"], t["w_above_m"], fmt(e_["beyond_Rt_sum"], "%.4e"), fmt(e_["beyond_Rt_max"], "%.4e"),
                e_["beyond_2gE_sum"], fmt(mx["beyond_Rt_sum"], "%.4e"), mx["t_over_rs"], fmt(e_["current_out_sum"], "%+.4e"), max(s_["norm_defect"] for s_ in ser)))
            S["B5"].append({"tag": La["tag"], "kappa": La["kappa"], "n_orbitals": La["n_orbitals"], "spectrum_launch_total": t,
                            "series": [{k_: s_[k_] for k_ in ("t_over_rs", "R_t", "two_gE", "beyond_Rt_sum", "beyond_Rt_max", "beyond_2gE_sum", "beyond_2gE_max", "current_out_sum")} for s_ in ser]})
        L.append("")
    # ---------------- C
    ch = {c["kappa"]: c for c in C["channels"]}
    L += ["## C. The delivered part 3 relabelled (the r_s/a 32 record at T/r_s %.3f)" % C["T0_over_rs"], "",
          "- The window: %s; X %.4f, w %.0f; the plateau shells n_L = R %.0f, n_R = R %.0f; the ramp shells %s." % (C["window"]["form"], C["window"]["X"], C["window"]["width"], C["window"]["n_L"],
                                                                                                                 C["window"]["n_R"], ", ".join("%.0f" % x for x in C["window"]["ramp_shells"])),
          "- The roots (part A): dense outer %.4f, inner %.4f; applied outer %.4f, inner %.4f." % (C["inside"]["dense"]["R_outer"], C["inside"]["dense"]["R_inner"], C["inside"]["applied"]["R_outer"], C["inside"]["applied"]["R_inner"]),
          "- The bands: %s." % "; ".join("kappa %d: energy %d shells (R %.0f to %.0f), momentum %d shells (R %.0f to %.0f)" % (
              k_, c["energy_band"]["shells"], c["energy_band"]["first_R"], c["energy_band"]["last_R"], c["momentum_band"]["shells"], c["momentum_band"]["first_R"], c["momentum_band"]["last_R"]) for k_, c in sorted(ch.items())),
          "- Inside the outer root: dense %s (margin %.4f sites), applied %s (margin %.4f sites); the prints flag %s." % (C["inside"]["dense"]["bands_inside_outer_root"], C["inside"]["dense"]["margin_to_outer"],
                                                                                                            C["inside"]["applied"]["bands_inside_outer_root"], C["inside"]["applied"]["margin_to_outer"], C["prints"]), "",
          "| kappa | theta | DeltaC(0) op | C_+(0) - C_-(0) op |", "|---|---|---|---|"]
    for k_, c in sorted(ch.items()):
        for nrm in c["norms"]:
            L.append("| %d | %.6f | %.6e | %.6e |" % (k_, nrm["theta"], nrm["DeltaC0_op"], nrm["Cplus_minus_Cminus_op"]))
    L += ["", "| kappa | the exact bound beyond R_t(t) (T/r_s: value) | over DeltaC(0) op | over C_+(0) - C_-(0) op | the collar D_det at most |", "|---|---|---|---|---|"]
    for k_, c in sorted(ch.items()):
        L.append("| %d | %s | %s | %s | %.3e |" % (k_, "; ".join("%s: %.3e" % (t_, v_) for t_, v_ in c["bound_beyond_Rt"].items()),
                                                    "; ".join("%s: %.4e" % (t_, v_["over_DeltaC0"]) for t_, v_ in c["ratios"].items()),
                                                    "; ".join("%s: %.4e" % (t_, v_["over_Cplus_minus_Cminus"]) for t_, v_ in c["ratios"].items()), c["collar_D_det_max"]))
    clocks = sorted({t_ for c in ch.values() for t_ in c["bound_beyond_Rt"]})
    L += ["", "- The numbers print: %s (the bands inside the outer root: dense %s, applied %s). The lane's record measured them beyond R_t(t) (part 3's D_det beyond the surface at T/r_s %s), "
          "not at the collar, whose D_det stayed at the rounding floor (at most %s): the sentence's \"reached the collar\" reads \"reached beyond the outer root\" for these numbers (flagged for the lead). "
          "ITEM7_s7_part3_sentence_v2.md stays withdrawn in its 2-preparation form." % (C["prints"], C["inside"]["dense"]["bands_inside_outer_root"], C["inside"]["applied"]["bands_inside_outer_root"],
                                                                                 " and ".join(clocks), "%.1e" % max(c["collar_D_det_max"] for c in ch.values())), ""]
    S["C"] = {"json": a.c_json, "prints": C["prints"]}
    # ---------------- D
    L += ["## D. Validation", ""]
    for fam, c in AS["strand_R_checks"].items():
        L.append("- A.2 (%s): the largest SO1 residual %.1e (outer), %s (annulus); the largest S_X agreement %.1e (outer), %s (annulus); target 1e-10; sigma_max <= 1/2 holds: %s; the largest stored ||C^2 - C||_op %.1e; the largest abs(F_D - F_C) %.1e." % (
            fam, c["outer_SO1_residual_max"], fmt(c["annulus_SO1_residual_max"], "%.1e"), c["outer_SO2_agreement_max"], fmt(c["annulus_SO2_agreement_max"], "%.1e"), c["SO1_bound_half_holds"],
            c["stored_purity_max"], c["FD_minus_FC_max"]))
    for F in AF:
        for rr in F["records"]:
            if rr.get("vacuum_control"):
                vc = rr["vacuum_control"]
                L.append("- A's vacuum control (%s, T/r_s %.3f): the exterior appended to %d sites; the slab blocks of P_- against the stored chain's, the largest difference %.1e." % (
                    F["family"], rr["T_over_rs"], vc["n_ext_sites"], max(q["max_abs_diff"] for q in vc["rows"])))
    for d in AS["device_checks"]:
        L.append("- A's device check (%s): %d readouts compared, the largest absolute difference %.1e (at %s)." % (d["label"], d["leaves_compared"], d["max_abs_diff"], d["max_abs_diff_at"]))
    L.append("- B's dense oracle (%s): %d records; the largest differences %s; pass at 1e-10: %s." % (O["fixture"], O["records"], "; ".join("%s %.1e" % (k_, v_) for k_, v_ in O["max_abs_diff"].items()), O["pass_1e-10"]))
    for nm, Ff, Hh in zip(names, BF, BH):
        for mode, Fx in (("frozen", Ff), ("history", Hh)):
            for La in Fx["launches"]:
                nc = BC.norm_checks(La); rows = BC.launch_rows(La, mode, Fx["m"])
                extra = ""
                if mode == "frozen":
                    un = [abs(r["positive_control_unaccounted"]) for r in rows if r["positive_control_unaccounted"] is not None]
                    un_r = [abs(r["positive_control_unaccounted"]) for r in rows if r["positive_control_unaccounted"] is not None and abs(abs(La["columns"][r["column"]]["launch"]["k0"]) - 2.0 * 3.141592653589793 / 64.0) < 1e-12]
                    extra = "; the Killing spectrum's largest change %.1e (target 1e-10); the drain (positive control): the largest unaccounted weight %.1e over the 12 columns, %.1e over the wavelength-r_s columns" % (
                        La["spectrum_conservation_max"], max(un) if un else float("nan"), max(un_r) if un_r else float("nan"))
                if mode == "history":
                    ser = La["series"]; tE = ser[-1]["after_launch_rs"]
                    prv = [s_ for s_ in ser if s_["after_launch_rs"] <= tE - 1.0 + 1e-9]
                    if prv:
                        g16 = [ser[-1]["w"]["beyond_16rs"][c_] - prv[-1]["w"]["beyond_16rs"][c_] for c_ in range(len(La["columns"]))]
                        extra += "; the weight beyond 16 r_s grew by up to %.3e over the last %.3f r_s before the window's end at T/r_s %.3f (the arrival clocks are relative to the window-end value)" % (
                            max(g16), tE - prv[-1]["after_launch_rs"], ser[-1]["t_over_rs"])
                cc = La.get("cubic_control")
                if cc:
                    extra += "; the cubic control (T/r_s %.3f to %.3f): the columns' largest element difference %.3e, the readouts' changes %s" % (
                        cc["window_rs"][0], cc["window_rs"][1], cc["columns_diff_max"], "; ".join("%s %.3e" % (k_, v_) for k_, v_ in cc["region_diff"].items()))
                L.append("- B, %s, %s, T/r_s %.3f, kappa %d: the norms' largest defect %.2e over %d steps (first above 1e-12 at %s r_s after launch; %.1e per step)%s." % (
                    nm, mode, La["T_launch_over_rs"], La["kappa"], nc["max"], nc["steps"], fmt(nc["first_above_1e-12_after_launch_rs"], "%.1f"), nc["max_over_steps"], extra))
    if PREP:
        L.append("- B.5's columns: %s." % "; ".join("%s kappa %d: orthonormality %s, inside-X weight minus lambda %s, the propagated occupied set's orthonormality %.1e" % (
            it["tag"], it["kappa"], fmt(it["orbitals_orthonormality"], "%.1e"), fmt(it["inside_X_minus_lambda_max"], "%.1e"), it["occupied_orthonormality_after_propagation"]) for it in PREP["items"] if it.get("R_outer") is not None))
    L += ["", "## Open points", "",
          "- B.1 (ii) and B.5 end at T/r_s 40: the extension to 100 needs S1's engine continuation B, which does not exist.",
          "- No massless channel (D14); no mutual information anywhere (D15).",
          "- The exterior-shell Killing spectra of B.4 are printed in the frozen runs' files every 25 r_s; this file does not reduce them to one number against the launch spectrum.",
          "- B.6's clocks are the sigma/a 8 member's own trapped episodes (its record has no root at T/r_s 30.0).",
          "- C's \"reached the collar\" wording reads \"reached beyond the outer root\" (flagged for the lead).",
          "- The drain's accounting: 1 - (the weight beyond R_cut at the last clock before the reflection) - (the bound-state overlap); continuum weight still inside R_cut at that clock (for example slow, near-threshold massive waves) counts as unaccounted; it is not resolved further here.",
          "- On the history runs the arrival clocks are window-limited: the weight beyond 16 r_s still grows at T/r_s 40 (D lists the growth over the last r_s).",
          "- Devices: the sigma/a 4 frozen and history runs and the oracle ran on the second H100's GPU; B.5's propagation and the sigma/a 8 runs on that pod's CPU (8 threads, the coordinator's cap); B.5's columns on the box CPU; part A on pod 1's GPU (sigma/a 4) and the box CPU (sigma/a 8, r_s/a 32), with the device check in D.", "",
          "## Inputs (md5)", ""] + ["- %s %s" % (v_, k_) for k_, v_ in S["inputs"].items()] + ["", "**Label (every output):** %s." % LABEL]
    open(a.md, "w").write("\n".join(L) + "\n")
    json.dump(S, open(a.out_json, "w"), indent=1, default=float)
    print("composed %s (%d lines) and %s; %d B columns; B.5 %s" % (a.md, len(L), a.out_json, len(blaunch), "yes" if ORB else "no"), flush=True)


if __name__ == "__main__":
    main()
