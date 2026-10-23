#!/usr/bin/env python
"""engine_single_object_F_horizon_v6.py (builder analyst, the HORIZON version 2 builder of the lead's turn 47; engine spec 38.4 (1) (i) to
(iv) and 39.5 (1) (v) to (x); brief lead/BRIEF_horizon_v2_builder.md): writes HORIZON_single_object_v2.md and its summary json from the
outputs of part A (version 1's files, unchanged), the B runs of version 2 (engine_single_object_B_v2.py: fixed_field and history with the
comparators and the signed readout; dust, B.5 version 2), B.5 version 2's columns (engine_single_object_B5_prep_v2.py), the stored states'
own readouts (engine_single_object_slices_v2.py: the full detector density, the five rates), part C (version 1's file) and, for the
regression of version 2's interior columns, version 1's B files (staged under neutral names, md5-identical). Never edits an input; every
number in the md is printed by this script from them; each input's md5 is printed. The ruled items enter the definitions, not as labels on
version 1 (39.5 (1)). A final check: none of sec. 4's words of requirements_new.txt, and no "escape", in the composed text.
Args: see main(). Refuses to overwrite."""
import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

LABEL = ("the single object's own cross-surface 2-point block and transfer on the solved history of the massive collapse, per radial channel; "
         "the interior beyond the hydrodynamic gate H sigma > 0.1 from T/r_s 11.75, the record's own limit; the massless channel, where present, "
         "a test channel under owner decision D14")
REG = ("beyond_outer_root", "beyond_2GE", "beyond_16rs", "collar_16_24", "far_32_48", "beyond_Rcut")
FORBIDDEN = ("frozen", "freez", "prediction", "escape", "toy model", "semiclassical", "mean-field", "discretized", "event horizon forms", "singularity forms",
             "no black object", "obligation", "bonds keep", "bonds always", "unitary completion", "rectangle", "rectangular", "allocation exponent", "uv complete",
             "remnant", "page fraction", "stationary luminosity", "static star", "emergent gravity", "derived einstein", "mutual information", "near-maximal")


def fmt(x, f="%.3e"):
    return (f % x) if isinstance(x, (int, float)) and not isinstance(x, bool) and x is not None and x == x else "-"


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def depth_name(l_):
    d = l_["depth"]
    return "annulus middle" if d.startswith("annulus") else ("2 sigma inside R_t" if "inside" in d else "R_t + 2 sigma (comparator)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-summary", required=True); ap.add_argument("--a-families", required=True); ap.add_argument("--c-json", required=True)
    ap.add_argument("--v1-oracle", required=True); ap.add_argument("--v1-fixed", required=True, help="version 1's fixed-field files, production then sigma/a 8")
    ap.add_argument("--v1-history", required=True, help="version 1's history files, production then sigma/a 8")
    ap.add_argument("--b2-fixed", required=True); ap.add_argument("--b2-history", required=True); ap.add_argument("--b2-dust", required=True)
    ap.add_argument("--b5-prep", required=True); ap.add_argument("--slices", required=True, help="slices_v2 jsons: production, continuation, sigma/a 8")
    ap.add_argument("--run-names", default="sigma/a 4 (production),sigma/a 8 (B.6 scale control)")
    ap.add_argument("--md", required=True); ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    for p in (a.md, a.out_json):
        if os.path.exists(p):
            sys.exit("refusing to overwrite %s" % p)
    names = a.run_names.split(",")
    sp = lambda s: s.split(",")
    inputs = [a.a_summary] + sp(a.a_families) + [a.c_json, a.v1_oracle] + sp(a.v1_fixed) + sp(a.v1_history) + sp(a.b2_fixed) + \
        [p.replace(".json", ".npz") for p in sp(a.b2_fixed)] + sp(a.b2_history) + [p.replace(".json", ".npz") for p in sp(a.b2_history)] + sp(a.b2_dust) + sp(a.b5_prep) + sp(a.slices)
    AS = json.load(open(a.a_summary)); AF = [json.load(open(p)) for p in sp(a.a_families)]; C = json.load(open(a.c_json)); O = json.load(open(a.v1_oracle))
    V1F = [json.load(open(p)) for p in sp(a.v1_fixed)]; V1H = [json.load(open(p)) for p in sp(a.v1_history)]
    BF = [json.load(open(p)) for p in sp(a.b2_fixed)]; BFZ = [np.load(p.replace(".json", ".npz")) for p in sp(a.b2_fixed)]
    BH = [json.load(open(p)) for p in sp(a.b2_history)]; BHZ = [np.load(p.replace(".json", ".npz")) for p in sp(a.b2_history)]
    BD = [json.load(open(p)) for p in sp(a.b2_dust)]; PREP = [json.load(open(p)) for p in sp(a.b5_prep)]; SL = [json.load(open(p)) for p in sp(a.slices)]
    now = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    S = {"script": "engine_single_object_F_horizon_v6.py", "composed_utc": now, "label": LABEL, "D15_D14": "observed: massive channel only, no second preparation, none of the quantity D15 removes",
         "inputs": {p: md5(p) for p in inputs}}
    L = ["# HORIZON_single_object_v2: the single object's own cross-surface block and transfer (S4 revised; engine spec 38.4 (1) and 39.5 (1))", "",
         "**Label (every output):** %s." % LABEL, "",
         "Composed %s by engine_single_object_F_horizon_v6.py (the HORIZON version 2 builder of the .lane-alt solve lane, analyst) from the files listed with their md5 at the end; every number "
         "below is printed by that script from those files. The owner's one-object ruling holds throughout (one object, no second preparation, D15 and D14 observed: the massive channel ma 0.2 only). "
         "Version 1 (HORIZON_single_object_v1.md) stands as delivered; this version carries the reviews #45, #46 and #51 in its definitions." % now, "",
         "## Definitions (the ruled items of 38.4 (1) and 39.5 (1))", "",
         "- (iii) Fixed-field: a record's applied fields held fixed while a column propagates (the launch record's h for 300 r_s); history: the solved history record by record (piecewise-constant applied fields), "
         "the production's records to T/r_s 40 and #36 B's continuation's after 40, to the last record held when each run started.",
         "- (i) The comparator: per launch record and channel a column at R_t + 2 sigma outside the outer root with each k_0 and envelope (both signs of k_0), beside the interior columns (the annulus middle and 2 sigma inside R_t, "
         "version 1's columns, reproduced: D). The arrival: the clock at which the weight beyond 16 r_s reaches half its value at the last readout before the chain-end reflection, interpolated linearly between steps of "
         "0.25 r_s (fixed field) or between records (history). The delay: the arrival minus the arrival of the comparator with the same k_0 and envelope, printed beside 2 r_s ln(1/(a omega_bar)) and kappa_in^-1 ln(1/(ak_0)).",
         "- (v) The reconstruction residuals (C_X - C_X^2 - K K^dagger and the two entropy routes) are internal checks of the Loewdin-reconstructed occupied projector; the stored covariance's own purity defect "
         "is printed beside them (its largest value on the production %s, #45)." % fmt(AS["strand_R_checks"]["rs64_sigma4"]["stored_purity_max"], "%.2e"),
         "- (vi) The share of a column's or the dust's weight at pi/4 <= |ak| < pi/2 is an unsigned windowed share (part A's untapered transform of the interior components, both momentum signs and both spinor "
         "components), kept as such. The outward readout is the signed projection on the positive intrinsic Dirac branch with the local group-velocity sign: the region's weight split by sine tapers of 32 sites "
         "with half overlap (their squares add to one on the region), each tapered piece's orthonormal transform zero-padded to 128; at each piece's centre the spinors u_+-(q) of D(q) = m sigma_z + sin q sigma_x + "
         "(kappa/R) sigma_y (the flow term (v/2) sin 2q is a multiple of the identity, so h's local eigenvectors are D's) and the local group velocities v_+-(q) = +-N sin q cos q/E(q) + v cos 2q (N, v the applied "
         "fields at the centre, v <= 0 the inflow); printed: the weight on the + branch with v_+ > 0 (+ out) and with v_+ <= 0 (+ in), the - branch's, and the + branch on the signed interval pi/4 < q < pi/2, "
         "which moves outward at every flow speed for N > 0, v <= 0 (#51; hreduce SO4); tapers of 16 and 64 sites are the control (D).",
         "- (vii) B.5 version 2 propagates the dust-tagged occupied columns (the sector with occupation f, all of them) and, as the comparator, the sea's orbitals localized inside the outer root (the right singular "
         "vectors of Pi_X Phi_sea with weight at least 0.99 inside, version 1's selection on the sea sector); these are the selected sets' readouts. The full detector density of the state's excess, Tr[Pi_D (C - P)] "
         "over the whole occupied set and the reference, is read from the stored states themselves (E). The radial operator h_m anticommutes with the on-site antiunitary (-1)^j sigma_y K (checked on every slice, E), "
         "so the sea and the reference each keep site density one: the excess number density beyond any cut is the dust's, Tr[Pi_D (C - P)] = Tr[Pi_D C_dust], and the pairs appear in energy, not in number.",
         "- (viii) The 99 percent classifier (the eigenvectors of the held h whose weight inside R_cut 24 r_s is at least 0.99) is a finite-box localization classifier, not the infinite-exterior bound-state "
         "projector; each column's Killing spectral weight above m is printed beside its mean energy omega_bar (a mean below m does not exclude support above the threshold).",
         "- (ix) Five rates, separate columns, never summed: (1) the fixed-cut number current through R_t, -2 Im Tr(h_XY X_YX) (outward positive); (2) the moving-cut reassignment, -rho(R_t) dR_t/dt (and its "
         "step form over the record step); (3) the complete reference term Gamma_P = P_dot + i[h, P] beyond the cut, Tr[Pi_Y Gamma_P] (number) and its Killing-energy form at the fixed h; (4) the Killing-energy "
         "flux through R_t, Tr[X (i/2)[h^2, Pi_Y]]; (5) the Gauss charge loss, -dE/dT of the record's charge.",
         "- (x) Every readout of weight beyond the outer root is a transfer beyond the moving root within the run.", ""]
    # ---------------------------------------------------------------- A
    L += ["## A. Correlations across the root (version 1's part A; the residuals as internal checks with the stored purity defect beside)", "",
          "Per radial channel kappa as stored, no 16 kappa weight; the dense root; X the sites inside the outer root. The full tables (both roots, both slab geometries, the quarter shares) are A_single_object_v2.md.", "",
          "| family | T/r_s | kappa | R_outer (X sites) | n sigma > 1e-12 | sigma_max(K) | norm_F(K) | internal checks: SO1 residual; S_X agreement | the stored covariance's purity defect | cut current F (D = C - P) | the dust's unsigned windowed share pi/4 <= abs(ak) < pi/2; abs(ak) >= pi/4 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in AS["rows"]:
        if r["root"] != "dense":
            continue
        bo = r["band_occupation"]
        L.append("| %s | %.3f | %d | %.4f (%d) | %d | %.6e | %.6e | %.1e; %.1e | %.1e | %+.4e | %s; %s |" % (
            r["family"], r["T_over_rs"], r["kappa"], r["R_outer"], r["iX"], r["n_above_1e-12"], r["sigma_max"], r["fro"], r["SO1_residual"], r["S_X_agreement"], r["stored_purity"],
            r["F_D"], fmt(bo["dust_pi4_pi2"], "%.4f"), fmt(bo["dust_ge_pi4"], "%.4f")))
    L += ["", "### A, item 3: the 4-site slabs at d = 2, 4, 8 sites (P07 sec. 10's geometry, the dense root): sigma_max of the cross block, correlations only", "",
          "| family | T/r_s | kappa | C (d = 2; 4; 8) | P_- Killing vacuum of the held fields | P continued reference | C - P_- | C - P |", "|---|---|---|---|---|---|---|---|"]
    for F in AF:
        for rr in F["records"]:
            for ch in (rr.get("channels") or []):
                sl = (ch["splits"].get("dense") or {}).get("slabs")
                if not sl:
                    continue
                g = lambda key: "; ".join(fmt((sl.get("P07_%d" % d) or {}).get(key, {}).get("sigma_max") if (sl.get("P07_%d" % d) or {}).get(key) else None, "%.4e") for d in (2, 4, 8))
                L.append("| %s | %.3f | %d | %s | %s | %s | %s | %s |" % (F["family"], rr["T_over_rs"], ch["kappa"], g("C"), g("P_minus"), g("P"), g("C_minus_P_minus"), g("C_minus_P")))
    L.append("")
    # ---------------------------------------------------------------- B
    L += ["## B. Transfer by wavelength and depth, per launch (fixed-field and history; with the comparators and the signed readout)", "",
          "Machinery: page_history_v6b's Chebyshev action (the record's applied fields on the engine's sites, the vacuum Painleve-Gullstrand exterior at the record's E beyond them, %s sites = 128 r_s; the Bessel tail %s). "
          "Fixed field: the launch record's h held 300 r_s in steps of 0.25 r_s (the end includes the chain-end reflection, the arrival does not). History: to the last record held at each run's start (printed). "
          "Columns 0 to 11: version 1's (the annulus middle and 2 sigma inside R_t; 3 (ak_0, sigma) = (2 pi/64, 16), (pi/4, 8), (pi/2, 4); both signs); 12 to 17: the comparators at R_t + 2 sigma. "
          "The sigma/a 8 member is a different object, printed beside, never averaged." % (BF[0].get("nsites"), BF[0].get("cheb_tol")), ""]
    blaunch = []
    for nm, Ff, Hh in zip(names, BF, BH):
        for mode, Fx in (("fixed-field", Ff), ("history", Hh)):
            for La in Fx["launches"]:
                span = ("held fixed %.0f r_s" % (La["T_end_over_rs"] - La["T_launch_over_rs"])) if mode == "fixed-field" else ("to T/r_s %.3f" % La["T_end_over_rs"])
                hdr = "### B, %s, %s: launch T/r_s %.3f, kappa %d (R_t %.4f, inner %s, kappa_in %s; 2GE %.4f; %d finite-box classifier states; %s)" % (
                    nm, mode, La["T_launch_over_rs"], La["kappa"], La["R_outer"], fmt(La["R_inner"], "%.4f"), fmt(La["kappa_in"], "%.5f"), La["two_gE"], La["classifier_states"], span)
                L += [hdr, "", "Launch content: the Killing spectrum on the launch record's h and the signed projection of the whole column at launch.", "",
                      "| column | depth | R_c | ak_0 | sigma | omega_bar | Killing weight above m; below -m; in the gap | + out | + in | - branch | + on pi/4 < q < pi/2 | unsigned windowed share | classifier overlap at launch |",
                      "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
                sg = La["signed_launch"]["whole"]
                for c_, c in enumerate(La["columns"]):
                    l_ = c["launch"]
                    L.append("| %d | %s | %.2f | %+.4f | %.0f | %+.4f | %.3e; %.3e; %.3e | %.4f | %.4f | %.4f | %.4f | %.4f | %.4e |" % (
                        c_, depth_name(l_), l_["R_c"], l_["k0"], l_["sigma"], c["omega_bar"], c["w_above_m"], c["w_below_minus_m"], c["w_in_gap"], sg["plus_out"][c_], sg["plus_in"][c_],
                        sg["minus"][c_], sg["plus_pi4_pi2"][c_], sg["unsigned_pi4_pi2"][c_], c["classifier_overlap_launch"]))
                lastsg = La["signed_series"][-1] if La["signed_series"] else None
                L += ["", "Transfer readouts (%s): the arrivals, the delays against the comparator beside the two forms, the weights at the end, the signed projection of the part beyond R_t at %s." % (
                    "the end includes the chain-end reflection" if mode == "fixed-field" else "run-limited: the history ends at T/r_s %.3f" % La["T_end_over_rs"],
                    (("%.0f r_s after launch" % lastsg["after_launch_rs"]) if mode == "fixed-field" else ("T/r_s %.3f" % lastsg["t_over_rs"])) if lastsg else "-"), "",
                      "| column | beyond 16 r_s, last before the reflection | arrival (r_s after launch) | comparator's arrival | delay: arrival minus the comparator's | 2 r_s ln(1/(a omega_bar)) | kappa_in^-1 ln(1/(ak_0)) | end: beyond R_t; beyond 2GE; collar 16-24; far 32-48; beyond R_cut | classifier overlap at the end | interior at the end | max radius (r_s) | beyond R_t: total; + out; + in |",
                      "|---|---|---|---|---|---|---|---|---|---|---|---|"]
                for c_, c in enumerate(La["columns"]):
                    e_ = c["end"]
                    bt = (lastsg or {}).get("beyond_Rt") or {}
                    L.append("| %d | %s | %s | %s | %s | %s | %s | %s | %.4e | %.4e | %.1f | %s; %s; %s |" % (
                        c_, fmt(c["beyond16_final_before_reflection"], "%.4e"), fmt(c["arrival_after_launch_rs"], "%.2f"), fmt(c.get("comparator_arrival_after_launch_rs"), "%.2f"),
                        fmt(c.get("delay_minus_comparator_rs"), "%+.2f"), fmt(c["pred_2rs_log"], "%.2f"), fmt(c["pred_kin_log"], "%.2f") if c["pred_kin_log"] is not None else "n/a (ak_0 >= 1)",
                        "; ".join(fmt(e_.get(k_), "%.3e") for k_ in REG), c["classifier_overlap_end"], c["interior_end"], c["max_radius_rs"],
                        fmt((bt.get("total") or [None] * 18)[c_], "%.3e"), fmt((bt.get("plus_out") or [None] * 18)[c_], "%.3e"), fmt((bt.get("plus_in") or [None] * 18)[c_], "%.3e")))
                    blaunch.append({"run": nm, "mode": mode, "T_launch_over_rs": La["T_launch_over_rs"], "kappa": La["kappa"], "column": c_, "depth": depth_name(c["launch"]),
                                    "R_c": c["launch"]["R_c"], "ak0": c["launch"]["k0"], "sigma": c["launch"]["sigma"], "comparator": c["launch"]["comparator"],
                                    **{k_: c.get(k_) for k_ in ("omega_bar", "w_above_m", "w_below_minus_m", "w_in_gap", "soft", "beyond16_final_before_reflection", "arrival_after_launch_rs",
                                                                "comparator_arrival_after_launch_rs", "delay_minus_comparator_rs", "pred_2rs_log", "pred_kin_log", "classifier_overlap_launch",
                                                                "classifier_overlap_end", "interior_end", "max_radius_rs", "end_includes_reflection")},
                                    **{"end_" + k_: e_.get(k_) for k_ in REG},
                                    "signed_launch": {k_: sg[k_][c_] for k_ in ("total", "plus_out", "plus_in", "minus", "minus_out", "minus_in", "plus_pi4_pi2", "minus_negpi2_negpi4", "unsigned_pi4_pi2")},
                                    "signed_beyond_Rt_last": {k_: (bt.get(k_) or [None] * 18)[c_] for k_ in ("total", "plus_out", "plus_in", "plus_pi4_pi2", "unsigned_pi4_pi2")} if bt else None,
                                    "T_end_over_rs": La["T_end_over_rs"]})
                L.append("")
    S["B_columns"] = blaunch
    # ---------------------------------------------------------------- B.5
    L += ["## B.5 version 2: the dust-tagged occupied columns on the solved history, the sea-localized orbitals beside as the comparator", "",
          "The selected sets' readouts (definition (vii)); the dust's columns carry the stored occupations f; every weight beyond R_t(t) is a transfer beyond the moving root within the run. "
          "The five rates for the dust sector: (1), (2) and (4) here per record (the reference term (3) has no dust counterpart: the dust has no reference; the record's Gauss charge loss (5) is global and printed beside).", ""]
    S["B5"] = []
    for nm, D_, P_ in zip(names, BD, PREP):
        L += ["### B.5, %s (%s)" % (nm, D_.get("columns_npz")), "",
              "| tag | kappa | provenance | target T/r_s | R_outer (X sites) | occupied (sea; dust) | stored purity defect | dust occupations | the dust's weight beyond R_t at launch (summed) | sea-localized orbitals (smallest lambda; the next below) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for it in P_["items"]:
            if it.get("R_outer") is None:
                continue
            L.append("| %s | %d | %s | %.3f | %.4f (%d) | %d; %d | %.1e | %s | %.4e | %d (%s; %s) |" % (
                it["tag"], it["kappa"], it["provenance"], it["T_target_over_rs"], it["R_outer"], it["iX"], it["n_sea"], it["n_dust"], it["stored_idempotency"],
                ", ".join("%.3f" % x for x in it["dust_occupations"]), it["dust_beyond_Rt_sum"], it["sealoc_n"], fmt(it["sealoc_lambda_min"], "%.6f"), fmt(it["sealoc_lambda_next_below"], "%.6f")))
        L.append("")
        for La in D_["launches"]:
            ld = La["launch"]["dust"]; ls = La["launch"]["sealoc"]
            L += ["#### %s, kappa %d: from T/r_s %.3f to %.3f" % (La["tag"], La["kappa"], La["T_launch_over_rs"], La["T_end_over_rs"]), "",
                  "- The launch spectra on the launch record's h (occupation-weighted sums): the dust (%d columns) below -m %.4f, in the gap %.4f, above m %.4f (summed mean energy %+.4f); the sea-localized "
                  "(%d orbitals) below -m %.4f, in the gap %.4f, above m %.4f." % (ld["n"], ld["w_below_minus_m"], ld["w_in_gap"], ld["w_above_m"], ld["mean_sum"], ls["n"], ls["w_below_minus_m"], ls["w_in_gap"], ls["w_above_m"]), "",
                  "| T/r_s | R_t | 2GE | dust beyond R_t (largest single) | dust beyond 2GE | sea-localized beyond R_t | sea-localized beyond 2GE | (1) dust fixed-cut number current | (2) dust moving-cut reassignment over the step | the dust's fixed-cut change over the step | (4) dust Killing-energy flux |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
            sd = La["series"]["dust"]; ss = La["series"]["sealoc"]
            for j, (p_, q_) in enumerate(zip(sd, ss)):
                t_ = p_["t_over_rs"]
                if not (abs(t_ - round(t_)) < 1e-9 or j == 0 or j == len(sd) - 1):
                    continue
                L.append("| %.3f | %s | %.4f | %s (%s) | %.4e | %s | %.4e | %s | %s | %s | %s |" % (
                    t_, fmt(p_["R_t"], "%.4f"), p_["two_gE"], fmt(p_["beyond_Rt_sum"], "%.4e"), fmt(p_["beyond_Rt_max"], "%.3e"), p_["beyond_2gE_sum"], fmt(q_["beyond_Rt_sum"], "%.4e"), q_["beyond_2gE_sum"],
                    fmt(p_.get("fixed_cut_number_current"), "%+.3e"), fmt(p_.get("moving_cut_reassignment_step"), "%+.3e"), fmt(p_.get("fixed_cut_change_step"), "%+.3e"), fmt(p_.get("killing_energy_flux"), "%+.3e")))
            sgr = La["signed_series"]["dust"]
            if sgr:
                L += ["", "The dust's signed projection (inside R_t; beyond R_t): + out; + in; the + branch on pi/4 < q < pi/2; the region's total.", "",
                      "| T/r_s | inside R_t: + out; + in; + on (pi/4, pi/2); total (summed over the columns) | beyond R_t: + out; + in; + on (pi/4, pi/2); total |", "|---|---|---|"]
                for q in sgr:
                    if not (abs(q["t_over_rs"] - round(q["t_over_rs"] / 4.0) * 4.0) < 1e-9 or q is sgr[0] or q is sgr[-1]):
                        continue
                    ins = q.get("inside_Rt") or {}; bey = q.get("beyond_Rt") or {}
                    f4 = lambda d_: "; ".join(fmt(sum(d_[k_]) if d_.get(k_) else None, "%.4f") for k_ in ("plus_out", "plus_in", "plus_pi4_pi2", "total"))
                    L.append("| %.3f | %s | %s |" % (q["t_over_rs"], f4(ins), f4(bey)))
            L.append("")
            S["B5"].append({"run": nm, "tag": La["tag"], "kappa": La["kappa"], "T_launch_over_rs": La["T_launch_over_rs"], "T_end_over_rs": La["T_end_over_rs"], "launch": {k_: {kk: v_ for kk, v_ in La["launch"][k_].items() if kk != "per_column"} for k_ in La["launch"]},
                            "series_dust": sd, "series_sealoc": [{k_: q_.get(k_) for k_ in ("t_over_rs", "beyond_Rt_sum", "beyond_2gE_sum", "fixed_cut_number_current")} for q_ in ss]})
    # ---------------------------------------------------------------- E
    L += ["## E. The stored states' own readouts: the full detector density and the five rates, per slice and channel", "",
          "engine_single_object_slices_v2.py on the stored slices (no propagation): C over the whole occupied set (Loewdin-re-orthonormalized), P the stored continued reference R2 (likewise); the dense outer root; "
          "the rates per lattice time unit, separate columns, never summed. The excess number beyond any cut is the dust's (definition (vii); the checks printed).", ""]
    S["E"] = []
    for SLx in SL:
        L += ["### E, %s (%s)" % (SLx["name"], SLx["record"]), "",
              "| T/r_s | kappa | dtype | R_t | dR_t/dT | excess number beyond R_t = the dust's (direct; complement minus direct) | beyond 2GE | beyond 16 r_s | excess Killing energy beyond R_t: D; dust; pairs | (1) fixed-cut number current: D; dust | (2) moving-cut reassignment rate: D; step form | (3) reference term: number; Killing energy at fixed h | (4) Killing-energy flux: D; dust | (5) Gauss charge loss -dE/dT | checks: particle-hole; sea and P site densities minus 1; the rebuilt reference against R2 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for row in SLx["slices"]:
            for ch in row.get("channels") or []:
                d = ch["detector"]; e_ = ch["energy_beyond"]; cu = ch["currents"]; ra = ch["reassignment"]; rt = ch.get("reference_term") or {}; dd = ch["density_deviation"]
                L.append("| %.3f | %d | %s | %.4f | %s | %.6e; %+.1e | %.6e | %.6e | %+.4e; %+.4e; %+.4e | %+.4e; %+.4e | %s; %s | %s; %s | %+.4e; %+.4e | %s | %.1e; %.1e, %.1e; %s |" % (
                    row["T_over_rs"], ch["kappa"], row["dtype"].replace("complex", "c"), row["R_outer"], fmt(row["dRt_dt"], "%+.4e"), d["beyond_Rt"]["D_direct"], d["beyond_Rt"]["D_complement"] - d["beyond_Rt"]["D_direct"],
                    d["beyond_2gE"]["D_complement"], d["beyond_16rs"]["dust"], e_["beyond_Rt"]["D"], e_["beyond_Rt"]["dust"], e_["beyond_Rt"]["sea_minus_P"], cu["D"]["number"], cu["dust"]["number"],
                    fmt(ra["D"]["rate"], "%+.4e"), fmt(ra["D"]["step_form"], "%+.4e"), fmt(rt.get("rate"), "%+.1e"), fmt(rt.get("energy_rate_fixed_h"), "%+.4e"), cu["D"]["killing_energy"], cu["dust"]["killing_energy"],
                    fmt(row["gauss_charge_loss_rate"], "%+.4e"), dd["particle_hole_check_max"], dd["sea_minus_1_max"], dd["P_minus_1_max"], fmt(rt.get("rebuild_vs_stored_residual_F"), "%.1e")))
                S["E"].append({"name": SLx["name"], "T_over_rs": row["T_over_rs"], "kappa": ch["kappa"], "dtype": row["dtype"], "R_t": row["R_outer"], "dRt_dt": row["dRt_dt"],
                               "detector": d, "energy_beyond": e_, "currents": cu, "reassignment": ra, "reference_term": {k_: v_ for k_, v_ in rt.items() if k_ != "rebuild_info"},
                               "gauss_charge_loss_rate": row["gauss_charge_loss_rate"], "density_deviation": dd, "signed_dust": ch["signed_dust"], "unsigned_dust_interior_A": ch["unsigned_dust_interior_A"]})
        L += ["", "The dust's signed projection at the slices (summed over its columns, occupation-weighted) beside part A's unsigned share of its interior:", "",
              "| T/r_s | kappa | inside R_t: + out; + in; - branch; + on pi/4 < q < pi/2; total | beyond R_t: + out; + in; total | part A's unsigned share pi/4 <= abs(ak) < pi/2 of the interior (quarters [0, pi/4), [pi/4, pi/2), [pi/2, 3pi/4], (3pi/4, pi]) |", "|---|---|---|---|---|"]
        for row in SLx["slices"]:
            for ch in row.get("channels") or []:
                ins = (ch["signed_dust"] or {}).get("inside_Rt") or {}; bey = (ch["signed_dust"] or {}).get("beyond_Rt") or {}
                q4 = (ch.get("unsigned_dust_interior_A") or {}).get("quarters")
                sm = lambda d_, k_: fmt(sum(d_[k_]) if d_.get(k_) else None, "%.4f")
                L.append("| %.3f | %d | %s; %s; %s; %s; %s | %s; %s; %s | %s |" % (row["T_over_rs"], ch["kappa"], sm(ins, "plus_out"), sm(ins, "plus_in"), sm(ins, "minus"), sm(ins, "plus_pi4_pi2"), sm(ins, "total"),
                                                                      sm(bey, "plus_out"), sm(bey, "plus_in"), sm(bey, "total"), ", ".join("%.4f" % x for x in q4) if q4 else "-"))
        L.append("")
    # ---------------------------------------------------------------- C
    ch = {c["kappa"]: c for c in C["channels"]}
    L += ["## C. The delivered part 3 relabelled (version 1's C, unchanged; the r_s/a 32 record at T/r_s %.3f)" % C["T0_over_rs"], "",
          "- The plateau shells n_L = R %.0f, n_R = R %.0f; the ramp shells %s; the dense outer root %.4f, the applied %.4f; inside the outer root: dense %s (margin %.4f sites), applied %s (margin %.4f sites)." % (
              C["window"]["n_L"], C["window"]["n_R"], ", ".join("%.0f" % x for x in C["window"]["ramp_shells"]), C["inside"]["dense"]["R_outer"], C["inside"]["applied"]["R_outer"],
              C["inside"]["dense"]["bands_inside_outer_root"], C["inside"]["dense"]["margin_to_outer"], C["inside"]["applied"]["bands_inside_outer_root"], C["inside"]["applied"]["margin_to_outer"]),
          "- Per channel: the bound beyond R_t(t) and its ratio to DeltaC(0)'s operator norm, the collar's covariance witness at most: %s." % "; ".join(
              "kappa %d: %s (ratios %s), collar at most %.1e" % (k_, ", ".join("T/r_s %s %.3e" % (t_, v_) for t_, v_ in c["bound_beyond_Rt"].items()), ", ".join("%.4e" % v_["over_DeltaC0"] for v_ in c["ratios"].values()), c["collar_D_det_max"])
              for k_, c in sorted(ch.items())),
          "- The numbers read \"reached beyond the outer root\" (the lead's turn 46 accepted the wording; the collar's witness at the rounding floor).", ""]
    # ---------------------------------------------------------------- D
    L += ["## D. Validation", ""]
    for fam, c in AS["strand_R_checks"].items():
        L.append("- A (%s): internal checks of the reconstructed projector, the largest SO1 residual %.1e (outer), %s (annulus), the largest S_X agreement %.1e (outer), %s (annulus); beside them the stored covariance's "
                 "largest purity defect %.2e; sigma_max <= 1/2 holds: %s." % (fam, c["outer_SO1_residual_max"], fmt(c["annulus_SO1_residual_max"], "%.1e"), c["outer_SO2_agreement_max"],
                                                                               fmt(c["annulus_SO2_agreement_max"], "%.1e"), c["stored_purity_max"], c["SO1_bound_half_holds"]))
    L.append("- B's dense oracle (version 1's, the machinery unchanged): %s; %d records; the largest differences %s; pass at 1e-10: %s." % (
        O["fixture"], O["records"], "; ".join("%s %.1e" % (k_, v_) for k_, v_ in O["max_abs_diff"].items()), O["pass_1e-10"]))
    regs = []
    for nm, V1, Fx, Zx, mode in [(names[0], V1F[0], BF[0], BFZ[0], "fixed-field"), (names[1], V1F[1], BF[1], BFZ[1], "fixed-field"), (names[0], V1H[0], BH[0], BHZ[0], "history"), (names[1], V1H[1], BH[1], BHZ[1], "history")]:
        for La in Fx["launches"]:
            q1 = [q for q in V1["launches"] if abs(q["T_launch_over_rs"] - La["T_launch_over_rs"]) < 1e-9 and q["kappa"] == La["kappa"]]
            if not q1:
                continue
            q1 = q1[0]; key = La["key"]; t2 = Zx["t_" + key]
            mx = 0.0; n_cmp = 0
            for s_ in q1["series"]:
                j = int(np.argmin(np.abs(t2 - s_["t"])))
                if abs(t2[j] - s_["t"]) > 1e-6:
                    continue
                for k_ in REG:
                    if k_ not in s_["w"]:
                        continue
                    a1 = np.array(s_["w"][k_], float); a2 = np.asarray(Zx["w_%s_%s" % (k_, key)][j][:12], float)
                    ok = np.isfinite(a1) & np.isfinite(a2)
                    if ok.any():
                        mx = max(mx, float(np.max(np.abs(a1[ok] - a2[ok])))); n_cmp += int(ok.sum())
            om = max(abs(c1["omega_bar"] - c2["omega_bar"]) for c1, c2 in zip(q1["columns"], La["columns"][:12]))
            regs.append({"run": nm, "mode": mode, "T_launch_over_rs": La["T_launch_over_rs"], "kappa": La["kappa"], "values_compared": n_cmp, "max_abs_diff": mx, "omega_bar_max_abs_diff": om,
                         "classifier_states_v1": q1.get("bound_states"), "classifier_states_v2": La["classifier_states"]})
    S["regression_v1"] = regs
    L.append("- Version 2's interior columns (0 to 11) against version 1's at version 1's clocks (the region weights; the mean energies; the classifier counts): %s." % "; ".join(
        "%s %s T/r_s %.1f kappa %d: %d values, largest difference %.1e, omega_bar %.1e, classifier states %s and %s" % (
            r["run"].split(" (")[0], r["mode"], r["T_launch_over_rs"], r["kappa"], r["values_compared"], r["max_abs_diff"], r["omega_bar_max_abs_diff"], r["classifier_states_v1"], r["classifier_states_v2"]) for r in regs))
    for nm, Fx in zip(names, BF):
        for La in Fx["launches"]:
            ctl = La.get("signed_launch_window_controls") or {}
            base = La["signed_launch"]["whole"]["plus_out"]
            dev = {k_: max(abs(x - y) for x, y in zip(v_["plus_out"], base)) for k_, v_ in ctl.items()}
            L.append("- B, %s, fixed-field T/r_s %.1f kappa %d: the Killing spectrum's largest change over the 300 r_s %.1e (a propagator check, target 1e-10); the norms' largest defect %.2e over %d steps; "
                     "the signed projection's + out at launch with tapers of %s sites against 32: the largest change over the columns %s." % (
                         nm.split(" (")[0], La["T_launch_over_rs"], La["kappa"], La["spectrum_conservation_max"], La["norm_defect_max"], La["steps"], " and ".join(sorted(ctl)), ", ".join("%s: %.3f" % (k_, v_) for k_, v_ in sorted(dev.items()))))
    for nm, Hx in zip(names, BH):
        for La in Hx["launches"]:
            cc = La.get("cubic_control")
            L.append("- B, %s, history T/r_s %.1f kappa %d (to %.3f): the norms' largest defect %.2e over %d records%s." % (
                nm.split(" (")[0], La["T_launch_over_rs"], La["kappa"], La["T_end_over_rs"], La["norm_defect_max"], La["steps"],
                ("; the piecewise-constant against cubic control on T/r_s %.3f to %.3f: the columns' largest difference %.3e, the readouts' changes %s" % (
                    cc["window_rs"][0], cc["window_rs"][1], cc["columns_diff_max"], "; ".join("%s %.3e" % (k_, v_) for k_, v_ in cc["region_diff"].items()))) if cc else ""))
    for P_ in PREP:
        L.append("- B.5's columns (%s): %s." % (os.path.basename(P_["record"]), "; ".join("%s kappa %d: stored purity defect %.1e, Loewdin %.1e, the occupied set's orthonormality after the propagation %.1e, sea-localized orthonormality %s and inside-X weight minus lambda %s" % (
            it["tag"], it["kappa"], it["stored_idempotency"], it["loewdin_orthonormality"], it["occupied_orthonormality_after_propagation"], fmt(it["sealoc_orthonormality"], "%.1e"), fmt(it["sealoc_inside_X_minus_lambda_max"], "%.1e"))
            for it in P_["items"] if it.get("R_outer") is not None)))
    # the dust: two routes (B.5's propagation against the stored states)
    tw = []
    for nm, D_, SLx in [(names[0], BD[0], [SL[0], SL[1]]), (names[1], BD[1], [SL[2]])]:
        for La in D_["launches"]:
            for sl in SLx:
                for row in sl["slices"]:
                    for chx in row.get("channels") or []:
                        if chx["kappa"] != La["kappa"]:
                            continue
                        q = [p_ for p_ in La["series"]["dust"] if abs(p_["t_over_rs"] - row["T_over_rs"]) < 1e-9 and p_.get("beyond_Rt_sum") is not None]
                        if q and row["T_over_rs"] >= La["T_launch_over_rs"] - 1e-9:
                            tw.append({"run": nm, "tag": La["tag"], "kappa": La["kappa"], "T_over_rs": row["T_over_rs"], "propagated": q[0]["beyond_Rt_sum"], "stored": chx["detector"]["beyond_Rt"]["dust"],
                                       "R_t_propagated": q[0]["R_t"], "R_t_stored": row["R_outer"]})
    S["dust_two_routes"] = tw
    if tw:
        L.append("- The dust's weight beyond R_t by two routes, B.5's propagation (B's machinery from its launch slice) against the stored state at the slice's clock (the engine's own evolution): %s." % "; ".join(
            "%s %s kappa %d T/r_s %.0f: %.5e against %.5e" % (t["run"].split(" (")[0], t["tag"], t["kappa"], t["T_over_rs"], t["propagated"], t["stored"]) for t in tw))
    phm = max((ch_["density_deviation"]["particle_hole_check_max"] for sl in SL for row in sl["slices"] for ch_ in (row.get("channels") or [])), default=None)
    dvm = max((max(ch_["density_deviation"]["sea_minus_1_max"], ch_["density_deviation"]["P_minus_1_max"]) for sl in SL for row in sl["slices"] for ch_ in (row.get("channels") or [])), default=None)
    rbm = max(((ch_.get("reference_term") or {}).get("rebuild_vs_stored_residual_F") or 0.0 for sl in SL for row in sl["slices"] for ch_ in (row.get("channels") or [])), default=None)
    rdm = max((abs(ch_["detector"]["beyond_Rt"]["routes_diff_D"]) for sl in SL for row in sl["slices"] for ch_ in (row.get("channels") or [])), default=None)
    L.append("- E's checks over every slice and channel: the particle-hole check ||S h_m^* S^dag + h_m||_max at most %s; the sea's and the reference's site densities differ from one by at most %s (the complex64 slices' "
             "rounding included); the reference rebuilt from the stored mask and applied fields against the stored R2, the residual at most %s; the direct and complement routes of the excess number differ by at most %s." % (
                 fmt(phm, "%.1e"), fmt(dvm, "%.1e"), fmt(rbm, "%.1e"), fmt(rdm, "%.1e")))
    S["E_checks"] = {"particle_hole_max": phm, "density_deviation_max": dvm, "rebuild_residual_max": rbm, "routes_diff_max": rdm}
    L += ["", "## Open points", "",
          "- The history runs and B.5 end at the last continuation record held when each started (printed per launch); later records extend them by the runs' resume (engine_single_object_B_v2.py --resume-json/--resume-npz), "
          "a new version of this file then.",
          "- The arrivals on the history are run-limited where the weight beyond 16 r_s still grows at the end; the delays on the history are a record (the comparator is ruled per fixed-field record).",
          "- The signed projection depends on the taper length (D prints the change at 16 and 64 sites); it is a kinematic split of the weight at a clock, not a forecast of what crosses later.",
          "- B.5's sums are the selected sets' readouts; the full detector density is E's, and by the particle-hole identity it is the dust's alone.",
          "- The Killing-energy shells of the exterior block are in the fixed-field runs' files every 25 r_s, not reduced here.",
          "- Devices: the fixed-field runs on pod 1's GPU (lane 2c); the histories, B.5, the columns and E on box GPU 0 (A6000); version 1's runs as version 1 states.", "",
          "## Inputs (md5)", ""] + ["- %s %s" % (v_, k_) for k_, v_ in S["inputs"].items()] + ["", "**Label (every output):** %s." % LABEL]
    text = "\n".join(L) + "\n"
    low = text.lower()
    hits = {w: low.count(w) for w in FORBIDDEN if w in low}
    wn = low.count("window") - low.count("windowed share") - low.count("window_rs")
    if hits or wn > 0:
        sys.exit("words found in the composed text: %s; 'window' outside 'windowed share': %d" % (hits, wn))
    open(a.md, "x").write(text)
    S["words_check"] = {"forbidden_found": hits, "window_outside_windowed_share": wn}
    json.dump(S, open(a.out_json, "x"), indent=1, default=float)
    print("composed %s (%d lines) and %s; %d B columns; %d B.5 launches; %d E rows" % (a.md, len(L), a.out_json, len(blaunch), len(S["B5"]), len(S["E"])), flush=True)


if __name__ == "__main__":
    main()
