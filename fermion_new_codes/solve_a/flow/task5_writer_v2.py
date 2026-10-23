#!/usr/bin/env python
"""Task 5 result writer, revision 1, writer version 2 (version 1's note (3) quoted one maximum over all masses for the late-time check and
note (4) printed "negative at ... none"; version 2 gives the late-time check per mass and words the sign per carrier) (builder analyst): solve/results/RESULT_task5_black_object.md from task5_report_v1.py's tables and the
records, with the header and the builder's notes (labelled; every number read from the JSONs).  Args: --report-md --report-json --records
(folder) --stamp --out."""
import argparse
import json
import math
import os


def f(v, d=4):
    if v is None:
        return "-"
    v = float(v)
    return "%.*g" % (d, v) if (v == 0 or 1e-3 <= abs(v) < 1e6) else "%.3e" % v


def demote(md):
    return ["#" + x if x.startswith("##") else ("## " + x[2:] if x.startswith("# ") else x) for x in open(md).read().split("\n")]


def main():
    ap = argparse.ArgumentParser()
    for k in ("--report-md", "--report-json", "--records", "--stamp", "--out"):
        ap.add_argument(k, required=True)
    a = ap.parse_args()
    R = lambda n: json.load(open(os.path.join(a.records, n)))
    rep = json.load(open(a.report_json))
    e0 = R("v5_e0_checks.json")
    e1 = R("v6_e1_m0.json")
    s32 = [s for s in e1["runs"][0]["series"] if abs(s["k0"] - math.pi / 32) < 1e-12 and s["sigma_x"] == 32.0][0]
    e1rows = rep["e1"]
    late = {m: max(r["max_late_minus_P"] for r in e1rows if r["max_late_minus_P"] is not None and r["mass"] == m) for m in sorted({r["mass"] for r in e1rows})}
    dyn = max(r["max_Pdyn_minus_P"] for r in e1rows if r["max_Pdyn_minus_P"] is not None)
    e4 = R("v6_e4_N4096.json")
    near = []
    for run in e4["runs"]:
        if run["mass"] != 0 or run["kappa_j"] != -1.0:
            continue
        for s in run["series"]:
            if s["control"]:
                continue
            rows = [r for r in s["rows"] if r["Re_over_rs"] <= 1.02]
            neg = [r["Re_over_rs"] for r in s["rows"] if r.get("one_plus_z") is not None and r["one_plus_z"] < 0]
            zp = [r["one_plus_z_plus"] for r in rows if r.get("one_plus_z_plus")]
            ss = s.get("symbol_series") or {}
            sr = [q for q in ss.get("rows", []) if q.get("local_slope_ln1pz") is not None]
            last = sr[-1] if sr else None
            first_none = next((q for q in ss.get("rows", []) if q.get("one_plus_z_point") and q.get("t_arr") is None), None)
            near.append((s["k0"], s["sigma_x"], neg, min(zp) if zp else None, max(zp) if zp else None, s["saturation"], last, first_none))
    e2 = [R("v5_e2_shard0.json"), R("v5_e2_shard1.json")]
    ch = [c for d in e2 for c in d["channels"]]
    b = {c["kappa_j"]: c for c in ch if c["mass"] == 0 and c["omega_rs"] == 4 and c["variant"] == "base" and "T_first_passage" in c}
    h = {c["kappa_j"]: c for c in ch if c["mass"] == 0 and c["omega_rs"] == 4 and c["variant"] == "sigma_omega_halved" and "T_first_passage" in c}
    ratio = [h[k]["T_first_passage"] / b[k]["T_first_passage"] for k in (-1.0, -2.0, -3.0, -4.0) if k in h and k in b and b[k]["T_first_passage"] > 0]
    e3 = R("v5_e3.json")
    delays = [r["t_exit_half"] - r["t_entry_half"] for r in e3["runs"] if r["t_exit_half"] and r["t_entry_half"]]
    sx = {o: next((c["sigma_x"] for c in ch if c["mass"] == 0 and c["omega_rs"] == o and c["variant"] == "base" and "sigma_x" in c), None)
          for o in (1.0, 4.0, 16.0)}
    vg = {o: next((c["vg0"] for c in ch if c["mass"] == 0 and c["omega_rs"] == o and c["variant"] == "base" and "vg0" in c), None) for o in (1.0, 4.0, 16.0)}
    steps = {(x["mass"], x["omega_rs"]): x["kappa_half"] for x in rep["e2"]}
    rs2, bc = e2[0]["rs"], e2[0]["b_c"]
    L = ["# Task 5 result, revision 1 (%s): the black object of the model on the flow chart, E0 to E4 (E5 not run)" % a.stamp, "",
         "Builder: analyst. Spec: solve/specs/SPEC_task5_black_object.md (lead, turn 11 T56). Every number is labelled \"prescribed flow (simpler "
         "version); PG radial; v declared\" (the profile v = -sqrt(r_s/R) continued by the ball v = -sqrt(r_s/R_m^3) R inside R_m = r_s/4; lapse 1, "
         "ruler 1); the tables below are emitted by task5_report_v1.py from the records in solve/results/task5/records/ (flow_chart_v5.py: E0, E2, "
         "E3 and the version-5 controls; flow_chart_v6.py: E1 and E4); the builder's notes are the builder's readings, labelled, their numbers read "
         "from the records by this writer (task5_writer_v1.py).", "",
         "## Builder's notes (labelled; for the lead)",
         "- (1) E0 fixed the cut current's sign: dN_in/dt = %s psi_in^H h_{in,out} psi_out (the derivation; the spec's sec. 7 draft wrote -2 Im), "
         "checked to %s by 64-point Gauss-Legendre integrals and the Richardson difference (with the draft's sign the error is %s); every flux in E1, "
         "E2 and E4 uses the checked sign." % (e0["i_cut_current"]["sign_fixed"], f(e0["i_cut_current"]["max_err_plus"], 2), f(e0["i_cut_current"]["max_err_minus"], 3)),
         "- (2) Version 6 (after version 5's first records; E0, E2, E3 unchanged): outgoing_root's bracket search returned None once the long-wave "
         "root fell below its step 0.02 (the symbol's arrival time was missing for every escaping frequency below about 0.02); E1's launch margin at "
         "the centre is 3 sigma_x (version 5's 4 sigma_x dropped every pi/32 launch below 1.05 r_s; with version 6 the pi/32 crossing is R_1/2 = %s r_s "
         "against R_esc = %s r_s); E4 adds the symbol's point-emission series." % (f(s32["R_half_measured"][0] / e1["rs"], 5) if s32["R_half_measured"] else "-",
                                                                                 f(s32["R_esc_over_rs"], 5)),
         "- (3) E1's dynamical check: the spec's P_dyn (the norm beyond R_ref when the centre's return first crosses the gate outward) differs from "
         "P_+ by up to %s, because the escaping part launched near the escape surface leaves slowly while the return through the declared centre "
         "crosses the gate at r_s/2 first (the gate fires before the escapers reach R_ref = 2 r_s); the norm beyond R_ref at the run's end (t = 10 r_s) "
         "lies below P_+ by at most %s. Builder's reading: P_+ is the escaping weight (whatever has omega > m leaves, the returned part keeping its "
         "Killing frequency, E3); the end-of-run norm checks it for the massless escapers and not for the slow massive ones, which have not passed "
         "R_ref by t = 10 r_s; a longer run (or the flux at R_ref integrated to convergence) is the check there." % (
             f(dyn, 3), ", ".join("%s at m = %s" % (f(v, 3), f(m)) for m, v in late.items()))]
    for k0, sxx, neg, zmin, zmax, sat, last, first_none in near:
        L.append("- (4) E4, carrier k0 a = %s (sigma_x %s): the whole-packet 1 + z = E_loc/omega_bar %s; the escaping part's 1 + z_+ stays within %s to %s for R_e <= 1.02 r_s (the lattice "
                 "saturation 1/(1 - cos k0) form is %s): the packet's escaping part is emitted from its outer half. The symbol's point-emission series: "
                 "%s%s." % (f(k0), f(sxx), ("is negative at R_e/r_s in %s (the packet straddles the surface; its trapped part has omega < 0)" % neg) if neg
                            else "stays positive at every R_e (it grows beyond the point law near the surface, the packet average over both sides)", f(zmin), f(zmax), f(sat), (
                     "the local slope d ln(1+z)/d(kappa t_arr) reaches %s at R_e - r_s = %s r_s (1 + z = %s)" % (
                         f(last["local_slope_ln1pz"], 3), f(last["Re_minus_rs_over_rs"], 2), f(last["one_plus_z_point"]))) if last else "no slope",
                     ("; from R_e - r_s = %s r_s (1 + z = %s) the WKB root disappears on the way out (the angular barrier omega < |kappa_j|/R of j = 1/2)" % (
                         f(first_none["Re_minus_rs_over_rs"], 2), f(first_none["one_plus_z_point"]))) if first_none else ""))
    L += ["- (5) E4's reading (builder, labelled): at r_s = 128 and j = 1/2 the e^{kappa t} regime of the point emission is bounded on the lattice "
          "symbol from both sides (the angular barrier cuts the long carriers at 1 + z of order E(k0) r_s, the short carrier saturates at "
          "1/(1 - cos k0)), so the spec's fit window (1 + z from 3 to a third of the saturation) measures the approach, not the law: the slopes of "
          "the table are below 1 for the packets and for the symbol alike. A clean test of the law on the symbol needs E(k0) r_s >> 1 + z >> 1 and "
          "1 + z << 1/(1 - cos k0) together, i.e. r_s of order 1e3 at k0 a = pi/32 (the symbol alone is seconds; packets resolving the emission point "
          "need sigma_x << R_e - r_s, which long carriers cannot meet near the surface).",
          "- (6) E2's first-passage crossing as specified measures the packet, not the channel: the packet enters r_s and the part that crossed returns "
          "through the declared centre on the outgoing lattice branch (E3: exit minus entry %s time units at r_s = 32) while the packet's own passage "
          "takes about 6 sigma_x/v_g (sigma_x %s at omega r_s = 1, %s at 4, %s at 16; v_g %s, %s, %s), so the head's return fires the gate at r_s/2 "
          "while the tail is still entering, and the integral up to the gate undercounts. The decisive control: sigma_omega halved (a packet twice as "
          "long) changes the massless T at omega r_s = 4 and |kappa| = 1 to 4 by the factors %s. The largest norm inside r_s over the run is a lower "
          "bound for the crossing (its column beside the spec's). The step of T against |kappa| sits at |kappa| = %s (omega r_s = 4; omega b_c = %s), "
          "%s (8; %s) and %s (16; %s): at omega a = 0.125 to 0.5 the lattice dispersion is not long-wave, and the geometric limit 27/4 is not "
          "approached (builder reading). Remedies for the lead to choose: the inward current of the ingoing branch alone at r_s (the state near the "
          "cut split by lattice momentum: the ingoing k near -omega/2 against the returning k near (2 omega)^(1/3)), a stationary scattering "
          "computation with the purely ingoing condition inside r_s, or shorter packets with the frequency average stated; and r_s larger at fixed "
          "omega r_s for the high-frequency end." % (
              ", ".join(f(x, 3) for x in delays), f(sx[1.0]), f(sx[4.0]), f(sx[16.0]), f(vg[1.0], 3), f(vg[4.0], 3), f(vg[16.0], 3), ", ".join(f(x, 3) for x in ratio),
              f((steps.get((0.0, 4.0)) or [None])[0]), f(4 / rs2 * bc), f((steps.get((0.0, 8.0)) or [None])[0]), f(8 / rs2 * bc),
              f((steps.get((0.0, 16.0)) or [None])[0]), f(16 / rs2 * bc)),
          "- (7) E3 rebuilt D3's packets (version 4's construction) and tracked the annuli; the returned packet's Killing mean and spread match the "
          "launched ones (the table). E5 (the chart map on the page code) is not run; the spec allows the derivation alone for the section.",
          "- (8) Records carry the script name, never a hash.", ""]
    L += demote(a.report_md)
    open(a.out, "w").write("\n".join(L) + "\n")
    print("wrote %s (%d lines)" % (a.out, len(L)))


if __name__ == "__main__":
    main()
