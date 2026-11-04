#!/usr/bin/env python
"""attrib_compose_v1.py (composition builder, the solve lane): writes W5_W4_formation_attribution_v1.md from attrib_readout_v1.json and
W3_completions_v1.md from w3_readout_v1.json (no run file is opened; every number is the producers'). Usage:
  python attrib_compose_v1.py --ladders-code DIR --attrib JSON --w3 JSON --sources JSON --out-dir DIR"""
import argparse
import json
import os
import sys

PROD_QUOTED = {"P64_s2": (12.000, 0.65), "P64_s4": (12.125, 0.66), "P64_s8": (12.750, 0.76)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladders-code", required=True)
    ap.add_argument("--attrib", required=True)
    ap.add_argument("--w3", required=True)
    ap.add_argument("--sources", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.ladders_code)
    from ladders_compose_v1 import f, e, table
    D = json.load(open(a.attrib)); W = json.load(open(a.w3)); S = json.load(open(a.sources))
    H = D["histories"]

    def fo(nm):
        return H[nm]["formation"]

    o = []
    o.append("# W5_W4_formation_attribution_v1: the formation offset's attribution on the (0.2, 64) member: the heavy sea's angular truncation "
             "(W5), the pair sources (W4 (b)), the step (tau 4 and tau 2 against tau 8; the conserving step), beside the dust comparator (W4 (a)) "
             "and the records' production numbers")
    o.append("")
    o.append("Composed by attrib_readout_v1.py (the producer, box CPU, stored files only) and attrib_compose_v1.py (this text, from the "
             "producer's json). The builder composes; no sentence here reads the physics. Formation is the first record holding a trapping "
             "zero (the ladder reader's rule, MASS_ladder_v1's definitions block), R_t the outer dense root, R_t/(2 g E) the record's own ratio "
             "at its charge E; the dust comparator's root is the zero of N - |v| on its dense fields, its fold clock by the integrator's event, "
             "its radius over its own 2 g E0 (the comparator conserves E to 1e-9). The engine records sit on a grid of tau/r_s (0.125 at "
             "tau 8, 0.0625 at tau 4, 0.03125 at tau 2): a formation clock is read to that grid, and two clocks on the same grid that agree "
             "are equal only to its spacing.")
    o.append("")
    o.append("## 1 The offset: the production histories against the dust comparator (W4 (a))")
    o.append("")
    rows = []
    for x in D["offsets"]:
        q = PROD_QUOTED.get(x["history"])
        rows.append([x["history"], "%s / %s" % (f(q[0], "%.3f"), f(q[1], "%.2f")) if q else "-", f(x["T_f"]), f(x["R_t_over_2gE"]),
                     x["comparator"], f(x["fold_T"]), f(x["fold_R_over_2gE0"]), f(x["first_record_T"]), f(x["first_record_outer_over_2gE0"]),
                     "%+.4f" % x["offset_T_vs_fold"], "%+.4f" % x["offset_T_vs_first_record"], "%+.4f" % x["offset_ratio_vs_fold"],
                     "%+.4f" % x["offset_ratio_vs_first_record"]])
    o.append(table(["history", "quoted T_f / ratio", "T_f (records)", "R_t/(2gE)", "dust comparator run", "fold T/r_s", "fold R/(2gE0)",
                    "first comparator record with a root T/r_s", "its outer root/(2gE0)", "offset T_f - fold", "offset T_f - first record",
                    "offset ratio - fold", "offset ratio - first record"], rows))
    o.append("")
    o.append("The quoted production numbers (12.000/0.65, 12.125/0.66, 12.750/0.76) are reproduced by the stored records to their printed "
             "digits. The comparator's variants (its own step, dispersion, kernel, release and lapse) are in section 4.")
    o.append("")
    o.append("## 2 The attribution table: each switch against its reference at formation (switch minus reference)")
    o.append("")
    groups = [("angular truncation (W5)", ["W5_k14_tau8", "W5_k20_tau8", "W5_k14_tau4", "W5_k20_tau4"]),
              ("W5's reference against the production (the same arguments, the W5 launcher)", ["W5_k10_tau8"]),
              ("pair sources off (W4 (b))", ["W4b_pairsoff"]),
              ("step tau 4 and tau 2 against tau 8", ["W5_k10_tau4", "W5_k14_tau4", "W5_k20_tau4", "P64_s4_tau4", "P64_s4_tau2"]),
              ("the conserving step (W2's dg history)", ["W2_dg"]),
              ("averaging scale (the stored spread)", ["P64_s2", "P64_s8"])]
    rows = []
    for gname, sws in groups:
        for sw in sws:
            for x in D["diffs"]:
                if x["switch"] != sw:
                    continue
                if gname.startswith("angular") and not x["reference"].startswith("W5_k10"):
                    continue
                if gname.startswith("step") and sw.startswith("W5") and not x["reference"].endswith("tau8"):
                    continue
                fa, fb = fo(sw), fo(x["reference"])
                rows.append([gname, sw, x["reference"], "%s / %s" % (f(fa["T"]), f(fb["T"])), "%+.4f" % x["dT_f"], "%+.3f" % x["dR_t"],
                             "%+.4f" % x["dR_t_over_2gE"], H[sw]["meta"]["tau"] / 64.0, "yes" if x["both_certified"] else "no"])
    o.append(table(["switch", "history", "reference", "T_f switch / reference", "dT_f (r_s)", "dR_t (a)", "d(R_t/(2gE))", "record grid of the switch (r_s)",
                    "both certified"], rows))
    o.append("")
    off4 = [x for x in D["offsets"] if x["history"] == "P64_s4"]
    if off4:
        o.append("Against the sigma/a 4 offset T_f - fold = %+.4f r_s (T_f - first comparator record %+.4f r_s; ratio %+.4f against the fold): "
                 "the switches above move the engine's formation clock by the amounts in the dT_f column, each read on its record grid." % (
                     off4[0]["offset_T_vs_fold"], off4[0]["offset_T_vs_first_record"], off4[0]["offset_ratio_vs_fold"]))
        o.append("")
    o.append("## 3 W5: the heavy sea's angular truncation (kappa_sea 10, 14, 20 at tau 8 and tau 4, from the common preparation, to T/r_s 14)")
    o.append("")
    rows = []
    for nm in [x for x in D["order"] if x.startswith("W5")]:
        h = H[nm]; ff = h["formation"]; c = h["convergence"]; s = D["w5_slice"].get(nm, {})
        rows.append([nm, h["meta"]["args"]["kappa_sea"], h["meta"]["tau"], f(ff["T"]), f(ff["R_t"], "%.3f"), f(ff["R_t_over_2gE"]), f(ff["E_over_E0"], "%.5f"),
                     f(ff["gate_sigma_over_Rt"], "%.3f"), e(ff["residual"]), ff["label"], f(c["last"]), f(c["last_converged"]),
                     ", ".join("%.4f" % x["T"] for x in c["excluded"]) or "none", f(h["counts"]["E0"], "%.9f")])
    o.append(table(["rung", "kappa_sea", "tau/a", "formation T/r_s", "R_t/a", "R_t/(2gE)", "E/E0", "gate sigma/R_t", "residual", "certification",
                    "last record", "last converged", "excluded records", "E0 (head)"], rows))
    o.append("")
    for nm in [x for x in D["order"] if x.startswith("W5")]:
        c = H[nm]["convergence"]
        if c["last_converged"] is None or c["last_converged"] < 14.0 - 1e-9:
            o.append("- %s: the record at 14 does not count (last converged record %s; %s); its slice readout at 14 below is that "
                     "record's, labelled." % (nm, f(c["last_converged"]), "first above 1e-3 at %s" % f(c["first_above_1e-3"]) if c["first_above_1e-3"] is not None else "capped records %s" % ", ".join("%.4f" % x["T"] for x in c["excluded"])))
    o.append("")
    o.append("### 3a The pairs at T/r_s 14 (per channel kappa: the pair count w (n_sea - ||R2^dag S||^2) and the bare energy w (Tr h P_sea - Tr h P_R2); "
             "the slice readout of w5_readout_v1)")
    o.append("")
    rows = []
    for nm in sorted(D["w5_slice"]):
        s = D["w5_slice"][nm]
        if not s.get("n_pairs_per_channel"):
            rows.append([nm, s.get("slice_note") or "no slice readout", "-", "-", "-", "-"])
            continue
        ks = sorted(s["n_pairs_per_channel"], key=int)
        rows.append([nm, " ".join("%s:%s" % (k, e(s["n_pairs_per_channel"][k])) for k in ks), " ".join("%s:%s" % (k, e(s["E_pairs_bare_per_channel"][k])) for k in ks),
                     e(s.get("sum_n_pairs")), e(s.get("sum_E_pairs_bare")), e(s.get("sum_E_pairs_killing"))])
    o.append(table(["rung", "pair count per channel", "bare pair energy per channel", "sum pair count", "sum bare pair energy", "sum Killing pair energy"], rows))
    o.append("")
    o.append("### 3b The inner-shell share above pi/4 and the 16 kappa E_kappa tail at T/r_s 14")
    o.append("")
    rows = []
    for nm in sorted(D["w5_slice"]):
        s = D["w5_slice"][nm]
        if not s.get("w_E_maj_pairs_per_channel"):
            continue
        ks = sorted(s["w_E_maj_pairs_per_channel"], key=int)
        rows.append([nm, f(s.get("R_50"), "%.1f"), f(s.get("inner_shell_particles"), "%.2f"), f(s.get("inner_shell_share_above_quarter")),
                     f(s.get("all_dust_share_above_quarter")), e(s.get("sum_w_E_maj_pairs")), e(s.get("sum_w_E_maj_heavy")),
                     f(s.get("top_channel_share_w_E_maj_pairs")), f(s.get("above_matched_share_w_E_maj_pairs")),
                     " ".join("%s:%s" % (k, e(s["w_E_maj_pairs_per_channel"][k])) for k in ks), e(s.get("eig_vs_stored_max")), s.get("pockets")])
    o.append(table(["rung", "R_50 (a)", "inner-shell dust count", "inner-shell share above pi/4", "whole dust share above pi/4", "sum 16 kappa E_kappa (pairs)",
                    "sum 16 kappa E_kappa (whole heavy)", "top channel's share of the pairs' tail", "share of channels above 10", "16 kappa E_kappa (pairs) per channel",
                    "eigenvalues against the slice's stored (max)", "pockets (occupied positive)"], rows))
    o.append("")
    o.append("Definitions (w5_readout_v1's): R_50 the radius enclosing half of the fields row's dust energy at 14; the share the number-weighted "
             "fraction of the inner dust on lattice momenta with |sin a k| > sin(pi/4); E_kappa^pairs = Tr[|h_m| (P_sea - P_R2)^2], "
             "E_kappa^heavy = Tr[|h_m| (C - P_-(h_m))^2] on the record's clocked operator h_m, each weighted by 16 kappa.")
    o.append("")
    o.append("## 4 W4 (a): the dust comparator's runs (dust_comparator_v1c on the production's stored T = 0 profile)")
    o.append("")
    rows = []
    for lab in D["w4a_order"]:
        w = D["w4a"].get(lab, {})
        if "absent" in w:
            rows.append([lab, "absent"] + ["-"] * 8)
            continue
        rows.append([lab, f(w.get("fold_T")), f(w.get("fold_R"), "%.3f"), f(w.get("fold_R_over_2gE0")), f(w.get("first_record_T")),
                     f(w.get("first_record_outer"), "%.3f"), f(w.get("first_record_outer_over_2gE0")), f(w.get("caustic_T")),
                     e(w.get("E_drift_net_max")), (w.get("end_label") or "")[:110]])
    o.append(table(["run (sigma, kernel, dispersion, release, lapse, step)", "fold T/r_s", "fold R (a)", "fold R/(2gE0)", "first record with a root T/r_s",
                    "its outer root (a)", "/(2gE0)", "central caustic T/r_s", "E drift net (max)", "end"], rows))
    o.append("")
    o.append("A second copy of the sigma/a 4 runs (busi's fundmarks w4a_pairsoff, default_s4) prints the same fold and first-record numbers to every "
             "digit of its log; it is listed in SOURCES and not repeated.")
    o.append("")
    o.append("## 5 W4 (b) and the conserving step")
    o.append("")
    rows = []
    for nm in ("P64_s4", "W4b_pairsoff", "W2_dg", "P64_s4_tau4", "P64_s4_tau2"):
        h = H[nm]; ff = h["formation"]; c = h["convergence"]
        eng = "; ".join("%s (%s to %s)" % (u["value"], f(u["first"], "%.3f"), f(u["last"], "%.3f")) for u in h["meta"]["engine_entries"][:3])
        ep = (h["episodes"] or [{}])[0]
        rows.append([nm, h["meta"]["label"], h["meta"]["script"], eng, f(ff["T"]), f(ff["R_t"], "%.3f"), f(ff["R_t_over_2gE"]), f(ff["E_over_E0"], "%.5f"),
                     e(ff["residual"]), f(ep.get("closes_at")) if ep.get("closes_at") is not None else "open", f(c["last"]), f(c["last_converged"]),
                     ", ".join("%.4f" % x["T"] for x in c["excluded"][:6]) or "none"])
    o.append(table(["history", "label", "script field", "engine entries", "formation T/r_s", "R_t/a", "R_t/(2gE)", "E/E0", "residual", "first episode closes",
                    "last record", "last converged", "excluded records"], rows))
    o.append("")
    for line in S.get("attrib_notes", []):
        o.append("- " + line)
    o.append("")
    o.append("## 6 SOURCES")
    o.append("")
    rows = [[nm, H[nm]["meta"]["read_from"], H[nm]["meta"]["bytes"], H[nm]["meta"]["md5"]] for nm in D["order"]]
    o.append(table(["history", "run json", "bytes", "md5"], rows))
    o.append("")
    for line in S.get("attrib_sources", []):
        o.append("- " + line)
    o.append("")
    open(os.path.join(a.out_dir, "W5_W4_formation_attribution_v1.md"), "w").write("\n".join(o))

    # ------------------------------------------------------------------ W3
    o = []
    WH = W["histories"]
    o.append("# W3_completions_v1: the completion control of the (0.2, 64) sigma/a 4 tau 8 member: the record's own history and the busi "
             "lane's completions from T/r_s 11.125 (frozen-sources and paired-avg to 40; point-core ended at 12.5), at matched clocks")
    o.append("")
    o.append("Composed by w3_readout_v1.py (the producer, box CPU, stored files only) and attrib_compose_v1.py. The builder composes; no "
             "sentence here reads the physics. The spread at a clock is max - min over the histories holding a counting record there "
             "(the count of histories beside it). Rows to 11.125 are the record's own records carried into each completion (identical).")
    o.append("")
    o.append("## 1 The histories")
    o.append("")
    rows = []
    for nm in W["order"]:
        h = WH[nm]; c = h["convergence"]
        eps = "; ".join("%s to %s (%s)" % (f(x["first"]), f(x["last"]), ("closes %s" % f(x["closes_at"])) if x["closes_at"] is not None else "open")
                        for x in h["episodes"][:4])
        rows.append([nm, h["label"], h["records"], f(c["last"]), f(c["last_converged"]), ", ".join("%.4f" % x["T"] for x in c["excluded"][:6]) or "none",
                     eps or "none", json.dumps(h["completion_args"]) if h["completion_args"] else "-"])
    o.append(table(["history", "label", "records", "last record", "last converged", "excluded records", "trapped episodes", "completion arguments (json)"], rows))
    o.append("")
    o.append("## 2 Matched clocks: E/E0, the outer root R_t (a), the far dust's bare shell energy beyond 16 and 24 r_s, min(N - |v|)")
    o.append("")
    rows = []
    for r in W["matched"]:
        per = r["per"]; sp = r["spread"]
        cells = [f(r["T"])]
        for nm in W["order"]:
            p = per.get(nm)
            cells.append("-" if p is None else "%s; %s; %s; %s; %s%s" % (f(p["EE0"], "%.5f"), f(p["R_t"], "%.3f") if p["R_t"] is not None else "none",
                                                                        e(p["dust_16"]), e(p["dust_24"]), f(p["minNv"]), "" if p["counts"] else " (n.c.)"))
        cells += ["%s (%d)" % (e(sp["EE0"]["spread"]), sp["EE0"]["n"]), "%s (%d)" % (f(sp["R_t"]["spread"], "%.3f"), sp["R_t"]["n"]),
                  "%s (%d)" % (e(sp["dust_16"]["spread"]), sp["dust_16"]["n"]), "%s (%d)" % (e(sp["dust_24"]["spread"]), sp["dust_24"]["n"])]
        rows.append(cells)
    o.append(table(["T/r_s"] + ["%s: E/E0; R_t; dust>16; dust>24; min(N-|v|)" % nm for nm in W["order"]] +
                   ["spread E/E0 (n)", "spread R_t (n holding a root)", "spread dust>16 (n)", "spread dust>24 (n)"], rows))
    o.append("")
    o.append("## 3 Not composed and labels")
    o.append("")
    for line in S.get("w3_notes", []):
        o.append("- " + line)
    o.append("")
    o.append("## 4 SOURCES")
    o.append("")
    rows = [[nm, WH[nm]["json"], WH[nm]["json_md5"], WH[nm]["fields"], WH[nm]["fields_md5"]] for nm in W["order"]]
    o.append(table(["history", "records json", "md5", "fields npz", "md5"], rows))
    o.append("")
    for line in S.get("w3_sources", []):
        o.append("- " + line)
    o.append("")
    open(os.path.join(a.out_dir, "W3_completions_v1.md"), "w").write("\n".join(o))
    print("wrote", a.out_dir)


if __name__ == "__main__":
    main()
