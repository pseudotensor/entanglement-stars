#!/usr/bin/env python
"""cv_compare.py (busi lane, correctness reading of CURVATURE_readout_v1): my recomputed numbers (cv_fields_*.json/npz, cv_rays_*.json)
beside the delivery's own outputs (curvature_v1_*.json, read for comparison only), group by group, with relative differences; plus the
cross-member checks the delivery asserts (B2 vs c19 node fields through 40, P64_s4 vs c19) from the stored nodes.
Usage: python cv_compare.py --mine DIR --deliv DIR --out FILE.md"""
import argparse
import json
import os

import numpy as np

NAMES = ["B2", "P64_s4_c19", "P64_s4", "P64_s2", "P64_s8", "P64_s4_tau4", "P64_s4_tau2"]
L = []


def P(s=""):
    L.append(s)


def rel(a, b):
    if a is None or b is None:
        return "-"
    if b == 0:
        return "%.2g" % abs(a - b)
    return "%.1e" % (abs(a - b) / abs(b))


def f3(x, fmt="%.4g"):
    return "-" if x is None else (fmt % x)


def pk(d, fmt="%.4g"):
    return "-" if d is None else ("%s (%.4g)" % (fmt % d["value"], d["T_over_rs"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mine", required=True); ap.add_argument("--deliv", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S = {n: json.load(open(os.path.join(a.mine, "fields", "cv_fields_%s.json" % n))) for n in NAMES}
    Z = {n: np.load(os.path.join(a.mine, "fields", "cv_fields_%s.npz" % n), allow_pickle=True) for n in NAMES}
    D = {n: json.load(open(os.path.join(a.deliv, "curvature_v1_%s.json" % n))) for n in NAMES}
    Ry = {}
    for fn in sorted(os.listdir(os.path.join(a.mine, "rays"))):
        if fn.endswith(".json"):
            Ry[fn[len("cv_rays_"):-5]] = json.load(open(os.path.join(a.mine, "rays", fn)))

    P("# cv_compare: my recomputation beside the delivery's outputs\n")
    # ---------------- group 1: inputs, records, gate, formation
    P("## G1 inputs, records, gate, formation, housekeeping\n")
    P("| member | records | converged (mine) / unconverged T | gauge not intact (count; T range) | gate barred / unbarred / strain: mine | delivery | formation T (zeros): mine | delivery | episodes (mine) |")
    P("|---|---|---|---|---|---|---|---|---|")
    for n in NAMES:
        s, d = S[n], D[n]
        gi = s["gauge_not_intact_T_over_rs"]
        g_m = "%s / %s / %s" % tuple(f3(s["gate"][k] and s["gate"][k]["T_over_rs"], "%g") for k in ("onet_barred", "onet_unbarred", "strain_unbarred"))
        g_d = "%s / %s / %s" % tuple(f3(d["gate"][k] and d["gate"][k]["T_over_rs"], "%g") for k in ("onet_barred", "onet_unbarred", "strain_unbarred"))
        P("| %s | %d | %d / %s | %d; %s | %s | %s | %g (%s) | %g (%s) | %s |" % (
            n, s["records"], s["n_converged"], ", ".join("%g" % x for x in s["unconverged_T_over_rs"]) or "none", s["n_gauge_not_intact"],
            ("%g to %g" % (min(gi), max(gi)) if gi else "-"), g_m, g_d, s["formation"]["T_over_rs"], ", ".join("%.2f" % z for z in s["formation"]["zeros"]),
            d["formation"]["T_over_rs"], ", ".join("%.2f" % z for z in d["formation"]["zeros"]), "; ".join("%g-%g" % (x, y) for x, y in s["episodes"])))
    P("\n| member | interp_floor N max / median (pre max; post max) | v max / median (pre; post) | N - N_applied max | v - v_applied max | my spline zeros vs stored: max abs diff (a) | records with a different zero count |")
    P("|---|---|---|---|---|---|---|")
    for n in NAMES:
        s = S[n]; i = s["interp_floor"]
        P("| %s | %.1e / %.1e (%s; %s) | %.1e / %.1e (%s; %s) | %.1e | %.1e | %.3g | %d %s |" % (
            n, i["N_max"], i["N_median"], f3(i["N_max_pre"], "%.1e"), f3(i["N_max_post"], "%.1e"), i["v_max"], i["v_median"], f3(i["v_max_pre"], "%.1e"), f3(i["v_max_post"], "%.1e"),
            s["N_minus_N_applied_max"], s["v_minus_v_applied_max"], s["my_zeros_vs_stored"]["max_abs_diff"], s["my_zeros_vs_stored"]["records_with_count_mismatch"],
            ("(T/r_s %s)" % ", ".join("%g" % x for x in s["my_zeros_vs_stored"]["count_mismatch_T_over_rs"])) if s["my_zeros_vs_stored"]["records_with_count_mismatch"] else ""))
    # cross-member node fields
    P("\nCross-member node fields (from the stored nodes):")
    for (x, y) in (("B2", "P64_s4_c19"), ("P64_s4", "P64_s4_c19"), ("B2", "P64_s4")):
        Tx, Ty = Z[x]["Trs"], Z[y]["Trs"]
        common = np.intersect1d(np.round(Tx, 6), np.round(Ty, 6))
        ix = np.searchsorted(np.round(Tx, 6), common); iy = np.searchsorted(np.round(Ty, 6), common)
        nn = min(Z[x]["Nn"].shape[1], Z[y]["Nn"].shape[1])
        dN = np.abs(Z[x]["Nn"][ix][:, :nn] - Z[y]["Nn"][iy][:, :nn]); dv = np.abs(Z[x]["vn"][ix][:, :nn] - Z[y]["vn"][iy][:, :nn])
        jm = int(np.argmax(dN.max(axis=1)))
        P("- %s vs %s: %d common records (T/r_s %g to %g), %d common nodes; max |dN| %.2e (at T/r_s %g), max |dv| %.2e; median per-record max |dN| %.1e" % (
            x, y, len(common), common[0], common[-1], nn, dN.max(), common[jm], dv.max(), np.median(dN.max(axis=1))))
    P("")
    # ---------------- group 2: spheres' curvature
    P("## G2 spheres' curvature K_t^2 against the cap (1/a^2)\n")
    P("| member | cap: mine / deliv | max K_t^2 (T): mine [r] / deliv | rel | centre limit at peak / value at r=0.25 | max/cap mine / deliv | max/gate scale | pre-gate max mine / deliv | at formation | nodes only | barred sites | applied sites | centre identity max | records above cap | above gate scale: mine / deliv |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for n in NAMES:
        s, d = S[n], D[n]
        P("| %s | %.4f / %.4f | %s [%.3g] / %s | %s | %.5g / %.5g | %.4f / %.4f | %.0f / %.0f | %s / %s | %.4g / %.4g | %s / %s | %s / %s | %s / %s | %s / %s | %d / %d | %d / %d |" % (
            n, s["cap"], d["cap"], pk(s["Kt2_max_all"], "%.5g"), s["Kt2_max_r_at_max"], pk(d["Kt2_max_all"], "%.5g"), rel(s["Kt2_max_all"]["value"], d["Kt2_max_all"]["value"]),
            s["Kt2_centre_limit_at_peak"], s["Kt2_025_at_peak"], s["frac_cap_max"], d["frac_cap_max"], s["over_gate_scale_max"], d["over_gate_scale_max"],
            pk(s["Kt2_max_pre"]), pk(d["Kt2_max_pre"]), s["at_formation"]["Kt2_max"], d["at_formation"]["Kt2_max"], pk(s["Kt2_nodes_max"]), pk(d["Kt2_nodes_max_all"]),
            pk(s["Kt2_bar_max"]), pk(d["Kt2_bar_max_all"]), pk(s["Kt2_app_max"]), pk(d["Kt2_app_max_all"]), pk(s["centre_identity_max"]), pk(d["Kt2_centre_series_max"]),
            s["records_above_cap"], d["records_Kt2_above_cap"], s["records_above_gate_scale"], d["records_Kt2_above_gate_scale"]))
    P("\n| member | centre limit vs identity: rel max (intact records) / all / at peak | deliv (spline 0.25 vs identity, max rel) | first record above gate scale | late range T>=15: min / max (T of max) | rise gate -> peak | max on certified records | pre-gate max on intact records | r of max in post-gate records: fraction at my first grid point |")
    P("|---|---|---|---|---|---|---|---|---|")
    for n in NAMES:
        s, d = S[n], D[n]; z = Z[n]
        post = np.arange(len(z["Trs"])) >= int(z["kgate"]); pre = ~post
        frac_first = float(np.mean(z["Kt2_rmax"][post] == z["rg"][0])) if post.any() else float("nan")
        intact = z["gauge_intact"]
        pre_int = float(np.max(z["Kt2_max"][pre & intact])) if (pre & intact).any() else float("nan")
        lr = s["late_range_T15_on"]
        P("| %s | %.2e / %.2e / %.2e | %.2e | %s | %s | %s | %s | %.4g | %.2f |" % (
            n, s["centre_limit_vs_identity"]["rel_max_intact"], s["centre_limit_vs_identity"]["rel_max_all"], s["centre_limit_vs_identity"]["rel_at_peak"],
            d["centre_spline_vs_series_rel_max"], f3(s["first_record_above_gate_scale_T_over_rs"], "%g"),
            ("%.4g / %.4g (%g)" % (lr["min"], lr["max"], lr["T_of_max"])) if lr else "-", f3(s["rise_gate_to_peak"], "%.1f"), pk(s["Kt2_max_certified"]), pre_int, frac_first))
    P("\nProfile of K_t^2 at the peak record and the tail 2gM = K_t^2 r^3 (mine):")
    for n in ("B2", "P64_s2", "P64_s8"):
        s = S[n]
        P("- %s: " % n + ", ".join("%s: %.3g" % (k, v) for k, v in s["profile_at_peak"].items()) + "; M(r) = K_t^2 r^3/(2g): " + ", ".join("%s: %.4g" % (k, v) for k, v in s["M_from_Kt2_r3_at_peak"].items()))
    P("\nB2 time series (mine; the delivery's sec. 3 table in the reading):")
    P("| T/r_s | " + " | ".join(S["B2"]["series"].keys()) + " |")
    P("|---|" + "---|" * len(S["B2"]["series"]))
    P("| max K_t^2 | " + " | ".join("%.3g" % v["Kt2_max"] for v in S["B2"]["series"].values()) + " |")
    P("| /cap | " + " | ".join("%.2g" % v["over_cap"] for v in S["B2"]["series"].values()) + " |")
    P("| norm c1 | " + " | ".join("%.3g" % v["norm_c1"] for v in S["B2"]["series"].values()) + " |")
    # width scaling and gaps
    k4, k2, k8 = S["P64_s4"]["Kt2_max_all"]["value"], S["P64_s2"]["Kt2_max_all"]["value"], S["P64_s8"]["Kt2_max_all"]["value"]
    P("\nWidth scaling of the peak (mine): s2/s4 = %.2f, s4/s8 = %.2f; exponents log ratio / log 2: %.2f, %.2f" % (k2 / k4, k4 / k8, np.log(k2 / k4) / np.log(2), np.log(k4 / k8) / np.log(2)))
    for n in ("P64_s4", "P64_s4_tau4", "P64_s4_tau2", "B2", "P64_s8"):
        ep = S[n]["episodes"]
        gaps = ["%.4g" % (ep[i + 1][0] - ep[i][1]) for i in range(len(ep) - 1)]
        P("- %s episodes %s; gaps (first record with a zero minus last record with a zero, r_s) %s" % (n, "; ".join("%g-%g" % (x, y) for x, y in ep), ", ".join(gaps) or "none"))
    P("")
    # ---------------- group 3: Riemann norm
    P("## G3 slice-frame Riemann norm (unbarred, r <= 256a)\n")
    P("| member | cad | pre-gate max (T; r): mine / deliv | rel | pre-gate max on intact records (mine) | post-gate max (T; r): mine / deliv | rel | abs Kretschmann pre / post: mine | deliv | components at post max (abs max over r at that record): mine | deliv | components at the point of the max (mine) | norm / K_t^2 at post max (mine) |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for n in NAMES:
        for c in ("1", "2", "4"):
            s, d = S[n]["norm"][c], D[n]["norm"][c]
            cm = s["comp_absmax_at_post_max_record"]; cd = d.get("components_at_post_max") or {}; cp = s["comp_at_post_max_point"]
            P("| %s | %s | %s; %s / %s; %s | %s | %s | %s; %s / %s; %s | %s | %s / %s | %s / %s | %s | %s | %s | %s |" % (
                n, c, pk(s["pre"]), f3(s["r_at_pre_max"], "%.3g"), pk(d["pre"]), f3(d.get("r_at_pre_max"), "%.3g"), rel(s["pre"]["value"], d["pre"]["value"]),
                pk(s["pre_intact"]), pk(s["post"]), f3(s["r_at_post_max"], "%.3g"), pk(d["post"]), f3(d.get("r_at_post_max"), "%.3g"), rel(s["post"]["value"], d["post"]["value"]),
                pk(s["kret_abs_pre"], "%.3g"), pk(s["kret_abs_post"], "%.3g"), pk(d["kret_pre"], "%.3g"), pk(d["kret_post"], "%.3g"),
                ", ".join("%.3g" % cm[k] for k in ("R0101", "R0202", "R0212", "R1212", "R2323")) if cm else "-",
                ", ".join("%.3g" % cd[k] for k in ("R0101", "R0202", "R0212", "R1212", "R2323")) if cd else "-",
                ", ".join("%.3g" % cp[k] for k in ("R0101", "R0202", "R0212", "R1212", "R2323")) if cp else "-", f3(s["norm_over_Kt2_at_post_max"], "%.2f")))
    P("\n| member | stencil | pre: mine / deliv | post: mine / deliv |")
    P("|---|---|---|---|")
    for n in NAMES:
        for key, dkey in (("spline_at_nodes", "spline@nodes"), ("FD2_at_nodes", "FD2@nodes"), ("FD4_at_nodes", "FD4@nodes"), ("barred_sites_FD2_from_site2", "barred_sites_FD2"), ("barred_sites_FD2_all_sites", None)):
            s = S[n]["stencil"][key]; d = D[n]["stencil"].get(dkey) if dkey else None
            P("| %s | %s | %s / %s | %s / %s |" % (n, key, pk(s["pre"]), pk(d["pre"]) if d else "-", pk(s["post"]), pk(d["post"]) if d else "-"))
    P("\nSource route (reviewer A's second route) at the nodes:")
    P("| member | r range | records | constraint rel max pre (energy, momentum with T_nr = -J, with +J) | pre intact | post (energy, -J, +J) | median resid/norm pre intact (R0101, R0202, R0212, R1212): mine | deliv text | post: mine | deliv text | norm max src / (N,v) pre: mine | deliv | post: mine | deliv | per-record ratio src/(N,v): pre intact min-max; post min-max |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    dtext = {"P64_s4": ("0.054, 0.081", "0.28, 0.29"), "P64_s4_c19": ("0.046, 0.052", "0.28, 0.29"), "P64_s8": ("0.035, 0.036", "0.16, 0.13"), "P64_s4_tau4": ("0.033, 0.041", "0.29, 0.27"),
             "P64_s4_tau2": ("0.026, 0.034", "0.30, 0.29"), "P64_s2": ("0.31, 0.27", "0.24, 0.33")}
    for n in NAMES:
        s = S[n].get("source_route"); d = D[n].get("reviewa_route")
        if s is None:
            P("| %s | no handoff | | | | | | | | | | | | | |" % n); continue
        cp, ci, co = s["constraint_rel_max_pre"], s["constraint_rel_max_pre_intact"], s["constraint_rel_max_post"]
        mp, mo = s["median_resid_over_norm_pre_intact"], s["median_resid_over_norm_post"]; rr = s["norm_ratio_src_over_Nv_per_record"]
        P("| %s | %g-%g | %d | %s, %s, %s | %s, %s | %s, %s, %s | %s | %s | %s | %s | %s / %s | %s / %s | %s / %s | %s / %s | %s-%s; %s-%s |" % (
            n, s["r_range"][0], s["r_range"][1], s["records"], f3(cp["energy"], "%.2e"), f3(cp["momentum_Tnr=-J"], "%.2e"), f3(cp["momentum_Tnr=+J"], "%.2e"),
            f3(ci["energy"], "%.2e"), f3(ci["momentum_Tnr=-J"], "%.2e"), f3(co["energy"], "%.2e"), f3(co["momentum_Tnr=-J"], "%.2e"), f3(co["momentum_Tnr=+J"], "%.2e"),
            ", ".join(f3(mp[k], "%.3f") for k in ("R0101", "R0202", "R0212", "R1212")), dtext.get(n, ("-", "-"))[0],
            ", ".join(f3(mo[k], "%.3f") for k in ("R0101", "R0202", "R0212", "R1212")), dtext.get(n, ("-", "-"))[1],
            f3(s["norm_src_max_pre"]), f3(s["norm_Nv_nodes_max_pre"]), f3(d["norm_src_pre"]) if d else "-", f3(d["norm_Nv_nodes_pre"]) if d else "-",
            f3(s["norm_src_max_post"]), f3(s["norm_Nv_nodes_max_post"]), f3(d["norm_src_post"]) if d else "-", f3(d["norm_Nv_nodes_post"]) if d else "-",
            f3(rr["min_pre_intact"], "%.2f"), f3(rr["max_pre_intact"], "%.2f"), f3(rr["min_post"], "%.2f"), f3(rr["max_post"], "%.2f")))
    P("")
    # ---------------- group 4: rays
    P("## G4 radial null rays\n")

    def rows_by(rows):
        return {(r_["label"] or ("%.6g" % r_["r0"]), r_["dir"]): r_ for r_ in rows}
    for n in NAMES:
        key = "%s_c1_dt0.125_h0.25" % n
        if key not in Ry:
            continue
        mine = Ry[key]; drows = D[n]["rays"]["1"]["rows"]; dm = rows_by(drows); mm = rows_by(mine["rows"])
        P("### %s: launch T/r_s %.4g (zeros %s), window to %.4g, zeros at the end mine %s / deliv %s, launches %d (deliv %d)\n" % (
            n, mine["T_start_over_rs"], ", ".join("%.2f" % z for z in mine["zeros_formation"]), mine["T_end_over_rs"], ", ".join("%.3f" % z for z in mine["zeros_end"]),
            ", ".join("%.3f" % z for z in D[n]["rays"]["1"]["zeros_end"]), len(mine["rows"]), len(drows)))
        P("| launch | dir | T at r=0 (lam there): mine / deliv | r end: mine / deliv | log10 eps end: mine / deliv | lam end: mine / deliv | dlam last 5 r_s: mine / deliv | dlneps/dT last 5 r_s: mine / deliv | d|r|/dT end: mine / deliv | K_t^2 at end: mine / deliv | status mine | status deliv |")
        P("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for lab in ("centre", "r_in/2", "r_in", "mid", "r_out", "1.5 r_out", "2 r_out", "4 r_out"):
            for dr in ("in", "out"):
                m = mm.get((lab, dr)); d = dm.get((lab, dr))
                if m is None or d is None:
                    continue
                P("| %s %.2f | %s | %s / %s | %.3f / %.3f | %.3f / %.3f | %.5g / %.5g | %.2e / %.2e | %s / %s | %s / %s | %s / %s | %s | %s |" % (
                    lab, m["r0"], dr, ("%.4f (%.4g)" % (m["T_centre_over_rs"], m["lam_centre"])) if m["T_centre_over_rs"] is not None else "-",
                    ("%.4f (%.4g)" % (d["T_centre_over_rs"], d["lam_centre"])) if d["T_centre_over_rs"] is not None else "-",
                    m["r_end"], d["r_end"], m["log10_eps_end"], np.log10(d["eps_end"]), m["lam_end"], d["lam_end"], m["dlam_last_5rs"], d["dlam_last_5rs"],
                    f3(m["dlneps_dT_last_5rs"], "%.4g"), f3(d["dlneps_dT_last_5rs"], "%.4g"), f3(m["drdT_end"], "%.3g"), f3(d["drdT_end"], "%.3g"),
                    f3(m["Kt2_end"], "%.4g"), f3(d["Kt2_at_end"], "%.4g"), m["status"], d["status"]))
        # all launches summary
        rows = mine["rows"]; nin = [r_ for r_ in rows if r_["dir"] == "in"]; nout = [r_ for r_ in rows if r_["dir"] == "out"]
        reach = [r_ for r_ in nin if r_["T_centre_over_rs"] is not None]
        esc_out = [r_["r0"] for r_ in nout if "escap" in r_["status"]]; not_out = [r_["r0"] for r_ in nout if "escap" not in r_["status"]]
        lam_c = [r_["lam_centre"] for r_ in reach]; t_c = [r_["T_centre_over_rs"] for r_ in reach]
        non_esc = [r_ for r_ in rows if "escap" not in r_["status"]]
        P("\nAll launches (mine): %d ingoing, %d reach r = 0 (lam there %.3g to %.3g; T/r_s %.3f to %.3f); outgoing escaping from r0 >= %s, not escaping for r0 <= %s; "
          "non-escaping rays: %d, end r %s to %s, log10 eps end %s to %s, dlam last 5 r_s max %s, dlneps/dT last 5 r_s %s to %s; escaping rays' largest log10 eps %s" % (
              len(nin), len(reach), min(lam_c) if lam_c else float("nan"), max(lam_c) if lam_c else float("nan"), min(t_c) if t_c else float("nan"), max(t_c) if t_c else float("nan"),
              f3(min(esc_out) if esc_out else None, "%.2f"), f3(max(not_out) if not_out else None, "%.2f"), len(non_esc),
              f3(min(r_["r_end"] for r_ in non_esc) if non_esc else None, "%.3f"), f3(max(r_["r_end"] for r_ in non_esc) if non_esc else None, "%.3f"),
              f3(min(r_["log10_eps_end"] for r_ in non_esc) if non_esc else None, "%.2f"), f3(max(r_["log10_eps_end"] for r_ in non_esc) if non_esc else None, "%.2f"),
              f3(max(abs(r_["dlam_last_5rs"]) for r_ in non_esc) if non_esc else None, "%.2e"),
              f3(min(r_["dlneps_dT_last_5rs"] for r_ in non_esc) if non_esc else None, "%.4g"), f3(max(r_["dlneps_dT_last_5rs"] for r_ in non_esc) if non_esc else None, "%.4g"),
              f3(max(r_["log10_eps_end"] for r_ in rows if "escap" in r_["status"]) if len(rows) > len(non_esc) else None, "%.2f")))
        # the delivery's summary for the same
        drs = drows; dnin = [r_ for r_ in drs if r_["dir"] == "in"]; dreach = [r_ for r_ in dnin if r_["T_centre_over_rs"] is not None]
        dnon = [r_ for r_ in drs if "escap" not in r_["status"]]
        P("Delivery's rows: %d ingoing, %d reach r = 0 (lam %.3g to %.3g; T/r_s %.3f to %.3f); non-escaping %d, end r %s to %s, log10 eps %s to %s, dlam last 5 r_s max %s\n" % (
            len(dnin), len(dreach), min(r_["lam_centre"] for r_ in dreach), max(r_["lam_centre"] for r_ in dreach), min(r_["T_centre_over_rs"] for r_ in dreach), max(r_["T_centre_over_rs"] for r_ in dreach),
            len(dnon), f3(min(r_["r_end"] for r_ in dnon) if dnon else None, "%.3f"), f3(max(r_["r_end"] for r_ in dnon) if dnon else None, "%.3f"),
            f3(min(np.log10(r_["eps_end"]) for r_ in dnon) if dnon else None, "%.2f"), f3(max(np.log10(r_["eps_end"]) for r_ in dnon) if dnon else None, "%.2f"),
            f3(max(abs(r_["dlam_last_5rs"]) for r_ in dnon) if dnon else None, "%.2e")))
        # growth split for the 40-window: inside the shell vs 38.6-44
        if mine["T_end_over_rs"] <= 40.0 + 1e-9:
            ins = [abs(r_["dlam_last_5rs"]) for r_ in non_esc if r_["r0"] < 38.0]; edge = [abs(r_["dlam_last_5rs"]) for r_ in non_esc if 38.0 <= r_["r0"] <= 44.5]
            P("dlam over the last 5 r_s at the window end: launched inside the formation shell (r0 < 38) max %s; launched at 38.6 to 44a max %s\n" % (f3(max(ins) if ins else None, "%.2e"), f3(max(edge) if edge else None, "%.2e")))
        # marks (B2 at 40 etc.)
        for mk, mv in mine["marks"].items():
            if float(mk) >= mine["T_end_over_rs"] - 1e-9 and n != "B2":
                continue
            lab_idx = {(mine["rows"][i]["label"] or "%.6g" % mine["rows"][i]["r0"], mine["rows"][i]["dir"]): i for i in range(len(mine["rows"]))}
            P("At T/r_s %s (mine; zeros %s): " % (mk, ", ".join("%.3f" % z for z in mv["zeros"]) if mv["zeros"] else "-") + "; ".join(
                "%s %s: r %.3f, log10 eps %.2f, lam %.4g, dlam5 %.2e, rate %.4g" % (lab, dr, mv["r"][i], mv["log10_eps"][i], mv["lam"][i], mv["dlam_last_5rs"][i], mv["dlneps_dT_last_5rs"][i])
                for (lab, dr), i in lab_idx.items() if lab in ("centre", "r_in", "mid", "r_out", "1.5 r_out", "4 r_out")))
        P("")
    # ray sensitivity variants
    P("### ray variants (mine): cadence, half step, table spacing\n")
    P("| run | in from r_out: T at r=0 (lam) | in from r_out: r end, log10 eps, lam | out from mid: r end, log10 eps, lam | out from centre: r end, log10 eps, lam | out from r_out: r end, status | out from 2 r_out: r end, status | zeros end |")
    P("|---|---|---|---|---|---|---|---|")
    for key in sorted(Ry):
        m = rows_by(Ry[key]["rows"]); a_ = m.get(("r_out", "in")); b_ = m.get(("mid", "out")); c_ = m.get(("centre", "out")); d_ = m.get(("r_out", "out")); e_ = m.get(("2 r_out", "out"))
        if a_ is None:
            continue
        P("| %s | %s | %.3f, %.2f, %.5g | %.3f, %.2f, %.5g | %.3f, %.2f, %.5g | %.2f, %s | %.1f, %s | %s |" % (
            key, ("%.4f (%.4g)" % (a_["T_centre_over_rs"], a_["lam_centre"])) if a_["T_centre_over_rs"] is not None else "-", a_["r_end"], a_["log10_eps_end"], a_["lam_end"],
            b_["r_end"], b_["log10_eps_end"], b_["lam_end"], c_["r_end"], c_["log10_eps_end"], c_["lam_end"], d_["r_end"], d_["status"], e_["r_end"], e_["status"],
            ", ".join("%.3f" % z for z in Ry[key]["zeros_end"])))
    P("\nDelivery's ray variants (its json, cadence 1/2/4/dt_half):")
    for n in ("B2", "P64_s4"):
        for c in ("1", "2", "4", "dt_half"):
            R_ = D[n]["rays"].get(c)
            if R_ is None:
                continue
            m = rows_by(R_["rows"]); a_ = m[("r_out", "in")]; b_ = m[("mid", "out")]; c_ = m[("centre", "out")]
            P("- %s %s: in from r_out T at r=0 %s (lam %s); r end %.3f, log10 eps %.2f, lam %.5g; out from mid %.3f, %.2f, %.5g; out from centre %.3f, %.2f, %.5g; zeros end %s" % (
                n, c, f3(a_["T_centre_over_rs"], "%.4f"), f3(a_["lam_centre"], "%.4g"), a_["r_end"], np.log10(a_["eps_end"]), a_["lam_end"], b_["r_end"], np.log10(b_["eps_end"]), b_["lam_end"],
                c_["r_end"], np.log10(c_["eps_end"]), c_["lam_end"], ", ".join("%.3f" % z for z in R_["zeros_end"])))
    # curvature at the ray ends (B2 at 100): from my field profiles at the last record
    z = Z["B2"]; rg = z["rg"]
    m = rows_by(Ry["B2_c1_dt0.125_h0.25"]["rows"])
    ends = sorted(set(round(r_["r_end"], 3) for r_ in Ry["B2_c1_dt0.125_h0.25"]["rows"] if "escap" not in r_["status"]))
    P("\nCurvature at the non-escaping rays' ends on B2's last record (mine): r end %s; K_t^2 %s; frame norm (c1) %s" % (
        ", ".join("%.3f" % x for x in ends[:3] + ends[-3:]), ", ".join("%.4g" % np.interp(x, rg, z["Kt2_prof"][-1]) for x in ends[:3] + ends[-3:]),
        ", ".join("%.4g" % np.interp(x, rg, z["norm_prof"][-1]) for x in ends[:3] + ends[-3:])))
    # the inner root's motion
    for n, mk in (("B2", 40.0), ("B2", 100.0), ("P64_s4", 40.0)):
        zz = Z[n]; Trs = zz["Trs"]; jj = int(np.argmin(np.abs(Trs - mk)))
        if abs(Trs[jj] - mk) > 1e-9:
            continue
        rin = zz["r_in_rec"]
        j1 = max(jj - 1, 0); j2 = jj
        P("- %s inner root at T/r_s %g: %.3f; its drift over the preceding record %.4g per a" % (n, mk, rin[jj], (rin[j2] - rin[j1]) / (zz["T"][j2] - zz["T"][j1]) if j2 > j1 else float("nan")))
    out = "\n".join(L)
    open(a.out, "w").write(out)
    print(out)


if __name__ == "__main__":
    main()
