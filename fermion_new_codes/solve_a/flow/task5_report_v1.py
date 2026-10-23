#!/usr/bin/env python
"""Task 5 report (SPEC_task5_black_object.md sec. 3; builder analyst): the tables of E0 to E4 from flow_chart_v5.py's and flow_chart_v6.py's
records; every number is read from the JSONs.  Args: --e0 --e1 (comma list) --e1-rmax --e4 --e4-ctl (comma list) --e3 --e2 (comma list)
--e2-rmax --json --md.  Missing inputs are skipped."""
import argparse
import json
import math

import numpy as np


def f(v, d=4):
    if v is None:
        return "-"
    if isinstance(v, str):
        return v
    v = float(v)
    if not math.isfinite(v):
        return "-"
    return "%.*g" % (d, v) if (v == 0 or 1e-3 <= abs(v) < 1e6) else "%.3e" % v


def load(p):
    return json.load(open(p)) if p else None


def e1_section(paths, rmax):
    L = ["## E1: the escape probability by wavelength (P_+ = ||Pi_{h > m} w||^2; r_s = 128, R_m = 32, N = 2048)", "",
         "| m | kappa_j | k0 a | sigma_x | R_esc/r_s | budget/r_s | R_turn/r_s | R_1/2/r_s (lattice) | R_1/2/r_s (symbol) | (lattice - R_esc)/budget | "
         "max abs P_+ - P_sym | max abs P_dyn_ext - P_+ (gate clean) | max abs (norm beyond R_ref at t_max) - P_+ | launches |", "|" + "---|" * 14]
    out = []
    for p in paths:
        d = load(p)
        rs = d["rs"]
        for run in d["runs"]:
            for s in run["series"]:
                rows = [r for r in s["rows"] if "P_plus" in r]
                dps = [abs(r["P_plus"] - r["P_sym"]) for r in rows]
                dd = [abs(r["dyn"]["P_dyn_ext"] - r["P_plus"]) for r in rows if "P_dyn_ext" in r.get("dyn", {})
                      and r["dyn"].get("gate_outward_before_t_min", 1) < 0.01]
                late = [abs(r["dyn"]["beyond_R_ref_tmax"] - r["P_plus"]) for r in rows if "beyond_R_ref_tmax" in r.get("dyn", {})]
                lat = s["R_half_measured"]
                rec = {"mass": run["mass"], "kappa_j": run["kappa_j"], "k0": s["k0"], "sigma_x": s["sigma_x"], "control": s["control"],
                       "R_esc_over_rs": s["R_esc_over_rs"], "budget_over_rs": s["budget"] / rs, "R_turn_over_rs": s["R_turn"] / rs,
                       "R_half_lattice_over_rs": [x / rs for x in lat], "R_half_symbol_over_rs": [x / rs for x in s["R_half_symbol"]],
                       "dev_over_budget": [(x - s["R_esc"]) / s["budget"] for x in lat], "max_P_minus_Psym": max(dps) if dps else None,
                       "max_Pdyn_minus_P": max(dd) if dd else None, "max_late_minus_P": max(late) if late else None, "n_clean": len(dd), "n": len(rows)}
                out.append(rec)
                L.append("| %s | %s | %s%s | %s | %s | %s | %s | %s | %s | %s | %s | %s (%d of %d) | %s | %d |" % (
                    f(run["mass"]), f(run["kappa_j"]), f(s["k0"]), " (control)" if s["control"] else "", f(s["sigma_x"]), f(s["R_esc_over_rs"], 5),
                    f(rec["budget_over_rs"], 3), f(rec["R_turn_over_rs"], 4), ", ".join(f(x, 5) for x in rec["R_half_lattice_over_rs"]) or "none",
                    ", ".join(f(x, 5) for x in rec["R_half_symbol_over_rs"]) or "none", ", ".join(f(x, 3) for x in rec["dev_over_budget"]) or "-",
                    f(rec["max_P_minus_Psym"], 3), f(rec["max_Pdyn_minus_P"], 3), len(dd), len(rows), f(rec["max_late_minus_P"], 3), len(rows)))
    if rmax:
        d = load(rmax)
        L += ["", "- The R_max control (N = %d, R_max = %s r_s): " % (d["N"], f(d["R_max"] / d["rs"])) + "; ".join(
            "m %s kappa_j %s k0 %s: R_1/2/r_s %s" % (f(run["mass"]), f(run["kappa_j"]), f(s["k0"]), ", ".join(f(x / d["rs"], 5) for x in s["R_half_measured"]))
            for run in d["runs"] for s in run["series"]) + "."]
    return L, out


def e4_section(path, ctl_paths):
    d = load(path)
    rs, kappa = d["rs"], d["kappa"]
    L = ["## E4: the clock (packets on the dust law's events; r_s = %s, N = %d, R_obs = %s r_s; t_cross = %s)" % (f(rs), d["N"], f(d["R_obs"] / rs),
                                                                                                              f(d["t_cross"], 6)), ""]
    out = {"series": []}
    for run in d["runs"]:
        for s in run["series"]:
            L += ["### m %s, kappa_j %s, k0 a %s, sigma_x %s%s (saturation 1/(1 - cos k0) form: %s)" % (
                f(run["mass"]), f(run["kappa_j"]), f(s["k0"]), f(s["sigma_x"]), " (control)" if s["control"] else "", f(s["saturation"], 5)), "",
                  "| R_e/r_s | t_e | P_+ | omega_bar | omega_bar_+ | 1+z (packet) | 1+z_+ (escaping) | 1+z symbol avg | 1+z point | tau_peak | tau_half | "
                  "tau_mean | tau_symbol | clean |", "|" + "---|" * 14]
            for r in s["rows"]:
                L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                    f(r["Re_over_rs"], 5), f(r["t_e"], 5), f(r["P_above"], 3), f(r["omega_mean"], 4), f(r.get("omega_mean_above"), 4), f(r.get("one_plus_z")),
                    f(r.get("one_plus_z_plus")), f(r.get("one_plus_z_symbol_avg")), f(r.get("one_plus_z_point")), f(r.get("tau_peak"), 5),
                    f(r.get("tau_half"), 5), f(r.get("tau_mean"), 5), f(r.get("tau_symbol"), 5), r.get("clean")))
            L.append("")
            for k, v in s["fits"].items():
                a, b = v["ln_1pz_vs_kappa_t_arr"], v["ln_dtarr_dte_vs_kappa_t_arr"]
                L.append("- fit %s over R_e/r_s %s: ln(1+z) slope %s (rms %s, n %s); ln(dt_arr/dt_e) slope %s (n %s)." % (
                    k, v["window_points"], f(a["slope"]) if a else "-", f(a["rms_residual"], 2) if a else "-", a["n"] if a else 0,
                    f(b["slope"]) if b else "-", b["n"] if b else 0))
            ss = s.get("symbol_series")
            if ss:
                rows = ss["rows"]
                pick = []
                for target in (1.0, 0.3, 0.1, 0.03, 0.01, 1e-3, 1e-4, 1e-5):
                    q = min(rows, key=lambda q: abs(math.log(q["Re_minus_rs_over_rs"]) - math.log(target)))
                    pick.append(q)
                L.append("- the symbol's point emissions (version 6): " + "; ".join(
                    "R_e - r_s = %s r_s: 1+z %s, t_arr %s, d ln(1+z)/d(kappa t_arr) %s, d ln(dt_arr/dt_e)/d(kappa t_arr) %s" % (
                        f(q["Re_minus_rs_over_rs"], 2), f(q.get("one_plus_z_point")), f(q.get("t_arr"), 6), f(q.get("local_slope_ln1pz"), 4),
                        f(q.get("local_slope_ln_dtarr_dte"), 4)) for q in pick) + ".")
            L.append("")
            out["series"].append({"mass": run["mass"], "kappa_j": run["kappa_j"], "k0": s["k0"], "sigma_x": s["sigma_x"], "fits": s["fits"]})
    for p in ctl_paths:
        c = load(p)
        L.append("- control %s (N %d, R_obs %s r_s): " % (p.split("/")[-1], c["N"], f(c["R_obs"] / c["rs"])) + "; ".join(
            "k0 %s sigma_x %s: %s" % (f(s["k0"]), f(s["sigma_x"]), ", ".join("%s slope %s" % (k, f((v["ln_1pz_vs_kappa_t_arr"] or {}).get("slope")))
                                                                       for k, v in s["fits"].items())) for run in c["runs"] if run["kappa_j"] == -1.0
            for s in run["series"]) + ".")
    return L, out


def e3_section(path):
    d = load(path)
    rs = d["rs"]
    L = ["## E3: what crosses from inside (D3's packets rerun; r_s = %s, R_m = %s, N = %d, R_inner = %s)" % (f(rs), f(d["Rm"]), d["N"], f(d["R_inner"])), "",
         "| kappa_j | omega r_s | entry (half) | exit (half) | symbol entry | symbol exit | launched omega (mean, spread) | returned beyond 2 r_s "
         "(mean, spread) | annuli: the largest-norm exit sample's mean k (k > 0 share) against the root at the midpoint [range] |", "|" + "---|" * 9]
    out = []
    for r in d["runs"]:
        tr = r["track"]
        tc = max(range(len(tr)), key=lambda i: tr[i]["in_Rinner"])
        ann = []
        for ia, key in enumerate(r["roots_inside"].keys()):
            best = None
            for q in tr[tc:]:
                a = q["annuli"][ia]
                if "mean_k_pos" in a and (best is None or a["norm"] > best["norm"]):
                    best = a
            rt = r["roots_inside"][key]
            ann.append("%s: %s (%s) vs %s [%s, %s]" % (key, f(best["mean_k_pos"], 3) if best else "-", f(best["weight_k_pos"], 3) if best else "-",
                                                     f(rt["mid"], 3), f(rt["hi"], 3), f(rt["lo"], 3)))
        sym = r["symbol_ray"]
        L.append("| %s | %s | %s | %s | %s | %s | %s, %s | %s, %s | %s |" % (
            f(r["kappa_j"]), f(r["omega_rs"]), f(r["t_entry_half"], 5), f(r["t_exit_half"], 5), f(sym.get("t_entry"), 5), f(sym.get("t_exit"), 5),
            f(r["launched"]["omega_mean"], 5), f(r["launched"]["omega_spread"], 3), f(r["returned_beyond_2rs"]["omega_mean"], 5),
            f(r["returned_beyond_2rs"]["omega_spread"], 3), "; ".join(ann)))
        out.append({"kappa_j": r["kappa_j"], "omega_rs": r["omega_rs"], "t_entry_half": r["t_entry_half"], "t_exit_half": r["t_exit_half"], "symbol": sym})
    return L, out


def e2_section(paths, rmax):
    ch = []
    meta = None
    for p in paths:
        d = load(p)
        meta = d
        ch += d["channels"]
    rs, bc = meta["rs"], meta["b_c"]
    L = ["## E2: the capture cross section (r_s = %s, R_m = %s; b_c = 3 sqrt(3) r_s/2 = %s; the geometric limit 27/4 = 6.75)" % (f(rs), f(meta["Rm"]), f(bc, 6)), "",
         "| m | omega r_s | omega b_c | channels | sigma/(pi r_s^2), the spec's T (gate at r_s/2) | sigma/(pi r_s^2), max norm inside r_s | m > 0: pi/p^2 form (spec's T) | "
         "|kappa| at T = 1/2 (spec's T, kappa < 0; > 0) | gate not triggered | packet Killing spread flags |", "|" + "---|" * 10]
    out = []
    for m in sorted({c["mass"] for c in ch}):
        for o in sorted({c["omega_rs"] for c in ch if c["mass"] == m}):
            rows = [c for c in ch if c["mass"] == m and c["omega_rs"] == o and c["variant"] == "base" and "T_first_passage" in c]
            if not rows:
                continue
            om = o / rs
            S1 = sum(abs(c["kappa_j"]) * c["T_first_passage"] for c in rows) * math.pi / om ** 2 / (math.pi * rs * rs)
            S2 = sum(abs(c["kappa_j"]) * c["N_inside_max"] for c in rows) * math.pi / om ** 2 / (math.pi * rs * rs)
            p2 = om * om - m * m
            S3 = S1 * om * om / p2 if m > 0 and p2 > 0 else None
            half = []
            for sg in (-1, 1):
                rr = sorted([c for c in rows if c["kappa_j"] * sg > 0], key=lambda c: abs(c["kappa_j"]))
                x = [abs(c["kappa_j"]) for c in rr]
                y = [c["T_first_passage"] for c in rr]
                h = None
                for i in range(len(x) - 1):
                    if y[i] >= 0.5 > y[i + 1]:
                        h = x[i] + (x[i + 1] - x[i]) * (y[i] - 0.5) / (y[i] - y[i + 1])
                half.append(h)
            ng = sum(1 for c in rows if c.get("t_gate") is None)
            nk = sum(1 for c in rows if c.get("killing_flag_spread_above_omega_over_6"))
            out.append({"mass": m, "omega_rs": o, "sigma_spec": S1, "sigma_maxnorm": S2, "sigma_pi_p2": S3, "kappa_half": half, "n": len(rows)})
            L.append("| %s | %s | %s | %d | %s | %s | %s | %s; %s | %d | %d |" % (f(m), f(o), f(om * bc, 4), len(rows), f(S1, 4), f(S2, 4), f(S3, 4),
                                                                              f(half[0], 4), f(half[1], 4), ng, nk))
    L += ["", "### Per channel (kappa_j < 0; the spec's T, the max norm inside r_s, the gate time)", ""]
    for m in sorted({c["mass"] for c in ch}):
        for o in sorted({c["omega_rs"] for c in ch if c["mass"] == m}):
            rows = sorted([c for c in ch if c["mass"] == m and c["omega_rs"] == o and c["variant"] == "base" and "T_first_passage" in c
                           and c["kappa_j"] < 0], key=lambda c: abs(c["kappa_j"]))
            if rows:
                L.append("- m %s, omega r_s %s: " % (f(m), f(o)) + "; ".join("%d: %s / %s" % (int(c["kappa_j"]), f(c["T_first_passage"], 3), f(c["N_inside_max"], 3))
                                                                         for c in rows) + ".")
    ctl = [c for c in ch if c["variant"] != "base" and "T_first_passage" in c]
    if ctl:
        L += ["", "### Controls at omega r_s = 4 (massless)", ""]
        for v in sorted({c["variant"] for c in ctl}):
            rr = sorted([c for c in ctl if c["variant"] == v], key=lambda c: c["kappa_j"])
            base = {c["kappa_j"]: c for c in ch if c["variant"] == "base" and c["omega_rs"] == 4 and c["mass"] == 0 and "T_first_passage" in c}
            L.append("- %s: " % v + "; ".join("%d: %s (base %s)" % (int(c["kappa_j"]), f(c["T_first_passage"], 4), f(base.get(c["kappa_j"], {}).get("T_first_passage"), 4))
                                              for c in rr) + ".")
    if rmax:
        d = load(rmax)
        L.append("- R_max doubled (N 4096): " + "; ".join("kappa_j %d omega r_s %s: %s / %s" % (int(c["kappa_j"]), f(c["omega_rs"]), f(c.get("T_first_passage"), 4),
                                                                                              f(c.get("N_inside_max"), 4)) for c in d["channels"] if "T_first_passage" in c) + ".")
    return L, out


def main():
    ap = argparse.ArgumentParser()
    for k in ("--e0", "--e1", "--e1-rmax", "--e4", "--e4-ctl", "--e3", "--e2", "--e2-rmax"):
        ap.add_argument(k, default=None)
    ap.add_argument("--json", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    L = ["# Task 5: the black object of the model, tables (task5_report_v1.py; prescribed flow (simpler version); PG radial; v declared)", ""]
    out = {}
    if a.e0:
        d = load(a.e0)
        i = d["i_cut_current"]
        L += ["## E0: checks", "", "- (i) the cut current: sign %s; the largest error of the Gauss-Legendre integral and of the Richardson difference: %s with "
              "+2 Im, %s with -2 Im (r_s = 32, N = 1024, D3's packet at omega r_s = 2). (ii) the launched packet's Killing mean %s and spread %s against "
              "the target %s. (iii) Q^H Q - 1: %s; the flat sea's idempotency: %s." % (i["sign_fixed"], f(i["max_err_plus"], 3), f(i["max_err_minus"], 3),
                                                                                       f(d["ii_killing_readout"]["omega_mean"], 6), f(d["ii_killing_readout"]["omega_spread"], 4),
                                                                                       f(d["ii_killing_readout"]["omega_target"], 6), f(d["iii_orthonormality"]["QhQ_minus_1_max"], 3),
                                                                                       f(d["iii_flat_sea_idempotency_max"], 3)), ""]
        out["e0"] = {"sign": i["sign_fixed"], "err_plus": i["max_err_plus"]}
    if a.e1:
        x, out["e1"] = e1_section(a.e1.split(","), a.e1_rmax)
        L += x + [""]
    if a.e4:
        x, out["e4"] = e4_section(a.e4, a.e4_ctl.split(",") if a.e4_ctl else [])
        L += x + [""]
    if a.e3:
        x, out["e3"] = e3_section(a.e3)
        L += x + [""]
    if a.e2:
        x, out["e2"] = e2_section(a.e2.split(","), a.e2_rmax)
        L += x + [""]
    L.append("- Every number above is read from the records by the script.")
    json.dump(out, open(a.json, "w"), indent=1, default=float)
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L)[:6000])


if __name__ == "__main__":
    main()
