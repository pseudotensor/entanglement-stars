#!/usr/bin/env python3
"""a4_beta_eff_tables_v1_1.py (solve, page_verify A4 v1.1): version 1's tables (with the 4 r_s windows, reviewer B minor 5) plus the v1.1 blocks
(the all-window scan, the formation comparator, the post-burst occupation, the thermal increments, the upper-bound test, the alternatives,
the outgoing spectrum). No recomputation. Usage: a4_beta_eff_tables_v1_1.py A4_beta_eff_v1.1.json A4_spectrum_v1.1.json OUT.md"""
import json, sys
import numpy as np


def f(v, p=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    return ("%%.%dg" % p) % v


def main():
    d = json.load(open(sys.argv[1]))
    rc, rf, ro = d["rows_cut"], d["rows_cut_forward"], d["rows_occ"]
    get = lambda w, st, cut: next((r for r in rc if r["window"] == w and r["set"] == st and r["cut"] == cut), None)
    geto = lambda w, st, sh: next((r for r in ro if r["window"] == w and r["set"] == st and (r["shifted_by"] > 0) == sh), None)
    order = ["formation", "formation_a", "formation_b", "formation_c", "burst", "burst_42_46"] + \
            ["late6_%d_%d" % (a, a + 6) for a in range(46, 100, 6)] + ["late10_%d_%d" % (a, a + 10) for a in range(46, 96, 10)] + \
            ["late10_90_100", "late_46_100", "check_80_100", "late4_48_52"] + ["late4_%d_%d" % (a, a + 4) for a in range(52, 100, 4)]
    L = []
    L.append("### Table 1. Cut-based beta_eff = T_H dS_cut / dE_collar per window (collar record clock; the 2 r_s cut lagged by the ray map)\n")
    L.append("Columns per cut: K = 18 against the K = 18 collar flow (L_18) | K = 18 against L_12 | kappa 1 alone (16 x_1 against kappa 1's flow) | kappa 1+2 against L_12. "
             "dS and dE are the window's changes (nats; lattice energy units). 'err' is T_H sqrt(floor^2 + (K c)^2 + interp_ctrl^2)/|dE| for the K = 18 sum (the floor alone for kappa 1). "
             "'unlagged' is the 2 r_s cut read on the collar's window without the lag.\n")
    L.append("| window (collar clock) | lag at A, B (r_s) | T_H | dE L_18 | dE L_12 | 16 r_s: dS K18 | beta K18/L18 (err) | beta K18/L12 | beta k1 | beta k12 | 2 r_s lagged: dS K18 | beta K18/L18 (err) | beta K18/L12 | beta k1 | beta k12 | 2 r_s unlagged: beta K18/L18, K18/L12, k1 |")
    L.append("|" + "---|" * 16)
    for w in order:
        a16 = get(w, "K18", "16"); a2 = get(w, "K18", "2")
        if a16 is None:
            continue
        b16 = get(w, "K18_vs_L12", "16"); k16 = get(w, "k1", "16"); kk16 = get(w, "k12", "16")
        b2 = get(w, "K18_vs_L12", "2"); k2 = get(w, "k1", "2"); kk2 = get(w, "k12", "2")
        L.append("| %s %g-%g | %s, %s | %s | %s | %s | %s | %s (%s) | %s | %s | %s | %s | %s (%s) | %s | %s | %s | %s, %s, %s |" % (
            w, a16["A"], a16["B"], f(a2["lag_A"]), f(a2["lag_B"]), f(a16["T_H"], 4), f(a16["dE_collar"]), f(b16["dE_collar"]),
            f(a16["dS_cut"]), f(a16["beta"]), f(a16["err_beta_from_dS"], 2), f(b16["beta"]), f(k16["beta"]), f(kk16["beta"]),
            f(a2["dS_cut"]), f(a2["beta"]), f(a2["err_beta_from_dS"], 2), f(b2["beta"]), f(k2["beta"]), f(kk2["beta"]),
            f(a2.get("beta_unlagged")), f(b2.get("beta_unlagged")), f(k2.get("beta_unlagged"))))
    L.append("\n### Table 2. The 2 r_s cut's own windows mapped forward to the collar (emission clock at 2 r_s -> arrival at 16 r_s by the ray map; collar flow interpolated linearly between checkpoints)\n")
    L.append("| emission window at 2 r_s | collar window | dS K18 | dE L_18 | beta K18/L18 | beta K18/L12 | beta k1 | beta k12 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for w in sorted(set(r["window"] for r in rf), key=lambda x: [r["emit_A"] for r in rf if r["window"] == x][0] * 100 + [r["emit_B"] for r in rf if r["window"] == x][0]):
        g = {r["set"]: r for r in rf if r["window"] == w}
        a = g["K18"]
        L.append("| %s %g-%g | %.2f-%.2f | %s | %s | %s | %s | %s | %s |" % (w, a["emit_A"], a["emit_B"], a["collar_A"], a["collar_B"], f(a["dS_cut"]),
                 f(a["dE_collar_interp"]), f(a["beta"]), f(g["K18_vs_L12"]["beta"]), f(g["k1"]["beta"]), f(g["k12"]["beta"])))
    L.append("\n### Table 3. Occupation-based beta_occ = T_H dS_occ / dE_occ on the exterior Killing shells beyond 24 r_s (diagonal entropy excess over the instantaneous vacuum), per window\n")
    L.append("'shifted' = the collar window moved by +%g r_s (the ray-map lag 16 -> 24 r_s, median %.3f r_s over emission 36-90), i.e. the same radiation as Table 1's row; 'own' = the window on the shells' clock. Bands of |e|/T_H^ref (T_H^ref = %.6g): %s.\n" % (
        d["lag"]["shells_shift_used"], d["lag"]["lag_16_24_median_36_90"], d["T_H_ref"], ", ".join(d["occ_bands_TH"])))
    L.append("| collar window | shells window | dS_occ K18 | dE_occ K18 | beta_occ K18 | beta_occ k1 | beta_occ k12 | beta per band K18 | share of dE per band K18 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for w in order:
        for sh in (True, False):
            a = geto(w, "K18", sh)
            if a is None:
                continue
            k1 = geto(w, "k1", sh); k12 = geto(w, "k12", sh)
            eb = np.array(a["dE_band"]); share = eb / a["dE_occ"] if a["dE_occ"] else eb * np.nan
            L.append("| %s %g-%g | %g-%g (%s) | %s | %s | %s | %s | %s | %s | %s |" % (w, a["A_collar"], a["B_collar"], a["A"], a["B"], "shifted" if sh else "own",
                     f(a["dS_occ"]), f(a["dE_occ"]), f(a["beta_occ"]), f(k1["beta_occ"]), f(k12["beta_occ"]),
                     ", ".join(f(x, 2) for x in a["beta_band"]), ", ".join(f(x, 2) for x in share)))
    cal = d["occ_calibration"]
    L.append("\n### Table 4. The estimator's thermal calibration (a Fermi-Dirac fill at T_H of the same stored e_j, against the block's ground), median over the clocks from 40 r_s\n")
    L.append("| set | total | per band (%s) |" % ", ".join(d["occ_bands_TH"]))
    L.append("|---|---|---|")
    t = np.array(cal["K18"]["t"])
    for nm in ("k1", "k12", "K18"):
        b = np.array(cal[nm]["beta_fill_band"], float)[t >= 40]
        L.append("| %s | %s | %s |" % (nm, f(float(np.nanmedian(np.array(cal[nm]["beta_fill"])[t >= 40])), 4), ", ".join(f(x, 3) for x in np.nanmedian(b, axis=0))))
    lg = d["lag"]
    L.append("\n### Table 5. The lag 2 -> 16 r_s (ray map in the record's fields) against the fixed-charge flight comparison\n")
    te = np.array(lg["t_emit_2rs"]); la = np.array(lg["lag_2_16"])
    L.append("| emission clock at 2 r_s | 12 | 14 | 20 | 26 | 30 | 40 | 50 | 60 | 70 | 79 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    L.append("| lag (r_s) | " + " | ".join(f(float(np.interp(x, te, la)), 4) for x in (12, 14, 20, 26, 30, 40, 50, 60, 70, 79)) + " |")
    L.append("\nFixed-charge flight 2 -> 16 r_s: %.3f r_s at 2gE_f (formation), %.3f r_s at 2gE(100 r_s); 16 -> 24 r_s: %.3f r_s (formation charge) against the ray map's median %.3f.\n" % (
        lg["fixed_charge_2_16_formation"], lg["fixed_charge_2_16_late_charge"], lg["fixed_charge_16_24_formation"], lg["lag_16_24_median_36_90"]))
    L.append("Occupation energy check (sum_j e_j (n_j - nvac_j) against the stored inventory E_D beyond 24 r_s): %s\n" % json.dumps(d["occ_energy_check_max_rel"]))
    v = d["v1_1"]
    L.append("\n### Table 6 (v1.1). Every window of 6 r_s or more on the 2 r_s checkpoint grid, 40 <= A < B <= 100 (406 per set and cut; the 2 r_s cut lagged)\n")
    L.append("| set, cut | max (window, dE) | min (window) | max, 6 r_s | max, 10 r_s | max, >= 20 r_s | windows > 0.252 | windows > 0.5 | max on the v1 grid (6 and 10 r_s from 46) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for k_, a in v["all_windows"].items():
        L.append("| %s | %.3f (%g-%g, %.1e) | %.3f (%g-%g) | %.3f (%g-%g) | %.3f (%g-%g) | %.3f (%g-%g) | %d | %d | %.3f |" % (k_, a["max"]["beta"], a["max"]["A"], a["max"]["B"], a["max"]["dE"],
                 a["min"]["beta"], a["min"]["A"], a["min"]["B"], a["max_len6"]["beta"], a["max_len6"]["A"], a["max_len6"]["B"], a["max_len10"]["beta"], a["max_len10"]["A"], a["max_len10"]["B"],
                 a["max_len_ge20"]["beta"], a["max_len_ge20"]["A"], a["max_len_ge20"]["B"], a["n_above_0.252"], a["n_above_0.5"], a["max_from46_len6_len10"]["beta"]))
    fo = v["formation"]
    L.append("\n### Table 7 (v1.1). The formation window 14-40 r_s at the 2 r_s cut against the Hawking comparator L_H(T_H) of kappa 1, 2\n")
    L.append("| excess kappa 1 | kappa 1+2 | K18 | comparator energy (integral of LH12_TH over 14-40) | its entropy at beta = 2 (2 E / T_H, T_H mean %.4e) | ratio kappa 1+2 / comparator | collar flow kappa 1, 2 over 36-62 | over the comparator's energy |" % fo["T_H_mean"])
    L.append("|---|---|---|---|---|---|---|---|")
    L.append("| %.3f | %.3f | %.3f | %.3e | %.3f | %.3f | %.4f | %.1f |" % (fo["excess_2rs_k1"], fo["excess_2rs_k12"], fo["excess_2rs_K18"], fo["comparator_energy_LH12_TH"], fo["comparator_entropy_at_beta2"], fo["ratio_k12_to_comparator"], fo["collar_flow12_36_62"], fo["collar_flow12_over_comparator"]))
    L.append("\n### Table 8 (v1.1). The occupation detector: post-burst (shells 56-92 = collar 46-82 shifted), burst only (shells 50-56), burst-inclusive (shells' own 46-100)\n")
    L.append("| window | set | dS_occ | dE_occ | beta_occ | energy share per band (%s) | beta per band |" % ", ".join(v["post_burst"]["bands"]))
    L.append("|---|---|---|---|---|---|---|")
    for lab, blk in (("post-burst 56-92", v["post_burst"]), ("burst 50-56", v["burst_only_50_56"]), ("burst-inclusive 46-100 own", v["burst_inclusive_46_100_own"])):
        for nm in ("K18", "k12", "k1"):
            r = blk[nm]
            L.append("| %s | %s | %.3f | %.4f | %.3f | %s | %s |" % (lab, nm, r["dS"], r["dE"], r["beta"], ", ".join("%.3f" % x for x in r["band_share"]), ", ".join(f(x, 2) for x in r["beta_band"])))
    L.append("\n### Table 9 (v1.1). Thermal references of the occupation estimator: the fill (Table 4) and a thermal increment n = g0 + chi (f - g0) (median over clocks from 40)\n")
    L.append("| set | fill | chi 0.3 | chi 0.1 | chi 0.03 |")
    L.append("|---|---|---|---|---|")
    for nm in ("k1", "k12", "K18"):
        ii = v["increments"][nm]
        L.append("| %s | %s | %.3f | %.3f | %.3f |" % (nm, f(float(np.nanmedian(np.array(cal[nm]["beta_fill"])[t >= 40])), 4), ii["chi_0.3"], ii["chi_0.1"], ii["chi_0.03"]))
    ub = v["upper_bound"]
    L.append("\nUpper-bound test (the diagonal excess S_occ against the stored von Neumann excess beyond the same 24 r_s cut): " + "; ".join(
        "%s: smallest margin per clock from 16 r_s %.4f (at %g), at 100 %.2f against %.2f nats, windows of 6 r_s or more from 56 with the S_occ increment above the von Neumann one %d of %d (smallest margin %.3f at %g-%g)" % (
            nm, u["min_clock_margin_from16"], u["clock_of_min"], u["at_100"][0], u["at_100"][1], u["windows_occ_above_vn"], u["windows_from56"], u["smallest_window_margin"][0], u["smallest_window_margin"][1], u["smallest_window_margin"][2])
        for nm, u in ub.items()) + ".\n")
    mg = v["margins"]["smallest"]
    L.append("Smallest occupation margins against the fill over the 6 and 10 r_s windows (shifted where available, own otherwise): " + "; ".join("%s %.2f (%s, %s clock, beta_occ %.3f)" % (nm, x["margin"], x["window"], x["clock"], x["beta_occ"]) for nm, x in mg.items()) + ".\n")
    al = v["alternatives"]
    L.append("\n### Table 10 (v1.1). Alternatives (K18 unless stated)\n")
    L.append("| alternative | value |")
    L.append("|---|---|")
    L.append("| fixed-charge lag (formation charge, %.3f r_s): 2 r_s cut 46-100 | %.4f |" % (al["lag_fixed_charge_formation"]["lag"], al["lag_fixed_charge_formation"]["beta_K18_2rs_46_100"]))
    L.append("| fixed-charge lag (charge at 100, %.3f r_s): 2 r_s cut 46-100 | %.4f |" % (al["lag_fixed_charge_late"]["lag"], al["lag_fixed_charge_late"]["beta_K18_2rs_46_100"]))
    sc = al["slope_clock"]
    L.append("| slope clock kappa_T/(2 pi) in place of T_H (%.3f T_H at 100): 16 r_s 46-100; 70-76 | %.4f; %.4f |" % (sc["T_slope_over_T_H_at_100"], sc["beta_K18_16_46_100"], sc["beta_K18_16_70_76"]))
    iv = al["inventory_denominator"]
    L.append("| exterior inventory beyond the collar as the denominator (dE %.4f against the flow's %.4f): 16 r_s 46-100 | %.4f |" % (iv["dE_inventory"], iv["dE_flow"], iv["beta_K18_16_46_100"]))
    oo = al["outward_only"]
    L.append("| collar split, incoming over outgoing at the %d checkpoints from 46 | max %.4f, median %.4f |" % (oo["checkpoints"], oo["in_over_out_max"], oo["in_over_out_median"]))
    rr = al["raw_cut_entropy_16_46_100"]
    L.append("| raw cut entropy change against the excess change, 16 r_s 46-100 (nats) | %.3f against %.3f |" % (rr["dS_raw"], rr["dS_excess"]))
    pc = al["per_channel"]
    for k_ in ("k1_46_100", "k1_80_100", "k2_46_100", "k2_80_100", "k3_46_100", "k3_80_100"):
        L.append("| %s: beta at 16 r_s; 2 r_s lagged (dS16, dE) | %.3f; %.3f (%.3f, %.2e) |" % (k_, pc[k_]["beta16"], pc[k_]["beta2_lagged"], pc[k_]["dS16"], pc[k_]["dE"]))
    es = pc["energy_share_46_100"]
    L.append("| collar energy share 46-100: kappa 1; 2; 1+2; 3-18 | %.3f; %.3f; %.3f; %.3f |" % (es["k1"], es["k2"], es["k1_2"], es["k3_18"]))
    for cut, bs in al["other_cuts_K18_6rs"].items():
        L.append("| %s r_s cut read forward by its ray-map lag, 6 r_s windows | %s |" % (cut, "; ".join("%g-%g: %.3f" % tuple(b_) for b_ in bs)))
    wd = al["occ_energy_window_diff_vs_inventory"]
    L.append("| occupation energy window differences against the inventory's (K18; kappa 1) | max %.3f, median %.4f; max %.3f, median %.4f |" % (wd["K18"]["max_rel"], wd["K18"]["median_rel"], wd["k1"]["max_rel"], wd["k1"]["median_rel"]))
    sp = json.load(open(sys.argv[2]))
    L.append("\n### Table 11 (v1.1). Post-burst outgoing spectrum, kappa 1 and 2, shells %g-%g (energy-weighted occupation per outgoing mode, relative to the instantaneous vacuum)\n" % tuple(sp["window_shells"]))
    L.append("| bin of abs(e)/T_H^ref | " + " | ".join(r["bin_TH"].split(" ")[0] for r in sp["rows"]) + " |")
    L.append("|---" * (len(sp["rows"]) + 1) + "|")
    L.append("| occupation | " + " | ".join(f(r["occupation_per_out_mode"], 2) for r in sp["rows"]) + " |")
    L.append("| energy share | " + " | ".join("%.3f" % r["share"] for r in sp["rows"]) + " |")
    L.append("\nOutgoing energy share above 10 T_H %.3f, above 24 T_H %.3f, above 26 T_H %.3f (total %.4f).\n" % (sp["share_above_10TH"], sp["share_above_24TH"], sp["share_above_26TH"], sp["total_dE_out"]))
    open(sys.argv[3], "w").write("\n".join(L) + "\n")
    print("wrote", sys.argv[3])


if __name__ == "__main__":
    main()
