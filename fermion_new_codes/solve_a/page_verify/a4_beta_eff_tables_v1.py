#!/usr/bin/env python3
"""a4_beta_eff_tables_v1.py (solve, page_verify A4): markdown tables from A4_beta_eff_v1.json (no recomputation).
Usage: a4_beta_eff_tables_v1.py A4_beta_eff_v1.json OUT.md"""
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
            ["late10_90_100", "late_46_100", "check_80_100"]
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
    open(sys.argv[2], "w").write("\n".join(L) + "\n")
    print("wrote", sys.argv[2])


if __name__ == "__main__":
    main()
