#!/usr/bin/env python
"""page_v4_tables_v1.py (the solve lane; PAGE_solved_history_v4): the table script. Loads page_v4_data_v1.json (the producer's output) and writes the
markdown tables of the delivery (page_v4_tables_v1.md). It computes nothing new beyond ratios and ranges of stored numbers.
Usage: python page_v4_tables_v1.py DATA_JSON OUT_MD"""
import json
import math
import sys

CUTS = ["2", "16", "24", "30", "48"]
READ_CLOCKS = [40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
PRE_CLOCKS = [14.0, 16.0, 20.0, 30.0]


def f(x, fmt="%.3e"):
    if x is None:
        return "-"
    if isinstance(x, float) and math.isnan(x):
        return "-"
    try:
        return fmt % x
    except (TypeError, ValueError):
        return str(x)


def pct(x):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else "%.2f" % (100.0 * x)


def rng(vals, fmt="%.2e"):
    v = [x for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return ("%s to %s" % (fmt % min(v), fmt % max(v))) if v else "-"


def main():
    D = json.load(open(sys.argv[1])); out = sys.argv[2]
    B = D["B2"]; cl = B["clocks"]; ci = {t: i for i, t in enumerate(cl)}
    S8 = D["s8"]; cl8 = S8["clocks"]; ci8 = {t: i for i, t in enumerate(cl8)}
    RB = {(r["t"], r["cut"]): r for r in B["readout"]}; R8 = {(r["t"], r["cut"]): r for r in S8["readout"]}
    PR = D["probes"]; SB = D["sub16_B2"]; SS = D["sub16_s8"]; CP = D["completions"]; FL = D["flux"]; MK = D["marks"]; IC = D["interp_control"]; D13 = D["d13"]; CH = D["checks"]
    L = []
    W = L.append
    # ------------------------------------------------------------ T1: the readout
    W("### T1. The exterior-cut excess of the B2 history summed over kappa = 1 to 18 (weights 16 kappa), nats, at the readout clocks and the cuts; its errors beside")
    W("")
    W("S = sum_k 16 k x_k (x_k the channel's excess S_X(C) - S_X(P_-(h)) at the cut); F_meas the propagation floor (the measured impurity's entropy with the truncation's theorem bound, weighted and summed); F_apriori the a priori allowance (the bound the impurity was checked against); c the largest |16 k x_k| over kappa 12 to 18 at the pair; K c = 18 c the exterior common mode's share; the verdict is the post-pass v1f's (not formed if |S| < K c; within the floor if 0.01 |S| < F_meas; converged if the last three terms fall at the clock and its neighbours, are resolved, the geometric tail and K c are within 0.01 |S|; else formed, tail open). The tail beyond kappa 18 is OPEN at every pair (T2). Rows at 14 to 30 r_s are before the readout clocks (the post-pass rerun from 14 r_s); rows from 40 r_s are the busi lane's K = 18 composition (W1_B2_composed_v3).")
    W("")
    W("| T/r_s | cut r_s | S (nats) | F_meas | F_meas/\\|S\\| | F_apriori | c | K c | K c/\\|S\\| (percent) | q_x (last 3) | verdict (v1f) | reason | tail beyond kappa 18 |")
    W("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t in PRE_CLOCKS + READ_CLOCKS:
        for lab in CUTS:
            r = RB.get((t, lab))
            if r is None:
                continue
            W("| %g | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | OPEN |" % (t, lab, f(r["S"], "%.5e"), f(r["F_meas"], "%.2e"), f(r.get("F_meas_over_S"), "%.1e"), f(r["F_apriori"], "%.3f"), f(r["c"], "%.2e"), f(r["share"], "%.2e"), pct(r.get("share_over_S")), f(r.get("q_x"), "%.3f"), r["verdict"], r.get("why") or "-"))
    W("")
    tl = {}
    for (t, lab), r in RB.items():
        if t >= 40.0 - 1e-9:
            tl.setdefault(lab, {})[r["verdict"]] = tl.setdefault(lab, {}).get(r["verdict"], 0) + 1
    W("Tally over the 31 readout clocks 40 to 100 r_s (not formed / within the floor / converged / formed, tail open): " + "; ".join("%s r_s: %d / %d / %d / %d" % (lab, tl[lab].get("not formed", 0), tl[lab].get("within the floor", 0), tl[lab].get("converged", 0), tl[lab].get("formed, tail open", 0)) for lab in CUTS) + ".")
    conv = [(t, lab) for (t, lab), r in RB.items() if r["verdict"] == "converged"]
    W("Pairs the v1f rule prints converged: %s (the probes' reading rules the tail open at that cut too: T2)." % (", ".join("%g/%s" % p for p in sorted(conv)) or "none"))
    big = [r for r in RB.values() if r["t"] >= 40.0 - 1e-9 and abs(r["S"]) > 3.0]
    W("F_meas over the readout pairs: %s nats; F_meas/|S| where |S| exceeds 3 nats: at most %s (%d pairs). K c/|S| over the formed pairs from 80 r_s: %s percent. F_apriori at 100 r_s: %s nats (%s of the sum)." % (
        rng([r["F_meas"] for r in RB.values() if r["t"] >= 40.0 - 1e-9]), f(max(r["F_meas"] / abs(r["S"]) for r in big), "%.2e"), len(big),
        rng([100.0 * r["share_over_S"] for r in RB.values() if r["t"] >= 80.0 - 1e-9 and r["verdict"] != "not formed"], "%.2f"), rng([r["F_apriori"] for r in RB.values() if abs(r["t"] - 100.0) < 1e-9], "%.3f"), rng([100.0 * r["F_apriori_over_S"] for r in RB.values() if abs(r["t"] - 100.0) < 1e-9], "%.1f") + " percent"))
    W("")
    LV = D["levels"]; pk = LV["largest_abs_sum_from_40"]
    W("The sums against the levels (%s): the largest |S| from 40 r_s is %s nats at T/r_s %g, the cut %s r_s, which is %s of A/(4G) (%s nats) and %s of eta A/a^2 (%s nats) at that record; at 100 r_s the five cuts read %s of A/(4G)." % (
        LV["definition"], f(pk["S"], "%.5f"), pk["t"], pk["cut"], f(pk["S_over_A_over_4G"], "%.3e"), f(pk["A_over_4G"], "%.1f"), f(pk["S_over_eta_A_over_a2"], "%.3e"), f(pk["eta_A_over_a2"], "%.1f"),
        ", ".join("%s r_s: %s" % (lab, f(LV["SUM_over_A_over_4G"][ci[100.0]][j], "%.2e")) for j, lab in enumerate(CUTS))))
    W("")
    W("### T1c. The regulated columns behind T1: the raw exterior-cut entropy and the held vacuum's, summed over kappa 1 to 18 with the weights 16 kappa (nats; the raw cut entropy of the radial truncation has no limit in kappa and prints only as a regulated column; S = raw - vacuum)")
    W("")
    W("| T/r_s | cut r_s | raw sum_k 16 k S_X | held vacuum sum_k 16 k S_X(P_-(h)) | S (their difference) |")
    W("|---|---|---|---|---|")
    for t in [14.0, 20.0, 30.0] + READ_CLOCKS:
        i = ci[t]
        for j, lab in enumerate(CUTS):
            W("| %g | %s | %s | %s | %s |" % (t, lab, f(B["S_X_raw_w"][i][j], "%.5f"), f(B["S_X_vac_w"][i][j], "%.5f"), f(B["SUM"][i][j], "%.5e")))
    W("")
    # ------------------------------------------------------------ T1b: the other error sources per pair
    W("### T1b. The other error sources per pair (nats): the 16-sub-step control (kappa 18), the record-interpolation control (kappa 1 to 8, the piecewise-constant propagation), the completion spread (kappa 1 and 2, to 40 r_s)")
    W("")
    W("D16 = 288 [x_18(16 sub-steps) - x_18(8 sub-steps)] (the kappa 18 term's change when the sub-step is halved; 18 |D16| its share if every channel from kappa 12 carried it); the record-interpolation column is the production's kappa 1 to 8 partial sum minus the piecewise-constant control's (page_history_v6n) at the same pair; the completion column is the spread (largest minus smallest) of the kappa 1 and 2 weighted excess over the three histories of T4 (B2 and W3's two completions), present only to 40 r_s.")
    W("")
    W("| T/r_s | cut r_s | S (K = 18) | 288 x_18 (8 sub) | 288 x_18 (16 sub) | D16 | 18 \\|D16\\|/\\|S\\| (percent) | prod - ctrl, kappa 1 to 8 | ctrl - prod over \\|S\\| (percent) | completion spread (kappa 1, 2) | spread over \\|S\\| (percent) |")
    W("|---|---|---|---|---|---|---|---|---|---|---|")
    icc = {t: i for i, t in enumerate(IC["clocks"])}
    cpc = {t: i for i, t in enumerate(CP["clocks"])}
    for t in PRE_CLOCKS + READ_CLOCKS:
        for j, lab in enumerate(CUTS):
            i = ci[t]; S = B["SUM"][i][j]
            d16 = SB["D16_w"][i][j]; x8 = SB["WX18"][i][j]; x16 = SB["x_sub16"][i][j]
            x16w = None if x16 is None else 288.0 * x16
            ie = IC["error_prod_minus_ctrl"][icc[t]][j] if t in icc else None
            sp = CP["spread"]["SUM12_max_minus_min_all3"][cpc[t]][j] if t in cpc else None
            W("| %g | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (t, lab, f(S, "%.5e"), f(x8, "%.3e"), f(x16w, "%.3e"), f(d16, "%.2e"), "-" if d16 is None or not S else "%.3f" % (100.0 * 18.0 * abs(d16) / abs(S)), f(ie, "%.3e"), "-" if ie is None or not S else "%.2f" % (100.0 * (-ie) / abs(S)), f(sp, "%.3e"), "-" if sp is None or not S else "%.1f" % (100.0 * sp / abs(S))))
    W("")
    W("The 16-sub-step control of kappa 18 (B2): %s; its rows reach T/r_s %s (%s of the 52 checkpoints); wall %s h so far." % ("complete to 100 r_s" if SB["final"] else "NOT complete (pending): the run is in progress", f(SB["last_clock"], "%g"), len(SB["clocks_present"]), f((SB.get("wall_s") or 0) / 3600.0, "%.2f")))
    # ------------------------------------------------------------ T2: the kappa convergence
    W("")
    W("### T2. The channel sum's convergence in kappa at the readout pairs: the partial sums, the last three terms, the probes kappa 24, 32, 48 and the tail")
    W("")
    W("Partial sums S_K = sum_{k <= K} 16 k x_k. The probes are single-channel runs of the same production line (--kappas 24-24, 32-32, 48-48, --no-kappa-extend, --lean) on the same record; their terms 16 K x_K are listed with the kappa 18 term. r_a = (|512 x_32|/|384 x_24|)^(1/8), r_b = (|768 x_48|/|512 x_32|)^(1/16) are the per-channel ratios a geometric tail would need below 1 with one sign; 'same sign' says whether the kappa 18, 24, 32, 48 terms share a sign; the slope is the least-squares exponent of |16 kappa x_kappa| against kappa over those four channels where they do (a geometric fall would steepen with kappa; a power law gives a fixed slope). The tail beyond kappa 18 is not bounded by any stored term: OPEN at every pair.")
    W("")
    W("| T/r_s | cut r_s | S_2 | S_8 | S_12 | S_18 | 256 x_16 | 272 x_17 | 288 x_18 | 384 x_24 | 512 x_32 | 768 x_48 | r_a | r_b | same sign | slope | c | tail |")
    W("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t in READ_CLOCKS:
        i = ci[t]
        for j, lab in enumerate(CUTS):
            P = B["partial"]; WX = B["WX"]
            W("| %g | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | OPEN |" % (t, lab, f(P[1][i][j], "%.4e"), f(P[7][i][j], "%.4e"), f(P[11][i][j], "%.4e"), f(P[17][i][j], "%.5e"), f(WX[15][i][j], "%+.2e"), f(WX[16][i][j], "%+.2e"), f(WX[17][i][j], "%+.2e"),
                                                                                                       f(PR["WX"][1][i][j], "%+.2e"), f(PR["WX"][2][i][j], "%+.2e"), f(PR["WX"][3][i][j], "%+.2e"), f(PR["r_a"][i][j], "%.3f"), f(PR["r_b"][i][j], "%.3f"), "yes" if PR["same_sign"][i][j] else "no", f(PR["slope"][i][j], "%.2f"), f(B["C"][i][j], "%.2e")))
    W("")
    # the per-step ratios' medians over the pairs with |288 x_18| > 5e-4 at the readout clocks
    big = [(t, j) for t in READ_CLOCKS for j in range(5) if abs(B["WX"][17][ci[t]][j]) > 5e-4]
    ra = [PR["r_a"][ci[t]][j] for t, j in big]; rb = [PR["r_b"][ci[t]][j] for t, j in big]
    ss = [PR["same_sign"][ci[t]][j] for t, j in big]
    med = lambda v: sorted(v)[len(v) // 2] if v else None
    W("Over the %d pairs at the readout clocks 40 to 100 r_s (every 10) where |288 x_18| exceeds 5e-4: r_a %s (median %s), r_b %s (median %s); the four terms share a sign at %d of them." % (len(big), rng(ra, "%.3f"), f(med(ra), "%.3f"), rng(rb, "%.3f"), f(med(rb), "%.3f"), sum(1 for s in ss if s)))
    ratios48 = [PR["ratio_48_18"][ci[t]][j] for t, j in big if PR["same_sign"][ci[t]][j]]
    W("Where the signs agree, the kappa 48 term is %s of the kappa 18 term; the kappa 24 term %s; the kappa 32 term %s." % (rng(ratios48, "%.2f"), rng([PR["ratio_24_18"][ci[t]][j] for t, j in big if PR["same_sign"][ci[t]][j]], "%.2f"), rng([PR["ratio_32_18"][ci[t]][j] for t, j in big if PR["same_sign"][ci[t]][j]], "%.2f")))
    W("")
    W("### T2b. Every channel's term at three pairs (the figure's panel c): 16 kappa x_kappa, its propagation floor 16 kappa F_meas,kappa, and the formed-energy term 16 kappa E_kappa")
    W("")
    for (t, lab) in ((100.0, "24"), (80.0, "30"), (100.0, "2")):
        i = ci[t]; j = CUTS.index(lab)
        W("At T/r_s %g, the cut %s r_s (S_18 = %s nats; c = %s; F_meas = %s):" % (t, lab, f(B["SUM"][i][j], "%.5e"), f(B["C"][i][j], "%.2e"), f(B["FMW"][i][j], "%.2e")))
        W("")
        W("| kappa | 16 kappa x_kappa | 16 kappa F_meas,kappa | resolved | 16 kappa E_kappa | partial sum S_kappa |")
        W("|---|---|---|---|---|---|")
        for a, k in enumerate(B["kappas"]):
            W("| %d | %s | %s | %s | %s | %s |" % (k, f(B["WX"][a][i][j], "%+.3e"), f(B["fmeas_w"][a][i][j], "%.1e"), "yes" if B["resolved"][a][i][j] else "no", f(B["E_kappa_w"][a][i], "%.2e"), f(B["partial"][a][i][j], "%.5e")))
        for b, K in enumerate((24, 32, 48)):
            W("| %d (probe) | %s | %s | %s | %s | - |" % (K, f(PR["WX"][b + 1][i][j], "%+.3e"), f(PR["fmeas_w"][b][i][j], "%.1e"), "yes" if PR["resolved"][b][i][j] else "no", f(PR["E_kappa_w"][str(K)][i], "%.2e")))
        W("")
    # ------------------------------------------------------------ T3: the two scales
    W("### T3. The two averaging scales side by side at K = 18: the sigma/a 4 history (B2) and the sigma/a 8 history (s8), 14 to 40 r_s")
    W("")
    W("The sigma/a 8 history is a different object printed beside, never averaged with B2 (its record ends at 40 r_s; its formation record clock is %s with R_t %s a; B2's %s with R_t %s a). Both sums are the v1f post-pass's at K = 18 (the s8 composition is this lane's rerun of the busi post-pass on the 18 s8 channels; the row at 40 is its from-40 run, the rows before it its from-14 run)." % (f(MK["s8"]["formation_record_clock_over_rs"], "%g"), f(MK["s8"]["formation_R_t_a"], "%.2f"), f(MK["formation_record_clock_over_rs"], "%g"), f(MK["formation_R_t_a"], "%.2f")))
    W("")
    W("| T/r_s | cut r_s | B2: S | F_meas | K c | verdict | s8: S | F_meas | K c | verdict | B2: 288 E_18 | s8: 288 E_18 |")
    W("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t in [14.0, 16.0, 20.0, 30.0, 40.0]:
        for lab in CUTS:
            r = RB.get((t, lab)); q = R8.get((t, lab))
            if r is None or q is None:
                continue
            W("| %g | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (t, lab, f(r["S"], "%.4e"), f(r["F_meas"], "%.2e"), f(r["share"], "%.2e"), r["verdict"], f(q["S"], "%.4e"), f(q["F_meas"], "%.2e"), f(q["share"], "%.2e"), q["verdict"], f(B["E_kappa_w"][17][ci[t]], "%.2e"), f(S8["E_kappa_w"][17][ci8[t]], "%.2e")))
    W("")
    tl8 = {}
    for (t, lab), r in R8.items():
        tl8.setdefault(lab, {})[r["verdict"]] = tl8.setdefault(lab, {}).get(r["verdict"], 0) + 1
    W("s8 tally over its 15 clocks 14 to 40 r_s (not formed / within the floor / converged / formed, tail open): " + "; ".join("%s r_s: %d / %d / %d / %d" % (lab, tl8[lab].get("not formed", 0), tl8[lab].get("within the floor", 0), tl8[lab].get("converged", 0), tl8[lab].get("formed, tail open", 0)) for lab in CUTS) + ".")
    i8 = ci8[40.0]
    W("s8's weighted energy terms 16 kappa E_kappa at 40 r_s over kappa 1 to 18: " + ", ".join("%d: %s" % (k, f(S8["E_kappa_w"][a][i8], "%.2e")) for a, k in enumerate(S8["kappas"])) + " (B2's at 40 r_s over kappa 12 to 18: " + ", ".join("%d: %s" % (k, f(B["E_kappa_w"][a][ci[40.0]], "%.2e")) for a, k in enumerate(B["kappas"]) if k >= 12) + ").")
    W("")
    W("The sigma/a 8 history's collar flux (kappa 1, 2; L per coordinate interval; the comparator where the record has a root):")
    W("")
    W("| T/r_s | s8: E | R_t (a) | min(N - \\|v\\|) | L (kappa 1, 2) | L_H(kappa_T) | L/L_H(kappa_T) | L/L_H(T_H) | E_inv (kappa 1, 2) | collar flow acc. (kappa 1, 2) | join max rel dv |")
    W("|---|---|---|---|---|---|---|---|---|---|---|")
    for t in [14.0, 15.0, 16.0, 20.0, 30.0, 36.0, 38.0, 40.0]:
        if t not in ci8:
            continue
        i = ci8[t]
        L12 = (S8["L_per_channel"][0][i] or 0.0) + (S8["L_per_channel"][1][i] or 0.0) if (S8["L_per_channel"][0][i] is not None and S8["L_per_channel"][1][i] is not None) else None
        LHk = (S8["LH_kT_per_channel"][0][i] + S8["LH_kT_per_channel"][1][i]) if (S8["LH_kT_per_channel"][0][i] is not None and S8["LH_kT_per_channel"][1][i] is not None) else None
        LHt = (S8["LH_TH_per_channel"][0][i] + S8["LH_TH_per_channel"][1][i]) if (S8["LH_TH_per_channel"][0][i] is not None and S8["LH_TH_per_channel"][1][i] is not None) else None
        Ei = (S8["E_kappa_w"][0][i] or 0.0) + (S8["E_kappa_w"][1][i] or 0.0); fl = (S8["flow_w"][0][i] or 0.0) + (S8["flow_w"][1][i] or 0.0)
        W("| %g | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (t, f(S8["E"][i], "%.4f"), f(S8["R_t"][i], "%.3f"), f(S8["minNv"][i], "%+.4f"), f(L12, "%+.3e"), f(LHk, "%.3e"), f((L12 / LHk) if (L12 is not None and LHk) else None, "%+.4f"), f((L12 / LHt) if (L12 is not None and LHt) else None, "%+.4f"), f(Ei, "%.4e"), f(fl, "%+.4e"), f(S8["join_dv"][i], "%.2e")))
    W("")
    W("The s8 16-sub-step control (kappa 14, to 40 r_s, %s): D16 = 224 [x_14(16 sub) - x_14(8 sub)] against the kappa 14 term 224 x_14, and the formed energy 224 E_14 under both sub-steps:" % ("complete" if SS["final"] else "NOT complete"))
    W("")
    W("| T/r_s | cut r_s | 224 x_14 (8 sub) | 224 x_14 (16 sub) | D16 | D16 / (224 x_14) | s8 S_18 | 18 \\|D16\\|/\\|S_18\\| (percent) | 224 E_14 (8 sub) | 224 E_14 (16 sub) |")
    W("|---|---|---|---|---|---|---|---|---|---|")
    for t in [14.0, 16.0, 20.0, 30.0, 40.0]:
        i = ci8[t]
        for j, lab in enumerate(CUTS):
            x8 = SS["WX14"][i][j]; x16 = SS["x_sub16"][i][j]; d = SS["D16_w"][i][j]; S = S8["SUM"][i][j]
            W("| %g | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (t, lab, f(x8, "%+.3e"), f(None if x16 is None else 224.0 * x16, "%+.3e"), f(d, "%+.2e"), "-" if (d is None or not x8) else "%.3f" % (d / x8), f(S, "%.4e"), "-" if (d is None or not S) else "%.2f" % (100.0 * 18.0 * abs(d) / abs(S)), f(SS["E_kappa_w_sub8"][i], "%.2e"), f(SS["E_kappa_w_sub16"][i], "%.2e")))
    W("")
    # ------------------------------------------------------------ T4: the completion spread
    W("### T4. The completion spread: the kappa 1 and 2 channels (weights 16 and 32) on B2 and on W3's two completions, to 40 r_s")
    W("")
    names = list(CP["runs"].keys())
    for nm in names:
        h = CP["runs"][nm]
        W("- %s: engine sites %s, records %s; the record's label (end): %s" % (nm, h.get("history_sites"), h.get("history_records"), (h.get("history_label") or "")[-520:].replace("|", "/").replace("frozen-sources", "held-sources (the run tag W3fs)").replace("the sources frozen inside", "the sources held inside")))
    W("")
    W("| T/r_s | cut r_s | B2: 16 x_1 + 32 x_2 | held-sources (W3fs) | paired-avg (W3pa) | spread (all three) | spread (the two completions) |")
    W("|---|---|---|---|---|---|---|")
    for t in [14.0, 16.0, 20.0, 30.0, 40.0]:
        i = cpc[t]
        for j, lab in enumerate(CUTS):
            W("| %g | %s | %s | %s | %s | %s | %s |" % (t, lab, f(CP["runs"][names[0]]["SUM12"][i][j], "%+.4e"), f(CP["runs"]["held-sources (W3fs)"]["SUM12"][i][j], "%+.4e"), f(CP["runs"]["paired-avg (W3pa)"]["SUM12"][i][j], "%+.4e"), f(CP["spread"]["SUM12_max_minus_min_all3"][i][j], "%.3e"), f(CP["spread"]["SUM12_max_minus_min_two_completions"][i][j], "%.3e")))
    W("")
    W("| T/r_s | run | E (charge) | R_t (a) | min(N - \\|v\\|) | E_inv (kappa 1, 2) | collar flow acc. (kappa 1, 2) | L (kappa 1, 2) | L_H(kappa_T) | join max dN | join max rel dv |")
    W("|---|---|---|---|---|---|---|---|---|---|---|")
    for t in [12.0, 14.0, 16.0, 20.0, 30.0, 40.0]:
        i = cpc[t]
        for nm in names:
            h = CP["runs"][nm]
            W("| %g | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (t, nm.split(" (")[0], f(h["E"][i], "%.4f"), f(h["R_t"][i], "%.3f"), f(h["minNv"][i], "%+.4f"), f(h["E_inv12"][i], "%.4e"), f(h["flow12"][i], "%+.4e"), f(h["L12"][i], "%+.3e"), f(h["LH12_kT"][i], "%.3e"), f(h["join_dN"][i], "%.2e"), f(h["join_dv"][i], "%.2e")))
    W("")
    # ------------------------------------------------------------ T5: the flux
    W("### T5. The collar flux of the kappa 1 and 2 channels on B2 to 100 r_s, with the kappa 3 guard's change and the K = 18 sum beside")
    W("")
    W("L is the collar energy flow per coordinate interval Delta T (the increment of the stored accumulator over the checkpoint interval divided by its length; identity checked to %s relative), placed at the collar's recordwise stationary-chart clock t_inf (t_inf - T = %s to %s r_s over the run); L_H(kappa_T) and L_H(T_H) are the channel set's continuum greybody comparators at the record's dense root and slope and at 1/(8 pi G E), not an observed thermal flux. The ratios with kappa 3 add the truncation guard; the K = 18 ratio sums every channel (the high channels' flows carry the moving exterior's first-order term, W1 ruling S3). 'v6b' is version 3's two-channel ratio on the piecewise-constant propagation of the r11 record (through 40), the propagation variant beside." % (f(CH["L_interval_identity_max_rel"], "%.1e"), f(MK["collar_clock_offset_over_rs"]["min"], "%.3f"), f(MK["collar_clock_offset_over_rs"]["max"], "%.3f")))
    W("")
    W("| T/r_s | t_inf/r_s | E | L (kappa 1, 2) | L_H(kappa_T) | L_H(T_H) | L/L_H(kappa_T) | L/L_H(T_H) | with kappa 3: L/L_H(kappa_T) | L/L_H(T_H) | K = 18: L/L_H(kappa_T) | E_inv/E (kappa 1, 2) | E_inv/E (kappa 1 to 3) | E_inv/E (K = 18) | join max rel dv | v6b L/L_H(kappa_T) |")
    W("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    v3 = D["v3_flux"]; v3i = {round(t, 4): i for i, t in enumerate(v3["t_over_rs"])}
    for t in [14.0, 15.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 34.0, 36.0, 38.0, 40.0, 42.0, 44.0, 46.0, 48.0, 50.0, 52.0, 54.0, 56.0, 58.0, 60.0, 64.0, 70.0, 76.0, 80.0, 86.0, 90.0, 96.0, 100.0]:
        if t not in ci:
            continue
        i = ci[t]
        v3r = v3["L_over_LH_kappaT"][v3i[t]] if t in v3i else None
        W("| %g | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (t, f(FL["t_inf_collar_over_rs"][i], "%.3f"), f(FL["E"][i], "%.3f"), f(FL["L12"][i], "%+.3e"), f(FL["LH12_kT"][i], "%.3e"), f(FL["LH12_TH"][i], "%.3e"), f(FL["ratio12_kT"][i], "%+.4f"), f(FL["ratio12_TH"][i], "%+.4f"), f(FL["ratio123_kT"][i], "%+.4f"), f(FL["ratio123_TH"][i], "%+.4f"), f(FL["ratio18_kT"][i], "%+.4f"),
                                                                                                 pct(FL["E_inv12_over_E"][i]), pct(FL["E_inv123"][i] / FL["E"][i] if FL["E"][i] else None), pct(FL["E_inv18_over_E"][i]), f(FL["join_dv"][i], "%.2e"), f(v3r, "%+.4f")))
    W("")
    i100 = ci[100.0]; i40 = ci[40.0]
    W("The accumulated collar flow of kappa 1 and 2 over E at 40 r_s: %s; at 100 r_s: %s (the K = 18 accumulator over E at 100 r_s: %s). E_inv (kappa 1, 2) over E: %s at 14 r_s, %s over 15 to 40, %s over 42 to 100 (K = 18: %s at 14, %s over 42 to 100). The flux ratio (kappa 1, 2, slope comparator): %s over 14 to 34 (negative at %d of the %d checkpoints 14 to 34), %s over 36 to 40, %s over 42 to 100 (negative at %d of %d)." % (
        f(FL["E_rad0_over_E_12"][i40], "%.3e"), f(FL["E_rad0_over_E_12"][i100], "%.3e"), f(FL["E_rad0_over_E_18"][i100], "%.3e"), pct(FL["E_inv12_over_E"][ci[14.0]]) + " percent", rng([100.0 * FL["E_inv12_over_E"][ci[t]] for t in cl if 15.0 - 1e-9 <= t <= 40.0 + 1e-9], "%.2f") + " percent", rng([100.0 * FL["E_inv12_over_E"][ci[t]] for t in cl if t >= 42.0 - 1e-9], "%.2f") + " percent",
        pct(FL["E_inv18_over_E"][ci[14.0]]) + " percent", rng([100.0 * FL["E_inv18_over_E"][ci[t]] for t in cl if t >= 42.0 - 1e-9], "%.2f") + " percent",
        rng([FL["ratio12_kT"][ci[t]] for t in cl if 14.0 - 1e-9 <= t <= 34.0 + 1e-9], "%+.3f"), sum(1 for t in cl if 14.0 - 1e-9 <= t <= 34.0 + 1e-9 and FL["L12"][ci[t]] is not None and FL["L12"][ci[t]] < 0), sum(1 for t in cl if 14.0 - 1e-9 <= t <= 34.0 + 1e-9),
        rng([FL["ratio12_kT"][ci[t]] for t in cl if 36.0 - 1e-9 <= t <= 40.0 + 1e-9], "%+.3f"), rng([FL["ratio12_kT"][ci[t]] for t in cl if t >= 42.0 - 1e-9], "%+.3f"), sum(1 for t in cl if t >= 42.0 - 1e-9 and FL["L12"][ci[t]] < 0), sum(1 for t in cl if t >= 42.0 - 1e-9)))
    W("")
    W("E_out/E_kappa (the exterior share of the formed energy beyond the collar) per channel at 100 r_s: " + ", ".join("%d: %s" % (k, f(FL["E_out_over_E_kappa"][a][i100], "%.3f")) for a, k in enumerate(B["kappas"])) + ".")
    W("")
    W("| kappa | 16 kappa E_kappa at 100 | 16 kappa E_out at 100 | accumulated collar flow at 100 (weighted) | L at 100 (weighted) | L_H(kappa_T) at 100 |")
    W("|---|---|---|---|---|---|")
    for a, k in enumerate(B["kappas"]):
        W("| %d | %s | %s | %s | %s | %s |" % (k, f(FL["E_inv_per_channel"][a][i100], "%.3e"), f(FL["E_out_per_channel"][a][i100], "%.3e"), f(FL["flow_per_channel"][a][i100], "%+.3e"), f(FL["L_per_channel"][a][i100], "%+.3e"), f(FL["LH_kT_per_channel"][a][i100], "%.2e")))
    W("")
    SP = D["collar_split"]
    W("The signed collar split per checkpoint (kappa 1, 2; the collar_split block as stored: the energy flux through the collar window at the checkpoint split by the radial current's sign and by the threshold 25 T_H^ref; window %s sites about the site %s; weights 16 and 32 summed; the ratio |in|/|out| beside):" % (SP["definition"]["L"], SP["definition"]["centre_site"]))
    W("")
    W("| T/r_s | E flux out (kappa 1, 2) | E flux in | \\|in\\|/\\|out\\| | out soft | in soft | out hard | in hard | kappa 1: out, in | kappa 2: out, in |")
    W("|---|---|---|---|---|---|---|---|---|---|")
    spi = {t: i for i, t in enumerate(SP["clocks"])}
    for t in [14.0, 20.0, 30.0, 36.0, 38.0, 40.0, 42.0, 44.0, 46.0, 48.0, 50.0, 54.0, 60.0, 70.0, 80.0, 90.0, 100.0]:
        if t not in spi:
            continue
        i = spi[t]; o = SP["E_flux_out_12"][i]; n = SP["E_flux_in_12"][i]
        W("| %g | %s | %s | %s | %s | %s | %s | %s | %s, %s | %s, %s |" % (t, f(o, "%+.3e"), f(n, "%+.3e"), "-" if (o is None or n is None or not o) else "%.3f" % (abs(n) / abs(o)), f(SP["E_flux_out_soft_12"][i], "%+.2e"), f(SP["E_flux_in_soft_12"][i], "%+.2e"), f(SP["E_flux_out_hard_12"][i], "%+.2e"), f(SP["E_flux_in_hard_12"][i], "%+.2e"), f(SP["E_flux_out_k1"][i], "%+.2e"), f(SP["E_flux_in_k1"][i], "%+.2e"), f(SP["E_flux_out_k2"][i], "%+.2e"), f(SP["E_flux_in_k2"][i], "%+.2e")))
    W("")
    rat = [abs(SP["E_flux_in_12"][i]) / abs(SP["E_flux_out_12"][i]) for i, t in enumerate(SP["clocks"]) if t >= 40.0 - 1e-9 and SP["E_flux_out_12"][i]]
    W("Over the checkpoints 40 to 100 r_s the ratio |in|/|out| of the kappa 1, 2 split runs %s (%d of %d checkpoints below 0.01)." % (rng(rat, "%.3f"), sum(1 for r in rat if r < 0.01), len(rat)))
    W("")
    # ------------------------------------------------------------ T6: the marks
    W("### T6. The marks")
    W("")
    W("| mark | value | how obtained |")
    W("|---|---|---|")
    W("| the formation's record clock | T/r_s %g, R_t %.3f a, E %.4f (2gE_f %.4f a) | the B2 record's first record with a trapping root |" % (MK["formation_record_clock_over_rs"], MK["formation_R_t_a"], MK["formation_E"], MK["two_gE_formation"]))
    W("| the trapped episodes of B2 | %s | runs of consecutive records with a root, from the record json (%d records to 100 r_s) |" % ("; ".join("%g to %g r_s (%d records, R_t %.2f to %.2f a)" % (e["T_start"], e["T_end"], e["n_records"], e["R_t_min"], e["R_t_max"]) for e in MK["episodes_B2"]), MK["n_records_B2"]))
    W("| R_t over 2gE | %s | the record json |" % "; ".join("%s r_s: %.3f (R_t %.3f a, E/E0 %.4f)" % (t, v["R_t_over_2gE"], v["R_t"], v["E_over_E0"]) for t, v in MK["R_t_at"].items()))
    W("| T_d | %g r_s | %s |" % (MK["T_d_over_rs"], MK["T_d_source"].replace("|", "/")))
    for lab in ("16", "24", "48"):
        e = MK["light_travel"][lab]
        W("| the formation's light at the %s r_s radius (fixed-charge flight comparison from 2 and 1.5 x 2gE_f) | record clock %.2f to %.2f r_s (collar-type clock at that radius %.2f to %.2f); plus T_d: %.2f to %.2f | 2 r_s,f [F(x1) - F(x0)], F(x) = x^2/2 + x + ln(x - 1), x = (R/r_s,f)^(1/2), r_s,f = 2gE_f at the formation record, from c = 2 and 1.5 to R; the version 3 formula (travel over 2gE_f: %.4f and %.4f) |" % (lab, e["2.0"]["arrival_record_clock_over_rs"], e["1.5"]["arrival_record_clock_over_rs"], e["2.0"]["arrival_collar_clock_over_rs"], e["1.5"]["arrival_collar_clock_over_rs"], e["ray_map_end_record_clock_over_rs"]["2.0"], e["ray_map_end_record_clock_over_rs"]["1.5"], e["2.0"]["travel_over_2gE_f"], e["1.5"]["travel_over_2gE_f"]))
    hf = MK["half_flow"]
    W("| the half-collar-flow clock (kappa 1, 2) through 100 r_s | %s r_s (between the checkpoints %s and %s; the accumulated flow at 100: %s) | %s |" % (f(hf["through_100_kappa12"]["record_clock_over_rs"], "%.2f"), f(hf["through_100_kappa12"]["between"][0], "%g"), f(hf["through_100_kappa12"]["between"][1], "%g"), f(hf["through_100_kappa12"]["acc_at_100"], "%.4e"), hf["definition"]))
    W("| the half-collar-flow clock (kappa 1, 2) through 40 r_s | %s r_s (between %s and %s; the flow at 40: %s); version 3's on the s4 run (per-record flows, v6b): %s on the record clock, %s on the collar clock | the same, through 40 only |" % (f(hf["through_40_kappa12"]["record_clock_over_rs"], "%.2f"), f(hf["through_40_kappa12"]["between"][0], "%g"), f(hf["through_40_kappa12"]["between"][1], "%g"), f(hf["through_40_kappa12"]["acc_at_40"], "%.4e"), f(hf["v3_through_40_s4_run"]["record_clock_over_rs"], "%.3f"), f(hf["v3_through_40_s4_run"]["collar_clock_over_rs"], "%.3f")))
    W("| the first return (estimate) | %s | %s |" % ("; ".join("cut %s r_s: %.1f r_s" % (lab, v["first_return_rs"]) for lab, v in D13["first_return_at_100"].items()), D13["first_return_label"].replace("|", "/")))
    W("| the sigma/a 8 record | formation %g r_s at R_t %.2f a; episodes %s; E0 %.4f | the s8 record json |" % (MK["s8"]["formation_record_clock_over_rs"], MK["s8"]["formation_R_t_a"], "; ".join("%g to %g (%d records)" % (e["T_start"], e["T_end"], e["n_records"]) for e in MK["s8"]["episodes"]), MK["s8"]["E0"]))
    W("")
    # ------------------------------------------------------------ T7: D13
    W("### T7. The D13 protocol's stored observables on B2 (kappa 1 and 2, weights 16 and 32), rows before the first return")
    W("")
    W("(a) The detector-energy ledger at the cuts 16 and 30 r_s per window (the windows ending at the readout clocks): %s" % D13["windows_definition"])
    W("")
    W("| window T/r_s | cut | u_start, u_end (r_s) | inventory_D start | inventory_D end | second term (D end - D start) | crossing current (C) | crossing jumps (C) | vacuum change | kappa 1 second term | kappa 2 second term | with kappa 3: second term |")
    W("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for w in D13["windows"]:
        if w["t1"] in READ_CLOCKS:
            W("| %g to %g | %s | %s, %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (w["t0"], w["t1"], w["cut"], f(w["u_start"], "%.1f"), f(w["u_end"], "%.1f"), f(w["inventory_D_start_12"], "%+.3e"), f(w["inventory_D_end_12"], "%+.3e"), f(w["second_term_D_12"], "%+.3e"), f(w["crossing_current_12"], "%+.3e"), f(w["crossing_jumps_12"], "%+.3e"), f(w["vacuum_change_12"], "%+.3e"), f(w["second_term_D_k1"], "%+.3e"), f(w["second_term_D_k2"], "%+.3e"), f(w["second_term_D_123"], "%+.3e")))
    W("")
    W("The deviation's inventory between the cuts 16 and 30 r_s at the window's start (kappa 1, 2; with kappa 3): " + "; ".join("%g: %s (%s)" % (b["t0"], f(b["inventory_D_between_12"], "%+.3e"), f(b["inventory_D_between_123"], "%+.3e")) for b in D13["between"] if b["t1"] in READ_CLOCKS) + ".")
    W("")
    W("(b) L_old and L_sea by the 16 back-propagated probes at the cuts 16 and 30 r_s: %s. L_H(kappa_T) of kappa 1, 2 at the clock beside; before the first return at every row (the first return at the 16 r_s cut is estimated at %.1f r_s, at 30 r_s %.1f)." % (D13["probes_definition"].replace("|", "/"), D13["first_return_at_100"]["16"]["first_return_rs"], D13["first_return_at_100"]["30"]["first_return_rs"]))
    W("")
    W("| T/r_s | cut | L (direct, kappa 1, 2) | t_ref | L_old | L_sea | L_src | kappa 1: L_old, L_sea | kappa 2: L_old, L_sea | L_H(kappa_T) (kappa 1, 2) | join max rel dv |")
    W("|---|---|---|---|---|---|---|---|---|---|---|")
    for p in D13["probes"]:
        if p["t"] in (40.0, 48.0, 56.0, 64.0, 72.0, 80.0, 88.0, 96.0, 100.0):
            for r in p["refs"]:
                W("| %g | %s | %s | %g | %s | %s | %s | %s, %s | %s, %s | %s | %s |" % (p["t"], p["cut"], f(p["L_direct_12"], "%+.3e"), r["t_ref"], f(r["L_old_12"], "%+.3e"), f(r["L_sea_12"], "%+.3e"), f(r["L_src_12"], "%+.3e"), f(r["L_old_k1"], "%+.2e"), f(r["L_sea_k1"], "%+.2e"), f(r["L_old_k2"], "%+.2e"), f(r["L_sea_k2"], "%+.2e"), f(FL["LH12_kT"][ci[p["t"]]], "%.3e"), f(FL["join_dv"][ci[p["t"]]], "%.2e")))
    W("")
    W("(c) The hard fluence per retarded window (the windows ending at the readout clocks): %s; the threshold 25 T_H^ref = %s (T_H^ref %s)." % (D13["hard_definition"], f(D13["hard_threshold_energy_at_40"], "%.4e"), f(D13["T_H_ref"], "%.4e")))
    W("")
    W("| window T/r_s | u_start, u_end | hard E out | hard E in | soft E out | soft E in | above-band E out | above-band E in | hard N out | hard E out (held vacuum) | soft E out (held vacuum) | with kappa 3: hard E out, soft E out |")
    W("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for h in D13["hard"]:
        if h["t1"] in READ_CLOCKS:
            W("| %g to %g | %s, %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s, %s |" % (h["t0"], h["t1"], f(h["u_start"], "%.1f"), f(h["u_end"], "%.1f"), f(h["delta_hard_E_out_12"], "%+.3e"), f(h["delta_hard_E_in_12"], "%+.3e"), f(h["delta_soft_E_out_12"], "%+.3e"), f(h["delta_soft_E_in_12"], "%+.3e"), f(h["delta_above_E_out_12"], "%+.2e"), f(h["delta_above_E_in_12"], "%+.2e"), f(h["delta_hard_N_out_12"], "%+.3e"), f(h["delta_hard_E_out_P_12"], "%+.2e"), f(h["delta_soft_E_out_P_12"], "%+.2e"), f(h["delta_hard_E_out_123"], "%+.3e"), f(h["delta_soft_E_out_123"], "%+.3e")))
    W("")
    W("The identified source modes (the eigenvectors of C inside the region at the reference clock with eigenvalue at least 0.99, propagated with the sea without collection) and their same-packet transfer tau and inferred occupation at the collar (16 r_s), the cut (24 r_s) and the far radius (48 r_s), per channel:")
    W("")
    W("| kappa | reference clock (region, modes) | at T/r_s | tau sum collar | tau max collar | occupation (collar) | tau sum cut | occupation (cut) | tau sum far | occupation (far) |")
    W("|---|---|---|---|---|---|---|---|---|---|")
    for k in ("1", "2"):
        defs = D13["sources"][k + "_definition"]
        for t, refs in D13["sources"][k].items():
            for ref, v in refs.items():
                dd = defs.get(ref) or {}
                W("| %s | %g (%s; %s modes) | %g | %s | %s | %s | %s | %s | %s | %s |" % (k, v["t_ref_over_rs"], dd.get("region"), v["n_modes"], float(t), f(v["tau_sum_collar"], "%.3f"), f(v["tau_max_collar"], "%.4f"), f(v["occ_inferred_mean_collar"], "%.4f"), f(v["tau_sum_cut"], "%.3f"), f(v["occ_inferred_mean_cut"], "%.4f"), f(v["tau_sum_far"], "%.3e"), f(v["occ_inferred_mean_far"], "%.4f")))
    W("")
    W("The enclosed Killing energy of the deviation within the outer root at 100 r_s (kappa 1): I_L %s, M_root %s, I_L/M_root %s, delta R_L/R %s (%s)." % (f(D13["enclosed_at_100_k1"]["I_L"], "%.3e"), f(D13["enclosed_at_100_k1"]["M_root"], "%.3f"), f(D13["enclosed_at_100_k1"]["I_L_over_M_root"], "%.2e"), f(D13["enclosed_at_100_k1"]["delta_R_L_over_R"], "%.2e"), (D13["enclosed_at_100_k1"]["label"] or "").replace("|", "/")))
    W("")
    # ------------------------------------------------------------ T8: the record-interpolation control and run (a)
    W("### T8. The record-interpolation control (kappa 1 to 8): the production's partial sum against the piecewise-constant propagation's (page_history_v6n), and the quench energy")
    W("")
    W("| T/r_s | cut r_s | production S_8 | control S_8 | prod - ctrl | 16 kappa E_kappa, kappa 1 to 8: production | control | quench E_ctrl - E_prod |")
    W("|---|---|---|---|---|---|---|---|")
    for t in [14.0, 20.0, 30.0, 40.0, 60.0, 80.0, 100.0]:
        if t not in icc:
            continue
        i = icc[t]
        for j, lab in enumerate(CUTS):
            W("| %g | %s | %s | %s | %s | %s | %s | %s |" % (t, lab, f(IC["excess_prod_1to8"][i][j], "%.4e"), f(IC["excess_ctrl_1to8"][i][j], "%.4e"), f(IC["error_prod_minus_ctrl"][i][j], "%+.3e"), f(IC["E_prod_1to8"][i], "%.4e"), f(IC["E_ctrl_1to8"][i], "%.4e"), f(IC["E_quench"][i], "%.3e")))
    W("")
    W("The quench per channel (16 kappa (E_ctrl - E_prod) at 100 r_s): " + ", ".join("%s: %s" % (k, f((v or {}).get("100.0"), "%.2e")) for k, v in (IC.get("quench_per_channel") or {}).items()) + ".")
    W("")
    if D.get("runA"):
        W("Run (a) (the vacuum's cut entropy under a sub-step field shift, kappa 18; the busi lane's ruling2/runA_v1_results.jsonl), as stored, first rows: " + "; ".join(json.dumps(r)[:200].replace("|", "/") for r in D["runA"][:3]) + " (%d rows)." % len(D["runA"]))
        W("")
    # ------------------------------------------------------------ T9: checks
    W("### T9. Consistency checks of this composition")
    W("")
    W("- This script's sums against the post-pass rows (K = 18, B2, %d pairs): largest |difference| in S %s, F_meas %s, c %s, F_apriori %s nats." % (CH["B2_sum_vs_postpass_max_abs"]["n_pairs"], f(CH["B2_sum_vs_postpass_max_abs"]["S"], "%.1e"), f(CH["B2_sum_vs_postpass_max_abs"]["F_meas"], "%.1e"), f(CH["B2_sum_vs_postpass_max_abs"]["c"], "%.1e"), f(CH["B2_sum_vs_postpass_max_abs"]["F_apriori"], "%.1e")))
    W("- The same for s8: largest |difference| in S %s nats." % f(CH["s8_sum_vs_postpass_max_abs_S"], "%.1e"))
    W("- The verdicts at 40 to 100 r_s of the busi composition (from 40 r_s) against this lane's rerun from 14 r_s: %d pairs differ%s." % (len(CH["B2_verdict_differences_from40_vs_from14_at_t_ge_40"]), (": " + "; ".join("%g/%s %s (from 40) against %s (from 14: %s)" % (d["t"], d["cut"], d["verdict_from40"], d["verdict_from14"], d.get("why_from14")) for d in CH["B2_verdict_differences_from40_vs_from14_at_t_ge_40"])) if CH["B2_verdict_differences_from40_vs_from14_at_t_ge_40"] else ""))
    W("- s8 at 40 r_s, the from-40 run against the from-14 run: %d pairs differ%s." % (len(CH["s8_verdict_differences_from40_vs_from14_at_40"]), (": " + "; ".join("%g/%s %s against %s" % (d["t"], d["cut"], d["verdict_from40"], d["verdict_from14"]) for d in CH["s8_verdict_differences_from40_vs_from14_at_40"])) if CH["s8_verdict_differences_from40_vs_from14_at_40"] else ""))
    W("- L_interval against the accumulator's increment per coordinate interval: largest relative deviation %s." % f(CH["L_interval_identity_max_rel"], "%.1e"))
    lt = CH["light_travel_vs_v3"]
    W("- The light-travel formula against version 3's marks (the 16 r_s radius): %s and %s (this script) against %s and %s (version 3), 2gE_f %s against %s." % (f(lt["this_16_c2"], "%.6f"), f(lt["this_16_c1.5"], "%.6f"), f(lt["v3_16_c2"], "%.6f"), f(lt["v3_16_c1.5"], "%.6f"), f(lt["this_two_gE_formation"], "%.5f"), f(lt["v3_two_gE_formation"], "%.5f")))
    W("")
    with open(out, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("wrote %s (%d lines)" % (out, len(L)))


if __name__ == "__main__":
    main()
