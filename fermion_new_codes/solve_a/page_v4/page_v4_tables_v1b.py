#!/usr/bin/env python
"""page_v4_tables_v1b.py (the solve lane; PAGE_solved_history_v4.1): the table script. Loads page_v4_data_v1.json (the v1b producer's output) and
writes the markdown tables of the delivery (page_v4_tables_v1b.md) and the per-pair export (page_v4_readout_export_v1b.tsv: one row per readout clock and
cut with S, F_meas, c, K c, the last three weighted terms and their floors, q_x and its neighbours, the v1f tail, the probe terms, the geometric
criterion, the conditional majorant, the automatic verdict and the final tail status). It computes nothing new beyond ratios and ranges of stored
numbers. v1b: the corrections of the correctness reading of version 4 (the verdict column and tally at 76/24, the criterion and conditional-majorant
columns, the first-order term against E_kappa per channel, the accumulator column removed from the paper-facing channel table, the percent headings,
the 2 and 30 r_s flight marks, the ledger's agreement column, the L_old ratios, the source modes' averages and vacuum occupations, run (a)'s table).
Usage: python page_v4_tables_v1b.py DATA_JSON OUT_MD OUT_TSV"""
import json
import math
import sys

CUTS = ["2", "16", "24", "30", "48"]
READ_CLOCKS = [40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
PRE_CLOCKS = [14.0, 16.0, 20.0, 30.0]


def isnum(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def f(x, fmt="%.3e"):
    if not isnum(x):
        return "-"
    try:
        return fmt % x
    except (TypeError, ValueError):
        return str(x)


def pct(x):
    return "-" if not isnum(x) else "%.2f" % (100.0 * x)


def rng(vals, fmt="%.2e"):
    v = [x for x in vals if isnum(x)]
    return ("%s to %s" % (fmt % min(v), fmt % max(v))) if v else "-"


def final_verdict(r):
    """the composition's verdict: the v1f rule's, with its one 'converged' printed as formed, tail open (final tail status OPEN)."""
    v = r["verdict"]
    return "formed, tail open (v1f rule: converged; final tail status OPEN, T2c)" if v == "converged" else v


def main():
    D = json.load(open(sys.argv[1])); out = sys.argv[2]; out_tsv = sys.argv[3] if len(sys.argv) > 3 else None
    B = D["B2"]; cl = B["clocks"]; ci = {t: i for i, t in enumerate(cl)}
    S8 = D["s8"]; cl8 = S8["clocks"]; ci8 = {t: i for i, t in enumerate(cl8)}
    RB = {(r["t"], r["cut"]): r for r in B["readout"]}; R8 = {(r["t"], r["cut"]): r for r in S8["readout"]}
    PR = D["probes"]; SB = D["sub16_B2"]; SS = D["sub16_s8"]; CP = D["completions"]; FL = D["flux"]; MK = D["marks"]; IC = D["interp_control"]; D13 = D["d13"]; CH = D["checks"]; LV = D["levels"]; SP = D["collar_split"]
    CMS = D["common_mode_spread_12_18"]; DT = D["d_term_unweighted"]; EKU = D["E_kappa_unweighted"]
    L = []
    W = L.append
    # ------------------------------------------------------------ T1: the readout
    W("### T1. The exterior-cut excess of the B2 history summed over kappa = 1 to 18 (weights 16 kappa), nats, at the readout clocks and the cuts; its errors beside")
    W("")
    W("S = sum_k 16 k x_k (x_k the channel's excess S_X(C) - S_X(P_-(h)) at the cut); F_meas the propagation floor (the measured impurity's entropy with the truncation's theorem bound, weighted and summed); F_apriori the a priori allowance (the bound the impurity was checked against, not an error); c the largest |16 k x_k| over kappa 12 to 18 at the pair; K c = 18 c the exterior common mode's share within K (its reading as the common mode's share assumes the common mode below kappa 12 is no larger than at 12 to 18; it bounds nothing beyond kappa 18). The verdict column is the post-pass v1f's automatic verdict (not formed if |S| < K c; within the floor if 0.01 |S| < F_meas; converged if the last three terms fall at the clock and its neighbours, are resolved, the geometric tail and K c are within 0.01 |S|; else formed, tail open), with the one automatic 'converged' (76 r_s, the 24 r_s cut) printed as formed, tail open: its final tail status is OPEN like every other pair's (T2c). The tail beyond kappa 18 is OPEN at every pair: no evaluated remainder bound is supplied (T2). Rows at 14 to 30 r_s are before the readout clocks (the post-pass rerun from 14 r_s); rows from 40 r_s are the busi lane's K = 18 composition (W1_B2_composed_v3).")
    W("")
    W("| T/r_s | cut r_s | S (nats) | F_meas | F_meas/\\|S\\| | F_apriori | c | K c | K c/\\|S\\| (percent) | q_x (last 3) | verdict (v1f; the final tail status is OPEN) | reason | tail beyond kappa 18 |")
    W("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t in PRE_CLOCKS + READ_CLOCKS:
        for lab in CUTS:
            r = RB.get((t, lab))
            if r is None:
                continue
            W("| %g | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | OPEN |" % (t, lab, f(r["S"], "%.5e"), f(r["F_meas"], "%.2e"), f(r.get("F_meas_over_S"), "%.1e"), f(r["F_apriori"], "%.3f"), f(r["c"], "%.2e"), f(r["share"], "%.2e"), pct(r.get("share_over_S")), f(r.get("q_x"), "%.3f"), final_verdict(r), r.get("why") or "-"))
    W("")
    tl = {}; tla = {}
    for (t, lab), r in RB.items():
        if t >= 40.0 - 1e-9:
            fv = "formed, tail open" if r["verdict"] == "converged" else r["verdict"]
            tl.setdefault(lab, {})[fv] = tl.setdefault(lab, {}).get(fv, 0) + 1
            tla.setdefault(lab, {})[r["verdict"]] = tla.setdefault(lab, {}).get(r["verdict"], 0) + 1
    W("Tally over the 31 readout clocks 40 to 100 r_s (not formed / within the floor / formed, tail open): " + "; ".join("%s r_s: %d / %d / %d" % (lab, tl[lab].get("not formed", 0), tl[lab].get("within the floor", 0), tl[lab].get("formed, tail open", 0)) for lab in CUTS) + ". The automatic v1f tally has the 24 r_s cut as 4 / 0 / 1 / 26 (one pair, 76/24, converged by the v1f rule alone; T2c prints the probes at that pair).")
    big = [r for r in RB.values() if r["t"] >= 40.0 - 1e-9 and abs(r["S"]) > 3.0]
    W("F_meas over the readout pairs: %s nats; F_meas/|S| where |S| exceeds 3 nats: at most %s (%d pairs). K c/|S| over the formed pairs from 80 r_s: %s percent; at 100 r_s: %s percent. F_apriori at 100 r_s: %s nats (%s of the sum; the allowance, not the error)." % (
        rng([r["F_meas"] for r in RB.values() if r["t"] >= 40.0 - 1e-9]), f(max(r["F_meas"] / abs(r["S"]) for r in big), "%.2e"), len(big),
        rng([100.0 * r["share_over_S"] for r in RB.values() if r["t"] >= 80.0 - 1e-9 and r["verdict"] != "not formed"], "%.2f"), rng([100.0 * r["share_over_S"] for r in RB.values() if abs(r["t"] - 100.0) < 1e-9], "%.2f"), rng([r["F_apriori"] for r in RB.values() if abs(r["t"] - 100.0) < 1e-9], "%.3f"), rng([100.0 * r["F_apriori_over_S"] for r in RB.values() if abs(r["t"] - 100.0) < 1e-9], "%.1f") + " percent"))
    pk = LV["largest_abs_sum_from_40"]
    W("The sums against the levels (%s): the largest |S| from 40 r_s is %s nats at T/r_s %g, the cut %s r_s (the last clock's value, not a peak: the 30 r_s cut falls from %s at 60 r_s to %s at 70 and rises again), which is %s of A/(4G) (%s nats) and %s of eta A/a^2 (%s nats) at that record's 2gE (A evaluated at the record's 2gE, not at the trapping root or the cut's radius); at 100 r_s the five cuts read %s of A/(4G). These compare the radial-channel excess with the stated area levels; they identify it with no three-dimensional cut entropy." % (
        LV["definition"], f(pk["S"], "%.5f"), pk["t"], pk["cut"], f(RB[(60.0, "30")]["S"], "%.5f"), f(RB[(70.0, "30")]["S"], "%.5f"), f(pk["S_over_A_over_4G"], "%.5e"), f(pk["A_over_4G"], "%.1f"), f(pk["S_over_eta_A_over_a2"], "%.5e"), f(pk["eta_A_over_a2"], "%.1f"),
        ", ".join("%s r_s: %s" % (lab, f(LV["SUM_over_A_over_4G"][ci[100.0]][j], "%.2e")) for j, lab in enumerate(CUTS))))
    # the common-mode spread over kappa 12 to 18 and the largest kappa 18 term
    sel = [(t, j) for t in READ_CLOCKS for j in range(5) if abs(B["WX"][17][ci[t]][j]) > 5e-4]
    unsel = [(t, j) for t in READ_CLOCKS for j in range(5) if abs(B["WX"][17][ci[t]][j]) <= 5e-4]
    W("The spread of the high channels' terms |16 kappa x_kappa| over kappa 12 to 18 (largest over smallest) at the 35 every-10 readout pairs: %s at the %d pairs where |288 x_18| exceeds 5e-4 (within 17 percent at 100/24, 11 percent at 80/30), %s at the other %d (a factor %s at 100/2, where the terms are 8e-5 to 2.1e-4): the terms are near one value where they are large and not uniformly flat to 10 percent." % (
        rng([CMS[ci[t]][j] for t, j in sel], "%.2f"), len(sel), rng([CMS[ci[t]][j] for t, j in unsel], "%.2f"), len(unsel), f(CMS[ci[100.0]][0], "%.2f")))
    allp = [(t, j) for t in cl if t >= 40.0 - 1e-9 for j in range(5)]
    tmax = max(allp, key=lambda p: abs(B["WX"][17][ci[p[0]]][p[1]]))
    W("The largest kappa 18 term over the 155 readout pairs is at %g r_s, the %s r_s cut: 288 x_18 = %s (the probes there %s, %s, %s at kappa 24, 32, 48); the second largest among the every-10 pairs is 80/30 (%s)." % (
        tmax[0], CUTS[tmax[1]], f(B["WX"][17][ci[tmax[0]]][tmax[1]], "%+.3e"), f(PR["WX"][1][ci[tmax[0]]][tmax[1]], "%+.3e"), f(PR["WX"][2][ci[tmax[0]]][tmax[1]], "%+.3e"), f(PR["WX"][3][ci[tmax[0]]][tmax[1]], "%+.3e"), f(B["WX"][17][ci[80.0]][3], "%+.3e")))
    W("The record-interpolation control (T8) carries the cuts 16 to 48 r_s and kappa 1 to 8 only: the 2 r_s cut's sums, the largest value among them, print with no record-interpolation spread.")
    W("")
    # ------------------------------------------------------------ T1c: the regulated columns
    W("### T1c. The regulated columns behind T1 (computed here from the channel files): the raw exterior-cut entropy and the held vacuum's, summed over kappa 1 to 18 with the weights 16 kappa (nats; the raw cut entropy of the radial truncation has no limit in kappa and prints only as a regulated column; S = raw - vacuum)")
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
    W("D16 = 288 [x_18(16 sub-steps) - x_18(8 sub-steps)] is the measured change of one weighted channel term under sub-step halving; 18 |D16| is a whole-retained-set proxy, conditional on the other 17 channels carrying no larger change, not the measured change of the full sum (a sum restricted to kappa 12 to 18 would contain seven such terms). The record-interpolation column is the production's kappa 1 to 8 partial sum minus the piecewise-constant control's (page_history_v6n) at the same pair (no control exists at the 2 r_s cut); the completion column is the spread (largest minus smallest) of the kappa 1 and 2 weighted excess over the three histories of T4 (B2 and W3's two completions), present only to 40 r_s.")
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
    rows = []
    for i, t in enumerate(cl):
        for j, lab in enumerate(CUTS):
            d = SB["D16_w"][i][j]; S = B["SUM"][i][j]; r = RB.get((t, lab))
            if d is not None and t >= 40.0 - 1e-9 and r is not None:
                rows.append((t, lab, d, S, r["verdict"], 18 * abs(d) / abs(S), abs(SB["WX18"][i][j])))
    formed = [x for x in rows if x[4] in ("formed, tail open", "converged")]; nf = [x for x in rows if x[4] not in ("formed, tail open", "converged")]
    bigs = sorted([x for x in formed if x[5] > 0.005], key=lambda x: -x[5])
    big1 = [x for x in rows if x[6] > 1e-3]
    W("The 16-sub-step control of kappa 18 (B2): %s. Over the %d readout pairs |D16| is %s to %s nats; as 18 |D16| it is below 0.5 percent of the sum at %d of the %d formed pairs and %s at the rest (%s), and %s to %s percent at the %d not-formed and within-the-floor pairs, where the sums are small. Where the kappa 18 term exceeds 1e-3 (%d pairs) the halved sub-step moves it by %s of itself at most (|D16|/|288 x_18|), so the construction's one-sub-step reference lag (W1 ruling v2 section 1(b)) is at most that fraction of the common mode there. The kappa 18 formed energy 288 E_18 under 16 sub-steps is %s to %s percent above its 8-sub-step value over 40 to 100 r_s (a rise, opposite in sign to the s8 kappa 14 control's 12 percent fall)." % (
        "complete to 100 r_s (ruling2 run (c): --interp-sub 16, kappa 18 only, --lean; wall %.2f h)" % (SB["wall_s"] / 3600.0) if SB["final"] else "NOT complete (its rows reach %s r_s)" % f(SB["last_clock"], "%g"),
        len(rows), f(min(abs(x[2]) for x in rows), "%.1e"), f(max(abs(x[2]) for x in rows), "%.1e"), len(formed) - len(bigs), len(formed), ("%.1f to %.1f percent" % (min(x[5] for x in bigs) * 100, max(x[5] for x in bigs) * 100)) if bigs else "-", ", ".join("%g/%s %.2f" % (x[0], x[1], x[5] * 100) for x in bigs),
        f(min(x[5] for x in nf) * 100, "%.2f"), f(max(x[5] for x in nf) * 100, "%.1f"), len(nf), len(big1), f(max(abs(x[2]) / x[6] for x in big1), "%.3f"),
        f(min(100 * (SB["E_kappa_w_sub16"][i] / SB["E_kappa_w_sub8"][i] - 1) for i, t in enumerate(cl) if t >= 40.0 - 1e-9 and SB["E_kappa_w_sub16"][i] is not None), "%.1f"), f(max(100 * (SB["E_kappa_w_sub16"][i] / SB["E_kappa_w_sub8"][i] - 1) for i, t in enumerate(cl) if t >= 40.0 - 1e-9 and SB["E_kappa_w_sub16"][i] is not None), "%.1f")))
    W("")
    # ------------------------------------------------------------ T2: the kappa convergence
    W("### T2. The channel sum's convergence in kappa at the readout pairs: the partial sums, the last three terms, the probes kappa 24, 32, 48, the geometric criterion and its conditional majorant, and the tail")
    W("")
    W("Partial sums S_K = sum_{k <= K} 16 k x_k. The probes are single-channel runs of the same production line (--kappas 24-24, 32-32, 48-48, --no-kappa-extend, --lean) on the same record, final at every one of the 52 checkpoints (every 2 r_s with 15); their terms 16 K x_K are listed with the kappa 18 term. r_a = (|512 x_32|/|384 x_24|)^(1/8) and r_b = (|768 x_48|/|512 x_32|)^(1/16) are per-step ratios of the weighted terms (ratios of the unweighted x differ by (4/3)^(1/8) and (3/2)^(1/16)); 'criterion' says whether the four terms fall from kappa 18 through 24, 32 and 48 with one sign and both per-step ratios below 1 (W1 ruling v2 section 2(b)); where it holds, the conditional majorant |288 x_18| q/(1 - q), q the larger ratio, is printed as the remainder that extrapolation would give, with its percent of |S|: it is a conditional estimate on a continuation premise, never applied here, and no pair prints a percentage. The slope is the least-squares exponent of |16 kappa x_kappa| against kappa over the four channels where they share a sign (a finite-range fit, not an asymptotic law). The tail beyond kappa 18 is OPEN at every pair: no evaluated, proved remainder bound is supplied.")
    W("")
    W("| T/r_s | cut r_s | S_2 | S_8 | S_12 | S_18 | 256 x_16 | 272 x_17 | 288 x_18 | 384 x_24 | 512 x_32 | 768 x_48 | r_a | r_b | same sign | criterion | conditional majorant (percent of \\|S\\|) | slope | c | tail |")
    W("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t in READ_CLOCKS:
        i = ci[t]
        for j, lab in enumerate(CUTS):
            P = B["partial"]; WX = B["WX"]
            cm = PR["conditional_majorant"][i][j]; cms = PR["conditional_majorant_over_S"][i][j]
            W("| %g | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | OPEN |" % (t, lab, f(P[1][i][j], "%.4e"), f(P[7][i][j], "%.4e"), f(P[11][i][j], "%.4e"), f(P[17][i][j], "%.5e"), f(WX[15][i][j], "%+.2e"), f(WX[16][i][j], "%+.2e"), f(WX[17][i][j], "%+.2e"),
                                                                                                                 f(PR["WX"][1][i][j], "%+.2e"), f(PR["WX"][2][i][j], "%+.2e"), f(PR["WX"][3][i][j], "%+.2e"), f(PR["r_a"][i][j], "%.3f"), f(PR["r_b"][i][j], "%.3f"), "yes" if PR["same_sign"][i][j] else "no", "met" if PR["criterion_met"][i][j] else "not met",
                                                                                                                 ("%s (%s)" % (f(cm, "%.2e"), pct(cms))) if isnum(cm) else "-", f(PR["slope"][i][j], "%.2f"), f(B["C"][i][j], "%.2e")))
    W("")
    ra = [PR["r_a"][ci[t]][j] for t, j in sel]; rb = [PR["r_b"][ci[t]][j] for t, j in sel]
    ss = [PR["same_sign"][ci[t]][j] for t, j in sel]
    met = [(t, j) for t, j in sel if PR["criterion_met"][ci[t]][j]]
    med = lambda v: sorted(v)[len(v) // 2] if v else None
    W("Over the %d pairs at the every-10 readout clocks where |288 x_18| exceeds 5e-4: r_a %s (median %s), r_b %s (median %s); the four terms share a sign at %d of them, and the criterion (the terms fall from kappa 18 through 24, 32 and 48 with one sign, both per-step ratios below 1) is met at %d: %s. The conditional majorant there reads %s of the sum (%s); it is not applied. For a 1/kappa law of the weighted terms the per-step ratios would be %.3f and %.3f (the medians are those of a power law near 1/kappa, whose partial sums grow as ln K); the fitted exponents over the %d same-sign pairs run %s, over all %d same-sign pairs of the 35 %s; the terms change sign between kappa 18 and 48 at %d of the %d selected pairs and at %d of the other %d; at the same cut the terms rise or change sign at neighbouring readout clocks (80/24, 80/30, 60/24, 100/48). These finite-range samples establish neither divergence nor the absence of eventual geometric decay; no unconditional percentage is certified for the completed sum." % (
        len(sel), rng(ra, "%.3f"), f(med(ra), "%.3f"), rng(rb, "%.3f"), f(med(rb), "%.3f"), sum(1 for s in ss if s), len(met), ", ".join("%g/%s" % (t, CUTS[j]) for t, j in met),
        rng([100.0 * PR["conditional_majorant_over_S"][ci[t]][j] for t, j in met], "%.2f") + " percent", "; ".join("%g/%s %s" % (t, CUTS[j], pct(PR["conditional_majorant_over_S"][ci[t]][j])) for t, j in met),
        (18.0 / 24.0) ** (1.0 / 8.0) if False else (24.0 / 32.0) ** (1.0 / 8.0), (32.0 / 48.0) ** (1.0 / 16.0),
        sum(1 for s in ss if s), rng([PR["slope"][ci[t]][j] for t, j in sel if PR["same_sign"][ci[t]][j]], "%.2f"), sum(1 for t in READ_CLOCKS for j in range(5) if PR["same_sign"][ci[t]][j]), rng([PR["slope"][ci[t]][j] for t in READ_CLOCKS for j in range(5) if PR["same_sign"][ci[t]][j]], "%.2f"),
        sum(1 for s in ss if not s), len(sel), sum(1 for t, j in unsel if not PR["same_sign"][ci[t]][j]), len(unsel)))
    ratios48 = [PR["ratio_48_18"][ci[t]][j] for t, j in sel if PR["same_sign"][ci[t]][j]]
    W("Where the signs agree among the selected pairs, the kappa 48 term is %s of the kappa 18 term; the kappa 24 term %s; the kappa 32 term %s. Over all 155 readout pairs (every 2 r_s) the criterion is met at %d of the %d pairs where |288 x_18| exceeds 5e-4 and at %d of the 155." % (
        rng(ratios48, "%.3f"), rng([PR["ratio_24_18"][ci[t]][j] for t, j in sel if PR["same_sign"][ci[t]][j]], "%.2f"), rng([PR["ratio_32_18"][ci[t]][j] for t, j in sel if PR["same_sign"][ci[t]][j]], "%.2f"),
        sum(1 for t in cl if t >= 40.0 - 1e-9 for j in range(5) if PR["selected_5e-4"][ci[t]][j] and PR["criterion_met"][ci[t]][j]), sum(1 for t in cl if t >= 40.0 - 1e-9 for j in range(5) if PR["selected_5e-4"][ci[t]][j]), sum(1 for t in cl if t >= 40.0 - 1e-9 for j in range(5) if PR["criterion_met"][ci[t]][j])))
    W("")
    W("### T2c. The pair the v1f rule prints converged (76 r_s, the 24 r_s cut) with the probes at that clock and its neighbours")
    W("")
    W("| T/r_s | cut | S_18 | 256 x_16 | 272 x_17 | 288 x_18 | q_x (last 3) | 384 x_24 | 512 x_32 | 768 x_48 | r_a | r_b | criterion | conditional majorant (percent of \\|S\\|) | v1f verdict | final tail status |")
    W("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t in [70.0, 72.0, 74.0, 76.0, 78.0, 80.0]:
        i = ci[t]; j = 2; r = RB.get((t, "24")) or {}
        cm = PR["conditional_majorant"][i][j]; cms = PR["conditional_majorant_over_S"][i][j]
        W("| %g | 24 | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | OPEN |" % (t, f(B["SUM"][i][j], "%.5e"), f(B["WX"][15][i][j], "%+.2e"), f(B["WX"][16][i][j], "%+.2e"), f(B["WX"][17][i][j], "%+.2e"), f(r.get("q_x"), "%.3f"), f(PR["WX"][1][i][j], "%+.2e"), f(PR["WX"][2][i][j], "%+.2e"), f(PR["WX"][3][i][j], "%+.2e"), f(PR["r_a"][i][j], "%.3f"), f(PR["r_b"][i][j], "%.3f"), "met" if PR["criterion_met"][i][j] else "not met", ("%s (%s)" % (f(cm, "%.2e"), pct(cms))) if isnum(cm) else "-", r.get("verdict", "-")))
    W("")
    r76 = RB[(76.0, "24")]
    W("At 76/24 the v1f rule's clauses hold (q_x %s over kappa 16 to 18, the neighbours 74 and 78 falling, the last three terms resolved, the geometric tail %s and K c %s within 0.01 |S| = %s); the probes at the same clock give the row above. The composition prints the pair formed, tail open with the final tail status OPEN like every other pair: the v1f rule's majorant is the last three channels' extrapolation and no remainder bound beyond kappa 18 is supplied at this pair either (T2)." % (
        f(r76.get("q_x"), "%.3f"), f(r76.get("x_tail"), "%.2e"), f(r76.get("share"), "%.2e"), f(r76.get("one_pct"), "%.2e")))
    W("")
    W("### T2b. Every channel's term at three pairs (the figure's panel c): 16 kappa x_kappa, its propagation floor 16 kappa F_meas,kappa, and the formed-energy term 16 kappa E_kappa")
    W("")
    for (t, lab) in ((100.0, "24"), (80.0, "30"), (50.0, "16"), (100.0, "2")):
        i = ci[t]; j = CUTS.index(lab)
        W("At T/r_s %g, the cut %s r_s (S_18 = %s nats; c = %s; F_meas = %s; the spread of |16 kappa x_kappa| over kappa 12 to 18 a factor %s):" % (t, lab, f(B["SUM"][i][j], "%.5e"), f(B["C"][i][j], "%.2e"), f(B["FMW"][i][j], "%.2e"), f(CMS[i][j], "%.2f")))
        W("")
        W("| kappa | 16 kappa x_kappa | 16 kappa F_meas,kappa | resolved | 16 kappa E_kappa | partial sum S_kappa |")
        W("|---|---|---|---|---|---|")
        for a, k in enumerate(B["kappas"]):
            W("| %d | %s | %s | %s | %s | %s |" % (k, f(B["WX"][a][i][j], "%+.3e"), f(B["fmeas_w"][a][i][j], "%.1e"), "yes" if B["resolved"][a][i][j] else "no", f(B["E_kappa_w"][a][i], "%.2e"), f(B["partial"][a][i][j], "%.5e")))
        for b, K in enumerate((24, 32, 48)):
            W("| %d (probe) | %s | %s | %s | %s | - |" % (K, f(PR["WX"][b + 1][i][j], "%+.3e"), f(PR["fmeas_w"][b][i][j], "%.1e"), "yes" if PR["resolved"][b][i][j] else "no", f(PR["E_kappa_w"][str(K)][i], "%.2e")))
        W("")
    W("The formed-energy majorant over the last three channels (q 0.878 to 0.897 from 40 r_s; the energy tail at most 1.1e-3 of the partial sum at K = 18, W1_B2_composed_v3) is a within-K statement: the probes' energies 16 kappa E_kappa at 100 r_s are %s, %s and %s at kappa 24, 32 and 48 against %s at kappa 18, so the last-three-channel envelope (which would put the kappa 24 term below 3.7e-5) is not continued by the probes; no energy tail beyond kappa 18 is certified." % (f(PR["E_kappa_w"]["24"][ci[100.0]], "%.2e"), f(PR["E_kappa_w"]["32"][ci[100.0]], "%.2e"), f(PR["E_kappa_w"]["48"][ci[100.0]], "%.2e"), f(B["E_kappa_w"][17][ci[100.0]], "%.2e")))
    W("")
    # ------------------------------------------------------------ T3: the two scales
    W("### T3. The two averaging scales side by side at K = 18: the sigma/a 4 history (B2) and the sigma/a 8 history (s8), 14 to 40 r_s")
    W("")
    W("The sigma/a 8 history is a different object printed beside, never averaged with B2 (its record ends at 40 r_s; its formation record clock is %s with R_t %s a; B2's %s with R_t %s a). Both sums are the v1f post-pass's at K = 18 (the s8 composition is this lane's rerun of the busi post-pass on the 18 s8 channels; the row at 40 is its from-40 run, the rows before it its from-14 run). R_t is the root the history record reports (the dense fields); min(N - |v|) is copied from the propagation row (the applied fields on the 8192-site chain): their conventions differ and the two are not used together as a trapping certificate here." % (f(MK["s8"]["formation_record_clock_over_rs"], "%g"), f(MK["s8"]["formation_R_t_a"], "%.2f"), f(MK["formation_record_clock_over_rs"], "%g"), f(MK["formation_R_t_a"], "%.2f")))
    W("")
    W("| T/r_s | cut r_s | B2: S | F_meas | K c | verdict | s8: S | F_meas | K c | verdict | B2: 288 E_18 | s8: 288 E_18 |")
    W("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t in [14.0, 16.0, 20.0, 30.0, 40.0]:
        for lab in CUTS:
            r = RB.get((t, lab)); q = R8.get((t, lab))
            if r is None or q is None:
                continue
            W("| %g | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (t, lab, f(r["S"], "%.4e"), f(r["F_meas"], "%.2e"), f(r["share"], "%.2e"), final_verdict(r), f(q["S"], "%.4e"), f(q["F_meas"], "%.2e"), f(q["share"], "%.2e"), q["verdict"], f(B["E_kappa_w"][17][ci[t]], "%.2e"), f(S8["E_kappa_w"][17][ci8[t]], "%.2e")))
    W("")
    tl8 = {}
    for (t, lab), r in R8.items():
        tl8.setdefault(lab, {})[r["verdict"]] = tl8.setdefault(lab, {}).get(r["verdict"], 0) + 1
    W("s8 tally over its 15 clocks 14 to 40 r_s (not formed / within the floor / converged / formed, tail open): " + "; ".join("%s r_s: %d / %d / %d / %d" % (lab, tl8[lab].get("not formed", 0), tl8[lab].get("within the floor", 0), tl8[lab].get("converged", 0), tl8[lab].get("formed, tail open", 0)) for lab in CUTS) + ". At 40 r_s |S| at the 16 to 48 r_s cuts is %s nats against K c %s." % (rng([abs(R8[(40.0, lab)]["S"]) for lab in CUTS[1:]], "%.4e"), rng([R8[(40.0, lab)]["share"] for lab in CUTS[1:]], "%.2e")))
    i8 = ci8[40.0]
    W("s8's weighted energy terms 16 kappa E_kappa at 40 r_s over kappa 1 to 18: " + ", ".join("%d: %s" % (k, f(S8["E_kappa_w"][a][i8], "%.2e")) for a, k in enumerate(S8["kappas"])) + " (B2's at 40 r_s over kappa 12 to 18: " + ", ".join("%d: %s" % (k, f(B["E_kappa_w"][a][ci[40.0]], "%.2e")) for a, k in enumerate(B["kappas"]) if k >= 12) + ").")
    W("")
    W("The sigma/a 8 history's collar flux (kappa 1, 2; L per coordinate interval; the comparator where the record has a root):")
    W("")
    W("| T/r_s | s8: E | R_t (a) | min(N - \\|v\\|) | L (kappa 1, 2) | L_H(kappa_T) | L/L_H(kappa_T) | L/L_H(T_H) | E_inv (kappa 1, 2) | E_inv/E, percent | collar flow acc. (kappa 1, 2) | flow over E | join max rel dv |")
    W("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t in [14.0, 15.0, 16.0, 20.0, 30.0, 36.0, 38.0, 40.0]:
        if t not in ci8:
            continue
        i = ci8[t]
        L12 = (S8["L_per_channel"][0][i] + S8["L_per_channel"][1][i]) if (isnum(S8["L_per_channel"][0][i]) and isnum(S8["L_per_channel"][1][i])) else None
        LHk = (S8["LH_kT_per_channel"][0][i] + S8["LH_kT_per_channel"][1][i]) if (isnum(S8["LH_kT_per_channel"][0][i]) and isnum(S8["LH_kT_per_channel"][1][i])) else None
        LHt = (S8["LH_TH_per_channel"][0][i] + S8["LH_TH_per_channel"][1][i]) if (isnum(S8["LH_TH_per_channel"][0][i]) and isnum(S8["LH_TH_per_channel"][1][i])) else None
        Ei = (S8["E_kappa_w"][0][i] or 0.0) + (S8["E_kappa_w"][1][i] or 0.0); fl = (S8["flow_w"][0][i] or 0.0) + (S8["flow_w"][1][i] or 0.0); E = S8["E"][i]
        W("| %g | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (t, f(E, "%.4f"), f(S8["R_t"][i], "%.3f"), f(S8["minNv"][i], "%+.4f"), f(L12, "%+.3e"), f(LHk, "%.3e"), f((L12 / LHk) if (L12 is not None and LHk) else None, "%+.4f"), f((L12 / LHt) if (L12 is not None and LHt) else None, "%+.4f"), f(Ei, "%.4e"), pct(Ei / E if E else None), f(fl, "%+.4e"), f(fl / E if E else None, "%.3e"), f(S8["join_dv"][i], "%.2e")))
    W("")
    rows8 = []
    for i, t in enumerate(cl8):
        for j, lab in enumerate(CUTS):
            d = SS["D16_w"][i][j]; S = S8["SUM"][i][j]; x = SS["WX14"][i][j]; v = (R8.get((t, lab)) or {}).get("verdict")
            if d is not None and t >= 14.0 - 1e-9 and v is not None:
                rows8.append((t, lab, d, S, v, 18 * abs(d) / abs(S), abs(d / x) if x else None))
    f8 = [x for x in rows8 if x[4] == "formed, tail open"]
    W("The s8 16-sub-step control (kappa 14, to 40 r_s, %s): D16 = 224 [x_14(16 sub) - x_14(8 sub)] against the kappa 14 term 224 x_14, and the formed energy 224 E_14 under both sub-steps (one weighted channel term's change; 18 |D16| the whole-set proxy of T1b). Over the %d formed pairs (%s) 18 |D16|/|S_18| is at most %s percent and |D16|/|224 x_14| at most %s (at %g/%s); at the not-formed pair 40/16 the term ratio is %s and at 40/48 %s." % (
        "complete" if SS["final"] else "NOT complete", len(f8), "; ".join("%s r_s: %d" % (lab, sum(1 for x in f8 if x[1] == lab)) for lab in CUTS if any(x[1] == lab for x in f8)), f(max(x[5] for x in f8) * 100, "%.2f"), f(max(x[6] for x in f8 if x[6] is not None), "%.3f"), *max(((x[0], x[1]) for x in f8 if x[6] is not None), key=lambda p: [x[6] for x in f8 if (x[0], x[1]) == p][0]),
        f([x[6] for x in rows8 if x[0] == 40.0 and x[1] == "16"][0], "%.3f"), f([x[6] for x in rows8 if x[0] == 40.0 and x[1] == "48"][0], "%.3f")))
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
    W("The completions arm at the first record failing the hydrodynamic gate at or after T/r_s 11.125 (the label above), restarted from the rung-4 run's stored slice at T/r_s 11.0 on engine v36g with 2048 engine sites; B2 is engine_collapse_v19's record on 4096 sites. The paired-avg completion runs at a charge above the record's (E 113.01 against 106.21 at 40 r_s, 6.4 percent; 114.43 against 109.28 at 14), so its exterior differs at zeroth order; the held-sources completion's charge is projected to the record's E. The spread below carries those differences with the interior's.")
    W("")
    W("| T/r_s | cut r_s | B2: 16 x_1 + 32 x_2 | held-sources (W3fs) | paired-avg (W3pa) | spread (all three) | spread (the two completions) | spread over \\|S_18\\| (percent) |")
    W("|---|---|---|---|---|---|---|---|")
    for t in [14.0, 16.0, 20.0, 30.0, 40.0]:
        i = cpc[t]
        for j, lab in enumerate(CUTS):
            S18 = B["SUM"][ci[t]][j]
            W("| %g | %s | %s | %s | %s | %s | %s | %s |" % (t, lab, f(CP["runs"][names[0]]["SUM12"][i][j], "%+.4e"), f(CP["runs"]["held-sources (W3fs)"]["SUM12"][i][j], "%+.4e"), f(CP["runs"]["paired-avg (W3pa)"]["SUM12"][i][j], "%+.4e"), f(CP["spread"]["SUM12_max_minus_min_all3"][i][j], "%.4e"), f(CP["spread"]["SUM12_max_minus_min_two_completions"][i][j], "%.4e"), "-" if not S18 else "%.1f" % (100.0 * CP["spread"]["SUM12_max_minus_min_all3"][i][j] / abs(S18))))
    W("")
    W("| T/r_s | run | E (charge) | R_t (a) | min(N - \\|v\\|) | E_inv (kappa 1, 2) | collar flow acc. (kappa 1, 2) | L (kappa 1, 2) | L_H(kappa_T) | join max dN | join max rel dv |")
    W("|---|---|---|---|---|---|---|---|---|---|---|")
    for t in [12.0, 14.0, 16.0, 20.0, 30.0, 40.0]:
        i = cpc[t]
        for nm in names:
            h = CP["runs"][nm]
            W("| %g | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (t, nm.split(" (")[0], f(h["E"][i], "%.4f"), f(h["R_t"][i], "%.3f"), f(h["minNv"][i], "%+.4f"), f(h["E_inv12"][i], "%.4e"), f(h["flow12"][i], "%+.4e"), f(h["L12"][i], "%+.3e"), f(h["LH12_kT"][i], "%.3e"), f(h["join_dN"][i], "%.2e"), f(h["join_dv"][i], "%.2e")))
    W("")
    W("The spread of the charge E over the three histories at the sampled clocks 14 to 40: %s (the two completions alone %s; 0 at the start)." % (rng([v for v, t in zip(CP["spread"]["E_max_minus_min_all3"], CP["clocks"]) if t >= 14.0 - 1e-9], "%.3f"), rng([v for v, t in zip(CP["spread"]["E_max_minus_min_two_completions"], CP["clocks"]) if t >= 14.0 - 1e-9], "%.3f")))
    W("")
    # ------------------------------------------------------------ T5: the flux
    W("### T5. The collar flux of the kappa 1 and 2 channels on B2 to 100 r_s, with the kappa 3 guard's change and the K = 18 sum beside")
    W("")
    W("L is the collar energy flow per coordinate interval Delta T (the increment of the stored accumulator over the checkpoint interval divided by its length; identity checked to %s relative), placed at the collar's recordwise stationary-chart clock t_inf (t_inf - T = %s to %s r_s over 14 to 100); L_H(kappa_T) and L_H(T_H) are the channel set's continuum greybody comparators at the record's dense root and slope and at 1/(8 pi G E), not an observed thermal flux. The ratios with kappa 3 add the truncation guard (a truncation diagnostic, not a bound on every omitted channel); the K = 18 ratio sums every channel's L (the high channels' accumulators carry the moving exterior's first-order term, W1 ruling S3, so that ratio is a diagnostic only). 'v6b' is version 3's two-channel ratio on the piecewise-constant propagation of the r11 record (through 40), the propagation variant beside. The E_inv/E columns are percentages." % (f(CH["L_interval_identity_max_rel"], "%.1e"), f(FL["t_inf_collar_over_rs"][ci[14.0]] - 14.0, "%.3f"), f(MK["collar_clock_offset_over_rs"]["max"], "%.3f")))
    W("")
    W("| T/r_s | t_inf/r_s | E | L (kappa 1, 2) | L_H(kappa_T) | L_H(T_H) | L/L_H(kappa_T) | L/L_H(T_H) | with kappa 3: L/L_H(kappa_T) | L/L_H(T_H) | K = 18: L/L_H(kappa_T) | E_inv/E, percent, kappa 1 and 2 | E_inv/E, percent, kappa 1 to 3 | E_inv/E, percent, K = 18 | join max rel dv | v6b L/L_H(kappa_T) |")
    W("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    v3 = D["v3_flux"]; v3i = {round(t, 4): i for i, t in enumerate(v3["t_over_rs"])}
    for t in [14.0, 15.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 34.0, 36.0, 38.0, 40.0, 42.0, 44.0, 46.0, 48.0, 50.0, 52.0, 54.0, 56.0, 58.0, 60.0, 62.0, 64.0, 66.0, 68.0, 70.0, 72.0, 74.0, 76.0, 78.0, 80.0, 82.0, 84.0, 86.0, 88.0, 90.0, 92.0, 94.0, 96.0, 98.0, 100.0]:
        if t not in ci:
            continue
        i = ci[t]
        v3r = v3["L_over_LH_kappaT"][v3i[t]] if t in v3i else None
        W("| %g | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (t, f(FL["t_inf_collar_over_rs"][i], "%.3f"), f(FL["E"][i], "%.3f"), f(FL["L12"][i], "%+.3e"), f(FL["LH12_kT"][i], "%.3e"), f(FL["LH12_TH"][i], "%.3e"), f(FL["ratio12_kT"][i], "%+.4f"), f(FL["ratio12_TH"][i], "%+.4f"), f(FL["ratio123_kT"][i], "%+.4f"), f(FL["ratio123_TH"][i], "%+.4f"), f(FL["ratio18_kT"][i], "%+.4f"),
                                                                                                 pct(FL["E_inv12_over_E"][i]), pct(FL["E_inv123"][i] / FL["E"][i] if FL["E"][i] else None), pct(FL["E_inv18_over_E"][i]), f(FL["join_dv"][i], "%.2e"), f(v3r, "%+.4f")))
    W("")
    i100 = ci[100.0]; i40 = ci[40.0]
    jd = [(t, FL["join_dv"][ci[t]]) for t in cl if 14.0 - 1e-9 <= t]
    first1e3 = next((t for t, v in jd if v is not None and v > 1e-3), None)
    W("The accumulated collar flow of kappa 1 and 2 over E (E_rad^(0)/E, a diagnostic): %s at 40 r_s, %s at 100 r_s. E_inv (kappa 1, 2) over E: %s at 14 r_s, %s over 15 to 40, %s over 42 to 100 (K = 18: %s at 14, %s over 42 to 100). The flux ratio (kappa 1, 2, slope comparator): %s over 14 to 34 where the comparator exists (negative L at %d of the %d checkpoints 14 to 34; the comparator is absent at 18 r_s, where the record has no root), %s over 36 to 40, %s over 42 to 46, %s over 48 to 60, %s over 62 to 100 (negative at %d of the %d checkpoints from 42); the comparator L_H(kappa_T) is %s times L_H(T_H) over 42 to 100. The join's largest relative deviation in v: %s over 14 to 56, %s at 58, %s at 60, %s at 64, first above 1e-3 at %s r_s (%s), %s at 80, %s at 100." % (
        f(FL["E_rad0_over_E_12"][i40], "%.3e"), f(FL["E_rad0_over_E_12"][i100], "%.3e"), pct(FL["E_inv12_over_E"][ci[14.0]]) + " percent", rng([100.0 * FL["E_inv12_over_E"][ci[t]] for t in cl if 15.0 - 1e-9 <= t <= 40.0 + 1e-9], "%.2f") + " percent", rng([100.0 * FL["E_inv12_over_E"][ci[t]] for t in cl if t >= 42.0 - 1e-9], "%.2f") + " percent",
        pct(FL["E_inv18_over_E"][ci[14.0]]) + " percent", rng([100.0 * FL["E_inv18_over_E"][ci[t]] for t in cl if t >= 42.0 - 1e-9], "%.2f") + " percent",
        rng([FL["ratio12_kT"][ci[t]] for t in cl if 14.0 - 1e-9 <= t <= 34.0 + 1e-9], "%+.4f"), sum(1 for t in cl if 14.0 - 1e-9 <= t <= 34.0 + 1e-9 and FL["L12"][ci[t]] is not None and FL["L12"][ci[t]] < 0), sum(1 for t in cl if 14.0 - 1e-9 <= t <= 34.0 + 1e-9),
        rng([FL["ratio12_kT"][ci[t]] for t in cl if 36.0 - 1e-9 <= t <= 40.0 + 1e-9], "%+.4f"), rng([FL["ratio12_kT"][ci[t]] for t in cl if 42.0 - 1e-9 <= t <= 46.0 + 1e-9], "%+.2f"), rng([FL["ratio12_kT"][ci[t]] for t in cl if 48.0 - 1e-9 <= t <= 60.0 + 1e-9], "%+.2f"), rng([FL["ratio12_kT"][ci[t]] for t in cl if t >= 62.0 - 1e-9], "%+.3f"), sum(1 for t in cl if t >= 42.0 - 1e-9 and FL["L12"][ci[t]] < 0), sum(1 for t in cl if t >= 42.0 - 1e-9),
        rng([FL["LH12_kT"][ci[t]] / FL["LH12_TH"][ci[t]] for t in cl if t >= 42.0 - 1e-9], "%.2f"), rng([v for t, v in jd if t <= 56.0 + 1e-9], "%.2e"), f(FL["join_dv"][ci[58.0]], "%.2e"), f(FL["join_dv"][ci[60.0]], "%.2e"), f(FL["join_dv"][ci[64.0]], "%.2e"), f(first1e3, "%g"), f(FL["join_dv"][ci[first1e3]] if first1e3 is not None else None, "%.2e"), f(FL["join_dv"][ci[80.0]], "%.2e"), f(FL["join_dv"][ci[100.0]], "%.2e")))
    W("")
    A100 = (v3.get("A_to100") or {}).get("rows") or []
    a100 = {round(r["t_over_rs"], 4): r for r in A100}
    comp = []
    for t in [42.0, 44.0, 46.0, 48.0, 50.0, 52.0, 54.0, 56.0]:
        r = a100.get(t)
        if r and isnum(r.get("L_over_LH_kappaT")) and isnum(FL["ratio12_kT"][ci[t]]):
            comp.append((t, r["L_over_LH_kappaT"], FL["ratio12_kT"][ci[t]], r["L_over_LH_kappaT"] / FL["ratio12_kT"][ci[t]] - 1.0))
    W("The propagation variant on the continuation: version 3's A-to-100 run (page_history_v6i, piecewise-constant, the first chain; its rows after 58 r_s carry the return marker) against this run's kappa 1, 2 ratio L/L_H(kappa_T) at 42 to 56 r_s: " + "; ".join("%g: %s against %s (%+.1f percent)" % (t, f(a, "%.3f"), f(b, "%.3f"), 100.0 * d) for t, a, b, d in comp) + "; the largest difference %s percent (at %g r_s). At 36 and 38 r_s version 3's s4 run (v6b) read %s and %s against this run's %s and %s." % (
        f(max(abs(d) for _, _, _, d in comp) * 100 if comp else None, "%.1f"), (max(comp, key=lambda x: abs(x[3]))[0] if comp else 0), f(v3["L_over_LH_kappaT"][v3i[36.0]], "%.4f"), f(v3["L_over_LH_kappaT"][v3i[38.0]], "%.4f"), f(FL["ratio12_kT"][ci[36.0]], "%.4f"), f(FL["ratio12_kT"][ci[38.0]], "%.4f")))
    W("")
    W("E_out/E_kappa (the exterior share of the formed energy beyond the collar) per channel at 100 r_s: " + ", ".join("%d: %s" % (k, f(FL["E_out_over_E_kappa"][a][i100], "%.3f")) for a, k in enumerate(B["kappas"])) + " (values above 1 are possible: a region may carry negative excess energy).")
    W("")
    W("The first-order term of W1 ruling v2 section 3 per channel at 100 r_s (unweighted; from the stored rows: d = (E_out - flow)/(16 kappa), the balance discrepancy between the exterior deviation energy beyond the collar and the accumulated collar flow, equal to W_exc - R, the moving fields' work on the exterior excess minus the held vacuum's rearrangement, neither measured separately here; E_kappa = E_inv/(16 kappa); the accumulated flow itself prints as an emitted energy nowhere beyond the kappa 1, 2 diagnostic above and stays in the audit data):")
    W("")
    W("| kappa | 16 kappa E_kappa (weighted) | 16 kappa E_out (weighted) | d = (E_out - flow)/(16 kappa) | E_kappa (unweighted) | d/E_kappa | L at 100 (weighted) | L_H(kappa_T) at 100 |")
    W("|---|---|---|---|---|---|---|---|")
    first_k = None
    for a, k in enumerate(B["kappas"]):
        d = DT[a][i100]; ek = EKU[a][i100]
        if first_k is None and k >= 8 and isnum(d) and isnum(ek) and d > ek:
            first_k = k
        W("| %d | %s | %s | %s | %s | %s | %s | %s |" % (k, f(FL["E_inv_per_channel"][a][i100], "%.3e"), f(FL["E_out_per_channel"][a][i100], "%.3e"), f(d, "%.3e"), f(ek, "%.3e"), f(d / ek if ek else None, "%.3f"), f(FL["L_per_channel"][a][i100], "%+.3e"), f(FL["LH_kT_per_channel"][a][i100], "%.2e")))
    W("")
    W("d is %s per channel over kappa 11 to 18 at 100 r_s and exceeds E_kappa from kappa %s on (d/E_kappa %s at kappa 12, %s at kappa 13)." % (rng([DT[a][i100] for a, k in enumerate(B["kappas"]) if 11 <= k <= 18], "%.3e"), f(first_k, "%d"), f(DT[11][i100] / EKU[11][i100], "%.3f"), f(DT[12][i100] / EKU[12][i100], "%.3f")))
    W("")
    W("The signed collar split per checkpoint (kappa 1, 2; the collar_split block as stored: the energy flux through the collar window at the checkpoint split by the radial current's sign and by the threshold 25 T_H^ref; window %s sites about the site %s; weights 16 and 32 summed; the ratio |in|/|out| beside):" % (SP["definition"]["L"], SP["definition"]["centre_site"]))
    W("")
    W("| T/r_s | E flux out (kappa 1, 2) | E flux in | \\|in\\|/\\|out\\| | out soft | in soft | out hard | in hard | kappa 1: out, in | kappa 2: out, in |")
    W("|---|---|---|---|---|---|---|---|---|---|")
    spi = {t: i for i, t in enumerate(SP["clocks"])}
    for t in [14.0, 20.0, 30.0, 36.0, 38.0, 40.0, 42.0, 44.0, 46.0, 48.0, 50.0, 54.0, 56.0, 60.0, 70.0, 78.0, 80.0, 90.0, 92.0, 100.0]:
        if t not in spi:
            continue
        i = spi[t]; o = SP["E_flux_out_12"][i]; n = SP["E_flux_in_12"][i]
        W("| %g | %s | %s | %s | %s | %s | %s | %s | %s, %s | %s, %s |" % (t, f(o, "%+.3e"), f(n, "%+.3e"), "-" if (o is None or n is None or not o) else "%.3f" % (abs(n) / abs(o)), f(SP["E_flux_out_soft_12"][i], "%+.2e"), f(SP["E_flux_in_soft_12"][i], "%+.2e"), f(SP["E_flux_out_hard_12"][i], "%+.2e"), f(SP["E_flux_in_hard_12"][i], "%+.2e"), f(SP["E_flux_out_k1"][i], "%+.2e"), f(SP["E_flux_in_k1"][i], "%+.2e"), f(SP["E_flux_out_k2"][i], "%+.2e"), f(SP["E_flux_in_k2"][i], "%+.2e")))
    W("")
    rat = [(t, abs(SP["E_flux_in_12"][i]) / abs(SP["E_flux_out_12"][i])) for i, t in enumerate(SP["clocks"]) if t >= 40.0 - 1e-9 and SP["E_flux_out_12"][i]]
    W("Over the checkpoints 40 to 100 r_s the ratio |in|/|out| of the kappa 1, 2 split runs %s; it is below 0.01 at %d of the %d checkpoints and above it at %d: %s." % (rng([r for _, r in rat], "%.3f"), sum(1 for _, r in rat if r < 0.01), len(rat), sum(1 for _, r in rat if r >= 0.01), ", ".join("%g (%.3f)" % (t, r) for t, r in rat if r >= 0.01)))
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
    for lab in ("2", "16", "24", "30", "48"):
        e = MK["light_travel"][lab]
        W("| the formation's light at the %s r_s radius (fixed-charge flight comparison from 2 and 1.5 x 2gE_f) | record clock %.2f to %.2f r_s; shifted by the 16 r_s collar's recordwise offset at the nearest checkpoint: %.2f to %.2f (not a clock conversion at that radius); plus T_d on the record clock: %.2f to %.2f | 2 r_s,f [F(x1) - F(x0)], F(x) = x^2/2 + x + ln(x - 1), x = (R/r_s,f)^(1/2), r_s,f = 2gE_f at the formation record, from c = 2 and 1.5 to R; the version 3 formula (travel over 2gE_f: %.4f and %.4f) |" % (lab, e["2.0"]["arrival_record_clock_over_rs"], e["1.5"]["arrival_record_clock_over_rs"], e["2.0"]["arrival_collar_clock_over_rs"], e["1.5"]["arrival_collar_clock_over_rs"], e["ray_map_end_record_clock_over_rs"]["2.0"], e["ray_map_end_record_clock_over_rs"]["1.5"], e["2.0"]["travel_over_2gE_f"], e["1.5"]["travel_over_2gE_f"]))
    hf = MK["half_flow"]
    W("| the half-collar-flow clock (kappa 1, 2) through 100 r_s | bracketed by the checkpoints %s and %s (the linear-interpolation estimate %s r_s; the accumulated flow at 100: %s) | %s |" % (f(hf["through_100_kappa12"]["between"][0], "%g"), f(hf["through_100_kappa12"]["between"][1], "%g"), f(hf["through_100_kappa12"]["record_clock_over_rs"], "%.2f"), f(hf["through_100_kappa12"]["acc_at_100"], "%.4e"), hf["definition"]))
    W("| the half-collar-flow clock (kappa 1, 2) through 40 r_s | bracketed by %s and %s (the estimate %s r_s; the flow at 40: %s); version 3's on the s4 run (per-record flows, v6b): %s on the record clock, %s on the collar clock | the same, through 40 only |" % (f(hf["through_40_kappa12"]["between"][0], "%g"), f(hf["through_40_kappa12"]["between"][1], "%g"), f(hf["through_40_kappa12"]["record_clock_over_rs"], "%.2f"), f(hf["through_40_kappa12"]["acc_at_40"], "%.4e"), f(hf["v3_through_40_s4_run"]["record_clock_over_rs"], "%.3f"), f(hf["v3_through_40_s4_run"]["collar_clock_over_rs"], "%.3f")))
    W("| the first return (estimate) | %s | %s |" % ("; ".join("cut %s r_s: %.1f r_s" % (lab, v["first_return_rs"]) for lab, v in D13["first_return_at_100"].items()), D13["first_return_label"].replace("|", "/")))
    W("| the sigma/a 8 record | formation %g r_s at R_t %.2f a; episodes %s; E0 %.4f | the s8 record json |" % (MK["s8"]["formation_record_clock_over_rs"], MK["s8"]["formation_R_t_a"], "; ".join("%g to %g (%d records)" % (e["T_start"], e["T_end"], e["n_records"]) for e in MK["s8"]["episodes"]), MK["s8"]["E0"]))
    W("")
    # ------------------------------------------------------------ T7: D13
    W("### T7. The D13 protocol's stored observables on B2 (kappa 1 and 2, weights 16 and 32), rows before the first return")
    W("")
    W("(a) The detector-energy ledger at the cuts 16 and 30 r_s per window (the windows ending at the readout clocks): %s. The agreement column is |crossing current - second term| over |second term|; the last column is the full-precision difference crossing jumps minus vacuum change, which is part of the balance and is kept." % D13["windows_definition"])
    W("")
    W("| window T/r_s | cut | u_start, u_end (r_s) | inventory_D start | inventory_D end | second term (D end - D start) | crossing current (C) | agreement (percent) | crossing jumps (C) | vacuum change | jumps - vacuum change | kappa 1 second term | kappa 2 second term | with kappa 3: second term |")
    W("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for w in D13["windows"]:
        if w["t1"] in READ_CLOCKS:
            st = w["second_term_D_12"]; cc = w["crossing_current_12"]
            W("| %g to %g | %s | %s, %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (w["t0"], w["t1"], w["cut"], f(w["u_start"], "%.1f"), f(w["u_end"], "%.1f"), f(w["inventory_D_start_12"], "%+.3e"), f(w["inventory_D_end_12"], "%+.3e"), f(st, "%+.4e"), f(cc, "%+.4e"), "-" if not st else "%.2f" % (100.0 * abs(cc - st) / abs(st)), f(w["crossing_jumps_12"], "%+.4e"), f(w["vacuum_change_12"], "%+.4e"), f((w["crossing_jumps_12"] - w["vacuum_change_12"]) if (isnum(w["crossing_jumps_12"]) and isnum(w["vacuum_change_12"])) else None, "%+.3e"), f(w["second_term_D_k1"], "%+.3e"), f(w["second_term_D_k2"], "%+.3e"), f(w["second_term_D_123"], "%+.3e")))
    W("")
    W("The deviation's inventory between the cuts 16 and 30 r_s at the window's start (kappa 1, 2; with kappa 3): " + "; ".join("%g: %s (%s)" % (b["t0"], f(b["inventory_D_between_12"], "%+.3e"), f(b["inventory_D_between_123"], "%+.3e")) for b in D13["between"] if b["t1"] in READ_CLOCKS) + ".")
    W("")
    W("(b) L_old and L_sea by the 16 back-propagated probes at the cuts 16 and 30 r_s, every probe clock (40 to 100 every 4 r_s): %s. L_H(kappa_T) of kappa 1, 2 at the clock beside; before the first return at every row (the first return at the 16 r_s cut is estimated at %.1f r_s, at 30 r_s %.1f). The ratio columns are |L_old| over |L| (the direct current) and, at the reference clock 12 r_s, |L_sea - L| over |L|." % (D13["probes_definition"].replace("|", "/"), D13["first_return_at_100"]["16"]["first_return_rs"], D13["first_return_at_100"]["30"]["first_return_rs"]))
    W("")
    W("| T/r_s | cut | L (direct, kappa 1, 2) | t_ref | L_old | L_sea | L_src | \\|L_old\\|/\\|L\\| | \\|L_sea - L\\|/\\|L\\| | kappa 1: L_old, L_sea | kappa 2: L_old, L_sea | L_H(kappa_T) (kappa 1, 2) | join max rel dv |")
    W("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in D13["probes"]:
        if True:
            for r in p["refs"]:
                Ld = p["L_direct_12"]
                W("| %g | %s | %s | %g | %s | %s | %s | %s | %s | %s, %s | %s, %s | %s | %s |" % (p["t"], p["cut"], f(Ld, "%+.3e"), r["t_ref"], f(r["L_old_12"], "%+.3e"), f(r["L_sea_12"], "%+.3e"), f(r["L_src_12"], "%+.3e"), f(abs(r["L_old_12"]) / abs(Ld) if (isnum(r["L_old_12"]) and Ld) else None, "%.2f"), f(abs(r["L_sea_12"] - Ld) / abs(Ld) if (isnum(r["L_sea_12"]) and Ld) else None, "%.3f"), f(r["L_old_k1"], "%+.2e"), f(r["L_sea_k1"], "%+.2e"), f(r["L_old_k2"], "%+.2e"), f(r["L_sea_k2"], "%+.2e"), f(FL["LH12_kT"][ci[p["t"]]], "%.3e"), f(FL["join_dv"][ci[p["t"]]], "%.2e")))
    W("")
    lo12 = [(p["t"], p["cut"], abs(r["L_old_12"]), abs(r["L_sea_12"] - p["L_direct_12"]) / abs(p["L_direct_12"])) for p in D13["probes"] for r in p["refs"] if r["t_ref"] == 12.0 and isnum(r["L_old_12"]) and p["L_direct_12"]]
    ratio40 = [(p["t"], abs(r["L_old_12"]) / abs(p["L_direct_12"])) for p in D13["probes"] for r in p["refs"] if r["t_ref"] == 40.0 and p["cut"] == "16" and p["t"] >= 72.0 and p["L_direct_12"]]
    ratio20 = [(p["t"], abs(r["L_old_12"]) / abs(p["L_direct_12"])) for p in D13["probes"] for r in p["refs"] if r["t_ref"] == 20.0 and p["cut"] == "16" and p["t"] >= 56.0 and p["L_direct_12"]]
    lo12b = [(p["t"], p["cut"], abs(r["L_old_12"]) / abs(p["L_direct_12"]), abs(r["L_sea_12"] - p["L_direct_12"]) / abs(p["L_direct_12"])) for p in D13["probes"] for r in p["refs"] if r["t_ref"] == 12.0 and abs(p["L_direct_12"]) > 1e-5]
    wl = max(lo12, key=lambda x: x[3])
    W("Over all 32 probes (16 clocks 40 to 100 every 4 r_s, 2 cuts): with the reference clock 12 r_s |L_old| exceeds 2e-7 at %d of the 32 and is at most %s (at %g r_s, the %s r_s cut, where L is %s); |L_sea - L|/|L| is at most %s (at %g r_s, the %s r_s cut, where L is %s) and at most %s over the %d probes where |L| exceeds 1e-5 (there |L_old|/|L| is at most %s); with the reference clock 40 r_s at the 16 r_s cut |L_old|/|L| is %s over 72 to 100 r_s (every 4 r_s), and with the reference clock 20 r_s it is %s over 56 to 100 r_s (%s at 56 r_s, where the join's departure is %s)." % (
        sum(1 for x in lo12 if x[2] > 2e-7), f(max(x[2] for x in lo12), "%.1e"), *max(lo12, key=lambda x: x[2])[:2], f([p["L_direct_12"] for p in D13["probes"] if (p["t"], p["cut"]) == tuple(max(lo12, key=lambda x: x[2])[:2])][0], "%.1e"), f(wl[3], "%.2f"), wl[0], wl[1], f([p["L_direct_12"] for p in D13["probes"] if (p["t"], p["cut"]) == (wl[0], wl[1])][0], "%.1e"), f(max(x[3] for x in lo12b), "%.3f"), len(lo12b), f(max(x[2] for x in lo12b), "%.3f"), rng([r for _, r in ratio40], "%.1f"), rng([r for _, r in ratio20], "%.1f"), f([r for t, r in ratio20 if t == 56.0][0], "%.0f"), f(FL["join_dv"][ci[56.0]], "%.1e")))
    W("")
    W("(c) Windowed changes of the hard and soft exterior-shell energy inventories per retarded window (the windows ending at the readout clocks): %s; the threshold 25 T_H^ref = %s (T_H^ref %s). These are signed inventory changes of the exterior Killing shells, not a crossing through a fixed detector." % (D13["hard_definition"], f(D13["hard_threshold_energy_at_40"], "%.4e"), f(D13["T_H_ref"], "%.4e")))
    W("")
    W("| window T/r_s | u_start, u_end | hard E out | hard E in | out + in (hard) | soft E out | soft E in | above-band E out | above-band E in | hard N out | hard E out (held vacuum) | soft E out (held vacuum) | with kappa 3: hard E out, soft E out |")
    W("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for h in D13["hard"]:
        if h["t1"] in READ_CLOCKS:
            W("| %g to %g | %s, %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s, %s |" % (h["t0"], h["t1"], f(h["u_start"], "%.1f"), f(h["u_end"], "%.1f"), f(h["delta_hard_E_out_12"], "%+.3e"), f(h["delta_hard_E_in_12"], "%+.3e"), f(h["delta_hard_E_out_12"] + h["delta_hard_E_in_12"] if (isnum(h["delta_hard_E_out_12"]) and isnum(h["delta_hard_E_in_12"])) else None, "%+.2e"), f(h["delta_soft_E_out_12"], "%+.3e"), f(h["delta_soft_E_in_12"], "%+.3e"), f(h["delta_above_E_out_12"], "%+.2e"), f(h["delta_above_E_in_12"], "%+.2e"), f(h["delta_hard_N_out_12"], "%+.3e"), f(h["delta_hard_E_out_P_12"], "%+.2e"), f(h["delta_soft_E_out_P_12"], "%+.2e"), f(h["delta_hard_E_out_123"], "%+.3e"), f(h["delta_soft_E_out_123"], "%+.3e")))
    W("")
    W("The identified source modes (the eigenvectors of C inside the region at the reference clock with eigenvalue at least 0.99, propagated with the sea without collection) and their same-packet transfer tau (summed over the set, its maximum over the set, and the average tau sum / modes) and inferred occupation at the collar (16 r_s), the cut (24 r_s) and the far radius (48 r_s), per channel; the last column is the set's occupation in the held vacuum of the reference clock (the stored occ_vac_extremes, smallest and largest over the set):")
    W("")
    W("| kappa | reference clock (region, modes) | at T/r_s | tau sum collar | tau max collar | tau average collar | occupation (collar) | tau sum cut | tau average cut | occupation (cut) | tau sum far | occupation (far) | vacuum occupation of the set (min, max) |")
    W("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k in ("1", "2"):
        defs = D13["sources"][k + "_definition"]
        for t, refs in D13["sources"][k].items():
            for ref, v in refs.items():
                dd = defs.get(ref) or {}; ov = dd.get("occ_vac_extremes") or [None, None]; n = v["n_modes"] or 0
                W("| %s | %g (%s; %s modes) | %g | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s, %s |" % (k, v["t_ref_over_rs"], dd.get("region"), n, float(t), f(v["tau_sum_collar"], "%.3f"), f(v["tau_max_collar"], "%.4f"), f(v["tau_sum_collar"] / n if n else None, "%.3f"), f(v["occ_inferred_mean_collar"], "%.4f"), f(v["tau_sum_cut"], "%.3f"), f(v["tau_sum_cut"] / n if n else None, "%.3f"), f(v["occ_inferred_mean_cut"], "%.4f"), f(v["tau_sum_far"], "%.3e"), f(v["occ_inferred_mean_far"], "%.4f"), f(ov[0], "%.6f"), f(ov[1], "%.6f")))
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
    W("The quench per channel (16 kappa (E_ctrl - E_prod) at 100 r_s): " + ", ".join("%s: %s" % (k, f((v or {}).get("100.0"), "%.2e")) for k, v in (IC.get("quench_per_channel") or {}).items()) + ". The control holds no 2 r_s cut and no channel above 8.")
    W("")
    W("### T8b. Run (a) of W1 ruling v2 section 2: the held vacuum's cut entropy under a sub-step field shift (kappa 18; the busi lane's ruling2/runA_v1_results.jsonl, as stored, every row)")
    W("")
    W("D_half and D_full are S_X^vac at the fields shifted by Delta_sub/2 and Delta_sub (the record interval over 8 sub-steps; the linear mode samples each sub-step at its midpoint, so Delta_sub/2 is the shift that applies) minus the stored S_X^vac at the record's fields; x_18 the stored kappa 18 excess (unweighted); the vacuum's reproduction at the record's fields ('repro') beside. The ruling's branch: D below x_18 by 10 or more where the weighted term 288 |x_18| exceeds 5e-4 reads the common mode as the exterior's physics, not the construction's reference lag.")
    W("")
    W("| T/r_s | cut | 288 x_18 (stored) | 288 D_half | 288 D_full | \\|x_18/D_half\\| | same sign | repro | dN/dt | dv/dt |")
    W("|---|---|---|---|---|---|---|---|---|---|")
    nsel = 0; ratios_sel = []; same = 0
    for row in sorted(D.get("runA") or [], key=lambda r: r.get("T_rs", 0)):
        for lab in CUTS:
            c = (row.get("cuts") or {}).get(lab)
            if not c:
                continue
            x = c.get("x18_stored"); dh = c.get("D_half"); df = c.get("D_full")
            r_ = abs(x / dh) if (isnum(x) and isnum(dh) and dh) else None
            if isnum(x) and abs(288.0 * x) > 5e-4 and r_ is not None:
                nsel += 1; ratios_sel.append(r_); same += 1 if (x * dh > 0) else 0
            W("| %g | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (row.get("T_rs"), lab, f(288.0 * x if isnum(x) else None, "%+.3e"), f(288.0 * dh if isnum(dh) else None, "%+.2e"), f(288.0 * df if isnum(df) else None, "%+.2e"), f(r_, "%.0f"), "yes" if (isnum(x) and isnum(dh) and x * dh > 0) else "no", f(c.get("repro"), "%.1e"), f(c.get("dN_dt"), "%+.2e"), f(c.get("dv_dt"), "%+.2e")))
    W("")
    if ratios_sel:
        W("Where 288 |x_18| exceeds 5e-4 (%d pairs): |x_18/D_half| is %s (median %s); the signs agree at %d of them. D_full is about 2 D_half (linear in the shift)." % (nsel, rng(ratios_sel, "%.0f"), f(sorted(ratios_sel)[len(ratios_sel) // 2], "%.0f"), same))
    else:
        W("No run (a) row was read.")
    W("")
    # ------------------------------------------------------------ T9: checks
    W("### T9. Consistency checks of this composition")
    W("")
    W("- This script's sums against the post-pass rows (K = 18, B2, %d pairs): largest |difference| in S %s, F_meas %s, c %s, F_apriori %s nats." % (CH["B2_sum_vs_postpass_max_abs"]["n_pairs"], f(CH["B2_sum_vs_postpass_max_abs"]["S"], "%.1e"), f(CH["B2_sum_vs_postpass_max_abs"]["F_meas"], "%.1e"), f(CH["B2_sum_vs_postpass_max_abs"]["c"], "%.1e"), f(CH["B2_sum_vs_postpass_max_abs"]["F_apriori"], "%.1e")))
    W("- The same for s8: largest |difference| in S %s nats." % f(CH["s8_sum_vs_postpass_max_abs_S"], "%.1e"))
    W("- The verdicts at 40 to 100 r_s of the busi composition (from 40 r_s) against this lane's rerun from 14 r_s: %d pairs differ%s." % (len(CH["B2_verdict_differences_from40_vs_from14_at_t_ge_40"]), (": " + "; ".join("%g/%s %s (from 40) against %s (from 14: %s)" % (d["t"], d["cut"], d["verdict_from40"], d["verdict_from14"], d.get("why_from14")) for d in CH["B2_verdict_differences_from40_vs_from14_at_t_ge_40"])) if CH["B2_verdict_differences_from40_vs_from14_at_t_ge_40"] else ""))
    W("- s8 at 40 r_s, the from-40 run against the from-14 run: %d pairs differ%s." % (len(CH["s8_verdict_differences_from40_vs_from14_at_40"]), (": " + "; ".join("%g/%s %s against %s" % (d["t"], d["cut"], d["verdict_from40"], d["verdict_from14"]) for d in CH["s8_verdict_differences_from40_vs_from14_at_40"])) if CH["s8_verdict_differences_from40_vs_from14_at_40"] else ""))
    W("- L_interval against the accumulator's increment per coordinate interval: largest relative deviation %s (the accumulator-difference definition; not a check of the exterior work or reference balance)." % f(CH["L_interval_identity_max_rel"], "%.1e"))
    lt = CH["light_travel_vs_v3"]
    W("- The light-travel formula against version 3's marks (the 16 r_s radius; the dimensionless travel over 2gE_f): %s and %s (this script) against %s and %s (version 3): differences %s and %s absolute, %s and %s relative; 2gE_f %s against %s (the B2 record's formation E against r11's)." % (f(lt["this_16_c2"], "%.6f"), f(lt["this_16_c1.5"], "%.6f"), f(lt["v3_16_c2"], "%.6f"), f(lt["v3_16_c1.5"], "%.6f"), f(lt["this_16_c2"] - lt["v3_16_c2"], "%.1e"), f(lt["this_16_c1.5"] - lt["v3_16_c1.5"], "%.1e"), f((lt["this_16_c2"] - lt["v3_16_c2"]) / lt["v3_16_c2"], "%.1e"), f((lt["this_16_c1.5"] - lt["v3_16_c1.5"]) / lt["v3_16_c1.5"], "%.1e"), f(lt["this_two_gE_formation"], "%.5f"), f(lt["v3_two_gE_formation"], "%.5f")))
    W("- The export page_v4_readout_export_v1b.tsv holds one row per readout clock and cut (155 rows from 40 r_s) with S, F_meas, F_apriori, c, K c, the last three weighted terms and their floors, q_x with its neighbours, the v1f tail, the probe terms, r_a, r_b, the criterion, the conditional majorant, the automatic verdict and the final tail status, for recomputing every filter of this delivery from stored data.")
    W("")
    with open(out, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("wrote %s (%d lines)" % (out, len(L)))
    if out_tsv:
        cols = ["t_over_rs", "cut_rs", "S", "F_meas", "F_apriori", "c", "K_c", "K_c_over_S", "term_16", "term_17", "term_18", "floor_16", "floor_17", "floor_18", "q_x", "q_x_prev", "q_x_next", "v1f_tail", "v1f_tail_kind", "probe_24", "probe_32", "probe_48", "r_a", "r_b", "same_sign", "criterion_met", "conditional_majorant", "conditional_majorant_over_S", "slope", "verdict_v1f", "reason_v1f", "final_tail_status", "D16_sub16", "source"]
        with open(out_tsv, "w") as fh:
            fh.write("\t".join(cols) + "\n")
            for (t, lab), r in sorted(RB.items()):
                if t < 40.0 - 1e-9:
                    continue
                i = ci[t]; j = CUTS.index(lab)
                nb = r.get("q_x_neighbours") or {}; ks = sorted(nb.keys(), key=float)
                qprev = nb[ks[0]] if ks and float(ks[0]) < t else None; qnext = nb[ks[-1]] if ks and float(ks[-1]) > t else None
                vals = [t, lab, r["S"], r["F_meas"], r["F_apriori"], r["c"], r["share"], r.get("share_over_S"), B["WX"][15][i][j], B["WX"][16][i][j], B["WX"][17][i][j], B["fmeas_w"][15][i][j], B["fmeas_w"][16][i][j], B["fmeas_w"][17][i][j], r.get("q_x"), qprev, qnext, r.get("x_tail"), r.get("x_tail_kind"),
                        PR["WX"][1][i][j], PR["WX"][2][i][j], PR["WX"][3][i][j], PR["r_a"][i][j], PR["r_b"][i][j], PR["same_sign"][i][j], PR["criterion_met"][i][j], PR["conditional_majorant"][i][j], PR["conditional_majorant_over_S"][i][j], PR["slope"][i][j], r["verdict"], r.get("why"), "OPEN", SB["D16_w"][i][j], r.get("source")]
                fh.write("\t".join("" if v is None or (isinstance(v, float) and math.isnan(v)) else (repr(v) if isinstance(v, float) else str(v)) for v in vals) + "\n")
        print("wrote %s" % out_tsv)


if __name__ == "__main__":
    main()
