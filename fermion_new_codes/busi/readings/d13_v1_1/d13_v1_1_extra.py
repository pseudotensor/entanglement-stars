#!/usr/bin/env python
"""Follow-up checks for the D13_classification_v1.1 reading, on d13_v1_1_recompute.json and the stored summaries: the join's range
over 14 to 56, the per-clock comparator extremes, the probe-window allowance on the 2 r_s checkpoint grid against the 4 r_s probe
grid, the (a) extremal windows, the inside-16 ratios per late window, the rows where the guard flips L_sea's sign, the back-propagation
check against the allowance, and whether PAGE v4.1 T5's |in|/|out| rows reproduce from the same stored collar_split block."""
import json

OUT = "<internal>/d13_v1_1"
R = json.load(open(OUT + "/d13_v1_1_recompute.json"))
col = R["collar"]
RS = 64.0
lines = []
P = lines.append
P("join min/max over 14..56: %.3e %.3e | over 0..56: %.3e" % (min(r["join_max_rel_dv"] for r in col if 14 <= r["t"] <= 56), max(r["join_max_rel_dv"] for r in col if 14 <= r["t"] <= 56), min(r["join_max_rel_dv"] for r in col if r["t"] <= 56)))
S1 = json.load(open("<internal>/v6o_B2_k1_summary.json")); S2 = json.load(open("<internal>/v6o_B2_k2_summary.json"))
r1 = {r["t_over_rs"]: r for r in S1["rows"]}; r2 = {r["t_over_rs"]: r for r in S2["rows"]}
LH = {t: r1[t]["L_H_kappaT"] + r2[t]["L_H_kappaT"] for t in r1 if r1[t].get("L_H_kappaT") and r2[t].get("L_H_kappaT")}
srt = sorted((v, t) for t, v in LH.items() if t >= 40)
P("L_H per clock 40..100: lowest %s; highest %s" % (srt[:4], srt[-3:]))
P("L_H at 14, 15, 16, 20: %s %s %s %s; None at: %s" % (LH.get(14.0), LH.get(15.0), LH.get(16.0), LH.get(20.0), [t for t in sorted(r1) if not r1[t].get("L_H_kappaT")]))


def trap(ts, vs):
    return sum(0.5 * (vs[i] + vs[i - 1]) * (ts[i] - ts[i - 1]) for i in range(1, len(ts)))


mins = []
for bw in R["probe_windows"]:
    ts = [t for t in sorted(LH) if bw["t0"] - 1e-9 <= t <= bw["t1"] + 1e-9]
    fine = trap([t * RS for t in ts], [LH[t] for t in ts])
    if bw["cut"] == "16" and bw["t_ref"] == 20.0:
        P("probe window %g-%g: L_H fluence on the 4 r_s probe grid %.4e, on the 2 r_s checkpoint grid %.4e, ratio %.4f" % (bw["t0"], bw["t1"], bw["LH_fl"], fine, fine / bw["LH_fl"]))
    mins.append((bw["err_total"] / (1e-4 * fine), bw["cut"], bw["t_ref"], bw["t0"]))
P("min total/allowance in (b) on the 2 r_s grid: %s; max: %s" % (min(mins), max(mins)))
A = R["ledger"]
for cut in ("16", "30"):
    rows = [r for r in A if r["cut"] == cut and r["allowance"]]
    m = min(rows, key=lambda r: r["ratio"]["kappa3"]); M = max(rows, key=lambda r: r["ratio"]["kappa3"])
    P("cut %s: kappa3/allow min at %g-%g: %.3f (k3 %.3e, allowance %.3e); max at %g-%g: %.3e" % (cut, m["t0"], m["t1"], m["ratio"]["kappa3"], m["second_term_k3"], m["allowance"], M["t0"], M["t1"], M["ratio"]["kappa3"]))
    mt = min(rows, key=lambda r: r["ratio_total"]); mi = min(rows, key=lambda r: r["ratio"]["interp"]); Mt = max(rows, key=lambda r: r["ratio_total"])
    P("   total min at %g-%g: %.2f, max at %g-%g: %.3e; interp min at %g-%g: %.3f" % (mt["t0"], mt["t1"], mt["ratio_total"], Mt["t0"], Mt["t1"], Mt["ratio_total"], mi["t0"], mi["t1"], mi["ratio"]["interp"]))
P("inside16/st per window 60..100 at 16: %s" % [(r["t0"], round(r["inside16_over_st"], 2)) for r in A if r["cut"] == "16" and r["t0"] >= 60])
P("inside16/st over 20..100 at 16, sorted: %s ... %s" % (sorted((round(abs(r["inside16_over_st"]), 3), r["t0"]) for r in A if r["cut"] == "16" and r["allowance"])[:3], sorted((round(abs(r["inside16_over_st"]), 1), r["t0"]) for r in A if r["cut"] == "16" and r["allowance"])[-3:]))
P("sign-not-robust rows (|k3 L_sea| >= |L_sea| fluence): %s" % [(b["cut"], b["t_ref"], b["t0"], "%.2e" % b["L_sea_fl"], "%.2e" % b["err"]["kappa3"]) for b in R["probe_windows"] if abs(b["k3_over_Lsea"]) >= 1])
P("k3/L_sea at 16 r_s t_ref 40: %s" % [(b["t0"], "%.1e" % b["k3_over_Lsea"]) for b in R["probe_windows"] if b["cut"] == "16" and b["t_ref"] == 40.0])
P("check/allowance range: %.1e to %.1e" % (min(b["err"]["backprop"] / b["allowance"] for b in R["probe_windows"]), max(b["err"]["backprop"] / b["allowance"] for b in R["probe_windows"])))
T5 = {14: 2.226, 20: 0.033, 30: 0.108, 36: 0.027, 38: 0.342, 40: 0.000, 42: 0.000, 44: 0.000, 46: 0.005, 48: 0.016, 50: 0.004, 54: 0.011, 56: 0.023, 60: 0.002, 70: 0.027, 78: 0.033, 80: 0.011, 90: 0.005, 92: 0.028, 100: 0.003}
bad = [(t, round(r["abs_in_over_abs_out"], 3), T5[t]) for r in col for t in [int(r["t"])] if t in T5 and abs(round(r["abs_in_over_abs_out"], 3) - T5[t]) > 0.0015]
P("PAGE v4.1 T5 |in|/|out| rows not reproduced from the stored collar_split: %s (of %d)" % (bad, len(T5)))
P("out, in at 14, 20, 30, 78: %s" % [(r["t"], "%.3e" % r["out"], "%.3e" % r["in"]) for r in col if r["t"] in (14.0, 20.0, 30.0, 78.0)])
# the 46-60 window: block-basis jumps inside it against the block-ground minus _P difference
J = [j for j in R["block_jumps"] if j["t"] in (50.0, 60.0) and j["k"] in (1, 2)]
P("block-ground hard-out jumps at 50 and 60, kappa 1+2: %.4f; the 46-60 window's block-ground minus _P hard out: %.4f" % (sum(j["d_hard_E_out_bg"] for j in J), [c["delta_hard_E_out_12"] - c["delta_hard_E_out_P_12"] for c in R["hard_soft"] if c["t0"] == 46.0][0]))
P("block-ground out+in jump cancellation at the block clocks (kappa 1): %s" % [(j["t"], "%.1e" % (j["d_hard_E_out_bg"] + j["d_hard_E_in_bg"])) for j in R["block_jumps"] if j["k"] == 1])
# the pre-50 block-ground window entries against the jumps they contain
P("block-ground hard out for the windows 6-20, 16-30, 26-40 (kappa 1+2): %s; the jumps at 10+20, 20+30, 30+40 (kappa 1+2): %s" % ([("%g-%g" % (c["t0"], c["t1"]), "%.4f" % c["delta_hard_E_out_12"]) for c in R["hard_soft"] if c["t0"] in (6.0, 16.0, 26.0)], [("%g+%g" % (a, b), "%.4f" % sum(j["d_hard_E_out_bg"] for j in R["block_jumps"] if j["t"] in (a, b) and j["k"] in (1, 2))) for a, b in ((10.0, 20.0), (20.0, 30.0), (30.0, 40.0))]))
# s8 probe at 30 r_s
P("s8 probes: %s" % R["s8_probes"])
# the L_H checkpoint mean vs trapezoid mean for the (a) windows from 40
P("(a) windows from 40: checkpoint-mean L_H range %s; trapezoid-mean range %s" % ((min(r["LH_mean_checkpoints"] for r in A if r["allowance"] and r["t0"] >= 40), max(r["LH_mean_checkpoints"] for r in A if r["allowance"] and r["t0"] >= 40)), (min(r["LH_mean_trapezoid"] for r in A if r["allowance"] and r["t0"] >= 40), max(r["LH_mean_trapezoid"] for r in A if r["allowance"] and r["t0"] >= 40))))
open(OUT + "/d13_v1_1_extra.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
