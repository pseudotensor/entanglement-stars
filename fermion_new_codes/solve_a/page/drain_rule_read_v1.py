#!/usr/bin/env python
"""drain_rule_read_v1.py (the lane's analyst builder; inbox #62 item 1, requirements_new.txt sec. 5 P1 as amended): reads the fixed-field
controls c15 and c30 (A's channel with the fields held at the records of T/r_s 15 and 30; page_history_v6e, tags v6eA_c15 and v6eA_c30) against the
amended drain rule on their existing rows before the chain end's reflection clock (the summary's reflection_clock, elapsed hold time), clause by
clause: (1) the held excitation conserved (E_inv over the hold, its largest relative change); (2) the energy identity closed at every checkpoint
(its largest residual); (3) no return: the incoming part at the collar below 1 percent of the outgoing on every pre-reflection row (the signed
split) -- the stored rows carry only the net collar flux L and its accumulation, so this clause is reported as not readable, with the rows where
the net flux is negative printed as a diagnostic only; (4) the collar flux's envelope (L/L_H at the back-averaged slope; the upper envelope as the
largest L/L_H over the remaining pre-reflection rows, printed; the test: the medians of L/L_H over the 4 quarters of the pre-reflection rows
falling from the second quarter on and a line over the last half falling) falling or not, and where not falling the excitation inside the collar with its
linear-extrapolation emptying clock (a least-squares line over the last half of the pre-reflection rows); (5) the complete drain (the excitation
inside the collar below 1 percent of its value at the hold's start) or 'a drain in progress'. Every number printed by the script; no run.
Usage: python drain_rule_read_v1.py RUNS_DIR OUT_MD OUT_JSON."""
import json
import sys
import time

import numpy as np


def read(path, tag):
    d = json.load(open("%s/%s/%s_summary.json" % (path, tag, tag)))
    rows = d["rows"]; rc = d["reflection_clock"]
    t0 = rows[0]["t_over_rs"]; t_refl = t0 + rc["T_refl_rs"]
    pre = [r for r in rows if r["t_over_rs"] < t_refl]
    out = {"tag": tag, "hold_start_rs": t0, "reflection_clock_rs": t_refl, "reflection_label": rc.get("label"), "rows": len(rows), "pre_rows": len(pre),
           "end_reason": d.get("end_reason"), "final": d.get("final")}
    Einv = np.array([r["E_inv_weighted"] for r in rows])
    out["c1_Einv_start"] = float(Einv[0]); out["c1_Einv_max_rel_change"] = float(np.max(np.abs(Einv - Einv[0])) / abs(Einv[0]))
    res = [abs(r["energy_identity_residual_weighted"]) for r in pre if r.get("energy_identity_residual_weighted") is not None]
    cres = [abs(r["collar_identity_residual_weighted"]) for r in pre if r.get("collar_identity_residual_weighted") is not None]
    out["c2_energy_identity_max_abs"] = max(res) if res else None
    out["c2_collar_identity_max_abs"] = max(cres) if cres else None
    out["c2_relative_to_Einv"] = (max(res) / abs(Einv[0])) if res else None
    L = [(r["t_over_rs"], r["L_at_t"], r["L_H_kappaT"], r.get("L_H_TH")) for r in pre if r.get("L_at_t") is not None]
    neg = [(t, l / lh) for t, l, lh, _ in L if l < 0]
    out["c3_readable"] = False
    out["c3_note"] = ("not readable: the stored rows hold the net collar flux and its accumulation only, no incoming/outgoing split at the collar; "
                      "the rows with a negative net flux before the reflection clock are a diagnostic, not the clause")
    out["c3_negative_net_rows"] = [[t, x] for t, x in neg]
    ratio = np.array([l / lh for _, l, lh, _ in L]); tt = np.array([t for t, _, _, _ in L])
    env = np.array([ratio[i:].max() for i in range(len(ratio))])
    out["c4_ratio_range"] = [float(ratio.min()), float(ratio.max())]
    h = len(ratio) // 2
    out["c4_ratio_range_last_half"] = [float(ratio[h:].min()), float(ratio[h:].max())]
    out["c4_ratio_quartiles_clock"] = [[float(tt[i]), float(ratio[i]), float(env[i])] for i in np.linspace(0, len(ratio) - 1, 9).astype(int)]
    q = np.array_split(np.arange(len(ratio)), 4)
    out["c4_quarter_medians"] = [float(np.median(ratio[i])) for i in q]
    out["c4_last_half_slope_per_rs"] = float(np.polyfit(tt[h:], ratio[h:], 1)[0])
    out["c4_envelope_falling"] = bool(out["c4_quarter_medians"][3] < out["c4_quarter_medians"][2] < out["c4_quarter_medians"][1] and out["c4_last_half_slope_per_rs"] < 0)
    ins = np.array([r["E_inv_inside_weighted"] for r in pre]); tin = np.array([r["t_over_rs"] for r in pre])
    hh = len(ins) // 2
    b, a0 = np.polyfit(tin[hh:], ins[hh:], 1)
    out["c4_inside_start"] = float(ins[0]); out["c4_inside_last_pre"] = float(ins[-1]); out["c4_inside_fraction_left"] = float(ins[-1] / ins[0])
    out["c4_inside_slope_per_rs"] = float(b)
    out["c4_emptying_clock_rs"] = (float(-a0 / b) if b < 0 else None)
    out["c4_emptying_after_last_pre_rs"] = (float(-a0 / b - tin[-1]) if b < 0 else None)
    out["c4_reservoir_location_note"] = ("the stored rows split the excitation at the collar only (inside, beyond); its location behind an inner "
                                         "surface is not in the rows")
    done = bool(ins[-1] < 0.01 * ins[0])
    out["c5_complete_drain"] = done
    out["c5_reading"] = "the complete drain" if done else "a drain in progress"
    out["S_esc_peak_pre"] = float(max(r["S_esc_weighted"] for r in pre)); out["S_esc_last_pre"] = float(pre[-1]["S_esc_weighted"])
    return out


def main():
    runs, md, js = sys.argv[1], sys.argv[2], sys.argv[3]
    R = [read(runs, t) for t in ("v6eA_c15", "v6eA_c30")]
    L = ["# The amended drain rule read on the fixed-field controls c15 and c30 (inbox #62 item 1; requirements_new.txt sec. 5 P1 as amended)", "",
         "Composed %s by code/page/drain_rule_read_v1.py from %s/v6eA_c15 and v6eA_c30 (page_history_v6e summaries); every number printed by the "
         "script; the rows read are those before each run's reflection clock (the hold's start plus the chain end's light round trip from the summary). "
         "No run." % (time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), runs), ""]
    for o in R:
        L += ["## %s (the fields held at the record of T/r_s %.0f)" % (o["tag"], o["hold_start_rs"]), "",
              "- rows %d, before the reflection clock T/r_s %.3f: %d; the run's end: %s" % (o["rows"], o["reflection_clock_rs"], o["pre_rows"], o["end_reason"]),
              "- clause 1, the held excitation conserved: E_inv %.6e at the hold's start, largest relative change over the hold %.1e: met" % (o["c1_Einv_start"], o["c1_Einv_max_rel_change"]),
              "- clause 2, the energy identity closed: largest |residual| before the reflection clock %.1e (%.1e of E_inv); the collar identity %s: met" % (
                  o["c2_energy_identity_max_abs"], o["c2_relative_to_Einv"], "%.1e" % o["c2_collar_identity_max_abs"] if o["c2_collar_identity_max_abs"] is not None else "not stored"),
              "- clause 3, no return (incoming below 1 percent of outgoing at the collar): %s; rows with a negative net flux before the reflection clock: %d%s" % (
                  o["c3_note"], len(o["c3_negative_net_rows"]), (" (T/r_s and L/L_H: " + ", ".join("%.3f %.3g" % (t, x) for t, x in o["c3_negative_net_rows"][:8]) + ")") if o["c3_negative_net_rows"] else ""),
              "- clause 4, the collar flux's envelope: L/L_H(kappa_T) before the reflection clock %.4g to %.4g (last half %.4g to %.4g); medians by quarter %s; "
              "a line over the last half %.3e per r_s; falling: %s; "
              "at 9 clocks (T/r_s, L/L_H, envelope): %s" % (o["c4_ratio_range"][0], o["c4_ratio_range"][1], o["c4_ratio_range_last_half"][0], o["c4_ratio_range_last_half"][1],
                                                            ", ".join("%.4g" % x for x in o["c4_quarter_medians"]), o["c4_last_half_slope_per_rs"], "yes" if o["c4_envelope_falling"] else "no", "; ".join("%.1f %.3g %.3g" % tuple(x) for x in o["c4_ratio_quartiles_clock"])),
              "- the excitation inside the collar: %.6e at the start, %.6e at the last row before the reflection clock (%.4f of it); a line over the last half of those rows "
              "falls at %.3e per r_s, reaching zero at T/r_s %s (%s r_s after that row; a linear extrapolation); %s" % (
                  o["c4_inside_start"], o["c4_inside_last_pre"], o["c4_inside_fraction_left"], o["c4_inside_slope_per_rs"],
                  "%.0f" % o["c4_emptying_clock_rs"] if o["c4_emptying_clock_rs"] else "never", "%.0f" % o["c4_emptying_after_last_pre_rs"] if o["c4_emptying_after_last_pre_rs"] else "-",
                  o["c4_reservoir_location_note"]),
              "- clause 5: %s (the excitation inside the collar %s 1 percent of its start before the reflection clock); S_esc's largest before the reflection clock %.6g, at its last row %.6g" % (
                  o["c5_reading"], "below" if o["c5_complete_drain"] else "not below", o["S_esc_peak_pre"], o["S_esc_last_pre"]), ""]
    L += ["## What the no-return clause needs", "",
          "The signed split at the collar on every row before the reflection clock: the collar flux's incoming and outgoing parts by the sign of the radial "
          "velocity within the collar shell (the windowed sign split D uses beyond 24 r_s), which needs the covariance at each row. Neither hold stored its "
          "states (pod 1's run directories hold the summary and tables only), so the split cannot be computed from the existing files. It needs the producer "
          "to save the split per row (or the covariance at the rows) and the holds re-run: the same holds on the same chain with the split added, or the "
          "ruled rerun on 16384 sites with the absorber (#62 item 2), with the split saved there. A choice for the lead.", ""]
    open(md, "w").write("\n".join(L))
    json.dump(R, open(js, "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
