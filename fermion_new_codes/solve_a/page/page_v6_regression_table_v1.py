#!/usr/bin/env python
"""page_v6_regression_table_v1.py (builder analyst, 2026-10-16; SPEC_final4b_engine.md 37.1 "Ruled" (3)): the regression of page_*_v6
against the uncapped v5 baseline control, every change attributed to a named repair. VALIDATION, not the paper's Page computation (#32).

Inputs (results/page/ctrl32_v5_runs/ as fetch_ctrl32_v5_v2.sh and fetch_page_v6_regression_v1.sh copy them): the baseline control's summary
(v5, uncapped), and the regression variants' summaries v6reg_R0 (v6 with every repair off, from the formation), v6reg_R0s (the same from
R0's block-0 checkpoint: the shared formation's resume), v6reg_R1 (repair 1, the exact hull), v6reg_R3 (repairs 1 to 4). The readouts of
37.1's Facts: S_max and its time, E_rad and the collar integral at 200 r_s against E_init, S_esc at 200 r_s, T_m/T_H, the mutual information
across r_t at the end (d 2, 4, 8, weighted), the band; and every block's S_esc. Attribution: R0 - baseline is the v6 code in v5 mode (zero
expected), R0s - R0 the resume from the stored formation (roundoff expected), R1 - R0s repair 1, R3 - R1 repair 3 (repairs 2 and 4 do not
enter the state: repair 2 is the ledger, printed from the stored sines; repair 4 reads only). Every number is printed by this script.
Usage: python page_v6_regression_table_v1.py RUNS_DIR OUT_MD"""
import datetime
import json
import os
import sys


def load(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def g(x, fmt="%.6e"):
    if x is None:
        return "-"
    try:
        return fmt % x
    except (TypeError, ValueError):
        return str(x)


def at_t(s, t_rs):
    for r in s["curve"]:
        if abs(r["t_over_rs1"] - t_rs) < 1e-9:
            return r
    return None


def readouts(s):
    if s is None or not s.get("curve"):
        return None
    cv = s["curve"]
    mx = max(cv, key=lambda r: r["S_esc_total"])
    r200 = at_t(s, 200.0) or cv[-1]
    st = (s.get("stages") or [{}])[-1]
    E_init = st.get("E_init_weighted")
    mi = {}
    for k, rd in (s.get("readouts") or {}).items():
        for c in rd.get("cross_surface") or []:
            mi[c["d"]] = mi.get(c["d"], 0.0) + 16.0 * float(k) * c["physical"]["I"]
    sr = (s.get("stage_readouts") or [{}])[0]
    return {"S_max": mx["S_esc_total"], "t_S_max": mx["t_over_rs1"], "S_esc_200": r200["S_esc_total"], "t_200": r200["t_over_rs1"],
            "E_rad_200": r200["E_rad_stage"], "E_init": E_init, "collar_200": r200["E_collar_integrated_stage"],
            "T_m_over_T_H": sr.get("T_m_over_T_H"), "MI_end": mi, "band_end": cv[-1].get("S_esc_band", cv[-1].get("uncertainty_total")),
            "blocks": s.get("n_blocks"), "stop": s.get("stop_reason"), "final": s.get("final"), "repairs": s.get("repairs"),
            "ledger": {k: {"v5": v.get("entropy_bound_dropped_pairs_v5"), "sharp": v.get("entropy_bound_dropped_pairs_sharp"), "used": v.get("entropy_bound_dropped_pairs"),
                           "n_sines": v.get("dropped_sines_n")} for k, v in (s.get("inversion") or {}).items()},
            "S_curve": {round(r["t_over_rs1"], 6): r["S_esc_total"] for r in cv}, "E_curve": {round(r["t_over_rs1"], 6): r["E_rad_stage"] for r in cv},
            "checks": s.get("checks") or {}, "validation": (s.get("validation") or {}).get("status")}


def main():
    runs, out = sys.argv[1:3]
    base = load(os.path.join(runs, "page", "runs", "ctrl32_v5_baseline", "ctrl32_v5_baseline_summary.json"))
    V = {n: load(os.path.join(runs, "page", "runs", "v6reg_%s" % n, "v6reg_%s_summary.json" % n)) for n in ("R0", "R0s", "R1", "R3")}
    R = {"baseline": readouts(base)}
    R.update({n: readouts(s) for n, s in V.items()})
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    L = ["# page_*_v6 regression against the uncapped v5 baseline control (engine spec 37.1 (3)): VALIDATION, not the paper's Page computation\n",
         "Written %s by code/page/page_v6_regression_table_v1.py from %s. Inbox #32 (the owner disputes the Page conclusion): reviewer A's ruling pending; nothing here prints for item 7. Every number is printed by the script.\n" % (now, runs),
         "## 1. The readouts of 37.1's Facts per run\n",
         "| run | repairs | final, stop, blocks | S_max | t at S_max /r_s1 | S_esc at 200 r_s | E_rad at 200 r_s | E_init | E_rad/E_init | collar integral at 200 r_s | T_m/T_H | MI at the end d 2, 4, 8 (weighted) | band at the end | validation |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in ("baseline", "R0", "R0s", "R1", "R3"):
        x = R.get(n)
        if x is None:
            L.append("| %s | - | not landed | - | - | - | - | - | - | - | - | - | - | - |" % n); continue
        rp = x["repairs"] or {"hull": "v5", "pair_ledger": "v5", "collector": "v5", "direct_every": 5, "end_rule": "v5"}
        L.append("| %s | hull %s, ledger %s, collector %s, direct every %s | %s, %s, %s | %.6f | %.1f | %.6f | %.6e | %s | %s | %.6e | %s | %s | %s | %s |" % (
            n, rp.get("hull"), rp.get("pair_ledger"), rp.get("collector"), rp.get("direct_every"), x["final"], x["stop"], x["blocks"], x["S_max"], x["t_S_max"],
            x["S_esc_200"], x["E_rad_200"], g(x["E_init"]), g(x["E_rad_200"] / x["E_init"] if x["E_init"] else None, "%.6f"), x["collar_200"],
            g(x["T_m_over_T_H"], "%.6f"), ", ".join("%.6f" % x["MI_end"][d] for d in sorted(x["MI_end"])) or "-", g(x["band_end"], "%.3e"), x["validation"]))
    L.append("\n## 2. The attribution (differences of the readouts; the largest |difference| of S_esc over the common block times)\n")
    L.append("| comparison | attributed to | dS_max | dS_esc at 200 | dE_rad at 200 | dcollar at 200 | max over blocks of dS_esc | max over blocks of dE_rad |")
    L.append("|---|---|---|---|---|---|---|---|")
    pairs = [("R0", "baseline", "the v6 code in v5 mode (zero expected)"), ("R0s", "R0", "the resume from R0's stored formation (roundoff expected)"),
             ("R1", "R0s", "repair 1, the exact hull"), ("R3", "R1", "repair 3, the active-subspace collector (repairs 2 and 4 do not enter the state)")]
    for a_, b_, lab in pairs:
        x, y = R.get(a_), R.get(b_)
        if x is None or y is None:
            L.append("| %s - %s | %s | (not landed) |||||| " % (a_, b_, lab)); continue
        common = sorted(set(x["S_curve"]) & set(y["S_curve"]))
        dS = max((abs(x["S_curve"][t] - y["S_curve"][t]) for t in common), default=None)
        dE = max((abs(x["E_curve"][t] - y["E_curve"][t]) for t in common), default=None)
        L.append("| %s - %s | %s | %s | %s | %s | %s | %s | %s |" % (a_, b_, lab, g(x["S_max"] - y["S_max"], "%+.3e"), g(x["S_esc_200"] - y["S_esc_200"], "%+.3e"),
                                                              g(x["E_rad_200"] - y["E_rad_200"], "%+.3e"), g(x["collar_200"] - y["collar_200"], "%+.3e"), g(dS, "%.3e"), g(dE, "%.3e")))
    L.append("\n## 3. Repair 2, the dropped-pair ledger (from the stored sines; it changes the band only)\n")
    for n in ("R0", "R3"):
        x = R.get(n)
        if x is None:
            continue
        L.append("- %s: %s" % (n, "; ".join("kappa %s: v5 sum h(s^2) %s, sharp %s (%s sines), used %s" % (k, g(v["v5"]), g(v["sharp"]), v["n_sines"], g(v["used"])) for k, v in x["ledger"].items())))
    x = R.get("R3")
    if x is not None:
        c = x["checks"]
        L.append("\n## 4. R3's certificates (the active collector)\n")
        L.append("- exact Z identity max %s; all-history Gram defect max %s; the collected modes' smallest outside weight %s; the remaining outside weight max %s; D D^dagger defect max %s; the collar identity's largest residual over E_init %s at %s checkpoints; the direct route: abs %s nats, the revised ratio %s; the energy identity %s." % (
            g(c.get("Z_identity_exact_max"), "%.3e"), g(c.get("gram_all_history_max"), "%.3e"), g(c.get("g_sel_min"), "%.6f"), g(c.get("outside_weight_remaining_max"), "%.4e"),
            g(c.get("DDdag_defect_max"), "%.3e"), g(c.get("collar_identity_residual_max_rel"), "%.3e"), c.get("collar_identity_checkpoints"), g(c.get("direct_abs_max"), "%.3e"),
            g(c.get("direct_revised_ratio_max"), "%.3e"), g(c.get("energy_identity_max_rel"), "%.3e")))
    with open(out, "w") as f:
        f.write("\n".join(L) + "\n")
    print("wrote %s: %s" % (out, ", ".join("%s %s" % (n, "landed" if R.get(n) else "-") for n in ("baseline", "R0", "R0s", "R1", "R3"))))


if __name__ == "__main__":
    main()
