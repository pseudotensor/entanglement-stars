#!/usr/bin/env python
"""page_v6_regression_table_v2.py (builder analyst, 2026-10-16; S1 E, the v6 regression on the pilot's prescribed background, stage 0
uncapped; SPEC_final4b_engine.md 37.1 "Ruled" (3)): page_v6_regression_table_v1.py with inbox #41 (D15): no mutual information is printed
(the regression target "MI 7.365/2.492/0.627" is dropped; the other stage-0 targets stand), and with the regression's two code versions
named: R0, R0s and R1 ran page_recursion_v6.py to block 10 (t 40 r_s), where v6's write_md failed on a None check, and resumed from their own
block-10 checkpoints under page_recursion_v6b.py (the same state code); R3 ran page_recursion_v6.py to its end (exit 0). The attribution's
largest block differences are printed for the blocks up to 10 and after 10 separately, so a resume effect would show. VALIDATION, not the
paper's Page computation.

Inputs (RUNS_DIR as fetch_ctrl32_v5_v2.sh and fetch_page_v6_v2.sh lay it out): page/runs/ctrl32_v5_baseline/ctrl32_v5_baseline_summary.json
(v5, uncapped) and page/runs/v6reg_<R0|R0s|R1|R3>/v6reg_<name>_summary.json (fetch_page_v6_v2.sh's copies, their mutual-information entries
removed). Every number is printed by this script. Usage: python page_v6_regression_table_v2.py RUNS_DIR OUT_MD"""
import datetime
import json
import os
import sys

NAMES = ("R0", "R0s", "R1", "R3")


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
    sr = (s.get("stage_readouts") or [{}])[0]
    return {"S_max": mx["S_esc_total"], "t_S_max": mx["t_over_rs1"], "S_esc_200": r200["S_esc_total"], "t_200": r200["t_over_rs1"],
            "E_rad_200": r200["E_rad_stage"], "E_init": st.get("E_init_weighted"), "collar_200": r200["E_collar_integrated_stage"],
            "T_m_over_T_H": sr.get("T_m_over_T_H"), "band_end": cv[-1].get("S_esc_band", cv[-1].get("uncertainty_total")),
            "blocks": s.get("n_blocks"), "stop": s.get("stop_reason"), "final": s.get("final"), "repairs": s.get("repairs"), "code": s.get("code", "page_recursion_v5.py"),
            "ledger": {k: {"v5": v.get("entropy_bound_dropped_pairs_v5"), "sharp": v.get("entropy_bound_dropped_pairs_sharp"), "used": v.get("entropy_bound_dropped_pairs"),
                           "n_sines": v.get("dropped_sines_n")} for k, v in (s.get("inversion") or {}).items()},
            "S_curve": {round(r["t_over_rs1"], 6): r["S_esc_total"] for r in cv}, "E_curve": {round(r["t_over_rs1"], 6): r["E_rad_stage"] for r in cv},
            "n_curve": len(cv), "checks": s.get("checks") or {}, "validation": (s.get("validation") or {}).get("status")}


def maxdiff(x, y, key, sel):
    common = sorted(t for t in set(x[key]) & set(y[key]) if sel(t))
    return max((abs(x[key][t] - y[key][t]) for t in common), default=None), len(common)


def main():
    runs, out = sys.argv[1:3]
    base = load(os.path.join(runs, "page", "runs", "ctrl32_v5_baseline", "ctrl32_v5_baseline_summary.json"))
    V = {n: load(os.path.join(runs, "page", "runs", "v6reg_%s" % n, "v6reg_%s_summary.json" % n)) for n in NAMES}
    R = {"baseline": readouts(base)}
    R.update({n: readouts(s) for n, s in V.items()})
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    L = ["# page_*_v6 regression against the uncapped v5 baseline control (S1 E; engine spec 37.1 (3)): VALIDATION, not the paper's Page computation\n",
         "Written %s by code/page/page_v6_regression_table_v2.py from %s. No mutual information is printed (inbox #41, D15: the regression target "
         "\"MI 7.365/2.492/0.627\" is dropped; the other stage-0 targets stand). R0, R0s and R1 ran page_recursion_v6.py to block 10 (t 40 r_s), where "
         "v6's write_md failed on a None check (g_sel_min with the v5 collector), and resumed from their own block-10 checkpoints under "
         "page_recursion_v6b.py (the same state code; the output writer fixed, no mutual information); R3 ran page_recursion_v6.py to its end. "
         "Every number is printed by the script.\n" % (now, runs),
         "## 1. The readouts per run\n",
         "| run | code (summary) | repairs | final, stop, blocks (curve rows) | S_max | t at S_max /r_s1 | S_esc at 200 r_s | E_rad at 200 r_s | E_init | E_rad/E_init | collar integral at 200 r_s | T_m/T_H | band at the end | validation |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in ("baseline",) + NAMES:
        x = R.get(n)
        if x is None:
            L.append("| %s | - | - | not landed | - | - | - | - | - | - | - | - | - | - |" % n)
            continue
        rp = x["repairs"] or {"hull": "v5", "pair_ledger": "v5", "collector": "v5", "direct_every": 5, "end_rule": "v5"}
        L.append("| %s | %s | hull %s, ledger %s, collector %s, direct every %s | %s, %s, %s (%d) | %.6f | %.1f | %.6f | %.6e | %s | %s | %.6e | %s | %s | %s |" % (
            n, x["code"], rp.get("hull"), rp.get("pair_ledger"), rp.get("collector"), rp.get("direct_every"), x["final"], x["stop"], x["blocks"], x["n_curve"],
            x["S_max"], x["t_S_max"], x["S_esc_200"], x["E_rad_200"], g(x["E_init"]), g(x["E_rad_200"] / x["E_init"] if x["E_init"] else None, "%.6f"),
            x["collar_200"], g(x["T_m_over_T_H"], "%.6f"), g(x["band_end"], "%.3e"), x["validation"]))
    L.append("\n## 2. The attribution (differences of the readouts; the largest |difference| over the common block times, up to t 40 r_s and after)\n")
    L.append("| comparison | attributed to | dS_max | dS_esc at 200 | dE_rad at 200 | dcollar at 200 | max dS_esc, t <= 40 (blocks) | max dS_esc, t > 40 (blocks) | max dE_rad, t <= 40 | max dE_rad, t > 40 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    pairs = [("R0", "baseline", "the v6 code in v5 mode (zero expected; R0 resumed at t 40 under v6b)"),
             ("R0s", "R0", "the resume from R0's stored formation (roundoff expected)"),
             ("R1", "R0s", "repair 1, the exact hull"),
             ("R3", "R1", "repair 3, the active-subspace collector (repairs 2 and 4 do not enter the state)")]
    early = lambda t: t <= 40.0 + 1e-9
    late = lambda t: t > 40.0 + 1e-9
    for a_, b_, lab in pairs:
        x, y = R.get(a_), R.get(b_)
        if x is None or y is None:
            L.append("| %s - %s | %s | (not landed) | | | | | | | |" % (a_, b_, lab))
            continue
        dSe, ne = maxdiff(x, y, "S_curve", early); dSl, nl = maxdiff(x, y, "S_curve", late)
        dEe, _ = maxdiff(x, y, "E_curve", early); dEl, _ = maxdiff(x, y, "E_curve", late)
        L.append("| %s - %s | %s | %s | %s | %s | %s | %s (%d) | %s (%d) | %s | %s |" % (
            a_, b_, lab, g(x["S_max"] - y["S_max"], "%+.3e"), g(x["S_esc_200"] - y["S_esc_200"], "%+.3e"), g(x["E_rad_200"] - y["E_rad_200"], "%+.3e"),
            g(x["collar_200"] - y["collar_200"], "%+.3e"), g(dSe, "%.3e"), ne, g(dSl, "%.3e"), nl, g(dEe, "%.3e"), g(dEl, "%.3e")))
    L.append("\n## 3. Repair 2, the dropped-pair ledger (from the stored sines; it changes the band only)\n")
    for n in ("R0", "R3"):
        x = R.get(n)
        if x is None:
            continue
        L.append("- %s: %s" % (n, "; ".join("kappa %s: v5 sum h(s^2) %s, sharp %s (%s sines), used %s" % (k, g(v["v5"]), g(v["sharp"]), v["n_sines"], g(v["used"]))
                                             for k, v in x["ledger"].items())))
    x = R.get("R3")
    if x is not None:
        c = x["checks"]
        L.append("\n## 4. R3's certificates (the active collector)\n")
        L.append("- exact Z identity max %s; all-history Gram defect max %s; the collected modes' smallest outside weight %s; the remaining outside weight max %s; "
                 "D D^dagger defect max %s; the collar identity's largest residual over E_init %s at %s checkpoints; the direct route: abs %s nats, the revised ratio %s; "
                 "the energy identity %s." % (
                     g(c.get("Z_identity_exact_max"), "%.3e"), g(c.get("gram_all_history_max"), "%.3e"), g(c.get("g_sel_min"), "%.6f"), g(c.get("outside_weight_remaining_max"), "%.4e"),
                     g(c.get("DDdag_defect_max"), "%.3e"), g(c.get("collar_identity_residual_max_rel"), "%.3e"), c.get("collar_identity_checkpoints"), g(c.get("direct_abs_max"), "%.3e"),
                     g(c.get("direct_revised_ratio_max"), "%.3e"), g(c.get("energy_identity_max_rel"), "%.3e")))
    with open(out, "w") as f:
        f.write("\n".join(L) + "\n")
    print("wrote %s: %s" % (out, ", ".join("%s %s" % (n, "landed" if R.get(n) else "-") for n in ("baseline",) + NAMES)))


if __name__ == "__main__":
    main()
