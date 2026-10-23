#!/usr/bin/env python
"""s11_pathfinder_report_v1.py (builder analyst; S11 of the quasi-frozen ruling, owner-approved (inbox #36)): composes
MASSLESS_ball_pathfinder_v1.md from the S11 run's own files, every number printed by this script from them:
- the engine_collapse_v33 run (record json with the head's preparation and the per-record "s11" entries, the fields npz, the s11 npz);
- the mass-label characteristic solve's json (engine_masslabel_char_v1.py on the run's T = 0 profile; the continuum control);
- page_history_v6c's summary jsons on the S11 history for the collectors R_cut 8 r_s and 16 r_s (when they exist);
- the S5 (d) charge ledger (engine_replay_v9.ledger_terms through engine_massive_channel_D_v2.ledger_at) at the listed clocks and the
  density route beyond 8 and 16 r_s at every record (engine_massive_channel_D_v2.density_route), the ball and the pairs apart.
The ruling's items: whether the preparation stays inside the chart's domain (M(r) >= 0 outside the central averaging cell; the stop and its
case), whether and where a trapping zero of N + v forms and for how many records it persists, the collar charge against the emitted
energy through the collar identity with -Delta W, E_inv and the creation rate, the collected spectrum, S_esc, and the massless matter's own
weight beyond the collar per record. The Page code's built-in label names the massive history (page_history_v6c's LABEL_A); on this history
it is relabelled here: channels 1 and 2 of the same massless operator, the sea propagated from its ground state at T = 0 on the record's
applied fields (in those channels the engine's own sea, whose sources the engine feeds back), the ball's matter not in it.
Usage: s11_pathfinder_report_v1.py --run S11ball/<tag> [--char char.json] [--page8 a_summary.json] [--page16 b_summary.json]
[--ledger-T 0,1,2,...] --md out.md --out-json out.json"""
import argparse
import importlib
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine_massive_channel_D_v2 as DV2  # noqa: E402

LABEL = ("the model's own massless object at the pathfinder size: the self-consistent massless collapse (S11), r_s/a 16, G/a^2 1/4, m 0, "
         "ell 2a (ell/2GE 0.125, below the critical width 0.26), tau 2; a radial channel reduction with weights 16 kappa (kappa 1 to 10)")
fmt = DV2.fmt


def F_in(R, rs):
    return R - 2.0 * math.sqrt(rs * R) + 2.0 * rs * math.log(1.0 + math.sqrt(R / rs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--char", default=None)
    ap.add_argument("--page8", default=None)
    ap.add_argument("--page16", default=None)
    ap.add_argument("--gate", default=None, help="compare_v33_gate_v1's printed result (the v32/v33 gate)")
    ap.add_argument("--ledger-T", dest="ledger_T", default=None, help="T/r_s clocks for the S5 (d) ledger (default every 2 r_s and the last record)")
    ap.add_argument("--every", type=int, default=1, help="print every n-th record in the per-record table (all in the json)")
    ap.add_argument("--md", required=True)
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args()
    d, tag = os.path.split(a.run)
    J = json.load(open(os.path.join(d, "collapse_%s.json" % tag)))
    Zf = np.load(os.path.join(d, "collapse_%s_fields.npz" % tag), allow_pickle=True)
    S11p = os.path.join(d, "collapse_%s_s11.npz" % tag)
    Z11 = np.load(S11p, allow_pickle=True) if os.path.exists(S11p) else None
    ra = J["args"]; rs = float(ra["rs"]); g = float(ra["g"]); ell = float(ra["ell"])
    head = J.get("head") or {}
    recs = [r for r in J["records"] if r.get("E") is not None]
    out = {"script": "s11_pathfinder_report_v1.py", "run": a.run, "label": LABEL, "utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}
    L = ["# The massless ball pathfinder (S11): the self-consistent massless collapse at r_s/a 16", "", "Label (every table): %s." % LABEL, "",
         "- run %s (%s; its records %d, the last at T/r_s %s); every number below is printed by s11_pathfinder_report_v1.py from the run's own files." % (
             a.run, J.get("script"), len(recs), fmt(recs[-1]["T"] / rs if recs else None, "%.4f")),
         "- the end: %s." % (json.dumps(J.get("end_s11"), default=float) if J.get("end_s11") else (json.dumps(J.get("end_v32"), default=float) if J.get("end_v32") else "no stop recorded")), "",
         "## 0. The engine: engine_collapse_v33", "",
         "- engine_collapse_v33.py is engine_collapse_v32.py with S11's options, every one off by default. The reason: S11's ruling (REVIEWA_RULING_quasifrozen.md line 84) needs what version 32 lacks: m a = 0 (version 32 computes the rest number N_B = r_s/(2 g m) before its preparation, a division by zero at m = 0), the ball's preparation from the chiral Slater seeds with the momentum tuned after the constraint solve, the per-record readouts (the raw and averaged sources, both residuals, the positive-branch reconstruction, the ball's weight beyond the collars) and the stop where M(r) < 0 outside the central averaging cell with the three cases; the field module engine_fields_v17 (N' = 0 on an empty interior) as an option. The lead accepted the version for S11 (inbox, turn 46: 38.0's \"no new collapse engine version\" closes its owned corrections and does not bind S11).",
         "- the gate (version 33 with every option off against version 32, one small massive configuration, compare_v33_gate_v1.py): %s." % (
             " / ".join(x.strip() for x in open(a.gate).read().strip().splitlines()) if (a.gate and os.path.exists(a.gate)) else "not supplied"), ""]
    # 1 the preparation
    b = head.get("ball") or {}
    rs_E = 2.0 * g * float(head.get("E0") or 0.0)
    L += ["## 1. The preparation (the chiral Slater seeds, the projection against the physical sea, the inward boost tuned after the constraint solve)", "",
          "- R0 %s (%s r_s), x0 %s, sigma %s; orbitals %s; k0 %s; the solved charge E0 %s against the target %s (E0 a; 2gE0 %s); the bare energy (weighted) %s = %s E0, against the thin-layer control mu/E = 2[s - ln(1 + s)]/s^2 at R = x0 (s = (r_s/R)^(1/2)): %s; at R = R0: %s." % (
              fmt(b.get("R0"), "%.3f"), fmt((b.get("R0") or 0) / rs, "%.3f"), fmt(b.get("x0"), "%.3f"), fmt(b.get("sigma"), "%.3f"), b.get("orbitals"), fmt(head.get("k0"), "%.10f"),
              fmt(head.get("E0"), "%.10f"), fmt(head.get("E_target"), "%.6f"), fmt(rs_E, "%.6f"), fmt(b.get("E_bare_weighted"), "%.6f"), fmt(b.get("E_bare_over_E"), "%.6f"),
              fmt((2 * (math.sqrt(rs_E / b["x0"]) - math.log(1 + math.sqrt(rs_E / b["x0"]))) / (rs_E / b["x0"])) if b.get("x0") and rs_E > 0 else None, "%.6f"),
              fmt((2 * (math.sqrt(rs_E / b["R0"]) - math.log(1 + math.sqrt(rs_E / b["R0"]))) / (rs_E / b["R0"])) if b.get("R0") and rs_E > 0 else None, "%.6f")),
          "- the projection: the Gram eigenvalues per channel %s (the rank test: the smallest %s); the seeds' norm discarded by the projection at most %s; the smallest |e| of h_m at the start %s." % (
              json.dumps(b.get("gram"), default=float), fmt((head.get("projection") or {}).get("gram_min"), "%.6e"), fmt(b.get("discarded_norm_max"), "%.4e"), json.dumps(b.get("zero_mode_min_abs_e"))),
          "- the profile: rho_ell's minimum over (ell, R0] %s and J_ell's maximum there %s (an inward current is negative); the energy fraction beyond R0 %s; the largest 2GM/r %s at r %s; M's minimum outside the cell %s; N at the centre %s, at R0 %s against the thin-shell control 1 + (r_s/R0)^(1/2) = %s; the fixed point's passes %s and its last change %s." % (
              fmt(b.get("rho_ell_min_inside_outside_cell"), "%.4e"), fmt(b.get("J_ell_max_inside_outside_cell"), "%.4e"), fmt(b.get("energy_fraction_beyond_R0"), "%.4e"),
              fmt(b.get("max_compactness_2gM_over_r"), "%.6f"), fmt(b.get("r_at_max_compactness"), "%.3f"), fmt(b.get("M_min_outside_cell"), "%.6e"), fmt(b.get("N_centre"), "%.6f"),
              fmt(b.get("N_at_R0"), "%.6f"), fmt(b.get("N_at_R0_thin_shell_control"), "%.6f"), head.get("passes"), fmt(head.get("last_change"), "%.2e")),
          "", "| channel | j | Killing energy | bare energy | J/rho | site density min | weight on the ingoing spinor at k < 0 | weight above the quarter zone |", "|---|---|---|---|---|---|---|---|"]
    for k_, rows_ in (b.get("per_orbital") or {}).items():
        for q in rows_:
            L.append("| %s | %d | %.6f | %.6f | %s | %.3e | %.6f | %.3e |" % (k_, q["j"], q["killing_energy"], q["bare_energy"], fmt(q.get("J_over_rho"), "%.5f"), q["site_density_min"],
                                                                         q["weight_ingoing_spinor_k_negative"], q["weight_above_quarter_zone"]))
    L += ["", "- the k0 search: %s." % "; ".join("k0 %.8f -> E %.10f" % (q["k0"], q["E"]) for q in (head.get("k0_history") or [])), ""]
    out["preparation"] = {"k0": head.get("k0"), "E0": head.get("E0"), "ball": b}
    # 2 per record
    L += ["## 2. Per record: the chart's domain, the trapping zeros, the massless matter's own weight beyond the collar. Label: %s" % LABEL, "",
          "| T/r_s | E/E0 | M min outside the cell (at r) | U_phys min outside the cell | rho + w J min outside the cell (ball; pairs) | roots of N + v | min(N - abs(v)) | ball beyond 8 r_s: weight; energy | ball beyond 16 r_s: weight; energy | pairs' energy beyond 8; 16 r_s | residuals W; 1/N | converged; kept residual | continuation | stop |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    rows = []
    for i, r in enumerate(recs):
        s = r.get("s11") or {}
        b8 = s.get("ball_beyond_8") or {}; b16 = s.get("ball_beyond_16") or {}
        sv = r.get("solve_v32") or {}
        row = {"T_over_rs": r["T"] / rs, "E_over_E0": r.get("E_over_E0"), "M_min": s.get("M_min_outside_cell"), "r_M_min": s.get("r_at_M_min"), "U_phys_min": s.get("U_phys_min_outside_cell"),
               "source_min": s.get("source_min_outside_cell"), "ball_source_min": s.get("ball_source_min_outside_cell"), "pairs_source_min": s.get("pairs_source_min_outside_cell"),
               "roots": r.get("trapping_zeros"), "min_N_minus_absv": r.get("min_N_minus_absv"), "ball8": b8, "ball16": b16, "res_W": s.get("residual_W"), "res_invN": s.get("residual_invN"),
               "converged": r.get("converged"), "kept_residual": sv.get("kept_residual"), "continuation": r.get("continuation_label"), "stop": (s.get("stop") or {}).get("case")}
        rows.append(row)
        if i % max(1, a.every) == 0 or i == len(recs) - 1 or row["stop"] or (row["roots"] and not (rows[i - 1]["roots"] if i else None)):
            L.append("| %.4f | %s | %s (%s) | %s | %s (%s; %s) | %s | %s | %s; %s | %s; %s | %s; %s | %s; %s | %s; %s | %s | %s |" % (
                row["T_over_rs"], fmt(row["E_over_E0"], "%.8f"), fmt(row["M_min"], "%+.4e"), fmt(row["r_M_min"], "%.2f"), fmt(row["U_phys_min"], "%+.3e"), fmt(row["source_min"], "%+.3e"),
                fmt(row["ball_source_min"], "%+.3e"), fmt(row["pairs_source_min"], "%+.3e"), ", ".join("%.3f" % x for x in (row["roots"] or [])) or "none", fmt(row["min_N_minus_absv"], "%.4f"),
                fmt(b8.get("weight"), "%.4e"), fmt(b8.get("energy_density_route"), "%+.4e"), fmt(b16.get("weight"), "%.4e"), fmt(b16.get("energy_density_route"), "%+.4e"),
                fmt(b8.get("pairs_energy_density_route"), "%+.4e"), fmt(b16.get("pairs_energy_density_route"), "%+.4e"), fmt(row["res_W"], "%.1e"), fmt(row["res_invN"], "%.1e"),
                row["converged"], fmt(row["kept_residual"], "%.2e"), (row["continuation"] or "")[:60], row["stop"] or "-"))
    out["records"] = rows
    # the domain and the trapping zero
    first_stop = next((q for q in rows if q["stop"]), None)
    neg_M = [q for q in rows if q["M_min"] is not None and q["M_min"] < 0]
    trapped = [q for q in rows if q["roots"]]
    runs = []
    for i, q in enumerate(rows):
        if q["roots"]:
            if i > 0 and rows[i - 1]["roots"] and runs:
                runs[-1][1] = q["T_over_rs"]; runs[-1][2] += 1          # the same uninterrupted stretch of records with a root
            else:
                runs.append([q["T_over_rs"], q["T_over_rs"], 1])
    L += ["", "- the chart's domain: %s; records with M < 0 outside the cell: %d of %d." % (
        ("the run stopped at T/r_s %.4f: %s" % (first_stop["T_over_rs"], first_stop["stop"])) if first_stop else "M(r) >= 0 outside the central averaging cell on every record written",
        len(neg_M), len(rows)),
          "- the trapping zeros of N + v: %s." % (("first at T/r_s %.4f (roots %s); %d records with a root in all; the uninterrupted stretches (first, last, records): %s; the last record's roots %s" % (
              trapped[0]["T_over_rs"], ", ".join("%.4f" % x for x in trapped[0]["roots"]), len(trapped), "; ".join("%.4f to %.4f (%d)" % tuple(x) for x in runs),
              ", ".join("%.4f" % x for x in (rows[-1]["roots"] or [])) or "none")) if trapped else "none on any record written"), ""]
    out["domain"] = {"first_stop": first_stop, "negative_M_records": len(neg_M)}; out["trapping"] = {"records_with_root": len(trapped), "stretches": runs}
    # 3 the ledger and the energy beyond the collars
    Rcs = [(8.0, 8.0 * rs), (16.0, 16.0 * rs)]
    dr = DV2.density_route(J, Zf, Rcs)
    FM = importlib.import_module("engine_fields_v17" if ra.get("fields_v17") else "engine_fields_v16")
    Tz = np.asarray(Zf["T"], float)
    if a.ledger_T:
        want = sorted({int(np.argmin(np.abs(Tz / rs - float(x)))) for x in a.ledger_T.split(",")})
    else:
        want = sorted({int(np.argmin(np.abs(Tz / rs - x))) for x in np.arange(0.0, Tz[-1] / rs + 1e-9, 2.0)} | {len(Tz) - 1})
    lg = [DV2.ledger_at(J, Zf, k, FM, Rcs) for k in want]
    L += ["## 3. The collar charge against the energy beyond the collars, with the S5 (d) ledger and -Delta W. Label: %s" % LABEL, "",
          "| T/r_s | E_G | E_G - E_G(0) | sum a + b + c (M1) from T = 0 | -(W - W(0)) | K_out; eps_adj; eps_G | M(8 r_s); E_G - M(8 r_s) | M(16 r_s); E_G - M(16 r_s) | density route beyond 8 r_s: ball; pairs | beyond 16 r_s: ball; pairs |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    drm = {round(q["T"], 6): q for q in dr}
    x0 = lg[0] if lg else None
    for x in lg:
        q = drm.get(round(x["T"], 6)) or {}
        abc = sum(r_["abc"] for r_ in dr if x0 and x0["T"] < r_["T"] <= x["T"] + 1e-9 and r_["abc"] is not None)
        b8 = q.get("beyond_8") or {}; b16 = q.get("beyond_16") or {}
        L.append("| %.4f | %.10f | %+.6e | %+.6e | %s | %s; %s; %s | %s; %s | %s; %s | %s; %s | %s; %s |" % (
            x["T"] / rs, x["E_G"], x["E_G"] - x0["E_G"], abc, fmt(-(x["W"] - x0["W"]) if (x.get("W") is not None and x0.get("W") is not None) else None, "%+.6e"),
            fmt(x.get("K_out"), "%+.3e"), fmt(x.get("eps_adj"), "%+.3e"), fmt(x.get("eps_G"), "%+.3e"), fmt(x.get("M_at_8"), "%.8f"), fmt((x["E_G"] - x["M_at_8"]) if x.get("M_at_8") is not None else None, "%+.6e"),
            fmt(x.get("M_at_16"), "%.8f"), fmt((x["E_G"] - x["M_at_16"]) if x.get("M_at_16") is not None else None, "%+.6e"), fmt(b8.get("K_out_dust"), "%+.6e"), fmt(b8.get("K_out_pairs"), "%+.6e"),
            fmt(b16.get("K_out_dust"), "%+.6e"), fmt(b16.get("K_out_pairs"), "%+.6e")))
    L += ["", "- M(R) here is the field module's mass_function W(R)/(2g) (= N^2 M; N = 1 beyond the matter); the ledger's E_G is the record's charge at the engine's collar (the chain end, 64 r_s).", ""]
    out["ledger"] = lg; out["density_route"] = dr
    # 4 the characteristic solve (the continuum control)
    if a.char and os.path.exists(a.char):
        C = json.load(open(a.char))
        res = C.get("result") or {}
        ft = res.get("first_trapping")
        first_root = trapped[0]["T_over_rs"] if trapped else None
        L += ["## 4. The mass-label characteristic solve beside the run (the continuum control: one ingoing stream, no central flux). Label: %s" % LABEL, "",
              "- the source profile: %s." % json.dumps(C.get("source"), default=float),
              "- the characteristic solve's first trapping: %s; the run's first root of N + v at T/r_s %s." % (json.dumps(ft, default=float), fmt(first_root, "%.4f")),
              "- the outermost label (at R %s, %s r_s): t_H/r_s %s integrated against %s in closed form; the closed-form crossing from R0 = 4 r_s, 1 + 2 ln(3/2) = %.10f." % (
                  fmt((C.get("outer_label_closed_form") or {}).get("R0"), "%.4f"), fmt((C.get("outer_label_closed_form") or {}).get("R0_over_rs"), "%.4f"),
                  fmt((res.get("outer_label_t_H") / (2 * g * float(C["source"]["E"]))) if (res.get("outer_label_t_H") is not None and C.get("source")) else None, "%.8f"),
                  fmt((C.get("outer_label_closed_form") or {}).get("t_H_over_rs"), "%.8f"), 1.0 + 2.0 * math.log(1.5)),
              "- unit tests: %s." % json.dumps(C.get("unit_tests"), default=float)[:1500], "",
              "| t/r_s | r of the outermost label | innermost free label | held mass fraction | s max (at u) | trapped labels | R_h | min label spacing | energy identity | the midpoint map's tau ||DV||/2 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for q in res.get("rows") or []:
            L.append("| %.4f | %.4f | %s | %.6f | %.6f (%.4f) | %d | %s | %.3e | %.2e | %.3e |" % (q["t_over_rs"], q["r_outer"], fmt(q.get("r_inner_unheld"), "%.4f"), q["held_mass_fraction"], q["s_max"], q["u_at_s_max"],
                                                                                 q["trapped_labels"], fmt(q.get("R_h_outer"), "%.4f"), q["min_label_spacing"] if q["min_label_spacing"] is not None else float("nan"),
                                                                                 q["energy_identity_rel"], q["midpoint_contraction_at_tau"]))
        L.append("")
        out["characteristic_solve"] = {"first_trapping": ft, "outer_label_t_H": res.get("outer_label_t_H")}
    # 5 the Page recursion alongside
    for lab_, p in (("R_cut 8 r_s (the collar 16/3 r_s)", a.page8), ("R_cut 16 r_s (the collar 32/3 r_s)", a.page16)):
        if not p or not os.path.exists(p):
            L += ["## 5. The Page recursion on this history, %s: not run yet (%s)." % (lab_, p), ""]
            continue
        P = json.load(open(p))
        L += ["## 5. The Page recursion of S1 on this history, %s (page_history_v6c.py; %s). Label: %s; relabelled: channels 1 and 2 of the same massless operator, the sea propagated from its ground state at T = 0 on the record's applied fields, the ball's matter not in it." % (
            lab_, "final" if P.get("final") else "running", LABEL), "",
            "| T/r_s | 2gE | E_inv | beyond the collar | W (creation) | Tr[h_dot (C - P)] | L (interval) | L_H(kappa_T) | L_H(T_H) | S_esc | band | E_remove | M_chi | energy identity | collar identity |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in P.get("rows") or []:
            L.append("| %.4f | %.4f | %.6e | %.6e | %s | %s | %s | %s | %s | %.6f | %.2e | %s | %s | %s | %s |" % (
                r["t_over_rs"], r.get("two_gE", float("nan")), r.get("E_inv_weighted", float("nan")), r.get("E_out_weighted", float("nan")), fmt(r.get("W_creation"), "%.4e"),
                fmt(r.get("W_hdot_weighted"), "%.4e"), fmt(r.get("L_interval"), "%.4e"), fmt(r.get("L_H_kappaT"), "%.4e"), fmt(r.get("L_H_TH"), "%.4e"), r.get("S_esc_weighted", float("nan")),
                r.get("band", float("nan")), fmt(r.get("E_remove_weighted"), "%.6e"), fmt(r.get("M_chi_weighted"), "%.6e"), fmt(r.get("energy_identity_residual_weighted"), "%.1e"),
                fmt(r.get("collar_identity_residual_weighted"), "%.1e")))
        L.append("")
    L.append("- every number above is printed by s11_pathfinder_report_v1.py from the files named; the r_s/a 32 run at ell 2a and 4a is the owner's decision after this file and is not launched by it.")
    open(a.md, "w").write("\n".join(L) + "\n")
    if a.out_json:
        json.dump(out, open(a.out_json, "w"), indent=1, default=float)
    print("composed %s (%d records)" % (a.md, len(rows)), flush=True)


if __name__ == "__main__":
    main()
