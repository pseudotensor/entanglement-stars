#!/usr/bin/env python
"""Task 2 report, job T2b, version 2 (the reviewer B red team of the task 2 judgment: version 1's table printed the nodal form's cumulative
marker-step and re-solve pieces only, under a label that read as the marker Hamiltonian's; version 2 prints both forms side by side and
labels them; no number changes) (SPEC_task2_static_row_diagnostics.md sec. 3; builder analyst): the monitors M8 to M11 of strand7_shell_v5.py
--t2b 1 (self E16, h = 1, dt = 0.01 r_s, R_max = 64 r_s, T_end = 60 r_s0), read from its record s7a_self_E16_t2b.json and the markers' file.

Per output (every 0.1 r_s; the table at 2 r_s intervals, the throat passage and the last output): M11 the energy balance in both forms
(nodal sum N S1 and markers' sum nu N W of the discrete functional F): F, F - F(0), the cumulative marker-step piece (fixed fields), the
cumulative exact re-solve piece, the cumulative injection estimate R1_pre.dv + R2_pre.du, the telescoping closure F - F(0) - (step + resolve)
(bookkeeping, zero to rounding) and the spec's closure F - F(0) - (step + inj) against the target 1e-3 F(0), the instantaneous
gather/deposit mismatch; M10: sigma_min(J), sigma_min(J_C), N_min and its node, P07's residual max|N - N_*|/max|N|, P28's (N - N_*).v_* and
N.v_* against dA/dT (two-point differences of the record's A); M9: the markers' W-weighted chi beside the nodal chi_kin (I3) and
N_* = 1/(3 + 2 chi) against the record's N_sh at the throat passage (the E-ladder spreads 0.002 in N and 0.015 in B/A, spec sec. 3); the
boundary charge's drift log(E_bdy/E_bdy(0)) beside F's.  Args: --record --markers --json --md."""
import argparse
import json
import math

import numpy as np


def f(v, d=5):
    if v is None:
        return "-"
    v = float(v)
    return "%.*g" % (d, v) if (v == 0 or 1e-3 <= abs(v) < 1e6) else "%.3e" % v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True)
    ap.add_argument("--markers", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    D = json.load(open(a.record))
    run = D["run"]
    recs = run["records"]
    h0 = run["t2b"]
    F0n, F0m = h0["F_nodal0"], h0["F_markers0"]
    E0 = recs[0]["E_bdy"]
    rows = []
    for i, r in enumerate(recs):
        o = r.get("t2b")
        if not o:
            continue
        dAdT = None
        if 0 < i < len(recs) - 1:
            dAdT = (recs[i + 1]["A"] - recs[i - 1]["A"]) / (recs[i + 1]["T"] - recs[i - 1]["T"])
        chi_m, chi_n = o["chi_markers_W"], o["chi_nodal"]
        rows.append({"T_over_rs": r["T_over_rs"], "log_Ebdy_drift": math.log(r["E_bdy"] / E0), "A": r["A"], "dA_dT": dAdT,
                     "F_nodal_minus_F0": o["F_nodal"] - F0n, "F_markers_minus_F0": o["F_markers"] - F0m,
                     "cum_step_nodal": o["cum_step_n"], "cum_resolve_nodal": o["cum_resolve_n"], "cum_inj": o["cum_inj"],
                     "cum_step_markers": o["cum_step_m"], "cum_resolve_markers": o["cum_resolve_m"],
                     "closure_telescoping_nodal": o["closure_telescoping_nodal"], "closure_telescoping_markers": o["closure_telescoping_markers"],
                     "closure_inj_nodal": o["remainder_inj_nodal"], "closure_inj_markers": o["remainder_inj_markers"],
                     "closure_inj_nodal_over_F0": o["remainder_inj_nodal"] / F0n, "closure_inj_markers_over_F0": o["remainder_inj_markers"] / F0m,
                     "mismatch": o["mismatch"], "sigma_min_J": o["sigma_min_J"], "sigma_min_JC": o["sigma_min_JC"], "N_min": o["N_min"],
                     "r_Nmin": o["r_Nmin"], "p07_residual": o["p07_residual"], "P28_N_minus_Nstar_dot_vstar": o["N_minus_Nstar_dot_vstar"],
                     "N_dot_vstar": o["N_dot_vstar"], "vstar_norm": o["vstar_norm"], "chi_markers_W": chi_m, "chi_nodal": chi_n,
                     "N_star_markers": 1 / (3 + 2 * chi_m), "N_star_nodal": 1 / (3 + 2 * chi_n), "N_sh": r["N_sh"], "B_over_A": r["B_over_A"],
                     "r_sh_over_A": r["r_sh_over_A"]})
    out = {"source": a.record, "F_nodal0": F0n, "F_markers0": F0m, "mismatch0": h0["mismatch0"], "throat_crossing": run.get("throat_crossing"),
           "kills": run.get("kills"), "stop": run.get("stop"), "steps": run.get("steps"), "rows": rows}
    tc = run.get("throat_crossing")
    if tc:
        T = tc["T_over_rs"]
        lo = max([q for q in rows if q["T_over_rs"] <= T], key=lambda q: q["T_over_rs"])
        hi = min([q for q in rows if q["T_over_rs"] >= T], key=lambda q: q["T_over_rs"])
        t = 0.0 if hi["T_over_rs"] == lo["T_over_rs"] else (T - lo["T_over_rs"]) / (hi["T_over_rs"] - lo["T_over_rs"])
        lerp = lambda k: lo[k] + t * (hi[k] - lo[k])
        out["throat"] = {"T_over_rs": T, "N_sh_record": tc["N_sh"], "B_over_A_record": tc["B_over_A"], "chi_markers_W": lerp("chi_markers_W"),
                         "chi_nodal": lerp("chi_nodal"), "N_star_markers": lerp("N_star_markers"), "N_star_nodal": lerp("N_star_nodal")}
        out["throat"]["N_sh_minus_N_star_markers"] = tc["N_sh"] - out["throat"]["N_star_markers"]
        out["throat"]["N_sh_minus_N_star_nodal"] = tc["N_sh"] - out["throat"]["N_star_nodal"]
    agg = lambda k, fn: fn([abs(q[k]) for q in rows if q[k] is not None])
    out["max"] = {k: agg(k, max) for k in ("closure_telescoping_nodal", "closure_telescoping_markers", "closure_inj_nodal_over_F0",
                                           "closure_inj_markers_over_F0", "p07_residual", "P28_N_minus_Nstar_dot_vstar", "mismatch")}
    out["min"] = {k: agg(k, min) for k in ("N_min", "sigma_min_J", "sigma_min_JC")}
    Z = np.load(a.markers)
    out["markers_file"] = {"outputs": int(len(Z["T"])), "markers": int(Z["r"].shape[1]), "T_first": float(Z["T"][0]), "T_last": float(Z["T"][-1])}
    sel = []
    for target in list(np.arange(0, 61, 2.0)) + ([tc["T_over_rs"]] if tc else []):
        q = min(rows, key=lambda q: abs(q["T_over_rs"] - target))
        if q not in sel:
            sel.append(q)
    sel.sort(key=lambda q: q["T_over_rs"])
    L = ["# Task 2 job T2b: the monitors M8 to M11 (task2_t2b_report_v2.py over strand7_shell_v5.py --t2b 1; self E16, h = 1, dt = 0.01 r_s, "
         "R_max = 64 r_s, T_end = 60 r_s0; static rows, external frame)", "",
         "- F(0): nodal %s, markers %s, mismatch(0) %s; steps %s; stop: %s; kills: %s." % (f(F0n, 10), f(F0m, 10), f(h0["mismatch0"]), run.get("steps"),
                                                                                       run.get("stop"), json.dumps(run.get("kills"))),
         "- Largest over the outputs: telescoping closure (nodal, markers) %s, %s; the spec's closure F - F(0) - (step + inj) over F(0) (nodal, "
         "markers) %s, %s (target 1e-3); P07 residual %s; |P28 (N - N_*).v_*| %s; |mismatch| %s." % tuple(f(out["max"][k]) for k in (
             "closure_telescoping_nodal", "closure_telescoping_markers", "closure_inj_nodal_over_F0", "closure_inj_markers_over_F0", "p07_residual",
             "P28_N_minus_Nstar_dot_vstar", "mismatch")),
         "- Smallest over the outputs: N_min %s, sigma_min(J) %s, sigma_min(J_C) %s." % tuple(f(out["min"][k]) for k in ("N_min", "sigma_min_J", "sigma_min_JC"))]
    if tc:
        th = out["throat"]
        L.append("- The throat passage r_sh = A at T = %s r_s: the record's N_sh %s and B/A %s; chi (markers, W-weighted) %s, chi_kin (nodal) %s; "
                 "N_* = 1/(3 + 2 chi): %s (markers), %s (nodal); N_sh - N_*: %s (markers), %s (nodal); the E-ladder spread 0.002 in N." % tuple(
                     f(x) for x in (th["T_over_rs"], th["N_sh_record"], th["B_over_A_record"], th["chi_markers_W"], th["chi_nodal"], th["N_star_markers"],
                                    th["N_star_nodal"], th["N_sh_minus_N_star_markers"], th["N_sh_minus_N_star_nodal"])))
    L += ["", "Columns: F is the discrete functional in its nodal form (sum N S1) or its markers' form (sum nu N W); \"step\" is the cumulative change "
          "of F over the marker steps at fixed fields, \"re-solve\" the cumulative exact change at the re-solves, \"inj\" the cumulative first-order "
          "estimate R1_pre.dv + R2_pre.du (one number, form-independent); the telescoping closure F - F(0) - (step + re-solve) is zero by construction.",
          "", "| T/r_s | log E_bdy drift | F - F0 (nodal, markers) | step (nodal, markers) | re-solve (nodal, markers) | inj | closure tel. | closure inj/F0 (nodal, markers) | mismatch | "
          "sigma_min J, J_C | N_min (r) | P07 | P28 (N-N*).v* | N.v* | dA/dT | chi markers, nodal | N_sh | r_sh/A |",
          "|" + "---|" * 18]
    for q in sel:
        L.append("| %s | %s | %s, %s | %s, %s | %s, %s | %s | %s | %s, %s | %s | %s, %s | %s (%s) | %s | %s | %s | %s | %s, %s | %s | %s |" % (
            f(q["T_over_rs"], 4), f(q["log_Ebdy_drift"]), f(q["F_nodal_minus_F0"]), f(q["F_markers_minus_F0"]), f(q["cum_step_nodal"]), f(q["cum_step_markers"]),
            f(q["cum_resolve_nodal"]), f(q["cum_resolve_markers"]), f(q["cum_inj"]),
            f(q["closure_telescoping_nodal"], 2), f(q["closure_inj_nodal_over_F0"], 3), f(q["closure_inj_markers_over_F0"], 3), f(q["mismatch"], 3),
            f(q["sigma_min_J"], 3), f(q["sigma_min_JC"], 3), f(q["N_min"], 4), f(q["r_Nmin"], 3), f(q["p07_residual"], 2),
            f(q["P28_N_minus_Nstar_dot_vstar"], 3), f(q["N_dot_vstar"], 3), f(q["dA_dT"], 3), f(q["chi_markers_W"], 4), f(q["chi_nodal"], 4),
            f(q["N_sh"], 4), f(q["r_sh_over_A"], 4)))
    L += ["", "- Every number above is emitted by the script from the record; the markers' file holds the per-marker r, p, nu, W, N, psi at %d outputs "
          "(%d markers)." % (out["markers_file"]["outputs"], out["markers_file"]["markers"])]
    json.dump(out, open(a.json, "w"), indent=1, default=float)
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
