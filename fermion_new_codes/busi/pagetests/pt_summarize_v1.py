#!/usr/bin/env python
"""pt_summarize_v1.py (busi, inbox #428): the decision tables of PAGE_tests_v1.md from the run jsons (no compute beyond reading).
Test A: per channel, the particle occupation over Gamma f_H(T_H(u)) at every omega and u for L_eff = 0 (held), 1, 3, 10, 30 L_H; the
late-u mean at omega = T_H, 2 T_H, 3 T_H against L_eff (reviewer A's test: proportional rise against saturation at Gamma f_H above ~7 L_H);
the hole (negative band) occupations at |omega| <= T_H (the soft exit). Test B: native / reset / QFT per packet on the ladder."""
import glob
import json
import os
import sys

import numpy as np


def testA(d):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "A_k*_L*.json"))):
        J = json.load(open(p))
        out[J["tag"]] = J
    lines = []
    for k in (1, 2):
        runs = {float(t.split("_L")[1]): J for t, J in out.items() if t.startswith("A_k%d_" % k)}
        if not runs:
            continue
        Ls = sorted(runs)
        oms = sorted({r["omega_over_TH"] for J in runs.values() for r in J["rows"] if r["omega_over_TH"] > 0})
        us = sorted({r["u_rs"] for J in runs.values() for r in J["rows"]})
        lines.append("\n## kappa %d: particle occupation / Gamma f_H(T_H(u)) (rows u in nominal r_s, columns L_eff/L_H)\n" % k)
        for om in oms:
            lines.append("omega/T_H %g" % om)
            lines.append("| u | " + " | ".join("L %g" % L for L in Ls) + " |")
            lines.append("|---|" + "---|" * len(Ls))
            for u in us:
                cells = []
                for L in Ls:
                    rr = [r for r in runs[L]["rows"] if r["omega_over_TH"] == om and r["u_rs"] == u]
                    cells.append(("%.4g" % rr[0]["ratio_TH"]) if rr and rr[0]["ratio_TH"] is not None else "-")
                lines.append("| %g | " % u + " | ".join(cells) + " |")
            lines.append("")
        lines.append("late mean (u >= 1000 r_s) of the ratio, and T_fit/T_H there:")
        for om in oms:
            row = []
            for L in Ls:
                rr = [r for r in runs[L]["rows"] if r["omega_over_TH"] == om and r["u_rs"] >= 1000]
                if rr:
                    row.append("L %g: %.4g (T/T_H %s)" % (L, np.mean([r["ratio_TH"] for r in rr]),
                                                        "%.3f" % np.mean([r["T_fit_over_TH"] for r in rr if r["T_fit_over_TH"]]) if any(r["T_fit_over_TH"] for r in rr) else "-"))
            lines.append("  omega/T_H %g: " % om + "; ".join(row))
        lines.append("checks per run: norm defect max, ws ladder spread max (relative), edge weight, tail acc, packet energy check:")
        for L in Ls:
            J = runs[L]
            nd = max(abs(r["norm_defect_T0"]) for r in J["rows"])
            sp = max((r["ws_spread"] / max(r["particle_or_hole"], 1e-300)) for r in J["rows"] if r["ws_spread"] is not None)
            ec = [r["energy_check"] for r in J["rows"] if r.get("energy_check")]
            lines.append("  L %g: norm defect %.1e, ws/2-ws/4 relative spread max %.2e, edge %.1e, Chebyshev tail acc %.1e, <h>/omega %s, run %.0f s, charge at end %.4f" % (
                L, nd, sp, J["monitor"]["edge_max"], J["stats"]["tail_acc"], ["%.4f" % e["mean_over_omega"] for e in ec][:4], J["seconds"], J["charge_t_max"]))
        neg = [(L, r) for L in Ls for r in runs[L]["rows"] if r["omega_over_TH"] < 0]
        if neg:
            lines.append("negative band (holes) against the positive band (the chain's pairing):")
            for L in Ls:
                for r in runs[L]["rows"]:
                    if r["omega_over_TH"] < 0:
                        q = [x for x in runs[L]["rows"] if x["omega_over_TH"] == -r["omega_over_TH"] and x["u_rs"] == r["u_rs"]]
                        if q:
                            lines.append("  L %g omega/T_H %g u %g: hole %.6e particle(+omega) %.6e" % (L, r["omega_over_TH"], r["u_rs"], r["particle_or_hole"], q[0]["particle_or_hole"]))
    return "\n".join(lines)


def testB(d):
    lines = []
    for p in sorted(glob.glob(os.path.join(d, "B_rs*.json"))):
        J = json.load(open(p))
        lines.append("\n## %s: r_s/a %g, chain %d, final root %.4f r_s, kappa r_s %.5f, first root t %.3f r_s, static from %.3f r_s" % (
            J["tag"], J["rs"], J["nsites"], J["final_root_rs"], J["kappa_rs"], J["first_root_rs"] or float("nan"), J["t_static_rs"]))
        lines.append("| omega r_s | t_col/r_s | native | reset t_d 8 | reset t_d 20 | QFT (Gamma_1 f_H) | native/QFT | T_fit/T_H | ws/2 rel |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for r in J["rows"]:
            if r["omega_rs"] < 0:
                continue
            wsr = abs(r["n_native_ws_half"] - r["n_native"]) / max(abs(r["n_native"]), 1e-300)
            lines.append("| %g | %g | %.4g | %s | %s | %.4g | %s | %s | %.1e |" % (
                r["omega_rs"], r["t_col_rs"], r["particle_or_hole_native"], ("%.3g" % r["particle_or_hole_reset_td8"]) if "particle_or_hole_reset_td8" in r else "-",
                ("%.3g" % r["particle_or_hole_reset_td20"]) if "particle_or_hole_reset_td20" in r else "-", r["n_QFT_stationary"],
                ("%.4g" % r["ratio_native_QFT"]) if r["ratio_native_QFT"] is not None else "-", ("%.3f" % r["T_fit_over_TH"]) if r["T_fit_over_TH"] else "-", wsr))
        nd = max(abs(r["norm_defect"]) for r in J["rows"])
        pairing = max(abs(a["particle_or_hole_native"] - b["particle_or_hole_native"]) for a in J["rows"] for b in J["rows"]
                      if a["omega_rs"] == -b["omega_rs"] and a["omega_rs"] > 0 and a["t_col_rs"] == b["t_col_rs"])
        od = [o for o in J["offdiag"] if o["omega_rs"] > 0]
        big = sorted(od, key=lambda o: -o["abs_native"])[:3]
        lines.append("norm defect max %.1e; particle-hole pairing max |diff| %.1e; cross-band max |C| %.1e; Chebyshev tail acc %.1e" % (
            nd, pairing, J["cross_band_max_abs_native"] or 0, J["stats"]["tail_acc"]))
        lines.append("largest within-band off-diagonals |C_ij| native (QFT): " + "; ".join("w r_s %g t_col %g-%g: %.3g (%.3g)" % (
            o["omega_rs"], o["t_col_i"], o["t_col_j"], o["abs_native"], o["abs_QFT"]) for o in big))
        if J.get("raymap"):
            lines.append("ray map kappa r_s(t_col): " + ", ".join("%g: %s" % (x["t_col_rs"], ("%.4f" % x["kappa_rs"]) if x.get("kappa_rs") else "-") for x in J["raymap"] if x))
    return "\n".join(lines)


if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "<internal>/page_tests"
    print("# Test A\n" + testA(os.path.join(base, "testA")))
    print("\n# Test B\n" + testB(os.path.join(base, "testB")))
