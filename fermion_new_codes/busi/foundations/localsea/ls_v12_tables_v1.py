#!/usr/bin/env python
"""ls_v12_tables_v1.py (busi, inbox #455): prints every v1.2 table from the stored outputs (no computation beyond sums, medians
and ratios), so that no number in FOUNDATIONS_localsea_v1.2.md is retyped. Inputs under <internal>/localsea:
ls_srcsign_v1.json, v12/ls_reviewbA_3d_v3.json (s8), v12/s4a/ and v12/s4b/ls_reviewbA_3d_v3.json (s4), v12/ls_audit_read_v2.json
(rates), ls_audit_read_v2.json (v1.1's rates, for the s4 comparison), v12/ls_scale_read_v3.json, audit/ls_chain_kmax_v1_s8_T16_K40.json,
ls_rate3q_read_v1.json and ls_rate3q_read_v2.json (if present). One thread."""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import json   # noqa: E402

import numpy as np   # noqa: E402

D = "<internal>/localsea"


def J(p):
    p = os.path.join(D, p)
    return json.load(open(p)) if os.path.exists(p) else None


print("== T1 source sign (ls_srcsign_v1): max|eps_pairs - (sea - R2)|, max|eps_pairs + (sea - R2)|, J likewise, scale")
S = J("ls_srcsign_v1.json")
for t, x in S.items():
    print("%-7s | eps %.1e vs %.1e (scale %.3e) | J %.1e vs %.1e (scale %.3e) | pol eps %.4f J %.4f | 3D eps %.3f J %.3f" % (
        t, x["eps_pairs_minus_(sea-R2)_maxabs"], x["eps_pairs_plus_(sea-R2)_maxabs"], x["pairs_eps_maxabs"], x["J_pairs_minus_(sea-R2)_maxabs"],
        x["J_pairs_plus_(sea-R2)_maxabs"], x["pairs_J_maxabs"], x["pol_eps_total(R2-0)"], x["pol_J_total(R2-0)"], x["rho3d_total"], x["J3d_total"]))

print("\n== T2 reviewer B A with the 3D cubic sea, both signs (ls_reviewbA_3d_v3)")
A = {}
for p in ("v12/ls_reviewbA_3d_v3.json", "v12/s4a/ls_reviewbA_3d_v3.json", "v12/s4b/ls_reviewbA_3d_v3.json"):
    x = J(p)
    if x:
        A.update({k: v for k, v in x.items() if k != "sign_note"})
order = ["s4_T07", "s4_T13", "s4_T14", "s4_T15", "s4_T16", "s8_T14", "s8_T16"]
for t in order:
    if t not in A:
        continue
    x = A[t]
    rec = x["solves"].get("record", {})
    print("%s | json E %.4f zeros %s min %.4f | re-solve E %.4f zeros %s slope %s | floors N %.1e v %.1e" % (
        t, x["record_E"] or np.nan, [round(z, 2) for z in (x["record_zeros"] or [])], x["record_min"] or np.nan, rec.get("E_collar", np.nan),
        [round(z, 2) for z in rec.get("zeros", [])], ("%.5f" % rec["outer_root_slope"]) if rec.get("outer_root_slope") else None,
        x["record_interp_floor_N"] or np.nan, x["record_interp_floor_v"] or np.nan))
    for lab in ("LS2_3d_correct", "LS2_3d_outside_4a_correct", "flat_P0_correct", "LS2_3d_reversed_v2", "flat_P0_reversed_v2"):
        s = x["solves"].get(lab)
        if not s:
            continue
        if s.get("failed"):
            print("   %-27s FAILED (d eps %.2f, d J %.2f)" % (lab, s["d_eps_total"], s["d_J_total"]))
            continue
        print("   %-27s zeros %-16s min %8.4f slope %s E %9.3f dE %+7.2f%% | d eps %+8.2f d J %+8.2f | max|dN| %.2e at %.0f max|dv| %.2e at %.0f" % (
            lab, [round(z, 2) for z in s["zeros"]], s["min"], ("%.5f" % s["outer_root_slope"]) if s.get("outer_root_slope") else "none",
            s["E_collar"], 100 * s["dE_rel"], s["d_eps_total"], s["d_J_total"], s["max_abs_dN"], s["r_max_dN"], s["max_abs_dv"], s["r_max_dv"]))
    it = x.get("LS2_3d_correct_selfconsistent")
    if it:
        print("   self-consistent b[u]: " + " | ".join(
            ("it %d resid %.2e E %.3f dE %+.2f%% step %+.1e" % (h["it"], h["b_residual_rel"], h["E_collar"], 100 * h["dE_rel_vs_record"], h["dE_rel_vs_prev"]))
            if not h.get("failed") else "it %d FAILED (resid %.2e)" % (h["it"], h["b_residual_rel"]) for h in it))

print("\n== T3 the reference's rate at the outer root (v12 rerun; s4 from the resume-mirror copy, s8 new)")
R12 = J("v12/ls_audit_read_v2.json"); R11 = J("ls_audit_read_v2.json")
for h, x in R12["rates"].items():
    rows = x["rows"]
    def med(key, sel=lambda r: True):
        v = [abs(r[key]) for r in rows if sel(r)]
        return (float(np.median(v)), float(np.max(v)), len(v)) if v else (None, None, 0)
    print("%s: rows %d (J exact %d) | ref rate med %.4f | J exact: med %.4f max %.4f | energy only med %.4f | resid(-2J) med %.3f, J exact %.3f | handoff check %.1e of %.0f" % (
        h, x["n_rows"], x["n_rows_Jexact"], x["median_abs_ref_rate_rel"], med("ref_rate_rel", lambda r: r["J_is_R2"])[0],
        med("ref_rate_rel", lambda r: r["J_is_R2"])[1], x["median_abs_ref_rate_energy_only_rel"], x["median_abs_resid_minus2J"],
        x["median_abs_resid_minus2J_rel_Jexact"], x["handoff_check_max"], x["handoff_scale"]))
    if h == "s8":
        Ts = [r["T_over_rs"] for r in rows]
        print("   s8 T range %.3f to %.3f; ref rate per row: %s" % (min(Ts), max(Ts), " ".join("%.4f" % r["ref_rate_rel"] for r in rows)))
        print("   s8 resid(-2J) per row: %s; resid(+2J) median %.3f" % (" ".join("%.3f" % r["resid_minus2J_rel"] for r in rows), x["median_abs_resid_plus2J"]))
    if R11 and h in R11.get("rates", {}):
        a = R11["rates"][h]
        print("   v1.1's %s: ref rate med %.4f, J exact %.4f, rows %d -> identical: %s" % (
            h, a["median_abs_ref_rate_rel"], a["median_abs_ref_rate_rel_Jexact"], a["n_rows"],
            all(abs(r1["ref_rate_rel"] - r2["ref_rate_rel"]) < 1e-12 for r1, r2 in zip(a["rows"], rows)) and len(a["rows"]) == len(rows)))

print("\n== T4 the lambda scaling at s4_T15 (ls_scale_read_v3): energy, then momentum, per window")
SC = J("v12/ls_scale_read_v3.json")
print("lambda = 1 recovery check:", SC["lam1_recovery_check"])
for lab in ("eps", "J"):
    print("[%s] r window | prod (lam 1) | lam 2 | lam 3 | e2 +- unc | e4 | e6+ | prod - LS2 | prod - LS4 | e2/prod" % lab)
    for w in SC["windows"]:
        print("  %6.1f to %6.1f | %.4e | %.4e | %.4e | %+.3e +- %.1e | %+.3e | %+.3e | %+.3e | %+.3e | %+.3f" % (
            w["r"][0], w["r"][1], w[lab + "_prod"], w[lab + "_lam2"], w[lab + "_lam3"], w[lab + "_2"], w[lab + "_2_unc"], w[lab + "_4"],
            w[lab + "_6plus"], w[lab + "_prod_minus_LS2"], w[lab + "_prod_minus_LS4"], w[lab + "_2"] / w[lab + "_prod"]))
print("field solves at s4_T15 (sign: Delta s = pol - b):")
for k, s in SC["solves"].items():
    if s.get("failed"):
        print("  %-15s FAILED" % k); continue
    print("  %-15s zeros %-16s min %.4f slope %.5f E %.3f %s" % (k, [round(z, 2) for z in s["zeros"]], s["min"], s["outer_root_slope"] or np.nan,
          s["E_collar"], ("dE %+.2f%% d eps %+.3f d J %+.3f max|dN| %.2e at %.0f max|dv| %.2e at %.0f" % (100 * s["dE_rel"], s["d_eps_total"],
          s["d_J_total"], s["max_abs_dN"], s["r_max_dN"], s["max_abs_dv"], s["r_max_dv"])) if "dE_rel" in s else ""))
print("record json E %.4f zeros %s; floors N %.1e v %.1e" % (SC["record_E"], SC["record_zeros"], SC["record_interp_floor_N"], SC["record_interp_floor_v"]))

print("\n== T5 channel truncation at s8_T16 (ls_chain_kmax_v1, K = 10, 20, 40)")
K = J("audit/ls_chain_kmax_v1_s8_T16_K40.json")
print("K = 10 check against stored:", K["check_K10"])
print("mixed / neg-upper / pos-lower over all 40 channels:", sum(v["mixed"] for v in K["per_channel"].values()),
      sum(v["neg_upper_character"] for v in K["per_channel"].values()), sum(v["pos_lower_character"] for v in K["per_channel"].values()))
pc = K["per_channel"]
print("per-channel eps_pol total at kappa 10, 20, 30, 40: %s; ratio kappa 40/39: %.4f" % (
    " ".join("%.4f" % pc[str(k)]["eps_pol_total"] for k in (10, 20, 30, 40)), pc["40"]["eps_pol_total"] / pc["39"]["eps_pol_total"]))
print("r window | eps K10 | K20 | K40 | 3D | K40/K10 | 3D/K10 | 3D/K40 | J K10 | J K40 | J 3D")
for w in K["windows"]:
    print("  %6.1f to %6.1f | %.4e | %.4e | %.4e | %.4e | %.2f | %.3g | %.3g | %+.3e | %+.3e | %+.3e" % (
        w["r"][0], w["r"][1], w["eps_K10"], w["eps_K20"], w["eps_K40"], w["rho3d"], w["eps_K40"] / w["eps_K10"], w["rho3d"] / w["eps_K10"],
        w["rho3d"] / w["eps_K40"], w["J_K10"], w["J_K40"], w["J3d"]))

for name in ("ls_rate3q_read_v1.json", "ls_rate3q_read_v2.json"):
    Q = J(name)
    if not Q:
        continue
    print("\n== T6 nonlinear rate coefficients (%s): L %s, q %s" % (name, Q["L"], " ".join("%.5f" % x for x in Q["q"])))
    for k, o in Q["coef"].items():
        print("  %-12s c/q^2 %s | full c0/c(q_min) %s resid %s | c0+c2q2 resid %s | c2q2+c4q4 resid %s" % (
            k, " ".join("%.5f" % v for v in o["c_over_q2"]), ("%+.4f" % o["full_c0_over_c_smallest_q"]) if "full_c0_over_c_smallest_q" in o else "-",
            ("%.1e" % o["full_resid_rel"]) if "full_resid_rel" in o else "-", ("%.3f" % o["c0_c2_3pt_resid_rel"]) if "c0_c2_3pt_resid_rel" in o else "-",
            ("%.3f" % o["c2_c4_3pt_resid_rel"]) if "c2_c4_3pt_resid_rel" in o else "-"))
