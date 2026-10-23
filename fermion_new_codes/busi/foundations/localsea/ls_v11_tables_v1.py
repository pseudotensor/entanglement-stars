#!/usr/bin/env python
"""ls_v11_tables_v1.py (busi, inbox #449): emits the markdown tables of FOUNDATIONS_localsea_v1.1.md from the stored outputs (no
retyped numbers): the slice audit (ls_audit_read_v2.json), the field-solve substitution (ls_reviewbA_3d_v2.json), the reference's rate
at the outer root (ls_audit_read_v2.json, rates), the nonlinear rate coefficients and their two-q extrapolation (exact/rate2_*.json),
the real-time check (exact/ls_exact_v1_evolve_L16.json)."""
import json
import os
import sys

D = sys.argv[1] if len(sys.argv) > 1 else "<internal>/localsea"
A = json.load(open(os.path.join(D, "ls_audit_read_v2.json")))
S = A["slice"]
order = ["s4_T07", "s4_T13", "s4_T14", "s4_T15", "s4_T16", "s8_T14", "s8_T16"]
print("## T1 slice checks")
print("| record | outer root R_t | min(N-|v|) | max eig vs stored | R2 vs P_neg (states) | positive occupied / negative empty | R2 vs lower band by character | mixed vectors | pairs vs stored |")
print("|---|---|---|---|---|---|---|---|---|")
for t in order:
    x = S[t]
    rt = "none" if x["R_t"] is None else "%.2f" % x["R_t"]
    print("| %s | %s | %.3f | %.1e | %.0f | %d / %d | %.1e | %d | %.1e |" % (t, rt, x["min_N_minus_absv"], x["max_eig_vs_stored"], x["max_R2_vs_neg_mismatch"],
          x["positive_occupied_total"], x["negative_empty_total"], x["R2_vs_lower_band_mismatch_max"], x["bN_mixed_total"], x["pairs_vs_stored_eps_pairs_max"]))
print()
print("## T2 radius by radius (shell totals, the engine's eps units)")
for t in order:
    x = S[t]
    print("### %s" % t)
    print("| r window | production polarization eps_R2 - eps_0 | J_R2 - J_0 | 3D local sea (2nd order) eps | J | ratio 3D/production (eps) | record sources eps_tot | R2 - P_neg (eps) |")
    print("|---|---|---|---|---|---|---|---|")
    for w in x["windows"]:
        lo, hi = w["r"]
        ratio = w["local_sea_3d_eps"] / w["pol_eps"] if w["pol_eps"] != 0 else float("nan")
        print("| %.1f to %.1f | %.4g | %.4g | %.4g | %.4g | %.3g | %.4g | %.2g |" % (lo, hi, w["pol_eps"], w["pol_J"], w["local_sea_3d_eps"], w["local_sea_3d_J"],
              ratio, w["sources_eps_tot"], w["R2_minus_neg_eps"]))
    print()
print("## T2b ratios outside the outer root")
for t in order:
    x = S[t]
    if x["R_t"] is None:
        continue
    ws = x["windows"]
    r1 = [w for w in ws if abs(w["r"][0] - x["R_t"]) < 1e-9][0]; r2 = [w for w in ws if abs(w["r"][0] - 2 * x["R_t"]) < 1e-9][0]
    print("%s: [R_t,2R_t] 3D/prod %.1f, 3D/sources %.2f; [2R_t,4R_t] 3D/prod %.0f, 3D/sources %.2f" % (t, r1["local_sea_3d_eps"] / r1["pol_eps"],
          r1["local_sea_3d_eps"] / r1["sources_eps_tot"], r2["local_sea_3d_eps"] / r2["pol_eps"], r2["local_sea_3d_eps"] / r2["sources_eps_tot"]))
print()
print("## T3 one field solve with the reference replaced (reviewer B A)")
p = os.path.join(D, "ls_reviewbA_3d_v2.json")
if os.path.exists(p):
    B = json.load(open(p))
    print("| record | variant | roots | min(N-|v|) | collar charge E | change in E | added eps total | added J total |")
    print("|---|---|---|---|---|---|---|---|")
    for t, x in B.items():
        E0 = x["record"]["E_collar"]
        for lab in ("record", "LS2_3d", "LS2_3d_outside_4a", "flat_P0"):
            y = x[lab]
            print("| %s | %s | %s | %.4f | %.3f | %+.2f%% | %.3g | %.3g |" % (t, lab, ", ".join("%.2f" % z for z in y["zeros"]), y["min"], y["E_collar"],
                  100 * (y["E_collar"] / E0 - 1), y["d_eps_total"], y["d_J_total"]))
print()
print("## T4 reference's rate at the outer root (sigma/a = 4, T/r_s 12.125 to 17.25)")
for h, x in A["rates"].items():
    print("%s: rows %d (%d with J exactly R2); median |ref rate|/largest %.4f (J-exact rows %.4f; energy part only %.4f); median |dM/dt - row(-2J)|/largest %.3f (J-exact rows %.3f), with +2J %.3f; handoff eps_R2 check on non-pocket rows max %.1e of scale %.0f" % (
        h, x["n_rows"], x["n_rows_Jexact"], x["median_abs_ref_rate_rel"], x["median_abs_ref_rate_rel_Jexact"], x["median_abs_ref_rate_energy_only_rel"],
        x["median_abs_resid_minus2J"], x["median_abs_resid_minus2J_rel_Jexact"], x["median_abs_resid_plus2J"],
        max(r_["handoff_eps_R2_maxdiff"] if "handoff_eps_R2_maxdiff" in r_ else 0 for r_ in []) if False else float("nan"), x["handoff_scale"]))
    late = [r_ for r_ in x["rows"] if r_["J_is_R2"] and r_["T_over_rs"] >= 14.7]
    early = [r_ for r_ in x["rows"] if r_["J_is_R2"] and r_["T_over_rs"] < 12.7]
    import numpy as np
    for lab, rr in (("before the pockets (T < 12.7)", early), ("after the pockets (T >= 14.75)", late)):
        if rr:
            print("   %s: %d rows, |ref rate|/largest median %.4f max %.4f; |resid(-2J)|/largest median %.3f" % (lab, len(rr), float(np.median([abs(r_["ref_rate_rel"]) for r_ in rr])),
                  float(max(abs(r_["ref_rate_rel"]) for r_ in rr)), float(np.median([abs(r_["resid_minus2J_rel"]) for r_ in rr]))))
print()
print("## T5 nonlinear rate coefficients")
keys = [("n_udot", "coef_nprime_udot"), ("u_ndot", "coef_ndot_uprime"), ("u_ndot", "coef_u_ndotprime"), ("J_n_ndot", "coef_ndot_nprime"),
        ("J_u_udot", "coef_udot_uprime"), ("J_u_udot", "coef_u_udotprime"), ("n_udot", "coef_n_udotprime"), ("J_n_ndot", "coef_n_ndotprime")]
rows = {}
for tag in ("rate2_L32_j1", "rate2_L64_j2", "rate2_L64_j1", "rate2_L64_j1_n24", "rate2_L96_j1"):
    p = os.path.join(D, "exact", "ls_exact_v1_%s.json" % tag)
    if not os.path.exists(p):
        print("%s: not present" % tag)
        continue
    r = json.load(open(p))
    for j, v in r["rate2"].items():
        rows[tag] = (v["q"], r["ny"], [v[a][b] for a, b in keys])
        print("| %.4f (%s, %d pts/axis) | %s |" % (v["q"], tag, r["ny"], " | ".join("%.4g" % c for c in rows[tag][2])))
if "rate2_L32_j1" in rows and "rate2_L64_j1" in rows:
    c1 = rows["rate2_L32_j1"][2]; c2 = rows["rate2_L64_j1"][2]
    print("two-q intercept (4 c(q/2) - c(q))/3 (assumes no q^4 term):", " | ".join("%.3g" % ((4 * b - a) / 3) for a, b in zip(c1, c2)))
    print("as a fraction of c(q/2):", " | ".join("%.3f" % (((4 * b - a) / 3) / b) for a, b in zip(c1, c2)))
    print("ratio c(q)/c(q/2):", " | ".join("%.2f" % (a / b) for a, b in zip(c1, c2)))
    print("c/q^2 at q and q/2 (an intercept-free q^2 + q^4 reading):", " | ".join("%.4f %.4f" % (a / rows["rate2_L32_j1"][0] ** 2, b / rows["rate2_L64_j1"][0] ** 2) for a, b in zip(c1, c2)))
if "rate2_L32_j1" in rows and "rate2_L64_j2" in rows:
    print("size control (64 sites j=2 vs 32 sites j=1, same q), rel:", " | ".join("%.3f" % (b / a - 1) for a, b in zip(rows["rate2_L32_j1"][2], rows["rate2_L64_j2"][2])))
if "rate2_L64_j1" in rows and "rate2_L64_j1_n24" in rows:
    print("transverse grid (24 vs 40 pts), rel:", " | ".join("%.3f" % (b / a - 1) for a, b in zip(rows["rate2_L64_j1"][2], rows["rate2_L64_j1_n24"][2])))
print()
print("## T6 real-time check")
E = json.load(open(os.path.join(D, "exact", "ls_exact_v1_evolve_L16.json")))
print("Lx %d ny %d nz %d m %.2f B %.0e dt %.3f" % (E["Lx"], E["ny"], E["nz"], E["m"], E["B"], E["dt"]))
for e in E["evolve"]:
    print("K %.4f vh %s R(K) %.4e tau %.0f T %.0f" % (e["K"][0], e["vh"], e["R"], e["tau"], e["T"]))
    for x in e["records"]:
        print("   t %.0f  Tr((C - P(t)) e_x)_K = %.4e %+.4ei  adiabatic %.4ei  ratio %s" % (x["t"], x["dev_K"][0], x["dev_K"][1], x["pred_iR_B0rate_over2"][1],
              "%.4f" % x["ratio_im"] if abs(x["pred_iR_B0rate_over2"][1]) > 1e-20 else "(prediction zero)"))
