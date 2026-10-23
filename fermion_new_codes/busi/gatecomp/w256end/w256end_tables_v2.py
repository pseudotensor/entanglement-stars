#!/usr/bin/env python
"""w256end_tables_v2.py (busi; #477/#478; v1 with table-safe names (no pipes in cells), the res pair, the stored-only statement): emits deliveries/busi/W256_s4_end_reads_v1.md whole from w256end_reads_v1.json (and the
md5 of every input). Usage: w256end_tables_v1.py READS_JSON OUT_MD"""
import json, sys
J = json.load(open(sys.argv[1])); OUT = sys.argv[2]
f = lambda x, s="%.4f": "-" if x is None else (s % x)  # noqa: E731
L = []; P = L.append
b = J["birth"]; Rb = J["R_b"]; A = J["answers"]; V = J["validation"]; tg = J["T_gate_per_test"]; root = J["root"][0]
rows = [r for r in J["global"]["rows"] if r["T"] >= 9.0 - 1e-9]
P("# W256_s4 end: the zero-card gate reads (inbox #477, #478)")
P("")
P("busi, analyst, 23 Oct 2026. Box CPU only (nice 10, 8 threads), no card. Run: W256_s4 run_f (r_s/a 256, sigma/a 4, 4096 sites, K 14, tau 16 a), engine RD-1d "
  "engine_collapse_v36nq_rd13mfd 59a641d2, fields engine_fields_v16 7a69faad. The gate code is the stopper's own: stop_flat_h100f_v2.py 068f8979, gate_common_v1 "
  "3ab38808, gate_profiles_v1 031e934f, at sigma 4 and line 0.1. Every table and number below is emitted by busi/code/gatecomp/w256end/w256end_tables_v2.py from "
  "w256end_reads_v2.json (box <internal>/w256end_reads, MD5SUMS there). Nothing here is ruled.")
P("")
P("## What could and could not be computed")
P("")
P("The run json and fields npz stop at T %s. The records 9.4375 to 9.625 exist only as log lines. The run saved slices every 0.25 r_s, so between 9.25 and 9.625 "
  "the only slices are **9.25 and 9.50** (md5 %s and %s, matching solve's MD5_slices_verified.txt). There are no slices at 9.375, 9.4375, 9.5625 or 9.625. "
  "The six-term needs the record's sources, and the record's sources need the record's state, so **the fields at 9.4375, 9.5625 and 9.625 cannot be re-solved "
  "zero-card**. Making them means propagating the engine from 9.25 or 9.50. The engine's banded eigh is CUDA-only (numba kernels), and the run peaked at 70 GB of "
  "device memory, more than an A6000 has. Box GPU 1 is busy in any case. So the gates below are on every stored record to 9.375, plus the re-solved 9.50 record. "
  "The records 9.4375, 9.5625 and 9.625 carry only their log scalars." % (f(J["json_last"]), J["slices"]["9.2500"]["md5"][:8], J["slices"]["9.5000"]["md5"][:8]))
P("")
P("## Headline (the two answers)")
P("")
P("1. **T_b = %s** (linear zero of min(N - |v|) between 9.375 (+%.4f, json) and 9.4375 (%.4f, log); the 3-point quadratic gives %s). R_b: r_min is not logged at 9.4375, "
  "so the gate_common birth radius cannot be formed exactly. The candidates are: R_t at 9.4375 (log), %.2f a, which is %.3f of 2GE0 (%.2f a) and %.3f of 2GE(T_b) (%.2f a); "
  "and r_min interpolated linearly between 9.375 (%s a, json) and 9.50 (%s a, re-solve) to T_b, %s a, which is %s of 2GE0 and %s of 2GE(T_b). "
  "The record's 2GE is %.2f a at 9.4375." % (
      f(b["T_birth"], "%.5f"), b["m_lo"], b["m_hi"], f(b["T_quad"], "%.5f"), Rb["R_t_9.4375_log"], Rb["R_t_9.4375_log_over"]["2gE0"], J["two_gE0"],
      Rb["R_t_9.4375_log_over"]["2gE_Tb"], Rb["two_gE_Tb"], f(Rb["r_min_9.375_json"], "%.1f"), f(Rb["r_min_9.50_resolve"], "%.1f"), f(Rb["R_b_lin_9.375_9.50"], "%.2f"),
      f((Rb["R_b_lin_9.375_9.50_over"] or {}).get("2gE0"), "%.3f"), f((Rb["R_b_lin_9.375_9.50_over"] or {}).get("2gE_Tb"), "%.3f"), Rb["two_gE_9.4375"]))
gs = J["global"]["T_gate"]
P("2. **Global gate (sigma ||K||_op and the six-term on nodes r >= sigma, line 0.1): T_gate = %s** (per test: sigma ||K||_op %s, six-term %s; reported only: sigma|H| %s, "
  "six-term at r < sigma %s). The stored-only read (to 9.375, what the run's stopper could see) gives T_gate %s: the crossing record 9.3125 and both its d_T neighbours (9.25, 9.375) are stored rows, so T_gate does not depend on the slice re-solve. The window T_gate - T_b is %s r_s." % (
      f(gs), f(tg["sigma_Kop"]), f(tg["six_nodes"]), f(tg["sigma_H (one-term, reported)"]), f(tg["six_centre r<sigma (reported)"]), f(J["global_stored_only"]["T_gate"]), f((gs - b["T_birth"]) if gs is not None else None, "%+.4f")))
P("3. **Root gate at 9.50** (nodes [R_t, 2R_t], R_t %.2f a): sigma ||K||_op %s, sigma |N'/N| %s, six-term %s (d_T backward over 9.375 to 9.50); the largest is %s against 0.1, so it %s." % (
    root["R_t"], f(root["sigma_Kop_R_2R"]), f(root["sigma_absNp_over_N"], "%.2e"), f(root["six_R_2R"]), f(root["max_of_three"]), "CROSSED" if A["root_gate_9.50_crossed"] else "did not cross"))
if gs is not None:
    ans1 = "YES: T_b %s <= T_gate %s, so formation is inside its gate" % (f(b["T_birth"], "%.5f"), f(gs)) if A["T_b_le_T_gate"] else "NO: T_gate %s < T_b %s, so the gate closed before formation" % (f(gs), f(b["T_birth"], "%.5f"))
else:
    ans1 = "UNDECIDED on the computable records (no test crossed by 9.50)"
P("4. **Is T_b <= T_gate (formation inside its gate)? %s.**" % ans1)
if gs is not None and gs <= 9.5 + 1e-9:
    ans2 = "YES: the global gate is already closed at %s, before 9.5625" % f(gs)
elif A["root_gate_9.50_crossed"]:
    ans2 = "YES (root gate crossed at 9.50)"
else:
    ans2 = "NOT DECIDABLE zero-card: no test crosses at or before 9.50, and 9.5625 has no slice and no fields"
P("5. **Was the gate already closed at or before 9.5625? %s.**" % ans2)
P("")
P("## Validation of the re-solve")
P("")
P("The method reconstructs the record's sources as the engine's own sources() at the record's converged cache: eps_tot = sum_k w_k [dens(h_k, sea_k) - dens(h_k, R2_k) + dens(h_k, dust_k, f_k)], "
  "and J_tot the same with Q_k (Chan and dens from engine_collapse_v5). The site fields come from the engine's solve: engine_fields_v16, rtol 1e-12, atol 1e-16, collar flag, "
  "support 1e-14, kernel-averaged. Both routes were run: the run's collocation route and the ivp route.")
P("")
P("| quantity at T 9.25 (slice re-solve against the stored row) | max abs difference | scale (max abs of the row) | bitwise |")
P("|---|---|---|---|")
for k in ("eps_tot", "J_tot", "eps_pairs", "J_pairs", "N_collocation", "v_collocation", "N_ivp", "v_ivp", "N_applied", "v_applied"):
    q = V[k]; P("| %s | %.2e | %.4g | %s |" % (k, q["max_abs"], q["scale"], "yes" if q["bitwise"] else "no"))
P("")
P("At 9.25 the re-solved charge differs from the record's E by %.1e (collocation) and %.1e (ivp), relative. min(N - |v|) on the site fields is %.6f at r %s; the json has %s at r %s." % (
    V["E_resolve_over_record"]["collocation"], V["E_resolve_over_record"]["ivp"], V["min_N_minus_absv_resolve"], f(V["r_min_resolve"], "%.0f"), f(V["min_N_minus_absv_record"], "%.6f"), f(V["r_min_record"], "%.1f")))
t5 = V["T9.50"]
P("")
P("At 9.50 (no stored row): the re-solved E/E0 is %.7f (collocation) and %.7f (ivp), against the log's %.7f. Against the slice's own accepted fields, the re-solved site fields differ by "
  "max |N - N_applied| %.1e and max |v - v_applied| %.1e, inside the run's stop rule (fields 1e-6). On the site fields, min(N - |v|) is %.4f at r %.0f, with the negative sites running from r %s to %s. "
  "The log has min(N - |v|) %.4f, R_t %.3f and r_in %.3f; the engine's min is on its own (dense) evaluation and lies below the site value, as at 9.25, where the json's min (%.6f) is also below the site value (%.6f); the site check is a consistency check, not a reproduction of the logged number." % (
      t5["E_over_E0_resolve"]["collocation"], t5["E_over_E0_resolve"]["ivp"], t5["E_over_E0_log"], t5["N_minus_Napplied"], t5["v_minus_vapplied"], t5["min_N_minus_absv_resolve"],
      t5["r_min_resolve"], f(t5["inner_neg_site"], "%.0f"), f(t5["outer_neg_site"], "%.0f"), t5["min_log"], t5["R_t_log"], t5["r_in_log"], V["min_N_minus_absv_record"], V["min_N_minus_absv_resolve"]))
P("")
P("## Records 9.0 to 9.625: birth bracket, charge, solver (json to 9.375; log beyond)")
P("")
P("| T/r_s | source | E/E0 (direct-trace collar charge) | 1 - E/E0 | min(N - abs v) | R_t | 2gE | R_t / 2gE | Newton it | pockets | inversions | kept residual | log res (two numbers as printed) |")
P("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for t in sorted(J["log"], key=float):
    q = J["log"][t]
    if "E_over_E0" not in q:
        continue
    src = "json+log" if float(t) <= J["json_last"] + 1e-9 else "log only"
    P("| %s | %s | %.7f | %.2e | %+.4f | %s | %.2f | %s | %d | %d | %d | %s | %s |" % (t, src, q["E_over_E0"], 1 - q["E_over_E0"], q["min"], f(q["R_t"], "%.2f"), q["two_gE"],
      f(q["R_t"] / q["two_gE"] if q["R_t"] else None, "%.3f"), q["it"], q["pockets"], q["inversions"], f(q.get("kept"), "%.2e"), " / ".join("%.1e" % x for x in q.get("res_pair") or [])))
P("")
P("The record after 9.625 (9.6875) did not converge; its kept residuals are in the log. E at T_b, interpolated linearly between 9.375 and 9.4375, is %.5f E0." % Rb["E_Tb_over_E0"])
P("")
P("## Global gates per record (sigma 4, line 0.1)")
P("")
P("gate_common_v1.gate_profile, the stopper's own: sigma ||K||_op over (0, r_max] on the record's site fields; the six-term max(|v'/N|, |v/(N r)|, |N'/N|, sqrt|D k_r|, sqrt|D k_t|, sqrt|N''/N|) "
  "on the dense fields re-solved from the record's sources (ivp, collar flag), on the nodes r = sigma, 2 sigma, ... up to r_max (this is what gates). The values at r < sigma are reported and never gate. "
  "r_max is 2 R_t at a record with a root. Before the root it is 2 R_b, with R_b = %.2f a (the r_min interpolation). d_T is the code's own rule on the rows that exist: "
  "9.375 is central over (9.3125, 9.50), a step of 0.1875, and 9.50 is backward over (9.375, 9.50), a step of 0.125." % J["global"]["R_birth_used"])
P("")
P("| T/r_s | r_max | sigma norm(K)_op | at r | sigma abs H (reported) | six-term (nodes) | at r | binding term | d_T | six-term r < sigma (reported) |")
P("|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    P("| %.4f | %.1f | %.4f | %.0f | %.4f | %s | %s | %s | %s | %s |" % (r["T"], r["rmax"], r["sigma_Kop"], r["r_Kop"], r["sigma_H"], f(r["six_nodes"]), f(r["r_six"], "%.0f"), r["six_dominant"], r["dT_mode"], f(r["six_centre"])))
P("| 9.4375, 9.5625, 9.625 | - | NOT COMPUTABLE (no slice, no fields) | | | | | | | |")
P("")
P("Earlier rows (from T_b - 1): the largest sigma ||K||_op over %s to 8.9375 is %s, and the largest six-term is %s." % (
    f(J["global"]["rows"][0]["T"]), f(max([r["sigma_Kop"] for r in J["global"]["rows"] if r["T"] < 9.0 - 1e-9] or [None]) if any(r["T"] < 9.0 - 1e-9 for r in J["global"]["rows"]) else None),
    f(max([r["six_nodes"] for r in J["global"]["rows"] if r["T"] < 9.0 - 1e-9 and r["six_nodes"] is not None] or [None]) if any(r["T"] < 9.0 - 1e-9 for r in J["global"]["rows"]) else None)))
P("")
P("Robustness checks. (i) With r_max = 2 R_t(9.4375) = %.1f a before the root, T_gate is %s. (ii) Stored rows only (9.375 with backward d_T): " % (2 * J["global_alt_rmax"]["R_birth_used"], f(J["global_alt_rmax"]["T_gate"]))
  + "; ".join("%.4f sigma||K||_op %.4f six %s (%s)" % (r["T"], r["sigma_Kop"], f(r["six_nodes"]), r["dT_mode"]) for r in J["global_stored_only"]["rows"] if r["T"] >= 9.25 - 1e-9) + ".")
P("")
P("Per term, sigma times the max on the nodes (value, r):")
P("")
P("| term | " + " | ".join(J["per_term"].keys()) + " |")
P("|---|" + "---|" * len(J["per_term"]))
for nm in list(J["per_term"].values())[0]:
    P("| %s | " % nm.replace("|", "abs ") + " | ".join("%.4f (r %.0f)" % tuple(J["per_term"][k][nm]) for k in J["per_term"]) + " |")
P("")
P("## Root gate (stop_flat_h100f_v2 root_row, postgate)")
P("")
P("| T/r_s | R_t | sigma norm(K)_op on [R_t, 2R_t] | sigma abs(N'/N) | six-term on nodes [R_t, 2R_t] | at r | d_T | max of three |")
P("|---|---|---|---|---|---|---|---|")
P("| %.4f | %.2f | %s | %s | %s | %s | %s | %s |" % (root["T"], root["R_t"], f(root["sigma_Kop_R_2R"]), f(root["sigma_absNp_over_N"], "%.2e"), f(root["six_R_2R"]), f(root["r_six"], "%.0f"), root["dT_mode"], f(root["max_of_three"])))
P("| 9.4375, 9.5625, 9.625 | from log: 69.71, 92.20, 95.57 | NOT COMPUTABLE | | | | | |")
P("")
P("The root did not close on any logged record (R_t grows 69.71, 84.16, 92.20, 95.57).")
P("")
P("## Inputs (md5)")
P("")
for k, q in J["inputs"].items():
    P("- %s: %s (%s)" % (k, q["path"], q["md5"]))
for k, q in J["slices"].items():
    P("- slice %s: %s (%s)" % (k, q["path"], q["md5"]))
open(OUT, "w").write("\n".join(L) + "\n")
print("wrote", OUT)
