#!/usr/bin/env python
"""engine_collapse_v26_smoke_report_v1.py (builder analyst; the brief BRIEF_v26_builder_S4_S9, the unit tests: "S4 (a) and (b) at small size:
the r_s 16 member (512 sites, R_0 = 96, (k_0, w) from 23.4's admissible set) through the preparation and 8 re-solves on fields v17,
printing the interior lapse, the continued interval, the iteration counts and E/E0 (23.4's smoke criteria)"). Reads engine_collapse_v26
massless runs (--run label=dir/tag, repeatable) and prints: the preparation (N_B, E0, passes, 23.4's packet diagnostics: the outgoing and
low-k weights, the interior's energy against E), S4 (b)'s slice energy against the thin and thick null values, S4 (c)'s real-skew defects,
the empty interior at T = 0; per record T/rs, E/E0, the iterations and the stop (converged, or the stall: residual, contraction ratios,
signed cosines, not contracting, the finite-difference sigma_min(I - DF)), the continued intervals, the empty interior, the interior lapse
N(0) against 1 + (r_s/R)^(1/2) at R_c and R_in, the shell's radii and rates against the thin null shell, -J/rho, the certificate's
failures (S4 (e)), the unresolved steps (S9 (a)), the transported pairs (S4 (d)); and 23.4's three smoke criteria per run (no continued
interval at T = 0 and over the re-solves; at most 10 iterations; |E/E0 - 1| within 1e-4). Every number is printed by the script.
Args: --run label=dir/tag (repeatable) --md --json."""
import argparse
import json
import os


def f(x, fm="%.6g"):
    return (fm % x) if isinstance(x, (int, float)) and not isinstance(x, bool) else ("-" if x is None else str(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=[]); ap.add_argument("--md", required=True); ap.add_argument("--json", required=True)
    a = ap.parse_args()
    out = {"script": "engine_collapse_v26_smoke_report_v1.py", "runs": []}
    L = ["## S4 (a), (b) at small size: the massless member at r_s 16 through the preparation and the re-solves on engine_collapse_v26 (engine_collapse_v26_smoke_report_v1.py)", ""]
    for spec in a.run:
        lab, path = spec.split("=", 1)
        d, tag = os.path.split(path)
        J = json.load(open(os.path.join(d, "collapse_%s.json" % tag)))
        A = J["args"]; H = J["head"]; sh = H.get("shell") or {}; dg = (sh.get("diagnostics") or {})
        se = H.get("slice_energy") or {}; t0 = se.get("thin_at_R0") or {}; tc = se.get("thin_at_Rc") or {}
        recs = [r for r in J["records"]]
        run = {"label": lab, "path": path, "args": {k: A.get(k) for k in ("rs", "nsites", "tau", "ell", "kappa_sea", "g", "m", "shell_R0_over_rs", "shell_width_over_rs", "shell_k0",
                                                                         "ops", "fields", "empty_rel", "stop_rule", "oriented_isolation", "mix", "maxit", "c_LZ", "site_fields")}}
        L += ["### %s: %s" % (lab, tag), "",
              "- member: r_s %s, %s sites, tau %s, ell %s, %s channels, g %s, m %s; R_0 %s r_s, w %s r_s, k_0 %s; v26: ops %s (isolation %s), fields %s (empty_rel %s), stop rule %s, mix %s, maxit %s, c_LZ %s." % (
                  A["rs"], A["nsites"], A["tau"], A["ell"], A["kappa_sea"], A["g"], A["m"], A.get("shell_R0_over_rs"), A.get("shell_width_over_rs"), A.get("shell_k0"), A.get("ops"),
                  A.get("oriented_isolation"), A.get("fields"), A.get("empty_rel"), A.get("stop_rule"), A.get("mix"), A.get("maxit"), A.get("c_LZ")),
              "- preparation: N_B %s, E0 %s, %s passes; packets %s per channel at k %s; occupied %s; the largest outgoing and low-k weights of the occupied packets %s, %s; the interior's (r < R_0 - 3w = %s) smoothed energy %s of E; N at the centre %s, inside the shell %s; the gauge at T = 0: %s." % (
                  f(H.get("N_B"), "%.4f"), f(H.get("E0"), "%.6f"), H.get("passes"), sh.get("packets_per_channel"), [round(x, 4) for x in (sh.get("ks") or [])],
                  sh.get("occupied"), f(max([p.get("outgoing_weight", 0.0) for p in (dg.get("packets") or {}).values()] or [None]), "%.2e"),
                  f(max([p.get("weight_below_1_over_w", 0.0) for p in (dg.get("packets") or {}).values()] or [None]), "%.2e"), f(dg.get("interior_radius")),
                  f(dg.get("interior_energy_over_E"), "%.3e"), f(dg.get("N_centre"), "%.8f"), f(dg.get("N_inside_shell"), "%.8f"), dg.get("gauge_T0")),
              "- S4 (b): the slice energy mu = sum_n eps_dust,n = %s (smoothed %s) over E = %s: mu/E %s; the thin null shell's mu/E = 2 [s - ln(1 + s)]/s^2 with s = (r_s/R)^(1/2), r_s = 2 g E = %s: %s at R_0 = %s (s %s), %s at the prepared R_c = %s; with the brief's s = r_s/R: %s at R_0, %s at R_c; the thick null value int dM/(1 + (2gM/r)^(1/2)) on the solved profile over E: %s; the formula at R = r_s: %s; the shell's -J/rho: %s." % (
                  f(se.get("mu_sites"), "%.8f"), f(se.get("mu_smoothed"), "%.8f"), f(se.get("E_collar"), "%.8f"), f(se.get("mu_sites_over_E"), "%.6f"), f(se.get("r_s_2gE"), "%.6f"),
                  f(t0.get("mu_over_E"), "%.6f"), f(t0.get("R"), "%.4f"), f(t0.get("s"), "%.6f"), f(tc.get("mu_over_E"), "%.6f"), f(tc.get("R"), "%.4f"),
                  f(t0.get("mu_over_E_brief_s"), "%.6f"), f(tc.get("mu_over_E_brief_s"), "%.6f"), f(se.get("mu_null_thick_over_E"), "%.6f"), f(se.get("formula_at_R_eq_rs"), "%.8f"),
                  f(se.get("mean_minus_J_over_rho"), "%.6f")),
              "- S4 (c): max |h + h^*| over max |h| per channel at the initial fields: %s." % json.dumps(H.get("realskew")),
              "- the empty interior at T = 0 (fields v17): %s." % ((H.get("empty_interior_T0") or {}).get("label") or (H.get("empty_interior_T0") or {}).get("rule")), "",
              "| T/rs | E/E0 | iterations | stop | scaled residual norms | ratios (cosines) | sigma_min(I - DF) FD (dim; sigma_max) | continued intervals | empty interior | N(0) | 1 + (r_s/R_c)^(1/2) | 1 + (r_s/R_in)^(1/2) | R_c (thin trajectory) | R_in (thin trajectory) | dR_c/dT (thin) | dR_in/dT (thin) | -J/rho | certificate failures | S9a unresolved (protected) | transported pair-substeps |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        crit_cont = []; crit_it = []; crit_E = []; stopped = []
        rows = []
        for r in recs:
            if r.get("E") is None:
                L.append("| %s | failed: %s |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |" % (f(r.get("T", 0) / A["rs"], "%.3f"), r.get("failed")))
                rows.append({"T_over_rs": r.get("T", 0) / A["rs"], "failed": r.get("failed")})
                stopped.append((r.get("T", 0) / A["rs"], r.get("failed")))
                continue
            v = r.get("v26") or {}; s4 = v.get("s4a") or {}; c = v.get("certificate") or {}; dd = v.get("step_diag") or {}; ei = v.get("empty_interior") or {}
            u = r.get("unconverged") or {}; st = u.get("stall") or {}; fd = st.get("fd") or {}
            g = r.get("gauge") or {}
            ivs = g.get("intervals") or []
            stop = "converged" if r.get("converged") else ("unconverged (residual %s%s%s)" % (f(u.get("residual"), "%.2e"), "; not contracting" if u.get("not_contracting") else "",
                                                                                              "; a solve failed at iteration %s" % (u.get("solve_failed") or {}).get("iteration") if u.get("solve_failed") else ""))
            rn = (v.get("stop") or {}).get("scaled_residual_norms") or []
            rat = st.get("ratios") or []; cos = st.get("signed_cosines") or []
            tcr = s4.get("thin_at_Rc") or {}; tir = s4.get("thin_at_Rin") or {}
            L.append("| %.3f | %.8f | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s (%s) | %s (%s) | %s (%s) | %s (%s) | %s | %s of %s | %s (%s) of %s | %s |" % (
                r["T_over_rs"], r["E_over_E0"], r["iterations"], stop, ", ".join("%.2e" % x for x in rn[:6]), ", ".join(("%.2e (%+.2f)" % (x, y)) if (x is not None and y is not None) else "-" for x, y in zip(rat[:5], cos[:5])),
                ("%s (%s; %s)" % (f(fd.get("sigma_min_I_minus_DF"), "%.3e"), fd.get("dim"), f(fd.get("sigma_max_I_minus_DF"), "%.3e"))) if fd else "-",
                ", ".join("[%.2f, %.2f]" % (x, y) for x, y in ivs) if ivs else "none", ("[0, %.3f]" % ei["r_e"]) if ei.get("r_e") is not None else "none",
                f(s4.get("N_centre"), "%.6f"), f(tcr.get("N_in"), "%.6f"), f(tir.get("N_in"), "%.6f"), f(s4.get("R_c"), "%.4f"), f(s4.get("R_c_thin_trajectory"), "%.4f"),
                f(s4.get("R_in"), "%.4f"), f(s4.get("R_in_thin_trajectory"), "%.4f"), f(s4.get("dRc_dT"), "%.4f"), f(s4.get("thin_Rdot_at_mid_Rc"), "%.4f"),
                f(s4.get("dRin_dT"), "%.4f"), f(s4.get("thin_Rdot_at_mid_Rin"), "%.4f"), f(s4.get("mean_minus_J_over_rho"), "%.4f"), c.get("failures"), c.get("channels"),
                dd.get("unresolved"), dd.get("unresolved_protected"), dd.get("pairs"), c.get("oriented_pair_substeps")))
            rows.append({"T_over_rs": r["T_over_rs"], "E_over_E0": r["E_over_E0"], "iterations": r["iterations"], "stop": stop, "scaled_residual_norms": rn, "ratios": rat, "cosines": cos,
                         "fd": fd, "continued": ivs, "empty_interior": ei.get("r_e"), "s4a": s4, "certificate": {"failures": c.get("failures"), "channels": c.get("channels")},
                         "step_diag": {"unresolved": dd.get("unresolved"), "protected": dd.get("unresolved_protected"), "pairs": dd.get("pairs")},
                         "transported": c.get("oriented_pair_substeps")})
            crit_cont.append(bool(ivs)); crit_it.append(int(r["iterations"])); crit_E.append(abs(float(r["E_over_E0"]) - 1.0))
        resolves = len([q for q in rows if not q.get("failed")]) - 1
        c1 = not any(crit_cont); c2 = max(crit_it) <= 10 if crit_it else None; c3 = max(crit_E) <= 1e-4 if crit_E else None
        run["records"] = rows
        run["criteria"] = {"no_continued_interval": c1, "records_with_continued_interval": sum(crit_cont), "at_most_10_iterations": c2, "largest_iterations": max(crit_it) if crit_it else None,
                           "E_within_1e-4": c3, "largest_abs_E_over_E0_minus_1": max(crit_E) if crit_E else None, "resolves": resolves,
                           "stopped": [{"T_over_rs": t_, "failed": m_} for t_, m_ in stopped], "resolves_requested": int(round(float(A["T_over_rs"]) * float(A["rs"]) / float(A["tau"])))}
        out["runs"].append(run)
        L += ["", "- 23.4's smoke criteria over T = 0 and the %d completed re-solves (of %d requested): no continued interval %s (%d records with one); at most 10 iterations %s (the largest %s); |E/E0 - 1| within 1e-4 %s (the largest %s)%s." % (
            resolves, run["criteria"]["resolves_requested"], "PASS" if c1 else "FAIL", sum(crit_cont), "PASS" if c2 else "FAIL", f(max(crit_it) if crit_it else None), "PASS" if c3 else "FAIL", f(max(crit_E) if crit_E else None, "%.3e"),
            ("; the run stopped at T/rs %s: %s" % (", ".join("%.3f" % t_ for t_, m_ in stopped), "; ".join(m_ for t_, m_ in stopped))) if stopped else ""), ""]
    L += ["- every number above is printed by the script from the runs' records."]
    open(a.md, "w").write("\n".join(L) + "\n")
    json.dump(out, open(a.json, "w"), indent=1, default=float)
    print("\n".join(L))


if __name__ == "__main__":
    main()
