#!/usr/bin/env python
"""pilot32_v5_classD_read_v1.py (builder analyst, 2026-10-16; SPEC_final4b_engine.md 36.1 "Ruled" (1), (3), (5) and 36.5 item 1).

Writes results/page/pilot32_v5_classD_read_v1.md: the r_s/a 32 Page pilot's class-D readouts (GAP03 8.2) as one record file. Every number
is printed by this script from the pilot's outputs (pilot32_v5_summary.json, pilot32_v5_tables.md, pilot32_v5_final.log, read from
lead/api/turn42/page_v5_logs/); log and table lines are copied verbatim (the log's NUL bytes from the 16:43 UTC quota incident stripped);
ratios and differences are computed here and marked "(script)". GAP03 8.2's premise is copied verbatim from GAP03_integrated_v1.md.
R4 (36.1 (1)): the "turnover established" flags and the turnover test's outcomes are not printed or cited; nothing here is a Page turnover.

Usage: python pilot32_v5_classD_read_v1.py SUMMARY TABLES FINAL_LOG GAP03 OUT_MD"""
import datetime
import hashlib
import json
import re
import sys

EDGES = [0, 1, 2, 3, 4, 6, 8, 12, 25, 1e9]


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def g(x, fmt="%.4e"):
    if x is None:
        return "-"
    try:
        return fmt % x
    except (TypeError, ValueError):
        return str(x)


def main():
    p_sum, p_tab, p_log, p_gap, p_out = sys.argv[1:6]
    s = json.load(open(p_sum))
    tab = open(p_tab, encoding="utf-8").read().splitlines()
    raw = open(p_log, "rb").read()
    n_nul = raw.count(b"\x00")
    log = raw.replace(b"\x00", b"").decode("utf-8", "replace").splitlines()
    gap = open(p_gap, encoding="utf-8").read().splitlines()
    cfg = s["settings"]; der = s["derived"]; lv = der["levels_initial"]
    rs = float(cfg["rs"]); curve = s["curve"]; stages = s["stages"]; chans = [str(k) for k in cfg["channel_list"]]
    L = []
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    L.append("# The Page pilot read as class D (r_s/a 32, page_*_v5 on the exact exterior sequence): the readouts of GAP03 8.2 (engine spec 36.1 (1))\n")
    L.append("Written %s by code/page/pilot32_v5_classD_read_v1.py (builder analyst; 36.5 item 1). Every number below is printed by the script from the files listed; lines in quotation marks are copied verbatim; \"(script)\" marks a ratio or difference computed by the script from the printed values. Inputs (md5):\n" % now)
    for p in (p_sum, p_tab, p_log, p_gap):
        L.append("- %s (%s)" % (p, md5(p)))
    L.append("- the final log holds %d NUL bytes (the 16:43 UTC quota incident), stripped before reading.\n" % n_nul)
    L.append("Run: %s; stop %s; %d blocks; final %s; validation status (as printed): %s\n" % (s.get("label"), s.get("stop_reason"), s.get("n_blocks"), s.get("final"), s["validation"]["status"]))
    L.append("Labels on this record (36.1 (1), (5)):")
    L.append("- The D8 and formation allowances are not bounds as written (icloud review R1 and R2, review_GAP03_page_v5_correctness.tex): every \"D8 bound\", \"allowance\" and \"uncertainty\" below is the code's ledger quantity, a label, not a certified bound.")
    L.append("- R3: FIFO pruning by present weight does not certify the all-history algebra; in this run no entry left the FIFO (checks fifo_exits = %s, fifo_exit_weight_max = %s)." % (s["checks"].get("fifo_exits"), s["checks"].get("fifo_exit_weight_max")))
    L.append("- R4: the 1e-3 formation test, the 1 percent transition test and the turnover flags are not certificates as written. The \"turnover established\" flags are not cited here, and nothing here is a Page turnover. The fall of S_esc is the escaped sector purifying itself (36.1 (ii)): a measurement with labelled certificates, not a certified bound.")
    L.append("- The code's own labels: %s; %s; %s\n" % (s.get("label_clock"), s.get("label_rule"), s.get("label_levels")))

    # 1 the stage ledger
    L.append("## 1. The stage ledger: emitted energy per stage against E_init,k\n")
    L.append("| stage k | E_k | duration/r_s1 | end | E_rad (memory route) | E_init,k (weighted) | E_rad/E_init,k (script) | dE target | L_mean | L_cont (all, T_H) | L_mean/L_cont (script) | S_esc at end |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for st in stages:
        L.append("| %d | %.6f | %.1f | %s | %.4e | %.4e | %.5f | %.4e | %.3e | %.3e | %.2f | %.5f |" % (
            st["k"], st["E_k"], st["duration_over_rs1"], st["end"], st["E_rad"], st["E_init_weighted"], st["E_rad"] / st["E_init_weighted"],
            st["dE_target"], st["L_mean"], st["L_cont"], st["L_mean"] / st["L_cont"], st["S_esc_end"]))
    L.append("\nThe log's stage lines (verbatim):\n")
    for ln in log:
        if re.search(r"stage \d ends \(", ln) or re.search(r"\] done: ", ln):
            L.append("> %s\n" % ln.strip())
    s0, s1 = stages[0], stages[1]
    E0, E1 = s0["E_k"], s1["E_k"]
    rep = s1["E_init_weighted"] - (s0["E_init_weighted"] - s0["E_rad"])
    L.append("- (script) E_init,0 - E_rad,0 = %.6e against E_init,1 = %.6e: the transition changed the stage's memory energy by %.3e, %.3e of the charge step E_0 - E_1 = %.6e (%.4e of E_0)." % (
        s0["E_init_weighted"] - s0["E_rad"], s1["E_init_weighted"], rep, rep / (E0 - E1), E0 - E1, (E0 - E1) / E0))
    L.append("- (script) E_init,0/E_0 = %.5e against the stage rule's dE/E %g; E_rad,0/E_init,0 = %.4f; E_rad,1/E_init,1 = %.5f." % (
        s0["E_init_weighted"] / E0, cfg["dE_frac"], s0["E_rad"] / s0["E_init_weighted"], s1["E_rad"] / s1["E_init_weighted"]))
    w = {k: 16.0 * float(k) for k in chans}
    einv_w = sum(w[k] * s["inversion"][k]["E_inv"] for k in chans)
    L.append("- The formation's inversion per channel (unweighted): %s; weighted sum_kappa 16 kappa E_inv = %.6e (script) against E_init,0 = %.6e." % (
        "; ".join("kappa %s E_inv %.6e, E_inv/E %.4e, N+ %.4f, N- %.4f, memory rank %s" % (k, s["inversion"][k]["E_inv"], s["inversion"][k]["E_inv_over_E"], s["inversion"][k]["N_plus"],
                                                                                   s["inversion"][k]["N_minus"], s["inversion"][k]["memory_rank"]) for k in chans), einv_w, s0["E_init_weighted"]))
    last = curve[-1]
    L.append("- The late flux, the last block (t %.1f r_s1): L collected %.3e, L collar %.3e, L_cont %.3e; L collected/L_cont %.3e (script).\n" % (
        last["t_over_rs1"], last["L_collected"], last["L_collar"], last["L_cont"], last["L_collected"] / last["L_cont"]))
    for sr in s.get("stage_readouts", []):
        L.append("- Stage %d frozen record: r_H,m %s, kappa_m %s, T_m/T_H %s; the thermal comparator over the run's channels L(T_m) %s, L(T_H) %s." % (
            sr["k"], g(sr.get("r_Hm"), "%.4f"), g(sr.get("kappa_m"), "%.6e"), g(sr.get("T_m_over_T_H"), "%.5f"), g(sr.get("L_comp_T_m"), "%.4e"), g(sr.get("L_comp_T_H"), "%.4e")))
    L.append("")

    # 2 the inverted population by bin; the drained share
    L.append("## 2. The inverted population by energy bin (the formation's) and the drained share\n")
    L.append("| kappa | " + " | ".join("[%g, %s)" % (EDGES[i], "inf" if EDGES[i + 1] >= 1e9 else "%g" % EDGES[i + 1]) for i in range(len(EDGES) - 1)) + " | total |")
    L.append("|---|" + "---|" * len(EDGES))
    for k in chans:
        b = s["inversion"][k]["inverted_by_bin_TH"]
        L.append("| %s | %s | %.4e |" % (k, " | ".join("%.3e" % x for x in b), sum(b)))
    L.append("\nBins in omega/T_H of the first stage's charge E_1 (the code's k = 0, which build_stage labels \"stage 1\"; bin_edges_TH %s); the entries are the inverted population (n_alpha above the reference, 1 - n_alpha below it) summed per bin, per channel, unweighted." % s["inversion"][chans[0]]["bin_edges_TH"])
    L.append("\nThe drained share by energy bin is not in the pilot's outputs: the summary holds the formation's inverted population by bin (above) and the last stage's collected spectrum (section 3), while stage 0's collected spectrum and stage 1's initial inversion by bin are not printed. Each control runs one stage, so its final spectrum is stage 0's collected spectrum against the same kind of formation inversion. ctrl32_v5_table_v1.md prints that ratio per control when the spectrum is complete.\n")

    # 3 the stage-1 spectrum
    L.append("## 3. The stage-1 spectrum: the collected occupations of the last stage at the end (omega/T_H bins)\n")
    for k in chans:
        sp = s["readouts"][k].get("spectrum") or []
        sd = s["readouts"][k].get("stage_direct") or {}
        L.append("kappa %s: %d collected modes in the stage record (stage_direct m %s, complete %s, S_direct of the stage's own collected set %s nats; not S_esc, the earlier stage's radiation being outside the record)." % (
            k, len(sp), sd.get("m"), sd.get("complete"), g(sd.get("S_direct"), "%.6e")))
        L.append("\n| bin omega/T_H | particles: count | sum of excess occupations | mean | holes: count | sum | mean |")
        L.append("|---|---|---|---|---|---|---|")
        npart = sum(1 for x in sp if x[2] > 0)
        for i in range(len(EDGES) - 1):
            sel = [x for x in sp if EDGES[i] <= abs(x[0]) < EDGES[i + 1]]
            pa = [x[1] for x in sel if x[2] > 0]; ho = [x[1] for x in sel if x[2] < 0]
            if not sel:
                continue
            L.append("| [%g, %s) | %d | %s | %s | %d | %s | %s |" % (EDGES[i], "inf" if EDGES[i + 1] >= 1e9 else "%g" % EDGES[i + 1], len(pa), g(sum(pa) if pa else None, "%.4e"),
                                                                 g(sum(pa) / len(pa) if pa else None, "%.3e"), len(ho), g(sum(ho) if ho else None, "%.4e"), g(sum(ho) / len(ho) if ho else None, "%.3e")))
        above = [x for x in sp if x[2] > 0 and abs(x[0]) >= 25]
        L.append("\n(script) kappa %s: %d of %d particle modes at omega at or above 25 T_H, their mean excess occupation %s.\n" % (
            k, len(above), npart, g(sum(x[1] for x in above) / len(above) if above else None, "%.3e")))
    L.append("The tables' lines (verbatim):\n")
    on = False
    for ln in tab:
        if "energy-resolved collected occupations of the last stage" in ln:
            on = True
        elif on and not ln.startswith("    - ["):
            on = False
        if on or (ln.startswith("- kappa ") and "accumulated truncation entropy bound" in ln):
            if ln.startswith("- kappa "):
                L.append("> " + ln[:ln.find("stage direct route")].rstrip() + " (the stage direct route's json follows in the tables)")
            else:
                L.append("> " + ln.strip())
    L.append("")

    # 4 the maximum and the saturation lines
    L.append("## 4. S_esc's maximum against eta A/a^2, A/(4G) and the thermal comparator\n")
    to = s["turnover"]
    tmax = to["t_S_max"]
    row_max = max(curve, key=lambda r: r["S_esc_total"])
    L.append("- S_max %.5f nats at t %.1f r_s1 (stage %d; turnover.t_S_max %.1f lattice units); eta A/a^2 (initial) %.1f, A/(4G) (initial) %.1f; S_max over them %.4e and %.4e (script)." % (
        to["S_max"], row_max["t_over_rs1"], row_max["stage"], tmax, lv["eta_A_over_a2"], lv["A_over_4G"], to["S_max"] / lv["eta_A_over_a2"], to["S_max"] / lv["A_over_4G"]))
    sd = lv["S_dot_cont"]
    L.append("- The thermal comparator S_dot_cont t (S_dot_cont %.6e per lattice time unit, levels_initial): over the same time as the maximum, t %.1f r_s1, %.4f nats (script); over the run, to the last row's t %.1f r_s1, %.4f nats (script). S_max over the first: %.3f (script)." % (
        sd, tmax / rs, sd * tmax, last["t"] / rs, sd * last["t"], to["S_max"] / (sd * tmax)))
    for ln in tab:
        if ln.startswith("Largest S_esc total"):
            cut = ln.find(" Turnover:")
            L.append("- The tables' saturation line (verbatim, its last sentence, a turnover flag, not cited: R4): \"%s\"" % (ln[:cut] if cut > 0 else ln).strip())
    L.append("- Flag (script check of the code): the tables' \"thermal comparator over the same time\" is page_recursion_v5.py line 1021's lv[\"S_dot_cont\"] * rows[-1][\"t\"], the run's last row (t %.1f r_s1), not the maximum's time t %.1f r_s1. Spec 36.1's facts and Ruled (1) read the tables' %.4f as the comparator over the same time as S_max; over that time it is %.4f." % (
        last["t"] / rs, tmax / rs, sd * last["t"], sd * tmax))
    L.append("")

    # 5 the fall
    L.append("## 5. S_esc(t) at every block: the rise and the fall to the end\n")
    L.append("Label (36.1 (5)): the fall rests on the memory route. The archive direct-route check found that the stored checkpoints hold every collected mode: the final checkpoint's FIFO, with stage 0's entries carried through the ramp, and stage 1's complete record. Neither checkpoint holds stage 0's initial memory or its coefficients, which were replaced at the transition (page_core_v5 line 1167), and the memory route keeps no correlation between stage 0's radiation and stage 1's initial memory. So no direct route over the whole archive at t 300 is formed from the stored records (results/page/pilot32_v5_ckpt_inspect_v1_stdout.txt, by code/page/inspect_pilot32_ckpt_v1.py). The rise's values stand on both routes at the stage-0 direct checkpoints (section 8).\n")
    L.append("| t/r_s1 | stage | S_esc total | %s | S_direct total | MI across r_t total (d 2) | L collected | L collar | rank (weighted) |" % " | ".join("S_esc kappa %s" % k for k in chans))
    L.append("|---|---|---|" + "---|" * len(chans) + "---|---|---|---|---|")
    for r in curve:
        mi = (r.get("MI_cross_total") or {})
        L.append("| %.1f | %d | %.5f | %s | %s | %s | %.3e | %.3e | %.0f |" % (
            r["t_over_rs1"], r["stage"], r["S_esc_total"], " | ".join("%.5e" % r["per_channel"][k]["S_new"] for k in chans), g(r.get("S_direct_total"), "%.5f"),
            g(mi.get("2"), "%.3e"), r["L_collected"], r["L_collar"], r["memory_rank_weighted"]))
    L.append("")

    # 6 the MI across r_t
    L.append("## 6. The mutual information across the trapping zero r_t (the exact stage state)\n")
    L.append("Weighted totals at the direct-route checkpoints (the curve's MI_cross_total, nats; d is the band width in sites on each side):\n")
    for r in curve:
        mi = r.get("MI_cross_total")
        if mi:
            L.append("- t %.1f r_s1 (stage %d): %s" % (r["t_over_rs1"], r["stage"], ", ".join("d %s: %.4e" % (d_, v_) for d_, v_ in sorted(mi.items(), key=lambda z: int(z[0])))))
    L.append("\nPer channel (unweighted) at each stage's end and at the run's end, physical I against its bound and the reference vacuum's I:\n")
    for st in stages:
        for k, lst in (st.get("cross_surface_end") or {}).items():
            L.append("- stage %d end, kappa %s: %s" % (st["k"], k, "; ".join("d %d: I %.4e (bound %.3e), reference vacuum %.4e" % (c["d"], c["physical"]["I"], c["physical"]["bound_2K2"], c["reference_vacuum"]["I"]) for c in lst)))
    for k in chans:
        lst = s["readouts"][k].get("cross_surface") or []
        L.append("- the run's end, kappa %s: %s" % (k, "; ".join("d %d: I %.4e (bound %.3e), reference vacuum %.4e" % (c["d"], c["physical"]["I"], c["physical"]["bound_2K2"], c["reference_vacuum"]["I"]) for c in lst)))
    L.append("\n(script) Weighted (16 kappa) physical I, the reference vacuum's I and their difference, per band width d:\n")
    ends = [("stage %d end" % st["k"], st.get("cross_surface_end") or {}) for st in stages] + [("the run's end", {k: s["readouts"][k].get("cross_surface") or [] for k in chans})]
    for lab, dct in ends:
        if not dct:
            continue
        ds = sorted({c["d"] for lst in dct.values() for c in lst})
        parts = []
        for d_ in ds:
            ip = sum(w[k] * c["physical"]["I"] for k, lst in dct.items() for c in lst if c["d"] == d_)
            iv = sum(w[k] * c["reference_vacuum"]["I"] for k, lst in dct.items() for c in lst if c["d"] == d_)
            parts.append("d %d: physical %.5e, vacuum %.5e, difference %.3e" % (d_, ip, iv, ip - iv))
        L.append("- %s: %s" % (lab, "; ".join(parts)))
    L.append("")

    # 7 the premise
    L.append("## 7. The premise, in GAP03 8.2's words, as a result of the model as declared (36.1 (1))\n")
    prem = None; dfn = None
    for ln in gap:
        if ln.startswith("- Drain (class D)"):
            i0 = ln.find("the premise responsible")
            i1 = ln.find(" The replacement tests")
            prem = ln[i0:i1] if (i0 >= 0 and i1 > i0) else None
            dfn = ln[:ln.find("; reported:")] if "; reported:" in ln else None
    L.append("GAP03_integrated_v1.md 8.2, the class's definition (verbatim): \"%s\"\n" % (dfn or "-"))
    L.append("The premise (verbatim): \"%s\"\n" % (prem or "-"))
    L.append("Read here (36.1 (1)) as a result of the model as declared, not a theorem of this record.\n")

    # 8 the revised criteria as labels
    L.append("## 8. 36.1 (3)'s revised criteria, applied as labels (no code edit of v5)\n")
    for tr in s["transitions"]:
        L.append("Transition %d -> %d (the code's numbering), attempts:" % (tr["k_from"], tr["k_from"] + 1))
        for at in tr["attempts"]:
            L.append("- tau_tr x %g: S-continuity defect %.3e, accumulated D8 bound %.3e nats (1 percent of S_esc %.3e), leak max %.2e; the printed check %s (the absolute 1e-6 then in force)" % (
                at["tau_factor"], at["S_continuity_defect_max"], at["D8_bound_weighted"], 0.01 * at["S_esc"], at["leak_max"], "pass" if at["pass"] else "FAIL"))
        k_ = tr["attempts"][-1]
        L.append("- (script) the kept attempt (tau_tr x %g): |defect| %.3e at or below its accumulated D8 bound %.3e: %s (the revised criterion: |S_after - S_before| at or below the transition's accumulated D8 bound)." % (
            k_["tau_factor"], abs(k_["S_continuity_defect_max"]), k_["D8_bound_weighted"], abs(k_["S_continuity_defect_max"]) <= k_["D8_bound_weighted"]))
        if len(tr["attempts"]) >= 2:
            d = [a["S_continuity_defect_max"] for a in tr["attempts"]]
            L.append("- (script) the defect's ratio per doubling: %s." % ", ".join("x%g/x%g %.4f" % (tr["attempts"][i + 1]["tau_factor"], tr["attempts"][i]["tau_factor"], d[i + 1] / d[i]) for i in range(len(d) - 1)))
    L.append("\nThe direct route against the memory route at its checkpoints (stage 0; the revised criterion: relative 1e-6 or absolute 1e-9 nats, whichever is larger):\n")
    L.append("| t/r_s1 | kappa | S_direct | S_memory | abs (nats) | rel | threshold max(1e-6 S, 1e-9) | passes (script) |")
    L.append("|---|---|---|---|---|---|---|---|")
    n_pass = n_all = 0
    for r in curve:
        for k in chans:
            pc = r["per_channel"][k]
            if "S_direct" in pc:
                a_ = abs(pc["S_direct"] - pc["S_new"]); thr = max(1e-6 * pc["S_new"], 1e-9); ok = a_ <= thr
                n_all += 1; n_pass += int(ok)
                L.append("| %.1f | %s | %.10e | %.10e | %.3e | %.3e | %.3e | %s |" % (r["t_over_rs1"], k, pc["S_direct"], pc["S_new"], a_, a_ / max(pc["S_new"], 1e-12), thr, ok))
    L.append("\n(script) %d of %d checkpoint-channels pass the revised direct-route criterion.\n" % (n_pass, n_all))

    # 9 the unresolved items and the other checks
    L.append("## 9. The uncollected weight (the cap) and the other checks\n")
    rows = []
    for r in curve:
        for k in chans:
            pc = r["per_channel"][k]
            an = max(pc.get("active_norm", 1.0) or 1.0, 1.0)
            rows.append((pc.get("weight_beyond_cut_uncollected", 0.0) / an, r["stage"], r["t_over_rs1"], k, pc.get("m_coll"), pc.get("m_cap_used"), pc.get("r_before")))
    rows.sort(reverse=True)
    top = rows[0]
    L.append("- (script) the largest uncollected weight over the active norm %.3e at stage %d t %.1f kappa %s (m_coll %s, m_cap_used %s, r_before %s); block-channels at or above 1e-6: %d of %d; block-channels where the cap could bind (r_before above m_cap_used): %d of %d." % (
        top[0], top[1], top[2], top[3], top[4], top[5], top[6], sum(1 for x in rows if x[0] >= 1e-6), len(rows),
        sum(1 for x in rows if (x[6] or 0) > (x[5] or 0)), len(rows)))
    c = s["checks"]
    keys = ["Z_identity_max", "energy_identity_max_rel", "energy_identity_checkpoints", "coverage_identity_max_rel", "memory_mismatch_max_rel", "clip_excursion_max",
            "fifo_overlap_after_max", "Xi_min_eig_min", "I_minus_DD_minus_Xi_min", "direct_vs_memory_max_rel", "direct_checkpoints", "uncollected_over_active_max",
            "uncollected_content_F_max", "uncollected_content_E_max_abs", "fifo_exits", "fifo_exit_weight_max", "fifo_oldest_weight_max", "fifo_depth_doublings",
            "post_split_inward_weight_max", "D8_bound_over_S_esc_at_transitions_max", "transition_leak_max", "transition_S_continuity_max", "transitions_flagged",
            "restores", "restores_inexact", "transition_device_peak_GB_max"]
    L.append("- The summary's checks: " + "; ".join("%s %s" % (k_, c.get(k_)) for k_ in keys) + ".")
    L.append("- The charge sequence's accumulated energy error bound: %s." % json.dumps(s.get("energy_error_bound")))
    L.append("\nRead in turn 45 with the controls (ctrl32_v5_table_v1.md). Item 7's text is turn 45's; nothing here prints as a Page turnover.")
    with open(p_out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("wrote %s (%d lines)" % (p_out, len(L)))


if __name__ == "__main__":
    main()
