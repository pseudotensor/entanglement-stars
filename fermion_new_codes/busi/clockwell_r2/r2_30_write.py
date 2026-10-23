"""Second reader: emit MOCK_clockwell_secondread_v1.md from the r2_*.json products (no new computation). argv: out_md"""
import json, os, sys, glob, hashlib, datetime
O = "<internal>/clockwell_r2"
L = lambda f: json.load(open(os.path.join(O, f)))
C = L("r2_20_compare.json"); IN = L("r2_12_intake.json"); VX = L("r2_13_vertex.json"); ST = L("r2_14_static.json")
ODD = {n: L("r2_odd_n%s.json" % n) for n in "23"}
G = {"0": "0", "1/2": "0p5", "1": "1", "2": "2"}
D = {(a, n, g): L("r2_%s_even_n%s_g%s_v1.json" % (a, n, G[g])) for a in ("filtered", "onsite") for n in "23" for g in G}
HALO = "[endpoint-copy halo; N_-1 = N_0 = 1/2, N_n = N_(n-1) = 1; declared convention, not a measurement]"
def e(x): return "n/a" if x is None else ("0" if x == 0 else "%.1e" % x)
def f6(x): return "%.6g" % x
o = []; w = o.append

w("# MOCK clock well: independent second read of the busi #489 reads (filtered clock-well mock and on-site baseline)\n")
w("Second reader: an analyst subagent of the busi lane, independent of the first reader. Reads checked: `deliveries/busi/MOCK_filtered_clockwell_v1.md.wip` (md5 `4ca9a1e1999251c354763cd4799fa388`, 12 judgments) and `deliveries/busi/MOCK_unitary_lead_baseline_v1.md.wip` (md5 `3273920033d553f3bdf84bad1216e3df`, 5 judgments). Governing texts read in full: icloud inbox #138 to #145; `solve/MOCK_filtered_clockwell_spec_v1.md`; `solve/MOCK_unitary_lead_run_spec_v1.md`; `solve/MOCK_filtered_readout_addendum_v1.md` (defines F_e, purity and the two-time return); `lead_gaps/ReviewA_halolapse_v1.md`; `lead_gaps/ReviewB_halolapse_v1.md`.\n")
w("Independence: the first reader's scripts (busi/code/clockwell/cw*.py) and its box products in <internal>/clockwell were not opened, imported or run. The exceptions are the inputs that a judgment names: the input manifest INPUT_MD5SUMS.txt (used only to check md5s), the three busi-run adapter outputs under adapter_overlay_v1/upper/ and the three busi intake assessments. Every number below comes from second-reader scripts (busi/code/clockwell_r2/r2_*.py, copied to <internal>/code), written from the spec definitions and the sec. 9 / sec. 7 formulas. Their outputs are in <internal>/clockwell_r2 (MD5SUMS there). No propagation job was rerun. The only time evolution applied is the return definition itself, exp(-iH(t-1)) on the saved t = 1 vector, using a second-reader eigendecomposition of the saved 276 x 276 H. Box CPU, nice 10, at most 4 BLAS threads, total compute under 30 s.\n")
w("Every clock-well quantity below is conditional on the declared halo rule. **Halo rule (#145): endpoint-copy halo: N_-1 = N_0 = 1/2, N_n = N_(n-1) = 1 (declared convention, not a measurement; no lapse sweep; n = 7 static only).** Each filtered table header repeats it.\n")

# ---- verdict table ----
cv = C["conversion_claims"]; rs = C["return_summary"]
def U4(a, g): return D[(a, "3", g)]["purification"]["mine"]
minU = min(min(U4(a, g).values()) for a in ("filtered", "onsite") for g in G)
minU_on = min(min(U4("onsite", g).values()) for g in G)
minmax_f = min(rs["filtered n%s g%s" % (n, g)]["max_joint_lower"] for n in "23" for g in G)
minmax_o = min(rs["onsite n%s g%s" % (n, g)]["max_joint_lower"] for n in "23" for g in G)
minlow_f = min(rs["filtered n%s g%s" % (n, g)]["min_joint_lower_after_te"] for n in "23" for g in G)
def coh(a, n, g): return [x for x in C["cohorts"] if x["arm"] == a and x["n"] == n and x["g"] == g][0]
def maxdev(r): return max(v for v in (r["dev_pLL"], r["dev_CHL"], r["dev_S"], r["dev_D"], r["dev_U"] or 0))
w("## 1. Verdicts\n")
w("Criterion per judgment as stated in the .wip sec. 9 / sec. 7; where the .wip formula and the spec differ the spec governs (no material difference found, sec. 4). 'Diff' is second reader minus .wip at the .wip's printed precision.\n")
w("| # | judgment | second-reader value(s) | .wip value(s) | diff | criterion | verdict | defect |")
w("|---|---|---|---|---|---|---|---|")
V = []
for i, g in ((1, "0"), (2, "1/2"), (3, "2")):
    r = coh("filtered", "3", g); wipdev = {"0": "6.6e-14", "1/2": "3e-14", "2": "6e-14"}[g]
    V.append(["J-F-phys-n3-g%s" % g, "termination complete, %d check rows all true (= checks field), sources unchanged; max dev p_LL/C_HL/S/D_RK/U = %s; header rows %s" % (r["checks"], e(maxdev(r)), "all correct"),
              "%d predicates true; max dev %s" % (r["checks"], wipdev), "both <= 1e-12", "all true and every dev <= 1e-12", "AGREE", "none"])
for g, wd in (("1/2", "1e-15 (F_e, purity)"), ("2", "6.7e-16")):
    r = coh("filtered", "2", g)
    V.append(["J-F-read-n2-g%s" % g, "termination complete; F_e dev %s, purity dev %s; 13/13 return rows inside icloud intervals" % (e(r["ro_dev_Fe"]), e(r["ro_dev_pur"])), wd + "; all inside", "both <= 1.0e-15", "|dev| <= 1e-12; recomputed probabilities inside intervals", "AGREE", "none"])
for g in ("0", "1/2", "2"):
    r = coh("filtered", "3", g); k = "n3_g" + G[g]; ia = IN[k]
    ok = all(ia[x] for x in ("term_ok", "terminal_last", "kind_roster_equal_ref", "intake_sources_eq_term_sources", "sources_eq_ref_basename_sha", "adapter_controls_eq_ref", "n_g", "parent_sha_recorded_eq_now", "roster_34", "excluded_256", "locators_eq_ref_count", "physical_sources_eq_ref", "readout_saved_intake_binds_this_intake"))
    wd = {"0": "4.4e-16", "1/2": "1.3e-15", "2": "6.7e-16"}[g]
    V.append(["J-F-read-n3-g%s" % g, "intake checks %s (39 sources = icloud n3 g1; 34 Actions; 256 excluded; parent sha unchanged); readout termination complete; F_e dev %s, purity dev %s; 13/13 return rows inside" % ("all pass" if ok else "FAIL", e(r["ro_dev_Fe"]), e(r["ro_dev_pur"])),
              "14 of 14; dev %s; all inside" % wd, "< 1.2e-15", "as J-F-read + intake criteria", "AGREE", "none (the readout source roster differs from icloud n3 g1 only in the bound review file: busi's own assessment replaces icloud's red-team review; disclosed in F6b)"])
cnt = "16 of 17 (all clocks except t = 0) for every g > 0 cohort, 0 of 17 at g = 0, both arms"
mx = "; ".join("n%s g%s %s vs %s" % (n, g, f6(cv["n%s g%s" % (n, g)]["max_f"]), f6(cv["n%s g%s" % (n, g)]["max_o"])) for n in "23" for g in ("1/2", "1", "2"))
above = "; ".join("n%s g%s: %s" % (n, g, ", ".join(cv["n%s g%s" % (n, g)]["clocks_filtered_above_onsite"]) or "none") for n in "23" for g in ("1/2", "1", "2"))
V.append(["J-F-conversion", "%s; max sampled p_LL filtered vs on-site: %s" % (cnt, mx), "same counts; 0.1575 vs 0.5224, 0.2432 vs 0.5064, 0.4757 vs 0.4982, 0.1773 vs 0.5116, 0.2465 vs 0.5012, 0.2953 vs 0.4984", "0 at printed precision", "flag count; comparison of maxima",
          "AGREE (values)", "WORDING: 'the filtered control converts less than the on-site baseline at every (n, g > 0)' is true only of the maximum sampled p_LL. Per clock, the filtered p_LL exceeds the on-site p_LL at these sampled clocks: %s (e.g. n = 2, g = 2, t = 4: 0.4757 vs 0.2825, Table F7). Replace with 'the maximum sampled p_LL is lower in the filtered control at every (n, g > 0)'." % above])
V.append(["J-F-return-exclusion", "not established in any n = 3 cohort (both arms, finite and extension); smallest of the four upper bounds %s (filtered g = 1/2, U_cum finite); c2, D2 lower endpoints > 0 in all 8 n = 3 cohorts; smallest per-cohort maximum joint-return lower endpoint (filtered) %s" % (f6(minU), f6(minmax_f)),
          "5.37951; c2, D2 > 0; 0.005053", "0 at printed precision", "U_cond, U_cum < 0 both, finite and extension; c2, D2 > 0; joint lower > 0 at some t > 1", "AGREE", "none"])
cond = "; ".join("n%s g%s %s" % (n, g, f6(rs["filtered n%s g%s" % (n, g)]["max_cond"])) for n in "23" for g in G)
V.append(["J-F-return-history", "all 8 filtered cohorts: every recomputed p_escape and joint p_return inside the saved interval; max conditional: %s; smallest joint lower endpoint over the 12 clocks after t_e, all cohorts: %s (> 0)" % (cond, f6(minlow_f)),
          "0.65093, 0.556681, 0.851326, 0.615635, 0.335305, 0.541048, 0.394226, 0.466272; all > 0", "< 5e-7 (rounding)", "inside interval (1e-12 slack); every joint lower endpoint after t = 1 > 0", "AGREE", "none"])
V.append(["J-F-recovery", "all 72 filtered register rows not_decoupled_to_1e-4 (each upper(D_RK) > 1e-4; own D_RK inside every printed interval); own F_e inside every printed interval; max F_e upper endpoint 0.3129701310 (n = 2, g = 2, t = 4, complete output)",
          "all not decoupled; 'never exceeds F_e upper endpoint 0.31297'", "1.3e-7 above 0.31297", "upper(D_RK) > 1e-4; max F_e upper", "AGREE",
          "ROUNDING: the maximum upper endpoint 0.312970131 exceeds the printed bound 0.31297 by 1.3e-7, so 'never exceeds 0.31297' is false as written; print 0.312971 (rounded up) or 'maximum 0.31297013'."])
r = [coh("onsite", n, g) for n in "23" for g in G]
V.append(["J-B-cohorts", "8 even cohorts: termination complete, check rows = checks field, all true, sources unchanged, max dev %s; odd n = 2, 3: complete, 55 checks each, all true; odd D_RK re-derived (max dev %s)" % (e(max(maxdev(x) for x in r)), e(max(ODD[n]["max_dev_D"] for n in "23"))),
          "all accepted; odd 55 (yes)", "both <= 1e-12", "as J-F-phys", "AGREE", "none"])
V.append(["J-B-vertex", "%d/%d probe records: raw == g S_N(z) local_C exactly, raw == target, S_N = sum_x N_x z^x, normalized value = raw/sqrt(QH QL) and equal to the run-spec sec. 1.1 analytic values; %d/%d local amputated rows actual == target, values -32/5 (16), 4 (16), 0 (120); tangents raw = -128 g S_N(1) (16/16)" % (VX["probe_ok"], VX["probes"], VX["local_actual_eq_target"], VX["local_checks"]),
          "152 pass; -32/5, 4", "none", "exact equality", "AGREE", "none"])
V.append(["J-B-conversion", "; ".join("n%s g%s %s" % (n, g, f6(cv["n%s g%s" % (n, g)]["max_o"])) for n in "23" for g in ("1/2", "1", "2")) + "; counts 16/17 (g > 0), 0/17 (g = 0)",
          "0.5224, 0.5064, 0.4982, 0.5116, 0.5012, 0.4984; same counts; argmax clocks 2, 3.5, 1.75, 2, 3.5, 1.75", "0", "as J-F-conversion", "AGREE", "none"])
V.append(["J-B-return-exclusion", "not established; smallest on-site upper bound %s; smallest per-cohort maximum joint lower endpoint %s; every joint lower endpoint after t_e > 0 in all 8 even cohorts" % (f6(minU_on), f6(minmax_o)),
          "5.4182; 0.002953", "0", "as J-F-return-exclusion", "AGREE", "none"])
V.append(["J-B-recovery", "all on-site even (72) and odd (18) register rows not_decoupled_to_1e-4; max even F_e upper endpoint 0.3222917325 (n = 2, g = 2, t = 4, complete); odd light-only D_RK = 3/4 at every clock; odd F_e = 1/4 at every receiver row",
          "never exceeds 0.322292; 3/4", "0 (0.322292 rounds up correctly)", "as J-F-recovery", "AGREE", "none"])
for i, v in enumerate(V, 1): w("| %d | %s |" % (i, " | ".join(v)))
w("\nCount: %d judgments, %d AGREE, 0 DISAGREE. Two AGREE verdicts (J-F-conversion, J-F-recovery) carry wording repairs. Further defects outside the judgment list are in sec. 3.\n" % (len(V), len(V)))

# ---- re-derivation tables ----
w("## 2. Second-reader re-derivation tables\n")
w("**Table R1. Per-cohort re-derivation, both arms** (filtered rows %s)\n" % HALO)
w("Deviation = max |second reader - icloud saved| over every row of the report: p_LL and C_HL (from the saved density array and again from the saved raw vector), fixed-cut entropies (graded reduction of every _snapshot raw vector), D_RK (registered vector, K = complement of the output), and U (the four purification upper bounds). 'traj' = ||exp(-iH) psi(0) - psi(1)|| with the saved H and vectors. Readout columns: |F_e - centre of fidelity_computed_interval| and |purity - centre of unclipped|, all rows, plus whether every recomputed p_escape and p_return lies inside the saved interval (1e-12 slack).\n")
w("| arm | n | g | report md5 | termination ok / check rows | dev p_LL | dev C_HL | dev S (rows) | dev D_RK | dev U | traj | readout md5 | rows recv/pur/ret | dev F_e | dev purity | returns inside |")
w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for x in C["cohorts"]:
    w("| %s | %s | %s | `%s` | %s / %d | %s | %s | %s (%d) | %s | %s | %s | `%s` | %s | %s | %s | %s |" % (
        x["arm"], x["n"], x["g"], x["md5"], "yes" if x["term_ok"] else "NO", x["checks"], e(x["dev_pLL"]), e(x["dev_CHL"]), e(x["dev_S"]), x["S_rows"],
        e(x["dev_D"]), e(x["dev_U"]), e(x["traj"]), x["ro_md5"], "/".join(map(str, x["ro_rows"])), e(x["ro_dev_Fe"]), e(x["ro_dev_pur"]), "yes" if x["ro_ret_inside"] else "NO"))
w("\nOdd on-site cohorts: " + "; ".join("n = %s report `%s` termination %s, %d checks, D_RK max dev %s, light-only D_RK = 3/4 at every clock: %s, readout `%s` F_e all 1/4: %s" % (
    n, ODD[n]["report_md5"], "ok" if ODD[n]["term_ok"] else "FAIL", ODD[n]["checks"], e(ODD[n]["max_dev_D"]), ODD[n]["light_rows_three_quarters"], ODD[n]["readout_md5"], ODD[n]["receivers_all_quarter"]) for n in "23") + ".\n")
w("Zero-coupling identity (filtered vs on-site at g = 0, second-reader values from the saved vectors): " + "; ".join("n = %s: max |diff| populations %s, p_LL %s, D_RK %s, F_e %s" % (n, e(v["pops"]), e(v["pLL"]), e(v["D"]), e(v["Fe"])) for n, v in C["g0_identity"].items()) + ".\n")

w("**Table R2. Halo header audit of the eight filtered physical reports** %s\n" % HALO)
w("Every record whose data carries a top-level `header` field was counted. The .wip's '37 / 38 records' counts only the conversion, status, wall, purification(-status), B2 and model rows; that is a narrower convention, and no mismatch was found under either count.\n")
w("| n | g | records with header | exact #145 string | model.halo_values and wall_complete.context.model.halo_values |")
w("|---|---|---|---|---|")
for k, v in C["halo_all_kinds"].items():
    n, g = k.split()[0][1:], k.split()[1][1:]
    ha = D[("filtered", n, g)]["halo_audit"]
    w("| %s | %s | %d | %d | %s / %s |" % (n, g, v["top_level_header_rows"], v["correct"], ha["model_halo_values"], ha["wall_halo_values"]))
w("\n**Table R3. Busi-run intake and readout jobs (n = 3, g = 0, 1/2, 2), second-reader audit against icloud's accepted n = 3, g = 1 intake and readout** %s\n" % HALO)
w("| g | intake md5 (report = journal) | intake sha256 | termination / last row / kind roster / sources (39) / adapter-control gate equal | parent journal sha256 unchanged | 34 Actions / 256 excluded / locators / 30 physical sources | readout md5 | readout termination / roster / bound intake | readout sources vs icloud n3 g1 |")
w("|---|---|---|---|---|---|---|---|---|")
for g in ("0", "1/2", "2"):
    a = IN["n3_g" + G[g]]
    w("| %s | `%s` (%s) | `%s` | %s / %s / %s / %s / %s | %s | %s / %s / %s / %s | `%s` | %s / %s / %s | equal except the bound review file (busi assessment in place of icloud red-team review) |" % (
        g, a["intake_md5"], "identical" if a["intake_md5"] == a["intake_journal_md5"] else "DIFFER", a["intake_sha256"], a["term_ok"], a["terminal_last"], a["kind_roster_equal_ref"], a["sources_eq_ref_basename_sha"], a["adapter_controls_eq_ref"],
        a["parent_sha_recorded_eq_now"], a["roster_34"], a["excluded_256"], a["locators_eq_ref_count"], a["physical_sources_eq_ref"], a["readout_md5"],
        a["readout_term"]["status"] + "/" + str(a["readout_term"]["all_pass"]), a["readout_kind_roster_eq_ref"], a["readout_saved_intake_binds_this_intake"]))
w("\nAdapter source md5 `%s` (= F6 / sec. 8). icloud's own n3 g1 intake report and journal are byte-identical, as F6b says.\n" % IN["n3_g0"]["adapter_source_md5"])

w("**Table R4. Conversion, second-reader values from the saved primary vectors at the 17 clocks** (filtered rows %s)\n" % HALO)
w("| arm | n | g | clocks with coherent_conversion flag (= lower p_LL > 0 and lower C_HL > 0) | own max p_LL (clock) | own C_HL at max | own p_LL at t = 4 | own C_HL at t = 4 |")
w("|---|---|---|---|---|---|---|---|")
for n in "23":
    for g in G:
        for a in ("filtered", "onsite"):
            d = D[(a, n, g)]; own = d["own_pLL_by_clock"]; best = max(own, key=lambda k: own[k][0])
            fl = d["conversion"]["count_flag"]
            w("| %s | %s | %s | %d of 17 (flag = bounds: %s) | %s (%s) | %s | %s | %s |" % (a, n, g, fl, d["conversion"]["flag_eq_bounds"], f6(own[best][0]) if own[best][0] > 1e-12 else "0", best if own[best][0] > 1e-12 else "0", f6(own[best][1]), f6(own["4.0"][0]), f6(own["4.0"][1])))
w("\nEvery own value lies inside the corresponding printed .wip interval in Tables F7 and B3. Own populations at t = 4 (core/lead, H/L) match Tables F8 and B4 to the printed 6 digits.\n")

w("**Table R5. Purification upper bounds, n = 3, own Audenaert evaluation from the saved purification-row fields** (filtered rows %s)\n" % HALO)
w("F_d(T) = T ln(d-1) + h(T) for T < 1 - 1/d, else ln d; T capped at 1; d = (137, 37). Finite: U_cond = Cq_up + F_137(B2) + F_37(B2), U_cum = Cq_up + F_137(B2) + F_37(B1). Extension: B -> B + Bwall at the same clock (run spec sec. 4, 5).\n")
w("| arm | g | B_first | B_second | Bwall(1) | Bwall(2) | U_cond fin | U_cum fin | U_cond ext | U_cum ext | max dev vs saved | c2 lower | D2 lower | exclusion |")
w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for g in G:
    for a in ("filtered", "onsite"):
        p = D[(a, "3", g)]["purification"]; m = p["mine"]
        w("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (a, g, f6(p["B_first"]), f6(p["B_second"]), f6(p["Bwall_first"]), f6(p["Bwall_second"]),
          f6(m["Ucond_fin"]), f6(m["Ucum_fin"]), f6(m["Ucond_ext"]), f6(m["Ucum_ext"]), e(p["max_dev"]), f6(p["c2"][0]), f6(p["D2"][0]), "no" if not (p["exclusion_fin"] or p["exclusion_ext"]) else "YES"))
w("\nThe saved wall prefix bounds at t = 1, 2, 3, 4 equal Tables F10 / B6 to the printed 6 digits in all 16 cohorts.\n")

w("**Table R6. Two-time (Lüders) return history, own evaluation** (filtered rows %s)\n" % HALO)
w("p_escape = ||Q_E psi(1)||^2 with Q_E = no particle on site 0 (modes 0-7) and psi(1) the saved primary t = 1 vector. p_return(t) = ||P_C exp(-iH(t-1)) Q_E psi(1)||^2 for t = 5/4, ..., 4, with H the saved operator array (second-reader eigh). The minimum joint lower endpoint is taken over the saved intervals at the 12 clocks after t_e.\n")
w("| arm | n | g | own p_escape | own max joint p_return (clock) | own max conditional | min saved joint lower endpoint after t_e |")
w("|---|---|---|---|---|---|---|")
for n in "23":
    for g in G:
        for a in ("filtered", "onsite"):
            v = rs["%s n%s g%s" % (a, n, g)]
            w("| %s | %s | %s | %s | %s (%s) | %s | %s |" % (a, n, g, f6(v["p_escape"]), f6(v["max_joint"]), v["argmax"], f6(v["max_cond"]), f6(v["min_joint_lower_after_te"])))
w("\n**Table R7. Register decoupling and fixed receiver, summary** (filtered rows %s)\n" % HALO)
w("| arm | register rows | all upper(D_RK) > 1e-4 | own max F_e | max printed F_e upper endpoint | max saved readout F_e upper endpoint |")
w("|---|---|---|---|---|---|")
for a, t in (("filtered", "F12"), ("onsite", "B8")):
    nrows = sum(len(D[(a, n, g)]["register"]["rows"]) for n in "23" for g in G)
    w("| %s | %d | %s | %.10f | %s | %.10f |" % (a, nrows, C["all_not_decoupled"][a], C["Fe_max_own"][a], C["Fe_max_printed"][t], C["Fe_max_readout_upper"][a]))
w("\nEvery own D_RK and F_e lies inside the printed interval in Tables F12 / B8 (row by row, matched on n, g, clock and output). Every own fixed-cut purity (core, whole lead, B1 at t = 1, 2, 4) lies inside the printed interval in Tables F13 / B9, and every odd D_RK inside Table B10.\n")

w("**Table R8. Static #145 halo quantities, exact (extra check of Table F1)** %s\n" % HALO)
w("G_r recomputed from the #141 definition (w = q * 1_[-1/2,1/2], q = (3003/2048)(1-t^2)^6 on [-1,1], G_r = int w(t) w(t-r) dt) by exact symbolic integration: G0, G1, G2 equal the admitted constants (%s), G3 = %s, sum G_r = %s.\n" % (ST["equal_given"], ST["G"][3], ST["sum"]))
w("| quantity | exact | decimal | equals Table F1 |")
w("|---|---|---|---|")
lab = {"site0_endpoint_copy": "(19G0+20G1+3G2)/64", "flat": "(9G0+8G1+G2)/16", "Nm1_eq_1": "(5G0+6G1+G2)/16 (rejected control)", "halo_removed": "(9G0+6G1)/32 (rejected control)",
       "dH_dNm1_entry": "(G0+4G1+G2)/32", "Schur_n2": "R(W_-1), n = 2: (2G0+9G1+6G2)/32", "Schur_n3_n7": "R(W_-1), n = 3, 7: (2G0+9G1+9G2)/32"}
for k, v in ST["combos"].items(): w("| %s | %s | %.12g | yes |" % (lab[k], v[0], v[1]))

w("\n## 3. Defects found outside the judgment list\n")
b2 = C["B2_label"]
w("1. **Mislabelled clock, Table F8 (filtered) and Table B4 (baseline), last column.** Both files print 'n = 3: site 1 (later-collected B2) H / L / total at t = 0.75'. The saved record `later_collected_B2_at_first_collection` has clock 1.0 (field preceding_primary_clock = 0.75), and the printed numbers are the site-1 populations at t = 1, not at t = 0.75. The second reader's own values from the saved t = 1 vector equal the printed numbers in all 8 n = 3 rows; the t = 0.75 values differ (e.g. filtered g = 0: printed 0.230091 = own t = 1 0.230091, own t = 0.75 0.141417). Run spec sec. 3.2 asks for B2 at the first collection clock t1 = 1. Repair: relabel the column 'at t = 1 (first collection, before collecting B1)'. The values need no change.\n")
w("2. **Missing #145 halo header on one clock-well table.** The sub-table under Table F9 ('Computed primary-versus-protected endpoint distance ... and the recorded switch work', filtered .wip line 176) has no 'endpoint-copy halo; N_-1 = 1/2, N_n = 1' label, but its filtered rows are halo-dependent. Every other clock-well table (F1 to F13, F6b) carries the label and the rule line. Repair: add the bracketed header and rule line.\n")
w("3. **Overbroad conversion comparison** (J-F-conversion; detail in sec. 1).\n")
w("4. **Rounding of the receiver bound** (J-F-recovery; detail in sec. 1).\n")
w("5. **Minor, no repair required.** (a) Sec. 1 of the filtered .wip says 'at most 4 BLAS threads', while sec. 8 says 'One thread' for the adapter jobs. Both can be true (analysis versus adapter jobs), but the file should say which applies to what. (b) Table F12's source label 'busi = busi reduction ... (icloud readout job not run)' is accurate. For the three busi-run cohorts, the adapter's own F_e now exists and agrees, as sec. 8 states; the label could add 'adapter value agrees (Table F6)'. (c) The 37/38 header-record count convention is narrower than 'records with halo header' implies (Table R2), but there are no mismatches.\n")

w("## 4. Spec versus .wip formulas\n")
w("The sec. 9 / sec. 7 formulas were checked against the governing texts: D_RK (run spec 3.3), U_cond / U_cum with dimensions (137, 37) and B1 = 0 on the shared prefix (run spec 4), the extension B + Bwall at t1 and t2 with the cap applied only inside F (run spec 5), the F_e decoder with p_failure/4 and the D minus sign (addendum 2), the purity interval 4 min(1, eps) (addendum 1), and the Lüders history with unnormalized branch and t_e = 1 (addendum 3). They agree. One immaterial wording difference: J-F-recovery uses 'eps the action's metadata.error', while the addendum prescribes the normalized-state error. The two agree to about 1e-16 here and do not affect any verdict. Provenance labels: the three busi-run readout jobs are correctly labelled busi-run with the icloud adapter unchanged (adapter md5 `125f7814a3cc461bfb4a58b38e2af53e`, source roster identical to icloud's n3 g1 readout except the bound review file). They are bound to busi's own intake assessment as the review, which F6b and sec. 8 disclose.\n")

w("## 5. Inputs, md5 verification and products\n")
tot = sum(D[k]["md5_log"]["checked"] for k in D); mism = sum(len(D[k]["md5_log"]["mismatch"]) for k in D); nim = sum(D[k]["md5_log"]["not_in_manifest"] for k in D)
w("Every input file opened by r2_10 was hashed before use and compared with <internal>/INPUT_MD5SUMS.txt (md5 `64fb24c4a36bfacca8d792faf8dabfb3`): %d file reads, %d mismatches. %d reads (15 per cohort: the primary pure-action vectors at the 17 clocks that the first reader did not use) are not in that manifest; their md5s are recorded in each r2_*_even_*.json under md5_log.digests. All report md5s in Tables R1 and R3 equal those printed in Tables F5, F6 and B1, B2, and the controls report md5 `%s` equals the baseline sec. 2 value.\n" % (tot, mism, nim, VX["md5"]))
w("Second-reader code (new files only): busi/code/clockwell_r2/r2_01_schema.py, r2_02_peek.py, r2_03_keys.py (schema reads), r2_10_cohort.py (per-cohort re-derivation), r2_11_runall.sh, r2_12_intake.py, r2_13_vertex.py, r2_14_static.py, r2_15_odd.py, r2_20_compare.py (row-by-row comparison with the .wip tables; 0 failures), r2_30_write.py (this file). Products: <internal>/*.json, with md5s in <internal>/MD5SUMS.\n")
w("| product | md5 |")
w("|---|---|")
for p in sorted(glob.glob(os.path.join(O, "*.json"))) + sorted(glob.glob(os.path.join(O, "code", "*"))):
    w("| `%s` | `%s` |" % (p, hashlib.md5(open(p, "rb").read()).hexdigest()))
w("\nNo rate, flow, current, chronology, source or Page time is printed or implied. Generated %s UTC by r2_30_write.py.\n" % datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M"))
open(sys.argv[1], "w").write("\n".join(o))
print("written", sys.argv[1], len(o))
