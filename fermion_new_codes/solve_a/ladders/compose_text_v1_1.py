"""compose_text_v1_1.py: the fixed wording of MASS_ladder_v1.1.md and SCALING_ladder_v1.1.md (imported by ladders_compose_v1_1.py). Every
number is read from the version 1.1 producer's json through the Doc object, or from the W4 (a) comparator's stored outputs where named;
nothing here computes a physical quantity. The corrections cite deliveries/lead_gaps/ReviewA_reading_LADDERS_v1.md by line."""
from ladders_compose_v1_1 import f, e, table

READING = "deliveries/lead_gaps/ReviewA_reading_LADDERS_v1.md"
MASS = ["M3b_s4", "P64_s4", "M3a_s4", "M3c_s4_nk", "M3d_s4"]
MASS_CTRL = ["M3b_s8", "P64_s2", "P64_s8", "P64_s4_tau4", "P64_s4_tau2", "P64_s4_c19", "M3a_s2", "M3a_s4_tau4", "M3c_s4_plain"]
MASS_PAIRS = {("M3b_s4", "M3b_s8"), ("P64_s4", "P64_s2"), ("P64_s4", "P64_s8"), ("P64_s4", "P64_s4_tau4"), ("P64_s4", "P64_s4_tau2"),
              ("P64_s4", "P64_s4_c19"), ("M3a_s4", "M3a_s2"), ("M3a_s4", "M3a_s4_tau4"), ("M3c_s4_nk", "M3c_s4_plain")}
SIZE = ["S4_16_s4", "RS32_s4", "P64_s4", "S4_128_s4"]
SIZE_CTRL = ["S4_16_s2", "S4_16_s4_tau1", "RS32_s2", "RS32_s4_tau1", "P64_s2", "P64_s8", "P64_s4_tau4", "P64_s4_tau2",
             "S4_128_s4_nk_v36m", "S4_128_s4_nk_v36q"]
PREP = ["S4_R2_m0p2", "S4_R2_m0p1"]
SIZE_PAIRS = {("S4_16_s4", "S4_16_s2"), ("S4_16_s4", "S4_16_s4_tau1"), ("RS32_s4", "RS32_s2"), ("RS32_s4", "RS32_s4_tau1"),
              ("P64_s4", "P64_s2"), ("P64_s4", "P64_s8"), ("P64_s4", "P64_s4_tau4"), ("P64_s4", "P64_s4_tau2"),
              ("S4_128_s4", "S4_128_s4_nk_v36m"), ("S4_128_s4", "S4_128_s4_nk_v36q"), ("S4_128_s4", "S4_128_s8"), ("S4_128_s4", "S4_128_s16"),
              ("S4_128_s8", "S4_128_s16"), ("P64_s4", "S4_R2_m0p2"), ("M3a_s4", "S4_R2_m0p1")}
LABELS = {
    "M3b_s4": "mass member (0.4, 64)", "P64_s4": "mass and size member (0.2, 64), the production's history (a reproduction of the v19 record)",
    "M3a_s4": "mass member (0.1, 64), near-maximal", "M3c_s4_nk": "mass member (0.1, 96)", "M3d_s4": "mass member (0.05, 326) (kappa_sea 12, plain, v36e)",
    "S4_16_s4": "size member (0.2, 16), near-maximal and outside the hydrodynamic hierarchy on both ends",
    "RS32_s4": "size member (0.2, 32), the 1-field production c31_rs32_A_R2_tau2_ell4_prep4_avg", "S4_128_s4": "size member (0.2, 128)",
    "M3b_s8": "sigma/a 8 control of M3b_s4 (its formation grazing)", "M3a_s2": "sigma/a 2 control of M3a_s4", "M3a_s4_tau4": "tau 4 control of M3a_s4 (to 16)",
    "M3c_s4_plain": "plain-iteration comparator of M3c_s4_nk", "P64_s2": "sigma/a 2 point at 64: the composite history (Newton across its seams)",
    "P64_s8": "sigma/a 8 point at 64", "P64_s4_tau4": "tau 4 control of P64_s4 (engine v19)", "P64_s4_tau2": "tau 2 control of P64_s4 (engine v25, to 21.5)",
    "P64_s4_c19": "the production record (engine v19); P64_s4 reproduces it", "S4_16_s2": "sigma/a 2 control of S4_16_s4", "S4_16_s4_tau1": "tau 1 control of S4_16_s4 (to 16)",
    "RS32_s2": "sigma/a 2 control of RS32_s4 (Newton, engine v32)", "RS32_s4_tau1": "tau 1 control of RS32_s4 (engine v25)",
    "S4_128_s4_nk_v36m": "Newton run of S4_128_s4 (v36m), stopped", "S4_128_s4_nk_v36q": "Newton run of S4_128_s4 (v36q), stopped",
    "S4_128_s8": "width point (128, sigma/a 8), RUNNING (version 1's snapshot)", "S4_128_s16": "width point (128, sigma/a 16), at the gate's edge",
    "S4_R2_m0p2": "preparation member, release at 2 r_s, (0.2, 64)", "S4_R2_m0p1": "preparation member, release at 2 r_s, (0.1, 64)",
}

DEFS = (
    "Definitions (version 1.1; the reading's R1, R2, R6 and reviewer B's R1, R4). A record counts when its converged entry is True (a Newton "
    "record without the entry: its kept residual) and its residual (engine_ladder_compare_v5.rec_residual: Newton records the kept residual, "
    "plain records the last iterate's or the stored unconverged residual) is at most 1e-3; for a Newton record the confirming residual "
    "(confirm_residual, the npz column conf) is tested as well, since confirm_v32's own convergence test does not test it: a record whose "
    "confirming residual exceeds the engine's stop rule (stop_fields 1e-6 on every member here) is relabelled UNCONVERGED. A capped record "
    "(converged False), a record above 1e-3 or a record failing the confirming test does not count and is named. Two extents print for every "
    "member: the 30.3 prefix, ending before the first record above 1e-3, and the certified prefix, ending before the first non-counting "
    "record of any kind, since a record kept unconverged propagates into every later record until a replay from the last certified record. "
    "The last converged record is the last record of the certified prefix. A formation is certified when its record lies inside the "
    "certified prefix, and only such formations enter a fit; a formation inside the 30.3 prefix but after a non-counting record prints as "
    "'certified under the 30.3 rule only' and enters no fit except as a labelled row. Formation is the first saved record holding a "
    "trapping zero (trapping_zeros non-empty), the birth lying in the interval (T_f minus the record step, T_f], whose width prints with "
    "each member; R_t at formation is the outer zero of the dense N + v on that first saved record and not the birth radius; the 2 zeros "
    "of every formation record print with no recorded tangency and a positive slope at R_t (the inner zero's simplicity is not tested). "
    "R_t/(2 g E) is the record's own stored ratio at its charge E (g 1/4): the root fraction. The root gate is sigma/R_t with sigma = ell. "
    "E/E0 is against the head's E0 of the ell 4 preparation, so it is not 1 at T = 0 for the re-averaged controls. N_B (m a)^3/1.087 is the "
    "nominal count ratio (to the displayed approximation the rest-energy ratio N_B m/E_max), not the record's charge ratio; the charge ratio "
    "E0/E_max uses the stored E0 and E_max = c_E/(g^(3/2) m^2) with c_E 0.135858. Labels: (n.c.) the record does not count; (unc.) the "
    "record lies after the member's certified prefix.")


def first_record_growth(d, names):
    vals = []
    for nm in names:
        fo = d.M[nm]["formation"]
        if not (fo and fo["certified"]):
            continue
        r0 = d.M[nm]["matched_from_formation"].get("0"); r5 = d.M[nm]["matched_from_formation"].get("0.5")
        if r0 and r5 and r0.get("R_t_over_2gE") is not None and r5.get("R_t_over_2gE") is not None and r5.get("certified"):
            vals.append((nm, r0["R_t_over_2gE"], r5["R_t_over_2gE"]))
    return vals


def growth_text(d, names):
    v = first_record_growth(d, names)
    if not v:
        return ""
    gs = [b - a for _, a, b in v]
    return ("Over the 0.5 r_s after the first saved record the root fraction grows by %s to %s of 2GE on the certified formed members (%s), so a "
            "first-record radius carries a sampling scatter of several hundredths of 2GE depending on where in the step the birth fell; no fit "
            "of R_t at formation and no intercept of T_f in any variable is a law before the crossing of N - |v| is located between records "
            "on every member (busi B7)." % (f(min(gs), "%.3f"), f(max(gs), "%.3f"), "; ".join("%s %s to %s" % (nm, f(a), f(b)) for nm, a, b in v)))


def spread_of(d, a_, b_):
    for s in d.D["spreads"]:
        if s["member"] == a_ and s["control"] == b_:
            return s
    return {}


def spreads_rows(d, nm):
    out = []
    for s in d.D["spreads"]:
        if s["member"] == nm and s.get("dT_f") is not None and s["kind"].startswith("control_"):
            out.append("%s %s: dT_f %+.4f, d(R_t/r_s) %+.4f (%s)" % (s["kind"].replace("control_", "").split(" ")[0], s["control"], s["dT_f"],
                                                                   s["dR_t_over_rs"], s.get("classes")))
    return "; ".join(out) if out else "no stored control holding a formation"


def fit_vs_controls(d, key):
    F = d.D["fits"].get(key)
    if not isinstance(F, dict):
        return ""
    h = ["member", "residual T_f", "residual R_t/r_s", "residual R_t/(2gE)", "the member's stored control spreads at formation (control's class)"]
    rows = [[nm, "%+.4f" % F["T_f"]["residual"][i], "%+.4f" % F["R_t_over_rs"]["residual"][i], "%+.4f" % F["R_t_over_2gE"]["residual"][i],
             spreads_rows(d, nm)] for i, nm in enumerate(F["members"])]
    return table(h, rows)


def values_at_others(d, key):
    F = d.D["fits"].get(key)
    if not isinstance(F, dict) or not F.get("value_at_other_members"):
        return "- none"
    lines = []
    for nm, v in F["value_at_other_members"].items():
        rec = v["record"]
        if "none_to" in rec:
            rtxt = "the member holds no trapping zero in its records to %s" % f(rec["none_to"])
        else:
            rtxt = "the member's record: T_f %s, R_t/r_s %s, R_t/(2gE) %s (%s)" % (f(rec["T_f"]), f(rec["R_t_over_rs"]), f(rec["R_t_over_2gE"]), rec["class"])
        lines.append("- %s: the fit's value T_f %s, R_t/r_s %s, R_t/(2gE) %s; %s" % (nm, f(v.get("T_f")), f(v.get("R_t_over_rs")), f(v.get("R_t_over_2gE")), rtxt))
    return "\n".join(lines)


def status_lines(d, names):
    lines = []
    for nm in names:
        M = d.M[nm]; fo = M["formation"]; v = M["convergence"]; c = M["counts"]
        if fo:
            s = "first saved zero %s at R_t %s a (%s r_s, root fraction %s, gate %s), %s" % (
                f(fo["T"]), f(fo["R_t"], "%.3f"), f(fo["R_t_over_rs"]), f(fo["R_t_over_2gE"]), f(fo["gate_sigma_over_Rt"], "%.3f"), fo["label"])
        else:
            s = "no trapping zero in its records to %s" % f(v["last"])
        lc = f(v["last_converged"]) if v["last_converged"] is not None else "none (its first record does not count)"
        extra = ""
        if v["first_noncounting"] is not None:
            extra = ", first non-counting record %s (%s)" % (f(v["first_noncounting"]), v["first_noncounting_gate"])
        lines.append("- %s %s (%s; %s sites, tau %g, N_B %s): %s; last converged record %s%s." % (
            nm, d.short(nm), LABELS.get(nm, ""), M["meta"]["nsites"], M["meta"]["tau"], f(c["N_B"], "%.0f"), s, lc, extra))
    return "\n".join(lines)


def sources_block(d, names, SRC):
    extra = ["catalog (chain columns): %s (md5 %s), written by %s" % (p, m5, SRC["catalog_script"]) for p, m5 in SRC["catalogs"]]
    extra += ["producer: %s (md5 %s), version 1's producer with the reading's corrections, rerun on the same stored files (the running "
              "S4_128_s8 from version 1's snapshot); helper engine_ladder_compare_v5.py (md5 %s)" % (SRC["producer"], SRC["producer_md5"], SRC["lc_md5"])]
    extra += ["composer: %s and %s; the readout json %s (md5 %s); the arrays %s (md5 %s), mirrored on the box at %s with its md5 manifest" % (
        SRC["composer"], SRC["composer_text"], SRC["readout_json"], SRC["readout_json_md5"], SRC["readout_npz"], SRC["readout_npz_md5"], SRC["box_dir"])]
    extra += ["engine of the catalog's rerun preparations: %s (md5 %s)" % (SRC["engine"], SRC["engine_md5"])]
    extra += ["the reading applied: %s (reviewer A Part A, reviewer B Part B, the lane's cross-check)" % READING]
    extra += ["W4 (a) comparator numbers quoted in 5.1: <internal>/s4_density_continuum_rest_lapse-on_adaptive.json "
              "(through deliveries/solve_a/W5_W4_formation_attribution_v1.md section 4)"]
    return d.sources(names, extra)


def changes_common(d):
    M = d.M
    p2 = M["P64_s2"]["convergence"]; r2 = M["RS32_s2"]["convergence"]; nc = M["M3c_s4_nk"]["convergence"]
    return [
        "Certified prefix with propagation (reading lines 61-65, reviewer A R1; 168-227, reviewer B R1): a capped record now ends the certified prefix; "
        "the 30.3 prefix prints beside it. Extents that move: M3c_s4_nk 22.0833 to %s; M3d_s4 0.3926 to %s; S4_128_s4 3.8125 to %s; "
        "P64_s4_tau4 40.0 to %s; S4_R2_m0p2 40.0 to %s; RS32_s2 and P64_s2 to none (see the next item); the definitions in the producer's "
        "docstring and json now state the rule applied." % (f(nc["last_converged"]), f(M["M3d_s4"]["convergence"]["last_converged"]),
                                                            f(M["S4_128_s4"]["convergence"]["last_converged"]), f(M["P64_s4_tau4"]["convergence"]["last_converged"]),
                                                            f(M["S4_R2_m0p2"]["convergence"]["last_converged"])),
        "Confirming-residual audit (lines 67-71, reviewer A R2; 229-294, reviewer B R2), from the stored records against stop_fields 1e-6: M3c_s4_nk fails at %s "
        "only (both already non-counting; its formation record's confirming residual %s, the maximum up to formation 2.0e-07, passes); P64_s2 fails at "
        "%d records from T = 0 (%s; the T = 0 record's confirming residual %s against a kept %s), so the sigma/a 2 composite has no certified prefix "
        "and its formation 12.000 prints as 'certified under the 30.3 rule only'; RS32_s2 fails at %d records from T = 0 (%s), besides its capped "
        "T = 0 record; the 2 S4_128 Newton runs fail from 0.5625 (certified to 0.5000). The convergence table gains the audit column and the stored "
        "True count." % (", ".join("%.4f" % t for t in nc["conf_fail_clocks"]), e(M["M3c_s4_nk"]["formation"]["confirming"]), p2["conf_fail_count"],
                         ", ".join("%.4f" % t for t in p2["conf_fail_clocks"]), e(next((x["confirming"] for x in p2["excluded"] if x["T"] == 0.0), None)),
                         e(next((x["residual"] for x in p2["excluded"] if x["T"] == 0.0), None)), r2["conf_fail_count"],
                         ", ".join("%.4f" % t for t in r2["conf_fail_clocks"])),
        "(unc.) labels (lines 73-77, reviewer A R3; 298-387, reviewer B R3): every matched-clock cell, formation-offset cell, episode closing and episode depth "
        "after a member's certified prefix carries (unc.); the episode depth prints over the certified records beside the raw one; control spreads "
        "are taken over clocks counting and certified in both histories, the raw spread beside them labelled.",
        "First-record clocks and radii (lines 91-95, reviewer A R6; 391-461, reviewer B R4): the bracket (previous record, its min(N - |v|), the record step) "
        "and the zero of the stored min(N - |v|) series in the bracket (linear and quadratic, a diagnostic, not an event location) print with "
        "every formation; c0 cells say 'not an intercept'; R_t fit rows say 'at the sampling scatter of the first record'.",
        "Fits relabelled 2-coefficient diagnostics, no exponent or law (lines 79-83 and 85-89, reviewer A R4 and R5; 465-519 and 523-612, reviewer B R5 and R6); "
        "the exact 3-point power forms print as dependences; residual rms prints per residual degree of freedom beside the per-member value "
        "(line 130, reviewer A M2).",
        "Onsets relabelled chain-column upper bounds (lines 103-107, reviewer A R8; 673-726, reviewer B R8); section 3 relabelled the dust-energy centroid "
        "(lines 109-113, reviewer A R9).",
        "E/E_max printed as the nominal count ratio with the charge ratio E0/E_max, E0, 2 g E0 and E0/(N_B m) beside it (lines 616-669, reviewer B R7; "
        "line 138, reviewer A M10; lines 930-947, reviewer B M2: 2 g E0 is 59.069 a for the (0.2, 64) members, not 59.05); the isotropic pressure-correction "
        "parameter (p_F/m)^2/5 prints as a release comparator (lines 949-967, reviewer B M3); the own coldness f_kin/0.3 is labelled the composer's "
        "cold-gas conversion (line 137, reviewer A M9).",
        "Minors applied: 'max over the records up to the last converged record' (line 129, M1); M3b_s8's grazing formation labelled (line 131, M3); "
        "E/E0 at T = 0 for re-averaged controls (line 132, M4); M3d_s4 described (line 136, M8); 'no tangency, positive slope at R_t' (line 139, M11); "
        "records dropped without E printed per member (line 140, M12; 0 on every member); P64_s4 labelled a reproduction of the v19 record "
        "(line 141, M13); the excluded lists are complete with their gates (reviewer B M4, lines 985-987).",
        "Not delivery changes, routed to the paper lane by the reading: reviewer A R10, R11, M5 to M7, M17 and reviewer B R9, R10, M5 (lines 115-125, 133-135, "
        "145, 730-838, 999-1011); the data each needs is in sections 2, 4 and 6 here.",
    ]


def mass_doc(d, D, SRC, PREP_RULE):
    names = MASS + MASS_CTRL
    F = D["fits"]
    o = ["# MASS_ladder_v1.1: the mass ladder at sigma/a 4, tau 8 (m a 0.4, 0.2 and 0.1 at r_s/a 64; 0.1 at r_s/a 96; 0.05 at r_s/a 326), "
         "composed from the stored members, with the corrections of the correctness reading", ""]
    o.append("Version 1.1 of MASS_ladder_v1 (which stays as issued). Composed by ladders_readout_v1_1.py (the producer, box CPU, stored run files "
             "only) and ladders_compose_v1_1.py (this text, from the producer's json). The builder composes; no sentence here reads the physics.")
    o.append("")
    o.append("## Changes from v1")
    o.append("")
    for c in changes_common(d):
        o.append("- " + c)
    s = spread_of(d, "M3b_s4", "M3b_s8")
    o.append("- 5.1 rewritten per the reading (lines 79-83, reviewer A R4): the (0.4, 64) member's formation lies %s r_s and %s in radius from its "
             "sigma/a 8 control, outside the acceptance criterion (0.5 r_s, 10 percent), and enters the diagnostic labelled so. Where the reading's "
             "replacement wording names the comparator's cold-limit clock 13.22 r_s, the stored W4 (a) comparator on the production's own T = 0 "
             "profile (sigma/a 4, adaptive) folds at 10.8157 r_s with its first root record at 10.8750: the 13.22 r_s is not that comparator's "
             "clock; both print, the reading's labelled as its value." % (f(s.get("dT_f"), "%.3f"), f(100 * (s.get("dR_t_rel") or 0), "%.1f%%")))
    o.append("")
    o.append("## 0 The members' end states")
    o.append("")
    o.append(status_lines(d, MASS))
    o.append("")
    o.append("Controls and comparators (not ladder members):")
    o.append("")
    o.append(status_lines(d, MASS_CTRL))
    o.append("")
    o.append(DEFS)
    o.append("")
    o.append("## 1 The member table")
    o.append("")
    o.append(PREP_RULE)
    o.append("")
    o.append(d.member_table(names, LABELS))
    o.append("")
    o.append("The p_F/m of the isotropic release form and its (p_F/m)^2/5 are release comparators, not measured pressure corrections of the "
             "anisotropic well fillings; the members' own f_kin (head) gives (p_F/m)^2 = f_kin/0.3 by the composer's cold-gas conversion, 0.35 to 0.53 "
             "on every member and non-monotone in m a. P64_s4 (v36d from T = 0) agrees with the v19 production record P64_s4_c19 to 1.2e-8 in E/E0 "
             "on all 321 clocks (6.3): a reproduction of the same arithmetic, not an independent route.")
    o.append("")
    o.append("## 2 The first saved trapping record per member, with the bracket, the root fraction and the gate")
    o.append("")
    o.append(d.formation_table(names))
    o.append("")
    o.append(growth_text(d, names))
    o.append("")
    fo = d.M["M3b_s8"]["formation"]; ep = d.M["M3b_s8"]["episodes"][0]
    o.append("M3b_s8's formation is grazing: min(N - |v|) %s at the record, %s over its %d-record episode, zeros %s a apart; the spread it sets "
             "beside M3b_s4 is that of a marginal episode. The mass direction compares a root at %s of the charge (0.4, 64), gate %s, with roots at %s "
             "(0.2, 64) and %s (0.1, 96): an interior core root against outer roots." % (
                 f(fo["minNv"]), f(ep["min_minNv"]), ep["records"], f(fo["zeros"][1] - fo["zeros"][0], "%.2f"),
                 f(d.M["M3b_s4"]["formation"]["R_t_over_2gE"]), f(d.M["M3b_s4"]["formation"]["gate_sigma_over_Rt"], "%.3f"),
                 f(d.M["P64_s4"]["formation"]["R_t_over_2gE"]), f(d.M["M3c_s4_nk"]["formation"]["R_t_over_2gE"])))
    o.append("")
    o.append("The trapped episodes per member (labels as in section 0):")
    o.append("")
    o.append(d.episodes_text(names))
    o.append("")
    o.append("## 3 The dust-energy centroid R_c against the onsets (the centroid's minimum over the certified records; not the core's compression "
             "radius, which the requirement reads on the half-mass radius with the dust's bare energy per particle there and which is not composed here)")
    o.append("")
    o.append(d.compression_table(names))
    o.append("")
    o.append("M3d_s4's centroid minimum is at T = 0: its centroid moves outward over its 16 certified records, unlike every other released member; "
             "not attributed here.")
    o.append("")
    o.append("## 4 The matched-clock readouts (every 2 r_s; (n.c.) the record does not count, (unc.) it lies after the certified prefix)")
    o.append("")
    o.append(d.matched_tables(MASS, "minNv", "### 4a min(N - |v|)"))
    o.append("")
    o.append(d.matched_tables(MASS, "EE0", "### 4b E/E0", "%.5f"))
    o.append("")
    o.append(d.matched_tables(MASS, "R_c_over_rs", "### 4c R_c/r_s (the dust-energy centroid)"))
    o.append("")
    o.append(d.matched_tables(MASS, "R_t_over_rs", "### 4d R_t/r_s (\"none\": no zero in the record)"))
    o.append("")
    o.append("### 4e At the first saved trapping record and after it (T_f + 0, 0.5, 1, 2, 4 r_s)")
    o.append("")
    o.append(d.from_formation_table(MASS + ["M3b_s8", "P64_s2", "P64_s8", "P64_s4_tau4", "P64_s4_tau2", "M3c_s4_plain"]))
    o.append("")
    o.append("## 5 The 2-coefficient diagnostics (no exponent, intercept or law is printed)")
    o.append("")
    o.append("Members with a certified formation: %s. Without one: %s." % (", ".join(F["mass_formed_certified"]), ", ".join(
        "%s (%s)" % (nm, ("none to %s" % f(d.M[nm]["convergence"]["last"])) if not d.M[nm]["formation"] else d.M[nm]["formation"]["class"])
        for nm in MASS if nm not in F["mass_formed_certified"])))
    o.append("")
    o.append("### 5.1 Y = c0 + c1 x on the 3 formed members, x = (p_F/m)^2 of the isotropic release form")
    o.append("")
    o.append("Under the assumption that Y depends on m a and r_s/a only through x = (p_F/m)^2 of the isotropic release form (x scales as (m a)^(-8/3) "
             "at fixed r_s/a and as (a/r_s)^(4/3) at fixed m a), the 3 formed members lie on one variable; both coefficients float, so this is a "
             "2-coefficient line with 1 residual degree of freedom and not a test of the asymptotic form, which fixes the cold-limit value at the "
             "comparator's clock and the slope at 1/5 of it (the reading gives the cold-limit value as 13.22 r_s for release at 4 r_s with the "
             "production's charge; the stored W4 (a) comparator on the production's own profile folds at 10.8157 r_s, first root record 10.8750): "
             "the fitted c0 and c1/c0 below are not those values. The residuals are of the size of the stored averaging-scale spreads. The "
             "(0.4, 64) member's formation lies %s r_s and %s in radius from its sigma/a 8 control (itself grazing), outside the acceptance "
             "criterion, and is printed in this fit labelled so and in no law." % (f(s.get("dT_f"), "%.3f"), f(100 * (s.get("dR_t_rel") or 0), "%.1f%%")))
    o.append("")
    o.append(d.fit_text("mass_diag_iso"))
    o.append("")
    o.append(fit_vs_controls(d, "mass_diag_iso"))
    o.append("")
    o.append("The diagnostic's value at the members it does not contain (each is below or beyond that member's records):")
    o.append("")
    o.append(values_at_others(d, "mass_diag_iso"))
    o.append("")
    o.append(d.power3_text("mass_power3_iso", "x"))
    o.append("")
    o.append("### 5.2 The same line with the members' own coldness x = f_kin/0.3 (the composer's cold-gas conversion; a diagnostic)")
    o.append("")
    o.append(d.fit_text("mass_diag_own"))
    o.append("")
    o.append(values_at_others(d, "mass_diag_own"))
    o.append("")
    o.append("### 5.3 The dependence at r_s/a 64 (2 formed members)")
    o.append("")
    rows = []
    for nm in ["M3b_s4", "P64_s4", "M3a_s4"]:
        fo = d.M[nm]["formation"]; c = d.M[nm]["counts"]
        rows.append([nm, f(d.M[nm]["meta"]["m"], "%g"), f(c["p_F_over_m_iso"], "%.3f"), f(c["p_F_over_m_own"], "%.3f"),
                     f(fo["T"]) if fo else "none to %s" % f(d.M[nm]["convergence"]["last"]), f(fo["bracket"]["width"], "%.3f") if fo else "-",
                     f(fo["R_t"], "%.3f") if fo else "-", f(fo["R_t_over_rs"]) if fo else "-", f(fo["R_t_over_2gE"]) if fo else "-",
                     f(fo["gate_sigma_over_Rt"], "%.3f") if fo else "-", fo["class"] if fo else "-"])
    o.append(table(["member", "m a", "p_F/m isotropic", "p_F/m own (cold-gas conversion)", "first saved zero T_f/r_s", "record step", "R_t/a",
                    "R_t/r_s", "root fraction", "gate", "class"], rows))
    o.append("")
    o.append("## 6 The convergence tables")
    o.append("")
    o.append("### 6.1 Per member")
    o.append("")
    o.append(d.convergence_table(names))
    o.append("")
    o.append("### 6.2 The records that do not count, with their gates (complete)")
    o.append("")
    o.append(d.excluded_text(names))
    o.append("")
    o.append("### 6.3 The stored controls (control minus member)")
    o.append("")
    o.append(d.spreads_table(MASS_PAIRS))
    o.append("")
    o.append("The (0.4, 64) and (0.1, 96) members have no stored step control; the (0.1, 96) member has no stored averaging-scale control (its plain "
             "comparator agrees with it over the common certified clocks); the (0.1, 64) member's controls hold no trapping zero; the (0.2, 64) step "
             "controls are certified to 13.375 (tau 4) and 12.750 (tau 2), and the sigma/a 2 composite has no certified record (6.1).")
    o.append("")
    o.append("## 7 The catalogs behind the chain-column onsets")
    o.append("")
    o.append(d.catalog_text(names))
    o.append("")
    o.append("## 8 Not composed, truncated or open")
    o.append("")
    for line in SRC.get("mass_open", []):
        o.append("- " + line)
    o.append("")
    o.append("## 9 Figure data and figures")
    o.append("")
    for line in SRC.get("mass_figs", []):
        o.append("- " + line)
    o.append("")
    o.append("## 10 SOURCES")
    o.append("")
    o.append(sources_block(d, names, SRC))
    o.append("")
    return "\n".join(o)


def scaling_doc(d, D, SRC, PREP_RULE):
    names = SIZE + SIZE_CTRL + ["S4_128_s8", "S4_128_s16"] + PREP + ["M3a_s4"]
    F = D["fits"]
    P3 = F.get("size_power3_with_uncertified") or {}
    rt2 = (P3.get("R_t_over_2gE") or [{}])[0]
    wd = F.get("width_per_doubling") or []
    rk = F.get("design_rank") or {}
    o = ["# SCALING_ladder_v1.1: the size ladder at m a 0.2, sigma/a 4 (r_s/a 16, 32, 64, 128), the width cross at r_s/a 64 and 128, and the "
         "2 r_s releases, composed from the stored members, with the corrections of the correctness reading", ""]
    o.append("Version 1.1 of SCALING_ladder_v1 (which stays as issued). Composed by ladders_readout_v1_1.py and ladders_compose_v1_1.py from "
             "stored files only. The builder composes; no sentence here reads the physics. The size member at r_s/a 32 is the 1-field production "
             "c31_rs32_A_R2_tau2_ell4_prep4_avg (tau 2, 1024 sites).")
    o.append("")
    o.append("## Changes from v1")
    o.append("")
    for c in changes_common(d):
        o.append("- " + c)
    o.append("- 5.2 retitled per the reading (lines 85-89, reviewer A R5). Its replacement wording says R_t/(2gE) admits no finite-intercept power through "
             "the 3 points; the stored values contradict it: the exact form through (32, 64, 128 uncertified) has p %s, Y_0 %s, A %s (reviewer B's "
             "-1.04, line 542, agrees); the T_f form has p %s, Y_0 %s as the reading states." % (
                 f(rt2.get("p"), "%.3f"), f(rt2.get("Y0"), "%.4f"), f(rt2.get("A"), "%.3g"),
                 f(((P3.get("T_f") or [{}])[0]).get("p"), "%.3f"), f(((P3.get("T_f") or [{}])[0]).get("Y0"), "%.3f")))
    w64 = [x for x in wd if x["r_s/a"] == 64]; w128 = [x for x in wd if x["r_s/a"] == 128]
    o.append("- 5.3 per the reading (lines 97-101, reviewer A R7; 523-612, reviewer B R6): with P64_s2 uncertified by the confirming-residual audit the "
             "certified width set is 4 points for 4 coefficients (no residual test), v1's 5-point fit prints labelled; the design rank of "
             "[1, z, x^2, h, zh, j, j^2, zj] on the certified formed rows is %s of 8 on %d rows (residual degrees of freedom %s), as reviewer B states "
             "(line 590 to 598). The reading's verdict (line 150) gives 'a width slope of 0.375 r_s per doubling of sigma/a at both sizes'; the "
             "certified records give %s r_s per doubling at 64 (sigma/a 4 to 8) and %s at 128 (8 to 16); the 0.375 at 64 is the average over "
             "sigma/a 2 to 8, which uses the uncertified sigma/a 2 point (%s r_s from 2 to 4)." % (
                 (rk.get("certified") or {}).get("rank"), len((rk.get("certified") or {}).get("rows", [])), (rk.get("certified") or {}).get("residual_dof"),
                 f(next((x["dT_f"] for x in w64 if x["from"] == "P64_s4"), None), "%.3f"), f(next((x["dT_f"] for x in w128 if x["from"] == "S4_128_s8"), None), "%.3f"),
                 f(next((x["dT_f"] for x in w64 if x["from"] == "P64_s2"), None), "%.3f")))
    sp = spread_of(d, "P64_s4", "P64_s2"); s816 = spread_of(d, "S4_128_s8", "S4_128_s16")
    o.append("- The certified factor-2 pair at 128 added to 6.3 (line 142, reviewer A M14): %s r_s and %s in radius at formation, max relative radius "
             "difference %s at %s over %s common certified zero records. The reading's 's2 against s4 radius difference reaches 11.3 a, 29 percent': "
             "the records give %s a and %s at most with the sigma/a 4 radius as denominator (%s at 16.0, reviewer B's 22.2 percent, line 790), over %s "
             "common counting zero records, all uncertified on the sigma/a 2 side." % (
                 f(s816.get("dT_f"), "%.3f"), f(100 * (s816.get("dR_t_rel") or 0), "%.1f%%"), f(100 * (s816.get("max_rel_dRt") or 0), "%.1f%%"),
                 f(s816.get("max_rel_dRt_at")), s816.get("Rt_common_count"), f(sp.get("raw_max_abs_dRt"), "%.3f"), f(100 * (sp.get("raw_max_rel_dRt") or 0), "%.1f%%"),
                 f(100 * (sp.get("raw_rel_dRt_at_16") or 0), "%.1f%%"), sp.get("raw_Rt_common_count")))
    o.append("- Section 1's sentence on the step (line 101, reviewer A R7) replaced; S4_128_s4's uncertified record described (line 143, reviewer A M15).")
    o.append("")
    o.append("## 0 The members' end states")
    o.append("")
    o.append(status_lines(d, SIZE))
    o.append("")
    o.append("Width points, preparation members, controls and comparators:")
    o.append("")
    o.append(status_lines(d, ["S4_128_s8", "S4_128_s16"] + PREP + SIZE_CTRL))
    o.append("")
    o.append(DEFS)
    o.append("")
    o.append("## 1 The member table")
    o.append("")
    o.append(PREP_RULE)
    o.append("")
    o.append(d.member_table(names, LABELS))
    o.append("")
    o.append("The step in units of r_s differs along the size ladder (tau/r_s 0.125 at 16 and 64, 0.0625 at 32 and 128), and sigma/r_s halves at "
             "each doubling at fixed sigma/a 4 (0.25, 0.125, 0.0625, 0.03125): the size direction at fixed sigma/a carries both, which the width "
             "cross does not separate (its model has no step term) and the step controls bound at r_s/a 64 only (and at 32 between tau 2 and "
             "tau 1); the (32, 4, 8) member is the step pair at a second size.")
    o.append("")
    o.append("## 2 The first saved trapping record per member, with the bracket, the root fraction and the gate")
    o.append("")
    o.append(d.formation_table(names))
    o.append("")
    o.append(growth_text(d, names))
    o.append("")
    c128 = d.M["S4_128_s4"]["convergence"]
    ab = [x for x in c128["excluded"] if "above 1e-3" in x["gate"]]
    o.append("S4_128_s4's uncertified formation record (%s) sits amid a failing solve: %d of its records lie above 1e-3, the residual reaching %s at "
             "%s; it prints beside the width cross labelled so and enters no law." % (
                 f(d.M["S4_128_s4"]["formation"]["T"]), len(ab), e(c128["residual_max"]), f(c128["residual_max_at"])))
    o.append("")
    o.append("The trapped episodes per member:")
    o.append("")
    o.append(d.episodes_text(names))
    o.append("")
    o.append("## 3 The dust-energy centroid R_c against the onsets (not the core's compression radius; not composed here)")
    o.append("")
    o.append(d.compression_table(names))
    o.append("")
    o.append("## 4 The matched-clock readouts (every 2 r_s; (n.c.) the record does not count, (unc.) it lies after the certified prefix)")
    o.append("")
    cols = SIZE + ["S4_128_s8", "S4_128_s16", "P64_s2", "P64_s8"]
    o.append(d.matched_tables(cols, "minNv", "### 4a min(N - |v|)"))
    o.append("")
    o.append(d.matched_tables(cols, "EE0", "### 4b E/E0", "%.5f"))
    o.append("")
    o.append(d.matched_tables(cols, "R_c_over_rs", "### 4c R_c/r_s (the dust-energy centroid)"))
    o.append("")
    o.append(d.matched_tables(cols, "R_t_over_rs", "### 4d R_t/r_s (\"none\": no zero in the record)"))
    o.append("")
    o.append("### 4e At the first saved trapping record and after it")
    o.append("")
    o.append(d.from_formation_table(SIZE + ["S4_128_s8", "S4_128_s16", "P64_s2", "P64_s8", "RS32_s2", "RS32_s4_tau1", "P64_s4_tau4", "P64_s4_tau2"] + PREP))
    o.append("")
    o.append("## 5 The dependences and 2-coefficient diagnostics (no exponent, intercept or law is printed)")
    o.append("")
    o.append("### 5.1 The size ladder at sigma/a 4: the dependence")
    o.append("")
    o.append("The fixed-width size ladder supplies %d certified first-record formation points (%s), insufficient to test even a fixed-exponent "
             "intercept-and-amplitude model; the free exponent on 4 members is not evaluable." % (len(F.get("size_formed_certified") or []),
                                                                                                 ", ".join(F.get("size_formed_certified") or [])))
    o.append("")
    rows = []
    for nm in SIZE:
        M = d.M[nm]; fo = M["formation"]; c = M["counts"]
        rows.append([nm, f(M["meta"]["rs"], "%g"), "%g (%s)" % (M["meta"]["tau"], f(c["tau_over_rs"], "%.5f")), f(c["sigma_over_rs"], "%.5f"),
                     (f(fo["T"]) + " (" + fo["class"] + ")") if fo else "none to %s" % f(M["convergence"]["last"]),
                     f(fo["bracket"]["width"], "%.4f") if fo else "-", f(fo["R_t"], "%.3f") if fo else "-", f(fo["R_t_over_rs"]) if fo else "-",
                     f(fo["R_t_over_2gE"]) if fo else "-", f(fo["gate_sigma_over_Rt"], "%.3f") if fo else "-",
                     f(M["convergence"]["last_converged"]) if M["convergence"]["last_converged"] is not None else "none"])
    o.append(table(["member", "r_s/a", "tau/a (tau/r_s)", "sigma/r_s", "first saved zero T_f/r_s (class)", "record step", "R_t/a", "R_t/r_s",
                    "root fraction", "gate", "last converged record"], rows))
    o.append("")
    o.append("### 5.2 A 2-coefficient fit in (a/r_s)^(1/3) (the composer's choice of variable; the paper's chi^(1/3) is the exponent of the shed "
             "energy share f_E, not of the formation clock, whose asymptotic form at fixed m a is the comparator's clock with a pressure correction "
             "of order chi^(4/3), nor of the root fraction, whose asymptotic form is an approach to 1): Y = c0 + c1 (a/r_s)^(1/3)")
    o.append("")
    o.append(d.fit_text("size_diag_third"))
    o.append("")
    o.append("With the uncertified formation record of the (0.2, 128) sigma/a 4 member inside (labelled; not a law):")
    o.append("")
    o.append(d.fit_text("size_diag_third_with_uncertified"))
    o.append("")
    o.append(values_at_others(d, "size_diag_third_with_uncertified"))
    o.append("")
    o.append(d.power3_text("size_power3_with_uncertified", "(a/r_s)"))
    o.append("")
    o.append("### 5.3 The width cross: Y = c0 + c1 z + c2 h + c3 z h, z = ln(a/r_s), h = ln(sigma/a) (a pair of straight lines in ln(sigma/a); "
             "c0 is not a continuum-size intercept)")
    o.append("")
    o.append("The 6 points: r_s/a 64 at sigma/a 2, 4, 8 and r_s/a 128 at sigma/a 4, 8, 16. Certified: (64, 4), (64, 8), (128, 8), (128, 16); the "
             "sigma/a 2 composite fails the confirming-residual audit from T = 0 and the (128, 4) formation record lies after its first record above "
             "1e-3. The z term carries one halving of tau/r_s between the sizes (0.125 at 64, 0.0625 at 128), whose share by the 64 step controls is "
             "-0.125 r_s and -0.019 in R_t/r_s per halving, measured at 1 size only. The (128, 8) point is read from version 1's snapshot of the "
             "records mirrored from its running card.")
    o.append("")
    o.append("The dependence per doubling of sigma/a (the records' numbers):")
    o.append("")
    o.append(table(["r_s/a", "from", "to", "dT_f (r_s)", "d(R_t/r_s)", "d(root fraction)", "both certified"],
                   [[x["r_s/a"], x["from"], x["to"], "%+.4f" % x["dT_f"], "%+.4f" % x["dR_t_over_rs"], "%+.4f" % x["dR_t_over_2gE"], "yes" if x["both_certified"] else "no"] for x in wd]))
    o.append("")
    o.append("On the 4 certified points (4 coefficients, no residual test):")
    o.append("")
    o.append(d.fit_text("width_cross_certified"))
    o.append("")
    o.append(values_at_others(d, "width_cross_certified"))
    o.append("")
    o.append("Version 1's 5-point set with the sigma/a 2 composite (labelled; 1 residual degree of freedom, all of it in the 3 points at 64):")
    o.append("")
    o.append(d.fit_text("width_cross_with_s2"))
    o.append("")
    o.append(values_at_others(d, "width_cross_with_s2"))
    o.append("")
    o.append("Every formation record of the 6 points (labelled; not a law):")
    o.append("")
    o.append(d.fit_text("width_cross_with_uncertified"))
    o.append("")
    o.append("The design rank (the requirements' edit 5): the model matrix [1, z, x^2, h, zh, j, j^2, zj] with x = 64 a/r_s, z = x^(4/3), "
             "h = (16 sigma/r_s)^2, j = 8 tau/r_s (the rank does not depend on the columns' normalization):")
    o.append("")
    rows = []
    for k, r in rk.items():
        rows.append([k, "; ".join("%s (%g, %g, %g)" % (nm, a_, b_, c_) for nm, (a_, b_, c_) in zip(r["rows"], r["rows_rs_sigma_tau"])), len(r["rows"]),
                     r["rank"], r["residual_dof"], ", ".join("%.4g" % v for v in r["singular_values"])])
    o.append(table(["row set", "rows (r_s/a, sigma/a, tau/a)", "rows", "rank (of 8)", "rows minus rank", "singular values"], rows))
    o.append("")
    o.append("The review's rank-7 null (z - 1)(2/3 - 7h/24 + j) holds on its own 8-row set; the certified rows here have full rank 8 on 8 rows, a "
             "saturated design with no residual test (full rank alone is not a calibration).")
    o.append("")
    o.append("### 5.4 The release radius (the 2 r_s preparation members beside the 4 r_s members)")
    o.append("")
    rows = []
    for a_, b_ in (("P64_s4", "S4_R2_m0p2"), ("M3a_s4", "S4_R2_m0p1")):
        for nm in (a_, b_):
            M = d.M[nm]; fo = M["formation"]
            rows.append([nm, f(M["meta"]["m"], "%g"), f(M["meta"]["R0_over_rs"], "%g"),
                         (f(fo["T"]) + " (" + fo["class"] + ")") if fo else "none to %s" % f(M["convergence"]["last"]),
                         f(fo["R_t"], "%.3f") if fo else "-", f(fo["R_t_over_rs"]) if fo else "-", f(fo["R_t_over_2gE"]) if fo else "-",
                         f(fo["gate_sigma_over_Rt"], "%.3f") if fo else "-", f(M["convergence"]["last_converged"])])
    o.append(table(["member", "m a", "R_0/r_s", "first saved zero T_f/r_s (class)", "R_t/a", "R_t/r_s", "root fraction", "gate", "last converged record"], rows))
    o.append("")
    o.append("## 6 The convergence tables")
    o.append("")
    o.append("### 6.1 Per member")
    o.append("")
    o.append(d.convergence_table(names))
    o.append("")
    o.append("### 6.2 The records that do not count, with their gates (complete)")
    o.append("")
    o.append(d.excluded_text(names))
    o.append("")
    o.append("### 6.3 The stored controls and the width and release pairs (second minus first)")
    o.append("")
    o.append(d.spreads_table(SIZE_PAIRS))
    o.append("")
    o.append("## 7 The catalogs behind the chain-column onsets, and the r_s/a 32 member's preparation")
    o.append("")
    o.append(d.catalog_text(names))
    o.append("")
    for line in SRC.get("rs32_prep", []):
        o.append(line)
    o.append("")
    o.append("## 8 Not composed, truncated or open")
    o.append("")
    for line in SRC.get("scaling_open", []):
        o.append("- " + line)
    o.append("")
    o.append("## 9 Figure data and figures")
    o.append("")
    for line in SRC.get("scaling_figs", []):
        o.append("- " + line)
    o.append("")
    o.append("## 10 SOURCES")
    o.append("")
    o.append(sources_block(d, names, SRC))
    o.append("")
    return "\n".join(o)
