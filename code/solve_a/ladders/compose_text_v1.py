"""compose_text_v1.py: the fixed wording of MASS_ladder_v1.md and SCALING_ladder_v1.md (imported by ladders_compose_v1.py). Every number
is read from the producer's json through the Doc object; nothing here computes a physical quantity."""
from ladders_compose_v1 import f, e, table

MASS = ["M3b_s4", "P64_s4", "M3a_s4", "M3c_s4_nk", "M3d_s4"]
MASS_CTRL = ["M3b_s8", "P64_s2", "P64_s8", "P64_s4_tau4", "P64_s4_tau2", "P64_s4_c19", "M3a_s2", "M3a_s4_tau4", "M3c_s4_plain"]
MASS_PAIRS = {("M3b_s4", "M3b_s8"), ("P64_s4", "P64_s2"), ("P64_s4", "P64_s8"), ("P64_s4", "P64_s4_tau4"), ("P64_s4", "P64_s4_tau2"),
              ("P64_s4", "P64_s4_c19"), ("M3a_s4", "M3a_s2"), ("M3a_s4", "M3a_s4_tau4"), ("M3c_s4_nk", "M3c_s4_plain")}
SIZE = ["S4_16_s4", "RS32_s4", "P64_s4", "S4_128_s4"]
SIZE_CTRL = ["S4_16_s2", "S4_16_s4_tau1", "RS32_s2", "RS32_s4_tau1", "P64_s2", "P64_s8", "P64_s4_tau4", "P64_s4_tau2",
             "S4_128_s4_nk_v36m", "S4_128_s4_nk_v36q"]
WIDTH = ["P64_s2", "P64_s4", "P64_s8", "S4_128_s4", "S4_128_s8", "S4_128_s16"]
PREP = ["S4_R2_m0p2", "S4_R2_m0p1"]
SIZE_PAIRS = {("S4_16_s4", "S4_16_s2"), ("S4_16_s4", "S4_16_s4_tau1"), ("RS32_s4", "RS32_s2"), ("RS32_s4", "RS32_s4_tau1"),
              ("P64_s4", "P64_s2"), ("P64_s4", "P64_s8"), ("P64_s4", "P64_s4_tau4"), ("P64_s4", "P64_s4_tau2"),
              ("S4_128_s4", "S4_128_s4_nk_v36m"), ("S4_128_s4", "S4_128_s4_nk_v36q"), ("S4_128_s4", "S4_128_s8"), ("S4_128_s4", "S4_128_s16"),
              ("P64_s4", "S4_R2_m0p2"), ("M3a_s4", "S4_R2_m0p1")}

LABELS = {
    "M3b_s4": "mass member (0.4, 64)", "P64_s4": "mass and size member (0.2, 64), the production's history", "M3a_s4": "mass member (0.1, 64), near-maximal",
    "M3c_s4_nk": "mass member (0.1, 96)", "M3d_s4": "mass member (0.05, 326)", "S4_16_s4": "size member (0.2, 16), near-maximal and outside the hydrodynamic hierarchy on both ends",
    "RS32_s4": "size member (0.2, 32), the 1-field production c31_rs32_A_R2_tau2_ell4_prep4_avg", "S4_128_s4": "size member (0.2, 128)",
    "M3b_s8": "sigma/a 8 control of M3b_s4", "M3a_s2": "sigma/a 2 control of M3a_s4", "M3a_s4_tau4": "tau 4 control of M3a_s4 (to 16)",
    "M3c_s4_plain": "plain-iteration comparator of M3c_s4_nk", "P64_s2": "sigma/a 2 point at 64: the composite history, labelled across its seams",
    "P64_s8": "sigma/a 8 point at 64", "P64_s4_tau4": "tau 4 control of P64_s4 (engine v19)", "P64_s4_tau2": "tau 2 control of P64_s4 (engine v25, to 21.5)",
    "P64_s4_c19": "the production record (engine v19), comparator of P64_s4", "S4_16_s2": "sigma/a 2 control of S4_16_s4", "S4_16_s4_tau1": "tau 1 control of S4_16_s4 (to 16)",
    "RS32_s2": "sigma/a 2 control of RS32_s4 (Newton)", "RS32_s4_tau1": "tau 1 control of RS32_s4 (engine v25)",
    "S4_128_s4_nk_v36m": "Newton run of S4_128_s4 (v36m), stopped", "S4_128_s4_nk_v36q": "Newton run of S4_128_s4 (v36q), stopped",
    "S4_128_s8": "width point (128, sigma/a 8), RUNNING", "S4_128_s16": "width point (128, sigma/a 16), at the gate's edge",
    "S4_R2_m0p2": "preparation member, release at 2 r_s, (0.2, 64)", "S4_R2_m0p1": "preparation member, release at 2 r_s, (0.1, 64)",
}

DEFS = ("Definitions (the house rules of COLLAPSE_ladder_v2.1 and engine_ladder_compare_v5, applied unchanged): a record holds a trapping zero "
        "when its trapping_zeros list is non-empty; R_t is the outermost zero of the dense N + v; formation is the first record holding a zero "
        "(at every formation below the record holds 2 simple zeros, no tangency, and the slope |d(N + v)/dr| at R_t is printed); the first "
        "episode closes at the first later record holding none; R_t/(2 g E) is the record's own stored ratio at the record's charge E (g 1/4); "
        "the root gate is sigma/R_t with sigma = ell, the Gaussian's per-axis standard deviation in lattice units; the residual is "
        "engine_ladder_compare_v5.rec_residual (Newton records: the kept residual; plain records: the last iterate's, or the stored unconverged "
        "residual). A record counts when its converged entry is True (a Newton record without the entry: its kept residual) and its residual "
        "is at most 1e-3; a capped record (converged False) or a record above 1e-3 does not count and is named; the reading rule 30.3 "
        "uncertifies a member from its first record above 1e-3; the last converged record is the last counting record before that; a "
        "formation is certified when its record counts and lies before that first record above 1e-3, and only certified formations enter a "
        "fit. E/E_max = N_B (m a)^3/1.087; p_F/m = kappa_max/((R_0/a)(m a)) with kappa_max = (9 pi N_B/32)^(1/3) (the isotropic release form, "
        "0.24 (2 g m a)^(-1/3) (a/r_s)^(2/3)/(m a) at R_0 = 4 r_s); the isotropic onset 1.222 N_B^(1/3) a = 4 a kappa_max/pi; the exclusion "
        "floor (3 N_B/(8 pi))^(1/3) a; the own onset 4 a n_rad,max with n_rad,max the largest radial-mode count of the member's own catalog "
        "at T = 0. \"n.c.\" marks a record that does not count.")


def spreads_rows(d, nm):
    out = []
    for s in d.D["spreads"]:
        if s["member"] == nm and s.get("dT_f") is not None and s["kind"].startswith("control_"):
            out.append("%s %s: dT_f %s, d(R_t/r_s) %s%s" % (s["kind"].replace("control_", ""), s["control"], "%+.4f" % s["dT_f"], "%+.4f" % s["dR_t_over_rs"],
                                                             "" if s.get("both_certified") else " (a formation uncertified)"))
    return "; ".join(out) if out else "no stored control holding a formation"


def fit_vs_controls(d, key):
    F = d.D["fits"].get(key)
    if not isinstance(F, dict):
        return ""
    h = ["member", "residual T_f", "residual R_t/r_s", "residual R_t/(2gE)", "the member's stored control spreads at formation"]
    rows = []
    for i, nm in enumerate(F["members"]):
        rows.append([nm, "%+.4f" % F["T_f"]["residual"][i], "%+.4f" % F["R_t_over_rs"]["residual"][i], "%+.4f" % F["R_t_over_2gE"]["residual"][i],
                     spreads_rows(d, nm)])
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
            rtxt = "the member's record: T_f %s, R_t/r_s %s, R_t/(2gE) %s (%s)" % (f(rec["T_f"]), f(rec["R_t_over_rs"]), f(rec["R_t_over_2gE"]),
                                                                                  "certified" if rec["certified"] else "uncertified")
        lines.append("- %s: the fit's value T_f %s, R_t/r_s %s, R_t/(2gE) %s; %s" % (nm, f(v.get("T_f")), f(v.get("R_t_over_rs")), f(v.get("R_t_over_2gE")), rtxt))
    return "\n".join(lines)


def status_lines(d, names):
    lines = []
    for nm in names:
        M = d.M[nm]; fo = M["formation"]; v = M["convergence"]; c = M["counts"]
        if fo:
            s = "formation %s at R_t %s a (%s r_s, %s of 2GE), %s" % (f(fo["T"]), f(fo["R_t"], "%.3f"), f(fo["R_t_over_rs"]), f(fo["R_t_over_2gE"]),
                                                                    "certified" if fo["certified"] else "UNCERTIFIED (%s)" % fo["label"])
        else:
            s = "no trapping zero in its records to %s" % f(v["last"])
        lines.append("- %s %s (%s; %s sites, tau %g, N_B %s, E/E_max %s): %s; last converged record %s%s." % (
            nm, d.short(nm), LABELS.get(nm, ""), M["meta"]["nsites"], M["meta"]["tau"], f(c["N_B"], "%.0f"), f(c["E_over_Emax"], "%.2f"), s,
            f(v["last_converged"]), (", first record above 1e-3 at %s" % f(v["first_above_1e-3"])) if v["first_above_1e-3"] is not None else ""))
    return "\n".join(lines)


def sources_block(d, names, SRC):
    extra = ["catalog (own onsets): %s (md5 %s), written by %s" % (p, m5, SRC["catalog_script"]) for p, m5 in SRC["catalogs"]]
    extra += ["producer: %s (md5 %s); its helper engine_ladder_compare_v5.py (md5 %s) for rec_residual, confirm_residual, pockets, slope_at_Rt" % (
        SRC["producer"], SRC["producer_md5"], SRC["lc_md5"])]
    extra += ["composer: %s and %s; the readout json %s (md5 %s); the arrays %s (md5 %s), mirrored on the box at %s with its md5 manifest" % (
        SRC["composer"], SRC["composer_text"], SRC["readout_json"], SRC["readout_json_md5"], SRC["readout_npz"], SRC["readout_npz_md5"], SRC["box_dir"])]
    extra += ["engine of the rerun preparations: %s (md5 %s)" % (SRC["engine"], SRC["engine_md5"])]
    extra += ["spec read: final4/requirements_new.txt (the ladder paragraph), final4/REVIEWA_REORIENT_5.md (the solve lane's on-return item), "
              "final4/MARKS_TRIAGE/busi.md item 8, final4/MARKS_TRIAGE/paper.md item 5 (06_scaling:359), busi/results/w11/ladder_manifest_v1.md"]
    return d.sources(names, extra)


def mass_doc(d, D, SRC, PREP_RULE):
    F = D["fits"]
    names = MASS + MASS_CTRL
    fi = F.get("mass_fixed_iso", {})
    out = []
    out.append("# MASS_ladder_v1: the mass ladder at sigma/a 4, tau 8 (m a 0.4, 0.2 and 0.1 at r_s/a 64; 0.1 at r_s/a 96; 0.05 at r_s/a 326), "
               "composed from the stored members with the formation clock and radius, the matched-clock readouts, the fixed-exponent test, "
               "the convergence tables and the figure data")
    out.append("")
    out.append("Composed by ladders_readout_v1.py (the producer, box CPU, stored run files only) and ladders_compose_v1.py (this text, from "
               "the producer's json). The builder composes; no sentence here reads the physics. Members, end states and digits are the "
               "records'; every input path and md5 is in SOURCES.")
    out.append("")
    out.append("## 0 The members' end states")
    out.append("")
    out.append(status_lines(d, MASS))
    out.append("")
    out.append("Controls and comparators (not ladder members):")
    out.append("")
    out.append(status_lines(d, MASS_CTRL))
    out.append("")
    out.append(DEFS)
    out.append("")
    out.append("## 1 The member table")
    out.append("")
    out.append(PREP_RULE)
    out.append("")
    out.append(d.member_table(names, LABELS))
    out.append("")
    out.append("## 2 The formation clock and radius per member, with the gate values")
    out.append("")
    out.append(d.formation_table(names))
    out.append("")
    out.append("The trapped episodes per member (first and last record holding a zero, R_t at both, the depth min(N - |v|), the closing):")
    out.append("")
    out.append(d.episodes_text(names))
    out.append("")
    out.append("## 3 The compression radius against the onsets")
    out.append("")
    out.append(d.compression_table(names))
    out.append("")
    out.append("## 4 The matched-clock readouts (every 2 r_s; the ladder members)")
    out.append("")
    out.append(d.matched_tables(MASS, "minNv", "### 4a min(N - |v|)"))
    out.append("")
    out.append(d.matched_tables(MASS, "EE0", "### 4b E/E0", "%.5f"))
    out.append("")
    out.append(d.matched_tables(MASS, "R_c_over_rs", "### 4c R_c/r_s (the dust-energy centroid)"))
    out.append("")
    out.append(d.matched_tables(MASS, "R_t_over_rs", "### 4d R_t/r_s (\"none\": no zero in the record)"))
    out.append("")
    out.append("### 4e At the formation clock and after it (T_f + 0, 0.5, 1, 2, 4 r_s)")
    out.append("")
    out.append(d.from_formation_table(MASS + ["M3b_s8", "P64_s2", "P64_s8", "P64_s4_tau4", "P64_s4_tau2", "M3c_s4_plain"]))
    out.append("")
    out.append("## 5 The fits")
    out.append("")
    nform = [nm for nm in MASS if d.M[nm]["formation"] and d.M[nm]["formation"]["certified"]]
    nnot = [nm for nm in MASS if nm not in nform]
    out.append("Members with a certified formation: %s; without one: %s. The requirement's rule: an exponent fixed by the asymptotic form "
               "tested on 3 members (1 residual degree of freedom), a free exponent only on 4." % (", ".join(nform), ", ".join(
                   "%s (%s)" % (nm, "none to %s" % f(d.M[nm]["convergence"]["last"]) if not d.M[nm]["formation"] else "uncertified") for nm in nnot)))
    out.append("")
    out.append("### 5.1 The fixed-exponent test on the 3 formed members: Y = c0 + c1 x, x = (p_F/m)^2 of the isotropic release form")
    out.append("")
    out.append("The exponent is fixed by the release form: x scales as (m a)^(-8/3) at fixed r_s/a and as (a/r_s)^(4/3) at fixed m a, so the "
               "3 formed members, 2 at r_s/a 64 and 1 at 96, lie on one variable; c1 is the coefficient of (p_F/m)^2 in the units of Y. "
               "This choice of variable is the composer's reading of the requirement's "
               "\"exponent fixed by the asymptotic form\" (the coldness correction it prints with each member) and is stated for the reading.")
    out.append("")
    out.append(d.fit_text("mass_fixed_iso"))
    out.append("")
    out.append("The residuals beside each member's stored control spreads (step, averaging scale, solver) at formation:")
    out.append("")
    out.append(fit_vs_controls(d, "mass_fixed_iso"))
    out.append("")
    out.append("The fit's value at the members it does not contain:")
    out.append("")
    out.append(values_at_others(d, "mass_fixed_iso"))
    out.append("")
    out.append("### 5.2 The same form with the members' own coldness, x = f_kin/0.3 from each stored preparation")
    out.append("")
    out.append(d.fit_text("mass_fixed_own"))
    out.append("")
    out.append(values_at_others(d, "mass_fixed_own"))
    out.append("")
    out.append("### 5.3 The dependence at r_s/a 64 (2 formed members; no fit is a law)")
    out.append("")
    rows = []
    for nm in ["M3b_s4", "P64_s4", "M3a_s4"]:
        fo = d.M[nm]["formation"]; c = d.M[nm]["counts"]
        rows.append([nm, f(d.M[nm]["meta"]["m"], "%g"), f(c["p_F_over_m_iso"], "%.3f"), f(c["p_F_over_m_own"], "%.3f"),
                     f(fo["T"]) if fo else "none to %s" % f(d.M[nm]["convergence"]["last"]), f(fo["R_t"], "%.3f") if fo else "-",
                     f(fo["R_t_over_rs"]) if fo else "-", f(fo["R_t_over_2gE"]) if fo else "-", f(fo["gate_sigma_over_Rt"], "%.3f") if fo else "-"])
    out.append(table(["member", "m a", "p_F/m isotropic", "p_F/m own (f_kin)", "T_f/r_s", "R_t/a", "R_t/r_s", "R_t/(2gE)", "gate"], rows))
    out.append("")
    out.append("## 6 The convergence tables")
    out.append("")
    out.append("### 6.1 Per member")
    out.append("")
    out.append(d.convergence_table(names))
    out.append("")
    out.append("### 6.2 The excluded records (capped or kept above 1e-3), named")
    out.append("")
    out.append(d.excluded_text(names))
    out.append("")
    out.append("### 6.3 The stored controls: step, averaging scale and solver (control minus member)")
    out.append("")
    out.append(d.spreads_table(MASS_PAIRS))
    out.append("")
    out.append("The (0.4, 64) and (0.1, 96) members have no stored step control; the (0.1, 96) member has no stored averaging-scale control; "
               "the (0.1, 64) member's controls hold no trapping zero, so they carry no formation spread. The (0.2, 64) step controls are "
               "the production's tau 4 (engine v19) and tau 2 (engine v25) histories on the same preparation (E0 equal to the member's), "
               "the member itself on v36d: their spreads carry the engine versions with the step.")
    out.append("")
    out.append("## 7 The catalogs behind the own onsets")
    out.append("")
    out.append(d.catalog_text(names))
    out.append("")
    out.append("## 8 Not composed, truncated or open")
    out.append("")
    for line in SRC.get("mass_open", []):
        out.append("- " + line)
    out.append("")
    out.append("## 9 Figure data and figures")
    out.append("")
    for line in SRC.get("mass_figs", []):
        out.append("- " + line)
    out.append("")
    out.append("## 10 SOURCES")
    out.append("")
    out.append(sources_block(d, names, SRC))
    out.append("")
    return "\n".join(out)


def scaling_doc(d, D, SRC, PREP_RULE):
    names = SIZE + SIZE_CTRL + ["S4_128_s8", "S4_128_s16"] + PREP + ["M3a_s4"]
    out = []
    out.append("# SCALING_ladder_v1: the size ladder at m a 0.2, sigma/a 4 (r_s/a 16, 32, 64, 128), the width cross at r_s/a 64 "
               "(sigma/a 2, 4, 8) and 128 (sigma/a 4, 8, 16), and the 2 r_s releases, composed from the stored members with the formation "
               "clock and radius, the matched-clock readouts, the fits, the convergence tables and the figure data")
    out.append("")
    out.append("Composed by ladders_readout_v1.py (the producer, box CPU, stored run files only) and ladders_compose_v1.py (this text, from "
               "the producer's json). The builder composes; no sentence here reads the physics. The size member at r_s/a 32 is the 1-field "
               "production c31_rs32_A_R2_tau2_ell4_prep4_avg (tau 2, 1024 sites); its preparation line is in section 7.")
    out.append("")
    out.append("## 0 The members' end states")
    out.append("")
    out.append(status_lines(d, SIZE))
    out.append("")
    out.append("Width points, preparation members, controls and comparators:")
    out.append("")
    out.append(status_lines(d, ["S4_128_s8", "S4_128_s16"] + PREP + SIZE_CTRL))
    out.append("")
    out.append(DEFS)
    out.append("")
    out.append("## 1 The member table")
    out.append("")
    out.append(PREP_RULE)
    out.append("")
    out.append(d.member_table(names, LABELS))
    out.append("")
    out.append("The step in units of r_s differs along the size ladder (tau/r_s 0.125 at 16 and 64, 0.0625 at 32 and 128), and sigma/r_s "
               "halves at each doubling at fixed sigma/a 4 (0.25, 0.125, 0.0625, 0.03125): the size direction at fixed sigma/a carries "
               "both, which the width cross and the step controls separate as far as the stored members reach.")
    out.append("")
    out.append("## 2 The formation clock and radius per member, with the gate values")
    out.append("")
    out.append(d.formation_table(names))
    out.append("")
    out.append("The trapped episodes per member:")
    out.append("")
    out.append(d.episodes_text(names))
    out.append("")
    out.append("## 3 The compression radius against the onsets")
    out.append("")
    out.append(d.compression_table(names))
    out.append("")
    out.append("## 4 The matched-clock readouts (every 2 r_s)")
    out.append("")
    cols = SIZE + ["S4_128_s8", "S4_128_s16", "P64_s2", "P64_s8"]
    out.append(d.matched_tables(cols, "minNv", "### 4a min(N - |v|)"))
    out.append("")
    out.append(d.matched_tables(cols, "EE0", "### 4b E/E0", "%.5f"))
    out.append("")
    out.append(d.matched_tables(cols, "R_c_over_rs", "### 4c R_c/r_s"))
    out.append("")
    out.append(d.matched_tables(cols, "R_t_over_rs", "### 4d R_t/r_s (\"none\": no zero in the record)"))
    out.append("")
    out.append("### 4e At the formation clock and after it")
    out.append("")
    out.append(d.from_formation_table(SIZE + ["S4_128_s8", "S4_128_s16", "P64_s2", "P64_s8", "RS32_s2", "RS32_s4_tau1", "P64_s4_tau4", "P64_s4_tau2"] + PREP))
    out.append("")
    out.append("## 5 The fits")
    out.append("")
    sf = D["fits"].get("size_formed_certified") or []
    out.append("### 5.1 The free exponent on the 4 size members: Y = Y_0 + A (a/r_s)^p")
    out.append("")
    out.append("Not evaluable as a law: the size members with a certified formation are %s (%d of 4). The dependence per member:" % (
        ", ".join(sf) if sf else "none", len(sf)))
    out.append("")
    rows = []
    for nm in SIZE:
        M = d.M[nm]; fo = M["formation"]; c = M["counts"]
        rows.append([nm, f(M["meta"]["rs"], "%g"), "%g (%s)" % (M["meta"]["tau"], f(c["tau_over_rs"], "%.5f")), f(c["sigma_over_rs"], "%.5f"),
                     (f(fo["T"]) + ("" if fo["certified"] else " (uncertified)")) if fo else "none to %s" % f(M["convergence"]["last"]),
                     f(fo["R_t"], "%.3f") if fo else "-", f(fo["R_t_over_rs"]) if fo else "-", f(fo["R_t_over_2gE"]) if fo else "-",
                     f(fo["gate_sigma_over_Rt"], "%.3f") if fo else "-", f(M["convergence"]["last_converged"])])
    out.append(table(["member", "r_s/a", "tau/a (tau/r_s)", "sigma/r_s", "T_f/r_s", "R_t/a", "R_t/r_s", "R_t/(2gE)", "gate", "last converged record"], rows))
    out.append("")
    out.append("### 5.2 The fixed exponent 1/3 in a/r_s (the paper's chi^(1/3)): Y = c0 + c1 (a/r_s)^(1/3)")
    out.append("")
    out.append(d.fit_text("size_fixed_third"))
    out.append("")
    out.append("With the uncertified formation record of the (0.2, 128) sigma/a 4 member beside (labelled; not a law):")
    out.append("")
    out.append(d.fit_text("size_fixed_third_with_uncertified"))
    out.append("")
    out.append(values_at_others(d, "size_fixed_third_with_uncertified"))
    out.append("")
    out.append("### 5.3 The width cross: Y = c0 + c1 z + c2 h + c3 z h, z = ln(a/r_s), h = ln(sigma/a)")
    out.append("")
    out.append("The 6 points: r_s/a 64 at sigma/a 2, 4, 8 and r_s/a 128 at sigma/a 4, 8, 16 (sigma/r_s 1/32, 1/16, 1/8 at both sizes). The "
               "certified points enter the law; the (128, 4) point's formation record is uncertified and is printed beside the fit's value "
               "there. The sigma/a 2 point at 64 is the composite history (Newton, across its seams), labelled; the (128, 8) point is read "
               "from the records mirrored from its running card (snapshot md5 in SOURCES).")
    out.append("")
    out.append("The design of the certified set: 3 points at r_s/a 64 and 2 at 128, so the 4 coefficients fit the 2 points at 128 exactly and "
               "the 1 residual degree of freedom lies in the 3 points at 64 (the h-linearity there); with the uncertified (128, 4) record "
               "inside, each size carries 1 residual degree of freedom.")
    out.append("")
    out.append(d.fit_text("width_cross_certified"))
    out.append("")
    out.append(values_at_others(d, "width_cross_certified"))
    out.append("")
    out.append("With the uncertified (128, 4) record inside (labelled; not a law):")
    out.append("")
    out.append(d.fit_text("width_cross_with_uncertified"))
    out.append("")
    out.append("### 5.4 The release radius (the 2 r_s preparation members beside the 4 r_s members)")
    out.append("")
    rows = []
    for a_, b_ in (("P64_s4", "S4_R2_m0p2"), ("M3a_s4", "S4_R2_m0p1")):
        for nm in (a_, b_):
            M = d.M[nm]; fo = M["formation"]
            rows.append([nm, f(M["meta"]["m"], "%g"), f(M["meta"]["R0_over_rs"], "%g"),
                         (f(fo["T"]) + ("" if fo["certified"] else " (uncertified)")) if fo else "none to %s" % f(M["convergence"]["last"]),
                         f(fo["R_t"], "%.3f") if fo else "-", f(fo["R_t_over_rs"]) if fo else "-", f(fo["R_t_over_2gE"]) if fo else "-",
                         f(fo["gate_sigma_over_Rt"], "%.3f") if fo else "-", f(M["convergence"]["last_converged"])])
    out.append(table(["member", "m a", "R_0/r_s", "T_f/r_s", "R_t/a", "R_t/r_s", "R_t/(2gE)", "gate", "last converged record"], rows))
    out.append("")
    out.append("## 6 The convergence tables")
    out.append("")
    out.append("### 6.1 Per member")
    out.append("")
    out.append(d.convergence_table(names))
    out.append("")
    out.append("### 6.2 The excluded records (capped or kept above 1e-3), named")
    out.append("")
    out.append(d.excluded_text(names))
    out.append("")
    out.append("### 6.3 The stored controls and the width and release pairs (second minus first)")
    out.append("")
    out.append(d.spreads_table(SIZE_PAIRS))
    out.append("")
    out.append("## 7 The catalogs behind the own onsets, and the r_s/a 32 member's preparation")
    out.append("")
    out.append(d.catalog_text(names))
    out.append("")
    for line in SRC.get("rs32_prep", []):
        out.append(line)
    out.append("")
    out.append("## 8 Not composed, truncated or open")
    out.append("")
    for line in SRC.get("scaling_open", []):
        out.append("- " + line)
    out.append("")
    out.append("## 9 Figure data and figures")
    out.append("")
    for line in SRC.get("scaling_figs", []):
        out.append("- " + line)
    out.append("")
    out.append("## 10 SOURCES")
    out.append("")
    out.append(sources_block(d, names, SRC))
    out.append("")
    return "\n".join(out)
