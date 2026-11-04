"""compose_text_v1_3.py: the fixed wording of MASS_ladder_v1.3.md and SCALING_ladder_v1.3.md (imported by ladders_compose_v1_3.py). Every
number is read from the version 1.3 producer's json through the Doc object, or from the W4 (a) comparator's stored outputs where named;
nothing here computes a physical quantity. The corrections cite deliveries/lead_gaps/ReviewA_reading_LADDERS_v1.md by line."""
from ladders_compose_v1_3 import f, e, table

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
    "S4_128_s8": "width point (128, sigma/a 8), finished to 40", "S4_128_s16": "width point (128, sigma/a 16), at the gate's edge",
    "S4_R2_m0p2": "preparation member, release at 2 r_s, (0.2, 64)", "S4_R2_m0p1": "preparation member, release at 2 r_s, (0.1, 64)",
}

DEFS = (
    "Definitions (version 1.3; %s). A record is certified when the self-consistent solve of its fields has converged to a residual of at "
    "most 1e-3, the residual being the largest change of N or v over one iteration of the solve at its final fields; a residual epsilon "
    "shifts an exterior root of N - |v| by 2 r_s epsilon, about a tenth of a spacing at the threshold here, and a history is certified to its "
    "last record before the first that fails. Residuals between 1e-4 and 1e-3 are marked \u2020 in the tables, and a reading past a "
    "history's certified extent says so where it is made. As applied: both the kept residual (engine_ladder_compare_v5.rec_residual) and the "
    "confirming residual (a Newton record's solve_v32 confirming residual; a plain record kept unconverged, the iteration_history entry "
    "after the kept iterate) must be at most 1e-3; a record failing either does not count and is named. Two extents print for every member: "
    "the 30.3 prefix, ending before the first record whose kept residual exceeds 1e-3, and the certified extent (the certified prefix), "
    "ending before the first non-counting record of any kind. The last converged record is the last record of the certified extent. A "
    "formation is certified when its record lies inside the certified extent, and only such formations enter a fit. Formation is the first "
    "saved record holding a trapping zero (trapping_zeros non-empty), the birth lying in the interval (T_f minus the record step, T_f], whose "
    "width prints with each member; R_t at formation is the outer zero of the dense N + v on that first saved record and not the birth "
    "radius; the 2 zeros of every formation record print with no recorded tangency and a positive slope at R_t (the inner zero's simplicity "
    "is not tested). R_t/(2 g E) is the record's own stored ratio at its charge E (g 1/4): the root fraction. The root gate is sigma/R_t "
    "with sigma = ell. E/E0 is against the head's E0 of the ell 4 preparation, so it is not 1 at T = 0 for the re-averaged controls. "
    "N_B (m a)^3/1.087 is the nominal count ratio (to the displayed approximation the rest-energy ratio N_B m/E_max), not the record's charge "
    "ratio; the charge ratio E0/E_max uses the stored E0 and E_max = c_E/(g^(3/2) m^2) with c_E 0.135858. Labels: (n.c.) the record does not "
    "count; (unc.) the record lies after the member's certified extent; \u2020 the record counts and is banded (kept or confirming residual "
    "between 1e-4 and 1e-3)." % "deliveries/lead_gaps/ReviewA_cert_rule_v1.md")


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
    extra += ["producer: %s (md5 %s), version 1.2's producer with the production certification rule, rerun on the same stored "
              "files; helper engine_ladder_compare_v5.py (md5 %s)" % (SRC["producer"], SRC["producer_md5"], SRC["lc_md5"])]
    extra += ["certification inputs checked by the producer: %s (md5 %s, LADDER_certification_list_v1.1) and %s (md5 %s, written by %s, the "
              "following-entry test)" % (SRC["certext"], SRC["certext_md5"], SRC["followup"], SRC["followup_md5"], SRC["followup_script"])]
    extra += ["composer: %s and %s; the readout json %s (md5 %s); the arrays %s (md5 %s), mirrored on the box at %s with its md5 manifest" % (
        SRC["composer"], SRC["composer_text"], SRC["readout_json"], SRC["readout_json_md5"], SRC["readout_npz"], SRC["readout_npz_md5"], SRC["box_dir"])]
    extra += ["engine of the catalog's rerun preparations: %s (md5 %s)" % (SRC["engine"], SRC["engine_md5"])]
    extra += ["the reading applied: %s (reviewer A Part A, reviewer B Part B, the lane's cross-check); the certification rule applied: "
              "deliveries/lead_gaps/ReviewA_cert_rule_v1.md" % READING]
    extra += ["W4 (a) comparator numbers quoted in 5.1: <internal>/s4_density_continuum_rest_lapse-on_adaptive.json "
              "(through deliveries/solve_a/W5_W4_formation_attribution_v1.md section 4)"]
    return d.sources(names, extra)


RULE = "deliveries/lead_gaps/ReviewA_cert_rule_v1.md"
CRIT_T, CRIT_R = 0.5, 0.10   # the acceptance criteria the ruling applies to the averaging controls (0.5 r_s in T_f, 10 percent in R_t)


def fT(x):
    """a clock as stored, without rounding (12.78125, not 12.7812)."""
    if x is None:
        return "none"
    return ("%.6f" % x).rstrip("0").rstrip(".")


def rng(ts):
    """compress sorted clocks into ranges of consecutive integers."""
    ts = sorted(ts)
    out, i = [], 0
    while i < len(ts):
        j = i
        while j + 1 < len(ts) and abs(ts[j + 1] - ts[j] - 1) < 1e-9:
            j += 1
        out.append(fT(ts[i]) if j == i else ("%s to %s" % (fT(ts[i]), fT(ts[j]))))
        i = j + 1
    return ", ".join(out)


def g4(x):
    if x is None:
        return "none"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int, float)):
        return "%.4g" % x
    return str(x)


def control_verdict(d, a_, b_):
    """the factor-2 averaging control against the criteria, at formation and over the common certified zero records (mechanical)."""
    s = spread_of(d, a_, b_)
    if s.get("dT_f") is None:
        return None
    dt, rf, rm = abs(s["dT_f"]), abs(s.get("dR_t_rel") or 0), s.get("max_rel_dRt")
    within = dt <= CRIT_T and rf <= CRIT_R
    txt = ("%s against %s (%s): |dT_f| %s r_s, |dR_t|/R_t at formation %s, max |dR_t|/R_t over %s common certified zero records %s at %s; "
           % (b_, a_, s.get("classes"), f(dt, "%.4f"), f(100 * rf, "%.1f%%"), s.get("Rt_common_count"),
              f(100 * rm, "%.1f%%") if rm is not None else "-", f(s.get("max_rel_dRt_at"))))
    if not s.get("both_certified"):
        return txt + "not both certified, no verdict"
    if within:
        txt += "within both criteria at formation"
        if rm is not None and rm > CRIT_R:
            txt += ", the radius difference over the common records exceeding 10 percent"
        return txt
    bits = []
    if dt > CRIT_T:
        bits.append("the clock spread exceeds 0.5 r_s")
    if rf > CRIT_R:
        bits.append("the radius spread at formation exceeds 10 percent")
    if rm is not None and rm > CRIT_R:
        bits.append("the radius difference over the common records exceeds 10 percent")
    return txt + "; ".join(bits) + ": the control fails"


def changes_v1_3(d, D, which, names):
    """v1.3: the change list against version 1.2 (old json attached by the composer as D["_old"], the diff as D["_diff"])."""
    O = D["_old"]["members"]; N = D["members"]; X = D.get("_diff") or {}
    o = ["## Changes from v1.2", ""]
    o.append("- The certification rule is the production rule (%s): a record is certified when its kept residual and its confirming residual "
             "are both at most 1e-3, and a history is certified to its last record before the first that fails; records counting with either "
             "residual between 1e-4 and 1e-3 are banded and carry a dagger (†) in every table. Only the certification inputs changed (the "
             "producer's counting test, ladders_readout_v1_3.py); every record, clock, radius and fit input is read as in version 1.2. The converged "
             "flag no longer gates. A Newton record's confirming residual is solve_v32's (version 1.2 held it to stop_fields 1e-6). A plain record "
             "kept unconverged is confirmed by its iteration_history entry after the kept iterate, which the ruling asks for before the tables "
             "recompose (the plain mixer applies the kept fields, so that entry is the map's change out of them); a plain record whose kept iterate "
             "is its last entry, and every converged record, rest on the kept residual." % RULE)
    sh = [c for c in D.get("certext_check", []) if c["name"] in names and c["certext_extent"] != c["producer_extent"]]
    if sh:
        parts = []
        for c in sh:
            v = N[c["name"]]["convergence"]
            x = next((y for y in (v.get("unconverged_plain") or []) if v["first_noncounting"] is not None and abs(y["T"] - v["first_noncounting"]) < 1e-9), {})
            parts.append("%s %s to %s (record %s: kept %s, following entry %s)" % (
                c["name"], fT(c["certext_extent"]), fT(c["producer_extent"]), fT(v["first_noncounting"]), e(x.get("kept")), e(x.get("following"))))
        agree = sum(1 for c in D.get("certext_check", []) if c["certext_extent"] == c["producer_extent"])
        o.append("- That test shortens extents of LADDER_certification_list_v1.1 (certext_production_v1.json, which tested the kept residual "
                 "alone on plain members): %s. It shortens no other member (certext_followup_v1.py); the producer's extents equal the list's on the "
                 "other %d of %d members." % ("; ".join(parts), agree, len(D.get("certext_check", []))))
    mv = []
    for nm in names:
        a_, b_ = O[nm]["convergence"]["certified_prefix_end"], N[nm]["convergence"]["certified_prefix_end"]
        if a_ != b_:
            mv.append("%s %s to %s" % (nm, fT(a_), fT(b_)))
    same = [nm for nm in names if O[nm]["convergence"]["certified_prefix_end"] == N[nm]["convergence"]["certified_prefix_end"]
            and O[nm]["convergence"]["certified_prefix_end"] != O[nm]["convergence"]["last"]]
    o.append("- Certified extents that move (T/r_s): %s. Unchanged and short of the last record: %s." % (
        "; ".join(mv) if mv else "none", ", ".join("%s (%s)" % (nm, fT(N[nm]["convergence"]["certified_prefix_end"])) for nm in same) or "none"))
    fc = [nm for nm in names if N[nm]["formation"] and O[nm]["formation"]["class"] != N[nm]["formation"]["class"]]
    unc = [nm for nm in names if N[nm]["formation"] and not N[nm]["formation"]["certified"]]
    o.append("- Formations that become certified: %s. No formation loses its certification; %s." % (
        "; ".join("%s at %s (v1.2: %s)" % (nm, fT(N[nm]["formation"]["T"]), O[nm]["formation"]["class"]) for nm in fc) or "none",
        "; ".join("%s's at %s stays uncertified" % (nm, fT(N[nm]["formation"]["T"])) for nm in unc) or "every formation is certified"))
    bi, bo = [], 0
    for nm in names:
        for x in N[nm]["convergence"]["banded_records"]:
            if x["certified"]:
                bi.append("%s %s (kept %s, confirming %s)" % (nm, fT(x["T"]), e(x["kept"]), e(x["confirming"])))
            else:
                bo += 1
    atf = [nm for nm in names if N[nm]["formation"] and N[nm]["formation"].get("banded")]
    o.append("- Banded records inside the certified extents, marked † wherever a table prints them: %s; %s. Counting banded records past an "
             "extent carry (unc.) † (%d in this document's members, listed in 6.1)." % (
                 "; ".join(bi) or "none", ("at a formation record: " + ", ".join(atf)) if atf else "none is a formation record", bo))
    # values per member
    vl = []
    for nm in names:
        a_, b_ = O[nm], N[nm]
        bits = []
        if a_["convergence"]["counting"] != b_["convergence"]["counting"]:
            bits.append("counting records %d to %d" % (a_["convergence"]["counting"], b_["convergence"]["counting"]))
        if len(a_["convergence"]["excluded"]) != len(b_["convergence"]["excluded"]):
            bits.append("non-counting records %d to %d" % (len(a_["convergence"]["excluded"]), len(b_["convergence"]["excluded"])))
        if a_["convergence"].get("first_noncounting_gate") != b_["convergence"].get("first_noncounting_gate") and b_["convergence"]["first_noncounting"] is not None:
            bits.append("first non-counting record %s (%s)" % (fT(b_["convergence"]["first_noncounting"]), b_["convergence"]["first_noncounting_gate"]))
        ra, rb = a_["Rc_min"], b_["Rc_min"]
        if ra.get("min_Rc") != rb.get("min_Rc"):
            bits.append("min R_c over the certified records %s to %s a at %s (%s r_s)" % (
                ("%s a at %s" % (f(ra.get("min_Rc"), "%.2f"), fT(ra.get("at")))) if ra.get("min_Rc") is not None else "none (no certified record)",
                f(rb.get("min_Rc"), "%.2f"), fT(rb.get("at")), f(rb.get("min_Rc_over_rs"), "%.3f")))
        for i, (ea, eb) in enumerate(zip(a_["episodes"], b_["episodes"])):
            if ea.get("min_cert") != eb.get("min_cert"):
                bits.append("episode %d depth over certified records %s to %s (at %s)" % (
                    i + 1, f(ea.get("min_cert")) if ea.get("min_cert") is not None else "none", f(eb.get("min_cert")), fT(eb.get("min_cert_at"))))
        cc = [float(t) for t, r in b_["matched"].items() if r and a_["matched"][t] and r["certified"] and not a_["matched"][t]["certified"]]
        cn = [float(t) for t, r in b_["matched"].items() if r and a_["matched"][t] and r["counts"] and not a_["matched"][t]["counts"] and r["certified"]]
        cu = [float(t) for t, r in b_["matched"].items() if r and a_["matched"][t] and r["counts"] and not a_["matched"][t]["counts"] and not r["certified"]]
        if cc:
            bits.append("(unc.) removed at matched clocks %s" % rng(cc))
        if cn:
            bits.append("(n.c.) removed at %s" % rng(cn))
        if cu:
            bits.append("(n.c.) becomes (unc.) at %s" % rng(cu))
        ff = [t for t, r in b_["matched_from_formation"].items() if r and a_["matched_from_formation"][t] and r["certified"] != a_["matched_from_formation"][t]["certified"]]
        if ff:
            bits.append("(unc.) removed at T_f + %s" % ", ".join(ff))
        if bits:
            vl.append("  - %s: %s" % (nm, "; ".join(bits)))
    o.append("- Values that change, per member (every other member value is unchanged):")
    o += vl or ["  - none"]
    sl = []
    keys = (("common_clocks", "common certified clocks"), ("common_to", "last common clock"), ("max_abs_dEE0", "max |d E/E0|"), ("max_abs_dminNv", "max |d min(N - |v|)|"),
            ("max_abs_dRc_over_rs", "max |d R_c|/r_s"), ("Rt_common_count", "common zero records"), ("max_abs_dRt", "max |d R_t| (a)"),
            ("max_rel_dRt", "max |d R_t|/R_t"), ("max_rel_dRt_at", "at"), ("classes", "classes"), ("raw_common_clocks", "raw common clocks"))
    for sa, sb in zip(D["_old"]["spreads"], D["spreads"]):
        if sb["member"] not in names or sb["control"] not in names:
            continue
        cv = lambda k, v: fT(v) if (k in ("common_to", "max_rel_dRt_at") and v is not None) else g4(v)
        ch = ["%s %s to %s" % (lab, cv(k, sa.get(k)), cv(k, sb.get(k))) for k, lab in keys if sa.get(k) != sb.get(k)]
        if ch:
            sl.append("  - %s against %s: %s" % (sb["control"], sb["member"], "; ".join(ch)))
    o.append("- Control spreads that change (6.3, control minus member, over clocks counting and certified in both):")
    o += sl or ["  - none"]
    if which == "SCALING":
        Fo, Fn = D["_old"]["fits"], D["fits"]
        wo, wn = Fo.get("width_cross_certified") or {}, Fn.get("width_cross_certified") or {}
        o.append("- Fits (5.3): the certified width cross gains P64_s2: %d points to %d (%s), residual degrees of freedom %d to %d; T_f coefficients "
                 "%s to %s, residuals %s; it equals version 1.2's labelled set with the sigma/a 2 composite, which therefore no longer prints "
                 "separately; P64_s2 leaves the uncertified list of the every-record fit; the sigma/a 2 to 4 step at 64 is now certified in the "
                 "per-doubling table; the design rank's certified rows go from %d to %d (adding RS32_s2 and P64_s2), rank %d to %d, rows beyond "
                 "the rank %d to %d. No size-ladder fit (5.1, 5.2) and no release pair (5.4) changes." % (
                     wo.get("n", 0), wn.get("n", 0), ", ".join(wn.get("members", [])), wo.get("dof", 0), wn.get("dof", 0),
                     ", ".join("%.4g" % c for c in (wo.get("T_f") or {}).get("coef", [])), ", ".join("%.4g" % c for c in (wn.get("T_f") or {}).get("coef", [])),
                     ", ".join("%+.4f" % r for r in (wn.get("T_f") or {}).get("residual", [])),
                     len(Fo["design_rank"]["certified"]["rows"]), len(Fn["design_rank"]["certified"]["rows"]), Fo["design_rank"]["certified"]["rank"],
                     Fn["design_rank"]["certified"]["rank"], Fo["design_rank"]["certified"]["residual_dof"], Fn["design_rank"]["certified"]["residual_dof"]))
        v32 = control_verdict(d, "RS32_s4", "RS32_s2")
        o.append("- The factor-2 averaging control at r_s/a 32 is now certified on both sides and fails: %s." % v32)
    else:
        nf = [k for k in D["fits"] if k.startswith("mass") and D["fits"][k] != D["_old"]["fits"].get(k)]
        o.append("- Fits: %s." % ("none of the mass diagnostics (5.1 to 5.3) changes" if not nf else "changed: " + ", ".join(nf)))
        o.append("- The sigma/a 2 control at 64 is now certified: %s." % control_verdict(d, "P64_s4", "P64_s2"))
    o.append("- Wording: the definitions paragraph, the end-state notes of section 8, the convergence table's columns (confirming residual "
             "above 1e-3 for both solvers, the band column) and the sentences that quoted the strict extents now read the producer's extents. "
             "The change lists of versions 1.1 and 1.2 stay in those versions. Every changed value with its old and new entry (%d paths, %d of "
             "them new per-record band flags) is in ladders_diff_v1_3.json." % (
                 X.get("n_changed", 0), sum(1 for r in X.get("changed", []) if r["path"].endswith("/banded"))))
    o.append("")
    return o


def mass_doc(d, D, SRC, PREP_RULE):
    names = MASS + MASS_CTRL
    F = D["fits"]
    o = ["# MASS_ladder_v1.3: the mass ladder at sigma/a 4, tau 8 (m a 0.4, 0.2 and 0.1 at r_s/a 64; 0.1 at r_s/a 96; 0.05 at r_s/a 326), "
         "composed from the stored members, with the corrections of the correctness reading", ""]
    o.append("Version 1.3 of MASS_ladder_v1 (versions 1, 1.1 and 1.2 stay as issued). Composed by ladders_readout_v1_3.py (the producer, box CPU, "
             "stored run files only) and ladders_compose_v1_3.py (this text, from the producer's json). The builder composes; no sentence here reads "
             "the physics.")
    o.append("")
    o += changes_v1_3(d, D, "MASS", names)
    s = spread_of(d, "M3b_s4", "M3b_s8")
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
    o.append("M3d_s4's centroid minimum is at T = 0: its centroid moves outward over its %d certified records, unlike every other released member; "
             "not attributed here." % (d.M["M3d_s4"]["cert_end_index"] + 1))
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
             "controls are certified to %s (tau 4) and %s (tau 2), and the sigma/a 2 composite to %s (6.1). %s." % (
                 f(d.M["P64_s4_tau4"]["convergence"]["last_converged"], "%g"), f(d.M["P64_s4_tau2"]["convergence"]["last_converged"], "%g"),
                 f(d.M["P64_s2"]["convergence"]["last_converged"], "%g"), control_verdict(d, "P64_s4", "P64_s2")[0].upper() + control_verdict(d, "P64_s4", "P64_s2")[1:]))
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
    o = ["# SCALING_ladder_v1.3: the size ladder at m a 0.2, sigma/a 4 (r_s/a 16, 32, 64, 128), the width cross at r_s/a 64 and 128, and the "
         "2 r_s releases, composed from the stored members, with the corrections of the correctness reading", ""]
    o.append("Version 1.3 of SCALING_ladder_v1 (versions 1, 1.1 and 1.2 stay as issued). Composed by ladders_readout_v1_3.py and ladders_compose_v1_3.py "
             "from stored files only. The builder composes; no sentence here reads the physics. The size member at r_s/a 32 is the 1-field production "
             "c31_rs32_A_R2_tau2_ell4_prep4_avg (tau 2, 1024 sites).")
    o.append("")
    o += changes_v1_3(d, D, "SCALING", names)
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
    wcc = F.get("width_cross_certified") or {}
    wpt = lambda nm: "(%g, %g)" % (d.M[nm]["meta"]["rs"], d.M[nm]["meta"]["sigma"])
    wun = [nm for nm in ["P64_s2", "P64_s4", "P64_s8", "S4_128_s4", "S4_128_s8", "S4_128_s16"] if nm not in wcc.get("members", [])]
    o.append(("The 6 points: r_s/a 64 at sigma/a 2, 4, 8 and r_s/a 128 at sigma/a 4, 8, 16. Certified: %s; not certified: %s. " % (
        ", ".join(wpt(nm) for nm in wcc.get("members", [])), "; ".join("%s (formation %s, certified extent %s, first kept residual above 1e-3 at %s)" % (
            wpt(nm), f(d.M[nm]["formation"]["T"], "%g"), f(d.M[nm]["convergence"]["last_converged"], "%g"),
            f(d.M[nm]["convergence"]["first_above_1e-3"], "%g")) for nm in wun) or "none"))
             + "The z term carries one halving of tau/r_s between the sizes (0.125 at 64, 0.0625 at 128), whose share by the 64 step controls is "
             "-0.125 r_s and -0.019 in R_t/r_s per halving, measured at 1 size only. The (128, 8) point is read from its final run json (finished to T/r_s 40).")
    o.append("")
    o.append("The dependence per doubling of sigma/a (the records' numbers):")
    o.append("")
    o.append(table(["r_s/a", "from", "to", "dT_f (r_s)", "d(R_t/r_s)", "d(root fraction)", "both certified"],
                   [[x["r_s/a"], x["from"], x["to"], "%+.4f" % x["dT_f"], "%+.4f" % x["dR_t_over_rs"], "%+.4f" % x["dR_t_over_2gE"], "yes" if x["both_certified"] else "no"] for x in wd]))
    o.append("")
    o.append("On the %d certified points (4 coefficients, %s):" % (wcc.get("n", 0), ("%d residual degree%s of freedom" % (wcc["dof"], "" if wcc["dof"] == 1 else "s"))
                                                                     if wcc.get("dof") else "no residual test"))
    o.append("")
    o.append(d.fit_text("width_cross_certified"))
    o.append("")
    o.append(values_at_others(d, "width_cross_certified"))
    o.append("")
    if isinstance(F.get("width_cross_with_s2"), dict):
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
    rc_ = rk.get("certified") or {}
    o.append("The review's rank-7 null (z - 1)(2/3 - 7h/24 + j) holds on its own 8-row set; the certified rows here have rank %s of 8 on %d rows, "
             "%d rows beyond the rank (full rank alone is not a calibration)." % (rc_.get("rank"), len(rc_.get("rows", [])), rc_.get("residual_dof", 0)))
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
    o.append("The factor-2 averaging controls against the acceptance criteria (0.5 r_s in T_f, 10 percent in R_t): %s." % "; ".join(
        x for x in (control_verdict(d, a_, b_) for a_, b_ in (("RS32_s4", "RS32_s2"), ("P64_s4", "P64_s2"), ("P64_s4", "P64_s8"),
                                                               ("S4_128_s8", "S4_128_s16"))) if x))
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
