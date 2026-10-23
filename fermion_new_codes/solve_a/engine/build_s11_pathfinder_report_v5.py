#!/usr/bin/env python
"""build_s11_pathfinder_report_v5.py (the solve lane, for the analyst builder; MASSLESS_ball_pathfinder v1.2 section 9.2 to 9.3;
ReviewA_reading_batch3 item J): writes s11_pathfinder_report_v5.py from s11_pathfinder_report_v4.py by exact replacements. Each OLD
is copied from version 4 as attached and must occur exactly once; if the box copy differs from the attached text in whitespace only, the
OLD is matched with runs of whitespace collapsed and the replacement is reported as "whitespace-normalized". The written script's
docstring states the version-5 changes. The dens call of section 9.2 is not in version 4 (the report prints head.ball as the engine's
preparation path stored it and calls no dens); what the report side owns is changed here, and the fixed call's output enters version 5
through its --prep argument (RUN_s11_pathfinder_report_v5.md). After writing, this build compiles the result, imports it beside
engine_massive_channel_D_v2 and runs the new helpers on synthetic inputs (the two sides agreeing, disagreeing, and reconciled by a --prep
json; the labels), ending with SELFTEST PASS.
Usage: python build_s11_pathfinder_report_v5.py [--src s11_pathfinder_report_v4.py] [--dst s11_pathfinder_report_v5.py] [--no-import-check]"""
import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

REPL = []

# 1 the docstring: the version-5 note in front of version 4's
REPL.append(("docstring", '''#!/usr/bin/env python
"""s11_pathfinder_report_v4.py (the lane's analyst builder; engine spec 41.1 (1)): version 4 = version 3 with the transport label extended by''',
'''#!/usr/bin/env python
"""s11_pathfinder_report_v5.py (the solve lane's build for the analyst builder; MASSLESS_ball_pathfinder v1.2 section 9, ReviewA_reading_batch3 item J):
version 5 = version 4 with (i) the two sides of the preparation energy printed and tested against each other on every run: the initial aggregate
records[0].E_dust_shells (the raw-source path over all filled columns) beside the stored per-orbital diagnostic head.ball.E_bare_weighted (the
engine's dens call on individual column slices); the line "the bare energy (weighted) X = Y E0" prints only from a side that reproduces the
aggregate to the printed digits (5e-7), otherwise it says so and prints no bare energy; the stored per-orbital table keeps version 4's row format
and is followed by its dictionary check (a positive-norm ingoing massless column's Killing/bare between 1 and N_centre + (largest 2GM/r)^(1/2),
J/rho negative with |J|/rho at most 1, site-density minima nonnegative); the recomputed per-orbital diagnostic is read from --prep (a json in
head.ball's shape written by the fixed call on the run's stored T = 0 state, with the strings script and cause) and printed with the same table,
the same check, its weighted sum against the aggregate and its columns against the stored table, the cause going to the footer; (ii) the object
label and the transport label composed per run from the run's own args, head and first evolved record in place of the S11 text hard-coded on
every report, the certified-segment sentence supplied by --transport-note for the run it belongs to; (iii) the stale closing sentence about the
r_s/a 32 run dropped and the footer naming this script. No number of sections 2 to 5 changes. Version 4's docstring follows.
s11_pathfinder_report_v4.py (the lane's analyst builder; engine spec 41.1 (1)): version 4 = version 3 with the transport label extended by'''))

# 2 the module-level S11 labels become per-run helpers; the preparation's two sides and their checks
REPL.append(("module labels to per-run helpers", '''LABEL = ("the model's own massless object at the pathfinder size: the self-consistent massless collapse (S11), r_s/a 16, G/a^2 1/4, m 0, "
         "ell 2a (ell/2GE 0.125, below the critical width 0.26), tau 2; a radial channel reduction with weights 16 kappa (kappa 1 to 10)")
fmt = DV2.fmt
LABEL_TRANSPORT = ("the reference transport on the LZ convention from T/r_s 0.125 (c_LZ 1, at most 2 sub-steps): the Weyl certificate fails on all 10 channels "
                   "at the first step, margins -0.34 to -2.17 (norm of dh 0.173 to 1.212 against gaps 0.0039 to 0.0229); certification would need about 100 "
                   "sub-steps per channel per evaluation")
LABEL_TRANSPORT = LABEL_TRANSPORT.rstrip(".") + ("; LZ convention from 0.125; over 1.125 to 1.5 the certified eigenvalue-route transport (31 to 46 pieces per channel, 0 failing, "
                                                "smallest margin 4.11e-06) reproduces these records bitwise and the full route reproduces them within the floor")   # engine spec 41.1 (1)''',
'''fmt = DV2.fmt
CRITICAL_WIDTHS = (0.228195, 0.260875)     # ell/2GE's two critical widths (spec 42.5 (2): with source averaging; #62); the label prints the inequality only
TOL_SIDES = 5e-7                           # the two sides of the preparation energy agree when they print alike at %.6f
TABLE_HEAD = "| channel | j | Killing energy | bare energy | J/rho | site density min | weight on the ingoing spinor at k < 0 | weight above the quarter zone |"
TABLE_RULE = "|---|---|---|---|---|---|---|---|"
AGG_KEYS = ("E_dust_shells", "E_dust", "E_ball_shells", "E_ball")   # where records[0] is read for the initial aggregate of the raw-source path


def _first(d, keys):
    for k in keys:
        if d.get(k) is not None:
            return d[k]
    return None


def sdiv(x, y):
    return (x / y) if (x is not None and y is not None and y != 0) else None


def sdiff(x, y):
    return abs(x - y) if (x is not None and y is not None) else None


def make_label(ra, head, rs, g, ell, tr):
    """the object label from this run's own args and head, in place of the S11 text versions 1 to 4 hard-coded on every report"""
    E0 = head.get("E0")
    two_gE = (2.0 * g * float(E0)) if E0 else None
    w = sdiv(ell, two_gE)
    if w is None:
        where = "ell/2GE0 not computable, E0 missing"
    elif w < CRITICAL_WIDTHS[0]:
        where = "below both critical widths %.4f and %.4f" % CRITICAL_WIDTHS
    elif w < CRITICAL_WIDTHS[1]:
        where = "between the critical widths %.4f and %.4f" % CRITICAL_WIDTHS
    else:
        where = "above both critical widths %.4f and %.4f" % CRITICAL_WIDTHS
    m = _first(ra, ("m", "ma", "mass"))
    tau = _first(ra, ("tau", "tau_avg", "tau_s"))
    orb = (head.get("ball") or {}).get("orbitals")
    nch = (tr or {}).get("channels")
    return ("the model's own massless object at the pathfinder size: the self-consistent massless collapse, r_s/a %g, G/a^2 %g, m %s, ell %g a (ell/2GE0 %s, %s), tau %s, "
            "orbitals %s, %s channels (this run's args and head; not an inherited header)" % (
                rs, g, "(not in args)" if m is None else m, ell, fmt(w, "%.4f"), where, "(not in args)" if tau is None else tau, orb,
                "(no evolved record)" if nch is None else nch))


def make_transport_label(tr, note):
    """the transport label from this run's own first evolved record, in place of the S11 text hard-coded on every report; the certified-segment
    sentence of the run it belongs to is appended from --transport-note and from nowhere else"""
    if not tr:
        base = "the reference transport: the run JSON has no evolved record with continuation entries"
    else:
        base = ("the reference transport as the record labels it, %s, the first record on the LZ convention at T/r_s %s: at the first evolved record (T/r_s %s) %s of %s channels Weyl-verified, "
                "margins %s to %s (norm of dh %s to %s against gaps %s to %s), sub-steps used at most %s; certification would need at most %s sub-steps per channel per evaluation, %s times the sub-steps used" % (
                    ((tr.get("label") or "").split(";")[0].strip() or "no label"), fmt(tr.get("first_LZ_record_T_over_rs"), "%.4f"), fmt(tr.get("T_over_rs"), "%.4f"), tr.get("verified"), tr.get("channels"),
                    fmt(tr.get("margin_min"), "%.4f"), fmt(tr.get("margin_max"), "%.4f"), fmt(tr.get("norm_dH_min"), "%.4f"), fmt(tr.get("norm_dH_max"), "%.4f"), fmt(tr.get("gap_min"), "%.5f"),
                    fmt(tr.get("gap_max"), "%.5f"), tr.get("nsub_used_max"), tr.get("nsub_needed_max"), fmt(tr.get("cost_factor"), "%.1f")))
    if note:
        base = base.rstrip(".") + "; " + note.strip().rstrip(".")
    return base


def table_rows(per_orbital):
    if isinstance(per_orbital, list):
        per_orbital = {"1": per_orbital}
    rows = []
    for k_, rows_ in (per_orbital or {}).items():
        for q in rows_:
            rows.append((k_, q))
    return rows


def table_lines(per_orbital, strict=True):
    """strict: version 4's row format to the character (the stored table, the regression check); not strict: the same columns through fmt, a missing entry printing as fmt prints None"""
    out_ = []
    for k_, q in table_rows(per_orbital):
        if strict:
            out_.append("| %s | %d | %.6f | %.6f | %s | %.3e | %.6f | %.3e |" % (k_, q["j"], q["killing_energy"], q["bare_energy"], fmt(q.get("J_over_rho"), "%.5f"), q["site_density_min"],
                                                                        q["weight_ingoing_spinor_k_negative"], q["weight_above_quarter_zone"]))
        else:
            out_.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (k_, q.get("j"), fmt(q.get("killing_energy"), "%.6f"), fmt(q.get("bare_energy"), "%.6f"), fmt(q.get("J_over_rho"), "%.5f"),
                                                             fmt(q.get("site_density_min"), "%.3e"), fmt(q.get("weight_ingoing_spinor_k_negative"), "%.6f"), fmt(q.get("weight_above_quarter_zone"), "%.3e")))
    return out_


def dictionary_check(per_orbital, bound):
    """H = N E + v.p: a positive-norm ingoing massless column's Killing/bare lies between 1 and the profile's largest N + |v|; its current is inward with
    |J|/rho at most 1; its site density is nonnegative. Counts per table; consistent only when every row passes every test."""
    rows = table_rows(per_orbital)
    c = {"n": len(rows), "ratios": [], "bare_positive": 0, "bare_below_killing": 0, "in_bound": 0, "J_over_rho_negative": 0, "abs_J_over_rho_le_1": 0,
         "abs_J_over_rho_max": None, "site_density_min_nonneg": 0, "site_density_min_smallest": None, "bound": bound}
    for _, q in rows:
        ke, be, jr, sd = q.get("killing_energy"), q.get("bare_energy"), q.get("J_over_rho"), q.get("site_density_min")
        r = sdiv(ke, be)
        c["ratios"].append(r)
        if be is not None and be > 0:
            c["bare_positive"] += 1
            if ke is not None and be < ke:
                c["bare_below_killing"] += 1
        if r is not None and bound is not None and 1.0 <= r <= bound:
            c["in_bound"] += 1
        if jr is not None:
            if jr < 0:
                c["J_over_rho_negative"] += 1
            if abs(jr) <= 1.0:
                c["abs_J_over_rho_le_1"] += 1
            c["abs_J_over_rho_max"] = abs(jr) if c["abs_J_over_rho_max"] is None else max(c["abs_J_over_rho_max"], abs(jr))
        if sd is not None:
            if sd >= -1e-12:
                c["site_density_min_nonneg"] += 1
            c["site_density_min_smallest"] = sd if c["site_density_min_smallest"] is None else min(c["site_density_min_smallest"], sd)
    n = c["n"]
    c["consistent"] = bool(n and bound is not None and all(c[k] == n for k in ("bare_positive", "bare_below_killing", "in_bound", "J_over_rho_negative", "abs_J_over_rho_le_1", "site_density_min_nonneg")))
    return c


def check_line(chk, what, b):
    if not chk["n"]:
        return "- the dictionary check on %s: no per-orbital rows." % what
    cmp_ = b.get("max_compactness_2gM_over_r")
    v_ = math.sqrt(float(cmp_)) if (cmp_ is not None and float(cmp_) >= 0) else None
    return ("- the dictionary check on %s (H = N E + v.p: a positive-norm ingoing massless column's Killing/bare lies between 1 and the profile's largest N + |v|): Killing/bare per orbital %s against the bound 1 to %s "
            "(N at the centre %s plus |v| = (largest 2GM/r)^(1/2) = %s, the flat-slice form); bare energies positive on %d of %d and below the Killing energy on %d of %d; ratios inside the bound on %d of %d; "
            "J/rho negative on %d of %d, |J|/rho at most %s (at most 1 on %d of %d); site-density minima nonnegative (above -1e-12) on %d of %d, the smallest %s: %s." % (
                what, ", ".join(fmt(r, "%.3f") for r in chk["ratios"]), fmt(chk["bound"], "%.3f"), fmt(b.get("N_centre"), "%.6f"), fmt(v_, "%.3f"), chk["bare_positive"], chk["n"], chk["bare_below_killing"], chk["n"],
                chk["in_bound"], chk["n"], chk["J_over_rho_negative"], chk["n"], fmt(chk["abs_J_over_rho_max"], "%.5f"), chk["abs_J_over_rho_le_1"], chk["n"], chk["site_density_min_nonneg"], chk["n"],
                fmt(chk["site_density_min_smallest"], "%.3e"), "consistent with the dictionary" if chk["consistent"] else "INCONSISTENT with the dictionary"))


def column_diff_line(stored, recomputed):
    keys = ("killing_energy", "bare_energy", "J_over_rho", "site_density_min", "weight_ingoing_spinor_k_negative", "weight_above_quarter_zone")
    A = {(str(k_), int(q["j"])): q for k_, q in table_rows(stored)}
    B = {(str(k_), int(q["j"])): q for k_, q in table_rows(recomputed)}
    common = sorted(set(A) & set(B))
    if not common:
        return "- the recomputed table against the stored table: no (channel, j) in common (%d stored rows, %d recomputed)." % (len(A), len(B))
    mx = {}
    for key in keys:
        vals = [v for v in (sdiff(A[c].get(key), B[c].get(key)) for c in common) if v is not None]
        mx[key] = max(vals) if vals else None
    return ("- the recomputed table against the stored table on %d common (channel, j) rows (%d stored, %d recomputed): the largest |difference| per column, Killing energy %s, bare energy %s, J/rho %s, "
            "site density min %s, ingoing weight %s, quarter-zone weight %s." % (len(common), len(A), len(B), fmt(mx["killing_energy"], "%.3e"), fmt(mx["bare_energy"], "%.3e"), fmt(mx["J_over_rho"], "%.3e"),
                                                                              fmt(mx["site_density_min"], "%.3e"), fmt(mx["weight_ingoing_spinor_k_negative"], "%.3e"), fmt(mx["weight_above_quarter_zone"], "%.3e")))


def preparation_sides(a_prep, run, b, head, recs, rs, rs_E):
    """the two sides of the preparation energy at T = 0 (the initial aggregate of the raw-source path as the engine stored it in records[0]; the stored
    per-orbital diagnostic head.ball; the recomputed diagnostic from --prep) and the lines they decide; the bare energy prints only from a side that
    reproduces the aggregate to the printed digits"""
    P = {"run": run, "prep_path": a_prep, "E0": head.get("E0"), "T0_over_rs": (recs[0]["T"] / rs) if recs else None}
    E0_ = P["E0"]
    P["agg_key"] = next((k for k in AGG_KEYS if recs and recs[0].get(k) is not None), None)
    agg = float(recs[0][P["agg_key"]]) if P["agg_key"] else None
    stored = b.get("E_bare_weighted")
    cmp_ = b.get("max_compactness_2gM_over_r")
    bound = (float(b["N_centre"]) + math.sqrt(float(cmp_))) if (b.get("N_centre") is not None and cmp_ is not None and float(cmp_) >= 0) else None
    chk_stored = dictionary_check(b.get("per_orbital"), bound)
    rows_s = table_rows(b.get("per_orbital"))
    bare_sum_stored = sum(q["bare_energy"] for _, q in rows_s if q.get("bare_energy") is not None) if rows_s else None
    kill_sum = sum(q["killing_energy"] for _, q in rows_s if q.get("killing_energy") is not None) if rows_s else None
    w_impl = sdiv(stored, bare_sum_stored)
    kill_w = (w_impl * kill_sum) if (w_impl is not None and kill_sum is not None) else None
    comp_x0 = (2 * (math.sqrt(rs_E / b["x0"]) - math.log(1 + math.sqrt(rs_E / b["x0"]))) / (rs_E / b["x0"])) if b.get("x0") and rs_E > 0 else None
    comp_R0 = (2 * (math.sqrt(rs_E / b["R0"]) - math.log(1 + math.sqrt(rs_E / b["R0"]))) / (rs_E / b["R0"])) if b.get("R0") and rs_E > 0 else None
    thin = "the thin-layer control mu/E = 2[s - ln(1 + s)]/s^2 at R = x0 (s = (r_s/R)^(1/2)): %s; at R = R0: %s" % (fmt(comp_x0, "%.6f"), fmt(comp_R0, "%.6f"))
    PREP = None; pb = {}; pE = None; chk_prep = None; prep_name = None; prep_cause = None; bare_sum_p = None
    if a_prep:
        PREP = json.load(open(a_prep)); pb = PREP.get("ball") or PREP
        prep_name = PREP.get("script") or os.path.basename(a_prep); prep_cause = PREP.get("cause") or pb.get("cause")
        pE = pb.get("E_bare_weighted"); chk_prep = dictionary_check(pb.get("per_orbital"), bound)
        rows_p = table_rows(pb.get("per_orbital"))
        bare_sum_p = sum(q["bare_energy"] for _, q in rows_p if q.get("bare_energy") is not None) if rows_p else None
    agree_stored = (sdiff(stored, agg) is not None and sdiff(stored, agg) <= TOL_SIDES)
    agree_prep = (sdiff(pE, agg) is not None and sdiff(pE, agg) <= TOL_SIDES)
    if agree_stored:
        bare_val = stored; bare_src = "the stored per-orbital diagnostic, equal to the initial aggregate %s to the printed digits" % fmt(agg, "%.10f")
    elif PREP and agree_prep:
        bare_val = pE; bare_src = "the recomputed per-orbital diagnostic of %s, equal to the initial aggregate %s to the printed digits; the stored diagnostic %s = %s E0 was the inconsistent side" % (
            prep_name, fmt(agg, "%.10f"), fmt(stored, "%.6f"), fmt(sdiv(stored, E0_), "%.6f"))
    else:
        bare_val = None; bare_src = None
    if bare_val is not None:
        bare_line = "- the bare energy (weighted) %s = %s E0, against %s (%s)." % (fmt(bare_val, "%.6f"), fmt(sdiv(bare_val, E0_), "%.6f"), thin, bare_src)
    elif agg is None:
        bare_line = ("- the bare energy (weighted): not printed; the run JSON stores no initial aggregate (none of records[0].E_dust_shells, E_dust, E_ball_shells, E_ball), so the stored per-orbital diagnostic "
                     "%s = %s E0 has no second side to be tested against and is not printed as the bare energy; %s." % (fmt(stored, "%.6f"), fmt(sdiv(stored, E0_), "%.6f"), thin))
    elif PREP:
        bare_line = ("- the bare energy (weighted): not printed; the recomputed per-orbital diagnostic (%s) gives %s = %s E0 against the initial aggregate %s = %s E0 (|difference| %s, factor %s): they disagree, "
                     "the aggregate becomes the next suspect, and neither value is printed as this run's bare energy; %s." % (
                         prep_name, fmt(pE, "%.6f"), fmt(sdiv(pE, E0_), "%.6f"), fmt(agg, "%.6f"), fmt(sdiv(agg, E0_), "%.6f"), fmt(sdiff(pE, agg), "%.3e"), fmt(sdiv(agg, pE), "%.4f"), thin))
    else:
        bare_line = ("- the bare energy (weighted): not printed; the stored per-orbital diagnostic %s = %s E0 disagrees with the initial aggregate %s = %s E0 (|difference| %s, factor %s): the stored diagnostic is "
                     "the inconsistent side (its dictionary check below), no recomputed diagnostic was supplied (--prep), and no value is printed as this run's bare energy; %s." % (
                         fmt(stored, "%.6f"), fmt(sdiv(stored, E0_), "%.6f"), fmt(agg, "%.6f"), fmt(sdiv(agg, E0_), "%.6f"), fmt(sdiff(stored, agg), "%.3e"), fmt(sdiv(agg, stored), "%.4f"), thin))
    two_sides = ("- the two sides of the preparation energy at T/r_s %s: the initial aggregate %s (the raw-source path over all filled columns) %s = %s E0; the stored per-orbital diagnostic "
                 "head.ball.E_bare_weighted (the dens call on individual column slices; the table below) %s = %s E0; the two sides %s. The implied channel weight E_bare_weighted / (the stored per-orbital bare "
                 "energies' sum %s) %s; the weighted Killing sum (that weight times the per-orbital Killing energies' sum %s) %s = %s E0; the initial aggregate over the weighted Killing sum %s." % (
                     fmt(P["T0_over_rs"], "%.4f"), ("records[0].%s" % P["agg_key"]) if P["agg_key"] else "records[0].E_dust_shells (not stored)", fmt(agg, "%.10f"), fmt(sdiv(agg, E0_), "%.6f"),
                     fmt(stored, "%.10f"), fmt(sdiv(stored, E0_), "%.6f"),
                     ("agree to the printed digits (|difference| %s)" % fmt(sdiff(stored, agg), "%.1e")) if agree_stored else (
                         ("disagree (|difference| %s, factor %s)" % (fmt(sdiff(stored, agg), "%.4e"), fmt(sdiv(agg, stored), "%.3f"))) if sdiff(stored, agg) is not None else "cannot be compared (a side is not stored)"),
                     fmt(bare_sum_stored, "%.6f"), fmt(w_impl, "%.4f"), fmt(kill_sum, "%.6f"), fmt(kill_w, "%.6f"), fmt(sdiv(kill_w, E0_), "%.4f"), fmt(sdiv(agg, kill_w), "%.4f")))
    stored_status = "- the stored per-orbital diagnostic (head.ball.per_orbital; the dens call on individual column slices at the preparation): %s." % (
        "consistent with the initial aggregate; its columns are read as the preparation's" if agree_stored else (
            "INCONSISTENT with the initial aggregate (the two sides above); its columns bare energy, J/rho and site density min are printed as that call's output and are not read as the preparation's"
            if sdiff(stored, agg) is not None else "untested, a side is not in the run JSON"))
    R0_line = "- R0 %s (%s r_s), x0 %s, sigma %s; orbitals %s; k0 %s; the solved charge E0 %s against the target %s (E0 a; 2gE0 %s); %s." % (
        fmt(b.get("R0"), "%.3f"), fmt((b.get("R0") or 0) / rs, "%.3f"), fmt(b.get("x0"), "%.3f"), fmt(b.get("sigma"), "%.3f"), b.get("orbitals"), fmt(head.get("k0"), "%.10f"),
        fmt(head.get("E0"), "%.10f"), fmt(head.get("E_target"), "%.6f"), fmt(rs_E, "%.6f"), thin)
    P.update(dict(agg=agg, stored=stored, bound=bound, chk_stored=chk_stored, bare_sum_stored=bare_sum_stored, kill_sum=kill_sum, kill_w=kill_w, w_impl=w_impl, thin=thin, PREP=PREP, pb=pb, pE=pE,
                  chk_prep=chk_prep, prep_name=prep_name, prep_cause=prep_cause, bare_sum_p=bare_sum_p, agree_stored=agree_stored, agree_prep=agree_prep, bare_val=bare_val, bare_src=bare_src,
                  R0_line=R0_line, two_sides=two_sides, bare_line=bare_line, stored_status=stored_status))
    return P


def preparation_table_lines(P, b):
    """the stored per-orbital table in version 4's row format with its dictionary check; then, from --prep, the recomputed table with its checks"""
    lines = table_lines(b.get("per_orbital"))
    lines.append(check_line(P["chk_stored"], "the stored per-orbital diagnostic", b))
    if P["PREP"]:
        pb, pE, agg = P["pb"], P["pE"], P["agg"]
        prun = P["PREP"].get("run")
        warn = "" if (not prun or prun == P["run"]) else "; WARNING its run field %s is not this run" % prun
        lines += ["", "- the recomputed per-orbital diagnostic (--prep %s; %s%s; the fixed call on the stored T = 0 state): its weighted sum E_bare_weighted %s = %s E0 against the initial aggregate %s: %s; its per-orbital bare energies' sum %s, the implied weight %s." % (
            P["prep_path"], P["prep_name"], warn, fmt(pE, "%.10f"), fmt(sdiv(pE, P["E0"]), "%.6f"), fmt(agg, "%.10f"),
            ("equal to the printed digits (|difference| %s)" % fmt(sdiff(pE, agg), "%.1e")) if P["agree_prep"] else (
                ("NOT equal (|difference| %s, factor %s)" % (fmt(sdiff(pE, agg), "%.4e"), fmt(sdiv(agg, pE), "%.4f"))) if sdiff(pE, agg) is not None else "not comparable (a side is missing)"),
            fmt(P["bare_sum_p"], "%.6f"), fmt(sdiv(pE, P["bare_sum_p"]), "%.4f")), "", TABLE_HEAD, TABLE_RULE]
        lines += table_lines(pb.get("per_orbital"), strict=False)
        lines += [check_line(P["chk_prep"], "the recomputed per-orbital diagnostic", b), column_diff_line(b.get("per_orbital"), pb.get("per_orbital"))]
    return lines


def preparation_out(P, head, b):
    return {"k0": head.get("k0"), "E0": head.get("E0"), "ball": b,
            "aggregate": {"key": P["agg_key"], "value": P["agg"], "over_E0": sdiv(P["agg"], P["E0"]), "T_over_rs": P["T0_over_rs"]},
            "stored_diagnostic": {"E_bare_weighted": P["stored"], "over_E0": sdiv(P["stored"], P["E0"]), "agrees_with_aggregate": P["agree_stored"], "check": P["chk_stored"], "implied_weight": P["w_impl"],
                                  "killing_sum": P["kill_sum"], "killing_sum_weighted": P["kill_w"]},
            "recomputed_diagnostic": ({"prep": P["prep_path"], "script": P["prep_name"], "cause": P["prep_cause"], "E_bare_weighted": P["pE"], "over_E0": sdiv(P["pE"], P["E0"]),
                                       "agrees_with_aggregate": P["agree_prep"], "check": P["chk_prep"], "ball": P["pb"]} if P["PREP"] else None),
            "bare_energy_printed": P["bare_val"], "bare_energy_source": P["bare_src"]}'''))

# 3 the two new arguments
REPL.append(("arguments", '''ap.add_argument("--md", required=True)''',
'''ap.add_argument("--prep", default=None, help="the recomputed per-orbital preparation diagnostic: a json in head.ball's shape (E_bare_weighted, per_orbital) written by the fixed dens call on this run's stored T = 0 state, with the strings script, cause and run")
    ap.add_argument("--transport-note", dest="transport_note", default=None, help="the certified-segment sentence appended to this run's transport label (the lead's 41.1 (1) verdict, for the run it belongs to and no other)")
    ap.add_argument("--md", required=True)'''))

# 4 the out dict (the labels are set once the first evolved record is read)
REPL.append(("out dict", '''out = {"script": "s11_pathfinder_report_v4.py", "run": a.run, "label": LABEL, "label_transport": LABEL_TRANSPORT, "utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}''',
'''out = {"script": "s11_pathfinder_report_v5.py", "run": a.run, "args": ra, "utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}'''))

# 5 the labels per run and the title
REPL.append(("labels per run and the title", '''out["transport"] = tr
    L = ["# The massless ball pathfinder (S11): the self-consistent massless collapse at r_s/a 16", "", "Label (every table): %s." % LABEL, "",
         "Transport (every table): %s." % LABEL_TRANSPORT, "",''',
'''out["transport"] = tr
    LABEL = make_label(ra, head, rs, g, ell, tr)
    LABEL_TRANSPORT = make_transport_label(tr, a.transport_note)
    out["label"] = LABEL; out["label_transport"] = LABEL_TRANSPORT
    L = ["# The massless ball pathfinder: the self-consistent massless collapse at r_s/a %g, ell %g a (run %s)" % (rs, ell, tag), "", "Label (every table): %s." % LABEL, "",
         "Transport (every table): %s." % LABEL_TRANSPORT, "",'''))

# 6 the run line
REPL.append(("run line", '''"- run %s (%s; its records %d, the last at T/r_s %s); every number below is printed by s11_pathfinder_report_v1.py from the run's own files." % (
             a.run, J.get("script"), len(recs), fmt(recs[-1]["T"] / rs if recs else None, "%.4f")),''',
'''"- run %s (%s; its records %d, the last at T/r_s %s); every number below is printed by s11_pathfinder_report_v5.py from the run's own files; the run's args: %s." % (
             a.run, J.get("script"), len(recs), fmt(recs[-1]["T"] / rs if recs else None, "%.4f"), json.dumps(ra, sort_keys=True, default=float)),'''))

# 7 section 0's heading
REPL.append(("section 0 heading", '''"## 0. The engine: engine_collapse_v33", "",''',
'''"## 0. The engine", "",
         "- the run JSON's own script string %s; the run directory %s; the paragraph below describes version 33, the base of this lineage, as inherited from version 1 of this script, and these strings do not decide the records' engine version." % (J.get("script"), a.run),'''))

# 8 the preparation lines: the two sides and the bare energy decided by them
REPL.append(("preparation lines", '''L += ["## 1. The preparation (the chiral Slater seeds, the projection against the physical sea, the inward boost tuned after the constraint solve)", "",
          "- R0 %s (%s r_s), x0 %s, sigma %s; orbitals %s; k0 %s; the solved charge E0 %s against the target %s (E0 a; 2gE0 %s); the bare energy (weighted) %s = %s E0, against the thin-layer control mu/E = 2[s - ln(1 + s)]/s^2 at R = x0 (s = (r_s/R)^(1/2)): %s; at R = R0: %s." % (
              fmt(b.get("R0"), "%.3f"), fmt((b.get("R0") or 0) / rs, "%.3f"), fmt(b.get("x0"), "%.3f"), fmt(b.get("sigma"), "%.3f"), b.get("orbitals"), fmt(head.get("k0"), "%.10f"),
              fmt(head.get("E0"), "%.10f"), fmt(head.get("E_target"), "%.6f"), fmt(rs_E, "%.6f"), fmt(b.get("E_bare_weighted"), "%.6f"), fmt(b.get("E_bare_over_E"), "%.6f"),
              fmt((2 * (math.sqrt(rs_E / b["x0"]) - math.log(1 + math.sqrt(rs_E / b["x0"]))) / (rs_E / b["x0"])) if b.get("x0") and rs_E > 0 else None, "%.6f"),
              fmt((2 * (math.sqrt(rs_E / b["R0"]) - math.log(1 + math.sqrt(rs_E / b["R0"]))) / (rs_E / b["R0"])) if b.get("R0") and rs_E > 0 else None, "%.6f")),''',
'''P1 = preparation_sides(a.prep, a.run, b, head, recs, rs, rs_E)
    L += ["## 1. The preparation (the chiral Slater seeds, the projection against the physical sea, the inward boost tuned after the constraint solve)", "",
          P1["R0_line"], P1["two_sides"], P1["bare_line"],'''))

# 9 the table header, preceded by the stored diagnostic's status
REPL.append(("table header", '''"", "| channel | j | Killing energy | bare energy | J/rho | site density min | weight on the ingoing spinor at k < 0 | weight above the quarter zone |", "|---|---|---|---|---|---|---|---|"]''',
'''"", P1["stored_status"], "", TABLE_HEAD, TABLE_RULE]'''))

# 10 the per-orbital tables (stored, then recomputed from --prep), the k0 search, the json
REPL.append(("per-orbital tables and the json", '''for k_, rows_ in (b.get("per_orbital") or {}).items():
        for q in rows_:
            L.append("| %s | %d | %.6f | %.6f | %s | %.3e | %.6f | %.3e |" % (k_, q["j"], q["killing_energy"], q["bare_energy"], fmt(q.get("J_over_rho"), "%.5f"), q["site_density_min"],
                                                                         q["weight_ingoing_spinor_k_negative"], q["weight_above_quarter_zone"]))
    L += ["", "- the k0 search: %s." % "; ".join("k0 %.8f -> E %.10f" % (q["k0"], q["E"]) for q in (head.get("k0_history") or [])), ""]
    out["preparation"] = {"k0": head.get("k0"), "E0": head.get("E0"), "ball": b}''',
'''L += preparation_table_lines(P1, b)
    L += ["", "- the k0 search: %s." % "; ".join("k0 %.8f -> E %.10f" % (q["k0"], q["E"]) for q in (head.get("k0_history") or [])), ""]
    out["preparation"] = preparation_out(P1, head, b)'''))

# 11 the footer: this script's name, the cause from --prep, the stale sentence dropped
REPL.append(("footer", '''L.append("- every number above is printed by s11_pathfinder_report_v1.py from the files named; the r_s/a 32 run at ell 2a and 4a is the owner's decision after this file and is not launched by it.")''',
'''L.append("- every number above is printed by s11_pathfinder_report_v5.py from the files named%s." % (
        (("; the recomputed per-orbital preparation diagnostic from %s (--prep %s), whose stated cause of the stored diagnostic's defect is: %s" % (P1["prep_name"], a.prep, P1["prep_cause"] or "not stated in the --prep json")) if P1["PREP"]
         else "; the per-orbital preparation diagnostic is the stored head.ball of the run JSON (no --prep supplied)")))'''))


def apply_one(text, name, old, new):
    n = text.count(old)
    if n == 1:
        return text.replace(old, new, 1), "exact"
    if n > 1:
        raise SystemExit("replacement %s: OLD occurs %d times (must be once); nothing written" % (name, n))
    pat = re.compile(r"\s+".join(re.escape(t) for t in old.split()))
    ms = list(pat.finditer(text))
    if len(ms) != 1:
        raise SystemExit("replacement %s: OLD not found (exact 0 times, with whitespace collapsed %d times); the box copy of version 4 differs from the attached text at: %r; nothing written" % (name, len(ms), old[:100]))
    m = ms[0]
    return text[:m.start()] + new + text[m.end():], "whitespace-normalized"


def helper_selftest(dst):
    import importlib
    import json
    import tempfile
    sys.path.insert(0, os.path.dirname(os.path.abspath(dst)))
    try:
        mod = importlib.import_module(os.path.splitext(os.path.basename(dst))[0])
    except Exception as e:
        raise SystemExit("import of the written script failed (%s: %s); run this build in the directory where version 4 ran, beside engine_massive_channel_D_v2.py, or pass --no-import-check" % (type(e).__name__, e))
    print("import check: ok (engine_massive_channel_D_v2 found)")
    fmt = mod.fmt
    good = [{"j": 0, "killing_energy": 0.12, "bare_energy": 0.08, "J_over_rho": -0.7, "site_density_min": 2e-7, "weight_ingoing_spinor_k_negative": 0.9, "weight_above_quarter_zone": 3e-3},
            {"j": 1, "killing_energy": 0.24, "bare_energy": 0.15, "J_over_rho": -0.8, "site_density_min": 3e-7, "weight_ingoing_spinor_k_negative": 0.93, "weight_above_quarter_zone": 3e-3}]
    bad = [dict(q) for q in good]
    bad[0].update(bare_energy=0.004, J_over_rho=0.3, site_density_min=-7e-3)
    bad[1].update(bare_energy=0.0075, J_over_rho=-1.5, site_density_min=-7e-3)
    AGG = 16.0 * (0.08 + 0.15)
    head = {"E0": 32.0, "E_target": 32.0, "k0": 0.02}
    base = {"R0": 64.0, "x0": 32.0, "sigma": 16.0, "orbitals": "1:2", "N_centre": 1.5, "max_compactness_2gM_over_r": 0.25}
    b_good = dict(base, E_bare_weighted=AGG, per_orbital={"1": good})
    b_bad = dict(base, E_bare_weighted=16.0 * (0.004 + 0.0075), per_orbital={"1": bad})
    rs, g, ell = 16.0, 0.25, 2.0
    rs_E = 2.0 * g * head["E0"]
    recs = [{"T": 0.0, "E_dust_shells": AGG}]
    # the two sides agree (the S11 situation)
    P = mod.preparation_sides(None, "run/x", b_good, head, recs, rs, rs_E)
    want = "- the bare energy (weighted) %s = %s E0, against the thin-layer control" % (fmt(AGG, "%.6f"), fmt(AGG / 32.0, "%.6f"))
    assert P["agree_stored"] and P["bare_val"] == AGG and P["bare_line"].startswith(want) and P["chk_stored"]["consistent"], P["bare_line"]
    assert ("at R = x0 (s = (r_s/R)^(1/2)): %s; at R = R0: %s" % (fmt(0.689227, "%.6f"), fmt(0.756279, "%.6f"))) in P["thin"], P["thin"]
    lines = mod.preparation_table_lines(P, b_good)
    assert lines[0] == "| 1 | 0 | 0.120000 | 0.080000 | %s | 2.000e-07 | 0.900000 | 3.000e-03 |" % fmt(-0.7, "%.5f"), lines[0]
    assert lines[2].endswith("consistent with the dictionary.") and "INCONSISTENT" not in lines[2], lines[2]
    # the stored diagnostic disagrees with the aggregate, no --prep (the pair's situation before the fix)
    P = mod.preparation_sides(None, "run/x", b_bad, head, recs, rs, rs_E)
    assert (not P["agree_stored"]) and P["bare_val"] is None and P["bare_line"].startswith("- the bare energy (weighted): not printed; the stored per-orbital diagnostic"), P["bare_line"]
    assert "INCONSISTENT with the initial aggregate" in P["stored_status"] and ("factor %s" % fmt(20.0, "%.3f")) in P["two_sides"], P["two_sides"]
    lines = mod.preparation_table_lines(P, b_bad)
    assert lines[2].endswith("INCONSISTENT with the dictionary."), lines[2]
    # --prep reproduces the aggregate (the pair's situation after the fix)
    fd, pth = tempfile.mkstemp(suffix=".json"); os.close(fd)
    json.dump({"script": "fixed_prep_selftest", "cause": "selftest", "run": "run/x", "E_bare_weighted": AGG, "per_orbital": {"1": good}}, open(pth, "w"))
    try:
        P = mod.preparation_sides(pth, "run/x", b_bad, head, recs, rs, rs_E)
        assert P["agree_prep"] and P["bare_val"] == AGG and P["bare_line"].startswith(want) and "the recomputed per-orbital diagnostic of fixed_prep_selftest" in P["bare_line"], P["bare_line"]
        lines = mod.preparation_table_lines(P, b_bad)
        cd = [x for x in lines if x.startswith("- the recomputed table against the stored table")]
        assert len(cd) == 1 and ("bare energy %s" % fmt(0.15 - 0.0075, "%.3e")) in cd[0], cd
        json.dumps(mod.preparation_out(P, head, b_bad), default=float)
    finally:
        os.remove(pth)
    lab = mod.make_label({"rs": 16, "g": 0.25, "ell": 2, "m": 0, "tau": 2}, dict(head, ball={"orbitals": "1:2"}), rs, g, ell, {"channels": 10})
    assert "r_s/a 16, G/a^2 0.25, m 0, ell 2 a (ell/2GE0 0.1250, below both critical widths 0.2282 and 0.2609), tau 2, orbitals 1:2, 10 channels" in lab, lab
    tl = mod.make_transport_label({"T_over_rs": 0.125, "channels": 10, "verified": 0, "margin_min": -2.1, "margin_max": -0.3, "norm_dH_min": 0.17, "norm_dH_max": 1.2, "gap_min": 0.0039, "gap_max": 0.0229,
                                   "nsub_used_max": 2, "nsub_needed_max": 106, "cost_factor": 91.4, "label": "LZ convention (c_LZ = 1) from T/rs 0.125 (channels 1 to 10); inversions 0",
                                   "first_LZ_record_T_over_rs": 0.125}, "a note.")
    assert tl.startswith("the reference transport as the record labels it, LZ convention (c_LZ = 1) from T/rs 0.125 (channels 1 to 10), the first record on the LZ convention at T/r_s 0.1250: "
                         "at the first evolved record (T/r_s 0.1250) 0 of 10 channels Weyl-verified") and tl.endswith("; a note"), tl
    print("helper self-test: ok (the two sides agreeing, disagreeing, reconciled by --prep; the labels)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(HERE, "s11_pathfinder_report_v4.py"))
    ap.add_argument("--dst", default=os.path.join(HERE, "s11_pathfinder_report_v5.py"))
    ap.add_argument("--no-import-check", dest="no_import", action="store_true", help="skip importing the written script and its helper self-test (only where engine_massive_channel_D_v2 is absent)")
    a = ap.parse_args()
    text = open(a.src).read()
    for name, old, new in REPL:
        text, mode = apply_one(text, name, old, new)
        print("replacement %-36s %s" % (name + ":", mode))
    for bad in ("the owner's decision after this file", "printed by s11_pathfinder_report_v1.py", 'LABEL = ("the model', '"script": "s11_pathfinder_report_v4.py"', "LABEL_TRANSPORT = ("):
        if bad in text:
            raise SystemExit("post-check failed: %r still in the result; nothing written" % bad)
    for good in ("def make_label(", "def make_transport_label(", "def preparation_sides(", "def dictionary_check(", '"--prep"', '"--transport-note"', "P1 = preparation_sides("):
        if good not in text:
            raise SystemExit("post-check failed: %r not in the result; nothing written" % good)
    print("post-checks: ok")
    try:
        compile(text, a.dst, "exec")
    except SyntaxError as e:
        raise SystemExit("compile failed: %s; nothing written" % e)
    print("compile: ok")
    open(a.dst, "w").write(text)
    print("wrote %s (%d lines)" % (a.dst, text.count("\n")))
    if a.no_import:
        print("import check and helper self-test: skipped (--no-import-check)")
    else:
        helper_selftest(a.dst)
    print("SELFTEST PASS")


if __name__ == "__main__":
    main()
