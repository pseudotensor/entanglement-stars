#!/usr/bin/env python
"""compose_relabel_v1_20.py (builder analyst; engine spec 38.7: "compose_relabel_v1_20 (v1.18's three and v1.19's four rows closed; the
convention row of 38.6 (1); the growth row of 38.3 (2); HORIZON v1's "frozen" read "fixed-field"; ITEM5 v9 read as v10; the capture row)",
with 38.2 (2), 38.3 (2), 38.4 (1), 38.5, 38.6 (1); and the lane's additions of 2026-10-16 from about 22:45 to 23:40 UTC: inbox #45 and #46
on HORIZON_single_object_v1 until its version 2; #47 with #49 on TEMPERATURE_solved_exterior_v1.5 until the lead's version 1.6; #48 on
PAGE_solved_history_v1a until its version 2; the builder's composition checks (ITEM5 version 10's 0.105; ASYMPTOTICS version 1.8's carried
row (3) lines); the capture run's version: version 10 named where version 11 runs): version 1.19 with the title's source list, the version
line and the sources line extended and the rows below added before the sources line. Every number in the builder's rows is printed from
its file (origin_term_p37_v1.md's summary line, fold_birth_summary_v2.md's episode-1 lines, the gate (a) JSONs, the full-precision
before-arrival JSON); every quoted review passage is checked against its source (line breaks joined, TeX markup and math delimiters
removed, its prose fragments and its numbers found) and a passage not found stops the compose; no pipe inside a cell (pipes in quoted
text written abs()). Refuses to overwrite. Args: --v119 --out --deliv --origin --fold --cap10 --log."""
import argparse
import glob
import json
import os
import re
import sys


def norm_tex(t):
    t = t.replace("\n", " ")
    t = re.sub(r"\\(emph|textbf|texttt|path)\{([^{}]*)\}", r"\2", t)
    t = t.replace("\\(", " ").replace("\\)", " ").replace("\\,", " ").replace("$", " ").replace("``", '"').replace("''", '"').replace("~", " ")
    t = t.replace("\\_", "_").replace("\\%", "%").replace("{", "").replace("}", "").replace("\\", "")
    t = re.sub(r"\s+", " ", t)
    return re.sub(r"\s+([,.;:)])", r"\1", t)


def main():
    ap = argparse.ArgumentParser()
    for k in ("v119", "out", "deliv", "origin", "fold", "cap10", "log"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    for p in (a.out, a.log):
        if os.path.exists(p):
            sys.exit("refusing to overwrite %s" % p)
    L = ["## compose_relabel_v1_20 (engine spec 38.7): the rows, the numbers printed from their files, the quote checks", ""]
    R = os.path.dirname(a.deliv.rstrip("/"))
    src = {"#45": os.path.join(R, "hreduce/web40f_single_object_identity.tex"), "#46": os.path.join(R, "solve_b/review_HORIZON_single_object_v1.tex"),
           "#47": os.path.join(R, "solve_b/review_TEMPERATURE_solved_exterior_v1_5_amendments.tex"), "#48": os.path.join(R, "solve_b/review_PAGE_solved_history_v1a.tex"),
           "#49": os.path.join(R, "solve_b/review_TEMPERATURE_solved_exterior_v1_5.tex")}
    txt = {k: norm_tex(open(p).read()) for k, p in src.items()}
    checks = []

    def found(key, *frags):
        for f in frags:
            ok = norm_tex(f) in txt[key]
            checks.append((key, f, ok))
            if not ok:
                open(a.log, "x").write("\n".join(L + ["| %s | %s | %s |" % c for c in checks]) + "\n\nSTOPPED: a quoted passage is not in %s\n" % src[key])
                sys.exit("not found in %s: %r" % (src[key], f))
    # the builder's numbers, printed from their files
    org = open(a.origin).read()
    cert = re.search(r"- (\d+) records certified of (\d+) by the rule", org)
    lm = re.search(r"records whose share with the local stress falls below 0\.800: tt (\d+) \(([^)]*)\), rr (\d+) \(([^)]*)\)", org)
    tt = [x.strip() for x in lm.group(2).split(",")]; rr = [x.strip() for x in lm.group(4).split(",")]
    n_below = len(set(tt) | set(rr))
    if int(lm.group(1)) != len(tt) or int(lm.group(3)) != len(rr):
        sys.exit("the origin-term summary's counts do not match its lists")
    L.append("- %s: %s of %s records certified; below the 0.8 share with the local stress: tt %d, rr %d, together %d records" % (a.origin, cert.group(1), cert.group(2), len(tt), len(rr), n_below))
    lead_conv = "the stress is the open chain's prefix flux with its origin term; under the local allocation of the P37 idea 20 of the 281 certified records fall below the 0.8 share"
    if "%d of the %s certified records" % (n_below, cert.group(1)) not in lead_conv:
        sys.exit("the lead's convention sentence does not carry the file's counts")
    fold = open(a.fold).read()
    grow = {}
    for tau in ("8", "4", "2"):
        sec = fold.split("### the production, tau %s\n" % tau, 1)[1].split("\n### ", 1)[0]
        ln = [x for x in sec.split("\n") if x.startswith("- R_t, episode 1 (")][0]
        nb = re.search(r"(\d+) with both sides", ln).group(1); op = re.search(r"matter term opposite in sign to dR/dT on (\d+) records", ln).group(1)
        med = re.search(r"\|defect term\|/\|matter term\| [^;]*?\(median ([0-9.e+-]+)\)", ln)
        grow[tau] = (op, nb, med.group(1) if med else None)
    L.append("- %s, the production's R_t, episode 1: %s" % (a.fold, "; ".join("tau %s: the matter term opposite to dR/dT on %s of %s records, the defect over the matter term median %s" % (t, g[0], g[1], g[2]) for t, g in grow.items())))
    gates = {}
    for p in sorted(glob.glob(os.path.join(a.cap10, "gate_a_v11_om*.json"))):
        g = json.load(open(p))
        port = [r["return_max_over_arrived"] for r in g["rows"] if r["case"].startswith("version 10's")]
        v11 = [r["return_max_over_arrived"] for r in g["rows"] if r["case"].startswith("v11 append 512")]
        gates[g["omega_over_m"]] = (min(port), max(port), max(v11), g["kappas"])
    L.append("- %s gate (a): %s" % (a.cap10, "; ".join("omega/m %g (kappa %s): version 10's port %.3f to %.3f, version 11's absorber at most %.1e" % (k, ", ".join(str(x) for x in v[3]), v[0], v[1], v[2]) for k, v in sorted(gates.items()))))
    ba = json.load(open(os.path.join(a.cap10, "capture_v9_t300_min_before_arrival_v1.json")))
    r15 = [r for r in ba["rows"] if r["background"].startswith("solved") and abs(r["e"] - 1.5) < 1e-9][0]
    L.append("- %s: the solved slice at omega/m 1.5, the smallest sigma/(pi b_c^2) before the arrival %.10f at %.1f r_s (three decimals %.3f)" % (
        os.path.join(a.cap10, "capture_v9_t300_min_before_arrival_v1.json"), r15["smallest_ratio_before"], r15["its_T_over_rs"], r15["smallest_ratio_before"]))
    gp = gates[1.1]; g2 = gates[2.0]
    vmax = max(v[2] for v in gates.values())
    # the review quotes, checked against their sources
    found("#46", "These per-channel readouts do not measure permanent escape or emitted positive particles",
          "Keep the reconstruction residual separate from the stored state's purity error",
          "is an unsigned windowed momentum share, not a projection onto the guaranteed outward positive intrinsic Dirac branch; that band need not have positive Killing energy",
          "B5 propagates complete selected orbitals and retains their spatial cross terms; its summed position weight omits the remaining occupied sector and reference, not those selected cross terms",
          "Its beyond-root readout cannot inherit the Gaussian probes' far-detector result",
          "localization classifier is not the infinite-exterior bound-state projector; a mean frequency below",
          "does not exclude above-threshold spectral support",
          "Fixed-cut number current, moving-cut reassignment, the complete reference term", "Killing-energy flux and Gauss charge loss remain distinct")
    found("#45", "These are internal checks of the", "reconstructed occupied projector; the separate largest stored-covariance purity defect is", "2.50", "5.87", "7.52",
          "They are per-channel correlations, with no angular multiplicity applied, and establish neither escape nor a three-dimensional entropy coefficient",
          "The reported second momentum quarter is a window in", "not a signed, band-resolved outward occupation")
    for k in ("#47", "#49"):
        found(k, "rounded extrema", "3.544", "1.051", "1.428508", "1.480122", "0.921823", "0.885221", "0.959497", "0.734071", "v2's lower-precision display recorded in its header")
    found("#47", "not a certified enclosure or an absolute slope error", "supplies the dominant change in", "The other supplied factors also change",
          "The delivered table is the v3 output, with v2's lower-precision display recorded in its header")
    found("#49", "not an absolute slope error or certified enclosure", "as its dominant contribution", "The other factors also change",
          "Explicitly supersede the clause that", "is v3 output, with v2's lower-precision display recorded in its header")
    found("#48", "band adds a propagation tail and unitarity defect; it is not a certified entropy error in nats",
          "Their printed values do not establish the matched work identity",
          "The same-clock energy and collar residuals are algebraically the same residual, not two balance tests",
          "0.03099214", "0.02637303", "the mode ledger alone is not positive escaped energy",
          "Keep the actual velocity-selected archive and its possible returns distinct from an asymptotic radiation space",
          "are comparators, not an observed thermal flux", "they are not physical partner occupations",
          "The arrival marker uses", "in its travel integral although the actual collar is", "the plotted L still uses Delta T, not Delta t_infty")
    r46, r45, r47, r49, r48 = ("review_HORIZON_single_object_v1.tex, inbox #46", "web40f_single_object_identity.tex, inbox #45",
                               "review_TEMPERATURE_solved_exterior_v1_5_amendments.tex, inbox #47", "review_TEMPERATURE_solved_exterior_v1_5.tex, inbox #49",
                               "review_PAGE_solved_history_v1a.tex, inbox #48")
    rows = [
        # 38.2 (1), (2)
        "| TEMPERATURE_solved_exterior_v1.4.md | read as TEMPERATURE_solved_exterior_v1.5.md (engine 38.2 (1), turn 46: TEMPERATURE_solved_exterior_v1.5_amendments.md's (a) to (f) applied by compose_temperature_v1_5b, every placeholder printed from the stored files its condition names; version 1.4 stays the record) |",
        "| version 1.18's three rows on TEMPERATURE_solved_exterior_v1.4.md (inbox #34: \"within about 12 percent\", the local conversion scale, the measured averaging term) | closed: TEMPERATURE_solved_exterior_v1.5.md replaces those clauses ((b), (d), (e); engine 38.2 (2)) |",
        "| version 1.19's four rows on TEMPERATURE_solved_exterior_v1.4.md (inbox #44: the counts' basis, the tau 4 counts' convention, D_1 and delta_conn with kappa_read unresolved, the kappa_0 clause) | closed: TEMPERATURE_solved_exterior_v1.5.md replaces those clauses ((e), (f), (c), (d); engine 38.2 (2)); its own remaining precisions are the three rows below (inbox #47, #49) |",
        # 38.6 (1)
        "| COLLAPSE_certificate_v1.2.md and COLLAPSE_certificate_statement_v1.3.md (the stress the shares are computed with) | \"%s\" (engine 38.6 (1); origin_term_p37_v1.md: %s of %s records certified; under the local stress tt %d and rr %d records below the 0.8 share, %d records); a convention line, not a recertification: no certificate version 1.3; the statement's version 1.4 in turn 47 carries the shell-0 convention beside it (engine 38.3 (3)) |" % (
            lead_conv, cert.group(1), cert.group(2), len(tt), len(rr), n_below),
        # 38.3 (2)
        "| COLLAPSE_records_v1.1 (every turn's label) and COLLAPSE_records_v1 of the tau 4 and tau 2 histories: the first episode's trapping radius R_t rising record by record from the formation | read with the growth identity (fold_birth_summary_v2.md): in the first episode the first surface's growth is carried by the rows' defect D_M, not by accretion through the surface; the matter term's sign is opposite to dR_t/dT on %s of %s records at tau 8, %s of %s at tau 4 and %s of %s at tau 2, the defect term exceeding the matter term by medians %s, %s and %s; no record line, label or item 5 sentence reads the surface as growing as matter falls in; insert 17's growth wording is not supported in the first episode on these records (engine 38.3 (2)) |" % (
            grow["8"][0], grow["8"][1], grow["4"][0], grow["4"][1], grow["2"][0], grow["2"][1], "%.2f" % float(grow["8"][2]), "%.2f" % float(grow["4"][2]), "%.2f" % float(grow["2"][2])),
        # 38.4 (1) (iii)
        "| HORIZON_single_object_v1.md and HORIZON_single_object_v1_summary.json: the label \"frozen\" (B's launches on the fields held at a record; the json's mode \"frozen\") | read \"fixed-field\" in every label (engine 38.4 (1) (iii); requirements_new.txt sec. 4); the json's key values unchanged; version 2 prints \"fixed-field\" |",
        # 38.3 (5), 38.5 (4), 38.6 (3)
        "| ITEM5_collapse_sentences_v9.md | read as ITEM5_collapse_sentences_v10_text.md, the consolidated text of versions 3 to 10 composed by compose_item5_v10 (engine 38.3 (5), 38.5 (4), 38.6 (3): sentence 6 in the failure form with the r_s/a 32 pair's numbers, the capture clause of 38.5); ITEM5_collapse_sentences_v10.md holds the replacements; versions 3 to 9 stay the record |",
        # the capture row (38.5)
        "| ASYMPTOTICS_checks_v1.7.md row (6) and ITEM5_collapse_sentences_v9.md's capture clause (C-capture version 9 at 40 r_s, and its run to 300 r_s) | read as ASYMPTOTICS_checks_v1.8.md row (6) and ITEM5_collapse_sentences_v10_text.md's capture clause: not settled within 56 r_s under the in-transit gate; the production's 2048-site chain, whose outer end has no absorber, returns weight from 56 to 58 r_s on in every probe, so no later reading is a non-return fraction; the maximum's 300 r_s run dropped (engine 38.5 (1), (2), (4)) |",
        "| ASYMPTOTICS_checks_v1.8.md row (6): \"A version 10 run with an outer absorbing port (engine spec 38.5 (3)) replaces this row when it lands\"; ITEM5_collapse_sentences_v10_text.md's capture clause: \"on a run that settles under the in-transit gate on the chain with an outer absorbing port (engine spec 38.5 (3))\" | the run is engine_capture_v11's, the absorber on 512 appended sites of the declared exterior (the lane's ruling of 2026-10-16 about 23:02 UTC; its gates (a) to (c) before any 300 r_s run): version 10's port, the inner port's construction on the last 16 sites, returns %.3f to %.3f of an arriving outgoing packet at omega/m 1.1 and %.4f to %.4f at omega/m 2, version 11's absorber at most %.1e at the four probes (test_capture_v11_absorber_v1, gate (a)); version 10's 40 r_s gate run stays a record; until the lead's next versions |" % (
            gp[0], gp[1], g2[0], g2[1], vmax),
        "| ITEM5_collapse_sentences_v10_text.md's capture clause: \"read 0.199, 0.126, 0.105 and 0.018\" | \"0.105\" reads \"%.3f\": the stored value %.7f (capture_v9_t300_prod19_min.json, the solved slice at omega/m 1.5, %.1f r_s), the four-decimal 0.1045 of capture_read_v3_v9_t300_min_final.md rounded a second time (the builder's composition check, agreed by the lane); until the lead's next version |" % (
            r15["smallest_ratio_before"], r15["smallest_ratio_before"], r15["its_T_over_rs"]),
        # the builder's check of ASYMPTOTICS v1.8 (the lane's instruction)
        "| ASYMPTOTICS_checks_v1.8.md row (3)'s carried record lines (version 1.7's): \"prediction\" (lines 474 and 476), \"frozen\" (lines 482 and 487), \"the frozen last records\" (line 482), the json key path \"1 - frozen/rHm_over_rs\" (line 494) | \"prediction\" reads \"formula\", \"frozen\" reads \"held fixed\", \"the frozen last records\" reads \"the last records held fixed\"; the path at line 494 names a json key, not wording (requirements_new.txt sec. 4; engine 38.0; the lane's instruction of 2026-10-16 about 23:35 UTC); until the lead's next version |",
        # inbox #45, #46: HORIZON_single_object_v1 until its version 2
        "| HORIZON_single_object_v1.md (B, B.5 and their readouts beyond the root) | these per-channel readouts \"do not measure permanent escape or emitted positive particles\" (%s, Usable paper scope); until HORIZON version 2 |" % r46,
        "| HORIZON_single_object_v1.md (A: the identity's residual and the two entropy routes) | \"Keep the reconstruction residual separate from the stored state's purity error\" (%s); the complete-cut residual 2.50e-14 and the entropy-route difference 5.87e-12 \"are internal checks of the Loewdin-reconstructed occupied projector; the separate largest stored-covariance purity defect is 7.52e-8\" (%s; its math written plain); until HORIZON version 2 |" % (r46, r45),
        "| HORIZON_single_object_v1.md (A, the band occupation, strand R's \"outward band\": the bin pi/4 <= abs(ak) < pi/2) and ITEM7_single_object_numbers_v1.md (\"sits at abs(ak) at or above pi/4, the outward band\"; pipes written abs()) | the bin \"is an unsigned windowed momentum share, not a projection onto the guaranteed outward positive intrinsic Dirac branch; that band need not have positive Killing energy\" (%s); \"The reported second momentum quarter is a window in abs(ak), not a signed, band-resolved outward occupation\" (%s); until HORIZON version 2 and the lead's next version of ITEM7_single_object_numbers |" % (r46, r45),
        "| HORIZON_single_object_v1.md (B.5: the selected interior orbitals' summed position weight) | \"B5 propagates complete selected orbitals and retains their spatial cross terms; its summed position weight omits the remaining occupied sector and reference, not those selected cross terms. Its beyond-root readout cannot inherit the Gaussian probes' far-detector result\" (%s); until HORIZON version 2 |" % r46,
        "| HORIZON_single_object_v1.md (B: the bound-state and below-threshold readings) and ITEM7_single_object_numbers_v1.md (\"overlaps the bound states\", \"mean Killing energy below m\") | \"The finite-box 99%% localization classifier is not the infinite-exterior bound-state projector; a mean frequency below m does not exclude above-threshold spectral support\" (%s); until HORIZON version 2 and the lead's next version of ITEM7_single_object_numbers |" % r46,
        "| HORIZON_single_object_v1.md (A's number current through the outer root; B's transfer readouts) and ITEM7_single_object_numbers_v1.md (\"the model's own number current through the outer root\") | \"Fixed-cut number current, moving-cut reassignment, the complete reference term Gamma_P = P_dot + i[h, P], Killing-energy flux and Gauss charge loss remain distinct\" (%s; its math written plain); until HORIZON version 2 |" % r46,
        "| HORIZON_single_object_v1.md (A, item 3: the 4-site slab blocks) and ITEM7_single_object_numbers_v1.md | \"They are per-channel correlations, with no angular multiplicity applied, and establish neither escape nor a three-dimensional entropy coefficient\" (%s); until HORIZON version 2 |" % r45,
        # inbox #47 and #49: TEMPERATURE_solved_exterior_v1.5 until the lead's version 1.6
        "| TEMPERATURE_solved_exterior_v1.5.md (the statement, from (d)): \"equals kappa_T to within between -3.5e-09 and 1.1e-08 on the five records\" | read as the reported relative deviation kappa_0/kappa_T - 1, with -3.5e-9 and 1.1e-8 its \"rounded extrema, not a certified enclosure or an absolute slope error. The table's underlying extrema are -3.544e-9 and 1.051e-8\" (%s, item 1; %s, item 1; their math written plain); until the lead's version 1.6 |" % (r47, r49),
        "| TEMPERATURE_solved_exterior_v1.5.md (the statement, from (d)): \"so the slope's swing between the phases sits in the source factor\" | read: the source factor \"supplies the dominant change in 4GE kappa_T\"; \"The other supplied factors also change: 2GE/R_H from 1.428508 to 1.480122, and N_H from 0.921823 to 0.885221, beside the source factor's 0.959497 to 0.734071\" (%s, item 2; %s, item 2); until the lead's version 1.6 |" % (r47, r49),
        "| TEMPERATURE_solved_exterior_v1.5.md (the condition carried from version 1.4): \"the collar formula kappa_0 enters version 1.5 when obs_p26_slope_ratios_v2 lands\" | superseded: \"The delivered table is the v3 output, with v2's lower-precision display recorded in its header\" (%s, item 3; %s, item 3: TEMPERATURE_kappa0_table_v1.md); until the lead's version 1.6 |" % (r47, r49),
        # inbox #48: PAGE_solved_history_v1a until its version 2
        "| PAGE_solved_history_v1a.md (S_esc's band 3.08e-09) | the \"band adds a propagation tail and unitarity defect; it is not a certified entropy error in nats\" (%s, item 1); until PAGE_solved_history version 2 |" % r48,
        "| PAGE_solved_history_v1a.md (the printed W beside the global field-work column; the same-clock energy and collar residuals) | \"Their printed values do not establish the matched work identity\"; \"The same-clock energy and collar residuals are algebraically the same residual, not two balance tests\" (%s, item 2); until PAGE_solved_history version 2 |" % r48,
        "| PAGE_solved_history_v1a.md (the final clock's mode ledger) | \"At the final clock the mode/contact split is E_remove = 0.03099214, M_chi = -0.02637303: the mode ledger alone is not positive escaped energy\" (%s, item 2; its math written plain); until PAGE_solved_history version 2 |" % r48,
        "| PAGE_solved_history_v1a.md (the collected archive; the L_H comparators; the inferred nu) | \"Keep the actual velocity-selected archive and its possible returns distinct from an asymptotic radiation space\"; the two-channel finite-integral L_H's \"are comparators, not an observed thermal flux\"; the large inferred nu = n_out/(Gamma f_FD) > 1: \"they are not physical partner occupations\" (%s, item 3); until PAGE_solved_history version 2 |" % r48,
        "| PAGE_solved_history_v1a.md and ITEM7_fig_page_draft_v1a.png (the arrival marker; the plotted L) | \"The arrival marker uses R_collar = 16 r_s,f in its travel integral although the actual collar is 16 r_s,0\"; \"the plotted L still uses Delta T, not Delta t_inf\" (%s, item 4; its math written plain); until PAGE_solved_history version 2 (the Page builder carries these rows there) |" % r48,
    ]
    for r_ in rows:
        if r_.count("|") != 3:
            sys.exit("a pipe inside a cell: %r" % r_[:120])
    t = open(a.v119).read()
    tm = re.search(r"^# RELABEL_table_v1\.19: (.*?)\(engine spec (.*?); inbox #41; inbox #34; inbox #44\)$", t, re.M)
    if not tm:
        sys.exit("version 1.19's title is not as read")
    t = t.replace(tm.group(0), "# RELABEL_table_v1.20: %s(engine spec %s; inbox #41; inbox #34; inbox #44; 38.2 to 38.7; inbox #45 to #49)" % (tm.group(1), tm.group(2)), 1)
    pm = re.search(r"Version 1\.19 = version 1\.18 with ", t)
    if not pm:
        sys.exit("version 1.19's version line is not as read")
    t = t.replace(pm.group(0), "Version 1.20 = version 1.19 with %d rows added (engine 38.2 to 38.7, turn 46: the temperature read as version 1.5 and version 1.18's three and 1.19's four rows closed, the stress convention, the first episode's growth, HORIZON's fixed-field label, item 5 read as version 10, the capture rows with version 11's absorber; inbox #45 and #46 on HORIZON_single_object_v1 until its version 2; inbox #47 and #49 on TEMPERATURE_solved_exterior_v1.5 until the lead's version 1.6; inbox #48 on PAGE_solved_history_v1a until its version 2; the builder's composition checks: item 5 version 10's 0.105 and ASYMPTOTICS version 1.8's carried row (3) lines). Version 1.19 = version 1.18 with " % len(rows), 1)
    si = t.index("\n- Sources: ")
    t = t[:si].rstrip("\n") + "\n" + "\n".join(rows) + "\n" + t[si:]
    sm = re.search(r"\n- Sources: ([^\n]*)", t)
    t = t.replace(sm.group(0), "\n- Sources: " + sm.group(1).rstrip(".") + "; solve/specs/SPEC_final4b_engine.md sec. 38; deliveries/solve_a/TEMPERATURE_solved_exterior_v1.5.md, ITEM5_collapse_sentences_v10_text.md, ASYMPTOTICS_checks_v1.8.md, ITEM7_single_object_numbers_v1.md, PAGE_solved_history_v1a.md; deliveries/hreduce/web40f_single_object_identity.tex (inbox #45); deliveries/solve_b/review_HORIZON_single_object_v1.tex (inbox #46), review_TEMPERATURE_solved_exterior_v1_5_amendments.tex (inbox #47), review_PAGE_solved_history_v1a.tex (inbox #48), review_TEMPERATURE_solved_exterior_v1_5.tex (inbox #49); solve/results/engine/CERT3/origin_term_p37_v1.md; solve/results/engine/LAD27/fold_birth_summary_v2.md; solve/results/engine/CAP10/ (gate_a_v11_om*.json, capture_v9_t300_min_before_arrival_v1.json).", 1)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "x").write(t)
    L += ["", "### the quote checks (line breaks joined, TeX markup removed)", "", "| source | passage | found |", "|---|---|---|"]
    L += ["| %s | %s | %s |" % (k, f.replace("|", "/"), "yes" if ok else "NO") for k, f, ok in checks]
    L += ["", "- rows added: %d; no pipe inside a cell; composed: %s from %s" % (len(rows), a.out, a.v119)]
    open(a.log, "x").write("\n".join(L) + "\n")
    print("wrote %s (%d rows added) and %s" % (a.out, len(rows), a.log))


if __name__ == "__main__":
    main()
