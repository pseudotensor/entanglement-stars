#!/usr/bin/env python
"""compose_relabel_v1_23.py (the lane's analyst builder; engine spec 40.5 (6), the lead's turn 48: "RELABEL_table_v1.23: the v1.22 hold
closed by v3; #55's five bullets and its closing limits as rows on HORIZON v2 until version 3 (composed when the second transfer route
lands); TEMPERATURE v1.6's head tail reads "versions 1.4 and 1.5 stay the record" (no version 1.7 for it alone); the figure's title
(closed by the figure's version 3); the gate (b) label (closed by ASYMPTOTICS 1.10); the S11 Page outputs' label (closed by the final
compose); the shell-0 line (closed by v1.4); the ASYMPTOTICS row open until v1.9 lands"): version 1.22 with the title's source list, the
version line and the sources line extended and 12 rows added before the sources line. The ASYMPTOTICS row is closed here because
ASYMPTOTICS_checks_v1.9 is delivered (engine_asymptotics_table_v13b) before this table. Every quoted string is asserted present in its
file. Args: --v122 --numbers3 --horizon2 --temp16 --fig2src --out."""
import argparse
import re


def main():
    ap = argparse.ArgumentParser()
    for k in ("v122", "numbers3", "horizon2", "temp16", "fig2src", "out"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    num3 = open(a.numbers3).read(); hz = open(a.horizon2).read(); t16 = open(a.temp16).read(); f2 = open(a.fig2src).read()
    assert num3.count("which are neither a transfer beyond it nor emitted particles") == 1
    assert t16.count("version 1.4 stays the record") == 1
    assert "## B. Transfer by wavelength and depth" in hz
    v22 = open(a.v122).read()
    assert v22.count("correlations across the root, not escape and not emitted particles") == 1
    asym_row = [ln for ln in v22.split("\n") if ln.startswith("| ASYMPTOTICS_checks_v1.8.md row (3)'s carried record lines")]
    assert len(asym_row) == 1
    rows = [
        "| version 1.22's row holding ITEM7_single_object_numbers_v2's printable paragraph on its closing words | closed: ITEM7_single_object_numbers_v3.md (engine 40.5 (5)) reads them \"correlations across the root, which are neither a transfer beyond it nor emitted particles\"; its printable paragraph is released (two routes, inbox #51); its record paragraph stays held (one route, inbox #55) |",
        "| version 1.21's row on ASYMPTOTICS_checks_v1.8.md row (3)'s carried record lines | closed: ASYMPTOTICS_checks_v1.9.md (engine 39.3 (4), 40.5 (2); engine_asymptotics_table_v13b with ASYMPTOTICS_row3_v1.9_replacements_v2.md) carries the replacements; \"the predicted shift\" at its lines 483 and 488 stands (requirements_new.txt sec. 6's own words) |",
        "| HORIZON_single_object_v2.md, B's end-region columns (six entries per launch) | read in order: beyond R_t, beyond 2GE, beyond 16 r_s, collar 16 to 24 r_s, far collar 32 to 48 r_s, beyond 24 r_s; the third label is missing in the file's header (inbox #55); open until HORIZON version 3 |",
        "| HORIZON_single_object_v2.md, the 32-site tapered signed weights (definition (vi) and every signed column) | positive windowed effects resolving a local positive intrinsic Dirac band by group-velocity sign, not global spectral projectors or fluxes; the 16- and 64-site controls are sensitivity tests; the unsigned Fourier share stays separate; intrinsic positivity does not require positive Killing energy (inbox #55); open until HORIZON version 3 |",
        "| HORIZON_single_object_v2.md, E's beyond-16 r_s column (the full detector density) | the dust's, not literal direct C - P (at the production's T/r_s 15, kappa 1: 1.24443e-16 and -1.39153e-10); exact sea/reference number cancellation requires paired half-filled states; the storage and reconstruction residuals are kept; it does not cancel the sea-reference energy or all resolved particle and hole occupations (inbox #55); open until HORIZON version 3 |",
        "| HORIZON_single_object_v2.md, the five rates and the number ledger (definition (ix), E, B.5) | the ledger combines the fixed-cut flux plus the membership change minus the complete reference term Tr Pi_Y (P_dot + i[h, P]); E's reference term is a neighbouring-rebuilt-sheet finite-difference estimator, not exact for the rounded stored R2; B.5's step increments, E's old-state membership rate and its continuous -rho dR_t/dt diagnostic have different conventions; rates per lattice time, row clocks nominal T/r_s; regional energy also needs field, reference and cut work; none of these channel quantities is the whole Gauss-charge loss (inbox #55); open until HORIZON version 3 |",
        "| HORIZON_single_object_v2.md, the comparators, arrivals and delays (definition (i)) | the outside comparators have the same carrier and envelope at different locations, not matched Killing energy; the arrival is the first crossing of half of each column's own terminal pre-flag beyond-16 r_s population and the delay its difference from that comparator's; the printed logarithms use the absolute frequency or carrier on their guarded domains, the launch 2GE distinct from the nominal r_s; not universal delay laws (inbox #55); open until HORIZON version 3 |",
        "| HORIZON_single_object_v2.md, any reading of exterior weight as transfer | the initial exterior tails, coherent cross blocks and moving-root membership are retained, so exterior population is not automatically transfer of an initially inside projected state; the production's B ends at T/r_s 47.5, B.5 at 48.125, the other preparation at 40; the fixed-field 300 r_s endpoints include finite-wall reflections; the 99 percent localized-eigenvector classifier is not an infinite-exterior bound-state projector; the stored purity and the reconstructed projector's identity residuals stay distinct; no permanent escape, flux to infinity, closed charge-loss ledger, physical Page turnover or completed-history no-horizon result follows (inbox #55's closing limits); open until HORIZON version 3 (composed when the second transfer route lands) |",
        "| TEMPERATURE_solved_exterior_v1.6.md, the head's tail \"version 1.4 stays the record\" | reads \"versions 1.4 and 1.5 stay the record\" (engine 40.5 (6); no version 1.7 for it alone) |",
        "| ITEM7_fig_page_draft_v2.png, its title \"the collapse epoch's escaping entropy and flux\" and its axis \"S_esc\" | closed: the figure's version 3 (ITEM7_fig_page_draft_v3.png, engine 40.2 (3), inbox #56) reads \"S_col, the collected sector's entropy by the retained-channel recursion (collected modes' outside weight 0.9935 to 0.9999 per block)\", the axis S_col, with the coordinate and flux qualifications on the figure; version 2's figure stays the record |",
        "| capture version 9's numbers wherever they print beside version 11's (ASYMPTOTICS row (6), ITEM5's capture clause) | carry \"version 11 within 6.9e-04 of sigma of version 9 at 40 r_s, the launched tails' outward part removed by the absorber\" (engine 40.5 (1), gate (b)); open until ASYMPTOTICS_checks_v1.10 |",
        "| S11's Page outputs (page_history_v6i on the massless ball's records): the codes' label \"the solved history of the massive collapse\" | reads as naming the m = 0 ball, the self-consistent massless collapse at r_s/a 16 (engine 40.1 (3)); open until MASSLESS_ball_pathfinder_v1's final compose |",
        "| COLLAPSE_certificate_statement_v1.3.md, the innermost shell's stress convention | the shell-0 convention check is a record: changing the innermost shell's convention moves its stress by up to 2.7 times the record's largest absolute S_r and the smoothed radial stress by up to 60 percent on the support, and moves no certified record across the criterion (0 flips of 281; engine 40.5 (4)); open until COLLAPSE_certificate_statement_v1.4 |",
    ]
    for r in rows:
        assert r.count("|") == 3, r[:60]
    t = v22
    tm = re.search(r"^# RELABEL_table_v1\.22: (.*?)\(engine spec (.*?)\)$", t, re.M)
    assert tm
    t = t.replace(tm.group(0), "# RELABEL_table_v1.23: %s(engine spec %s; 40.1 (3), 40.2 (3), 40.5 (1) to (6); inbox #55, #56)" % (tm.group(1), tm.group(2)), 1)
    pm = re.search(r"Version 1\.22 = version 1\.21 with ", t)
    assert pm
    t = t.replace(pm.group(0), "Version 1.23 = version 1.22 with %d rows added (engine 40.5 (6), the lead's turn 48: version 1.22's hold closed by ITEM7_single_object_numbers_v3; version 1.21's ASYMPTOTICS row closed by ASYMPTOTICS_checks_v1.9; inbox #55's five labels and its closing limits as rows on HORIZON_single_object_v2 until version 3; TEMPERATURE v1.6's head tail; the figure's title closed by its version 3; the gate (b) label until ASYMPTOTICS 1.10; the S11 Page outputs' label until the final compose; the shell-0 line until the certificate statement v1.4). Version 1.22 = version 1.21 with " % len(rows), 1)
    si = t.index("\n- Sources: ")
    t = t[:si].rstrip("\n") + "\n" + "\n".join(rows) + "\n" + t[si:]
    sm = re.search(r"\n- Sources: ([^\n]*)", t)
    assert sm
    t = t.replace(sm.group(0), "\n- Sources: " + sm.group(1).rstrip(".") + "; solve/specs/SPEC_final4b_engine.md sec. 40; deliveries/solve_b/review_HORIZON_single_object_v2.tex (inbox #55) and review_PAGE_solved_history_v2.tex (inbox #56); deliveries/solve_a/ITEM7_single_object_numbers_v3.md, ASYMPTOTICS_checks_v1.9.md, ITEM7_fig_page_draft_v3.png, HORIZON_single_object_v2.md, TEMPERATURE_solved_exterior_v1.6.md.", 1)
    low = "\n".join(rows).lower()
    for w in ("frozen", "freez", "prediction"):
        assert w not in low, w
    open(a.out, "x").write(t)
    print("wrote %s (%d rows added)" % (a.out, len(rows)))


if __name__ == "__main__":
    main()
