#!/usr/bin/env python
"""compose_relabel_v1_25.py (the lane's analyst builder; engine spec 42.4 (4), 44.2 (1), 47.1 (1), 47.8 item 1; inbox #81):
version 1.24 with the title's source list, the version line and the sources line extended and the rows below added before the sources
line; the rows version 1.24 left open stay as they are. Every quoted string is asserted present in its file.
Args: --v124 --page3 --para2 --figsrc --figcode --scol1 --out."""
import argparse

REL = ("the raw selector's (page_history_v6b), which web40g's P02 reading finds rejects directions whose archive-removed remnant is "
       "entirely outside; superseded by version 3b")


def main():
    ap = argparse.ArgumentParser()
    for k in ("v124", "page3", "para2", "figsrc", "figcode", "scol1", "out"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    v24 = open(a.v124).read(); p3 = open(a.page3).read(); pa = open(a.para2).read()
    fs = open(a.figsrc).read(); fc = open(a.figcode).read(); s1 = open(a.scol1).read()
    for q, txt in (("| S_esc | band | S_vac (archive) | S_esc - S_vac |", p3), ("Readout 10: the largest S_esc 0.001147 nats", p3),
                   ("## 9. The variant attribution of S_col at T/r_s 40", p3), ("S_col(40) 0.00114744 nats", p3),
                   ("The selected archive's entropy S_col (the velocity-selected, co-propagating collected modes", fc),
                   ("0.00114744", pa), ("the size of the omitted feedback", fs), ("geometry's work on the sea", pa), ("outside weight 0.9935 to 0.9999", fs),
                   ("6.55", s1)):
        assert q in txt, q
    rows = [
        "| PAGE_solved_history_v3.md sec. 1 (and 1b, 2, 4): the S_esc, S_vac (archive), S_esc - S_vac, S_B, S(M) - S(M'), collected and archive counts and outside-weight-minimum columns | " + REL + " (engine 47.1 (1)); the corrected collector is the generalized selector (page_history_v6j --collector-select gen), whose archive prints only after 47.1 (2)'s checks (i) to (vii); open until PAGE_solved_history version 3b |",
        "| PAGE_solved_history_v3.md readout 10: \"the largest S_esc 0.001147 nats ... against eta A/a^2 32572.9: 3.5227e-08; against A/(4G) 35436.1: 3.2381e-08\" | " + REL + " (engine 47.1 (1)); the ratios are the raw selector's record and print nowhere; open until version 3b |",
        "| PAGE_solved_history_v3.md sec. 9: the variant attribution of S_col at T/r_s 40 (S_col(40) 0.00114744 nats; the spread 0.000664845 to 0.00146431) | " + REL + "; the variants changed the same defective selection, so the spread is not the digits' error (engine 47.1 (1); inbox #81 (c)); open until the variants rerun under the corrected selector (47.1 (2) (vi)) and version 3b |",
        "| ITEM7_fig_page_v4.png, its headline \"The selected archive's entropy S_col (the velocity-selected, co-propagating collected modes by the retained-channel recursion; collected modes' outside weight 0.9935 to 0.9999 per block)\" and its top panel | " + REL + "; item 7 prints not this figure until version 3b (engine 47.1 (1); inbox #81 correction 1) |",
        "| ITEM7_solved_history_paragraph_v2.md: the sentence \"The selected archive's entropy S_col ... reaches 0.00114744 nats at 40 and has not fallen or saturated\", its spread and the outside-weight range | " + REL + "; item 7 prints no selected-archive digit, old or new, until version 3b; it prints the theorem sentences, the window sentence, the marks and, under their labels and after their second route, version 3's collector-independent readouts (the collar flow, E_inv, L/L_H, E_rad^(0)/E 4.3417e-05, the controls' drain readings) (engine 47.1 (1)) |",
        "| ITEM7_fig_page_v4's caption and label and PAGE v3's label (the feedback clause), ITEM7_solved_history_paragraph_v2.md (the remaining clauses): \"E_rad^(0)/E = 4.3417e-05 ..., the size of the omitted feedback\", \"moving with the geometry's work on the sea\", the flux, held-field, band-top, join-check and D16 clauses | the cold reading's corrections 2 to 8 (deliveries/busi/ReviewA_cold_reading_PAGE_solved_history_v3.md sec. (f); inbox #81): NOT CLEARED as filled; the net signed collar flow through 40 over E(40) is a diagnostic, not the size of the omitted feedback; open until version 3b |",
        "| SCOL_attribution_v1.md: the parenthetical carrying #62's figure (the kappa 3 guard's share 6.55) | prints as no value (engine 44.2 (1)); traced in SCOL_attribution_v1b as the guard's weighted E_inv at T/r_s 14 against the charge; closed by version 1b; S_col rows of v1 and v1b are the raw selector's (row above) until version 1c |",
        "| INTERIOR_core_statement_v2.md | withdrawn from print; a record of the same numbers; the paper's item 5 keeps icloud's replacement (commit b6fba001eb) (engine 42.4 (3)) |",
        "| COLLAPSE_certificate_statement_v1.3: \"the frozen records 35.375 and 37.75\", \"windows\" in the head, \"the late window\" | \"the records 35.375 and 37.75 with the fields held fixed\", \"intervals\", \"the late interval\" (engine 42.4 (2)); open until version 1.4 |",
        "| every S11 output (MASSLESS_ball_pathfinder files) from T/r_s 20.125 | carries the restart's label (busi 5.2): S11 restarted from its slice at 20.125 after its gate (E/E0 within 1.75e-08, min(N - |v|) within 1.86e-08, masks 10 of 10 on 20.125 to 20.625) (engine 42.5 (2)); open until MASSLESS_ball_pathfinder_v1's final compose |",
        "| the ell 2a member's \"no formation\" reading and the ell 4a member's records | \"no trapping zero through T/r_s 10.6, the lowest min(N - |v|) 0.035, on a member whose ell/2GE 0.25 lies between the critical widths 0.2282 (with source averaging) and 0.2609: a 'no formation' to 40 prints as the width theorem's statement where 0.2282 applies and not as a scale dependence; the requirements' scale control of the massless object is the r_s/a 32 pair's\" (engine 42.5 (2)); the ell 4a member on engine_collapse_v36b --pair-workers 2 cites the v36b gate record (results/engine/V36b/) in its label (47.4 (1)) |",
    ]
    head, src = v24.rsplit("\n- Sources:", 1)
    title_old = "41.5 (3); inbox #57)"
    assert title_old in head.split("\n", 1)[0], "title"
    head = head.replace("RELABEL_table_v1.24", "RELABEL_table_v1.25", 1).replace(title_old, "41.5 (3); inbox #57; 42.4 (4), 44.2 (1), 47.1 (1); inbox #81)", 1)
    vl = "Version 1.24 = version 1.23"
    assert vl in head
    head = head.replace(vl, "Version 1.25 = version 1.24 with 11 rows added (engine 47.1 (1): the raw selector's S_col columns, readout 10, sec. 9, the figure's headline and the paragraph's archive sentence, superseded by version 3b; inbox #81's corrections 2 to 8 on the paragraph as filled; 44.2 (1): SCOL v1's parenthetical; 42.4 (4): INTERIOR v2, version 1.3's four words, the S11 restart label, the ell 4a member's reading). " + vl, 1)
    out = head.rstrip("\n") + "\n" + "\n".join(rows) + "\n\n- Sources:" + src.rstrip("\n") + "; solve/specs/SPEC_final4b_engine.md secs. 42.4, 42.5, 44.2, 47.1; deliveries/busi/ReviewA_cold_reading_PAGE_solved_history_v3.md; deliveries/solve_a/PAGE_solved_history_v3.md, ITEM7_solved_history_paragraph_v2.md, ITEM7_fig_page_v4_SOURCES.txt, SCOL_attribution_v1.md\n"
    open(a.out, "w").write(out)
    print("rows added", len(rows))


if __name__ == "__main__":
    main()
