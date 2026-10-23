#!/usr/bin/env python
"""compose_relabel_v1_24.py (the lane's analyst builder; engine spec 41.5 (3), the lead's turn 49: "RELABEL_table_v1.24: #56's five corrections
on PAGE v2 and the filled sentence (closed by version 3); the figure's title (closed by version 4); the clause "entirely above 25 T_H" in every
PAGE version carrying it (closed by version 3); D's outputs from 58; INTERIOR v1's four identifications (closed by FILE 1, delivered); the S11
outputs' transport label (closed by the final compose); the rows v1.23 left open"; and 41.4 (3): "ITEM7_solved_history_sentence_v1 does not
print; a RELABEL row until version 3"): version 1.23 with the title's source list, the version line and the sources line extended and the rows
below added before the sources line; the rows version 1.23 left open stay as they are. Every quoted string is asserted present in its file.
Args: --v123 --page2 --sentence1 --interior1 --out."""
import argparse
import re


def main():
    ap = argparse.ArgumentParser()
    for k in ("v123", "page2", "sentence1", "interior1", "out"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    v23 = open(a.v123).read(); sen = open(a.sentence1).read(); i1 = open(a.interior1).read(); p2 = open(a.page2).read()
    for q, txt in (("9.8 percent of E over the record pair 12 to 14", sen), ("entirely above 25 T_H by energy", sen), ("the energy identity to 8.1e-11", sen),
                   ("largest averaged density", i1), ("reaches no saturated core", i1), ("From the formation on, the averaging length is no longer small", i1),
                   ("the slice where the hydrodynamic description ends, at the formation", i1)):
        assert q in txt, q
    rows = [
        "| PAGE_solved_history_v2.md and ITEM7_solved_history_sentence_v1.md: \"9.8 percent of E over the record pair 12 to 14\" and \"the energy identity to 8.1e-11\" | E_inv(14)/E(14) against the instantaneous Killing ground state, not the 12 to 14 increment or a localized formation emission; the 8.1e-11 is a same-clock energy decomposition residual, not matched field or reference work; the accumulated field-jump work is supplied for the v6f controls only (inbox #56 item 1); open until PAGE_solved_history version 3 |",
        "| every PAGE_solved_history version and ITEM7 sentence carrying \"entirely above 25 T_H\" (by energy) | struck: the hard-share routine weights its overflow bin [40, 1e30) T_H by its midpoint, about 5e29 T_H, so its unit share establishes no hard-energy dominance; the collected bins print as bins (inbox #56 item 2; engine 41.0 (i)); open until PAGE_solved_history version 3 |",
        "| PAGE_solved_history_v2.md and its figure: the collar's clock, the flux L and the marks 36.2143, 35.5703 to 37.4881 and 38.5243 | the plotted coordinate is a recordwise stationary-chart substitution and the flux energy per coordinate interval Delta T, not per a solved common arrival clock; 36.2143 halves only the net signed flow available through 40; 35.5703 to 37.4881 is a fixed-charge flight comparison; 38.5243 adds a leading T_d term with an unfixed matching constant, not a measured endpoint (inbox #56 item 3); open until PAGE_solved_history version 3 |",
        "| PAGE_solved_history_v2.md: the fixed-field controls' fractions 0.091 and 0.76 of the held excitation and their late flux ranges | signed whole-hold collar totals including 37 rows after each chain's reflection clock, not pre-return fractions; the full pre-marker flux ranges are 5.36975 to 6.26798 (c15) and -0.0171821 to 0.0756697 (c30) times the comparator; both controls conserve the whole-chain excitation and fail their entropy and flux drain rule; no universal emptying time and no completed drain follows (inbox #56 item 4; engine 41.0 (iii)); open until PAGE_solved_history version 3 |",
        "| PAGE_solved_history_v2.md: the kept transfer ledger's W_R = 0 | the ledger's last restart and its end both fall at T/r_s 40, so W_R = 0 has zero elapsed transfer time; its residual W_H is not an independently measured positive port; the history to 100 and its anchored ledger are pending in version 2 (inbox #56 item 5); open until PAGE_solved_history version 3 |",
        "| PAGE_solved_history_v2.md and ITEM7_solved_history_sentence_v1.md: \"E_rad^(0)/E\" | the net selected-channel collar flow at 16 r_s through T/r_s 40 divided by the final charge (0.00461114474455826 / 106.20577526191055), a diagnostic; positive permanently escaped energy, charge loss and a bound on the omitted feedback are not established (inbox #56) |",
        "| ITEM7_solved_history_sentence_v1.md | does not print (engine 41.4 (3)); item 7's paragraph is ITEM7_solved_history_paragraph_v2_template.md filled from PAGE version 3; open until version 3 |",
        "| ITEM7_fig_page_draft_v3.png's title \"S_col, the collected sector's entropy by the retained-channel recursion (collected modes' outside weight 0.9935 to 0.9999 per block)\" | reads \"The selected archive's entropy S_col (the velocity-selected, co-propagating collected modes by the retained-channel recursion; collected modes' outside weight 0.9935 to 0.9999 per block)\", the caption carrying #56's five qualifications (engine 41.4 (1)); version 3 is R2's reading copy and prints nowhere; open until the figure's version 4 |",
        "| D's outputs on every continuation slice from T/r_s 58 (D_v2, D_v5 and every later D table) | labelled \"after the chain end's first return\" (D_return_marker_v2: the incoming positive-band energy beyond 24 r_s positive and above 1 percent of the outgoing at every window, the outgoing above 1e-3, first at 58); the paper's D numbers are those at or before 56 (engine 41.2 (1)) |",
        "| INTERIOR_core_statement_v1.md: \"the core's largest averaged density\" | closed: INTERIOR_core_statement_v2.md (delivered) reads the unaveraged radial-channel site density against the cubic exclusion ceiling, a comparator (inbox #57) |",
        "| INTERIOR_core_statement_v1.md: \"the collapse reaches no saturated core\" | closed: INTERIOR_core_statement_v2.md (delivered); the density comparison supplies no actual cubic saturation test (inbox #57) |",
        "| INTERIOR_core_statement_v1.md: \"From the formation on, the averaging length is no longer small\" | closed: INTERIOR_core_statement_v2.md (delivered): the six-term geometry diagnostic exceeds the 0.1 gate on every sampled slice, already on the root-free slice 12; the samples do not locate its first crossing (inbox #57; engine 41.0 (ii)) |",
        "| INTERIOR_core_statement_v1.md: \"the slice where the hydrodynamic description ends, at the formation\" | closed: INTERIOR_core_statement_v2.md (delivered); no formation endpoint is identified by the samples (inbox #57; engine 41.0 (ii)) |",
        "| every S11 output (MASSLESS_ball_pathfinder files, s11_pathfinder_report_v3's tables) | carries \"LZ convention from 0.125; over 1.125 to 1.5 the certified eigenvalue-route transport (31 to 46 pieces per channel, 0 failing, smallest margin 4.11e-06) reproduces these records bitwise and the full route reproduces them within the floor\" (engine 41.1 (1)); open until MASSLESS_ball_pathfinder_v1's final compose |",
    ]
    for r in rows:
        assert r.count("|") == 3, r[:60]
    t = v23
    tm = re.search(r"^# RELABEL_table_v1\.23: (.*?)\(engine spec (.*?)\)$", t, re.M)
    assert tm
    t = t.replace(tm.group(0), "# RELABEL_table_v1.24: %s(engine spec %s; 41.1 (1), 41.2 (1), 41.4 (1), (3), 41.5 (3); inbox #57)" % (tm.group(1), tm.group(2)), 1)
    pm = re.search(r"Version 1\.23 = version 1\.22 with ", t)
    assert pm
    t = t.replace(pm.group(0), "Version 1.24 = version 1.23 with %d rows added (engine 41.5 (3), the lead's turn 49: inbox #56's corrections on PAGE v2 and the filled sentence until version 3; the clause on 25 T_H struck; E_rad^(0)/E's meaning; the filled sentence not printing; the figure's title until version 4; D's outputs from 58; INTERIOR v1's four identifications closed by version 2; the S11 outputs' transport label until the final compose). Version 1.23 = version 1.22 with " % len(rows), 1)
    si = t.index("\n- Sources: ")
    t = t[:si].rstrip("\n") + "\n" + "\n".join(rows) + "\n" + t[si:]
    sm = re.search(r"\n- Sources: ([^\n]*)", t)
    assert sm
    t = t.replace(sm.group(0), "\n- Sources: " + sm.group(1).rstrip(".") + "; solve/specs/SPEC_final4b_engine.md sec. 41; deliveries/solve_b/review_INTERIOR_core_statement_v1.tex (inbox #57); deliveries/solve_a/INTERIOR_core_statement_v1.md and _v2.md, ITEM7_solved_history_sentence_v1.md, PAGE_solved_history_v2.md; solve/results/engine/Dmassive_v5/D_return_marker_v2.md.", 1)
    low = "\n".join(rows).lower()
    for w in ("frozen", "freez", "prediction"):
        assert w not in low, w
    open(a.out, "x").write(t)
    print("wrote %s (%d rows added)" % (a.out, len(rows)))


if __name__ == "__main__":
    main()
