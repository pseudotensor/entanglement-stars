#!/usr/bin/env python
"""item7_fig_page_draft_v6.py (the lane's analyst builder; engine spec 41.4 (1)): the figure version 4, item7_fig_page_draft_v5.py with the title of 41.4 (1) verbatim; nothing else changed. v5's docstring: item7_fig_page_draft_v5.py: the figure version 3, item7_fig_page_draft_v4.py with the title
the lead's text of 40.2 (3), the axis S_col, the coordinate and flux qualifications and the marks' meanings (#56) on the figure itself, and the
definition, #48's corrections and #56's qualifications in the SOURCES file; the data and the marks unchanged. v4's docstring follows.
item7_fig_page_draft_v4.py (builder analyst, 2026-10-17): item7_fig_page_draft_v3.py with the run's label printed whole under the plot,
wrapped to the figure's width (v3 cut it at 230 characters and its one line ran past the right edge). Nothing else changes.

v3's docstring: item7_fig_page_draft_v3.py (builder analyst, 2026-10-16; engine spec 38.1 (2), inbox #48): item7_fig_page_draft_v2.py with the axis labels of
icloud's check of v1a: L is per checkpoint interval Delta T (the coordinate clock), not Delta t_inf; L_H a comparator; the collar's clock the
retarded clock in each record's stationary chart (not a solved common arrival map). The SOURCES file prints every definition of the json's
figure block (the marks, L, T_d's imported premises). v2 drew the version 1a figure; version 2's is drawn by this file.

v2's docstring: item7_fig_page_draft_v2.py (builder analyst, 2026-10-16; engine spec 38.1 (2)): item7_fig_page_draft_v1.py with the draft label in the
title and two sets of marks, read from the summary json's "figure" block (nothing is computed here; colours black, red, blue):
  --marks v1  version 1's: the formation's record clock, the formation plus T_d, the formation's light-cone arrival from 2 x 2gE;
  --marks v2  38.1 (2)'s: the axis the collar's clock; the formation's record clock a reference line; the formation's arrival at the collar the
              measured half-collar-flow clock, with the light-travel range from 2 and 1.5 times 2gE as a shaded band; the ray-map epoch's end
              at the collar as that arrival plus T_d (the formation plus T_d is a surface clock and is not drawn).
The SOURCES file carries the definitions as the json gives them. Usage:
python item7_fig_page_draft_v5.py SUMMARY_JSON OUT_PNG OUT_SOURCES --marks v2"""
import datetime
import hashlib
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402


def main():
    args = sys.argv[1:]
    mode = "v2"
    if "--marks" in args:
        i = args.index("--marks"); mode = args[i + 1]; args = args[:i] + args[i + 2:]
    p_json, p_png, p_src = args[:3]
    raw = open(p_json, "rb").read()
    J = json.loads(raw)
    F = J["figure"]
    x = F["t_inf_over_rs"]; S = F["S_esc"]; mk = F["marks"]
    lab_draft = F.get("draft_label") or J.get("draft_label")
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.0, 6.6), sharex=True)
    a1.plot(x, S, "k-o", ms=3, lw=1.2, label="S_col (the selected archive's entropy, weighted, nats)")
    a1.set_ylabel("S_col (nats)")
    for key, val, col in (("L_over_LH_kappaT", "L/L_H at kappa_T", "r"), ("L_over_LH_TH", "L/L_H at T_H = 1/(4 pi 2gE)", "b")):
        pts = [(xx, abs(v), v < 0) for xx, v in zip(x, F[key]) if v is not None and v != 0]
        if pts:
            a2.semilogy([p[0] for p in pts], [p[1] for p in pts], col + "-", lw=1.0, label=val + " (|.|; open: negative)")
            a2.semilogy([p[0] for p in pts if not p[2]], [p[1] for p in pts if not p[2]], col + "o", ms=3)
            neg = [p for p in pts if p[2]]
            if neg:
                a2.semilogy([p[0] for p in neg], [p[1] for p in neg], col + "o", ms=4, mfc="none")
    a2.set_ylabel("|L| / L_H\n(L: collar energy per coordinate interval Delta T,\nnot per a solved common arrival clock; L_H a comparator)", fontsize=7)
    a2.set_xlabel("t_inf / r_s at the collar: a recordwise stationary-chart substitution\n(each record's retarded clock at R_collar), not a solved common arrival map", fontsize=7)
    lines = []
    if mode == "v1":
        lines = [(mk.get("formation_t_inf_over_rs"), "k", "--", "formation T/r_s %s" % mk.get("formation_T_over_rs")),
                 (mk.get("formation_plus_T_d_t_inf_over_rs"), "b", "-.", "formation + T_d (T_d/r_s %s)" % mk.get("T_d_over_rs")),
                 (mk.get("arrival_t_inf_over_rs"), "r", ":", "the formation's arrival at the collar (light travel from 2 x 2gE)")]
    else:
        v2 = mk.get("v2") or {}
        lines = [(v2.get("formation_record_clock_over_rs"), "k", "--", "the formation's record clock T/r_s %s (reference)" % v2.get("formation_record_clock_over_rs")),
                 (v2.get("arrival_half_flow_t_inf_over_rs"), "r", "-", "the half-collar-flow clock (halves the net signed flow through T/r_s 40)"),
                 (v2.get("raymap_end_t_inf_over_rs"), "b", "-.", "arrival + T_d (T_d/r_s %s; a leading T_d term, matching constant unfixed; not a measured end)" % v2.get("T_d_over_rs"))]
        band = v2.get("light_travel_band_t_inf_over_rs")
        if band and all(b is not None for b in band):
            for ax in (a1, a2):
                ax.axvspan(min(band), max(band), color="r", alpha=0.12, lw=0, label="fixed-charge flight comparison from 2 and 1.5 x 2gE" if ax is a1 else None)
    for v, col, ls, lab in lines:
        if v is not None:
            for ax in (a1, a2):
                ax.axvline(v, color=col, ls=ls, lw=1.0, label=lab if ax is a1 else None)
    a1.legend(fontsize=6.5, loc="upper left"); a2.legend(fontsize=6.5, loc="lower right")
    title = "The selected archive's entropy S_col (the velocity-selected, co-propagating collected modes by the retained-channel recursion; collected modes' outside weight 0.9935 to 0.9999 per block)"   # engine spec 41.4 (1), the lead's title verbatim
    import textwrap
    if lab_draft:
        title += "\n" + "\n".join(textwrap.wrap(lab_draft, 92))
    a1.set_title("\n".join(textwrap.wrap(title.split("\n")[0], 100) + title.split("\n")[1:]), fontsize=7.5)
    import textwrap
    QUAL = ("The massless test channel (D14) on the stored massive collapse, sources not fed back. S_col: the coherent entropy of the velocity-selected, co-propagating "
            "archive collected at the collar, not established as the asymptotic radiation space. The clock: a recordwise stationary-chart substitution. The flux: collar "
            "energy per coordinate interval Delta T; L_H comparators, not an observed thermal flux. The marks: the half-collar-flow clock halves only the net signed flow "
            "available through T/r_s 40; the shaded band is a fixed-charge flight comparison; arrival + T_d adds a leading T_d term with an unfixed matching constant. "
            "The run's E_rad^(0)/E is the net selected-channel collar flow at 16 r_s through T/r_s 40 over the final charge, a diagnostic only. "
            "No Page curve, evaporation, thermal emission or partner return is claimed. The run label (the json's) is in the SOURCES file.")
    foot = textwrap.wrap(QUAL, 175)
    fig.text(0.01, 0.004, "\n".join(foot), fontsize=5, va="bottom")
    fig.tight_layout(rect=(0, 0.012 + 0.011 * len(foot), 1, 1))
    fig.savefig(p_png, dpi=160)
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    with open(p_src, "w") as f:
        f.write("ITEM7 figure sources (written %s by code/page/item7_fig_page_draft_v6.py (the figure version 4, title of engine spec 41.4 (1)), marks %s)\n" % (now, mode))
        if lab_draft:
            f.write("draft label: %s\n" % lab_draft)
        f.write("data: %s (md5 %s), its figure block, copied from the run summaries by the composer\n" % (p_json, hashlib.md5(raw).hexdigest()))
        f.write("run inputs (from the summary json's inputs): %s\n" % json.dumps({k: v for k, v in (J.get("inputs") or {}).items() if v.get("landed")}))
        f.write("x: the collar's clock t_inf_collar / 64 (the summary rows' retarded clock at R_collar, from the run's first record)\n")
        f.write("top: S_esc_weighted (the direct route on the cumulative collected set; band %s at the end)\n" % (F["band"][-1] if F.get("band") else "-"))
        f.write("bottom: |L_interval| / L_H for both comparators (L_H(kappa_T): the record's dense root and slope, which differ from the applied chain root; L_H(T_H): 1/(4 pi 2gE)); L_interval is the collar energy flow over each checkpoint interval divided by its coordinate length Delta T, not Delta t_inf; the L_H are comparators (the two-channel continuum greybody over a finite range), not an observed thermal flux; open markers: L < 0\n")
        f.write("band: the propagation's Chebyshev tail plus the sea's unitarity deviation, not a certified entropy error in nats\n")
        f.write("definition (engine spec 40.2 (3)): S_col is the collected sector's entropy by the retained-channel recursion, with the collected modes' outside weight 0.9935 to 0.9999 per block (38.1); the requirements' name 'escaping entropy' is used for it only in that labelled form (#27, #48); plotted from the json's S_esc_weighted column\n")
        f.write("#48's corrections (39.1 (3)): (1) the band a propagation tail, not a certified error; (2) E_inv on the instantaneous P_-; (3) W and the field-work column not a matched identity; (4) E_remove 0.03099214 and M_chi -0.02637303 side by side (the mode ledger alone is not positive escaped energy); the archive, comparators and nu labels; the arrival band 35.570 to 37.488 on the collar's clock with the half-flow arrival 36.214 inside it; L per Delta T\n")
        f.write("#56's qualifications (icloud's R2 read of version 2): the selected archive's entropy (velocity-selected, co-propagating; not the asymptotic radiation space); the x coordinate a recordwise stationary-chart substitution; the flux energy per Delta T, not per a solved common arrival clock; the 36.2143 mark halves only the net signed flow available through 40; 35.5703 to 37.4881 a fixed-charge flight comparison; the 38.5243 mark adds a leading T_d term with an unfixed matching constant, not a measured endpoint\n")
        if mode == "v1":
            for k_ in ("formation_T_over_rs", "formation_t_inf_over_rs", "T_d_over_rs", "T_d_source", "formation_plus_T_d_t_inf_over_rs", "arrival_t_over_rs",
                       "arrival_t_inf_over_rs", "arrival_travel_over_2gE", "two_gE_formation", "arrival_definition", "half_collar_flow_t_over_rs", "half_collar_flow_definition"):
                f.write("mark %s: %s\n" % (k_, mk.get(k_)))
        else:
            for k_, v_ in (mk.get("v2") or {}).items():
                f.write("mark %s: %s\n" % (k_, v_))
        f.write("label: %s\n" % F.get("label"))
    print("wrote %s and %s (marks %s)" % (p_png, p_src, mode))


if __name__ == "__main__":
    main()
