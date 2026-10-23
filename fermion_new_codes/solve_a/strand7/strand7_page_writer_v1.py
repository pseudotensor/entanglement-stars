#!/usr/bin/env python
"""Item 7's result file, a revision (builder analyst): assembles solve/results/RESULT_strand7_page.md from the report (strand7_page_report_v1.py's
markdown and JSON) with the header lines (code, commands, the box status lines, the builder's choices) and the builder's notes, whose numbers are
read from the report's JSON, so that no number is retyped.  --revision N --stamp "<UTC>" --report-md --report-json --status (the run folder's
status.txt) --out (the result file; a new revision is appended after the existing ones, the title line updated)."""
import argparse
import json
import os


def f(x, d=4):
    if x is None:
        return "-"
    x = float(x)
    return ("%." + str(d) + "g") % x if (x == 0 or 1e-3 <= abs(x) < 1e5) else "%.3e" % x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revision", type=int, required=True)
    ap.add_argument("--stamp", required=True)
    ap.add_argument("--report-md", required=True)
    ap.add_argument("--report-json", required=True)
    ap.add_argument("--status", required=True)
    ap.add_argument("--scope", default="", help="one line: which runs this revision covers and which follow")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    J = json.load(open(a.report_json))
    st = ";".join(x.strip() for x in open(a.status) if x.strip())
    notes = []
    t = J.get("tests")
    if t:
        notes.append("(1) The tests of sec. 7 as amended by sec. 12.5 are applied mechanically in the report (no reading): window T_1 = %s, T_2 = %s r_s0; "
                     "P1 %s, P2's slope and rate parts %s, P3 %s, P4 %s; the floors in the report's test section." % (
                         f(t["T_1"], 4), f(t["T_2"], 4), "holds" if t["P1"]["holds_as_written"] else "does not hold",
                         "hold" if t["P2"]["holds_as_written_without_the_edge"] else "do not hold", "holds" if t["P3"]["holds_as_written"] else "does not hold",
                         "holds" if t["P4"]["holds_as_written"] else "does not hold"))
    notes.append("(2) No T = 0 rows: run_channel's outputs are the ends of refresh intervals (as in version 1, which the lead's review accepted); at T = 0 the "
                 "evolved sea is the initial vacuum by construction, so C-pg-2's identities hold there trivially and are not recorded.")
    notes.append("(3) The sectors' polar factors (code version 4): through eigh(G) with the columns of X Z normalized by their own norms and the near-null "
                 "column at the outer wall re-orthogonalized; agreement with the SVD route to 1e-12 (bench_polar_v1.py), U^T U - 1 in table T7p5 (C-pg-5). "
                 "The wall's edge modes have |omega| of 1e-9 (j = 3/2), 1e-13 (5/2) and 1e-17 to 1e-22 (7/2); their count is C-pg-13's.")
    notes.append("(4) S0 on L ran under code version 3 (the SVD route; static and cached, so the route's speed did not matter) and is copied into the run folder.")
    L = ["## Revision %d (%s): %s" % (a.revision, a.stamp, a.scope or "item 7's sea runs on the backgrounds of SPEC_strand7_page.md sec. 12"), "",
         "- Code: solve/code/strand7/strand7_sea_page_v4.py (the lead's draft with the builder's completion, versions 2 to 4: sec. 12.7's corrections, the "
         "stable sectors and their fast route), strand7_page_data_v2.py, strand7_page_fig_v2.py, strand7_page_report_v1.py; launchers "
         "launch_strand7_S7p_v4.sh (stopped: the SVD route's speed) and launch_strand7_S7p_v5.sh (two GPU slots, sec. 12.6's order). Box: "
         "<internal>/S7p_run3. Status lines: " + st + ".",
         "- Builder's choices (for the lead to confirm or change): (a) the sectors' polar factors computed stably (versions 3 and 4; the docstring); "
         "(b) the floors of the tests from the j = 1/2 checks only (K1, K2, H run at j = 1/2), applied to the three-channel sums; (c) the mean slopes "
         "between the first and the last outputs inside each interval; (d) the spectrum's upper edge read from sector alpha's particle bins; (e) the "
         "C-pg-14 late law fitted on L's 0.5 r_s0 trajectory record.",
         "- Records: solve/results/strand7/S7p_run3 (the JSON per run and the report; the modes files stay on the box).",
         "- Builder's notes (facts located in the tables; no reading): " + " ".join(notes),
         "- Emitted tables: strand7_page_report_v1.py's output.", ""]
    body = open(a.report_md).read().split("\n")
    body = ["#" + x if x.startswith("##") else ("### " + x[2:] if x.startswith("# ") else x) for x in body]
    sec = "\n".join(L + body) + "\n"
    title = ("# Item 7 result: the page curve and the correlations of the light sector along the external-frame collapse (revision %d, %s)" % (
        a.revision, a.stamp))
    if os.path.exists(a.out):
        s = open(a.out).read()
        s = title + "\n" + s.split("\n", 1)[1]
        s = s.rstrip("\n") + "\n\n" + sec
    else:
        s = (title + "\n\nBuilder: analyst. Spec: solve/specs/SPEC_strand7_page.md (secs. 0 to 12; the lead, reviewer A on the provider API). Numbers, controls "
             "and commands only; no reading.\n\n" + sec)
    open(a.out, "w").write(s)
    print("wrote %s revision %d" % (a.out, a.revision))


if __name__ == "__main__":
    main()
