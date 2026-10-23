#!/usr/bin/env python
"""Item 7's result file, a revision, version 2 (builder analyst; for strand7_page_report_v2.py's JSON under SPEC_strand7_page.md sec. 13):
assembles solve/results/RESULT_strand7_page.md from the report's markdown and JSON with the header lines (code, commands, the box status
lines, the builder's choices) and the builder's notes, whose numbers are read from the report's JSON, so that no number is retyped.  The
title follows sec. 13.1's labels (the probe's excess-entanglement curve along the prescribed locked collapse; never "the Page curve").
--revision N --stamp "<UTC>" --report-md --report-json --status (the run folder's status.txt) --scope (one line) --out (the result file; a
new revision is appended after the existing ones, the title line updated)."""
import argparse
import json
import os


def f(x, d=4):
    if x is None:
        return "-"
    x = float(x)
    return ("%." + str(d) + "g") % x if (x == 0 or 1e-3 <= abs(x) < 1e5) else "%.3e" % x


def yn(x):
    if x is None:
        return "not decidable (a floor or an entry missing)"
    if isinstance(x, str):
        return x
    return "holds" if x else "does not hold"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revision", type=int, required=True)
    ap.add_argument("--stamp", required=True)
    ap.add_argument("--report-md", required=True)
    ap.add_argument("--report-json", required=True)
    ap.add_argument("--status", required=True)
    ap.add_argument("--scope", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    J = json.load(open(a.report_json))
    st = ";".join(x.strip() for x in open(a.status) if x.strip())
    notes = []
    mk = J.get("markers_L", {})
    t = J.get("tests", {})
    if t.get("testable"):
        P1, P2, P3, P4 = t["P1"], t["P2"], t["P3"], t["P4"]
        notes.append("(1) The tests of sec. 13.5 applied mechanically in the report (no reading): the window [T_1, T_w] = [%s, %s] r_s0 (T_2 = %s, T_2p = %s), "
                     "the late interval from %s r_s0; P1 %s; P2a %s, P2b %s, P2c %s, P2d %s; P3 %s; P4 %s; every floor beside its test and per output in T7p8." % (
                         f(mk.get("T_1"), 5), f(J.get("T_w"), 5), f(mk.get("T_2"), 5), f(mk.get("T_2p"), 5), f(J.get("late_start"), 5), yn(P1["holds"]),
                         yn(P2["P2a"]), yn(P2["P2b"]), yn(P2["P2c"]), yn(P2["P2d"]), yn(P3["holds"]), yn(P4["holds"])))
    else:
        notes.append("(1) The window is not testable as sec. 13.3 defines it (T_1 = %s, T_w = %s r_s0)." % (f(mk.get("T_1"), 5), f(J.get("T_w"), 5)))
    notes.append("(2) c_band = %s from F0's j = 1/2 modes (a declared coefficient)%s." % (
        f(J.get("c_band"), 6), ("; on the flat background j = 3/2 and 5/2 give " + ", ".join(
            "%s" % f(max(v["max_abs_omega_times_h"] for v in d.values()), 6) for jj, d in J["cband_computed"].items() if jj != "0.5") +
            " (strand7_cband_v1.py; j = 1/2 recomputed there as the route's control)") if J.get("cband_computed") else ""))
    notes.append("(3) No T = 0 rows: run_channel's outputs are the ends of refresh intervals; at T = 0 the evolved sea is the initial vacuum by construction.")
    notes.append("(4) The sectors' polar factors (code version 4): through eigh(G) with the columns of X Z normalized by their own norms and the near-null "
                 "column at the outer wall re-orthogonalized; agreement with the SVD route to 1e-12 (bench_polar_v1.py); U^T U - 1 in T7p5 (C-pg-5). S0 on L ran "
                 "under code version 3 (static and cached) and is copied into the run folder.")
    notes.append("(5) B40 (sec. 13.7) was inserted into the running queue's job list before K1 at 01:28 UTC (under the queue's lock; nothing stopped); the "
                 "launcher file is unchanged.")
    mc = J.get("markers_C")
    if mc:
        notes.append("(6) C's markers from its S7a record and stored fields: T_1 = %s, T_2 = %s, T_2p = %s, T_lw = %s r_s0; c_+^(r) from the fields against "
                     "the record's monitor to %s." % (f(mc.get("T_1"), 5), f(mc.get("T_2"), 5), f(mc.get("T_2p"), 5), f(mc.get("T_lw"), 5),
                                                      f(mc.get("c_plus_r_from_fields_vs_record_max_rel"), 2)))
    if J.get("marker_shifts_R_minus_B40"):
        notes.append("(7) R's markers minus B40's (r_s0): " + ", ".join("%s %s" % (k, f(v, 4)) for k, v in J["marker_shifts_R_minus_B40"].items()) + ".")
    L = ["## Revision %d (%s): %s" % (a.revision, a.stamp, a.scope or "item 7's sea runs on the backgrounds of SPEC_strand7_page.md secs. 12 and 13"), "",
         "- Code: solve/code/strand7/strand7_sea_page_v4.py (the lead's draft with the builder's completion, versions 2 to 4), strand7_page_markers_v1.py, "
         "strand7_page_report_v2.py, strand7_page_data_v3.py, strand7_page_fig_v3.py, strand7_cband_v1.py; launcher launch_strand7_S7p_v5.sh (two GPU slots, "
         "sec. 12.6's order, B40 inserted per sec. 13.7). Box: <internal>/S7p_run3. Status lines: " + st + ".",
         "- Builder's choices (for the lead to confirm or change): (a) the markers' crossings interpolated log-linearly between the trajectory's 0.5 r_s0 "
         "samples on L and the S7a record's 0.1 r_s outputs on C (the record's own 0.1 r_s0 window printed beside on L); (b) the late interval starts at "
         "max(T_2, T_2p) + 5 r_s0; (c) P2d's edge: the largest bin upper limit with count >= 2 and mean > 1e-3 over both sectors and every channel; (d) the "
         "raw-odds line over the 8 bins with sec. 13.5's qualifying rule (the record's own [1, 6] T_H line not used); (e) the floors' differences at P's "
         "output times (K1 and K2 to 36 r_s0), P at j = 1/2; (f) a test whose floor or entry is missing at a needed output is printed as not decidable.",
         "- Records: solve/results/strand7/S7p_run3 (the JSON per run and the report; the modes files stay on the box).",
         "- Builder's notes (facts located in the tables; no reading): " + " ".join(notes),
         "- Emitted tables: strand7_page_report_v2.py's output.", ""]
    body = open(a.report_md).read().split("\n")
    body = ["#" + x if x.startswith("##") else ("### " + x[2:] if x.startswith("# ") else x) for x in body]
    sec = "\n".join(L + body) + "\n"
    title = ("# Item 7 result: the probe's excess-entanglement curve and the correlations along the prescribed locked collapse L, with the rows' own "
             "collapse C as the comparison (revision %d, %s)" % (a.revision, a.stamp))
    if os.path.exists(a.out):
        s = open(a.out).read()
        s = title + "\n" + s.split("\n", 1)[1]
        s = s.rstrip("\n") + "\n\n" + sec
    else:
        s = (title + "\n\nBuilder: analyst. Spec: solve/specs/SPEC_strand7_page.md (secs. 0 to 13; the lead, reviewer A on the provider API). Numbers, controls "
             "and commands only; no reading.\n\n" + sec)
    open(a.out, "w").write(s)
    print("wrote %s revision %d" % (a.out, a.revision))


if __name__ == "__main__":
    main()
