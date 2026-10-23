#!/usr/bin/env python
"""Strand 7 result, revision 2's section, version 2 (builder analyst; version 2 letters the page-background notes (3a), (3b), ... and states
the P runs' re-queue once): assembles the header lines, the builder's notes with their numbers read from the
report JSONs (strand7_shell_report_v3.py --out-json), and the emitted tables (the reports' markdown), so that no number is retyped.
Args: --s7a-md/--s7a-json (S7a_run2's report), --s7p-md/--s7p-json (S7p_fields' report), --status-s7a/--status-s7p (the status files),
--stamp (the revision's UTC time), --out (the section's markdown)."""
import argparse
import json


def f(x, d=4):
    if x is None:
        return "-"
    x = float(x)
    if x == 0:
        return "0"
    if 1e-3 <= abs(x) < 1e5:
        return ("%." + str(d) + "g") % x
    return "%.3e" % x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s7a-md", required=True)
    ap.add_argument("--s7a-json", required=True)
    ap.add_argument("--s7p-md", required=True)
    ap.add_argument("--s7p-json", required=True)
    ap.add_argument("--status-s7a", required=True)
    ap.add_argument("--status-s7p", required=True)
    ap.add_argument("--stamp", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    A = json.load(open(a.s7a_json))
    P = json.load(open(a.s7p_json))
    sa = ";".join(x.strip() for x in open(a.status_s7a) if x.strip())
    sp = ";".join(x.strip() for x in open(a.status_s7p) if x.strip())
    held = {t: x for t, x in A.items() if t.startswith("heldlock")}
    selfr = {t: x for t, x in A.items() if t.startswith("self")}
    notes = []
    if held:
        rr = ", ".join("%s %s" % (t.split("_")[1], f(x["end"]["r_sh_over_A"])) for t, x in held.items())
        notes.append("(1) Every held-lock run stops at its first output with R_sh - 4A < 1e-6 r_s, at r_sh/A = %s (E_0: value; table S2), "
                     "after D1 and with D2 at the stop (table S2); table R3 gives rule (iii)'s late fit against frozen_schw's." % rr)
        lay = []
        for t, x in held.items():
            if x.get("layers"):
                y = x["layers"][-1]
                lay.append("%s: %s cells; R(r_p99)/4A %s with the code's splines, %s linear; the exterior form at r_p99 %s" % (
                    t.split("_")[1], f(y["width_cells"], 3), f(y["R_p99_4A"], 7), f(y["R_p99_4A_lin"], 7), f(y["Rext_p99_4A"], 7)))
        notes.append("(2) Table L (the builder's diagnostic, not in the spec): at the held-lock stop the shell spans less than one grid cell, and "
                     "the areal radius of its outer material relative to 4A depends on the interpolation inside that cell (%s). The stop rule and "
                     "D2 compare the median's areal radius, R_sh = r_sh psi(r_sh)^2 from the code's splines, with 4A." % "; ".join(lay))
    for k, (t, x) in enumerate(P.items()):
        e = x["end"]
        tc = x.get("throat_crossing")
        notes.append("(3%s) The page background %s (strand7_shell_v3.py, --stop-rule page: the run continued past the S7a stop): stop '%s' "
                     "at T = %s r_s0; A/A(0) - 1 = %s, c_+^(r) = %s, r_sh/A = %s at the end%s (the S7p_fields tables below)." % (
                         "abcdefgh"[k], t, x["stop"], f(e["T_over_rs"], 5), f(e["A_rel"]), f(e["c_plus_r"]), f(e["r_sh_over_A"]),
                         "; the throat crossing r_sh = A at T = %s r_s0 with c_+ %s" % (f(tc["T_over_rs"], 5), f(tc["c_plus_r"])) if tc else ""))
    if P:
        notes.append("The P runs of item 7 were re-queued to end at T = 30 r_s0, the h = 1 held-lock background's end (launch_strand7_S7p_v2.sh), "
                     "instead of 60.")
    for t, x in selfr.items():
        e = x["end"]
        tc = x.get("throat_crossing")
        ri = x.get("rule_i", {})
        notes.append("(4) %s: stop '%s' at T = %s r_s; %s; at the end r_sh/A = %s, R_sh/4A = %s, B/A = %s, c_+^(r) = %s, A/A(0) - 1 = %s; "
                     "rule (i) as written (table R1): freeze conditions %s, runaway conditions %s." % (
                         t, x["stop"], f(e["T_over_rs"], 5),
                         "the throat crossing at T = %s r_s with c_+ %s and B/A %s" % (f(tc["T_over_rs"], 5), f(tc["c_plus_r"]), f(tc["B_over_A"]))
                         if tc else "no throat crossing", f(e["r_sh_over_A"]), f(e["R_sh"] / e["R_trap"], 5), f(e["B_over_A"]), f(e["c_plus_r"]),
                         f(e["A_rel"]), "met" if ri.get("freeze_as_written") else "not met", "met" if ri.get("runaway_as_written") else "not met"))
    L = ["## Revision 2 (%s): S7a revision 2, the continuation and the held-lock diagnostic (SPEC_strand7_collapse.md sec. 9.2), and the page "
         "background" % a.stamp, "",
         "- Code: solve/code/strand7/strand7_shell_v2.py (from v1: --run heldlock, the continuation's stop rule, the sec. 9.2 monitors, the throat "
         "crossing in the head), launch_strand7_S7a_rev2_v1.sh (nine single-thread jobs, four at a time: the held-lock and continuation runs at "
         "E_0 = 16, 32 (N_p = 2e4) and 8, the warm variant B' (v_perp = 0.1) at E_0 = 8, C3's step-halving at E_0 = 16 for both); "
         "strand7_shell_v3.py (from v2: --stop-rule page, which ends a run only at T_end, N(r_sh) < 1e-12, a Newton failure, a non-positive lapse "
         "at a marker or the centre guard) with launch_strand7_S7p_fields_v1.sh (item 7's backgrounds, two threads); the report "
         "strand7_shell_report_v3.py (tables S2, TC, R1, R3, P, L, T7b1, C3, C4, M; version 2 ran once on the held-lock trio and differs only in "
         "table L's interpolation). On the box: <internal>/s7_collapse (S7a_run2, S7p_fields). Status lines: S7a_run2: %s. "
         "S7p_fields: %s." % (sa, sp),
         "- Builder's choices (for the lead to confirm or change): (a) rule (i)'s 'c_+ -> 0' is read as c_+^(r)(r_sh) below 1e-3 at the last "
         "record (D2's threshold), and the rates are the largest over consecutive outputs in the last 5 r_s; (b) rule (iii)'s late fit uses the last "
         "10 r_s before the first record with R_sh - 4A <= 1e-6 r_s (that record excluded), against frozen_schw E_0 = 8 case A of revision 1; (c) "
         "table L is a diagnostic the spec does not ask for (labelled); (d) the page background runs with --stop-rule page from strand7_shell_v3.py "
         "(the S7a stop at R_sh - 4A withdrawn there too), held lock, E_0 = 16, h = 1 and 0.5, E_0 = 32, and self E_0 = 16, to 60 r_s0.",
         "- Records: solve/results/strand7/S7a_run2 and solve/results/strand7/S7p_fields (the JSON per run, the reports' markdown and JSON; the "
         "fields files stay on the box).",
         "- Builder's notes (facts located in the tables below; no reading): " + " ".join(notes),
         "- Emitted tables: strand7_shell_report_v3.py's output over S7a_run2, then over S7p_fields.", ""]
    for p in (a.s7a_md, a.s7p_md):
        body = open(p).read().split("\n")
        body = ["#" + x if x.startswith("##") else ("### " + x[2:] if x.startswith("# ") else x) for x in body]
        L += body + [""]
    open(a.out, "w").write("\n".join(L) + "\n")
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
