#!/usr/bin/env python
"""Strand 7 result, revision 2 part 2's section (builder analyst): assembles the header lines, the builder's notes with their numbers read
from the reports' JSONs, and the emitted tables of S7a part 2 (SPEC_strand7_collapse.md secs. 10.2 and 11): the shell report over S7a_run2
and S7a_run3 (strand7_shell_report_v4.py: the passage record with A/A(0) - 1, the areal record, the renamed flags), the part 2 report
(strand7_part2_report_v2.py: F_disc and its calibration, the dt ladder with T_valid and its labelled variants, the E-ladder, the moved
exterior, the record checks), the coherent benchmark (strand7_coherent_v2.py) and the areal characteristics (strand7_areal_char_v4.py), so that
no number is retyped.  Appends the section to --out and updates the title line.  Args: --shell-md --shell-json --part2-md --part2-json
--coherent-md --coherent-json --areal-md --areal-json --status (comma list of status files) --stamp --out."""
import argparse
import json


def f(x, d=4):
    if x is None:
        return "-"
    x = float(x)
    if x == 0:
        return "0"
    return ("%." + str(d) + "g") % x if 1e-3 <= abs(x) < 1e5 else "%.3e" % x


def demote(md):
    return ["#" + x if x.startswith("##") else ("### " + x[2:] if x.startswith("# ") else x) for x in open(md).read().split("\n")]


def main():
    ap = argparse.ArgumentParser()
    for k in ("--shell-md", "--shell-json", "--part2-md", "--part2-json", "--coherent-md", "--coherent-json", "--areal-md", "--areal-json",
              "--status", "--stamp", "--out"):
        ap.add_argument(k, required=True)
    a = ap.parse_args()
    SJ, PJ, CJ, AJ = (json.load(open(p)) for p in (a.shell_json, a.part2_json, a.coherent_json, a.areal_json))
    st = " | ".join(";".join(x.strip() for x in open(p) if x.strip()) for p in a.status.split(","))
    notes = []
    cal = PJ.get("static_calibration")
    if cal:
        notes.append("(1) F_disc's boundary term (4 pi K A_n B_n/R_max, the exterior form's own contribution beyond the box) calibrated on C1's static member: "
                     "F_disc/(4 pi K A) - 1 = %s with the nodal sum (the discrete identity) and %s with the markers' sum (the gather/deposit mismatch)." % (
                         f(cal["rel_nodal"], 3), f(cal["rel_markers"], 3)))
    if PJ.get("F_disc"):
        s_ = PJ["sensitivities_for_item7"].get("F_disc", {})
        notes.append("(2) Along self E16 (stored fields): the largest |F_disc - 4 pi K A_tail|/(4 pi K A_tail) %s, the largest |F_disc/F_disc(0) - 1| %s." % (
            f(s_.get("max_abs_defect_rel"), 3), f(s_.get("max_abs_F_rel"), 3)))
    lad = PJ.get("ladder", {})
    notes.append("(3) The dt ladder (dt = %s r_s): T_valid as written %s r_s; the labelled variants (r_sh/A relative; the charge alone) %s and %s r_s; the median "
                 "observed orders %s." % (", ".join(f(x, 3) for x in lad.get("dt_over_rs", [])), f(lad.get("T_valid_over_rs"), 4),
                                         f(lad.get("T_valid_relative_r_sh_over_rs"), 4), f(lad.get("T_valid_charge_only_over_rs"), 4),
                                         json.dumps({k: (round(v, 3) if v is not None else None) for k, v in (lad.get("median_order") or {}).items()})))
    if PJ.get("rmax"):
        d = PJ["rmax"]["diffs"]
        notes.append("(4) The exterior moved to %s r_s: the largest relative differences from the base, %s." % (
            f(PJ["rmax"]["R_max"][1], 4), "; ".join("%s %s before and %s after the passage" % (k, f(v["before"], 3), f(v["after"], 3)) for k, v in d.items())))
    if PJ.get("E_ladder"):
        notes.append("(5) The E-ladder's spread at the passage (not a pure refinement): %s." % ", ".join(
            "%s %s" % (k, f(v, 3)) for k, v in PJ["E_ladder"]["spread"].items()))
    runs = CJ.get("runs", {})
    notes.append("(6) The coherent benchmark (version 2): %s." % "; ".join(
        "%s: max |A/A(0) - 1| %s, the virial residual %s, the release relation %s, the passage at %s r_s0" % (
            k, f(r["max_abs_A_rel"], 3), f(r["max_abs_virial_residual_over_A"], 3), f(r["max_abs_u2_minus_release_relation"], 3),
            f(r["throat_crossing"]["T_over_rs0"], 6) if r.get("throat_crossing") else "-") for k, r in runs.items()))
    sm = AJ.get("summary", {})
    bn = [q for q in AJ.get("rows", []) if q.get("both_negative_nonkink_nodes")]
    rat = [x["psidot_r_over_Adot"] for q in bn for x in q["both_negative_nonkink_nodes"] if x.get("psidot_r_over_Adot") is not None]
    notes.append("(7) The areal characteristics (version 4): outputs with a both-negative node outside the kink zone %s (c3, c5, bw), inside it %s; at every such "
                 "node outside the kink zone psi_dot r/A_dot lies in [%s, %s] and A_dot < 0 at %d of the %d outputs." % (
                     json.dumps(sm.get("outputs_with_both_negative_nonkink")), json.dumps(sm.get("outputs_with_both_negative_kink")),
                     f(min(rat), 7) if rat else "-", f(max(rat), 7) if rat else "-", sum(1 for q in bn if q["A_dot"] < 0), len(bn)))
    ar = {t: v.get("areal_record") for t, v in SJ.items() if v.get("areal_record")}
    if ar:
        notes.append("(8) The areal record (ii): %s." % "; ".join(
            "%s %s" % (t, ("first at T = %s r_s, %s" % (f(v["first"]["T_over_rs"], 5), v["first"]["fraction_bracket"])) if v.get("first") else
                       "not reached in %d outputs" % v["outputs"]) for t, v in ar.items()))
    L = ["## Revision 2, part 2 (%s): S7a part 2 (SPEC_strand7_collapse.md secs. 10.2 and 11)" % a.stamp, "",
         "- Code: strand7_shell_v4.py (the dt/4 rung, R_max = 128 r_s, B' with the total v^2: launch_strand7_S7a_part2_v1.sh), strand7_shell_report_v4.py, "
         "strand7_part2_report_v2.py, strand7_coherent_v2.py, strand7_areal_char_v4.py (versions 2 and 3 superseded in the same session: version 3 added the "
         "both-negative listing, version 4 stops a ray at the first node), strand7_part2_facts_v1.py. Box: <internal>/s7_collapse "
         "(S7a_run2, S7a_run3, S7a_part2). Status lines: " + st + ".",
         "- Builder's choices (for the lead to confirm or change): (a) T_valid is printed as written (sec. 11.3 (b)) with two labelled variants beside it; "
         "(b) the areal characteristics' version 3 lists the both-negative nodes outside the kink zone with psi_dot r/A_dot (the builder's addition); (c) the "
         "rays from the fold's midpoint (r_p99 + A)/2 are added beside the lead's 2A starts; (d) the areal record's resolved fraction is a bracket from the "
         "three recorded quantiles (the markers' radii are not stored); (e) the dt/2 rung's fields are not stored, so the differencing control is the three "
         "estimates over the 0.1 r_s outputs.",
         "- Records: solve/results/strand7/S7a_part2 and S7a_run3 (the JSONs, the reports; the fields files stay on the box).",
         "- Builder's notes (facts located in the tables below; no reading): " + " ".join(notes),
         "- Emitted tables: the shell report (version 4), the part 2 report (version 2), the coherent benchmark (version 2), the areal characteristics "
         "(version 4).", ""]
    for p in (a.shell_md, a.part2_md, a.coherent_md, a.areal_md):
        L += demote(p) + [""]
    s = open(a.out).read()
    title = ("# Strand 7 result, revision 2 (parts 1 and 2): the external-frame collapse with quasi-static constraints, S7a revisions 1 and 2 and the "
             "page background (%s)" % a.stamp)
    s = title + "\n" + s.split("\n", 1)[1]
    s = s.rstrip("\n") + "\n\n" + "\n".join(L) + "\n"
    open(a.out, "w").write(s)
    print("wrote %s (part 2 appended)" % a.out)


if __name__ == "__main__":
    main()
