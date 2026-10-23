#!/usr/bin/env python
"""Strand S result writer, revision 1 (builder analyst): assembles solve/results/RESULT_strandS_graded_entropy.md from the records of jobs S0,
S1 and S2 (and the builder's plan run1, superseded), with the builder's notes' numbers read from the JSONs, so that no number is retyped.
Args: --s0-json --s0-md --s1-json --s1-md --s2-json --s2-md --run1-json --sphere-note (a text line on S3's state) --stamp --out."""
import argparse
import json
import math


def f(v, d=6):
    if v is None:
        return "-"
    v = float(v)
    return "%.*g" % (d, v) if (v == 0 or 1e-3 <= abs(v) < 1e5) else "%.4e" % v


def demote(md):
    return ["#" + x if x.startswith("##") else ("### " + x[2:] if x.startswith("# ") else x) for x in open(md).read().split("\n")]


def main():
    ap = argparse.ArgumentParser()
    for k in ("--s0-json", "--s0-md", "--s1-json", "--s1-md", "--s2-json", "--s2-md", "--run1-json", "--sphere-note", "--stamp", "--out"):
        ap.add_argument(k, required=True)
    a = ap.parse_args()
    S0, S1, S2, R1 = (json.load(open(p)) for p in (a.s0_json, a.s1_json, a.s2_json, a.run1_json))
    ext = S2["extrapolations"]
    tg = S2["targets"]
    rows = S2["rows"]
    small = [r for r in rows if r["aM"] == min(x["aM"] for x in rows)]
    c3max = max(abs(c["S4_minus_2S2"]) for c in S1["C3"])
    c3mu = [c for c in S1["C3"] if abs(c["S4_minus_2S2"]) == c3max][0]
    eps = 2.220446049250313e-16
    floor_c3 = 4 * c3mu["L"] * eps * abs(math.log(eps))
    r1 = {(tuple(r["rect"]), r["aM"]): r for r in R1["rows"]}
    cmp_run1 = []
    for r in rows:
        if r["rect"] == "light" and (tuple([0.0, 0.5, 0.5, 1.0]), r["aM"]) in r1:
            q = r1[(tuple([0.0, 0.5, 0.5, 1.0]), r["aM"])]
            cmp_run1.append((r["aM"], r["c_S"], q["graded_over_aM2J"]))
    worst_run1 = max(abs(x[1] / x[2] - 1) for x in cmp_run1) if cmp_run1 else None
    c7 = S2["C7"]
    v1max = max(abs(c["V1_dS_gr"]) for c in c7)
    lit = [(c["rect"], c["aM"], c["V2_literal_dS_gr_rel"], c["V2_literal_dk0_rel"]) for c in c7]
    rep = [(c["rect"], c["aM"], c["V2_repair_dS_gr_rel"], c["V2_repair_dk0_rel"], c["V2_repair_eta_star_change"]) for c in c7]
    s0 = S0["fits"]["d32_nk24"][0]
    L = ["# Strand S result, revision 1 (%s): the graded cut entropy of the 4-packet sea on the flat [100] cut" % a.stamp, "",
         "Builder: analyst. Spec: solve/specs/SPEC_strandS_graded_entropy.md (lead, final4 turn 9). Numbers, controls and commands only; the "
         "builder's notes are labelled and carry no physics reading. Every table below is emitted by the scripts named; this file is assembled by "
         "solve/code/strandS/strandS_result_writer_v1.py from their JSONs.", "",
         "## Header",
         "- Code (solve/code/strandS/): strandS_S0_fit_v1.py (job S0, C5), strandS_table_v1.py (job S1: the 2-component chain table, the fixed-alpha "
         "fit, C3, C4; launch_strandS_S1_v1.sh), strandS_flat_v3.py (job S2: the graded cut, hat k_0 by strand5_J2e_v2's two routes, C1, C2, C6, C7, "
         "C8, the extrapolations; launch_strandS_S2_v1.sh); the paper's lattice_entropy.py and eta.py unmodified (box copies in "
         "<internal>/orient). Box: <internal>/sS_entropy (S0, S1, S2).",
         "- Records: solve/results/strandS/S0, S1, S2 (JSONs, the emitted markdown, the logs); run1 (the builder's plan before the spec: "
         "strandS_flat_v1.py) is kept in solve/results/strandS/run1 and superseded.",
         "- The spherical cut (job S3): " + a.sphere_note, "",
         "## The decision numbers (from S2's extrapolations on aM <= %s; the fit X(aM) = X_0 + c (aM)^2 log(1/aM) + c' (aM)^2)" % f(S2["meta"]["fit_max"]),
         "| rectangle | c_S -> | 1/(3 pi) | fit residual | R_gr -> | 2/9 | fit residual | hat k_0/((aM)^2 j) -> | 3/pi^2 |", "|---|---|---|---|---|---|---|---|---|"]
    for rect, e in ext.items():
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (rect, f(e["c_S"]["limit"], 10), f(tg["a1_one_over_3pi"], 10), f(e["c_S"]["max_resid"], 2),
                                                               f(e["R_gr"]["limit"], 10), f(tg["two_ninths"], 10), f(e["R_gr"]["max_resid"], 2),
                                                               f(e["k0_over_aM2j"]["limit"], 10), f(tg["three_over_pi2"], 10)))
    L += ["", "At the smallest spacing (aM = %s): %s." % (f(small[0]["aM"]), "; ".join(
        "%s: c_S %s, R_gr %s, the unsigned 2 pi eta_*/(3 j (aM)^2) %s" % (r["rect"], f(r["c_S"], 9), f(r["R_gr"], 9), f(r["unsigned_record_form"], 5)) for r in small)), "",
          "## Builder's notes (labelled; for the lead to confirm or change)",
          "- (1) C5, the stored-table fit (job S0) does not return a_1 = 1/(3 pi): strand 3's (32,24) ladder gives a_1 = %s on q <= 0.1 (largest residual of the "
          "quotient %s); eta.value's uniform transverse grid has no point within pi/48 of a node in each direction, so its zone sum is analytic in q^2 below "
          "that scale. The table route resolves the nodes (S2's a_1 is c_S's limit above)." % (f(s0["a_1"], 4), f(s0["max_resid"], 3)),
          "- (2) C7 V2: the lead's literal V2 (the rectangle's copy at 2 m_s with weights -eta_s/4) is not neutral at the J order: the copy's own J is minus the "
          "rectangle's (the identity sum (-eta/4)(4 m^2) log(4 m^2) = -J - log 4 sum eta m^2 = -J), so the combined inventory has J = 0 and both hat S_gr and "
          "hat k_0 change at order (aM)^2 (" + "; ".join("%s aM %s: %s, %s" % (x[0], f(x[1]), f(x[2], 4), f(x[3], 4)) for x in lit) + "). The builder's repair "
          "(the J-neutral block x' = (1, 2, 3, 4)/4 M^2 with weights solving sum c' = sum c' x' = sum c' x' log x' = 0, c'_1 = 1/4) changes hat S_gr and hat "
          "k_0 at order (aM)^4 log while eta_* changes at order 1 (" + "; ".join("%s aM %s: %s, %s, Delta eta_* %s" % (x[0], f(x[1]), f(x[2], 3), f(x[3], 3), f(x[4], 6)) for x in rep)
          + "). V1 (the exact pair at q' = 0.3): the largest |Delta hat S_gr| %s." % f(v1max, 3),
          "- (3) C3: S_4 - 2 S_2 is at most %s (mu = %s, L = %d) against the spec's 1e-12; the eigh rounding floor n eps |log eps| for the 4-component interval "
          "there (n = 4L) is %s. The table's L-ladder differences are at the same floor (largest %s)." % (
              f(c3max, 3), f(c3mu["mu"]), c3mu["L"], f(floor_c3, 3), f(max(abs(x["diff"]) for x in S1["ladder"]), 3)),
          "- (4) hat k_0: the lead's draft integrates in s = log t up to t = e^60 with scipy's ive, which returns NaN beyond an argument of about 1e9; the "
          "production uses strand5_J2e_v2's route 1 (quad in u = t^(1/2) to t = 1e7 with the analytic tail; the rectangle's weight in closed form with "
          "expm1) and route 2 (mpmath tanh-sinh, 30 digits) as the check: the largest |route 1 - route 2| over the rows is %s." % f(max(abs(r["k0_route_diff"]) for r in rows), 2),
          "- (5) The quadrature ladder (npan, ngl, nphi doubled one at a time) moves hat S_gr by at most %s relative over every row; the fit-to-table junction "
          "at mu_min is %s." % (f(max(r["ladder_max_rel"] for r in rows), 2), f(S2["junction_fit_minus_table"], 3)),
          "- (6) The builder's plan run (run1, before the spec: L = 12/mu, nfft = 8L, the free-alpha fit below 0.015) agrees with S2 on the light rectangle's "
          "c_S to %s relative at the common spacings." % f(worst_run1, 2) if worst_run1 is not None else "- (6) run1: no common rows.",
          "- (7) The second rectangle is the spec's (1, 2, 2, 3)/3 (M the heaviest packet); run1 used (1, 2, 2, 3) in units of M^2, so its aM labels differ "
          "by sqrt 3 for that rectangle.", "",
          "## Job S0 (C5)"] + demote(a.s0_md) + ["", "## Job S1 (the table, C3, C4)"] + demote(a.s1_md) + ["", "## Job S2 (the graded cut, C1, C2, C6, C7, C8)"] + demote(a.s2_md)
    open(a.out, "w").write("\n".join(L) + "\n")
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
