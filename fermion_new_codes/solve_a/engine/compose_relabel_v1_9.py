#!/usr/bin/env python
"""compose_relabel_v1_9.py (builder analyst; inbox #11: the hreduce lane's review of COLLAPSE_ladder_v1, dm7_ladder_review.tex and
redteam_dm7_ladder_review.tex, "apply what it requires to the ladder and its labels"): version 1.8 with the title's source list and the
sources line extended and six rows added before the sources line: (1) COLLAPSE_ladder_v1.md's head reads as COLLAPSE_ladder_v1.1.md; (2)
ITEM5_collapse_sentences_v4.md sentence 3 withdrawn, read with COLLAPSE_ladder_v1.1's verdict lines until the lead's version 5; (3) its
sentence 4's averaging-scale clause withdrawn likewise; (4) this lane's kernel variance read as ell^2; (5) the residual over the root slope
read as a sensitivity diagnostic; (6) a Newton-Krylov member's residuals read as two diagnostics. The withdrawal rows quote the review's and
COLLAPSE_ladder_v1.1's own numbers (--ladder-v11: its factor-2 verdict line is quoted verbatim). Args: --v18 --ladder-v11 --out."""
import argparse
import re


def main():
    ap = argparse.ArgumentParser()
    for k in ("v18", "ladder-v11", "out"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    t = open(a.v18).read()
    lv = open(a.ladder_v11).read()
    vl = re.search(r"^- the factor-2 reading \(sigma/a 4, 8, all records\): formation spread [^;]*; largest R_t spread [^;]*;", lv, re.M)
    assert vl, "no factor-2 verdict line in the ladder file"
    verdict = vl.group(0)[2:].rstrip(";").replace("|", "/")
    tm = re.search(r"^# RELABEL_table_v1\.8: (.*?)\(engine spec (.*?)\)$", t, re.M)
    assert tm
    t = t.replace(tm.group(0), "# RELABEL_table_v1.9: %s(engine spec %s; the hreduce lane's DM7 review, inbox #11)" % (tm.group(1), tm.group(2)), 1)
    pm = re.search(r"Version 1\.8 = version 1\.7 with ", t)
    assert pm
    t = t.replace(pm.group(0), "Version 1.9 = version 1.8 with six rows added (inbox #11: the hreduce lane's review of the averaging ladder). Version 1.8 = version 1.7 with ", 1)
    rows = ["| COLLAPSE_ladder_v1.md (its head, the lead's text of turn 39) | read as COLLAPSE_ladder_v1.1.md: the head is withdrawn (it read an earlier subset of the records); until the lead's new head (turn 40) the reading is the script's verdict line \"%s\" (dm7_ladder_review.tex item 1) |" % verdict,
            "| ITEM5_collapse_sentences_v4.md, sentence 3 (the averaging-scale control) | withdrawn until the lead's version 5 (turn 40): its \"radius at matched clocks agrees within 5.4 percent\" holds on an earlier subset only; read with \"%s\" (COLLAPSE_ladder_v1.1.md; dm7_ladder_review.tex items 1 and 2: both factor-2 criteria fail, and the factor-4 test is not evaluable with the sigma/a 2 member's records) |" % verdict,
            "| ITEM5_collapse_sentences_v4.md, sentence 4, its clause from \"and across the averaging scale\" | withdrawn until the lead's version 5 (turn 40): the sigma/a 8 member holds trapping zeros at T/rs 27 to 29.5, so \"none at 8 by 40\" is false for the full table, and the pocket range omits the count printed at 14.5 (redteam_dm7_ladder_review.tex DMR1); read with COLLAPSE_ladder_v1.1's tables |",
            "| any text giving this lane's averaging kernel a per-axis variance ell^2 + a^2/12 (sigma/a 4.010, 8.005 or 16.003; sigma^2 16.083333 or 64.083333) | reads sigma^2 = ell^2 (sigma/a = ell/a): the back-map is the three-dimensional Gaussian of standard deviation ell with no cell convolution, so the a^2/12 term is absent; the paper's compact kernel, ell^2/15 + a^2/12, is a different object (dm7_ladder_review.tex item 3) |",
            "| \"R_t bound\" (the self-consistent residual over the root slope, in the ladder's tables) | reads \"the residual over the root slope\": a sensitivity diagnostic, not a root-error bound (no inverse-response bound is supplied; dm7_ladder_review.tex item 4) |",
            "| a Newton-Krylov member's \"residual\" | reads as two diagnostics, never identified: the kept residual (the retained evaluation's, Newton's acceptance) and the confirming plain step's (dm7_ladder_review.tex item 4; engine_ladder_compare_v4 prints both) |"]
    si = t.index("\n- Sources: ")
    t = t[:si].rstrip("\n") + "\n" + "\n".join(rows) + "\n" + t[si:]
    sm = re.search(r"\n- Sources: ([^\n]*)", t)
    assert sm
    t = t.replace(sm.group(0), "\n- Sources: " + sm.group(1).rstrip(".") + "; deliveries/hreduce/dm7_ladder_review.tex and redteam_dm7_ladder_review.tex; deliveries/solve_a/COLLAPSE_ladder_v1.1.md.", 1)
    open(a.out, "w").write(t)
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
