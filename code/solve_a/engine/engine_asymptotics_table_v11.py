#!/usr/bin/env python
"""engine_asymptotics_table_v11.py (builder analyst; engine spec 38.5 (1): "FILE 5 is row (6)'s text for ASYMPTOTICS_checks_v1.8, composed
by engine_asymptotics_table_v11 (new): the version 9 tables at 40 r_s and the 300 r_s minimum's table verbatim under the heading ...; the
version 6 to 8 tables stay the record of the plateau reading"; FILE 5 = ASYMPTOTICS_row6_v1.8_text.md: "engine_asymptotics_table_v11
composes version 1.8 from version 1.7 with this row's heading and sentence in place of version 1.7's heading, the version 9 tables at 40
r_s (both phases) and the 300 r_s minimum's table from results/engine/CAP7/capture_read_v3_v9_t300_min_final.md copied verbatim under
it, the version 6, 7 and 8 tables kept as the record of the plateau reading"): a composition of the delivered version 1.7 (printed by
engine_asymptotics_table_v10; no record is re-read and no row re-printed), line for line: the title's version and script names read
ASYMPTOTICS_checks_v1.8 and engine_asymptotics_table_v11 (once each); row (6)'s heading line "### (6) The capture run's non-return
fraction" replaced by "### " and FILE 5's quoted heading, FILE 5's quoted sentence inserted under it as its own paragraph; the version 6,
7, 8 and 9 tables that follow kept verbatim in version 1.7's order; the 300 r_s minimum's reading (capture_read_v3_v9_t300_min_final.md)
inserted verbatim after the version 9 tables, before the closing line, under a label line; one line after the closing line names the
version 1.8 changes. Each replacement's count printed and checked (1); every other line of version 1.7 is asserted equal in the output.
FILE 5's heading and sentence are asserted free of sec. 4's replaced words (frozen, freez, window, prediction); version 1.7's own
occurrences elsewhere (row (3)'s record labels) are listed by line, unchanged (FILE 5 does not touch them). Reading aid in the log, never
an edit: the sentence's numbers beside the files FILE 5's condition names. Refuses to overwrite. Args: --v17 --row6 --t300 --cap40 --out
--log."""
import argparse
import os
import re
import sys

REPLACED = ("frozen", "freez", "window", "prediction")


def main():
    ap = argparse.ArgumentParser()
    for k in ("v17", "row6", "t300", "cap40", "out", "log"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    for p in (a.out, a.log):
        if os.path.exists(p):
            sys.exit("refusing to overwrite %s" % p)
    L = ["## engine_asymptotics_table_v11 (engine spec 38.5 (1)): ASYMPTOTICS_checks_v1.8 composed from version 1.7 and FILE 5; the replacements, their counts and the checks", "",
         "| item | old | new | count | expected |", "|---|---|---|---|---|"]
    src = open(a.v17).read()
    lines = src.split("\n")
    r6 = open(a.row6).read().split("\n\n")
    hp = [p for p in r6 if p.startswith("Heading:")]; sp_ = [p for p in r6 if p.startswith("Sentence (under the heading")]
    cp = [p for p in r6 if p.startswith("Condition:")]
    if len(hp) != 1 or len(sp_) != 1:
        sys.exit("FILE 5's heading or sentence not found")
    hq = re.findall(r'"([^"]*)"', hp[0]); sq = re.findall(r'"([^"]*)"', sp_[0])
    if len(hq) != 1 or len(sq) != 1:
        sys.exit("FILE 5's heading or sentence quotes are not one each: %d, %d" % (len(hq), len(sq)))
    heading, sentence = hq[0], sq[0]
    low_new = (heading + " " + sentence).lower()
    bad = {w: low_new.count(w) for w in REPLACED if w in low_new}
    if bad:
        sys.exit("sec. 4 words in FILE 5's text: %s" % bad)
    out = []
    counts = {"title version": 0, "title script": 0, "row (6) heading": 0, "closing line": 0}
    old_head = "### (6) The capture run's non-return fraction"
    t300 = open(a.t300).read().rstrip("\n").split("\n")
    for i, ln in enumerate(lines):
        if i == 0:
            if ln.count("ASYMPTOTICS_checks_v1.7") == 1 and ln.count("(engine_asymptotics_table_v10.py;") == 1:
                counts["title version"] += 1; counts["title script"] += 1
                ln = ln.replace("ASYMPTOTICS_checks_v1.7", "ASYMPTOTICS_checks_v1.8").replace(
                    "(engine_asymptotics_table_v10.py;", "(engine_asymptotics_table_v11.py: version 1.7, printed by engine_asymptotics_table_v10.py, with row (6)'s heading and sentence of ASYMPTOTICS_row6_v1.8_text.md and the 300 r_s minimum's reading added;")
            out.append(ln); continue
        if ln == old_head:
            counts["row (6) heading"] += 1
            out += ["### " + heading, "", sentence]
            continue
        if ln.startswith("- every number above is printed by the script from the stored records"):
            counts["closing line"] += 1
            out += ["- C-capture version 9 to 300 r_s at the production's last minimum (T/rs 37.75; slice 302; tag ccap9t300_prod19_min), as engine_capture_read_v3 printed it (capture_read_v3_v9_t300_min_final.md, verbatim):", ""]
            out += t300 + [""]
            out.append(ln)
            out.append("- version 1.8 (engine_asymptotics_table_v11): row (6)'s heading and sentence are the lead's (ASYMPTOTICS_row6_v1.8_text.md, engine spec 38.5 (1)); the 300 r_s reading is engine_capture_read_v3's output verbatim; every other line is version 1.7's.")
            continue
        out.append(ln)
    for k, n in counts.items():
        L.append("| %s | %s | %s | %d | 1 |" % (k, {"title version": "ASYMPTOTICS_checks_v1.7", "title script": "(engine_asymptotics_table_v10.py;", "row (6) heading": old_head,
                                                    "closing line": "(the closing line kept)"}[k],
                                              {"title version": "ASYMPTOTICS_checks_v1.8", "title script": "(engine_asymptotics_table_v11.py: version 1.7 ... added;", "row (6) heading": "### " + heading[:80] + " ... and FILE 5's sentence under it",
                                               "closing line": "the 300 r_s reading inserted before it, one line after it"}[k], n))
        if n != 1:
            open(a.log, "x").write("\n".join(L) + "\n\nSTOPPED: %s counted %d\n" % (k, n))
            sys.exit("%s: count %d, expected 1" % (k, n))
    # every other line of version 1.7 is in the output, in order
    text = "\n".join(out)
    kept = [ln for i, ln in enumerate(lines) if i != 0 and ln != old_head]
    it = iter(out)
    missing = 0
    for ln in kept:
        for o in it:
            if o == ln:
                break
        else:
            missing += 1
    occ = []
    for i, ln in enumerate(lines):
        lw = ln.lower()
        for w in REPLACED:
            if w in lw:
                occ.append("line %d: %s x%d" % (i + 1, w, lw.count(w)))
    nums_t300 = []
    for ln in t300:
        if ln.startswith("| solved") or ln.startswith("| control"):
            c = [x.strip() for x in ln.strip().strip("|").split("|")]
            nums_t300.append("%s %s: C_m %s, departure %s, smallest before the arrival %s" % (c[0][:7], c[1][-11:], c[3], c[6], c[14]))
    nums40 = []
    for ln in open(a.cap40).read().split("\n"):
        if ln.startswith("| solved") or ln.startswith("| control"):
            c = [x.strip() for x in ln.strip().strip("|").split("|")]
            nums40.append("%s %s: %s" % (c[0][:7], c[1][-11:], c[12]))
    L += ["", "### the checks", "",
          "- version 1.7's other %d lines appear in the output in order: %s" % (len(kept), "yes" if missing == 0 else "NO (%d missing)" % missing),
          "- FILE 5's heading and sentence: none of sec. 4's replaced words (%s)" % ", ".join(REPLACED),
          "- version 1.7's own occurrences of those words, unchanged (FILE 5 does not touch them; for the lead): %s" % ("; ".join(occ) if occ else "none"),
          "- FILE 5's condition: %s" % (cp[0] if cp else "(none)"),
          "", "### reading aid (the files' numbers beside the sentence's; the lead's text copied, never edited)", "",
          "- %s: %s" % (a.t300, "; ".join(nums_t300)),
          "- %s: %s" % (a.cap40, "; ".join(nums40))]
    if missing:
        open(a.log, "x").write("\n".join(L) + "\n\nSTOPPED: version 1.7 lines missing\n")
        sys.exit("version 1.7 lines missing: %d" % missing)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "x").write(text)
    L.append("- composed: %s from %s, %s and %s" % (a.out, a.v17, a.row6, a.t300))
    open(a.log, "x").write("\n".join(L) + "\n")
    print("wrote %s and %s" % (a.out, a.log))


if __name__ == "__main__":
    main()
