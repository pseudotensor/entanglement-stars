#!/usr/bin/env python
"""compose_relabel_v1_14.py (builder analyst; engine spec 34.2 (1): "the builder composes RELABEL_table_v1.14 with those rows closed and
the certificate row of 34.3 (4)"): version 1.13 with the title's source list and the sources line extended and three rows added before the
sources line: TEMPERATURE_solved_exterior_v1.1 read as v1.2; version 1.13's seven rows closed; label B2's innermost-node clauses and
COLLAPSE_certificate_statement_v1.1's innermost-node factors read as the uncorrected normalization's record. The stored texts are quoted
after this script checks that each occurs in its file (nothing retyped from memory). Args: --v113 --labels37 --cert11 --out."""
import argparse
import re

B2A = "differ by factors 2.4 to 3.9 at the innermost node r = 4 at every late record"
B2B = "the innermost-node maxima above are the direct route's and are not certified"
C11 = "The innermost node r = 4 differs between the routes by a factor 1.0 at 12.125, 3.5 at 12.75 and 3 to 11.5 at every record from 13.0 on"


def main():
    ap = argparse.ArgumentParser()
    for k in ("v113", "labels37", "cert11", "out"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    l37 = open(a.labels37).read(); c11 = open(a.cert11).read()
    for txt, src, nm in ((B2A, l37, "labels37"), (B2B, l37, "labels37"), (C11, c11, "cert11")):
        assert txt in src, "not found in %s: %r" % (nm, txt[:60])
    print("the three stored clauses found verbatim")
    rows = [
        "| TEMPERATURE_solved_exterior_v1.1.md | read as TEMPERATURE_solved_exterior_v1.2.md (engine 34.2 (1), turn 42: the icloud lane's delta check's four literal items applied, the conclusion unchanged; version 1.1 stays the record) |",
        "| version 1.13's seven rows on TEMPERATURE_solved_exterior_v1.1.md (the umbrella row and the six clause rows) | closed: TEMPERATURE_solved_exterior_v1.2.md replaces those clauses (engine 34.2 (1)) |",
        "| label B2's \"%s\" and \"%s\" (COLLAPSE_records_v1.1_labels_turn37.md, the r_s/a 32 set under engine_certificate_v2), and COLLAPSE_certificate_statement_v1.1's \"%s\" | read as the uncorrected normalization's record; the certified windows are COLLAPSE_certificate_statement_v1.2's (engine 34.3 (4); COLLAPSE_certificate_v1.2, engine_certificate_v3) |" % (B2A, B2B, C11),
    ]
    t = open(a.v113).read()
    tm = re.search(r"^# RELABEL_table_v1\.13: (.*?)\(engine spec (.*?); its delta check of version 1\.1, inbox #16\)$", t, re.M)
    assert tm
    t = t.replace(tm.group(0), "# RELABEL_table_v1.14: %s(engine spec %s; its delta check of version 1.1, inbox #16; 34.2, 34.3)" % (tm.group(1), tm.group(2)), 1)
    pm = re.search(r"Version 1\.13 = version 1\.12 with ", t)
    assert pm
    t = t.replace(pm.group(0), "Version 1.14 = version 1.13 with %d rows added (engine 34.2 (1) and 34.3 (4), turn 42: the temperature sentence read as version 1.2 and version 1.13's seven rows closed; the certificate's innermost-node clauses read as the uncorrected normalization's record). Version 1.13 = version 1.12 with " % len(rows), 1)
    si = t.index("\n- Sources: ")
    t = t[:si].rstrip("\n") + "\n" + "\n".join(rows) + "\n" + t[si:]
    sm = re.search(r"\n- Sources: ([^\n]*)", t)
    assert sm
    t = t.replace(sm.group(0), "\n- Sources: " + sm.group(1).rstrip(".") + "; solve/specs/SPEC_final4b_engine.md secs. 34.2, 34.3; deliveries/solve_a/TEMPERATURE_solved_exterior_v1.2.md, COLLAPSE_certificate_statement_v1.2.md, COLLAPSE_records_v1.1_labels_turn37.md, COLLAPSE_certificate_statement_v1.1.md.", 1)
    open(a.out, "w").write(t)
    print("wrote %s (%d rows added)" % (a.out, len(rows)))


if __name__ == "__main__":
    main()
