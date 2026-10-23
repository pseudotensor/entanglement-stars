#!/usr/bin/env python
"""compose_relabel_v1_16.py (builder analyst; engine spec 35.1 (3): "RELABEL_table_v1.16: the eight rows closed as (1) and (2); one row
added: the certificate file's head line 'The lead's reading of these tables is turn 41's' reads 'turn 42's and 43's (engine spec 34.3, 35.1;
the statement COLLAPSE_certificate_statement_v1.3)', the file's tables unchanged"): version 1.15 with the title's source list and the sources
line extended and five rows added before the sources line (the two texts read as their versions 1.3; version 1.15's three #19 rows and five
#20 rows closed; the certificate head's line). The head line is quoted after this script checks it occurs in COLLAPSE_certificate_v1.2.md.
Args: --v115 --cert12 --out."""
import argparse
import re

HEAD = "The lead's reading of these tables is turn 41's"


def main():
    ap = argparse.ArgumentParser()
    for k in ("v115", "cert12", "out"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    assert HEAD in open(a.cert12).read(), "the head line not found in COLLAPSE_certificate_v1.2.md"
    rows = [
        "| TEMPERATURE_solved_exterior_v1.2.md | read as TEMPERATURE_solved_exterior_v1.3.md (engine 35.1 (1), turn 43: the icloud lane's two literal precisions applied; version 1.2 stays the record; the icloud lane cleared version 1.3, inbox #22) |",
        "| version 1.15's three rows on TEMPERATURE_solved_exterior_v1.2.md (the umbrella row and the two clause rows) | closed: TEMPERATURE_solved_exterior_v1.3.md replaces those clauses (engine 35.1 (1)) |",
        "| COLLAPSE_certificate_statement_v1.2.md | read as COLLAPSE_certificate_statement_v1.3.md where the wording and windows change (engine 35.1 (2), turn 43; versions 1.1 and 1.2 stay the record; COLLAPSE_certificate_v1.2 stays the certificate) |",
        "| version 1.15's five rows on COLLAPSE_certificate_statement_v1.2.md (the umbrella row and the four clause rows) | closed: COLLAPSE_certificate_statement_v1.3.md replaces those clauses (engine 35.1 (2)) |",
        "| COLLAPSE_certificate_v1.2.md's head line \"%s\" | reads \"turn 42's and 43's (engine spec 34.3, 35.1; the statement COLLAPSE_certificate_statement_v1.3)\", the file's tables unchanged (engine 35.1 (3); the icloud lane's check of the certificate, inbox #18) |" % HEAD,
    ]
    t = open(a.v115).read()
    tm = re.search(r"^# RELABEL_table_v1\.15: (.*?)\(engine spec (.*?); the icloud lane's checks of the temperature v1\.2 and the certificate statement v1\.2, inbox #19, #20\)$", t, re.M)
    assert tm
    t = t.replace(tm.group(0), "# RELABEL_table_v1.16: %s(engine spec %s; the icloud lane's checks of the temperature v1.2 and the certificate statement v1.2, inbox #19, #20; 35.1)" % (tm.group(1), tm.group(2)), 1)
    pm = re.search(r"Version 1\.15 = version 1\.14 with ", t)
    assert pm
    t = t.replace(pm.group(0), "Version 1.16 = version 1.15 with %d rows added (engine 35.1 (3), turn 43: the temperature sentence and the certificate statement read as their versions 1.3, version 1.15's eight rows closed, the certificate head's turn line). Version 1.15 = version 1.14 with " % len(rows), 1)
    si = t.index("\n- Sources: ")
    t = t[:si].rstrip("\n") + "\n" + "\n".join(rows) + "\n" + t[si:]
    sm = re.search(r"\n- Sources: ([^\n]*)", t)
    assert sm
    t = t.replace(sm.group(0), "\n- Sources: " + sm.group(1).rstrip(".") + "; solve/specs/SPEC_final4b_engine.md sec. 35.1; deliveries/solve_a/TEMPERATURE_solved_exterior_v1.3.md, COLLAPSE_certificate_statement_v1.3.md; deliveries/solve_b/review_TEMPERATURE_solved_exterior_v1_3.tex.", 1)
    open(a.out, "w").write(t)
    print("wrote %s (%d rows added)" % (a.out, len(rows)))


if __name__ == "__main__":
    main()
