#!/usr/bin/env python
"""compose_certificate_statement_v1_4.py (the lane's analyst builder; engine spec 41.5 (2); the lead's turn 49 FILE 2,
COLLAPSE_certificate_statement_v1.4_amendments.md): version 1.4 = the cleared version 1.3 with the amendments' lines, nothing of version 1.3
changed: (1) the convention line and (2) the origin-term line inserted, in that order, after the certified-count sentence (the sentence ending
"is not certified (0.050 of the nodes in the tt row)."); (3) the scope line only if version 1.3 lacks it (version 1.3 is read as carrying it when
it holds "separately in each row", "the centre lies outside the retained support" and "not that the rows vanish"; the decision printed); (4) a
record section, not printable, with the summary lines of results/engine/CERT3v9/s5_rows_extract_v1.md copied verbatim under the heading
"S5 (a) and (b): read in version 1.5". The quoted lines are parsed from the amendments file, never retyped. The Condition's assertion: none
of requirements_new.txt sec. 4's replaced words in the composed file; if version 1.3's own text carries one, the compose stops (nothing of
version 1.3 may change) and the log lists each occurrence for the lead. Refuses to overwrite. Args: --v13 --amend --extract --out --log."""
import argparse
import os
import re
import sys

REPLACED = ("frozen", "freez", "window", "prediction")


def main():
    ap = argparse.ArgumentParser()
    for k in ("v13", "amend", "extract", "out", "log"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    for p in (a.out, a.log):
        if os.path.exists(p):
            sys.exit("refusing to overwrite %s" % p)
    v13 = open(a.v13).read(); am = open(a.amend).read(); ex = open(a.extract).read()
    q = {}
    for n in (1, 2, 3):
        m = re.search(r'^%d\. [^"]*"(.*)"\s*$' % n, am, re.M)
        if not m:
            sys.exit("amendment %d not parsed" % n)
        q[n] = m.group(1)
    L = ["## compose_certificate_statement_v1_4 (engine spec 41.5 (2))", "", "- inputs: %s; %s; %s" % (a.v13, a.amend, a.extract)]
    anchor = "is not certified (0.050 of the nodes in the tt row)."
    if v13.count(anchor) != 1:
        sys.exit("the certified-count sentence's end is not found once")
    has_scope = all(s in v13 for s in ("separately in each row", "the centre lies outside the retained support", "not that the rows vanish"))
    L.append("- amendment 3 (the scope line): version 1.3 %s; %s" % ("carries it (the three phrases present)" if has_scope else "lacks it", "not added" if has_scope else "added after line 2"))
    ins = " " + q[1] + " " + q[2] + ("" if has_scope else " " + q[3])
    body = v13.replace(anchor, anchor + ins, 1)
    summ = [ln for ln in ex.split("\n") if ln.startswith("- tt, ") or ln.startswith("- rr, ") or ln.startswith("- every number above")]
    body = body.rstrip("\n") + "\n\n## S5 (a) and (b): read in version 1.5\n\n(a record section, not printable: the summary lines of results/engine/CERT3v9/s5_rows_extract_v1.md, verbatim)\n\n" + "\n".join(summ) + "\n"
    head_old = body.split("\n", 1)[0]
    L.append("- inserted after the certified-count sentence: amendments 1 and 2%s; the record section with %d summary lines" % ("" if has_scope else " and 3", len(summ)))
    low = body.lower()
    occ = []
    for w in REPLACED:
        for m in re.finditer(w, low):
            occ.append("%s at offset %d: \"...%s...\"" % (w, m.start(), body[max(0, m.start() - 60):m.end() + 30].replace("\n", " ")))
    from_v13 = [o for o in occ]
    if occ:
        L += ["", "STOPPED: requirements_new.txt sec. 4's replaced words in the composed file, all inherited from version 1.3's own text (nothing of version 1.3 may change):"] + ["  - " + o for o in from_v13]
        open(a.log, "x").write("\n".join(L) + "\n")
        sys.exit("sec. 4 words in version 1.3's text: %d occurrences (log %s)" % (len(occ), a.log))
    open(a.out, "x").write(body)
    L.append("- composed: %s; none of sec. 4's replaced words" % a.out)
    open(a.log, "x").write("\n".join(L) + "\n")
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
