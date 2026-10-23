#!/usr/bin/env python
"""compose_ladder_rs32_v1.py (builder analyst; engine spec 34.1 (3): "When both hold records through T/rs 21 the builder runs
engine_ladder_compare_v5 on the two members (the matched-time table every 0.5 r_s plus the formation clocks; 29.1's criterion under
30.3's rule per record) into COLLAPSE_ladder_rs32_v1.md with this head, copied"; 36.2 (4) and 37.9 keep it): the title line, the lead's
head extracted verbatim from the spec (the quoted text after "into COLLAPSE_ladder_rs32_v1.md with this head, copied:"), then the
comparison file engine_ladder_compare_v5 printed, verbatim. Checks: the head found exactly once; every member's json holds a record at
T/rs 21 or later; the output does not exist. Args: --spec --compare-md --member-json (repeatable) --out."""
import argparse
import datetime
import json
import os
import re
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True); ap.add_argument("--compare-md", required=True)
    ap.add_argument("--member-json", action="append", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit("refusing to overwrite %s" % a.out)
    s = open(a.spec).read()
    hs = re.findall(r'into COLLAPSE_ladder_rs32_v1\.md with this head, copied: "(.*?)" Item 5', s, re.S)
    if len(hs) != 1:
        sys.exit("the head found %d times" % len(hs))
    last = []
    for p in a.member_json:
        r = json.load(open(p))["records"]
        last.append((os.path.basename(p), len(r), r[-1]["T_over_rs"]))
        if r[-1]["T_over_rs"] < 21.0:
            sys.exit("%s ends at T/rs %g, before 21" % (p, r[-1]["T_over_rs"]))
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M")
    title = ("# COLLAPSE_ladder_rs32_v1: the r_s/a 32 fixed-preparation pair (engine spec 34.1 (3); the lead's head copied verbatim, the tables "
             "printed by engine_ladder_compare_v5 and copied verbatim, composed by compose_ladder_rs32_v1 at %s UTC from the members' jsons: %s; "
             "builder analyst; the builder copies, never edits)" % (now, "; ".join("%s, %d records to T/rs %g" % x for x in last)))
    cmp_ = open(a.compare_md).read()
    open(a.out, "w").write(title + "\n\n" + hs[0] + "\n\n" + cmp_.rstrip("\n") + "\n")
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
