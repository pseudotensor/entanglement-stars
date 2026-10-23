#!/usr/bin/env python
"""compose_ladder_v1_6_v1.py (solve builder 9, analyst; the lead's instruction after turn 59: the busi lane's reviewer A API cold reading of
COLLAPSE_ladder_v1.5, CLEARED WITH CHANGES): COLLAPSE_ladder_v1.6 = the cold reading's items 1 to 3 (their quoted texts verbatim) in the head,
its items 4 to 11 (the red team's applied wordings) verbatim as the corrections carried, readings_48_1_3_v1.md whole with the inner-root
difference column item 1 asks for (v32 minus tightened, from the table's 4-decimal roots), a factual status line for turn 59 beside item 2,
and version 1.5 below as the record (never overwritten). Light parsing only."""
import re
import sys

READ, V15, RD, OUT = sys.argv[1:5]
L = open(READ).read().split("\n")
items = {}
for ln in L:
    m = re.match(r"^(\d+)\. (.*)$", ln)
    if m:
        items[int(m.group(1))] = m.group(2)
q = lambda s: re.findall(r'"(.*?)"(?=[ ,.;]|$)', s)


def quoted(n):
    s = items[n]; i = s.index('"'); j = s.rindex('"')
    return s[i + 1:j]


def inner_col(text):
    out = []; inroots = False
    for ln in text.split("\n"):
        if ln.startswith("## "):
            inroots = ln.startswith("## Roots")
        if inroots and ln.startswith("| T/r_s | v32: zeros"):
            ln = ln + " inner root difference |"
        elif inroots and ln.startswith("|---|"):
            ln = ln + "---|"
        elif inroots and re.match(r"^\| 1[12]\.\d{4} \|", ln):
            c = [x.strip() for x in ln.strip("|").split("|")]
            a, b = c[1], c[4]
            ln = ln + ((" %+.2e |" % (float(a.split(",")[0]) - float(b.split(",")[0]))) if (a != "none" and b != "none") else " - |")
        out.append(ln)
    return "\n".join(out)


rd = inner_col(open(RD).read())
v15 = open(V15).read()
head = ["# COLLAPSE_ladder_v1.6: the busi lane's cold reading of version 1.5 (CLEARED WITH CHANGES) applied: its items 1 to 3 in the head, its items 4 to 11 carried; version 1.5 below as the record",
        "",
        "Version 1.5 stands as the record: the snapshot on the v32 member to T/r_s 18.125, composed under turn 57's 49.4 (1). This version applies <internal>/ReviewA_cold_reading_COLLAPSE_ladder_v1.5.md (red team: 14 defects, every one applied in its wording; the item texts below are the reading's, verbatim).",
        "",
        "## Head",
        "",
        "(Item 1, the solver-spread sentence.) " + quoted(1),
        "",
        "(Item 2, the point line.) " + quoted(2),
        "",
        "Status of item 2 under turn 59 (engine spec 51.1, facts only): the continuation on engine_collapse_v36b ended at T/r_s 16.875 by the engine's end rule (direct solve kept 3.45e-03, stage 2 kept 3.35e-03, the occupied set changing there); its records 16.125 to 16.875 are kept on /data as the stall's record. The 16.125 re-solve under the tightened rule converged (kept 2.19e-12) with the inner root 1.40e-07 from the member's. The point's run is now the relaunch from the same upcast T0016 slice on engine_collapse_v36c --nk-set resolve (the set-resolving Newton solve; off gate bitwise against v36b over 136 records), labelled \"continued on v36c (--nk-set resolve)\"; until it passes 17.5 within its gate, the member's stored records to 18.125 are the history. The seam table and engine_ladder_compare_v5 on the composed history go into this file's next version once its records land. The stored-tolerance v36b history from T = 0 on pod 1 is retired at the owner's word, its records to T/r_s 1.25 and its restart r1 to 1.875 and its T/r_s 1 slice kept (<internal>/v36b_member_retired). The member's last record is 18.125 (146 records), against the LEAD entry's 17.5.",
        "",
        "(Item 3, the factor-4 reading.) " + quoted(3),
        "",
        "## Corrections carried (the cold reading's items 4 to 11, verbatim)",
        ""]
for n in range(4, 12):
    head += ["%d. %s" % (n, items[n]), ""]
head += ["## readings_48_1_3_v1.md, whole (item 3), with the inner-root difference column (item 1; v32 minus tightened, from the 4-decimal roots)", "", rd, "",
         "## Version 1.5 (the record, unchanged)", ""]
open(OUT, "w").write("\n".join(head) + "\n" + v15)
print("written", OUT)
