#!/usr/bin/env python
"""m325_port_measure_v1.py (busi builder, 20 Oct 2026; Reorientation 6 item 7): measures the port of v36g's completion (--completion point-core)
onto the merged Newton lineage v36y. Reads build_engine_collapse_v36g.py's REPLACEMENTS (exec of the module text before main(), no build), counts
each OLD in v36d and v36y, the lines each NEW adds, the v36g-marked lines of the built v36g over v36d, and prints where each OLD lands in v36y
with its enclosing def. Text only; no engine is imported or run.
Usage: python -I m325_port_measure_v1.py BUILDER V36D V36G V36Y"""
import difflib, sys
bld, fd, fg, fy = sys.argv[1:5]
src = open(bld).read(); ns = {}; exec(src[:src.index("\ndef main")], ns)
d = open(fd).read(); g = open(fg).read(); y = open(fy).read()
yl = y.splitlines()
def enclosing(text, pos):
    lines = text[:pos].splitlines()
    for l in reversed(lines):
        if l.startswith("def "):
            return l.split("(")[0][4:]
    return "<module>"
tot = 0
for name, old, new in [r[:3] for r in ns["REPLACEMENTS"]]:
    add = len(new.splitlines()) - len(old.splitlines()); tot += add
    py = y.find(old); pd_ = d.find(old)
    print("%-38s in v36d %d (def %s), in v36y %d at line %d (def %s), adds %d lines" % (name, d.count(old), enclosing(d, pd_), y.count(old),
          y[:py].count("\n") + 1 if py >= 0 else -1, enclosing(y, py) if py >= 0 else "-", add))
print("replacements %d, lines added by the replacements %d" % (len(ns["REPLACEMENTS"]), tot))
dl = list(difflib.unified_diff(d.splitlines(), g.splitlines(), lineterm="", n=0))
plus = [l for l in dl if l.startswith("+") and not l.startswith("+++")]
minus = [l for l in dl if l.startswith("-") and not l.startswith("---")]
print("v36g over v36d: %d lines added or changed, %d removed or changed; added lines carrying 'v36g W3': %d" % (len(plus), len(minus), sum("v36g W3" in l for l in plus)))
print("v36y mentions of --completion / point-core / CPL[: %d / %d / %d" % (y.count("--completion"), y.count("point-core"), y.count("CPL[")))
for i, l in enumerate(yl, 1):
    if l.lstrip().startswith("for step in "):
        print("v36y line %d (def %s): %s" % (i, enclosing(y, len("\n".join(yl[:i - 1]))), l.strip()[:120]))
