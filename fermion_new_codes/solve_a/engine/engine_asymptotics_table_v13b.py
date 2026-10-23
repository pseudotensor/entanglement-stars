#!/usr/bin/env python
"""engine_asymptotics_table_v13b.py (the lane's analyst builder; engine spec 40.5 (2), the lead's turn 48 FILE 1,
ASYMPTOTICS_row3_v1.9_replacements_v2.md): engine_asymptotics_table_v13.py with the version 2 map: items 1, 3, 4 and 5 of the version 1 map
as written there (counts 1, 1, 2, 1; old and new strings parsed from the version 1 map's own quotes), item 2's old string the version 1
map's and its new string the version 2 map's quote ("Item 2's new string reads ..."), count 1; and the version 2 map's item 3: one added
first line, its quoted text, no replacement. Every v13 assertion stands with the added line accounted for (the composed file's lines after
the first equal version 1.8's except the six occurrence lines 474, 476, 482 twice, 487, 494). Refuses to overwrite.
Args: --v18 --map (the version 1 map) --map2 (the version 2 map) --log12 --out --log. v13's docstring follows.
engine_asymptotics_table_v13.py (builder analyst; engine spec 39.3 (4): "FILE 4 now: ASYMPTOTICS_checks_v1.9 by
engine_asymptotics_table_v13 (new), the six carried occurrences replaced literally, counts 1 each, nothing else changed; version 1.8
stays delivered until 1.9 lands"; FILE 4, ASYMPTOTICS_row3_v1.9_replacements.md: "composes version 1.9 from the delivered version 1.8
with these alone, each count printed and asserted as stated, and asserts that none of requirements_new.txt sec. 4's replaced words
remains anywhere in the composed file"; the lane's instruction of 2026-10-17 about 00:50 UTC: "the counts as stated: 1, 1, 1, 2, 1"):
the delivered version 1.8 with FILE 4's five numbered replacements (six occurrences) applied literally, each old and new string and each
stated count parsed from FILE 4's own quotes, never retyped. Asserted, each a stop on failure: the stated counts are 1, 1, 1, 2, 1; each
old string's count in version 1.8, and at its turn, equals its stated count; the occurrences' lines are the six that
compose_asymptotics_v1_8_v12.md lists (474, 476, 482 twice, 487, 494); the lines changed are exactly those lines and every other line of
version 1.8 is unchanged (the head and the version line included: "with these alone", "nothing else changed"); none of sec. 4's replaced
words ("frozen", "freez", "window", "prediction", any case) and none of the further sec. 4 phrases (FORBIDDEN) anywhere in the composed
file. The remaining occurrences of the stem "predict" (none of them sec. 4's word "prediction") are listed by line for the lead,
unchanged. Refuses to overwrite. Args: --v18 --map --log12 --out --log."""
import argparse
import hashlib
import os
import re
import sys

REPLACED = ("frozen", "freez", "window", "prediction")
FORBIDDEN = ("toy model", "emergent gravity", "derived einstein", "derived from the fermions", "semiclassical", "mean-field", "mean field",
             "discretized", "an event horizon forms", "a singularity forms", "no black object", "obligation", "bonds keep their strength",
             "bonds always cross", "unitary completion", "rectangle", "allocation exponent", "uv complete", "near-maximal", "remnant",
             "page fraction", "page curve", "stationary luminosity", "static star", "exact for irrotational", "must radiate at t_h",
             "set by the masses", "0.99965", "(a/r)^2")
STATED = [1, 1, 1, 2, 1]


def lines_of(text, s):
    out, i = [], text.find(s)
    while i >= 0:
        out.append(text.count("\n", 0, i) + 1)
        i = text.find(s, i + 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    for k in ("v18", "map", "map2", "log12", "out", "log"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    for p in (a.out, a.log):
        if os.path.exists(p):
            sys.exit("refusing to overwrite %s" % p)
    md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()  # noqa: E731
    L = ["## compose_asymptotics_v1_9_v13b (engine_asymptotics_table_v13b; engine spec 39.3 (4), 40.5 (2)): ASYMPTOTICS_checks_v1.9 from version 1.8 with the version 2 map alone", "",
         "- inputs: version 1.8 %s (md5 %s); the version 1 map %s (md5 %s); the version 2 map %s (md5 %s); %s (md5 %s)" % (a.v18, md5(a.v18), a.map, md5(a.map), a.map2, md5(a.map2), a.log12, md5(a.log12)), ""]

    def stop(msg):
        open(a.log, "x").write("\n".join(L) + "\n\nSTOPPED: %s\n" % msg)
        sys.exit(msg)

    t18 = open(a.v18).read()
    items = []
    for ln in open(a.map).read().split("\n"):
        m = re.match(r'^(\d)\. "([^"]*)" reads "([^"]*)" \(count (\d+)[;)]', ln)
        if m:
            items.append((int(m.group(1)), m.group(2), m.group(3), int(m.group(4))))
    m2 = open(a.map2).read()
    mm = re.search(r'''^2\. Item 2's new string reads "([^"]*)" \(count (\d+);''', m2, re.M)
    if not mm or int(mm.group(2)) != 1:
        stop("the version 2 map's item 2 is not as stated")
    items = [(n, o, (mm.group(1) if n == 2 else w), c) for n, o, w, c in items]
    ma = re.search(r'^3\. The composed file begins with one added line, new text, no replacement: "([^"]*)"', m2, re.M)
    if not ma:
        stop("the version 2 map's item 3 is not found")
    FIRST = ma.group(1)
    L.append("- the version 2 map: item 2's new string \"%s\" (count 1); the added first line \"%s\"" % (mm.group(1), FIRST))
    L.append("- the numbered replacements: %d, numbered %s, stated counts %s (the lane's statement: %s)" % (
        len(items), [x[0] for x in items], [x[3] for x in items], STATED))
    if [x[0] for x in items] != [1, 2, 3, 4, 5] or [x[3] for x in items] != STATED:
        stop("FILE 4's replacements or counts are not as stated")
    listed = []
    for ln in open(a.log12).read().split("\n"):
        m = re.match(r"^  - line (\d+): (\w+), in ", ln)
        if m:
            listed.append((int(m.group(1)), m.group(2)))
    L.append("- compose_asymptotics_v1_8_v12.md's listed occurrences: %s" % "; ".join("line %d %s" % x for x in listed))
    L += ["", "| item | old | new | stated count | count in version 1.8 | count at its turn | its lines in version 1.8 |", "|---|---|---|---|---|---|---|"]
    t = t18
    occ_lines = []
    for n, old, new, cnt in items:
        c0 = t18.count(old); c1 = t.count(old); ls = lines_of(t18, old)
        occ_lines += ls
        L.append("| %d | %s | %s | %d | %d | %d | %s |" % (n, old, new, cnt, c0, c1, ", ".join(str(x) for x in ls)))
        if c0 != cnt or c1 != cnt:
            stop("item %d counted %d in version 1.8 and %d at its turn, stated %d" % (n, c0, c1, cnt))
        t = t.replace(old, new)
    if sorted(occ_lines) != sorted(x[0] for x in listed):
        stop("the occurrences' lines %s are not the v12 log's %s" % (sorted(occ_lines), sorted(x[0] for x in listed)))
    L.append("- the occurrences' lines %s equal the v12 log's listed lines: yes" % sorted(occ_lines))
    t = FIRST + "\n" + t
    a18, a19 = t18.split("\n"), t.split("\n")[1:]
    if len(a18) != len(a19):
        stop("the line counts differ: %d and %d" % (len(a18), len(a19)))
    changed = [i + 1 for i, (x, y) in enumerate(zip(a18, a19)) if x != y]
    L.append("- lines changed: %s; every other of version 1.8's %d lines unchanged (the head and the version line included)" % (changed, len(a18)))
    if changed != sorted(set(occ_lines)):
        stop("the changed lines %s are not the occurrences' lines %s" % (changed, sorted(set(occ_lines))))
    low = t.lower()
    hits = {w: low.count(w) for w in REPLACED + FORBIDDEN if w in low}
    L.append("- sec. 4's replaced words (%s) and the further sec. 4 phrases (%s) anywhere in the composed file, any case: %s" % (
        ", ".join(REPLACED), ", ".join(FORBIDDEN), "none" if not hits else "FOUND %s" % hits))
    if hits:
        stop("sec. 4 words remain: %s" % hits)
    pred = []
    for i, ln in enumerate(t.split("\n"), 1):
        for m in re.finditer(r"(?i)predict\w*", ln):
            pred.append("line %d: %s, in \"%s\"" % (i, m.group(0), ln[max(0, m.start() - 60):m.end() + 40]))
    L.append("- for the lead, unchanged (FILE 4 has no replacement for them; not sec. 4's word \"prediction\"): the stem \"predict\" remains %d times%s" % (
        len(pred), (": " + "; ".join(pred)) if pred else ""))
    L += ["", "### the six changed lines, as composed (the added first line above them)", ""]
    for i in changed:
        L.append("- line %d of version 1.8 (line %d of version 1.9): %s" % (i, i + 1, a19[i - 1].replace("|", "/")))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "x").write(t)
    L += ["", "- composed: %s (md5 %s), %d lines; nothing else of version 1.8 changed; every count above printed by the script from %s and %s." % (
        a.out, hashlib.md5(t.encode()).hexdigest(), len(a19) + 1, a.v18, a.map + " and " + a.map2)]
    open(a.log, "x").write("\n".join(L) + "\n")
    print("wrote %s and %s" % (a.out, a.log))


if __name__ == "__main__":
    main()
