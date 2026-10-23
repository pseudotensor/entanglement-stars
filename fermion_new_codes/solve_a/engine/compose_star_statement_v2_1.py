#!/usr/bin/env python
"""compose_star_statement_v2_1.py (builder analyst; engine spec 27.4: "the statement's last sentence replaced by: "..." (STAR_family_statement_v2.1.md,
BUILDER.)"): composes STAR_family_statement_v2.1 mechanically from the delivered version 2: its last sentence (from "(0.1, 1/16) has converged"
to the end) replaced verbatim by the quoted text of spec 27.4, the header line naming the version. Args: --v2 --spec --out."""
import argparse
import re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", required=True); ap.add_argument("--spec", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    v2 = open(a.v2).read().rstrip("\n"); spec = open(a.spec).read()
    sec = spec[spec.index("## 27."):]
    m = re.search(r'the statement\'s last sentence replaced by: "(.*?)" \(STAR_family_statement_v2\.1\.md, BUILDER\.\)', sec, re.S)
    assert m, "27.4's replacement text not found"
    new = m.group(1)
    start = "(0.1, 1/16) has converged"
    assert v2.count(start) == 1, "the last sentence's start is not unique"
    i = v2.index(start)
    assert "\n" not in v2[i:], "the last sentence is not the file's last line"
    lines = v2[:i].split("\n")
    assert lines[0].startswith("# The static star families: the lead's statement for handoff version 2 ("), lines[0]
    lines[0] = ("# The static star families: the lead's statement for handoff version 2.1 (engine spec 27.4: version 2's statement, spec 26.4, with its last "
                "sentence replaced by 27.4's text; the builder's tables follow it, printed by engine_star_handoff_v3)")
    out = "\n".join(lines) + new + "\n"
    open(a.out, "w").write(out)
    print("old last sentence: %s" % v2[i:]); print("new last sentence: %s" % new); print("%d chars" % len(out))


if __name__ == "__main__":
    main()
