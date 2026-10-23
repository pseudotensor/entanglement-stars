#!/usr/bin/env python
"""compose_j3p_labels_turn41_v1.py (builder analyst; engine spec 33.2 (2): "Labels, fixed now and carried in the version 9 files' heads
by composition"): for each run named there, a new md beside the record (suffix _l41) that is the record's title line, one label line
quoting 33.2 (2) (read from the spec by this script, nothing retyped), then the record verbatim. The 36.25 run: J3P_raymap_v9_tau4_swingmax;
the 39.5 run: J3P_raymap_v8_tau4_min (no version 9 exists for it); the production's maximum: J3P_raymap_v9_prod19_max; its minimum:
J3P_raymap_v8_prod19_min (the step-partner sentence on both). Args: --spec --j3dir."""
import argparse
import os
import re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True); ap.add_argument("--j3dir", required=True)
    a = ap.parse_args()
    t = open(a.spec).read()
    sec = t[t.index("### 33.2"):t.index("### 33.3")]
    m = re.search(r"\(2\) Labels, fixed now and carried in the version 9 files' heads by composition: the 36\.25 run \"([^\"]+)\"; the 39\.5 run \"([^\"]+)\"; (a production run's step partner is the tau 4 run at the matched phase \([^)]*\))\.", sec)
    assert m, "33.2 (2) not found"
    l3625, l395, lstep = m.group(1), m.group(2), m.group(3)
    jobs = [("J3P_raymap_v9_tau4_swingmax.md", "the 36.25 run: \"%s\"" % l3625),
            ("J3P_raymap_v8_tau4_min.md", "the 39.5 run: \"%s\"" % l395),
            ("J3P_raymap_v9_prod19_max.md", "the production's maximum: %s" % lstep),
            ("J3P_raymap_v8_prod19_min.md", "the production's minimum: %s" % lstep)]
    for f, lab in jobs:
        p = os.path.join(a.j3dir, f)
        L = open(p).read().split("\n")
        assert L[0].startswith("#"), p
        out = os.path.join(a.j3dir, f[:-3] + "_l41.md")
        assert not os.path.exists(out), out
        open(out, "w").write("\n".join([L[0], "", "- label (engine spec 33.2 (2), the lead's turn 41; composed by compose_j3p_labels_turn41_v1 onto %s, whose body follows verbatim): %s" % (f, lab)] + L[1:]))
        print("wrote %s" % out)


if __name__ == "__main__":
    main()
