#!/usr/bin/env python
"""compose_ladder_v1_4_v2.py (builder analyst): version 1 with FILE 4's paragraph withdrawn in place of being set: FILE 4 states the case
in which the interim solve ends at T/rs 0.750 (34.1 (1), (2)), and the interim solve met the stop rule at 0.750 and continues, so by
34.1 (2) it is the member and the lead's paragraph for that case is turn 44's; a builder's line stating this replaces version 1.3's
sigma/a 2 paragraph (--withdrawn-line). Nothing else changes. Version 1: compose_ladder_v1_4_v1.py (builder analyst; engine spec 34.5 item 3: "COLLAPSE_ladder_v1.4 composed from FILE 4 (its sigma/a 2
paragraph replacing version 1.3's; the other head paragraphs verbatim), version 1.3's tables and the interim job's table
(compose_ladder_v1_4, a new file; re-run as the interim records change)"): assembles the file mechanically, nothing retyped: (1) a title
line naming the parts; (2) version 1.3's head (its lines after the title up to its first table heading), every paragraph verbatim except
the one beginning "The sigma/a 2 member (the Newton solve):", which is replaced by FILE 4's paragraph (its lines after its own title);
(3) version 1.3's table sections verbatim; (4) engine_ladder_compare_v5's tables with the interim solve (v32, --nk-restarts 2) as
sigma/a 2, read at the stated time. Args: --v13 --head-file --interim-tables --interim-time --out."""
import argparse
import os

OLD = "The sigma/a 2 member (the Newton solve):"


def main():
    ap = argparse.ArgumentParser()
    for k in ("v13", "head-file", "interim-tables", "interim-time", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--withdrawn-line", required=True)
    a = ap.parse_args()
    L = open(a.v13).read().rstrip("\n").split("\n")
    assert L[0].startswith("# COLLAPSE_ladder_v1.3:"), L[0][:60]
    hi = [i for i, l in enumerate(L) if l.startswith("## ")]
    assert hi
    head, body = L[1:hi[0]], L[hi[0]:]
    F = open(a.head_file).read().rstrip("\n").split("\n")
    assert F[0].startswith("# COLLAPSE_ladder_v1.4:"), F[0][:60]
    newp = [l for l in F[1:] if l.strip()]
    assert len(newp) == 1, "FILE 4 holds %d paragraphs" % len(newp)
    idx = [i for i, l in enumerate(head) if l.startswith(OLD)]
    assert len(idx) == 1, "version 1.3's sigma/a 2 paragraph found %d times" % len(idx)
    head = head[:idx[0]] + [a.withdrawn_line] + head[idx[0] + 1:]
    t = open(a.interim_tables).read().rstrip("\n").split("\n")
    assert t[0].startswith("## ")
    out = ["# COLLAPSE_ladder_v1.4: version 1.3's head with its sigma/a 2 paragraph withdrawn until the lead's turn 44 (the lead's turn 42 paragraph, COLLAPSE_ladder_v1.4_head_sigma2.md, states the case in which the interim solve ends at T/rs 0.750; the interim solve met the stop rule there and continues), the other paragraphs verbatim; version 1.3's tables verbatim; and the interim sigma/a 2 solve on engine_collapse_v32 with --nk-restarts 2 (40 Krylov vectors per cycle, 2 restarts; from T = 0 on the second H100 since 14:08:13 UTC) as engine_ladder_compare_v5 reads it at %s, re-run as its records change (composed by compose_ladder_v1_4_v2)" % a.interim_time]
    out += head + body + [""]
    out += ["## The interim sigma/a 2 solve on engine_collapse_v32 with --nk-restarts 2 at %s (%s)" % (a.interim_time, os.path.basename(a.interim_tables)), ""] + t[1:] + [""]
    open(a.out, "w").write("\n".join(out) + "\n")
    print("wrote %s: %d lines" % (a.out, len(out)))


if __name__ == "__main__":
    main()
