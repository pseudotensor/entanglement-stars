#!/usr/bin/env python
"""compose_relabel_v1_10.py (builder analyst; engine spec 32.6 item 3: "RELABEL_table_v1.10 (COLLAPSE_ladder_v1 and v1.1 read as v1.2;
ITEM5 v4 as v5; version 1.9's two withdrawn-sentence rows closed by v5; ASYMPTOTICS' kernel labels)"): version 1.9 with the title's source
list and the sources line extended and four rows added before the sources line. Args: --v19 --out."""
import argparse
import re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v19", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t = open(a.v19).read()
    tm = re.search(r"^# RELABEL_table_v1\.9: (.*?)\(engine spec (.*?); the hreduce lane's DM7 review, inbox #11\)$", t, re.M)
    assert tm
    t = t.replace(tm.group(0), "# RELABEL_table_v1.10: %s(engine spec %s; the hreduce lane's DM7 review, inbox #11; 32.6)" % (tm.group(1), tm.group(2)), 1)
    pm = re.search(r"Version 1\.9 = version 1\.8 with ", t)
    assert pm
    t = t.replace(pm.group(0), "Version 1.10 = version 1.9 with four rows added (engine 32.6 item 3: the ladder and item 5 relabels of turn 40, version 1.9's two withdrawal rows closed, the checks table's kernel labels). Version 1.9 = version 1.8 with ", 1)
    rows = ["| COLLAPSE_ladder_v1.md and COLLAPSE_ladder_v1.1.md | read as COLLAPSE_ladder_v1.2.md (engine 32.1 (1), turn 40: the lead's head on the full overlap, engine_ladder_compare_v4's tables) |",
            "| ITEM5_collapse_sentences_v4.md | read as ITEM5_collapse_sentences_v5.md (engine 32.1 (2), turn 40) |",
            "| version 1.9's rows withdrawing ITEM5_collapse_sentences_v4's sentence 3 and sentence 4's averaging-scale clause | closed: ITEM5_collapse_sentences_v5.md's sentences 3 and 4 replace them (turn 40) |",
            "| ASYMPTOTICS_checks_v1.md to _v1.3.md: the kernel labels sigma/a 4.010, 8.005 and 16.003 and the variances 16.083333 and 64.083333 | read sigma/a 4, 8 and 16 with sigma^2 = ell^2 (engine 32.3 (3)); obs_j3p_scope_v2's rows in row (3) are computed at ell^2 + a^2/12 (ASYMPTOTICS_checks_v1.4 marks them) |"]
    si = t.index("\n- Sources: ")
    t = t[:si].rstrip("\n") + "\n" + "\n".join(rows) + "\n" + t[si:]
    sm = re.search(r"\n- Sources: ([^\n]*)", t)
    assert sm
    t = t.replace(sm.group(0), "\n- Sources: " + sm.group(1).rstrip(".") + "; solve/specs/SPEC_final4b_engine.md secs. 32.1, 32.3, 32.6; deliveries/solve_a/COLLAPSE_ladder_v1.2.md, ITEM5_collapse_sentences_v5.md.", 1)
    open(a.out, "w").write(t)
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
