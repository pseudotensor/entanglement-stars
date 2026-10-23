#!/bin/bash
# Strand 2b, result revision 3 (assembly v7): adds the sec. 14 item 2 cube uniqueness rerun (c8unique1) and its post-processing
# (c8unique1post) as a section after the sec. 13 check; the rest of revision 2's text is kept verbatim (revision 2 is in git history).
set -eu
W=${QI2_ROOT}/.wt-solve_a
B=${QI2_ROOT}/s2b_row
OUT=$W/solve/results/strand2b
PY=${QI2_ROOT}/python
mkdir -p $OUT/c8unique1 $OUT/c8unique1post
timeout 300 ssh -n -o BatchMode=yes compute-host "cd $B && OMP_NUM_THREADS=1 $PY code/strand2b_unique_report_v1.py --dir-unique c8unique1 --dir-post c8unique1post --dir-records c8pass1v5/enclosure --md c8unique1post/RESULT_unique_tables_v1.md > c8unique1post/report_v1.log 2>&1"
for f in status.txt unique_k1.log unique_k10000.log unique_C8_b20_m0.1_gM0.2_gB1_L1_k1.json unique_C8_b20_m0.1_gM0.2_gB1_L1_k10000.json; do
  timeout 120 scp -q compute-host:$B/c8unique1/$f $OUT/c8unique1/ || echo "missing c8unique1/$f"; done
for f in status.txt post_k1.log post_k10000.log post_C8_b20_m0.1_gM0.2_gB1_L1_k1.json post_C8_b20_m0.1_gM0.2_gB1_L1_k10000.json RESULT_unique_tables_v1.md; do
  timeout 120 scp -q compute-host:$B/c8unique1post/$f $OUT/c8unique1post/ || echo "missing c8unique1post/$f"; done
R=$W/solve/results/RESULT_strand2b_row.md
python3 - "$R" "$OUT/c8unique1post/RESULT_unique_tables_v1.md" <<'PYEOF'
import sys
R, T = sys.argv[1], sys.argv[2]
s = open(R).read()
tables = open(T).read()
old_title = "# Strand 2b result, revision 2 (tests T3, T6): the commuting static row on R4 and on the cube\n"
assert s.startswith(old_title), "unexpected title"
s = s.replace(old_title, "# Strand 2b result, revision 3 (tests T3, T6): the commuting static row on R4 and on the cube\n", 1)
anchor = "the sec. 12(c) columns. Revision 1 (R4 pass 1 alone) is in git history; its run record and readings carry over unchanged.\n"
assert s.count(anchor) == 1
s = s.replace(anchor, anchor + "Revision 3 adds the sec. 14 item 2 cube uniqueness rerun and its post-processing (the section after the sec. 13 check);\n"
              "revision 2's text is otherwise unchanged and is in git history.\n", 1)
sec = """## Sec. 14 item 2: the cube uniqueness rerun (strand2b_c8_unique_v1.py, launch_strand2b_C8_unique_v1.sh, c8unique1/; post-processing strand2b_c8_unique_post_v1.py, launch_strand2b_C8_unique_post_v1.sh, c8unique1post/; tables strand2b_unique_report_v1.py)

- Box compute-host, CPU: the starts in 8 single-thread workers, the post-processing's enclosure Jacobians in 12; the two enclosure cases of
  sec. 12(b) ((2,0), m 0.1, (g_M, g_B, Lambda) = (0.2, 1, 1), k = 1 and 1e4) with the projector rank fixed at the case's flat-start value.
  Records in solve/results/strand2b/c8unique1/ and c8unique1post/.
- Builder note 1: the run accepted a limit at residual 1e-11 (the library default). At k = 1 that is below the case's own evaluation
  noise (the record's residual_noise_floor, U1), and the continuation solution itself sits above it, so no start was accepted there at
  1e-11. The post-processing counts every positive limit at or below the record's noise floor, the standard the continuation solution
  meets, and groups the limits into orbits of the cube's 48 symmetries (signed permutations of the centred positions acting on the site
  labels); each orbit's representative was checked with the case driver's objects and enclosure design (U2).
- Builder note 2: at k = 1 the uniform continuation solution was reached from none of the 16 starts; its U2 row is recomputed by the same
  script as a control and reproduces the record's enclosure values (T2 of the cube enclosure section below). The two starts not at
  the floor at k = 1 ended at residuals of order one (U1).
- For the lead's reading under sec. 6 D1 (multiple distinct positive solutions: report all).

""" + tables.replace("### U", "### Sec. 14 U") + "\n"
anchor2 = "## T1 Solutions (min over sites of psi and N shown; full vectors in the JSON; on C8 the final residual is the record's residual)\n"
i = s.index(anchor2)
s = s[:i] + sec + s[i:]
open(R, "w").write(s)
print("inserted at", i)
PYEOF
