#!/bin/bash
# Strand 2b, result revision 4 (assembly v8): adds sec. 15 items 1, 2 and 4 (strand2b_sec15_v1.py, strand2b_sec15_item4_v1.py,
# strand2b_sec15_report_v1.py) as a section after the sec. 14 item 2 section; items 3 and 5 follow in a later revision.
set -eu
W=${QI2_ROOT}/.wt-solve_a
B=${QI2_ROOT}/s2b_row
OUT=$W/solve/results/strand2b
PY=${QI2_ROOT}/python
mkdir -p $OUT/sec15run1 $OUT/sec15item4
timeout 300 ssh -n -o BatchMode=yes compute-host "cd $B && OMP_NUM_THREADS=1 $PY code/strand2b_sec15_report_v1.py --items12 sec15run1/sec15_items12.json --item4 sec15item4/item4.json --md sec15run1/RESULT_sec15_tables_v1.md > sec15run1/report_v1.log 2>&1"
for f in status.txt run.log sec15_items12.json RESULT_sec15_tables_v1.md; do timeout 120 scp -q compute-host:$B/sec15run1/$f $OUT/sec15run1/ || echo "missing sec15run1/$f"; done
timeout 120 scp -q compute-host:$B/sec15item4/item4.json $OUT/sec15item4/ || echo "missing item4.json"
R=$W/solve/results/RESULT_strand2b_row.md
python3 - "$R" "$OUT/sec15run1/RESULT_sec15_tables_v1.md" <<'PYEOF'
import sys
R, T = sys.argv[1], sys.argv[2]
s = open(R).read()
tables = open(T).read()
old_title = "# Strand 2b result, revision 3 (tests T3, T6): the commuting static row on R4 and on the cube\n"
assert s.startswith(old_title), "unexpected title"
s = s.replace(old_title, "# Strand 2b result, revision 4 (tests T3, T6): the commuting static row on R4 and on the cube\n", 1)
anchor = "revision 2's text is otherwise unchanged and is in git history.\n"
assert s.count(anchor) == 1
s = s.replace(anchor, anchor + "Revision 4 adds sec. 15 items 1, 2 and 4 (the section after the sec. 14 item 2 section); items 3 and 5 follow.\n", 1)
sec = """## Sec. 15 items 1, 2 and 4 (strand2b_sec15_v1.py and strand2b_sec15_item4_v1.py, tables strand2b_sec15_report_v1.py; s2b_row/sec15run1, sec15item4)

- Box compute-host, CPU, seconds: code/strand2b_sec15_v1.py --out sec15run1 --r4 pass1 --c8 c8pass1v5; code/strand2b_sec15_item4_v1.py
  --jacgap jacgap2/jacgap_v2.json --out sec15item4. Records in solve/results/strand2b/sec15run1/ and sec15item4/.
- Item 1: the two cases are the pass-1 records R4_b10_m0.1_gM0_gB0_L1_k1 and R4_b11_m0.1_gM0.05_gB0_L1_k1 (both d_0 = 2); the clock
  component raised is N_0; the corrected projector response is a subclass in the test script (w_i - w_mm in the denominators, the
  smallest such distance returned); strand2b_lib_v6.py is unchanged (item 3 makes the repaired library a new file). J_FD is the central
  difference of the complete F (state and d rho^P re-evaluated at each displaced point).
- Item 2: E, E' and E'' are the k = 1 record's E_block(1), dE/db and d^2E/db^2 at h = 1e-4 (k-independent); the fitted c_1 is the
  report's fit_k over k in {10, 100, 1e3, 1e4}. In the (0,0) rows the block energy is zero to rounding, so the closed form and the fit
  both vanish there and their ratio is not meaningful.
- Item 4: as specified; the opposite sign is reported beside it.

""" + tables.replace("### Sec. 15", "### Sec. 15") + "\n"
anchor2 = "## T1 Solutions (min over sites of psi and N shown; full vectors in the JSON; on C8 the final residual is the record's residual)\n"
i = s.index(anchor2)
s = s[:i] + sec + s[i:]
open(R, "w").write(s)
print("inserted at", i)
PYEOF
