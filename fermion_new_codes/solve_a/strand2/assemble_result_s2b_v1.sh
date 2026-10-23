#!/bin/bash
# Strand 2b, result revision 1 (R4 pass 1): generate the timing-case tables on the box, fetch pass 1's tables, summary, logs and
# records, and assemble solve/results/RESULT_strand2b_row.md (text assembly only; numbers come from the generated files).
set -eu
W=${QI2_ROOT}/.wt-solve_a
B=${QI2_ROOT}/s2b_row
OUT=$W/solve/results/strand2b
PY=${QI2_ROOT}/python
mkdir -p $OUT/pass1 $OUT/timing7b
timeout 300 ssh -n -o BatchMode=yes compute-host "cd $B && OMP_NUM_THREADS=1 nice -n 10 $PY code/strand2b_report_R4_v3.py --run timing7b --baseline strand1_RESULTS_native_kscan_tables.md --md timing7b/RESULT_timing7b_tables.md > timing7b/report_v3.log 2>&1 && tar czf pass1_records.tar.gz pass1/case_*.json pass1/native/native_q*.json pass1/native/native_ruler_variance.json"
for f in status.txt batch.log RESULT_strand2b_R4_pass1_tables_v3.md SUMMARY_pass1_v2.md native_variance.log report_v3.log native_cases.txt; do
  timeout 120 scp -q compute-host:$B/pass1/$f $OUT/pass1/ || echo "missing pass1/$f"
done
timeout 120 scp -q compute-host:$B/pass1/native/native.log $OUT/pass1/native.log || echo "missing native.log"
timeout 120 scp -q compute-host:$B/pass1_records.tar.gz $OUT/pass1/ || echo "missing records"
for f in case.log case_R4_b21_m0.1_gM0.2_gB1_L1_k1.json RESULT_timing7b_tables.md; do
  timeout 120 scp -q compute-host:$B/timing7b/$f $OUT/timing7b/ || echo "missing timing7b/$f"
done
timeout 120 scp -q compute-host:$B/timing7a/case.log $OUT/timing7b/timing7a_case.log || echo "missing timing7a log"
R=$W/solve/results/RESULT_strand2b_row.md
{
echo "# Strand 2b result, revision 1 (tests T3, T6): the commuting static row on R4, pass 1 (solutions and lock pieces at every case)"
echo
echo "Builder: analyst. Spec: solve/specs/SPEC_strand2b_row.md with the lead's rulings in secs. 9-11. Numbers, controls and"
echo "commands only; no physics reading. Revision 1 covers R4 pass 1 (no enclosure, no uniqueness scan) and the sec. 11.3 native"
echo "control; pass 2 (enclosure and uniqueness, scope from the lead) and the C8 cases follow as revisions."
echo
echo "## Inputs"
echo "- Code (solve/code/strand2/): strand2b_lib_v6.py (the reduced static system in z = (psi, N): rho^P_x[psi] on the block in"
echo "  the fixed c-basis of the Fock sector, the state rho = P_low/d_0 eliminated, the adjoint row, Newton, continuation),"
echo "  strand2b_row_R4_v8.py (one case: continuation, Newton, the lock pieces, control C13; enclosure and uniqueness optional),"
echo "  strand2b_batch_R4_v8.py (the case grid, single-thread workers), strand2b_native_R4_v1.py (control sec. 11.3: the frozen"
echo "  code's scalar source e_x^nat(s), wick, at U = 2 g_M of the row, through the identical machinery; and the ruler variances"
echo "  from strand 0's full_v3/mats), strand2b_report_R4_v3.py (tables), strand2b_summary_v2.py (run summary); strand2a_lib.py"
echo "  and strand0_lib.py (C13's Slater vectors); frozen native modules in frozen/."
echo "- Earlier versions in the same folder are kept as run (library v1-v5, driver v1-v7: fixed rank, stencil, speed, Newton"
echo "  noise floor and key fixes, each stated in the file's docstring); report v1 and v2 had key defects (v2's docstring) and"
echo "  are not used for the tables below."
echo "- Box compute-host, conda env qi2; 12 single-thread workers for the law cases, xargs -P 12 for the native control."
echo
echo "## Commands"
echo "solve/code/strand2/launch_strand2b_R4_pass1_v1.sh in ${QI2_ROOT}/s2b_row (output pass1/), then"
echo "strand2b_report_R4_v3.py --run pass1 --baseline strand1_RESULTS_native_kscan_tables.md (a copy of"
echo "review_output/native_action_20261012/RESULTS_native_kscan_tables.md) and strand2b_summary_v2.py --run pass1. Records:"
echo "solve/results/strand2b/pass1/pass1_records.tar.gz (every case JSON, native control JSON, ruler variances)."
echo
echo "Run record (status.txt):"
sed 's/^/    /' $OUT/pass1/status.txt
echo "Law batch: $(tail -1 $OUT/pass1/batch.log)"
echo
echo "## Run summary (emitted by strand2b_summary_v2.py)"
cat $OUT/pass1/SUMMARY_pass1_v2.md
echo
echo "## Readings and conventions taken by the builder"
echo "- The state's projector rank d_0 is fixed per solve at the flat start's degeneracy (lead's ruling, sec. 11.1); the actual"
echo "  multiplet splitting and the gap to the next level are reported at every solution (T1)."
echo "- d rho^P_x/d psi_y at fixed state: a fourth-order central stencil at h = 1e-3 in the fixed c-basis; the spec's"
echo "  second-order differences at h = 1e-4 and 1e-5 are evaluated for the agreement check (run summary)."
echo "- Newton: tolerance 1e-9 along the continuation path, 1e-11 at lambda = 1; the best iterate is kept and the iteration stops"
echo "  at the residual's noise floor, which is measured at every solution (residual of F at z + 1e-12 noise; it grows with k);"
echo "  a final residual above 1e-11 is flagged in the record and the solution is still reported (run summary)."
echo "- The (2,1) additivity row compares c_1[(2,1)] with 2 c_1[(1,0)] + c_1[(0,1)] (the hole block (0,1) as the hole"
echo "  contribution; builder addition)."
echo "- Ring4's (2,0) rows are the free two-particle control (lead's sec. 9); they are reported with the others."
echo "- Native control: the frozen code's sector q is the block (N = 4 + q), its lowest multiplet the state, U = 2 g_M of the"
echo "  row (0, 0.1, 0.4), the same K and k; C13 does not apply (the record's C13 field says skipped)."
echo
echo "## Cost of the enclosure and uniqueness stages (pass 2), from the builder's timing case R4 (2,1), m 0.1, (0.2, 1), Lambda 1, k 1"
echo "Without them (timing7a): $(grep -h 'done in' $OUT/timing7b/timing7a_case.log)"
echo "With both (timing7b): $(grep -h 'done in' $OUT/timing7b/case.log)"
echo "The timing case's enclosure and uniqueness tables (strand2b_report_R4_v3.py --run timing7b):"
echo
awk '/^## T2 Enclosure/{p=1;print;next} p&&/^## /{p=0} p' $OUT/timing7b/RESULT_timing7b_tables.md
echo
sed '1,2d' $OUT/pass1/RESULT_strand2b_R4_pass1_tables_v3.md
} > $R
echo "wrote $R ($(wc -l < $R) lines)"
