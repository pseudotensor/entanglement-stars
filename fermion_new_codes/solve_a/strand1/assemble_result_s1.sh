#!/bin/bash
# Strand 1: fetch the small outputs of the box rerun and assemble solve/results/RESULT_strand1_baseline.md.
# Text assembly only (scp, grep, sed); every number in the result file is copied from a box-generated file.
set -eu
W=${QI2_ROOT}/.wt-solve_a
B=${QI2_ROOT}/s1_baseline
OUT=$W/solve/results/strand1_box
mkdir -p $OUT
for f in status.txt pytest.log kscan.log run.log report.log kscan_report.log compare_s1.json compare_s1.txt launch_s1_baseline.sh; do
  timeout 60 scp -q compute-host:$B/$f $OUT/ || echo "missing $f"
done
# the regenerated tables are identical to the committed ones (compare_s1.txt), so they are fetched to the scratchpad only
SCR=${QI2_ROOT}/s1_tables
mkdir -p $SCR
for f in native_tables.md RESULTS_native_kscan_tables.md; do
  timeout 60 scp -q compute-host:$B/$f $SCR/ || echo "missing $f"
done
R=$W/solve/results/RESULT_strand1_baseline.md
T=$SCR/native_tables.md
K=$SCR/RESULTS_native_kscan_tables.md
{
echo "# Strand 1 result: the frozen native-action ring4 lock test rerun from the builder's own code"
echo
echo "Builder: analyst (this lane's builder). Spec: the goal's strand 1 (${QI2_ROOT}/GOAL_lane-alt.md)."
echo "No physics reading is attached here; the lead judges."
echo
echo "## Inputs"
echo "- Code: ${QI2_ROOT}/native_action_20261012 at branch solve_a (PR3 head 3aa696f8c9b),"
echo "  byte-identical on the box (md5 list in solve/code/strand1/box_md5.txt): native_core.py and native_config.py (last changed"
echo "  bba9ec75d56), native_block.py, native_sources.py, native_run.py, native_report.py (102af54cef3, the AGG10 sec. 70 run),"
echo "  native_kscan.py, native_kscan_report.py (d7a545fd74b, the AGG10 sec. 73 k-scan), tests/test_native_action.py, tests/test_native_core.py."
echo "- Stored results compared against: native_summary.json, native_tables.md, native_jobs.tar.gz (322 job files),"
echo "  native_kscan_summary.json, RESULTS_native_kscan_tables.md, out_kscan/native_kscan.json.gz, all from the same folder."
echo "- Box: compute-host (host gpu), conda env qi2 (python 3.12.12, numpy 2.2.6, scipy 1.17.0), one thread per worker, nice 10."
echo
echo "## Command"
echo "solve/code/strand1/launch_s1_baseline.sh, run in $B: pytest on the two test files; native_kscan.py --workers 16;"
echo "native_kscan_report.py; native_run.py --workers 14 --which pinned,main,cube,register; native_report.py. Then"
echo "solve/code/strand1/compare_s1_baseline.py (leaf-by-leaf comparison of every JSON number and line-by-line table diff)."
echo
echo "## Run record (status.txt)"
sed 's/^/    /' $OUT/status.txt
echo
echo "pytest: $(grep -E 'passed|failed' $OUT/pytest.log | tail -1)"
echo "k-scan: $(grep 'tasks' $OUT/kscan.log | head -1); $(tail -1 $OUT/kscan.log)"
echo "run: $(grep 'jobs,' $OUT/run.log | head -1); $(tail -1 $OUT/run.log); jobs reporting ERROR: $(grep -c ERROR $OUT/run.log || true)"
echo
echo "## Comparison with the stored results (compare_s1.txt, verbatim)"
sed 's/^/    /' $OUT/compare_s1.txt
echo
echo "## Line 9 rows on ring4 at k = 1 (regenerated native_tables.md, copied by grep)"
sed -n '/^### Line 9/,/^### Line 10/p' $T | grep -E '^\| lattice|^\|---|^\| ring4 \| (J_W6|J_CVC|SJ) \| wick'
echo
echo "## k-scan: lock-defect fits (regenerated RESULTS_native_kscan_tables.md, first fit table, copied by grep)"
sed -n '/^| promotion | m | q | form/,/^$/p' $K
} > $R
echo "wrote $R"
