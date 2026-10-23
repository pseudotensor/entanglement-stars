#!/bin/bash
# Strand 3 addendum result (assembly v2: the title states which parts are present): generate the tables on the box (strand3_addendum_report_v1.py over addendum1/), fetch them with the run
# records and JSON, and assemble solve/results/RESULT_strand3_addendum.md (text assembly only).  Parts present at the time of the run are
# tabulated; a later run of this script (or a new version) adds the rest.
set -eu
W=${QI2_ROOT}/.wt-solve_a
B=${QI2_ROOT}/s3_entropy
OUT=$W/solve/results/strand3_addendum
PY=${QI2_ROOT}/python
mkdir -p $OUT
timeout 300 ssh -n -o BatchMode=yes compute-host "cd $B && OMP_NUM_THREADS=1 $PY code/strand3_addendum_report_v1.py --dir addendum1 --ball64 run1/balls/ball_q0_c0_0_0.json --md addendum1/RESULT_tables.md > addendum1/report.log 2>&1; cd addendum1 && tar czf ../addendum1_json.tar.gz \$(ls addA.json addB.json addD.json C/addC.json C/masses_addC.json C/planar/*.json C/sphere/*.json E/*.json 2>/dev/null)"
for f in RESULT_tables.md status.txt report.log addA.log addB.log addD.log; do timeout 120 scp -q compute-host:$B/addendum1/$f $OUT/ 2>/dev/null || echo "missing $f"; done
timeout 300 scp -q compute-host:$B/addendum1_json.tar.gz $OUT/ || echo "missing json archive"
R=$W/solve/results/RESULT_strand3_addendum.md
{
if [ -f $OUT/addB.log ]; then echo "# Strand 3 addendum result: the red team's settling computations (parts A to E)"; else
echo "# Strand 3 addendum result, revision 1: the red team's settling computations, parts A, C, D and E (part B queued: its 12 workers start when the lane's other jobs free the threads)"; fi
echo
echo "Builder: analyst. Spec: solve/specs/SPEC_strand3_addendum.md (lead, 2026-10-14). Numbers, controls and commands only; no reading."
echo
echo "## Inputs"
echo "- Code (solve/code/strand3/): strand3_addAB_v1.py (parts A and B: the nested transverse quadrature, the Wilson-type family through"
echo "  lattice_entropy_wilson.py and the 8-valley family through a copy of the paper's interval covariance with the mass replaced),"
echo "  strand3_addCD_v1.py (part C's job file and assembly, part D), strand3_planar.py and strand3_batches.py (part C's planar and sphere"
echo "  runs, unchanged from strand 3), the paper's orient/balls.py (part E, unmodified), strand3_addendum_report_v1.py (tables)."
echo "- The paper's modules are the strand 3 copies (md5 in RESULT_strand3_entropy.md)."
echo "- Box compute-host, conda env qi2, CPU; part A with 12 single-thread workers; part E with 4 BLAS threads."
echo
echo "## Commands"
echo "launch_strand3_addA_v1.sh, launch_strand3_addD_v1.sh, launch_strand3_addCE_v1.sh, launch_strand3_addB_v3.sh (part B; versions 1 and 2 were waiters replaced for the lane's thread cap) in"
echo "${QI2_ROOT}/s3_entropy (output addendum1/); records in solve/results/strand3_addendum/."
echo
echo "Run record (status.txt):"
sed 's/^/    /' $OUT/status.txt
echo
echo "## Readings and conventions taken by the builder"
echo "- Quadrature: K_1 = (pi/72) min(12, max(1, round(12 kappa 72/pi))) (so the inner square is a whole number of cells of the 24-grid);"
echo "  the inner square carries 48 x 48 midpoint points; the middle region is the 24 x 24 grid of [-pi/6, pi/6]^2 without the inner cells;"
echo "  the outer region is the standard 12 x 12 grid outside the pi/6 square (140 channels)."
echo "- Part B uses nfft = 16 L at L = 64 and 128 (the spec names the lengths only)."
echo "- Part C takes the masses 4 q_s from strand 3's masses.json (not the spec's rounded values), the massless light packet's (32,24)"
echo "  value and mesh-2 sphere from strand 3's run1."
echo "- Part D: the slab R = {x_1 in {0, 1}} has width 2, so every site of R is a boundary site; the vertex is reported at the two R sites"
echo "  (0,0,0), (1,0,0) and at (2,0,0) outside R. The transported region is read as the image of R's modes under V = diag(1, U^dag)"
echo "  (W = V[:, R]); the image under V^dag (part B's frame A2) and R's own sites (frame A1) are tabulated beside it."
echo
sed '1,2d' $OUT/RESULT_tables.md
} > $R
echo "wrote $R ($(wc -l < $R) lines)"
