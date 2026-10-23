#!/bin/bash
# Strand 3: fetch run1's outputs from the box and assemble solve/results/RESULT_strand3_entropy.md (text assembly only;
# the tables come verbatim from strand3_report.py).
set -eu
W=${QI2_ROOT}/.wt-solve_a
B=${QI2_ROOT}/run1
OUT=$W/solve/results/strand3
mkdir -p $OUT/planar $OUT/sphere $OUT/balls
for f in status.txt masses.json masses.log planar_cpu.log planar_gpu0.log planar_gpu2.log sphere_batch.log balls_batch.log partB.json partB.log \
         partC.json partC.log verify_basic.log scan.json scan.log report.log pytest.log RESULT_strand3_tables.md; do
  timeout 120 scp -q compute-host:$B/$f $OUT/ || echo "missing $f"
done
timeout 300 scp -q "compute-host:$B/planar/*.json" $OUT/planar/ || echo "missing planar"
timeout 120 scp -q "compute-host:$B/sphere/*.json" $OUT/sphere/ || echo "missing sphere"
timeout 120 scp -q "compute-host:$B/balls/*.json" $OUT/balls/ || echo "missing balls"
R=$W/solve/results/RESULT_strand3_entropy.md
{
echo "# Strand 3 result (test T10): the spatial algebra and the vacuum's cut-entropy area coefficient against A/(4G)"
echo
echo "Builder: analyst. Spec: solve/specs/SPEC_strand3_entropy.md (lead, 2026-10-14; with its later R^static column and the"
echo "B.3 correction). Numbers, controls and commands only; no physics reading. Every table is emitted by"
echo "solve/code/strand3/strand3_report.py from the files in solve/results/strand3/."
echo
echo "## Inputs"
echo "- The paper's entropy modules, unmodified (box copies in code/orient/; md5 and commits in solve/code/strand3/ORIENT_SOURCES.txt):"
sed 's/^/    /' $W/solve/code/strand3/ORIENT_SOURCES.txt
echo "- New code (solve/code/strand3/): strand3_masses.py, strand3_planar.py, strand3_planar_gpu.py, sphere_mass.py (sphere.py with mass"
echo "  passed and ./orient on the import path; diff in the file's docstring), strand3_batches.py, strand3_partB_v2.py (v1 ran only in"
echo "  the smoke test), lattice_entropy_wilson.py, strand3_partC.py, strand3_scan.py, strand3_report.py, tests/strand3_controls_test.py."
echo "- Box compute-host (host gpu), conda env qi2; GPUs 0 and 2 for the depth-64 planar runs (GPU 1 was occupied by another"
echo "  user's process at launch); 14 single-thread CPU workers for the other planar runs."
echo
echo "## Command"
echo "solve/code/strand3/launch_strand3_full_v1.sh, run in ${QI2_ROOT}/s3_entropy (output run1/)."
echo
echo "Run record (status.txt):"
sed 's/^/    /' $OUT/status.txt
echo
echo "pytest: $(grep -E 'passed|failed' $OUT/pytest.log | tail -1)"
echo
echo "## Readings taken by the builder"
echo "- A.1 runs at the exact q_s of every scan row (the set Q is the exact values; 4-digit rounding only labels them), so A.5"
echo "  uses eta_100 at the exact q_s as its text requires."
echo "- Part C: one evaluation at chain length L = 64 (the finest length of the depth-16 ladder, 4L = 256) with nfft = 16 L = 1024,"
echo "  the setting of the spec's control eta.value((1,0,0), L, 12, 16 L, q)."
echo "- C5: the entropy of an exact projector computed from eigenvalues carries a round-off floor of order n eps |log eps|; the"
echo "  restricted-spectrum distance from {0, 1} is reported beside S_R (the lead's B.3 correction)."
echo
sed '1,2d' $OUT/RESULT_strand3_tables.md
} > $R
echo "wrote $R ($(wc -l < $R) lines)"
