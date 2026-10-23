#!/bin/bash
# Strand 5 J2b (spec sec. 10(c)) and the J4 kernels' packets (sec. 10(d)): strand5_kernel_packets_v1.py, 8 single-thread workers, sequential:
# set A (2c's equal-rectangle masses, L 256), set C (further masses, L 256), set D (massless, twist 0.371, L 128 and 256), set B (physical
# heavy masses, twist 0.371, L 256 and 512); the J4 points' packets (twist 0): (0.03, 0.3) and (0.01, 0.1) at L 256, and the two light
# masses at L 512.  Output s5_star/J2b_run1 and s5_star/J4kernels_run1.
set -u
D=${QI2_ROOT}/s5_star
cd $D || exit 1
mkdir -p J2b_run1 J4kernels_run1
date -u '+start %Y-%m-%d %H:%M:%S UTC' > J2b_run1/status.txt
PY="nice -n 10 ${QI2_ROOT}/python code/strand5_kernel_packets_v1.py"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
$PY --out J2b_run1 --set A --L 256 >> J2b_run1/run.log 2>&1; echo "A256 exit $? $(date -u +%H:%M:%S)" >> J2b_run1/status.txt
$PY --out J2b_run1 --set C --L 256 >> J2b_run1/run.log 2>&1; echo "C256 exit $? $(date -u +%H:%M:%S)" >> J2b_run1/status.txt
$PY --out J2b_run1 --set D --L 128 >> J2b_run1/run.log 2>&1; echo "D128 exit $? $(date -u +%H:%M:%S)" >> J2b_run1/status.txt
$PY --out J2b_run1 --set D --L 256 >> J2b_run1/run.log 2>&1; echo "D256 exit $? $(date -u +%H:%M:%S)" >> J2b_run1/status.txt
$PY --out J2b_run1 --set B --L 256 >> J2b_run1/run.log 2>&1; echo "B256 exit $? $(date -u +%H:%M:%S)" >> J2b_run1/status.txt
$PY --out J4kernels_run1 --set C --L 256 --masses 0.03,0.30149626863362670,0.42532340636273950,0.01,0.10049875621120890,0.14177446878757824 >> J4kernels_run1/run.log 2>&1; echo "J4 L256 exit $? $(date -u +%H:%M:%S)" >> J2b_run1/status.txt
$PY --out J2b_run1 --set B --L 512 >> J2b_run1/run.log 2>&1; echo "B512 exit $? $(date -u +%H:%M:%S)" >> J2b_run1/status.txt
$PY --out J4kernels_run1 --set C --L 512 --masses 0.03,0.01 >> J4kernels_run1/run.log 2>&1; echo "J4 light L512 exit $? $(date -u +%H:%M:%S)" >> J2b_run1/status.txt
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> J2b_run1/status.txt
echo DONE >> J2b_run1/status.txt
