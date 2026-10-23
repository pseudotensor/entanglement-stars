#!/bin/bash
# GAP10 as delivered (the lead's code, unrepaired), on the pod's otherwise idle GPU (builder; inbox #22): the T0 tests and
# the T6 flat-space control, the parts that the icloud review's repairs (R1 to R6) do not touch. The repaired build
# replaces this code when it lands; these outputs are its environment check.
cd ${QI2_ROOT}/orig
OUT=${QI2_ROOT}/gap10; mkdir -p $OUT
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
python csr_tests.py > $OUT/T0_tests_orig.log 2>&1; echo "T0 exit $? $(date -u +%H:%M:%S)" >> $OUT/status_orig.txt
python csr_matrix.py > $OUT/matrix_orig.sh 2>&1
grep -v '^#' $OUT/matrix_orig.sh | grep csr_run | grep "T6_flat" > $OUT/t6_cmds.sh
cd ${QI2_ROOT}/gap10
while read c; do (cd ${QI2_ROOT}/orig && $c) >> $OUT/T6_orig.log 2>&1; echo "T6 line exit $? $(date -u +%H:%M:%S)" >> $OUT/status_orig.txt; done < $OUT/t6_cmds.sh
echo "all done $(date -u +%H:%M:%S)" >> $OUT/status_orig.txt
