#!/bin/bash
# J4 version 2, stage A on the box (builder): the version-2 tests, the fit self-test and the version-2 analysis of the
# version-1 data (no new cases). Also the red team's inputs.
cd ${QI2_ROOT}/j4
PY=${QI2_ROOT}/python; J4=${QI2_ROOT}/j4
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
nice -n 10 $PY j4_tests_v2.py > $J4/tests_v2.log 2>&1; echo "tests_v2 exit $?" >> $J4/stageA_v2_status.txt
nice -n 10 $PY j4_fit_v2.py --selftest > $J4/selftest_v2.log 2>&1; echo "selftest_v2 exit $?" >> $J4/stageA_v2_status.txt
nice -n 10 $PY j4_fit_v2.py --dirs $J4/pod_copy/prod --out $J4/J4_fit_v2_stage0.json > $J4/J4_fit_v2_stage0.log 2>&1; echo "fit_v2_stage0 exit $?" >> $J4/stageA_v2_status.txt
