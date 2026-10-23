#!/bin/bash
# J4 RUN_ORDER_v3 steps 0 to 2 on the pod (builder): the version-3 tests, the gate-B fit on the records that exist,
# the exact test of the prefix assembly. The running stage-C driver is not touched.
cd ${QI2_ROOT}/j4
J4=${QI2_ROOT}/j4
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
st() { echo "$1 $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> $J4/v3_steps_status.txt; }
python j4_tests_v3.py > $J4/tests_v3.log 2>&1; st "tests_v3 exit $?"
python j4_fit_v3.py --selftest > $J4/selftest_v3.log 2>&1; st "selftest_v3 exit $?"
python j4_fit_v3.py --dirs $J4/prod $J4/prod_v2 --out $J4/J4_fit_v3_gateB.json --plot $J4/J4_fit_v3_gateB > $J4/J4_fit_v3_gateB.log 2>&1; st "gateB fit exit $?"
python j4_prefix_tests_v3.py --v1-dir $J4/prod --v2-dir $J4/prod_v2 --cache-dir $J4/prod_v2/kcache --threads 4 > $J4/prefix_tests_v3.log 2>&1; st "prefix tests exit $?"
st "all done"
