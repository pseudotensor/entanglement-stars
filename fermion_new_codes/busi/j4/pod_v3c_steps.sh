#!/bin/bash
# J4 version 2, RUN_ORDER_v3c sections 1 to 3 on the pod CPU (builder), beside stages D and E (untouched).
cd ${QI2_ROOT}/j4
J4=${QI2_ROOT}/j4; ST=$J4/v3c_steps_status.txt
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python j4_tests_v3c.py > $J4/tests_v3c.log 2>&1; echo "tests_v3c exit $? $(date -u +%H:%M:%S)" >> $ST
python j4_fit_v3c.py --selftest --out $J4/selftest_v3c.json > $J4/selftest_v3c.log 2>&1; echo "selftest_v3c exit $? $(date -u +%H:%M:%S)" >> $ST
python j4_fit_v3c.py --dirs $J4/prod $J4/prod_v2 --wide-upto 40 --out $J4/J4_fit_v3c_gateC.json --plot $J4/J4_fit_v3c_gateC \
  > $J4/J4_fit_v3c_gateC.log 2>&1; echo "fit_v3c_gateC exit $? $(date -u +%H:%M:%S)" >> $ST
echo "all done $(date -u +%H:%M:%S)" >> $ST
