#!/bin/bash
# J4 version 2, RUN_ORDER_v3d sections 1 to 3 on the pod CPU (builder), beside stages D and E (untouched).
cd ${QI2_ROOT}/j4
J4=${QI2_ROOT}/j4; ST=$J4/v3d_steps_status.txt
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
D="--dirs $J4/prod $J4/prod_v2 --wide-upto 40"
python j4_tests_v3d.py > $J4/tests_v3d.log 2>&1; echo "tests_v3d exit $? $(date -u +%H:%M:%S)" >> $ST
python j4_fit_v3d.py --selftest --out $J4/selftest_v3d.json > $J4/selftest_v3d.log 2>&1; echo "selftest_v3d exit $? $(date -u +%H:%M:%S)" >> $ST
python j4_fit_v3d.py $D --out $J4/J4_fit_v3d_gateC.json --plot $J4/J4_fit_v3d_gateC > $J4/J4_fit_v3d_gateC.log 2>&1; echo "fit_v3d_gateC exit $? $(date -u +%H:%M:%S)" >> $ST
python j4_fit_v3d.py $D --joint-mmax 64 --out $J4/J4_fit_v3d_gateC_m64.json > $J4/J4_fit_v3d_gateC_m64.log 2>&1; echo "fit_v3d_gateC_m64 exit $? $(date -u +%H:%M:%S)" >> $ST
echo "all done $(date -u +%H:%M:%S)" >> $ST
