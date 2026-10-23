#!/bin/bash
# J4 version 2, RUN_ORDER_v3b sections 0, 2 and 3 on the pod CPU (builder). Stage C had finished before v3b arrived, so the
# gate-B fit of section 1 and gate C of section 3 read the same records (8 complete radii, 12 to 48): one fit, named gateC.
# Stage D (section 4) was already running with W = 2, the decision's default branch; a 1/2 or sqrt2/2 detection stops it.
cd ${QI2_ROOT}/j4
J4=${QI2_ROOT}/j4; ST=$J4/v3b_steps_status.txt
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python -m pip install -q matplotlib > $J4/pip_matplotlib.log 2>&1; echo "pip matplotlib exit $?" >> $ST
python j4_tests_v3b.py > $J4/tests_v3b.log 2>&1; echo "tests_v3b exit $? $(date -u +%H:%M:%S)" >> $ST
python j4_fit_v3b.py --selftest > $J4/selftest_v3b.log 2>&1; echo "selftest_v3b exit $? $(date -u +%H:%M:%S)" >> $ST
python j4_fit_v3b.py --dirs $J4/prod $J4/prod_v2 --out $J4/J4_fit_v3b_gateC.json --plot $J4/J4_fit_v3b_gateC \
  > $J4/J4_fit_v3b_gateC.log 2>&1; echo "fit_v3b_gateC exit $? $(date -u +%H:%M:%S)" >> $ST
python j4_prefix_tests_v3b.py --v1-dir $J4/prod --v2-dir $J4/prod_v2 --cache-dir $J4/prod_v2/kcache --threads 4 \
  > $J4/prefix_tests_v3b.log 2>&1; echo "prefix_tests_v3b exit $? $(date -u +%H:%M:%S)" >> $ST
echo "all done $(date -u +%H:%M:%S)" >> $ST
