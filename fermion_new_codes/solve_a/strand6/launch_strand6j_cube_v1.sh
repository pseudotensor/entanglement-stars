#!/bin/bash
# Strand 6 (j) J4, the cube's (2,0) variances (strand6j_cube_var_v1.py; one thread, minutes), started when the lane runs at most 15 busy
# processes.  Builder analyst.
D=${QI2_ROOT}/j
cd $D || exit 1
date -u '+queued %Y-%m-%d %H:%M:%S UTC' > run2/status.txt
until [ "$(bash ${QI2_ROOT}/lane_threads_v1.sh)" -le 15 ]; do sleep 30; done
date -u '+start %H:%M:%S UTC' >> run2/status.txt
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
nice -n 10 ${QI2_ROOT}/python code/strand6/strand6j_cube_var_v1.py --encl cases_c8/case_C8_b20_m0.1_gM0.2_gB1_L1_k1.json,cases_c8/case_C8_b20_m0.1_gM0.2_gB1_L1_k10000.json --classes cases_c8/item5.json --md run2/RESULT_j_cube_tables_v1.md --out-json run2/j_cube.json > run2/run.log 2>&1
echo "exit $? $(date -u +%H:%M:%S)" >> run2/status.txt
echo DONE >> run2/status.txt
