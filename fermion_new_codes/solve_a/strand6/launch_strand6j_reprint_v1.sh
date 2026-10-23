#!/bin/bash
# Strand 6 (j) reprint (JUDGMENT_strand6_validation.md sec. 12): strand6j_constraint_var_v2.py (ring4) and strand6j_cube_var_v2.py (the cube),
# seconds each, one thread, started when the lane runs at most 15 busy processes.  Builder analyst.
D=${QI2_ROOT}/j
cd $D || exit 1
mkdir -p run3
date -u '+queued %Y-%m-%d %H:%M:%S UTC' > run3/status.txt
until [ "$(bash ${QI2_ROOT}/lane_threads_v1.sh)" -le 15 ]; do sleep 20; done
date -u '+start %H:%M:%S UTC' >> run3/status.txt
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
nice -n 10 ${QI2_ROOT}/python code/strand6/strand6j_constraint_var_v2.py --cases "cases/case_R4_*.json" --out-json run3/j_ring4_v2.json --md run3/RESULT_j_ring4_tables_v2.md > run3/run_ring4.log 2>&1
echo "ring4 exit $? $(date -u +%H:%M:%S)" >> run3/status.txt
nice -n 10 ${QI2_ROOT}/python code/strand6/strand6j_cube_var_v2.py --encl cases_c8/case_C8_b20_m0.1_gM0.2_gB1_L1_k1.json,cases_c8/case_C8_b20_m0.1_gM0.2_gB1_L1_k10000.json --classes cases_c8/item5.json --md run3/RESULT_j_cube_tables_v2.md --out-json run3/j_cube_v2.json > run3/run_cube.log 2>&1
echo "cube exit $? $(date -u +%H:%M:%S)" >> run3/status.txt
echo DONE >> run3/status.txt
