#!/bin/bash
# Strand 6 (j) revision 4 (JUDGMENT_strand6_validation.md sec. 13, the hand-back's item 3): strand6j_constraint_var_v3.py (ring4, the reality
# lines over the 45 records) and strand6j_cube_var_v4.py (the cube, the caption amended), seconds each, one thread, started when the lane runs
# at most 15 busy processes.  Builder analyst.
D=${QI2_ROOT}/j
cd $D || exit 1
mkdir -p run4
date -u '+queued %Y-%m-%d %H:%M:%S UTC' > run4/status.txt
until [ "$(bash ${QI2_ROOT}/lane_threads_v1.sh)" -le 15 ]; do sleep 20; done
date -u '+start %H:%M:%S UTC' >> run4/status.txt
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
nice -n 10 ${QI2_ROOT}/python code/strand6/strand6j_constraint_var_v3.py --cases "cases/case_R4_*.json" --out-json run4/j_ring4_v3.json --md run4/RESULT_j_ring4_tables_v3.md > run4/run_ring4.log 2>&1
echo "ring4 v3 exit $? $(date -u +%H:%M:%S)" >> run4/status.txt
nice -n 10 ${QI2_ROOT}/python code/strand6/strand6j_cube_var_v4.py --encl cases_c8/case_C8_b20_m0.1_gM0.2_gB1_L1_k1.json,cases_c8/case_C8_b20_m0.1_gM0.2_gB1_L1_k10000.json --classes cases_c8/item5.json --md run4/RESULT_j_cube_tables_v4.md --out-json run4/j_cube_v4.json > run4/run_cube.log 2>&1
echo "cube v4 exit $? $(date -u +%H:%M:%S)" >> run4/status.txt
echo DONE >> run4/status.txt
