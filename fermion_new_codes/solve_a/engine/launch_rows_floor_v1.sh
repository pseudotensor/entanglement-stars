#!/bin/bash
# engine_rows_floor_v1 (engine spec 26.1): R28b's floor candidates on the regime member at ell 4 and 8, and the residual with d_t k set to
# zero at T/rs 0, 5, 10 on the production's v17 segment (the H100 replay's handoff). Box CPU, one at a time. Builder analyst.
cd ${QI2_ROOT}/engine
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
PY=${QI2_ROOT}/python
nice -n 10 $PY code/engine_rows_floor_v1.py --mode static --family STAT/star_s416_ell4_avg.json --NB 416 --md CCT/rows_floor_s416_ell4_v1.md > CCT/log_rows_floor_s416_ell4.log 2>&1; echo "CCT rows floor s416 ell4 exit $? $(date -u +%H:%M:%S) UTC" >> CCT/status.txt
nice -n 10 $PY code/engine_rows_floor_v1.py --mode static --family STAT/star_s416_ell8_avg.json --NB 416 --md CCT/rows_floor_s416_ell8_v1.md > CCT/log_rows_floor_s416_ell8.log 2>&1; echo "CCT rows floor s416 ell8 exit $? $(date -u +%H:%M:%S) UTC" >> CCT/status.txt
nice -n 10 $PY code/engine_rows_floor_v1.py --mode records --run H100b/C3prod/c17_rs64_A_R2_tau8_ell4_avg --handoff H100b/REPLAY2/replay2_c17_rs64_h100_handoff.npz --times 0,5,10 --md CCT/rows_floor_prod_0_5_10_v1.md > CCT/log_rows_floor_prod.log 2>&1; echo "CCT rows floor production 0 5 10 exit $? $(date -u +%H:%M:%S) UTC" >> CCT/status.txt
