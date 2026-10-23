#!/bin/bash
# engine_rows_variants_v1 (engine spec 25.3): the regime member at ell 4 and 8, the production at T/rs 5, 10 (the v17 segment, the H100
# replay's handoff) and 20.625 (native). Box CPU. Builder analyst.
cd ${QI2_ROOT}/engine
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
PY=${QI2_ROOT}/python
nice -n 10 $PY code/engine_rows_variants_v1.py --mode static --family STAT/star_s416_ell4_avg.json --NB 416 --md CCT/rows_variants_s416_ell4_v1.md > CCT/log_rows_variants_s416_ell4.log 2>&1; echo "CCT rows variants s416 ell4 exit $? $(date -u +%H:%M:%S) UTC" >> CCT/status.txt
nice -n 10 $PY code/engine_rows_variants_v1.py --mode static --family STAT/star_s416_ell8_avg.json --NB 416 --md CCT/rows_variants_s416_ell8_v1.md > CCT/log_rows_variants_s416_ell8.log 2>&1; echo "CCT rows variants s416 ell8 exit $? $(date -u +%H:%M:%S) UTC" >> CCT/status.txt
nice -n 10 $PY code/engine_rows_variants_v1.py --mode records --run H100b/C3prod/c17_rs64_A_R2_tau8_ell4_avg --handoff H100b/REPLAY2/replay2_c17_rs64_h100_handoff.npz --times 5,10 --md CCT/rows_variants_prod_5_10_v1.md > CCT/log_rows_variants_prod_5_10.log 2>&1; echo "CCT rows variants production 5 10 exit $? $(date -u +%H:%M:%S) UTC" >> CCT/status.txt
nice -n 10 $PY code/engine_rows_variants_v1.py --mode records --run H100b/C3prod19/c19_rs64_A_R2_tau8_ell4_avg_r11 --handoff H100b/C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11_handoff.npz --times 20.625 --md CCT/rows_variants_prod_20.6_v1.md > CCT/log_rows_variants_prod_20.6.log 2>&1; echo "CCT rows variants production 20.625 exit $? $(date -u +%H:%M:%S) UTC" >> CCT/status.txt
