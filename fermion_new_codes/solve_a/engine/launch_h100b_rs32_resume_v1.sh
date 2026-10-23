#!/bin/bash
# The r_s 32 tau 8 comparator of engine spec 21.3 on the new H100, resumed on engine_collapse_v19 from T/rs 14.5 (the stored v17 member's
# last record before the halving rule's first subdivision at 14.75; launch_h100b_v19_v2.sh asked for 14.625, which is no record time at
# r_s 32 and tau 8, so its job stopped at once). A light job (1024 sites) beside the production and the c_LZ ladder. Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
S=h100b_status.txt
echo "lane3 c19_rs32_tau8_r145 start $(date -u +%H:%M:%S) UTC" >> $S
python code/engine_collapse_v19.py --m 0.2 --g 0.25 --rs 32 --R0-over-rs 4 --nsites 1024 --ell 4 --kappa-sea 10 --reference R2 --site-fields averaged --reading A --tau 8 \
  --T-over-rs 40 --out C3set19 --gpu 0 --tag c19_rs32_A_R2_tau8_ell4_avg_r145 --resume-from C3set/set17_rs32_A_R2_tau8_ell4_avg --resume-T-over-rs 14.5 > logs_h100b_c19_rs32_tau8_r145.log 2>&1
echo "lane3 c19_rs32_tau8_r145 exit $? $(date -u +%H:%M:%S) UTC" >> $S
