#!/bin/bash
# launch_dust_root_history_box_v1b.sh (the lane's analyst builder; engine spec 42.2 (5), 42.3 (4)): engine_dust_root_history_v1b.py on box GPU 1
# (B5 version 2's dust columns at T/r_s 12.125 and 15.0, kappa 1 and 2, on the production to 40 and B's continuation after it, to T/r_s 56); nice 10, 4 threads.
cd ${QI2_ROOT}/engine || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PAGE_V6_PATH=${QI2_ROOT}/v6
mkdir -p DUST
echo "dust_v1b start $(date -u '+%Y-%m-%d %H:%M:%S UTC') pid $$" >> DUST/status.txt
nice -n 10 ${QI2_ROOT}/python code/engine_dust_root_history_v1b.py --columns-npz S4one/v2/B5v2_cols_prod.npz \
  --record H100b/C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json --fields H100b/C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11_fields.npz --fields-module engine_fields_v14 \
  --record2 C3ext19/collapse_c19_rs64_A_R2_tau8_ell4_avg_ext100.json --fields2 C3ext19/collapse_c19_rs64_A_R2_tau8_ell4_avg_ext100_fields.npz --switch-T-rs 40 \
  --kappas 1,2 --T-end-rs 56 --jump-every-rs 4 --gpu 1 --out-json DUST/dust_root_history_v1b.json --out-npz DUST/dust_root_history_v1b.npz > DUST/dust_root_history_v1b.log 2>&1
echo "dust_v1b exit $? $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> DUST/status.txt
