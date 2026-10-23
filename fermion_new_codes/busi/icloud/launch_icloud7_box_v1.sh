#!/bin/bash
# launch_icloud7_box_v1.sh (busi lane): icloud item 7's recomputation on the box CPU, RUN_dust_readouts_v1.md step 3 (kappa 1, the 2 required
# channel-records), resumable by rerunning the same command. Records the load at launch.
O=${QI2_ROOT}/icloud; C=${QI2_ROOT}/icloud; P=${QI2_ROOT}/DUST
echo "icloud7 launch $(date -u +%Y-%m-%dT%H:%M:%SZ) load $(cut -d' ' -f1 /proc/loadavg) free_GB $(free -g | awk '/Mem:/{print $7}')" >> $O/icloud7_status.txt
cd $O || exit 1
nohup nice -n 10 ${QI2_ROOT}/python -I -B $C/dust_readouts_v1.py \
  --engine-dir ${QI2_ROOT}/code --page-v6-path ${QI2_ROOT}/v6 \
  --columns-npz ${QI2_ROOT}/B5v2_cols_prod.npz \
  --record ${QI2_ROOT}/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json \
  --fields ${QI2_ROOT}/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11_fields.npz \
  --record2 ${QI2_ROOT}/collapse_c19_rs64_A_R2_tau8_ell4_avg_ext100.json \
  --fields2 ${QI2_ROOT}/collapse_c19_rs64_A_R2_tau8_ell4_avg_ext100_fields.npz \
  --switch-T-rs 40 --fields-module engine_fields_v14 --nsites 8192 --m 0.2 \
  --tags T12p125,T15p000 --kappas 1 --T-end-rs 56 --jump-every-rs 4 --checkpoint-every-rs 4 \
  --propagator chebyshev --cheb-tol 1e-15 --eig-driver evr \
  --producer-json $P/dust_root_history_v1.json --producer-json $P/dust_root_history_v1b.json --producer-npz $P/dust_root_history_v1b.npz \
  --out-dir $O --tag dust_readouts_v1 --threads 16 --tol 1e-6 > $O/dust_readouts_v1.log 2>&1 < /dev/null &
echo "icloud7 pid $! $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> $O/icloud7_status.txt
