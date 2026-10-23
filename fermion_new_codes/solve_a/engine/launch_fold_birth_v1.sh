#!/bin/bash
# engine_fold_birth_v1 (engine spec 36.4 (2)) on the ladder members sigma/a 4 and 8 and the production; box CPU, nice 10, 4 threads;
# outputs in engine/TAN27/ on the box (copied to the lane's results/engine/LAD27/). The copy of the launcher that ran at 17:57:09 UTC
# (TAN27/run_fold_birth_v1.sh on the box, the same lines). Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
echo "fold_birth_v1 start $(date -u +%H:%M:%S) UTC" >> TAN27/status.txt
nice -n 10 ${QI2_ROOT}/python code/engine_fold_birth_v1.py \
  --run "sigma/a 4 member=H100b/C3lad27/collapse_c27_rs64_A_R2_tau8_ell4_prep4_avg.json=COLL/COLLAPSE_records_v1_c27_rs64_A_R2_tau8_ell4_prep4_avg_data.npz" \
  --run "sigma/a 8 member=H100b/C3lad27/collapse_c27_rs64_A_R2_tau8_ell8_prep4_avg.json=COLL/COLLAPSE_records_v1_c27_rs64_A_R2_tau8_ell8_prep4_avg_data.npz" \
  --run "the production (c19_rs64)=H100b/C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json=COLL11/COLLAPSE_records_v1.1_c19_rs64_data.npz" \
  --t-max 20.0 --md TAN27/fold_birth_v1.md --out-json TAN27/fold_birth_v1.json > TAN27/log_fold_birth_v1.log 2>&1
echo "fold_birth_v1 exit $? $(date -u +%H:%M:%S) UTC" >> TAN27/status.txt
