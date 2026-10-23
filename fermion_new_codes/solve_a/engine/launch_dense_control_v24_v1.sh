#!/bin/bash
# The point-sampled r_s 64 control (engine spec 24.6, box GPU 0) on engine_collapse_v24 (fields v16), resumed from the stored v17 dense
# run (C3/c17_rs64_A_R2_tau8_ell4_dense, stopped at 23:06 UTC for the v19 check) at its last stored record T/rs 1.25, to T/rs 40.
# Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
echo "GPU0 c24_rs64_dense_r125 start $(date -u +%H:%M:%S) UTC (engine_collapse_v24, gpu 0)" >> status.txt
nice -n 10 ${QI2_ROOT}/python code/engine_collapse_v24.py --m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 2048 --ell 4 --tau 8 --T-over-rs 40 \
  --kappa-sea 10 --reference R2 --site-fields dense --out C3 --gpu 0 --tag c24_rs64_A_R2_tau8_ell4_dense_r125 --resume-from C3/c17_rs64_A_R2_tau8_ell4_dense --resume-T-over-rs 1.25 > logs_gpu0_c24_rs64_dense_r125.log 2>&1
echo "GPU0 c24_rs64_dense_r125 exit $? $(date -u +%H:%M:%S) UTC" >> status.txt
