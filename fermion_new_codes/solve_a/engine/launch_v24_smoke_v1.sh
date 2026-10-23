#!/bin/bash
# engine_collapse_v24's smoke test on box GPU 2 after vacflow v9 (its launcher's list ends with the r_s 32 replay): a fresh run at r_s 16
# (512 sites, tau 4, T/rs 1.5) and a resume from V19T/t17_rs16 at T/rs 1.0, against the v19 smoke runs (t19b_rs16, t19rb_rs16). Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
while ps -eo args | grep -q "[l]aunch_engine_gpu_v4.sh 2 code/list_gpu2_v11.txt"; do sleep 30; done
export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
A="--m 0.2 --g 0.25 --rs 16 --R0-over-rs 4 --reading A --nsites 512 --ell 4 --tau 4 --T-over-rs 1.5 --kappa-sea 10 --reference R2 --site-fields averaged"
echo "v24 smoke start $(date -u +%H:%M:%S)" >> V19T/smoke_status_v24.txt
nice -n 10 ${QI2_ROOT}/python code/engine_collapse_v24.py $A --out V19T --gpu 0 --tag t24_rs16 > V19T/log_t24_rs16.log 2>&1; echo "v24 fresh exit $? $(date -u +%H:%M:%S)" >> V19T/smoke_status_v24.txt
nice -n 10 ${QI2_ROOT}/python code/engine_collapse_v24.py $A --out V19T --gpu 0 --tag t24r_rs16 --resume-from V19T/t17_rs16 --resume-T-over-rs 1.0 > V19T/log_t24r_rs16.log 2>&1; echo "v24 resume exit $? $(date -u +%H:%M:%S)" >> V19T/smoke_status_v24.txt
echo "DONE $(date -u +%H:%M:%S)" >> V19T/smoke_status_v24.txt
