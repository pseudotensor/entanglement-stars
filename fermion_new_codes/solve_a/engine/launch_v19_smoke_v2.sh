#!/bin/bash
# Version 2 of the smoke test of engine_collapse_v19 on box GPU 0 (r_s 16, 512 sites, tau 4, T/rs 1.5), after the first stopped on the
# infinite Landau-Zener time at unchanged fields (engine_ops_v2 fixed: the longest window): waits for version 1's launcher (PID 2285979)
# to exit, then a fresh v19 run (one slice of each precision) and a v19 run resumed from version 1's v17 run (t17_rs16) at T/rs 1.0; one
# process at a time. Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
while kill -0 2285979 2>/dev/null; do sleep 10; done
PY=${QI2_ROOT}/python
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
A="--m 0.2 --g 0.25 --rs 16 --R0-over-rs 4 --reading A --nsites 512 --ell 4 --tau 4 --T-over-rs 1.5 --kappa-sea 10 --reference R2 --site-fields averaged"
echo "v19 fresh start $(date -u +%H:%M:%S)" >> V19T/smoke_status_v2.txt
nice -n 10 $PY code/engine_collapse_v19.py $A --out V19T --gpu 0 --tag t19b_rs16 --slices-c128 0.5 --slices-c64 1.0 --slice-dir V19T/slices > V19T/log_t19b_rs16.log 2>&1; echo "v19 fresh exit $? $(date -u +%H:%M:%S)" >> V19T/smoke_status_v2.txt
nice -n 10 $PY code/engine_collapse_v19.py $A --out V19T --gpu 0 --tag t19rb_rs16 --resume-from V19T/t17_rs16 --resume-T-over-rs 1.0 > V19T/log_t19rb_rs16.log 2>&1; echo "v19 resume exit $? $(date -u +%H:%M:%S)" >> V19T/smoke_status_v2.txt
echo "DONE $(date -u +%H:%M:%S)" >> V19T/smoke_status_v2.txt
