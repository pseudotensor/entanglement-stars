#!/bin/bash
# run_h2ab_timing_il_v1.sh (busi, analyst): the interleaved timing, 2 processes x 2 reps x 20 records = 80 solves per arm, nice 10, H1 limits
set -u
B=${QI2_ROOT}/H2AB; PY=${QI2_ROOT}/python
IN=${QI2_ROOT}/collapse_RUNG2c40b_srf_fields_T0012.00.npz
export OMP_NUM_THREADS=70 MKL_NUM_THREADS=70 OPENBLAS_NUM_THREADS=70
cd $B/timing
env -u H2_PW_ONE -u H2_LOCAL_SPLINE -u H2_RESIDUAL_PAIR nice -n 10 timeout 4h $PY -I $B/code/h2_timing_interleaved_v1.py $B/src_tm $IN 2 20 2 0 > il_p0.log 2>&1 &
env -u H2_PW_ONE -u H2_LOCAL_SPLINE -u H2_RESIDUAL_PAIR nice -n 10 timeout 4h $PY -I $B/code/h2_timing_interleaved_v1.py $B/src_tm $IN 2 20 2 2 > il_p1.log 2>&1 &
wait
uptime >> il_p1.log
echo done > TIMING_IL_DONE
