#!/bin/bash
# run_h2ab_timing_v1.sh (busi, analyst; PERF H2-A/H2-B): seconds per solve for off, A, B, A+B on the same 80 solves (20 records x 4),
# the four arms concurrently (same box load), plus the measurement copy at A+B and at off for the W and residuals_at split; nice 10, H1 limits
set -u
B=${QI2_ROOT}/H2AB; PY=${QI2_ROOT}/python
IN=${QI2_ROOT}/collapse_RUNG2c40b_srf_fields_T0012.00.npz; ELL=2; NREC=20; REPS=4
export OMP_NUM_THREADS=70 MKL_NUM_THREADS=70 OPENBLAS_NUM_THREADS=70
cd $B/timing
mkdir -p $B/src_tm && cp -p $B/src_v16b/engine_fields_v4.py $B/src_v16b/engine_fields_v12.py $B/src_v16b/engine_fields_v14.py $B/src_v16b/engine_fields_kcache_v1.py $B/src_tm/
$PY -I $B/code/mk_tm_copy_v1.py $B/src_v16b/engine_fields_v16b.py $B/src_tm/engine_fields_v16b_tm.py
U="env -u H2_PW_ONE -u H2_LOCAL_SPLINE -u H2_RESIDUAL_PAIR"
$U H2_PW_ONE=0 H2_LOCAL_SPLINE=0 nice -n 10 timeout 4h $PY -I $B/code/h2_timing_v1.py $B/src_v16b $IN $ELL $NREC $REPS engine_fields_v16b > off.log 2>&1 &
$U H2_PW_ONE=1 H2_LOCAL_SPLINE=0 nice -n 10 timeout 4h $PY -I $B/code/h2_timing_v1.py $B/src_v16b $IN $ELL $NREC $REPS engine_fields_v16b > A.log 2>&1 &
$U H2_PW_ONE=0 H2_LOCAL_SPLINE=1 nice -n 10 timeout 4h $PY -I $B/code/h2_timing_v1.py $B/src_v16b $IN $ELL $NREC $REPS engine_fields_v16b > B.log 2>&1 &
$U H2_PW_ONE=1 H2_LOCAL_SPLINE=1 nice -n 10 timeout 4h $PY -I $B/code/h2_timing_v1.py $B/src_v16b $IN $ELL $NREC $REPS engine_fields_v16b > AB.log 2>&1 &
$U nice -n 10 timeout 4h $PY -I $B/code/h2_timing_v1.py $B/src_orig $IN $ELL $NREC $REPS engine_fields_v16 > orig.log 2>&1 &
$U H2_PW_ONE=1 H2_LOCAL_SPLINE=1 nice -n 10 timeout 4h $PY -I $B/code/h2_timing_v1.py $B/src_tm $IN $ELL $NREC $REPS engine_fields_v16b_tm > AB_tm.log 2>&1 &
$U nice -n 10 timeout 4h $PY -I $B/code/h2_timing_v1.py $B/src_tm $IN $ELL $NREC $REPS engine_fields_v16b_tm > off_tm.log 2>&1 &
wait
{ for f in orig off A B AB off_tm AB_tm; do echo "== $f: $(grep '^solves' $f.log)"; grep '^W integration' $f.log; done; } > timing_summary.txt 2>&1
echo done > TIMING_DONE
