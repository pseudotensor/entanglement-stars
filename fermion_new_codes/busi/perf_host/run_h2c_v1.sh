#!/bin/bash
# run_h2c_v1.sh (busi, analyst; PERF H2-C): build the --with-c candidate as engine_fields_v16bc.py, the sec. 3 gate (original -> off, off -> C,
# off -> A+B+C; repeated twice for C, concurrency), then the interleaved AB vs ABC timing; box CPU, nice 10, H1 limits, H2_CPU_QUOTA=22.1 (R1's)
set -u
B=${QI2_ROOT}/H2AB; PY=${QI2_ROOT}/python
IN=${QI2_ROOT}/collapse_RUNG2c40b_srf_fields_T0012.00.npz; ELL=2; NREC=20
export OMP_NUM_THREADS=70 MKL_NUM_THREADS=70 OPENBLAS_NUM_THREADS=70
mkdir -p $B/src_v16bc $B/gate_c
cp -p $B/src_orig/engine_fields_v4.py $B/src_orig/engine_fields_v12.py $B/src_orig/engine_fields_v14.py $B/src_orig/engine_fields_kcache_v1.py $B/src_orig/engine_fields_v16.py $B/src_v16bc/
[ -f $B/src_v16bc/engine_fields_v16bc.py ] || $PY -I $B/code/build_engine_fields_v16b.py --src $B/src_v16bc/engine_fields_v16.py --with-c --dst $B/src_v16bc/engine_fields_v16bc.py
cd $B/gate_c
U="env -u H2_PW_ONE -u H2_LOCAL_SPLINE -u H2_RESIDUAL_PAIR"
$U nice -n 10 timeout 3h $PY -I $B/code/h2_field_gate_v1.py $B/src_v16bc $IN $ELL $NREC 1 off.pkl engine_fields_v16bc > off.log 2>&1 &
$U H2_RESIDUAL_PAIR=1 H2_CPU_QUOTA=22.1 nice -n 10 timeout 3h $PY -I $B/code/h2_field_gate_v1.py $B/src_v16bc $IN $ELL $NREC 1 C.pkl engine_fields_v16bc > C.log 2>&1 &
$U H2_RESIDUAL_PAIR=1 H2_CPU_QUOTA=22.1 nice -n 10 timeout 3h $PY -I $B/code/h2_field_gate_v1.py $B/src_v16bc $IN $ELL $NREC 1 C2.pkl engine_fields_v16bc > C2.log 2>&1 &
$U H2_PW_ONE=1 H2_LOCAL_SPLINE=1 H2_RESIDUAL_PAIR=1 H2_CPU_QUOTA=22.1 nice -n 10 timeout 3h $PY -I $B/code/h2_field_gate_v1.py $B/src_v16bc $IN $ELL $NREC 1 ABC.pkl engine_fields_v16bc > ABC.log 2>&1 &
wait
{
$PY -I $B/code/h2_field_gate_v1.py --compare $B/gate/original.pkl off.pkl
for c in C C2 ABC; do $PY -I $B/code/h2_field_gate_v1.py --compare off.pkl $c.pkl; done
md5sum $B/src_v16bc/engine_fields_v16bc.py
} > gate_c_summary.txt 2>&1
$U H2_CPU_QUOTA=22.1 nice -n 10 timeout 4h $PY -I $B/code/h2_timing_il_c_v1.py $B/src_v16bc $IN $ELL 20 2 0 > il_c_p0.log 2>&1 &
$U H2_CPU_QUOTA=22.1 nice -n 10 timeout 4h $PY -I $B/code/h2_timing_il_c_v1.py $B/src_v16bc $IN $ELL 20 2 1 > il_c_p1.log 2>&1 &
wait
uptime >> il_c_p1.log
echo done > C_DONE
