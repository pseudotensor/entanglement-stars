#!/bin/bash
# run_h2ab_gate_stretch_v1.sh (busi; the 24 stretch records (3 segments, a gauge interval, a zero) of the T0012 mirror;, analyst; PERF H2-A/H2-B): the review's sec. 3 gate and the unit coverage on the box CPU, nice 10, H1 limits
set -u
B=${QI2_ROOT}/H2AB; PY=${QI2_ROOT}/python
IN=${QI2_ROOT}/collapse_RUNG2c40b_srf_fields_T0012.00.npz; ELL=2; NREC=20; RECS=4,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,43,47,51,55,67,79; export H2_GATE_RECS=$RECS
export OMP_NUM_THREADS=70 MKL_NUM_THREADS=70 OPENBLAS_NUM_THREADS=70
mkdir -p $B/gate_stretch; cd $B/gate_stretch
U="env -u H2_PW_ONE -u H2_LOCAL_SPLINE -u H2_RESIDUAL_PAIR"
$U nice -n 10 timeout 3h $PY -I $B/code/h2_field_gate_v1.py $B/src_orig $IN $ELL $NREC 1 original.pkl engine_fields_v16 > original.log 2>&1 &
$U nice -n 10 timeout 3h $PY -I $B/code/h2_field_gate_v1.py $B/src_v16b $IN $ELL $NREC 1 off.pkl engine_fields_v16b > off.log 2>&1 &
$U H2_PW_ONE=1 H2_LOCAL_SPLINE=0 nice -n 10 timeout 3h $PY -I $B/code/h2_field_gate_v1.py $B/src_v16b $IN $ELL $NREC 1 A.pkl engine_fields_v16b > A.log 2>&1 &
$U H2_PW_ONE=0 H2_LOCAL_SPLINE=1 nice -n 10 timeout 3h $PY -I $B/code/h2_field_gate_v1.py $B/src_v16b $IN $ELL $NREC 1 B.pkl engine_fields_v16b > B.log 2>&1 &
$U H2_PW_ONE=1 H2_LOCAL_SPLINE=1 nice -n 10 timeout 3h $PY -I $B/code/h2_field_gate_v1.py $B/src_v16b $IN $ELL $NREC 1 AB.pkl engine_fields_v16b > AB.log 2>&1 &
$U nice -n 10 timeout 3h $PY -I $B/code/h2_unit_cov_v1.py $B/src_v16b $IN $ELL $NREC > unit_cov.log 2>&1 &
wait
{
for f in original off A B AB; do echo "== $f: $(tail -1 $f.log)"; done
$PY -I $B/code/h2_field_gate_v1.py --compare original.pkl off.pkl
for c in A B AB; do $PY -I $B/code/h2_field_gate_v1.py --compare off.pkl $c.pkl; done
for f in original off A B AB; do echo "$f W_nfev: $(grep -o 'W_nfev \[[0-9]*\]' $f.log | tr -d 'W_nfev[]' | tr '\n' ' ')"; done
tail -3 unit_cov.log
md5sum $B/src_orig/engine_fields_v16.py $B/src_v16b/engine_fields_v16b.py $B/code/*.py
} > gate_summary.txt 2>&1
echo done > GATE_STRETCH_DONE
