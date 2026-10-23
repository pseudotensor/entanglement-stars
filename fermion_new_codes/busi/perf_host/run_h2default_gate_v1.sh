#!/bin/bash
# run_h2default_gate_v1.sh (busi, analyst; inbox #343): the review's sec. 3 bytewise replay gate on engine_fields_v16c, box CPU, nice 10,
# 4 jobs x 2 threads (8 in all). off (H2_PW_ONE=0 H2_LOCAL_SPLINE=0) must equal v16's artifacts and default (both unset) v16b's A+B artifacts,
# on the last 20 records and the 24 stretch records of the T0012 mirror (H2AB/gate, H2AB/gate_stretch: the references, read only).
set -u
H=${QI2_ROOT}/H2AB; B=${QI2_ROOT}/H2default; PY=${QI2_ROOT}/python
IN=${QI2_ROOT}/collapse_RUNG2c40b_srf_fields_T0012.00.npz; ELL=2; NREC=20
RECS=4,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,43,47,51,55,67,79
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
G=$H/code/h2_field_gate_v1.py
mkdir -p $B/gate; cd $B/gate
U="env -u H2_PW_ONE -u H2_LOCAL_SPLINE -u H2_RESIDUAL_PAIR -u H2_GATE_RECS"
$U H2_PW_ONE=0 H2_LOCAL_SPLINE=0 nice -n 10 timeout 3h $PY -I $G $B/src_v16c $IN $ELL $NREC 1 off.pkl engine_fields_v16c > off.log 2>&1 &
$U nice -n 10 timeout 3h $PY -I $G $B/src_v16c $IN $ELL $NREC 1 default.pkl engine_fields_v16c > default.log 2>&1 &
$U H2_GATE_RECS=$RECS H2_PW_ONE=0 H2_LOCAL_SPLINE=0 nice -n 10 timeout 3h $PY -I $G $B/src_v16c $IN $ELL $NREC 1 off_stretch.pkl engine_fields_v16c > off_stretch.log 2>&1 &
$U H2_GATE_RECS=$RECS nice -n 10 timeout 3h $PY -I $G $B/src_v16c $IN $ELL $NREC 1 default_stretch.pkl engine_fields_v16c > default_stretch.log 2>&1 &
wait
{
for f in off default off_stretch default_stretch; do echo "== $f: $(head -1 $f.log) | $(tail -1 $f.log)"; done
$PY -I $G --compare $H/gate/original.pkl off.pkl
$PY -I $G --compare $H/gate/AB.pkl default.pkl
$PY -I $G --compare $H/gate_stretch/original.pkl off_stretch.pkl
$PY -I $G --compare $H/gate_stretch/AB.pkl default_stretch.pkl
for f in off default off_stretch default_stretch; do echo "$f W_nfev: $(grep -o 'W_nfev \[[0-9]*\]' $f.log | tr -d 'W_nfev[]' | tr '\n' ' ')"; done
md5sum $B/src_v16c/*.py $G $B/code/*.py
} > gate_summary.txt 2>&1
echo done > GATE_DONE
