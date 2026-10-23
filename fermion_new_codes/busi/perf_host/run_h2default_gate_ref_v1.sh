#!/bin/bash
# run_h2default_gate_ref_v1.sh (busi, analyst; inbox #343): the same-thread references for run_h2default_gate_v1.sh. The field solve's
# bytes depend on the BLAS thread count (H2AB's references were made at 70 threads; at 2 threads v16's own W nfev differ), so v16 (H2AB/src_orig)
# and v16b A+B (H2AB/src_v16b, H2_PW_ONE=1 H2_LOCAL_SPLINE=1) are re-run at the gate's 2 threads; then off -> v16 and default -> v16b A+B.
set -u
H=${QI2_ROOT}/H2AB; B=${QI2_ROOT}/H2default; PY=${QI2_ROOT}/python
IN=${QI2_ROOT}/collapse_RUNG2c40b_srf_fields_T0012.00.npz; ELL=2; NREC=20
RECS=4,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,43,47,51,55,67,79
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
G=$H/code/h2_field_gate_v1.py
cd $B/gate
U="env -u H2_PW_ONE -u H2_LOCAL_SPLINE -u H2_RESIDUAL_PAIR -u H2_GATE_RECS"
$U nice -n 10 timeout 3h $PY -I $G $H/src_orig $IN $ELL $NREC 1 v16_t2.pkl engine_fields_v16 > v16_t2.log 2>&1 &
$U H2_PW_ONE=1 H2_LOCAL_SPLINE=1 nice -n 10 timeout 3h $PY -I $G $H/src_v16b $IN $ELL $NREC 1 v16bAB_t2.pkl engine_fields_v16b > v16bAB_t2.log 2>&1 &
$U H2_GATE_RECS=$RECS nice -n 10 timeout 3h $PY -I $G $H/src_orig $IN $ELL $NREC 1 v16_t2_stretch.pkl engine_fields_v16 > v16_t2_stretch.log 2>&1 &
$U H2_GATE_RECS=$RECS H2_PW_ONE=1 H2_LOCAL_SPLINE=1 nice -n 10 timeout 3h $PY -I $G $H/src_v16b $IN $ELL $NREC 1 v16bAB_t2_stretch.pkl engine_fields_v16b > v16bAB_t2_stretch.log 2>&1 &
wait
{
for f in v16_t2 v16bAB_t2 v16_t2_stretch v16bAB_t2_stretch; do echo "== $f: $(head -1 $f.log) | $(tail -1 $f.log)"; done
echo "-- off -> v16 (same 2 threads)"; $PY -I $G --compare v16_t2.pkl off.pkl; $PY -I $G --compare v16_t2_stretch.pkl off_stretch.pkl
echo "-- default -> v16b A+B (same 2 threads)"; $PY -I $G --compare v16bAB_t2.pkl default.pkl; $PY -I $G --compare v16bAB_t2_stretch.pkl default_stretch.pkl
echo "-- off -> default (v16c's own on/off)"; $PY -I $G --compare off.pkl default.pkl; $PY -I $G --compare off_stretch.pkl default_stretch.pkl
for f in v16_t2 off default v16bAB_t2; do echo "$f W_nfev: $(grep -o 'W_nfev \[[0-9]*\]' $f.log | tr -d 'W_nfev[]' | tr '\n' ' ')"; done
} > gate_ref_summary.txt 2>&1
echo done > GATE_REF_DONE
