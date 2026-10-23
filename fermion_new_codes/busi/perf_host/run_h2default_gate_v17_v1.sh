#!/bin/bash
# run_h2default_gate_v17_v1.sh (busi, analyst; inbox #343): engine_fields_v17 (11828365) over v16c against v17 over v16, the review's
# sec. 3 gate with module engine_fields_v17: src_v17ref = v16 1ffcf5e8 + v17; src_v17c = v16c's content under the name engine_fields_v16.py
# (the kit route) + v17; options at their defaults (unset: v16c on). Inputs: the T0012 mirror itself (sources at the centre: v17 delegates
# to v16's solve) and its shell copy (mk_v17_shell_input_v1.py, R <= 40 zeroed: v17's own empty-interior path). Box CPU, nice 10, 4 x 2 threads.
set -u
H=${QI2_ROOT}/H2AB; B=${QI2_ROOT}/H2default; PY=${QI2_ROOT}/python
IN=${QI2_ROOT}/collapse_RUNG2c40b_srf_fields_T0012.00.npz; ELL=2; NREC=20
SH=$B/inputs/shell_R40_T0012.npz; RC=${QI2_ROOT}/code
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
G=$H/code/h2_field_gate_v1.py
for d in src_v17ref src_v17c; do
  [ -d $B/$d ] || { mkdir -p $B/$d; cp -p $B/src_v16c/engine_fields_kcache_v1.py $B/src_v16c/engine_fields_v12.py $B/src_v16c/engine_fields_v14.py $B/src_v16c/engine_fields_v4.py $RC/engine_fields_v17.py $B/$d/; }
done
[ -f $B/src_v17ref/engine_fields_v16.py ] || cp -p $B/src_v16c/engine_fields_v16.py $B/src_v17ref/engine_fields_v16.py
[ -f $B/src_v17c/engine_fields_v16.py ] || cp $B/src_v16c/engine_fields_v16c.py $B/src_v17c/engine_fields_v16.py
mkdir -p $B/inputs; [ -f $SH ] || $PY -I $B/code/mk_v17_shell_input_v1.py $IN $SH 40
mkdir -p $B/gate_v17; cd $B/gate_v17
U="env -u H2_PW_ONE -u H2_LOCAL_SPLINE -u H2_RESIDUAL_PAIR -u H2_GATE_RECS"
$U nice -n 10 timeout 3h $PY -I $G $B/src_v17ref $IN $ELL $NREC 1 ref_mirror.pkl engine_fields_v17 > ref_mirror.log 2>&1 &
$U nice -n 10 timeout 3h $PY -I $G $B/src_v17c $IN $ELL $NREC 1 c_mirror.pkl engine_fields_v17 > c_mirror.log 2>&1 &
$U nice -n 10 timeout 3h $PY -I $G $B/src_v17ref $SH $ELL $NREC 1 ref_shell.pkl engine_fields_v17 > ref_shell.log 2>&1 &
$U nice -n 10 timeout 3h $PY -I $G $B/src_v17c $SH $ELL $NREC 1 c_shell.pkl engine_fields_v17 > c_shell.log 2>&1 &
wait
{
md5sum $B/src_v17ref/*.py $B/src_v17c/*.py $SH
for f in ref_mirror c_mirror ref_shell c_shell; do echo "== $f: $(head -1 $f.log) | $(tail -1 $f.log)"; done
$PY -I $G --compare ref_mirror.pkl c_mirror.pkl
$PY -I $G --compare ref_shell.pkl c_shell.pkl
for f in ref_mirror ref_shell; do echo "== path $f"; $PY -I $B/code/v17_path_probe_v1.py $f.pkl | tail -4; done
} > gate_v17_summary.txt 2>&1
echo done > GATE_V17_DONE
