#!/bin/bash
# rd13_replay_v1.sh (busi, analyst; RD-1/RD-3 self-test part (c)): one arm of the ISO_128 nkqh2 stored-segment replay on the box GPU
# (CUDA_VISIBLE_DEVICES=$GPU, the engine's --gpu 0), from the run's T0005.00 complex128 slice, the production's own arguments from its json,
# 8 threads, nice 10, H2 on (engine_fields_v16.py = v16c in the code dir), the end record's slice written.
#   rd13_replay_v1.sh ARM ENGINE END_T_OVER_RS [extra engine options ...]
set -u
ARM=$1; ENG=$2; END=$3; shift 3
B=${QI2_ROOT}/RD1_build; PY=${QI2_ROOT}/python
I=${QI2_ROOT}/iso128_h2; TAG0=lad_ISO_128_s4_nkqh2_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_newton_v36nq
SL=$I/states_h2/orbitals_${TAG0}_T0005.00.npz
GPU=${GPU:-2}
out=$B/replay/$ARM; mkdir -p $out/states
args=$($PY $B/build/rd13_cmd_from_args_v1.py $I/run_h2/collapse_${TAG0}.json out=$out tag=rd13_$ARM slice_dir=$out/states slices_c128=$END gpu=0)
cmd="$PY -u $B/code/$ENG $args --resume-slice $SL --resume-run $I/run_h2/$TAG0 --segment-end-T-over-rs $END $*"
echo "$cmd" > $out/cmd.txt
md5sum $B/code/$ENG $B/code/engine_banded_eigh_v1.py $B/code/engine_fields_v16.py $SL > $out/md5_inputs.txt 2>&1
echo "[$(date -u +%FT%TZ)] start $ARM on GPU $GPU" >> $out/arm.log
cd $B/code
env CUDA_VISIBLE_DEVICES=$GPU NUMPY_MADVISE_HUGEPAGE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    nice -n 10 $cmd > $out/log.txt 2>&1
rc=$?
echo "[$(date -u +%FT%TZ)] end $ARM rc $rc" >> $out/arm.log
exit $rc
