#!/bin/bash
# rd13_replay_v2.sh (busi, analyst; RD-1/RD-3 self-test part (c)): rd13_replay_v1.sh for the P64_s4 Newton member (qi2_h100_1, engine
# v36nq + H2 + --pair-workers 2, ISO_128's Newton block, 2048 sites): one arm from the stored T0011.50 complex128 slice to END with the
# production's arguments from its json word for word (gpu 0 under CUDA_VISIBLE_DEVICES=$GPU, the end record's slice written), nice 10,
# OMP/MKL/OPENBLAS 2 per process (the parent and its 2 pair workers: at most 6 threads), H2 on (engine_fields_v16.py = v16c).
#   rd13_replay_v2.sh ARM ENGINE END [extra engine options ...]
set -u
ARM=$1; ENG=$2; END=$3; shift 3
B=${QI2_ROOT}/RD1_build; PY=${QI2_ROOT}/python
P=${QI2_ROOT}/final; TAG0=lad_P64_s4_newton_v36nq_h100_1
SL=$P/states/orbitals_${TAG0}_T0011.50.npz
GPU=${GPU:-2}
out=$B/replay64/$ARM; mkdir -p $out/states
args=$($PY $B/build/rd13_cmd_from_args_v1.py $P/run/collapse_${TAG0}.json out=$out tag=rd13_$ARM slice_dir=$out/states slices_c128=$END gpu=0)
cmd="$PY -u $B/code/$ENG $args --resume-slice $SL --resume-run $P/run/$TAG0 --segment-end-T-over-rs $END $*"
echo "$cmd" > $out/cmd.txt
md5sum $B/code/$ENG $B/code/engine_banded_eigh_v1.py $B/code/engine_fields_v16.py $SL > $out/md5_inputs.txt 2>&1
echo "[$(date -u +%FT%TZ)] start $ARM on GPU $GPU" >> $out/arm.log
cd $B/code
env CUDA_VISIBLE_DEVICES=$GPU NUMPY_MADVISE_HUGEPAGE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    nice -n 10 $cmd > $out/log.txt 2>&1
rc=$?
echo "[$(date -u +%FT%TZ)] end $ARM rc $rc" >> $out/arm.log
exit $rc
