#!/bin/bash
# merge_replay64_v1.sh (busi, MERGE-1, self-test (c); the reviewer A tolerance ruling sec. 4 (A), (B), (C)): one arm of the P64_s4 Newton
# member's stored segment (qi2_h100_1, v36nq + H2 + --pair-workers 2, 2048 sites, K 10) from its T0011.50 complex128 slice to END (default
# 12.5: 8 records across the first formation record 12.125), the production's arguments from its json word for word (RD-1's
# rd13_cmd_from_args_v1.py, read only; gpu 0 under CUDA_VISIBLE_DEVICES=$GPU), the end record's slice written; the merged engine
# engine_collapse_v36nq_rd13mf.py from MERGE_build/$CODE (default code16e: engine_fields_v16.py = v16e 83fe8b2e, RD-2b ADOPTED WITH
# CONDITIONS, ruling v2), nice 10, OMP/MKL/OPENBLAS/NUMBA $THR per
# process (the parent and its 2 pair workers). rd13m_replay_v1.sh with the merged engine and code dir.
#   merge_replay64_v1.sh ARM END [extra engine options ...]
set -u
ARM=$1; END=$2; shift 2
M=${QI2_ROOT}/MERGE_build; PY=${QI2_ROOT}/python
P=${QI2_ROOT}/final; TAG0=lad_P64_s4_newton_v36nq_h100_1
SL=$P/states/orbitals_${TAG0}_T0011.50.npz
GPU=${GPU:-1}; THR=${THR:-1}; CODE=${CODE:-code16e}; [ "$GPU" = 0 ] && { echo "never cuda:0"; exit 2; }
ENG=engine_collapse_v36nq_rd13mf.py
out=$M/replay64/$ARM; mkdir -p $out/states
args=$($PY ${QI2_ROOT}/rd13_cmd_from_args_v1.py $P/run/collapse_${TAG0}.json out=$out tag=mf_$ARM slice_dir=$out/states slices_c128=$END gpu=0)
cmd="$PY -u $M/$CODE/$ENG $args --resume-slice $SL --resume-run $P/run/$TAG0 --segment-end-T-over-rs $END $*"
echo "$cmd" > $out/cmd.txt
md5sum $M/$CODE/$ENG $M/$CODE/engine_banded_eigh_v1.py $M/$CODE/engine_fields_v16.py $SL > $out/md5_inputs.txt 2>&1
echo "[$(date -u +%FT%TZ)] start $ARM on GPU $GPU, $THR threads per process" >> $out/arm.log
cd $M/$CODE
env CUDA_VISIBLE_DEVICES=$GPU NUMPY_MADVISE_HUGEPAGE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=$THR MKL_NUM_THREADS=$THR OPENBLAS_NUM_THREADS=$THR NUMBA_NUM_THREADS=$THR \
    nice -n 10 $cmd > $out/log.txt 2>&1
rc=$?
echo "[$(date -u +%FT%TZ)] end $ARM rc $rc" >> $out/arm.log
exit $rc
