#!/bin/bash
# rd13_extend_v1.sh (busi, analyst; RD-1/RD-3 replay extended to the ruling's 8 records): arm NEW continues arm SRC from SRC's own end slice
# (its run dir as --resume-run, its json's arguments word for word, so the B arm keeps --eigh banded --jvp sternheimer) to END; the new
# run's json holds SRC's records (copied by the resume) plus the new ones. Same GPU, threads and environment as rd13_replay_v1.sh.
#   rd13_extend_v1.sh NEW SRC ENGINE END
set -u
NEW=$1; SRC=$2; ENG=$3; END=$4
B=${QI2_ROOT}/RD1_build; PY=${QI2_ROOT}/python; GPU=${GPU:-2}
S=$B/replay/$SRC; SL=$(ls $S/states/orbitals_rd13_${SRC}_T*.npz | tail -1)
out=$B/replay/$NEW; mkdir -p $out/states
args=$($PY $B/build/rd13_cmd_from_args_v1.py $S/collapse_rd13_${SRC}.json out=$out tag=rd13_$NEW slice_dir=$out/states slices_c128=$END gpu=0)
cmd="$PY -u $B/code/$ENG $args --resume-slice $SL --resume-run $S/rd13_$SRC --segment-end-T-over-rs $END"
echo "$cmd" > $out/cmd.txt
md5sum $B/code/$ENG $B/code/engine_banded_eigh_v1.py $B/code/engine_fields_v16.py $SL > $out/md5_inputs.txt 2>&1
echo "[$(date -u +%FT%TZ)] start $NEW from $SRC on GPU $GPU" >> $out/arm.log
cd $B/code
env CUDA_VISIBLE_DEVICES=$GPU NUMPY_MADVISE_HUGEPAGE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    nice -n 10 $cmd > $out/log.txt 2>&1
rc=$?
echo "[$(date -u +%FT%TZ)] end $NEW rc $rc" >> $out/arm.log
exit $rc
