#!/bin/bash
# run_v16e_hold_trace_box_v1.sh (busi; ReviewA_v16e_v1.md sec. 2 (i), supporting run, no card): the 5.1875 record alone, replayed on
# a box A6000 from arm Fd5's own T0005.12 slice (MERGE-1: Fd's engine, code dir and arguments to 5.125, its 2 records bitwise Fd's),
# with Fd's command word for word except the paths, --segment-end 5.1875, slices_c128 5.1875, gpu 0 under CUDA_VISIBLE_DEVICES=$GPU
# (never 0), and --nk-cache-host on (bitwise on the card at 4096 and on the box at 2048, MERGE-1 sec. 3 and 4; the dense arm does not
# fit an A6000 without it). The trace hook (tracehook/sitecustomize.py) logs every field solve's route; it changes nothing computed.
# The box is not the card: whether this path is Fd's is read from its log, not assumed.  usage: ARM=tr1 GPU=2 run_..._v1.sh [extra]
set -u
ARM=${ARM:-tr1}; GPU=${GPU:-2}; [ "$GPU" = 0 ] && { echo "never cuda:0"; exit 2; }
V=${QI2_ROOT}/v16e_hold; PY=${QI2_ROOT}/python; CODE=$V/code16e
F5=${QI2_ROOT}/Fd5
O=$V/trace_box/$ARM; mkdir -p $O/states
base=$(head -1 $F5/RUN.txt | sed 's/^.*Fd5 start: //')
args=$(echo "$base" | sed -e 's#^[^ ]* -B -u [^ ]*engine_collapse_v36nq_rd13mf.py ##' \
  -e "s#--out [^ ]*#--out $O#" -e "s#--tag [^ ]*#--tag vh_$ARM#" -e "s#--slice-dir [^ ]*#--slice-dir $O/states#" \
  -e "s#--slices-c128 [^ ]*#--slices-c128 5.1875#" -e "s#--resume-slice [^ ]*#--resume-slice $F5/orbitals_mf_Fd5_T0005.12.npz#" \
  -e "s#--resume-run [^ ]*#--resume-run $F5/mf_Fd5#" -e "s#--segment-end-T-over-rs [^ ]*#--segment-end-T-over-rs 5.1875#")
cmd="$PY -B -u $CODE/engine_collapse_v36nq_rd13mf.py $args --nk-cache-host on $*"
{ echo "$(date -u +%FT%TZ) $ARM start on box GPU $GPU: $cmd"; md5sum $CODE/engine_collapse_v36nq_rd13mf.py $CODE/engine_banded_eigh_v1.py $CODE/engine_fields_v16.py $F5/orbitals_mf_Fd5_T0005.12.npz; nvidia-smi; } > $O/RUN.txt 2>&1
cd $CODE
env CUDA_VISIBLE_DEVICES=$GPU NUMPY_MADVISE_HUGEPAGE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8 PYTHONPATH=$V/tracehook QI2_TRACE_DIR=$O/trace \
    nice -n 10 $cmd > $O/log.txt 2>&1
rc=$?
echo "$(date -u +%FT%TZ) $ARM rc $rc" >> $O/RUN.txt
exit $rc
