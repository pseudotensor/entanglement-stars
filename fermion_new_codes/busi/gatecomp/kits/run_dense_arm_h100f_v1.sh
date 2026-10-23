#!/bin/bash
# run_dense_arm_h100f_v1.sh (busi, GATECOMP kit (ii), ReviewA_s3b_v1 sec. 3): a W256 member's dense entry arm on h100-f: the member's own
# line with --eigh dense (lines_h100f_dense/W256_sS.args, one word different), from the member's slice FROM (the last slice before its
# first round-2 record), 4 records (--segment-end-T-over-rs FROM + 0.25), against the banded member's own records there: the mode read
# (A: equal counts and kept residuals to the reported digits, X-D on the table's lines; ReviewA_s3b_v1 sec. 1). Runs under the shared lock
# after the member (both do not fit 80 GB together). usage: run_dense_arm_h100f_v1.sh S FROM   (FROM like 11.50)
set -u
S=$1; FROM=$2; W=${QI2_ROOT}/busi_gatecomp; CODE=${QI2_ROOT}/code16d; PY=${QI2_ROOT}/python; M=$W/W256_s$S
A=$W/lines_h100f_dense/W256_s$S.args; TAG=$(tr ' ' '\n' < $A | awk 'p{print; exit} $0=="--tag"{p=1}')
FT=$(printf '%07.2f' $FROM); END=$(python3 -c "print('%.4f' % ($FROM + 0.25))"); O=$M/dense_arm_T$FT
SL=$M/states/orbitals_${TAG}_T$FT.npz; [ -e $SL ] || { echo "no slice $SL"; exit 9; }
mkdir -p $O/run $O/states
export OMP_NUM_THREADS=14 MKL_NUM_THREADS=14 OPENBLAS_NUM_THREADS=14 NUMEXPR_NUM_THREADS=14 NUMBA_NUM_THREADS=14 NUMPY_MADVISE_HUGEPAGE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset QI2_FIELDS_ROUTE QI2_FIELDS_WARM QI2_FIELDS_FLAT H2_PW_ONE H2_LOCAL_SPLINE
export PYTHONPATH=${QI2_ROOT}/peakhook
exec 9>>${QI2_ROOT}/busi_gpu.lock; flock 9
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 30; done
ARGS="$(sed -e "s#--slices-c128 [^ ]*#--slices-c128 40.0000#" -e "s#--slice-dir [^ ]*#--slice-dir $O/states#" -e "s#--out [^ ]*#--out $O/run#" $A) --resume-slice $SL --resume-run $M/run/$TAG --segment-end-T-over-rs $END"
echo "$(date -u +%FT%TZ) dense arm start: $ARGS" > $O/RUN.txt; md5sum $CODE/engine_collapse_v36nq_rd13mf.py $SL >> $O/RUN.txt
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -lms 5000 > $O/smi.csv 2>&1 & SMI=$!
( cd $CODE && $PY -B -u $CODE/engine_collapse_v36nq_rd13mf.py $ARGS ) > $O/log.txt 2>&1 < /dev/null
echo "$(date -u +%FT%TZ) dense arm rc $?" >> $O/RUN.txt; kill $SMI; flock -u 9
