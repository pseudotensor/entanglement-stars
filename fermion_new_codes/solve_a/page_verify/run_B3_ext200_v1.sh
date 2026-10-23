#!/bin/bash
# run_B3_ext200_v1.sh (solve, page_verify B3; plan LEAD_reviewa_page_verify_routes_v1.md sec. B3 and day 0): the heavy extension of B
# (engine_collapse_v19, 2048 sites, the r_s 64 ell 4 tau 8 member) from its T/r_s 100 restart slice to T/r_s 200, with Brerun98's gated
# slice runner run_collapse_v19_from_slice_v1.py and B's arguments unchanged except --T-over-rs 200, --resume-T-over-rs 100, the slice list,
# the output paths and the tag. Steps: (0) the kit's md5 table (md5sum -c, exit 2 on a mismatch), (1) the dry run b3_preflight_v1.py
# (argparse of the engine plus the runner's static gate; exit 3 on a failure), (2) the runner. Nothing else.
# Usage: KIT=<dir with code/ in/ MANIFEST_B3.md5> PY=<python> GPU=<index> [OUT=<dir>] [SDIR=<dir>] [SLICES=104,108,...] [MINFREE=60] bash run_B3_ext200_v1.sh
#   card (H100): KIT=${QI2_ROOT}/B3ext PY=${QI2_ROOT}/python GPU=0 bash ${QI2_ROOT}/run_B3_ext200_v1.sh
#   box:         KIT=${QI2_ROOT}/kit PY=${QI2_ROOT}/python GPU=<free index> bash .../run_B3_ext200_v1.sh
set -u
KIT=${KIT:?KIT}; PY=${PY:?PY}; GPU=${GPU:?GPU}
OUT=${OUT:-$KIT/out}; SDIR=${SDIR:-$KIT/states}; MINFREE=${MINFREE:-60}
SLICES=${SLICES:-$(seq -s, 104 4 200)}
TAG=c19_rs64_A_R2_tau8_ell4_avg_ext200_r100
RES=$KIT/in/c19_rs64_A_R2_tau8_ell4_avg_ext100_r98
SLICE=$KIT/in/orbitals_c19_rs64_A_R2_tau8_ell4_avg_ext100_r98_T0100.00.npz
ST=$KIT/status_B3_ext200.txt
mkdir -p $OUT $SDIR $KIT/logs
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8} NUMPY_MADVISE_HUGEPAGE=0
echo "B3_ext200 start $(date -u +%FT%TZ) host $(hostname) py $PY gpu $GPU" >> $ST
( cd $KIT && md5sum -c --quiet MANIFEST_B3.md5 ) >> $ST 2>&1 || { echo "B3_ext200 KIT md5 FAIL $(date -u +%FT%TZ)" >> $ST; exit 2; }
echo "B3_ext200 KIT_OK $(date -u +%FT%TZ)" >> $ST
ARGS="--m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 2048 --ell 4 --kappa-sea 10 --tau 8 --T-over-rs 200 --reference R2 --half-offset 0 \
 --collar-policy flag --support-rel 1e-14 --out $OUT --site-fields averaged --stop-E 1e-7 --stop-fields 1e-6 --maxit 20 --mix plain --delta-rtol-every 10 \
 --commutator-every 1 --streams 0 --m1-simpson-steps auto --c-LZ 1.0 --window-max 4 --resume-from $RES --resume-T-over-rs 100 --slices-c128 $SLICES \
 --slice-dir $SDIR --slice $SLICE --min-free-gb $MINFREE --gpu $GPU --tag $TAG"
nice -n 10 $PY $KIT/code/b3_preflight_v1.py $KIT/code -- $ARGS > $KIT/logs/preflight_$TAG.json 2> $KIT/logs/preflight_$TAG.err
rc=$?; echo "B3_ext200 preflight rc $rc $(date -u +%FT%TZ)" >> $ST
[ $rc -eq 0 ] || exit 3
cd $KIT/code || exit 4
nice -n 10 $PY run_collapse_v19_from_slice_v1.py $ARGS > $KIT/logs/$TAG.log 2>&1 &
P=$!; echo $P > $KIT/logs/$TAG.pid; echo "B3_ext200 engine pid $P $(date -u +%FT%TZ)" >> $ST
wait $P; rc=$?
echo "B3_ext200 exit $rc $(date -u +%FT%TZ)" >> $ST
exit $rc
