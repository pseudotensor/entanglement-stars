#!/bin/bash
# launch_v6q_B3_card_v1.sh (solve, page_verify B3 light part, 20 Oct): busi's launch_v6q_B2_card_v1.sh (9ae0d5c4 / md5 of the box copy in the
# notes) with the record and fields of B3_ext200 (engine_collapse_v19, 2048 sites, records 0 to 200 r_s) in place of B2's 4096-site record,
# and three changes forced by that record: --nsites 16384 (the chain at least as long as the history in light-crossing terms: 256 r_s for
# 200 r_s, as B2's 8192 = 128 r_s for 100 r_s), --join-rs 16,32 (the join window inside the engine's 2048 sites; B2's 32,64 lies beyond
# them), --T-end-rs 200. Every other flag is B2's. Steps: (0) the kit's md5 tables (w1/MANIFEST_B3L.md5 for the code relative to PAGE, w1/MANIFEST_B3L_inputs.md5 for the record and fields relative to B3D; exit 2 on a mismatch), (1) the dry run
# b3l_preflight_v1.py (v6q's own argparse, the History load, the chain and join checks; exit 3 on a failure), (2) page_history_v6q.
# Usage: W1_PY=<python> [W1_DRYRUN=1: stop after the dry run] [W1_HARDMAX=16] [W1_MAX_WALL_HOURS=24] [W1_TEND=200] [W1_TAGSUF=] [B3D=<dir of the record>] [PAGE=<page dir>] bash launch_v6q_B3_card_v1.sh <card> <a> <b>
#   tourn1b, kappa 1: W1_PY=${QI2_ROOT}/python bash ${QI2_ROOT}/launch_v6q_B3_card_v1.sh tourn1b 1 1
set -u
export NUMPY_MADVISE_HUGEPAGE=0; CARD=$1; A=$2; B=$3
PAGE=${PAGE:-${QI2_ROOT}/page}; cd $PAGE || exit 4
export PYTHONPATH=$PAGE/code_v6:$PAGE/code_d13 PAGE_CODE_V6=$PAGE/code_v6 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=${B3D:-${QI2_ROOT}/out}; T=c19_rs64_A_R2_tau8_ell4_avg_ext200_r100; MAXH=${W1_MAX_WALL_HOURS:-24}; TEND=${W1_TEND:-200}
TAG=v6q_B3_k${A}to${B}${W1_TAGSUF:-}; mkdir -p runs/$TAG logs w1
ST=$PAGE/v6o_status.txt
echo "$TAG launch on $CARD $(date -u +%FT%TZ) cap_h $MAXH T_end $TEND gpu_used_MiB $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)" >> $ST
( cd $PAGE && md5sum -c --quiet w1/MANIFEST_B3L.md5 ) >> $ST 2>&1 || { echo "$TAG KIT md5 FAIL (code) $(date -u +%FT%TZ)" >> $ST; exit 2; }
( cd $D && md5sum -c --quiet $PAGE/w1/MANIFEST_B3L_inputs.md5 ) >> $ST 2>&1 || { echo "$TAG KIT md5 FAIL (inputs) $(date -u +%FT%TZ)" >> $ST; exit 2; }
echo "$TAG KIT_OK $(date -u +%FT%TZ)" >> $ST
ARGS="--record $D/collapse_$T.json --fields $D/collapse_${T}_fields.npz --fields-key applied --nsites 16384 --kappas $A-$B --kappa-hard-max ${W1_HARDMAX:-16} --no-kappa-extend \
 --interp linear --interp-sub 8 --cheb-tol 1e-16 --sector auto --sector-propagate on --sector-check-every-rs 10 --audit on --cuts-rs 2,16,24,30,48 --R-cut-rs 24 \
 --R-collar-rs 16 --join-rs 16,32 --collect-every-rs 2 --extra-ck-rs 15,30 --ext-every-rs 10 --vac-every-rs 10 --T-end-rs $TEND --save-every 4 --device cuda:0 \
 --threads 4 --out runs/$TAG --tag $TAG --resume --max-wall-hours $MAXH --columns-at-rs 12,20,40 --columns-every-rs 0"
nice -n 10 ${W1_PY:-python} w1/b3l_preflight_v1.py -- $ARGS > logs/preflight_$TAG.json 2> logs/preflight_$TAG.err
rc=$?; echo "$TAG preflight rc $rc $(date -u +%FT%TZ)" >> $ST
[ $rc -eq 0 ] || exit 3
[ "${W1_DRYRUN:-0}" = "1" ] && { echo "$TAG dry run only (W1_DRYRUN=1) $(date -u +%FT%TZ)" >> $ST; exit 0; }
nice -n 10 ${W1_PY:-python} code_d13/page_history_v6q.py $ARGS >> logs/$TAG.log 2>&1 &
echo $! > w1/$TAG.pid; wait $!; rc=$?
echo "$TAG exit $rc $(date -u +%FT%TZ)" >> $ST
exit $rc
