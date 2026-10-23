#!/bin/bash
# launch_v36d_from0_continue_pod7_v2.sh: version 1 for pod 7 (inbox busi #153, #154): the same paths under /workspace; a gate mode gate18 that reruns T/r_s 18 to 18.125 aside from the T0018 slice and compares the 2 stored records at twice the floor (the T0017 slice stays on /data); no pausing disk guard (pod 7's 466 GB volume holds a whole rung; #153 asks the mirror to run at the link's pace instead). Version 1's header follows.
# launch_v36d_from0_continue_pod2_v1.sh (the busi lane as solve builder; spec 55b, turn 63c; inbox busi #144 item 5): the formation ladder's rungs
# sigma/a 4 and 8 on engine_collapse_v36d continued from their own T/r_s 18 slices to 40 on pod 2, launch_v36d_relaunch_pod4_v1.sh's form
# (--resume-slice SLICE --resume-run the rung's own run, a new tag <rung>_to40), the rungs' 55.1 arguments (launch_v36d_from0_rungs_pod2_v2.sh)
# with --T-over-rs 40 and --slices-c128 19..40. Modes:
#   gate L      : the resume test 55b (1) asks: from the rung's T0017.00 slice to T/r_s 18.0 aside (gate_ell<L>/, slice 18 only), then
#                 compare_member_floor_v2 --criterion twice against the rung's stored records; the test slice is deleted after the compare.
#   continue L  : from the rung's T0018.00 slice to 40, slices 19..40 in the rung's states dir, with the disk guard of 55b (2): the rung is
#                 paused (SIGSTOP) while /workspace has less than 2 slices' room (6 GB) and resumed (SIGCONT) above 9 GB.
L=${2:?ell 4 or 8}; MODE=${1:?gate or continue}
W=${QI2_ROOT}/v36d_from0; cd $W || exit 1
PY=${QI2_ROOT}/python; E=${QI2_ROOT}/engine_collapse_v36d.py; TAG=c32_rs64_A_R2_tau8_ell${L}_prep4_avg_plain_v36d_from0
BASE="--m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 2048 --kappa-sea 10 --reference R2 --site-fields averaged --prep-ell 4 --tau 8 --gpu 0 --mix plain --field-kernel-cache --v36-fixes --ell $L"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
st() { echo "$* $(date -u +%FT%TZ)" >> status.txt; }
if [ "$MODE" = gate18 ]; then
  G=$W/gate18_ell$L; mkdir -p $G/states && cd $G || exit 1
  S=$W/states/$TAG/orbitals_${TAG}_T0018.00.npz; [ -f $S ] || { st "gate18 ell $L: no T0018 slice"; exit 3; }
  st "gate18 ell $L start from $S (md5 $(md5sum $S | cut -c1-32))"
  nice -n 10 $PY $E $BASE --T-over-rs 18.125 --slices-c128 "" --slice-dir states --out . --tag ${TAG}_gate18 --resume-slice $S --resume-run $W/$TAG > gate_run.log 2>&1
  rc=$?; st "gate18 ell $L run exit $rc"
  $PY ${QI2_ROOT}/compare_member_floor_v2.py --run $G/${TAG}_gate18 --ref $W/$TAG --criterion twice --md $G/gate_compare.md --out-json $G/gate_compare.json > $G/gate_compare.log 2>&1
  rc2=$?; st "gate18 ell $L compare exit $rc2: $(tail -1 $G/gate_compare.log)"; exit $rc2
fi
if [ "$MODE" = gate ]; then
  G=$W/gate_ell$L; mkdir -p $G/states && cd $G || exit 1
  S=$W/states/$TAG/orbitals_${TAG}_T0017.00.npz; [ -f $S ] || { st "gate ell $L: no T0017 slice"; exit 3; }
  st "gate ell $L start from $S (md5 $(md5sum $S | cut -c1-32))"
  nice -n 10 $PY $E $BASE --T-over-rs 18.0 --slices-c128 18 --slice-dir states --out . --tag ${TAG}_gate --resume-slice $S --resume-run $W/$TAG > gate_run.log 2>&1
  rc=$?; st "gate ell $L run exit $rc"
  $PY ${QI2_ROOT}/compare_member_floor_v2.py --run $G/${TAG}_gate --ref $W/$TAG --criterion twice --md $G/gate_compare.md --out-json $G/gate_compare.json > $G/gate_compare.log 2>&1
  rc2=$?; st "gate ell $L compare exit $rc2: $(tail -1 $G/gate_compare.log)"
  F=$G/states/orbitals_${TAG}_gate_T0018.00.npz; [ -f "$F" ] && rm -f "${F:?}" && st "gate ell $L test slice deleted"
  exit $rc2
fi
[ "$MODE" = continue ] || { echo "mode gate or continue"; exit 2; }
S=$W/states/$TAG/orbitals_${TAG}_T0018.00.npz; [ -f $S ] || { st "continue ell $L: no T0018 slice"; exit 3; }
[ -e ${TAG}_to40.json ] || [ -e collapse_${TAG}_to40.json ] && { st "continue ell $L: refusing, ${TAG}_to40 exists"; exit 1; }
st "continue ell $L start from $S (md5 $(md5sum $S | cut -c1-32)) gpu_used_MiB $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) load $(cut -d' ' -f1 /proc/loadavg)"
nice -n 10 $PY $E $BASE --T-over-rs 40 --slices-c128 $(seq -s, 19 40) --slice-dir states/$TAG --out . --tag ${TAG}_to40 --resume-slice $S --resume-run $W/$TAG > logs/${TAG}_to40.log 2>&1 &
P=$!; echo $P > rung_ell${L}_to40.pid; st "continue ell $L pid $P"
wait $P; st "continue ell $L exit $?"; exit 0
wait $P; st "continue ell $L exit $?"
