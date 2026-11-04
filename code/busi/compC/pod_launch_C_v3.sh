#!/bin/bash
# Computation C's pathfinder, one member on an H100 pod, with the restart path (the resumability audit, inbox #86). pod_launch_C_v2.sh plus 2 modes:
#   pod_launch_C_v3.sh s4|s2 [BASE]                    a launch from T = 0, exactly as version 2;
#   pod_launch_C_v3.sh s4|s2 resume SLICE [BASE]       the restart: the member's run files are copied to BASE/snap/<stamp>/ (the engine reads the
#                                                      stored rows from that copy, never from files it is rewriting), then the engine continues from
#                                                      the complex128 orbitals slice SLICE into the same out dir and tag (version 34 (a) with
#                                                      light_resume_v36: the light state comes from the same slice). Refused while the member's
#                                                      recorded PID is alive;
#   pod_launch_C_v3.sh s4|s2 gate SLICE [N] [BASE]     the resume gate: the same restart for N records (default 2) into BASE/gate/<stamp>/ with no
#                                                      slices written, then compare_resume_gate_v1.py against the snapshot: BITWISE, WITHIN-FLOOR or
#                                                      FAIL. Safe beside a live member (it reads a snapshot) if the device has room (about 33 GB).
# The slice must be on the pod: c_mirror_v3 keeps the latest 2 orbitals slices there and the older ones on the box (${QI2_ROOT}/busi_bulk) and the
# owner's dataset; a restart from an older one needs it copied back first (scp from the box). Restart loss: at most the records after SLICE.
M=${1:?member s4 or s2}; MODE=fresh
case ${2:-} in resume|gate) MODE=$2; SLICE=${3:?slice}; shift 3;; *) shift 1;; esac
NREC=2; if [ "$MODE" = gate ] && [[ ${1:-} =~ ^[0-9]+$ ]]; then NREC=$1; shift; fi
B=${1:-${QI2_ROOT}/busi_compC}
case $M in s4) ELL=4; TAG=C32_s4_ell4_tau2_twofield_v3;; s2) ELL=2; TAG=C32_s2_ell2_tau2_twofield_v3;; *) echo "member s4 or s2"; exit 2;; esac
cd $B || exit 1; mkdir -p results/C32 states logs
export BUSI_ENGINE_DIR=$B/engine BUSI_PAGE_DIR=$B/engine PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SL4="--slices-c128 0,4,8,12,13,14,15,16,17,18,19,20,22,24,28,32,36,40 --slices-light-c128 1,2,3,5,6,7,9,10,10.5,11,11.5,12.5,13.5,14.5,15.5,16.5,17.5,18.5,19.5,20.5,21,21.5,22.5,23,23.5,25,26,27,29,30,31,33,34,35,37,38,39"
COMMON="--m 0.2 --g 0.25 --rs 32 --R0-over-rs 4 --reading A --nsites 2048 --prep-ell 4 --kappa-sea 10 --tau 2 --T-over-rs 40 --reference R2 --half-offset 0 --collar-policy flag --support-rel 1e-14 --site-fields averaged --stop-E 1e-7 --stop-fields 1e-6 --maxit 20 --mix newton --nk-maxiter 30 --nk-krylov 40 --nk-restarts 0 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3 --nk-jvp exact --delta-rtol-every 10 --commutator-every 1 --streams 0 --m1-simpson-steps auto --c-LZ 1.0 --window-max 4 --light-kappas 1,2 --light-guard-kappas 3 --light-coupling on --light-transport banded --light-d-min 1e-7 --s11-collars 8,16,24 --s11-stop 1 --field-kernel-cache --v36-fixes"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
st() { echo "C32_$M $* $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> $B/status.txt; }
gpuinfo="load $(cut -d' ' -f1 /proc/loadavg) gpu_used_MiB $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) free_MiB $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)"
if [ "$MODE" = fresh ]; then
  st "launch $gpuinfo"
  nice -n 10 python $B/code/engine_twofield_v3.py $COMMON $SL4 --ell $ELL --slice-dir $B/states --gpu 0 --out $B/results/C32 --tag $TAG > $B/logs/C32_$M.log 2>&1 &
  PID=$!; echo $PID > $B/C32_$M.pid; st "pid $PID"
  wait $PID; st "exit $?"; exit 0
fi
[ -f "$SLICE" ] || { echo "no slice $SLICE on this pod (copy it back from the box first)"; exit 3; }
TS=$(python -c "import numpy as np,sys; Z=np.load(sys.argv[1]); print(repr(float(Z['T'])/float(Z['rs'])))" "$SLICE") || exit 3
SNAP=$B/snap/${stamp}_$M; mkdir -p $SNAP
for try in 1 2 3; do   # a live member rewrites its files every 5 records: copy until the json and every npz read whole
  cp -p $B/results/C32/collapse_${TAG}.json $B/results/C32/collapse_${TAG}_*.npz $SNAP/ || exit 4
  python -c "import json,glob,sys,numpy as np; json.load(open(sys.argv[1])); [np.load(f, allow_pickle=True).files for f in glob.glob(sys.argv[2])]" \
    $SNAP/collapse_${TAG}.json "$SNAP/collapse_${TAG}_*.npz" 2>/dev/null && break
  [ $try = 3 ] && { echo "snapshot unreadable 3 times"; exit 4; }; sleep 20
done
if [ "$MODE" = resume ]; then
  if [ -f $B/C32_$M.pid ] && kill -0 "$(cat $B/C32_$M.pid)" 2>/dev/null; then echo "member $M still running (PID $(cat $B/C32_$M.pid)); not restarted"; exit 5; fi
  st "resume from $SLICE (T/rs $TS), snapshot $SNAP; $gpuinfo"
  nice -n 10 python $B/code/engine_twofield_v3.py $COMMON $SL4 --ell $ELL --slice-dir $B/states --gpu 0 --out $B/results/C32 --tag $TAG \
    --resume-slice "$SLICE" --resume-run $SNAP/$TAG > $B/logs/C32_${M}_resume_$stamp.log 2>&1 &
  PID=$!; echo $PID > $B/C32_$M.pid; st "pid $PID (resumed)"
  wait $PID; st "exit $? (resumed)"; exit 0
fi
G=$B/gate/${stamp}_$M; mkdir -p $G/states
TE=$(python -c "import sys; print(repr(float(sys.argv[1]) + int(sys.argv[2]) * 2.0 / 32.0))" "$TS" "$NREC")
st "resume gate from $SLICE (T/rs $TS to $TE), snapshot $SNAP, out $G; $gpuinfo"
nice -n 10 python $B/code/engine_twofield_v3.py $COMMON --slices-c128 "" --slices-light-c128 "" --ell $ELL --slice-dir $G/states --gpu 0 --out $G --tag $TAG \
  --resume-slice "$SLICE" --resume-run $SNAP/$TAG --segment-end-T-over-rs $TE > $G/gate_run.log 2>&1
rc=$?; st "resume gate run exit $rc"
python $B/code/compare_resume_gate_v1.py --stored-dir $SNAP --gate-dir $G --tag $TAG --slice-T-over-rs $TS --out $G/resume_gate.json > $G/resume_gate.log 2>&1
rc2=$?; st "resume gate compare exit $rc2: $(tail -1 $G/resume_gate.log)"; exit $rc2
