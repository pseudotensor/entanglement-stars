#!/bin/bash
# pod6_gate_v7_c64s4off_v2.sh: version 1 without /usr/bin/time (absent on pod 6; version 1 exited 127 before any run): the wall time by date. Version 1 follows.
# pod6_gate_v7_c64s4off_v1.sh (busi lane; inbox #158, #172): box_gate_v7_c64s4off_v1.sh moved to pod 6's H100, idle after the control's end at
# T/r_s 12.781 (#172): the same device and environment as the C runs, so the comparisons against the stored records are the decisive ones too.
# Its header follows. box_gate_v7_c64s4off_v1.sh (busi lane; inbox #158; RUN_engine_twofield_v7.md section 4): the v7 performance option's gate on the box's free
# A6000 (GPU 2), from the C64 sigma/a 4 coupling-off control's stored T/r_s 12 orbitals slice (the /data keep copy, md5-checked) and a snapshot
# of its live run files from pod 6: 4 records with --perf-v7 off, then 4 with it on, each against the stored records (compare_resume_gate_v1),
# then on against off directly and by the self-test's compare-runs. The box environment (numpy 2.2.6, scipy 1.17.0, torch 2.8.0, an A6000)
# differs from the pods', so the against-stored verdicts measure the device and environment change too; on against off on the same device is
# the decisive bitwise test. The two runs go one after the other. Writes everything under ${QI2_ROOT}/v7gate.
set -u
B=${QI2_ROOT}/busi_compC; cd $B || exit 1; mkdir -p snap gate logs
PY=python
M=c64s4off; TAG=C64_s4_ell4_tau2_twofield_v3_off; RS=64; ELL=4; COUP=off; JVP="--nk-jvp exact"
SN=orbitals_${TAG}_T0012.00.npz
st() { echo "$(date -u +%FT%TZ) $*" >> $B/gate_v7_status.txt; }
[ -f states/$SN ] || { st "STOP: no $SN on the pod (copy it back from the box's keep copy first)"; exit 2; }
SLICE=$B/states/$SN
TS=$($PY -c "import numpy as np,sys; Z=np.load(sys.argv[1]); print(repr(float(Z['T'])/float(Z['rs'])))" $SLICE)
TE=$($PY -c "import sys; print(repr(float(sys.argv[1]) + 4 * 2.0 / float(sys.argv[2])))" $TS $RS)
stamp=$(date -u +%Y%m%dT%H%M%SZ); SNAP=$B/snap/${stamp}_$M; mkdir -p $SNAP
for try in 1 2 3 4 5; do
  cp -p results/C64/collapse_${TAG}.json results/C64/collapse_${TAG}_*.npz $SNAP/ || true
  $PY -c "import json,glob,sys,numpy as np; json.load(open(sys.argv[1])); [np.load(f, allow_pickle=True).files for f in glob.glob(sys.argv[2])]" \
    $SNAP/collapse_${TAG}.json "$SNAP/collapse_${TAG}_*.npz" 2>/dev/null && break
  [ $try = 5 ] && { st "STOP: snapshot unreadable 5 times"; exit 4; }; sleep 30
done
st "gate start: slice $SN (T/rs $TS to $TE), snapshot $SNAP; GPU $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader); load $(cut -d' ' -f1 /proc/loadavg)"
export BUSI_ENGINE_DIR=$B/engine BUSI_PAGE_DIR=$B/engine PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COMMON="--m 0.2 --g 0.25 --rs $RS --R0-over-rs 4 --reading A --nsites 2048 --prep-ell 4 --kappa-sea 10 --tau 2 --T-over-rs 56 --reference R2 --half-offset 0 --collar-policy flag --support-rel 1e-14 --site-fields averaged --stop-E 1e-7 --stop-fields 1e-6 --maxit 20 --mix newton --nk-maxiter 30 --nk-krylov 40 --nk-restarts 0 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3 $JVP --delta-rtol-every 10 --commutator-every 1 --streams 0 --m1-simpson-steps auto --c-LZ 1.0 --window-max 4 --light-kappas 1,2 --light-guard-kappas 3 --light-coupling $COUP --light-transport banded --light-d-min 1e-7 --s11-collars 8,16,24 --s11-stop 1 --field-kernel-cache --v36-fixes"
for mode in v7off v7on; do G=$B/gate/${stamp}_${M}_$mode; mkdir -p $G/states; PERF=""; [ $mode = v7on ] && PERF="--perf-v7"
  t0=$(date +%s); nice -n 10 $PY $B/code/engine_twofield_v7.py $COMMON --slices-c128 "" --slices-light-c128 "" --ell $ELL --slice-dir $G/states --gpu 0 --out $G --tag $TAG \
    --resume-slice $SLICE --resume-run $SNAP/$TAG --segment-end-T-over-rs $TE $PERF > $G/gate_run.log 2>&1
  rc=$?; st "$mode run exit $rc (elapsed $(( $(date +%s) - t0 )) s)"
  $PY $B/code/compare_resume_gate_v1.py --stored-dir $SNAP --gate-dir $G --tag $TAG --slice-T-over-rs $TS --out $G/resume_gate.json > $G/resume_gate.log 2>&1
  st "$mode against stored: exit $?: $(tail -1 $G/resume_gate.log)"
done
$PY $B/code/compare_resume_gate_v1.py --stored-dir $B/gate/${stamp}_${M}_v7off --gate-dir $B/gate/${stamp}_${M}_v7on --tag $TAG --slice-T-over-rs $TS --out $B/gate/${stamp}_${M}_v7on/on_vs_off.json > $B/gate/${stamp}_${M}_v7on/on_vs_off.log 2>&1
st "on against off: exit $?: $(tail -1 $B/gate/${stamp}_${M}_v7on/on_vs_off.log)"
$PY $B/code/engine_twofield_v7_selftest.py compare-runs $B/gate/${stamp}_${M}_v7off/$TAG $B/gate/${stamp}_${M}_v7on/$TAG --after $TS > $B/gate/${stamp}_${M}_v7on/compare_runs.log 2>&1
st "compare-runs: exit $?: $(tail -1 $B/gate/${stamp}_${M}_v7on/compare_runs.log)"
st "gate done"
