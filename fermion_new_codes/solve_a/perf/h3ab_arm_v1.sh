#!/bin/bash
# solve 8a7b5274, PERF H3 A/B on h100-f (busi ready_jobs_v3 sec. 35): ONE arm of the c64s8L segment T0013.00 -> 14.0 (32 records) under the
# bench_protocol 4.1 telemetry wrapper. Runs ON h100-f from BASE=${QI2_ROOT}/busi_compC (staged by h3ab_h100f_stage_v1.sh).
# pod_launch_C_v12t.sh (f28d75c2)'s segment mode reproduced word for word (the kit c_twofield_v9c2 run.sh form): snapshot copy of the run files,
# args.cut_transport record -> sector (cut_transport_before kept), then ENGINE_SNAPARGS anchor_passage="lz" (anchor_passage_before kept), then
#   nice -n 10 python code/$ENGINE $COMMON --slices-c128 "" --slices-light-c128 "" --ell 8 --slice-dir $G/states --gpu 0 --out $G --tag $TAG
#     --resume-slice $S --resume-run $SNAP/$TAG --segment-end-T-over-rs 14.0 $ENGINE_EXTRA
# Differences from the launcher, identical in every arm: the H1 thread limits from threads.env (14 = 16 vCPU - 2) in place of the launcher's 16;
# the python is h3ab_ptracer_v1.py (beside the engines) which only calls prctl(PR_SET_PTRACER_ANY) so the unprivileged py-spy can attach under
# yama ptrace_scope 1, then runpy-runs the engine unchanged. No npz cache wrapper (the restart of this 417-row run fits: 53 GB RSS on L40S/A6000).
# Usage: bash h3ab_arm_v1.sh H3_0|AC|ABC
ARM=${1:?arm}; BASE=${QI2_ROOT}/busi_compC; cd $BASE || exit 1
PYBIN=${QI2_ROOT}/bin; PY=$PYBIN/python; export PATH=$PYBIN:$PATH
TAG=C64_s8_ell8_tau2_twofield_v4L; S=$BASE/states/orbitals_${TAG}_T0013.00.npz
ON="--anchor-passage lz --perf-solve-memo on --perf-eig-cache-gib 4 --perf-dens-memo on --perf-uhx-cache-gib 2 --perf-writer on"
case $ARM in
  H3_0) ENGINE=engine_twofield_v9c3c.py; EM=c6f15b2e234cf9653ad374710c5cd14c; ENGINE_EXTRA="$ON";;
  AC)   ENGINE=engine_twofield_v9c4.py;  EM=152ec1c6193f452922d343196dbf9b18; ENGINE_EXTRA="$ON --perf-h3-band-gather --perf-h3-weight-gather";;
  ABC)  ENGINE=engine_twofield_v9c4.py;  EM=152ec1c6193f452922d343196dbf9b18; ENGINE_EXTRA="$ON --perf-h3-band-gather --perf-h3-weight-gather --perf-h3-anchor-workers 8";;
  *) echo "unknown arm $ARM"; exit 2;;
esac
ENGINE_SNAPARGS='anchor_passage="lz"'
set -a; . $BASE/threads.env; set +a
export BUSI_ENGINE_DIR=$BASE/engine BUSI_PAGE_DIR=$BASE/engine PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
# gates
[ "$(md5sum < code/$ENGINE | cut -c1-32)" = $EM ] || { echo "GATE engine md5"; exit 3; }
[ "$(md5sum < code/assign_primitive_v1f.py | cut -c1-32)" = 9c9abc1b116f1314f97f576d9cd620c7 ] || { echo "GATE assign_primitive_v1f md5"; exit 3; }
[ "$(md5sum < pod_launch_C_v12t.sh | cut -c1-32)" = f28d75c278fc4b3356371d163ab27a01 ] || { echo "GATE launcher md5"; exit 3; }
[ "$(cd engine && ls *.py | wc -l)" = 325 ] && [ "$(cd engine && LC_ALL=C md5sum *.py | LC_ALL=C sort -k2 | md5sum | cut -c1-32)" = c381cd185fba63601a8d5448cb8a1829 ] || { echo "GATE engine dir"; exit 3; }
[ "$(md5sum < $S | cut -c1-32)" = d17763e7aeb7dbeca8af72c5df814b8e ] || { echo "GATE slice md5"; exit 3; }
$PY -c "import numpy,scipy,torch,sys; sys.exit(0 if (numpy.__version__,scipy.__version__,torch.__version__)==('1.26.3','1.14.1','2.4.1+cu124') and torch.cuda.is_available() else 1)" || { echo "GATE python stack"; exit 3; }
APPS=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader); [ -z "$APPS" ] || { echo "GATE the GPU holds compute processes: $APPS"; exit 3; }
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); [ "$FREE" -ge 76600 ] || { echo "GATE $FREE MiB free < 76600 (the launcher's guard)"; exit 3; }
if [ $ARM = ABC ]; then $PY -c "import threadpoolctl" 2>/dev/null || { echo "GATE threadpoolctl absent: the B arm is dropped (spec sec. 35)"; exit 3; }; fi
# the launcher's snapshot and segment dir
stamp=$(date -u +%Y%m%dT%H%M%SZ); G=$BASE/gate/H3AB_${ARM}_$stamp; SNAP=$G/snap; T=$G/telemetry; mkdir -p $G/states $SNAP $T
cp -p results/C64v4L/collapse_${TAG}.json results/C64v4L/collapse_${TAG}_*.npz $SNAP/ || exit 4
$PY -c "import json,sys; p=sys.argv[1]; J=json.load(open(p)); c=J['args'].get('cut_transport','record'); (c != 'sector') and (J['args'].__setitem__('cut_transport_before', c), J['args'].__setitem__('cut_transport', 'sector'), json.dump(J, open(p,'w'), default=float))" $SNAP/collapse_$TAG.json || exit 4
for _kv in $ENGINE_SNAPARGS; do $PY -c "import json,sys; p=sys.argv[1]; k,v=sys.argv[2].split('=',1); J=json.load(open(p)); J['args'][k+'_before']=J['args'].get(k); J['args'][k]=json.loads(v); json.dump(J, open(p,'w'), default=float)" $SNAP/collapse_$TAG.json "$_kv" || exit 4; done
COMMON="--m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 2048 --prep-ell 4 --kappa-sea 10 --tau 2 --T-over-rs 56 --reference R2 --half-offset 0 --collar-policy flag --support-rel 1e-14 --site-fields averaged --stop-E 1e-7 --stop-fields 1e-6 --maxit 20 --mix newton --nk-maxiter 30 --nk-krylov 40 --nk-restarts 0 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3 --nk-jvp exact --delta-rtol-every 10 --commutator-every 1 --streams 0 --m1-simpson-steps auto --c-LZ 1.0 --window-max 4 --light-kappas 1,2 --light-guard-kappas 3 --light-coupling on --light-transport banded --light-d-min 1e-7 --s11-collars 8,16,24 --s11-stop 1 --field-kernel-cache --v36-fixes --reference-solve lagged --cut-transport sector --light-max-pieces 8192"
{ echo "arm $ARM engine $ENGINE $EM extra [$ENGINE_EXTRA] snapargs [$ENGINE_SNAPARGS]"; env | grep -E '^(OMP|MKL|OPENBLAS|NUMEXPR)_NUM_THREADS|^CUDA_|^PYTORCH|^BUSI'; nvidia-smi --query-gpu=name,memory.total,driver_version,uuid --format=csv,noheader; nproc; } > $G/arm.txt
# telemetry (bench_protocol 4.1 item 5; mpstat is absent on h100-f, so vmstat)
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm --format=csv -l 1 > $T/smi.csv 2>/dev/null & P1=$!
nvidia-smi dmon -s pucm -d 1 -o DT > $T/dmon.log 2>/dev/null & P2=$!
vmstat -t 5 > $T/cpu_vmstat.log 2>/dev/null & P3=$!
t0=$(date +%s); echo "start $(date -u +%FT%TZ)" > $G/wall.txt
nice -n 10 $PY $BASE/code/h3ab_ptracer_v1.py $BASE/code/$ENGINE $COMMON --slices-c128 "" --slices-light-c128 "" --ell 8 --slice-dir $G/states --gpu 0 --out $G --tag $TAG \
  --resume-slice "$S" --resume-run $SNAP/$TAG --segment-end-T-over-rs 14.0 $ENGINE_EXTRA > $G/segment_run.log 2>&1 &
EP=$!; echo $EP > $T/pid.txt; echo "engine pid $EP telemetry pids $P1 $P2 $P3 start $(date -u +%FT%TZ) dir $G"
# py-spy: 240 s once the records are running, from the end of record 13.3125 (so the sample covers the heavy 13.344) in every arm
( while kill -0 $EP 2>/dev/null; do grep -q "^T/rs 13.312 E/E0" $G/segment_run.log 2>/dev/null && break; sleep 5; done
  kill -0 $EP 2>/dev/null && $PYBIN/py-spy record --pid $EP --native --nonblocking --rate 100 --duration 240 --format speedscope -o $T/pyspy.json > $T/pyspy.out 2>&1
  kill -0 $EP 2>/dev/null && $PYBIN/py-spy dump --pid $EP --nonblocking > $T/pyspy_dump.txt 2>&1 ) & P4=$!
wait $EP; rc=$?; t1=$(date +%s)
echo "end $(date -u +%FT%TZ) rc $rc wall_s $((t1 - t0))" >> $G/wall.txt
sleep 2; kill $P1 $P2 $P3 2>/dev/null; wait $P4 2>/dev/null
$PY -c "import json,sys; J=json.load(open(sys.argv[1])); R=[r for r in J['records'] if r.get('E') is not None]; print(len(R), 'records, the last at T/rs', R[-1]['T_over_rs'])" $G/collapse_$TAG.json >> $G/wall.txt 2>&1
cat $G/wall.txt; exit $rc
