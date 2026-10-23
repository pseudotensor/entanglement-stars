#!/bin/bash
# launch_s11_interim_T10_v1.sh (the lane's analyst builder; engine spec 39.4 (3): "an interim composition by s11_pathfinder_report_v3 at
# T/r_s 10 for the owner"): waits on the box until the S11 run's json holds a record at T/r_s >= 10, then copies the run's json and npz files
# (stable for 30 s, re-stated after the copy; the run is only read) to S11ball/snap_T10/, and composes
# S11ball/MASSLESS_ball_pathfinder_interim_T10_v1.md and .json by s11_pathfinder_report_v3.py with version 2's partial composition's inputs
# (the characteristic solve char_v2, the v32/v33 gate GATE33/gate_compare_v1.txt), every 4th record in the per-record table (all in the json).
# CPU only, nice 10, 1 thread. Usage (from ${QI2_ROOT}/engine): setsid nohup bash code/launch_s11_interim_T10_v1.sh &
cd ${QI2_ROOT}/engine || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=${QI2_ROOT}/python
TAG=s11_ball_rs16_ell2_tau2_v1
SRC=S11ball; SNAP=S11ball/snap_T10; LOG=S11ball/interim_T10_v1.log
mkdir -p $SNAP
now() { date -u +%H:%M:%S; }
last_T() { $PY -I -c "import json,sys; r=json.load(open(sys.argv[1]))['records']; print(r[-1]['T_over_rs'] if r else 0)" "$SRC/collapse_$TAG.json" 2>/dev/null || echo 0; }
sig() { stat -c '%Y %s' $SRC/collapse_${TAG}.json $SRC/collapse_${TAG}_*.npz 2>/dev/null | tr '\n' ' '; }
echo "$(now) UTC: waiting for a record at T/r_s >= 10" >> $LOG
while :; do
  T=$(last_T)
  if awk "BEGIN{exit !($T >= 10 - 1e-9)}"; then
    s=$(sig); newest=$(stat -c %Y $SRC/collapse_${TAG}.json $SRC/collapse_${TAG}_*.npz | sort -n | tail -1)
    if [ $(( $(date +%s) - newest )) -ge 30 ]; then
      cp $SRC/collapse_${TAG}.json $SRC/collapse_${TAG}_*.npz $SNAP/
      if [ "$(sig)" = "$s" ]; then break; fi
    fi
  fi
  sleep 60
done
echo "$(now) UTC: snapshot at T/r_s $(last_T) ($(md5sum $SNAP/collapse_${TAG}.json | cut -d' ' -f1))" >> $LOG
nice -n 10 $PY code/s11_pathfinder_report_v3.py --run $SNAP/$TAG --char $SRC/char_v2_${TAG}.json --gate GATE33/gate_compare_v1.txt --every 4 \
  --md $SRC/MASSLESS_ball_pathfinder_interim_T10_v1.md --out-json $SRC/MASSLESS_ball_pathfinder_interim_T10_v1.json >> $LOG 2>&1
echo "$(now) UTC: report exit $?" >> $LOG
