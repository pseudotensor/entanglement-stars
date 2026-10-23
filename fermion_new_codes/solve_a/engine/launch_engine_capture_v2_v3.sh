#!/bin/bash
# Version 3 of launch_engine_capture_v2_v1.sh: version 2 on the primary restarted on engine_collapse_v12 (the owner's request of 15 October).
# Version 2: on the primary restarted on engine_collapse_v11, and it waits for the primary's own exit line in
# status.txt ("GPU1 primary_rs64_A_R2_tau8_v11 exit"; version 1 checked its guard as soon as the npz existed). Version 1: C-capture (engine 13.3;
# engine_capture_v2.py) on the primary's last slice, on the CPU (sparse), with the exact-exterior control and the
# absorbing-port comparator: waits until the primary's fields npz exists and its record reaches T/rs 30 (the guard of the J3' waiter), then runs
# the massless probes (omega r_s 2, 4, 8) and the fermion's (a omega 0.25, 0.3, 0.4 at m a 0.2). 4 threads, nice 10.
# Usage (on the box, detached, from engine): nohup bash code/launch_engine_capture_v2_v3.sh > CAP/launcher_v3.log 2>&1 < /dev/null &
# Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
mkdir -p CAP
PY=${QI2_ROOT}/python
REC=C3/collapse_primary_rs64_A_R2_tau8_v12.json
FLD=C3/collapse_primary_rs64_A_R2_tau8_v12_fields.npz
echo "C-capture v2 waiter (v3) start $(date -u +%H:%M:%S) UTC (after the v12 primary's exit line)" >> status.txt
until grep -q "^GPU1 primary_rs64_A_R2_tau8_v12 exit" status.txt && [ -f "$FLD" ]; do sleep 120; done
sleep 60
Tend=$($PY -c "import json; r=[q for q in json.load(open('$REC'))['records'] if 'T_over_rs' in q]; print(r[-1]['T_over_rs'] if r else 0)")
if ! $PY -c "import sys; sys.exit(0 if float('$Tend') >= 30.0 else 1)"; then
  echo "C-capture v2 not started: the primary's record ends at T/rs $Tend (below 30) $(date -u +%H:%M:%S) UTC" >> status.txt
  exit 0
fi
echo "C-capture v2 start $(date -u +%H:%M:%S) UTC (CPU, 4 threads)" >> status.txt
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 nice -n 10 $PY code/engine_capture_v2.py --fields $FLD --records $REC --slice -1 --control --absorber \
  --out-json CAP/capture_v2_primary_last.json --md CAP/capture_v2_primary_last.md > CAP/capture_v2_primary_last.log 2>&1 < /dev/null
echo "C-capture v2 exit $? $(date -u +%H:%M:%S) UTC" >> status.txt
