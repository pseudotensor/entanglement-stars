#!/bin/bash
# run_slices144_B_v3.sh (v3: the driver through run_slices144_v1.py, the single-import launcher the prod and s8 run used; v2 stopped on OPS None, the __main__ double import) (v2: ENGINE is ${QI2_ROOT}/code, as the prod and s8 run used; v1 stopped on engine_fields_v14 not found) (the busi lane as solve builder; RUN_slices144_v1 section 3.2; #193 item 11 (b) INTERIOR_continuation_v1): B's 22
# slices on the box (58 to 100; T100 is Brerun98's), interior and page readouts, box CPU; the code is the box copy in
# ${QI2_ROOT}/slices144 (the run file's $CODE is the checkout path, which the box does not hold). Finished slices are
# skipped on relaunch. Never stops or kills any process.
export CODE=${QI2_ROOT}/slices144 ENGINE=${QI2_ROOT}/code PAGE=${QI2_ROOT}/v6
export OUT=${QI2_ROOT}/out SCRATCH=${QI2_ROOT}/scratch
P=${QI2_ROOT}/python
waitload() { while [ "$(awk -v n=$(nproc) '{printf "%d", 100*$1/n}' /proc/loadavg)" -ge 90 ]; do sleep 60; done; }
COMMON="--engine-code $ENGINE --page-code $PAGE --threads 16 --out $OUT --scratch $SCRATCH --max-load 90"
cd $CODE || exit 1
waitload; nice -n 10 $P -B $CODE/run_slices144_v1.py $COMMON --history B --clocks 58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96,98,100 --readouts interior,page > $OUT/B_onbox_v3.log 2>&1
echo "B on-box exit $? $(date -u +%FT%TZ)" >> $OUT/B_onbox.status
