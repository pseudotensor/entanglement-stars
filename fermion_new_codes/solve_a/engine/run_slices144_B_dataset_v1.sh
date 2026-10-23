#!/bin/bash
# run_slices144_B_dataset_v1.sh (the busi lane as solve builder; RUN_slices144_v1 section 3.3; #193 item 11 (b) INTERIOR_continuation_v1 on
# B to 100 and B2): waits for the on-box B pass (run_slices144_B_v4.sh's status line), then the dataset-only slices one at a time (the
# driver downloads, sha256-checks, reads and deletes each; --min-free-gb its default): B 40 to 56, B2 at 100, then B2 every slice. The
# dataset token only from gr_exports.sh sourced inside this process (never printed or written). Finished slices are skipped on relaunch.
# Never stops or kills any process.
export CODE=${QI2_ROOT}/slices144 ENGINE=${QI2_ROOT}/code PAGE=${QI2_ROOT}/v6
export SLICES144_PAGE_CODE=${QI2_ROOT}/raymap
export OUT=${QI2_ROOT}/out SCRATCH=${QI2_ROOT}/scratch
P=${QI2_ROOT}/python
waitload() { while [ "$(awk -v n=$(nproc) '{printf "%d", 100*$1/n}' /proc/loadavg)" -ge 90 ]; do sleep 60; done; }
COMMON="--engine-code $ENGINE --page-code $PAGE --threads 16 --out $OUT --scratch $SCRATCH --max-load 90"
until [ -s $OUT/B_onbox.status ]; do sleep 120; done
source ${QI2_ROOT}/gr_exports.sh > /dev/null 2>&1
cd $CODE || exit 1
st() { echo "$* $(date -u +%FT%TZ)" >> $OUT/B_dataset.status; }
waitload; nice -n 10 $P -B $CODE/run_slices144_v1.py $COMMON --history B --clocks 40,42,44,46,48,50,52,54,56 --readouts interior,page > $OUT/B_dataset_40_56.log 2>&1; st "B 40-56 exit $?"
waitload; nice -n 10 $P -B $CODE/run_slices144_v1.py $COMMON --history B2 --clocks 100 --readouts interior,page > $OUT/B2_100.log 2>&1; st "B2 100 exit $?"
waitload; nice -n 10 $P -B $CODE/run_slices144_v1.py $COMMON --history B2 --clocks all --readouts interior,page > $OUT/B2_all.log 2>&1; st "B2 all exit $?"
st "dataset pass done"
