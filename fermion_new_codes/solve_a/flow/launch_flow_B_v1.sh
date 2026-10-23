#!/bin/bash
# Task 3 part (b) B0 and B1 (SPEC_task3_prescribed_flow.md sec. 5; flow_B_conversion_v1.py job files): the record's conversion package
# (flow/record_spectro, unchanged copies of review_output/layer_response_2026-10-02/spectro_work's lattice.py, campaign.py, tscan_report.py)
# on GPU 1 (kappa a = 1/8, then half of 1/32) and GPU 0 (the other half of 1/32, once this lane has no process there); each start gated on
# the lane's busy processes (at most 15) and on no process of this lane on the GPU (another user's idle litellm contexts on GPU 1 are not
# this lane's, as for K1 and S1's ladder); then the report.  nice 10, one CPU thread per slot.  Builder analyst.
F=${QI2_ROOT}/flow
cd $F || exit 1
PY=${QI2_ROOT}/python
mine_on() { for pid in $(nvidia-smi -i $1 --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do [ "$(ps -o user= -p $pid 2>/dev/null)" = "jon" ] && return 0; done; return 1; }
gate() { until [ "$(bash ${QI2_ROOT}/lane_threads_v1.sh)" -le 15 ] && ! mine_on $1; do sleep 30; done; }
run() {  # run GPU DIR [shard args]
  local g=$1; shift; local d=$1; shift
  gate $g
  date -u "+B $d $* start %H:%M:%S on cuda:$g" >> B/status.txt
  (cd record_spectro && CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 $PY -u campaign.py work ../B/$d --device cuda "$@" >> ../B/work_${d}_gpu$g.log 2>&1)
  echo "B $d $* exit $? $(date -u +%H:%M:%S)" >> B/status.txt
}
( run 1 k8; run 1 k32 --shard 1 --shards 2 ) &
( sleep 20; run 0 k32 --shard 0 --shards 2 ) &
wait
$PY flow_B_conversion_v1.py report B/k8 B/k32 --package record_spectro --json B/flow_B_report_v1.json --md B/RESULT_flow_B_v1.md > B/report.log 2>&1
echo "B report exit $? $(date -u +%H:%M:%S)" >> B/status.txt
echo DONE >> B/status.txt
