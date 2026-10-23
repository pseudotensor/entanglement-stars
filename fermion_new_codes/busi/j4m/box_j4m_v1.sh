#!/bin/bash
# box_j4m_v1.sh (busi lane, WEB40G busi task 4): J4's massive rerun on the box CPU (RUN_ORDER_j4m_v1.md, box route), resumable: rerun the same
# script and the driver skips every complete record. Stage 1 the R 12 gate, stage 2 all 282 cases, stage 3 the fit. 4 workers of 4 threads
# (16 threads in all), nice 10. Records on ${QI2_ROOT}/records_box; the load is recorded at each stage.
B=${QI2_ROOT}/j4m_v1; C=${QI2_ROOT}/j4m; PY=${QI2_ROOT}/python
mkdir -p $B/records_box $B/cache_box $B/logs $B/fit
st() { echo "j4m $* $(date -u +%Y-%m-%dT%H:%M:%SZ) load $(cut -d' ' -f1 /proc/loadavg)" >> $B/status.txt; }
cd $C || exit 1
st "gate R 12 start"
nice -n 10 $PY -B j4m_driver_v1.py --radii 12 --outdir $B/records_box --cache-dir $B/cache_box --workers 4 --threads 4 --nice 10 --python $PY > $B/logs/driver_gate.log 2>&1
rc=$?; st "gate exit $rc"; [ $rc = 0 ] || exit $rc
$PY -B j4m_fit_v1.py --records $B/records_box --radii 12 --out-json $B/fit/J4m_fit_gate_v1.json > $B/logs/fit_gate.log 2>&1; st "gate fit exit $?"
st "full start"
nice -n 10 $PY -B j4m_driver_v1.py --radii 12 14 16 18 20 22 24 26 28 --outdir $B/records_box --cache-dir $B/cache_box --workers 4 --threads 4 --nice 10 --python $PY > $B/logs/driver_full.log 2>&1
rc=$?; st "full exit $rc"; [ $rc = 0 ] || exit $rc
$PY -B j4m_fit_v1.py --records $B/records_box --out-json $B/fit/J4m_fit_box_v1.json --out-tsv $B/fit/J4m_per_radius_box_v1.tsv > $B/logs/fit_full.log 2>&1
st "fit exit $?: $(grep -m1 ACCEPTANCE $B/logs/fit_full.log)"
