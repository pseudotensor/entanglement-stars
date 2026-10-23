#!/bin/bash
# T7.1 version 6, the plane-normal scan (SPEC_final4b_engine.md sec. 13.4 (1)): engine_ppn_v6.py on member 0 of star_t7_final.json, CPU, 1 thread, nice 10.
# Usage (on the box, detached): nohup bash launch_engine_ppn6_v1.sh > T7/ppn6_launcher.log 2>&1 < /dev/null &
# Builder analyst.
cd ${QI2_ROOT}/engine
PY=${QI2_ROOT}/python
echo "T7.1 ppn6 plane-normal scan start $(date -u +%H:%M:%S) UTC (CPU, 1 thread)" >> status.txt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 $PY code/engine_ppn_v6.py --star T7/star_t7_final.json --member 0 --out T7 --tag t7m0 > T7/ppn6_t7m0.log 2>&1 < /dev/null
echo "T7.1 ppn6 exit $? $(date -u +%H:%M:%S) UTC" >> status.txt
