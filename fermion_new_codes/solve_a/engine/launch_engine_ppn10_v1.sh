#!/bin/bash
# T7.1 version 10 (SPEC_final4b_engine.md sec. 17.2): engine_ppn_v10.py on member 0 of star_t7_final.json, two CPU processes in parallel (1 thread
# each, nice 10): the rays (three a omega, the lambda = 0 extrapolation, the fits) and the [011] orbits (m a, A/r_g, signed theta0, both senses,
# the windows and the tolerance ladder). Usage (on the box, detached, from engine): nohup bash code/launch_engine_ppn10_v1.sh > T7/ppn10_launcher.log 2>&1 < /dev/null &
# Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
PY=${QI2_ROOT}/python
run() { local tag=$1; shift; echo "T7.1 ppn10 $tag start $(date -u +%H:%M:%S) UTC (CPU, 1 thread)" >> status.txt
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 $PY code/engine_ppn_v10.py --star T7/star_t7_final.json --member 0 "$@" --out T7 --tag $tag > T7/ppn10_$tag.log 2>&1 < /dev/null
  echo "T7.1 ppn10 $tag exit $? $(date -u +%H:%M:%S) UTC" >> status.txt; }
run t7m0_rays --skip-orbits &
run t7m0_orbits --skip-rays &
wait
