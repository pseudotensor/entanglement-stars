#!/bin/bash
# engine_massfluct_v2 on the (0.1, 1/4) family's closed-count members (SPEC_final4b_engine.md: "engine_massfluct after each family"), CPU,
# 2 threads, nice 10, sequential: members 1, 3, 5, 7, 9, 11, 13, 17 of Sfam5/star_m0.1_g0.25_ell8.json (N_B 112, 208, 416, 560, 800, 944, 1056,
# 1136). Usage (on the box, detached, from engine): nohup bash code/launch_engine_MF_v1.sh > MF/launcher_v1.log 2>&1 < /dev/null &
# Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
mkdir -p MF
PY=${QI2_ROOT}/python
for i in 1 3 5 7 9 11 13 17; do
  echo "MF sfam5 member $i start $(date -u +%H:%M:%S) UTC (CPU, 2 threads)" >> status.txt
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 $PY code/engine_massfluct_v2.py --star Sfam5/star_m0.1_g0.25_ell8.json --member $i \
    --out MF --tag sfam5_m$i > MF/sfam5_m$i.log 2>&1 < /dev/null
  echo "MF sfam5 member $i exit $? $(date -u +%H:%M:%S) UTC" >> status.txt
done
echo "MF sfam5 DONE $(date -u +%H:%M:%S) UTC" >> status.txt
