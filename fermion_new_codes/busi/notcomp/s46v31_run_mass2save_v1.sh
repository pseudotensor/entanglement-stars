#!/bin/bash
# busi #467 item 2: prod 15 operators for the leakage pilot (s46v31_mass2save_v1.py) on box GPU 1 (cuda:1), nice 10, 8 threads.
# Measured cost of the same pipeline on cuda:1 (s46v3 prod 35 log): about 1565 s; with --check-rout about 25 percent more.
PY=${QI2_ROOT}/python
C=${QI2_ROOT}/code
mkdir -p ${QI2_ROOT}/ops ${QI2_ROOT}/logs
cd $C
echo "$(date -u +%FT%TZ) start" > ${QI2_ROOT}/mass2save_p15.status
nice -n 10 $PY -B s46v31_mass2save_v1.py --slice prod 15 --check-rout --device cuda:1 --threads 8 > ${QI2_ROOT}/mass2save_p15.log 2>&1
echo "$(date -u +%FT%TZ) end rc=$?" >> ${QI2_ROOT}/mass2save_p15.status
