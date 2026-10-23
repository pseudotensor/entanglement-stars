#!/bin/bash
# J2 (SPEC_final4b_observables.md sec. 2): the Bisognano-Wichmann check with the flow term, L = 512, 1024, 2048, m_c = 0, 0.05, 0.2,
# 2 workers x 1 thread, on the box CPU; starts when the engine's count diagnostic (diag_pass0_count_v2.py, 2 threads) has ended, so the
# lane stays within 16 threads.  Builder analyst.
cd ${QI2_ROOT}/obs || exit 1
mkdir -p J2
while ps -eo args | grep -q "[d]iag_pass0_count_v2.py"; do sleep 30; done
echo "J2 start $(date -u +%H:%M:%S) UTC (CPU, 2 workers)" >> status.txt
OMP_NUM_THREADS=1 nice -n 10 ${QI2_ROOT}/python code/obs_bw_modular_v1.py --L 512 1024 2048 --mc 0.0 0.05 0.2 --workers 2 \
  --out-json J2/J2_bw_modular_v1.json --md J2/J2_bw_modular_v1.md > J2/J2_bw_modular_v1.log 2>&1
echo "J2 exit $? $(date -u +%H:%M:%S) UTC" >> status.txt
