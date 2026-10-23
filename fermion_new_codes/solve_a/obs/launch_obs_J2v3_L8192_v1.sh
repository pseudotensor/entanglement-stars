#!/bin/bash
# B9 (observables 11.5): the one L = 8192 row, after J2's L = 4096 set: waits until the L = 4096 process (obs_bw_modular_v2.py --L 4096) has
# exited, then runs obs_bw_modular_v3.py at L = 8192, (0, 0.6), W = 256, the centre cut, m_c = 0 (with its v = 0 reference) on 16 host
# threads, nice 10. Usage (on the box, in ${QI2_ROOT}/obs, detached):
#   nohup bash code/launch_obs_J2v3_L8192_v1.sh > J2v2/launcher_L8192_v1.log 2>&1 < /dev/null &
# Builder analyst.
cd ${QI2_ROOT}/obs
PY=${QI2_ROOT}/python
echo "J2v3 L8192 waiter start $(date -u +%H:%M:%S) UTC (after the L 4096 set)" >> status.txt
while ps -eo args | awk '$2=="code/obs_bw_modular_v2.py" && $3=="--L" && $4=="4096"' | grep -q .; do sleep 60; done
echo "J2v3 L8192 start $(date -u +%H:%M:%S) UTC (CPU, 16 threads)" >> status.txt
OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 nice -n 10 $PY code/obs_bw_modular_v3.py --L 8192 --W 256 --mc 0 --profiles 0:0.6 --shifts 0 \
  --out-json J2v2/J2v3_L8192.json --md J2v2/J2v3_L8192.md > J2v2/J2v3_L8192.log 2>&1 < /dev/null
echo "J2v3 L8192 exit $? $(date -u +%H:%M:%S) UTC" >> status.txt
