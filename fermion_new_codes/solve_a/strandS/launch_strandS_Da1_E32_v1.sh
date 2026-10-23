#!/bin/bash
# Strand S diagnostic D-a1 at the largest size (E0 = 32, r_s0 = 64, r_h/h about 60; strandS_dos_v3.py) on the first GPU with no process of
# this lane (another user's idle litellm contexts on GPU 1 are not this lane's), one process, nice; gated on the lane's busy processes (at
# most 15).  Builder analyst.
cd ${QI2_ROOT}/sS_entropy || exit 1
mkdir -p Da1v3
mine_on() { for pid in $(nvidia-smi -i $1 --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do [ "$(ps -o user= -p $pid 2>/dev/null)" = "jon" ] && return 0; done; return 1; }
g=""
while [ -z "$g" ]; do
  for c in 2 0 1; do if ! mine_on $c && [ "$(bash ${QI2_ROOT}/lane_threads_v1.sh)" -le 15 ]; then g=$c; break; fi; done
  [ -z "$g" ] && sleep 30
done
date -u "+Da1 E32 start %H:%M:%S UTC (cuda:$g)" >> Da1v3/status.txt
STRAND7_CODE=${QI2_ROOT}/code OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 \
  ${QI2_ROOT}/python code/strandS_dos_v3.py --E0 32 --Rmax-over-rs 20 --device cuda:$g --out-json Da1v3/dos_E32.json > Da1v3/run_E32.log 2>&1
echo "Da1 E32 exit $? $(date -u +%H:%M:%S)" >> Da1v3/status.txt
