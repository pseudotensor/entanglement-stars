#!/bin/bash
# Strand S S3's control (SPEC sec. 9.5 (c)): the normal (2,1,0) (the largest mesh-2 weight) with the cap at 1024 (g_lim = 24/1024), on the first
# GPU with no process of this lane, one process (strandS_sphere_v4.py --gpu), gated on the lane's busy processes (at most 15).  Builder analyst.
cd ${QI2_ROOT}/sS_entropy || exit 1
mkdir -p S3v3ctl
mine_on() { for pid in $(nvidia-smi -i $1 --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do [ "$(ps -o user= -p $pid 2>/dev/null)" = "jon" ] && return 0; done; return 1; }
g=""
while [ -z "$g" ]; do
  for c in 0 1 2; do if ! mine_on $c && [ "$(bash ${QI2_ROOT}/lane_threads_v1.sh)" -le 15 ]; then g=$c; break; fi; done
  [ -z "$g" ] && sleep 30
done
date -u "+S3 cap1024 (2,1,0) start %H:%M:%S UTC (cuda:$g)" >> S3v3ctl/status.txt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 ${QI2_ROOT}/python code/strandS_sphere_v4.py \
  --normal 2,1,0 --Lcap 1024 --gpu $g --out-json S3v3ctl/sphere_210_cap1024.json > S3v3ctl/run_210_cap1024.log 2>&1
echo "S3 cap1024 (2,1,0) exit $? $(date -u +%H:%M:%S)" >> S3v3ctl/status.txt
