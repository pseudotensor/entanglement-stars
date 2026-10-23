#!/bin/bash
# launch_single_object_B_s8_frozen28_lane2c_v1.sh (builder analyst; the coordinator's faster option for B.6's frozen record T/r_s 28.0): the
# same run as launch_single_object_B_s8_frozen28_p2cpu_v4.sh on pod 2's CPU, on pod 1's GPU through lane 2c's runner: the sigma/a 8 history
# (C3lad27 c27 ell8, md5-equal to pod 2's; engine_fields_v16), frozen record T/r_s 28.0 to 300 r_s, kappa 1 and 2, --gpu 0. A memory gate in
# the pattern of the Page builder's wait launcher: the GPU's free memory at least NEED_GPU_GB (default 25; B main's process held 18.8 GB) on
# three polls 20 s apart, waiting at most 1 h (exit 8, nothing launched). Output S4one/B_s8_frozen28_gpu.json; the CPU job for T/r_s 28 is
# stopped by its exact PID once this file holds both channels.
cd ${QI2_ROOT}/engine || exit 1
mkdir -p S4one
S=S4one/status_B.txt
need=${NEED_GPU_GB:-25}
deadline=$(( $(date +%s) + 3600 ))
du=$(du -sm /workspace | tail -1 | cut -f1)
if [ "$du" -ge 40000 ]; then echo "B_s8_frozen28_gpu skipped: /workspace $du MB $(date -u +%H:%M:%S) UTC" >> $S; exit 1; fi
ok=0; free=0
while :; do
  line=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  tot=${line%,*}; used=${line#*,}
  free=$(( (tot - used) / 1024 ))
  if [ "$free" -ge "$need" ]; then ok=$((ok + 1)); else ok=0; fi
  [ "$ok" -ge 3 ] && break
  if [ "$(date +%s)" -gt "$deadline" ]; then echo "B_s8_frozen28_gpu: the GPU's free memory stayed below $need GB (last $free GB) for 1 h: no launch $(date -u +%H:%M:%S) UTC" >> $S; exit 8; fi
  sleep 20
done
echo "B_s8_frozen28_gpu start $(date -u +%H:%M:%S) UTC (/workspace $du MB; GPU free $free GB)" >> $S
python code/engine_single_object_B_v1.py --mode frozen --gpu 0 --record C3lad27/collapse_c27_rs64_A_R2_tau8_ell8_prep4_avg.json \
    --fields C3lad27/collapse_c27_rs64_A_R2_tau8_ell8_prep4_avg_fields.npz --fields-module engine_fields_v16 --launch-T 28.0 --kappas 1,2 --frozen-rs 300 \
    --out-json S4one/B_s8_frozen28_gpu.json > S4one/log_B_s8_frozen28_gpu.log 2>&1
echo "B_s8_frozen28_gpu exit $? $(date -u +%H:%M:%S) UTC" >> $S
