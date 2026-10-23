#!/bin/bash
# launch_single_object_B2_fixed_lane2c_v1.sh (builder analyst, the HORIZON version 2 builder; engine spec 38.4 (1) (i), (iii), (iv) and 39.5
# (1) (vi), (viii)): engine_single_object_B_v2.py --mode fixed_field on pod 1's GPU through lane 2c's runner: (1) the production's records
# T/r_s 15.0 and 30.0 held fixed for 300 r_s in steps of 0.25 r_s, kappa 1 and 2, the 12 interior columns of version 1 and the 6 comparators at
# R_t + 2 sigma, the signed branch readout every 5 r_s; (2) the sigma/a 8 history (C3lad27 c27 ell8, engine_fields_v16) at T/r_s 15.0 and 28.0
# likewise. A memory gate in the pattern of the Page builder's wait launcher: the GPU's free memory at least NEED_GPU_GB (default 25) on three
# polls 20 s apart, at most 1 h (exit 8, nothing launched); a run is skipped when /workspace holds 40000 MB or more. Outputs on the container
# disk ${QI2_ROOT}/S4one_v2 (the pod rule); status lines in ${QI2_ROOT}/status.txt. Started by one queue line in queue_lane2c.txt (the runner cds
# to ${QI2_ROOT}/engine).
cd ${QI2_ROOT}/engine || exit 1
OUT=${QI2_ROOT}/S4one_v2
mkdir -p $OUT
S=$OUT/status.txt
need=${NEED_GPU_GB:-25}
gate() {
  deadline=$(( $(date +%s) + 3600 )); ok=0; free=0
  while :; do
    line=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
    tot=${line%,*}; used=${line#*,}; free=$(( (tot - used) / 1024 ))
    if [ "$free" -ge "$need" ]; then ok=$((ok + 1)); else ok=0; fi
    [ "$ok" -ge 3 ] && return 0
    if [ "$(date +%s)" -gt "$deadline" ]; then echo "$1: the GPU's free memory stayed below $need GB (last $free GB) for 1 h: no launch $(date -u +%H:%M:%S) UTC" >> $S; return 8; fi
    sleep 20
  done
}
run() {
  tag=$1; shift
  du=$(du -sm /workspace | tail -1 | cut -f1)
  if [ "$du" -ge 40000 ]; then echo "$tag skipped: /workspace $du MB $(date -u +%H:%M:%S) UTC" >> $S; return 1; fi
  gate $tag || return 8
  echo "$tag start $(date -u +%H:%M:%S) UTC (/workspace $du MB; GPU free $free GB)" >> $S
  python code/engine_single_object_B_v2.py "$@" > $OUT/log_$tag.log 2>&1
  echo "$tag exit $? $(date -u +%H:%M:%S) UTC" >> $S
}
run B2_fixed_prod --mode fixed_field --gpu 0 --record C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json --fields C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11_fields.npz \
    --fields-module engine_fields_v14 --launch-T 15.0,30.0 --kappas 1,2 --fixed-rs 300 --step-rs 0.25 --signed-every-rs 5 --out-json $OUT/B2_fixed_prod.json --out-npz $OUT/B2_fixed_prod.npz
run B2_fixed_s8 --mode fixed_field --gpu 0 --record C3lad27/collapse_c27_rs64_A_R2_tau8_ell8_prep4_avg.json --fields C3lad27/collapse_c27_rs64_A_R2_tau8_ell8_prep4_avg_fields.npz \
    --fields-module engine_fields_v16 --launch-T 15.0,28.0 --kappas 1,2 --fixed-rs 300 --step-rs 0.25 --signed-every-rs 5 --out-json $OUT/B2_fixed_s8.json --out-npz $OUT/B2_fixed_s8.npz
echo "all done $(date -u +%H:%M:%S) UTC" >> $S
