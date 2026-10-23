#!/bin/bash
# launch_single_object_B_s8_frozen_p2cpu_v3.sh (builder analyst; B.6's frozen records on their own, 8 threads, nice 10, started when B.5's
# CPU job has ended, so that this builder's pod 2 CPU jobs stay at about 16 threads combined (the coordinator's cap): version 2's launcher
# (launch_single_object_B_s8_p2cpu_v2.sh, history first, then frozen) is stopped by its exact PID before this starts, and its running history
# job finishes on its own. The same run as version 2's frozen line: the sigma/a 8 history (C3lad27 c27 ell8, engine_fields_v16), frozen records
# T/r_s 15.0 and 28.0 to 300 r_s, kappa 1 and 2, on pod 2's CPU (--gpu -1). Started directly by nohup. Output S4one/B_s8_frozen_v2.json.
cd ${QI2_ROOT}/engine || exit 1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
S=S4one/status_B.txt
REC=C3lad27/collapse_c27_rs64_A_R2_tau8_ell8_prep4_avg.json
FLD=C3lad27/collapse_c27_rs64_A_R2_tau8_ell8_prep4_avg_fields.npz
du=$(du -sm /workspace | tail -1 | cut -f1)
if [ "$du" -ge 40000 ]; then echo "B_s8_frozen skipped: /workspace $du MB $(date -u +%H:%M:%S) UTC" >> $S; exit 1; fi
echo "B_s8_frozen start $(date -u +%H:%M:%S) UTC (/workspace $du MB; CPU 8 threads; launcher v3)" >> $S
nice -n 10 python code/engine_single_object_B_v1.py --mode frozen --gpu -1 --record $REC --fields $FLD --fields-module engine_fields_v16 --launch-T 15.0,28.0 \
    --kappas 1,2 --frozen-rs 300 --out-json S4one/B_s8_frozen_v2.json > S4one/log_B_s8_frozen_v2.log 2>&1
echo "B_s8_frozen exit $? $(date -u +%H:%M:%S) UTC" >> $S
