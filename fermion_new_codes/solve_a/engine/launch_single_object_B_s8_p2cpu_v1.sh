#!/bin/bash
# launch_single_object_B_s8_p2cpu_v1.sh (builder analyst; inbox #41's revised S4 part B, B.6: "the sigma/a 8 history as the scale control"):
# engine_single_object_B_v1.py on the sigma/a 8 member's solved history (C3lad27/collapse_c27_rs64_A_R2_tau8_ell8_prep4_avg, field module
# engine_fields_v16, as part A used): (1) frozen records T/r_s 15.0 and 28.0 to 300 r_s; (2) the solved history from those clocks to T/r_s 40,
# with the cubic control on the first 8 r_s after T/r_s 15.0; kappa 1 and 2; the massive channel only (D14); no mutual information (D15).
# T/r_s 28.0 stands in for the primary's 30.0, where the sigma/a 8 history is untrapped (its second episode runs T/r_s 26.625 to 29.75 by the
# record's stored zeros). Runs on pod 2's CPU (--gpu -1; 16 threads, nice 10): at 20:50 UTC the GPU had 14.8 GB free beside the ladder
# member, the replay and the Page builder's A variant (about 7 GB at that variant's 32.5 GB peak), and B main's process held 18.8 GB.
# Started directly by nohup, not by a queue runner. Outputs in S4one/; status lines in S4one/status_B.txt; a run is skipped when /workspace
# holds 40000 MB or more.
cd ${QI2_ROOT}/engine || exit 1
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
S=S4one/status_B.txt
REC=C3lad27/collapse_c27_rs64_A_R2_tau8_ell8_prep4_avg.json
FLD=C3lad27/collapse_c27_rs64_A_R2_tau8_ell8_prep4_avg_fields.npz
run() {
  tag=$1; shift
  du=$(du -sm /workspace | tail -1 | cut -f1)
  if [ "$du" -ge 40000 ]; then echo "B_$tag skipped: /workspace $du MB $(date -u +%H:%M:%S) UTC" >> $S; return 1; fi
  echo "B_$tag start $(date -u +%H:%M:%S) UTC (/workspace $du MB; CPU 16 threads)" >> $S
  nice -n 10 python code/engine_single_object_B_v1.py "$@" > S4one/log_B_$tag.log 2>&1
  echo "B_$tag exit $? $(date -u +%H:%M:%S) UTC" >> $S
}
run s8_frozen --mode frozen --gpu -1 --record $REC --fields $FLD --fields-module engine_fields_v16 --launch-T 15.0,28.0 --kappas 1,2 --frozen-rs 300 --out-json S4one/B_s8_frozen.json
run s8_history --mode history --gpu -1 --record $REC --fields $FLD --fields-module engine_fields_v16 --launch-T 15.0,28.0 --kappas 1,2 --T-end-rs 40 --cubic-window-rs 8 --out-json S4one/B_s8_history.json
echo "s8 all done $(date -u +%H:%M:%S) UTC" >> $S
