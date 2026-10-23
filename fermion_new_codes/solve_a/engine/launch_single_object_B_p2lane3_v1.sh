#!/bin/bash
# launch_single_object_B_p2lane3_v1.sh (builder analyst; inbox #41's revised S4 part B, the coordinator's go after page_*_v6b's self-test passed;
# the second H100's runner p2lane3): engine_single_object_B_v1.py with page_*_v6b's machinery (${QI2_ROOT}/code_v6, md5-equal to the
# worktree's code/page), one run after the other on the GPU: (1) D's 256-site dense oracle (3 launch columns, every readout against the full
# propagator); (2) B.1 (i): the production's records T/r_s 15.0 and 30.0 frozen, to 300 r_s, kappa 1 and 2, the 12 launch columns per record and
# channel, the exterior shells every 25 r_s; (3) B.1 (ii): the solved history from those clocks to T/r_s 40, the piecewise-constant against cubic
# control on the first 8 r_s after T/r_s 15.0. The massive channel (ma 0.2) only; no massless channel (D14); no mutual information (D15). The
# chain-end reflection clock is printed per column (and the Page builder's 208 r_s). Outputs in S4one/; a run is skipped when /workspace holds
# 40000 MB or more. Started by one queue line in queue_p2c.txt (the runner cds to ${QI2_ROOT}/engine).
cd ${QI2_ROOT}/engine || exit 1
mkdir -p S4one
S=S4one/status_B.txt
REC=C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json
FLD=C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11_fields.npz
run() {
  tag=$1; shift
  du=$(du -sm /workspace | tail -1 | cut -f1)
  if [ "$du" -ge 40000 ]; then echo "B_$tag skipped: /workspace $du MB $(date -u +%H:%M:%S) UTC" >> $S; return 1; fi
  echo "B_$tag start $(date -u +%H:%M:%S) UTC (/workspace $du MB)" >> $S
  python code/engine_single_object_B_v1.py "$@" > S4one/log_B_$tag.log 2>&1
  echo "B_$tag exit $? $(date -u +%H:%M:%S) UTC" >> $S
}
run oracle --mode oracle --gpu 0 --out-json S4one/B_oracle.json
run frozen --mode frozen --gpu 0 --record $REC --fields $FLD --launch-T 15.0,30.0 --kappas 1,2 --frozen-rs 300 --out-json S4one/B_frozen.json
run history --mode history --gpu 0 --record $REC --fields $FLD --launch-T 15.0,30.0 --kappas 1,2 --T-end-rs 40 --cubic-window-rs 8 --out-json S4one/B_history.json
echo "all done $(date -u +%H:%M:%S) UTC" >> $S
