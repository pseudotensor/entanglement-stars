#!/bin/bash
# launch_single_object_A_v2_lane2c_v1.sh (builder analyst; inbox #41's revised S4 part A, moved to pod 1's GPU by inbox #42 and onto lane 2c by
# the coordinator's correction): engine_single_object_A_v2.py --gpu 0 on the production's 10 stored slices (r_s/a 64, tau 8, sigma/a 4;
# T/r_s 12 to 40; states/c19_rs64_A_R2_tau8_ell4_avg_r11 on this pod), the dense roots by engine_fields_v14 (the run's own module), the
# exterior-appended Killing-vacuum control on T/r_s 15, kappa 1. Post-processing only: no propagation, no mutual information. Outputs (small:
# the json and the log) in S4one/; skipped when /workspace holds 40000 MB or more. Started by one queue line in queue_lane2c.txt (the runner
# cds to ${QI2_ROOT}/engine).
cd ${QI2_ROOT}/engine || exit 1
mkdir -p S4one
du=$(du -sm /workspace | tail -1 | cut -f1)
if [ "$du" -ge 40000 ]; then echo "A_rs64_sigma4 skipped: /workspace $du MB $(date -u +%H:%M:%S) UTC" >> S4one/status.txt; exit 1; fi
echo "A_rs64_sigma4 start $(date -u +%H:%M:%S) UTC (/workspace $du MB)" >> S4one/status.txt
S=$(for t in 12 15 16 20 24 25 28 30 35 40; do printf "states/c19_rs64_A_R2_tau8_ell4_avg_r11/orbitals_c19_rs64_A_R2_tau8_ell4_avg_r11_T00%s.00.npz," $t; done | sed 's/,$//')
python code/engine_single_object_A_v2.py --family rs64_sigma4 --gpu 0 --run C3prod19/c19_rs64_A_R2_tau8_ell4_avg_r11 --fields-module engine_fields_v14 \
    --slices "$S" --vacuum-control 15,1 --out-json S4one/A_rs64_sigma4.json > S4one/A_rs64_sigma4.log 2>&1
echo "A_rs64_sigma4 exit $? $(date -u +%H:%M:%S) UTC" >> S4one/status.txt
