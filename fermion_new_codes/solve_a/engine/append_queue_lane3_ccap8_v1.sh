#!/bin/bash
# C-capture version 8 on pod 1 lane 3 (the version 7 run at slice 283 stopped at 14:28:15 UTC on a one-point branch): the production's
# last maximum T/rs 35.375 (slice 283) first, then the gate, version 8 at the last minimum (slice 302) against version 7's md there
# (every line but the title and the run's length expected identical when no channel is excluded). Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
grep -q "^ccap8_prod19_max	" queue_lane3.txt && { echo "already queued"; exit 0; }
F="--fields C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11_fields.npz --records C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json --fields-key applied"
printf '# %s: C-capture version 8 (version 7 with the one-point-branch guard): slice 283, then the gate at slice 302 against version 7\n' "$(date -u +%H:%M:%S) UTC" >> queue_lane3.txt
printf 'ccap8_prod19_max\tmkdir -p CAP && python code/engine_capture_v8.py %s --slice 283 --control --absorber --workers 8 --out-json CAP/capture_v8_prod19_max.json --md CAP/capture_v8_prod19_max.md\n' "$F" >> queue_lane3.txt
printf 'ccap8_gate_prod19_min\tmkdir -p CAP && python code/engine_capture_v8.py %s --slice 302 --control --absorber --workers 8 --out-json CAP/capture_v8_gate_prod19_min.json --md CAP/capture_v8_gate_prod19_min.md\n' "$F" >> queue_lane3.txt
echo "# ccap7_prod19_max stopped at 14:28:15 UTC (a one-point branch at launch: flow_chart_v4.build_packet's np.gradient); ccap8_prod19_max and ccap8_gate_prod19_min queued on lane 3 (engine_capture_v8) $(date -u +%H:%M:%S) UTC" >> h100b_status.txt
tail -2 queue_lane3.txt | cut -c1-140
