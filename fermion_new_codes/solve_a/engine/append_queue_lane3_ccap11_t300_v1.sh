#!/bin/bash
# C-capture version 11 to 300 r_s or until settled at the last minimum (the lane's ruling on engine spec 38.5 (3): "Then the minimum to 300
# r_s on v11 on pod 1 lane 3 (8 workers), the maximum only if the minimum settles"), appended only after gates (a), (b) and (c) are printed
# and pass: version 11's gate line (ccap11_gate_prod19_min) with --T-max-over-rs 300; the absorber 512:0.05:3 as gated. Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
grep -q "^ccap11t300_prod19_min	" queue_lane3.txt && { echo "already queued"; exit 0; }
F="--fields C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11_fields.npz --records C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json --fields-key applied"
printf '# %s: C-capture version 11 to 300 r_s or until settled (the lane ruling on engine spec 38.5 (3); gates (a) to (c) passed): slice 302\n' "$(date -u +%H:%M:%S) UTC" >> queue_lane3.txt
printf 'ccap11t300_prod19_min\tmkdir -p CAP && python code/engine_capture_v11.py %s --slice 302 --control --absorber --workers 8 --T-max-over-rs 300 --append-out 512 --gamma-out 0.05 --power-out 3 --out-json CAP/capture_v11_t300_prod19_min.json --md CAP/capture_v11_t300_prod19_min.md\n' "$F" >> queue_lane3.txt
echo "# ccap11t300_prod19_min queued on lane 3 (engine_capture_v11 to 300 r_s, gates (a) to (c) passed) $(date -u +%H:%M:%S) UTC" >> h100b_status.txt
tail -2 queue_lane3.txt | cut -c1-160
