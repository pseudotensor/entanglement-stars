#!/bin/bash
# C-capture version 11 to 300 r_s or until settled at the last MAXIMUM (slice 283; only if the minimum settles) (the lane's ruling on engine spec 38.5 (3): "Then the minimum to 300
# r_s on v11 on pod 1 lane 3 (8 workers), the maximum only if the minimum settles"), appended only after gates (a), (b) and (c) are printed
# and pass: version 11's gate line (ccap11_gate_prod19_min) with --T-max-over-rs 300; the absorber 512:0.05:3 as gated. Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
grep -q "^ccap11t300_prod19_max	" queue_lane3.txt && { echo "already queued"; exit 0; }
F="--fields C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11_fields.npz --records C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json --fields-key applied"
printf '# %s: C-capture version 11 to 300 r_s or until settled (the lane ruling on engine spec 38.5 (3); the minimum settled): slice 283\n' "$(date -u +%H:%M:%S) UTC" >> queue_lane3.txt
printf 'ccap11t300_prod19_max\tmkdir -p CAP && python code/engine_capture_v11.py %s --slice 283 --control --absorber --workers 8 --T-max-over-rs 300 --append-out 512 --gamma-out 0.05 --power-out 3 --out-json CAP/capture_v11_t300_prod19_max.json --md CAP/capture_v11_t300_prod19_max.md\n' "$F" >> queue_lane3.txt
echo "# ccap11t300_prod19_max queued on lane 3 (engine_capture_v11 to 300 r_s at the maximum, the minimum settled) $(date -u +%H:%M:%S) UTC" >> h100b_status.txt
tail -2 queue_lane3.txt | cut -c1-160
