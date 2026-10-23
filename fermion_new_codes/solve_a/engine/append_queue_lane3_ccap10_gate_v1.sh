#!/bin/bash
# C-capture version 10's gate run on pod 1 lane 3 (engine spec 38.5 (3): "the gate a 40 r_s run of the minimum agreeing with version 9's
# readings within 1e-3 of sigma at every reading, the absorbed weight printed (expected below the 0.01 threshold before 40)"): the last
# minimum (slice 302), version 9's line (ccap9_prod19_min: --control --absorber --workers 8, the reading times to 40 r_s) with
# engine_capture_v10.py (the outer port at its defaults: the inner port's construction on the last 16 sites); read afterwards by
# compare_capture_v10_gate_v1 against CAP/capture_v9_prod19_min.json. The held tag ccap9t300_prod19_max stays skipped. Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
grep -q "^ccap10_gate_prod19_min	" queue_lane3.txt && { echo "already queued"; exit 0; }
F="--fields C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11_fields.npz --records C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json --fields-key applied"
printf '# %s: C-capture version 10 gate (engine spec 38.5 (3)): slice 302 to 40 r_s, the outer port on, against version 9 (ccap9_prod19_min)\n' "$(date -u +%H:%M:%S) UTC" >> queue_lane3.txt
printf 'ccap10_gate_prod19_min\tmkdir -p CAP && python code/engine_capture_v10.py %s --slice 302 --control --absorber --workers 8 --out-json CAP/capture_v10_gate_prod19_min.json --md CAP/capture_v10_gate_prod19_min.md\n' "$F" >> queue_lane3.txt
echo "# ccap10_gate_prod19_min queued on lane 3 (engine_capture_v10 gate, engine spec 38.5 (3)) $(date -u +%H:%M:%S) UTC" >> h100b_status.txt
tail -2 queue_lane3.txt | cut -c1-140
