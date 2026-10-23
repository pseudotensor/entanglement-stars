#!/bin/bash
# C-capture version 11's gate (b) on pod 1 lane 3 (the lane's ruling of about 23:02 UTC on engine spec 38.5 (3): "(b) a 40 r_s run of the
# minimum agreeing with version 9's readings within 1e-3 of sigma at every reading"; gate (a), the outgoing-packet return below 1e-3, by
# test_capture_v11_absorber_v1 on the box; gate (c), the cost per step against version 10's, read from this run's and ccap10_gate's step
# lines): the last minimum (slice 302), version 9's ccap9_prod19_min line (--control --absorber --workers 8, to 40 r_s) with
# engine_capture_v11.py and the absorber the box test chose (--append-out 512 --gamma-out 0.05 --power-out 3); behind ccap10_gate_prod19_min
# on the same lane; read afterwards by compare_capture_v11_gate_v1 against CAP/capture_v9_prod19_min.json. The held tag
# ccap9t300_prod19_max stays skipped. Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
grep -q "^ccap11_gate_prod19_min	" queue_lane3.txt && { echo "already queued"; exit 0; }
F="--fields C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11_fields.npz --records C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json --fields-key applied"
printf '# %s: C-capture version 11 gate (b) (the lane ruling on engine spec 38.5 (3)): slice 302 to 40 r_s, the appended absorber 512:0.05:3, against version 9 (ccap9_prod19_min)\n' "$(date -u +%H:%M:%S) UTC" >> queue_lane3.txt
printf 'ccap11_gate_prod19_min\tmkdir -p CAP && python code/engine_capture_v11.py %s --slice 302 --control --absorber --workers 8 --append-out 512 --gamma-out 0.05 --power-out 3 --out-json CAP/capture_v11_gate_prod19_min.json --md CAP/capture_v11_gate_prod19_min.md\n' "$F" >> queue_lane3.txt
echo "# ccap11_gate_prod19_min queued on lane 3 behind ccap10_gate_prod19_min (engine_capture_v11 gate (b)) $(date -u +%H:%M:%S) UTC" >> h100b_status.txt
tail -2 queue_lane3.txt | cut -c1-160
