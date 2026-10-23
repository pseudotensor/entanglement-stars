#!/bin/bash
# C-capture version 9 on pod 1 lane 3 (engine spec 34.4 (2): every probe's stop in version 7 at the minimum and version 8 at the maximum
# had 0.64 to 0.81 of sigma in transit, so each is rerun once with the settle rule gated on the in-transit share below 10 percent; the
# lead's "engine_capture_v8" for this file takes the name version 9): the last minimum (slice 302), then the last maximum (slice 283),
# behind the version 8 gate now on the lane; --workers 8 as before. Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
grep -q "^ccap9_prod19_min	" queue_lane3.txt && { echo "already queued"; exit 0; }
F="--fields C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11_fields.npz --records C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json --fields-key applied"
printf '# %s: C-capture version 9 (the settle rule gated on the in-transit share, engine spec 34.4 (2)): slice 302, then 283\n' "$(date -u +%H:%M:%S) UTC" >> queue_lane3.txt
printf 'ccap9_prod19_min\tmkdir -p CAP && python code/engine_capture_v9.py %s --slice 302 --control --absorber --workers 8 --out-json CAP/capture_v9_prod19_min.json --md CAP/capture_v9_prod19_min.md\n' "$F" >> queue_lane3.txt
printf 'ccap9_prod19_max\tmkdir -p CAP && python code/engine_capture_v9.py %s --slice 283 --control --absorber --workers 8 --out-json CAP/capture_v9_prod19_max.json --md CAP/capture_v9_prod19_max.md\n' "$F" >> queue_lane3.txt
echo "# ccap9_prod19_min and ccap9_prod19_max queued on lane 3 behind the version 8 gate (engine_capture_v9, the in-transit gate of 34.4 (2)) $(date -u +%H:%M:%S) UTC" >> h100b_status.txt
tail -2 queue_lane3.txt | cut -c1-120
