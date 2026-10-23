#!/bin/bash
# C-capture version 9 to 300 r_s or until settled (engine spec 35.2 (3): one further run per phase, no new engine version, --T-max-over-rs
# 300, everything else as the version 9 lines, --workers 8; the minimum first, then the maximum, on lane 3 one after the other: pod 1's
# cgroup allows 22.1 CPUs and its load already stands near 44, so two runs of 8 workers in parallel would slow the Page pilot); the first
# 20 reading times are gated against the version 9 runs afterwards (compare_capture_v9_inrun_gate_v3). Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
grep -q "^ccap9t300_prod19_min	" queue_lane3.txt && { echo "already queued"; exit 0; }
F="--fields C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11_fields.npz --records C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json --fields-key applied"
printf '# %s: C-capture version 9 to 300 r_s or until settled (engine spec 35.2 (3)): slice 302, then 283\n' "$(date -u +%H:%M:%S) UTC" >> queue_lane3.txt
printf 'ccap9t300_prod19_min\tmkdir -p CAP && python code/engine_capture_v9.py %s --slice 302 --control --absorber --workers 8 --T-max-over-rs 300 --out-json CAP/capture_v9_t300_prod19_min.json --md CAP/capture_v9_t300_prod19_min.md\n' "$F" >> queue_lane3.txt
printf 'ccap9t300_prod19_max\tmkdir -p CAP && python code/engine_capture_v9.py %s --slice 283 --control --absorber --workers 8 --T-max-over-rs 300 --out-json CAP/capture_v9_t300_prod19_max.json --md CAP/capture_v9_t300_prod19_max.md\n' "$F" >> queue_lane3.txt
echo "# ccap9t300_prod19_min and ccap9t300_prod19_max queued on lane 3 (engine_capture_v9 to 300 r_s or until settled, engine spec 35.2 (3)) $(date -u +%H:%M:%S) UTC" >> h100b_status.txt
tail -2 queue_lane3.txt | cut -c1-100
