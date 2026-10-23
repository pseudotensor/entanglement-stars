#!/bin/bash
# Pod 1 lane 3: appends the two C-capture version 7 runs (builder analyst, 2026-10-16; engine spec 32.3 (4), 32.6 item 8; brief
# lead/BRIEF_capture_v7_builder.md): the production's last minimum (slice 302, T/rs 37.75) first, then the last maximum (slice 283,
# T/rs 35.375), version 6's other arguments (--fields-key applied --control --absorber) and --workers 8. One comment line first (time from
# date -u). Idempotent: does nothing when a ccap7_prod19_min line is already in the queue. Prints the queue's tail with tabs shown.
cd ${QI2_ROOT}/engine || exit 1
Q=queue_lane3.txt
F=C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11_fields.npz
J=C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json
if grep -q "^ccap7_prod19_min"$'\t' "$Q"; then echo "already queued"; tail -3 "$Q" | cat -A | cut -c1-120; exit 0; fi
printf '# %s UTC: C-capture version 7 (engine spec 32.3 (4), 32.6 item 8): reading times every 2 r_s to 40 r_s after launch or until settled; the last minimum, then the last maximum; --workers 8 (the serial path to the bit on the unitary readings: CAP7TEST tests 2 and 3); builder analyst\n' "$(date -u +%H:%M:%S)" >> "$Q"
printf 'ccap7_prod19_min\tmkdir -p CAP && python code/engine_capture_v7.py --fields %s --records %s --fields-key applied --slice 302 --control --absorber --workers 8 --out-json CAP/capture_v7_prod19_min.json --md CAP/capture_v7_prod19_min.md\n' "$F" "$J" >> "$Q"
printf 'ccap7_prod19_max\tmkdir -p CAP && python code/engine_capture_v7.py --fields %s --records %s --fields-key applied --slice 283 --control --absorber --workers 8 --out-json CAP/capture_v7_prod19_max.json --md CAP/capture_v7_prod19_max.md\n' "$F" "$J" >> "$Q"
tail -3 "$Q" | cat -A
md5sum code/engine_capture_v7.py
