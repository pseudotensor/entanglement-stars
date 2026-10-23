#!/bin/bash
# The owner's ruling D7 (REVIEWA_REVIEW_web40e.md section 4, approved about 04:30 UTC 2026-10-16): the star-family ladders stop past the
# predicted turning points (S3: (0.2, 1/4) past N_B 200, (0.2, 1/16) past 1300, (0.1, 1/16) past 9500, (0.05, 1/4) past 9300). Stops
# (1) the running (0.2, 1/16) extension (engine_star_v6 --m 0.2 --g 0.0625 ... --tag m0.2_g0.0625_ell8_ext, bisecting near N_B 2150, its
# members kept in Sfam6/star_m0.2_g0.0625_ell8_ext.json) and (2) the armed (0.05, 1/4) ladder beyond N_B,max (the waiter
# launch_star_ladder7_v1.sh on the (0.1, 1/64) member's PID; no member started). Only the exact PIDs with the exact arguments are signalled.
# Run on the box. Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
EXT=$1; LAD=$2
a1=$(ps -o args= -p $EXT 2>/dev/null); a2=$(ps -o args= -p $LAD 2>/dev/null)
case "$a1" in *"code/engine_star_v6.py --m 0.2 --g 0.0625 "*"--tag m0.2_g0.0625_ell8_ext"*) ;; *) echo "PID $EXT is not the (0.2, 1/16) extension: '$a1'; nothing done"; exit 1;; esac
case "$a2" in "bash code/launch_star_ladder7_v1.sh "*) ;; *) echo "PID $LAD is not the ladder7 waiter: '$a2'; nothing done"; exit 1;; esac
kill -TERM $EXT && echo "star_v6 m0.2_g0.0625_ell8_ext stopped (owner ruling D7: no ladder past the predicted turning point) $(date -u +%H:%M:%S) UTC" >> status.txt
kill -TERM $LAD && echo "GPU2 star ladder7 disarmed before its start (owner ruling D7) $(date -u +%H:%M:%S) UTC" >> status.txt
sleep 5
for p in $EXT $LAD; do ps -o pid= -p $p >/dev/null 2>&1 && echo "PID $p still alive" || echo "PID $p ended"; done
