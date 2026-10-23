#!/bin/bash
# launch_replay_v9_lane5_v1.sh (builder analyst; engine spec 37.5 (1) and (2): "Device: pod 1 lane 5 for the production and the r_s 32 set,
# then the ladder members"; the brief lead/BRIEF_s7_copyslot_replay9_builder.md): engine_replay_v9 with the per-record hooks, one run after
# the other on the lane's GPU: (1) the production: its v17 parent's first 89 records (T/rs 0 to 11, the v17 path, replay v2's masks and
# stresses of REPLAY2/replay2_c17_rs64_h100 compared), then the v19 child resumed at T/rs 11 (version 8's path; its native records 11.125 to
# 40); (2) the r_s 32 set: the set17 tau 8, tau 4, tau 16 and reading-B members (the v17 path, replay v2's outputs compared); (3) the ladder
# members c27 ell 4 and ell 8 (version 8's path). Outputs in CERT3v9/; each run's start and exit in CERT3v9/status.txt; a run is skipped
# when /workspace holds 40000 MB or more (the brief's stop). Started by one queue line in queue_lane5.txt (the lane runner cds to
# ${QI2_ROOT}/engine).
cd ${QI2_ROOT}/engine || exit 1
mkdir -p CERT3v9
S=CERT3v9/status.txt
run() {
  tag=$1; shift
  du=$(du -sm /workspace | tail -1 | cut -f1)
  if [ "$du" -ge 40000 ]; then echo "$tag skipped: /workspace $du MB (stop at 40000) $(date -u +%H:%M:%S) UTC" >> $S; return 1; fi
  echo "$tag start $(date -u +%H:%M:%S) UTC (/workspace $du MB)" >> $S
  python code/engine_replay_v9.py "$@" > CERT3v9/log_replay9_$tag.log 2>&1
  echo "$tag exit $? $(date -u +%H:%M:%S) UTC" >> $S
}
run c17_rs64_to11 --run C3prod/c17_rs64_A_R2_tau8_ell4_avg --records 89 --gpu 0 --compare-prefix REPLAY2/replay2_c17_rs64_h100 \
    --md CERT3v9/replay9_c17_rs64_to11.md --out-json CERT3v9/replay9_c17_rs64_to11.json --out-npz CERT3v9/replay9_c17_rs64_to11
run c19_rs64_r11 --run C3prod19/c19_rs64_A_R2_tau8_ell4_avg_r11 --gpu 0 \
    --md CERT3v9/replay9_c19_rs64_r11.md --out-json CERT3v9/replay9_c19_rs64_r11.json --out-npz CERT3v9/replay9_c19_rs64_r11
for t in A_R2_tau8_ell4_avg A_R2_tau4_ell4_avg A_R2_tau16_ell4_avg B_R2_tau8_ell4_R0x4_avg; do
  run set17_rs32_$t --run C3set/set17_rs32_$t --gpu 0 --compare-prefix REPLAY2/replay2_set17_rs32_$t \
      --md CERT3v9/replay9_set17_rs32_$t.md --out-json CERT3v9/replay9_set17_rs32_$t.json --out-npz CERT3v9/replay9_set17_rs32_$t
done
for t in ell4 ell8; do
  run c27_rs64_$t --run C3lad27/c27_rs64_A_R2_tau8_${t}_prep4_avg --gpu 0 \
      --md CERT3v9/replay9_c27_rs64_${t}_prep4.md --out-json CERT3v9/replay9_c27_rs64_${t}_prep4.json --out-npz CERT3v9/replay9_c27_rs64_${t}_prep4
done
echo "all done $(date -u +%H:%M:%S) UTC" >> $S
