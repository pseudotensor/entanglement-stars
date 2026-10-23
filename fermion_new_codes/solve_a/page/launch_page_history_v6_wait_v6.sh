#!/bin/bash
# launch_page_history_v6_wait_v6.sh (builder analyst, 2026-10-17): wait_v5 with its first wait keyed on the last lane 5 queue line whose tag
# starts with WAIT_PREFIX (default page_v6f_A_s4k3), read again at every poll, so a re-queued s4k3 (a later tag with another device gate)
# is waited for as well: the wait ends when the status file holds "lane5 <that tag> exit".
# wait_v5's text: launch_page_history_v6_wait_v5.sh (builder analyst, 2026-10-17): the wait for A to T/r_s 100 (s4to100, code_v6/launch_page_history_v6_v7.sh
# on page_history_v6g.py) on pod 1 lane 1. It waits for (1) the status line WAIT_LINE in ${QI2_ROOT}/h100b_status.txt (default the
# exit of lane 5's page_v6f_A_s4k3b, whose gate needs 60 GB of the device beside nothing: A's measured peak, s4tr's 38.88 GiB reserved,
# does not fit beside it), (2) B's continuation on pod 1 (its json and fields npz in ${QI2_ROOT}/C3ext19, copied by the box's
# sync_ext_to_pod1_v1.sh), (3) NEED_GPU_GB free on the device (default 42) on three polls 20 s apart; then it runs the launcher v7.
# Every wait ends at WAIT_H hours (default 10) with exit 7 (the status line or B's files) or 8 (the memory), nothing launched.
# A's end is set by B's (B's own projection 17:47 to 18:33 UTC; A takes about 25 s per record, B about 140 s), so the start after lane 5
# moves nothing. Usage: bash code_v6/launch_page_history_v6_wait_v6.sh s4to100
set -euo pipefail
cd ${QI2_ROOT}/page
name=${1:?usage: launch_page_history_v6_wait_v6.sh s4to100}
need=${NEED_GPU_GB:-42}
wait_h=${WAIT_H:-10}
wpre=${WAIT_PREFIX:-page_v6f_A_s4k3}
EXTD=${QI2_ROOT}/C3ext19
EXTT=c19_rs64_A_R2_tau8_ell4_avg_ext100
now() { date -u +%H:%M:%S; }
deadline=$(( $(date +%s) + wait_h * 3600 ))
last_tag() { grep -aoE "^${wpre}[A-Za-z0-9_]*" ${QI2_ROOT}/queue_lane5.txt | tail -1; }
wtag=$(last_tag)
[ -n "$wtag" ] || { echo "$(now) UTC: no lane 5 queue line starts with $wpre: no launch"; exit 7; }
echo "$(now) UTC: $name waits for the exit of lane 5's last $wpre* line ($wtag now; read again at every poll)"
while :; do
  wtag=$(last_tag)
  grep -aqF "lane5 $wtag exit" ${QI2_ROOT}/h100b_status.txt && break
  if [ "$(date +%s)" -gt "$deadline" ]; then echo "$(now) UTC: 'lane5 $wtag exit' did not appear within $wait_h h: no launch"; exit 7; fi
  sleep 60
done
echo "$(now) UTC: $(grep -aF "lane5 $wtag exit" ${QI2_ROOT}/h100b_status.txt | tail -1)"
until [ -f "$EXTD/collapse_$EXTT.json" ] && [ -f "$EXTD/collapse_${EXTT}_fields.npz" ]; do
  if [ "$(date +%s)" -gt "$deadline" ]; then echo "$(now) UTC: B's continuation did not reach pod 1 ($EXTD) within $wait_h h: no launch"; exit 7; fi
  sleep 60
done
echo "$(now) UTC: B's continuation is on pod 1 ($EXTD)"
ok=0; free=0
while :; do
  line=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  tot=${line%,*}; used=${line#*,}
  free=$(( (tot - used) / 1024 ))
  if [ "$free" -ge "$need" ]; then ok=$((ok + 1)); else ok=0; fi
  [ "$ok" -ge 3 ] && break
  if [ "$(date +%s)" -gt "$deadline" ]; then echo "$(now) UTC: the device's free memory stayed below $need GB (last $free GB): no launch"; exit 8; fi
  sleep 20
done
echo "$(now) UTC: the device has $free GB free (need $need GB on three polls 20 s apart): launching $name"
exec bash code_v6/launch_page_history_v6_v7.sh "$name"
