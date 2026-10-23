#!/bin/bash
# Item 7's sea runs, version 6 (SPEC_strand7_page.md sec. 14.6, the lead's edge-mode ruling): the jobs moved from version 5's queue (K2, H, R,
# J, C at j = 3/2 and 5/2) and the P j = 5/2 rerun, all on strand7_sea_page_v5.py (the polar vacuum), after its validation: V5VAL runs F0 and
# S0 on v5 and validate_v5_v1.py, which writes S7p_run3/v5_validated or v5_validation_failed; every other job waits for v5_validated and the
# queue stops on a failure.  Two slots on GPUs 1 and 2 (one process per GPU, checked free at each start: K1 of version 5's queue still runs on
# GPU 1), one CPU thread each, each start gated on the lane's busy processes (at most 15).  Order (sec. 14.6): K2 then H; R | J; the P j = 5/2
# rerun | C at 3/2; C at 5/2.  Builder analyst.
D=${QI2_ROOT}/s7_collapse
cd $D || exit 1
O=S7p_run3
PY=${QI2_ROOT}/python
L="--background locked --rs0 32"
CF="--background fields --fields S7p_fields/s7a_self_E16_h1_fields.npz --s7a-json S7p_fields/s7a_self_E16_h1.json"
cat > $O/jobs_v6.txt <<JOBS
V5VAL|-|-
K2|v5|$L --channels 0.5 --T-end-over-rs 36 --tau-over-rs 0.25 --check-times 0,36
H|v5|--background locked --rs0 32 --h 0.5 --Rmax-over-rs 40 --channels 0.5 --T-end-over-rs 40 --check-times 0,24,40
R|v5|--background locked --rs0 64 --Rmax-over-rs 40 --channels 0.5 --T-end-over-rs 40 --check-times 0,24,40
J|v5|$L --channels 3.5 --T-end-over-rs 36 --tau-over-rs 0.25 --check-times 0,36
P5_j2.5|v5|$L --channels 2.5 --T-end-over-rs 40 --check-times 0,24,40
C_j1.5|v5|$CF --channels 1.5 --T-end-over-rs 60 --check-times 0,30,60
C_j2.5|v5|$CF --channels 2.5 --T-end-over-rs 60 --check-times 0,30,60
JOBS
echo 0 > $O/next_v6.txt
date -u "+v6 queued %H:%M:%S (strand7_sea_page_v5.py after its validation)" >> $O/status.txt
gpu_free() { [ -z "$(nvidia-smi -i $1 --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; }
slot() {
  local g=$1
  while true; do
    local line
    line=$(flock $O/next_v6.txt bash -c 'k=$(cat '$O'/next_v6.txt); n=$(wc -l < '$O'/jobs_v6.txt); if [ "$k" -ge "$n" ]; then echo END; else sed -n "$((k+1))p" '$O'/jobs_v6.txt; echo $((k+1)) > '$O'/next_v6.txt; fi')
    [ "$line" = "END" ] && break
    local tag=${line%%|*}; local rest=${line#*|}; local wt=${rest%%|*}; local args=${rest#*|}
    if [ "$wt" = "v5" ]; then
      until [ -f $O/v5_validated ] || [ -f $O/v5_validation_failed ]; do sleep 30; done
      if [ -f $O/v5_validation_failed ]; then date -u "+$tag not started: v5 validation failed %H:%M:%S" >> $O/status.txt; continue; fi
    fi
    until [ "$(bash ${QI2_ROOT}/lane_threads_v1.sh)" -le 15 ] && gpu_free $g; do sleep 30; done
    if [ "$tag" = "V5VAL" ]; then
      date -u "+V5VAL start %H:%M:%S on cuda:$g" >> $O/status.txt
      nice -n 10 $PY code/strand7_sea_page_v5.py --device cuda:$g --background flat --rs0 32 --channels 0.5 --T-end-over-rs 20 --tau-over-rs 0.5 \
        --out-times 0,2,4,8,12,16,20 --check-times 0,20 --out $O --tag V5_F0 > $O/V5_F0.log 2>&1
      nice -n 10 $PY code/strand7_sea_page_v5.py --device cuda:$g $L --static-from-over-rs 0 --channels 0.5 --T-end-over-rs 20 --tau-over-rs 0.5 \
        --out-times 0,2,4,8,12,16,20 --check-times 0,20 --out $O --tag V5_S0 > $O/V5_S0.log 2>&1
      nice -n 10 $PY code/validate_v5_v1.py --v5-f0 $O/s7p_V5_F0.json --v4-f0 S7p_run1/s7p_F0.json --v5-s0 $O/s7p_V5_S0.json --v4-s0 $O/s7p_S0_L.json \
        --dir $O --out $O/v5_validation.txt > $O/V5VAL.log 2>&1 || touch $O/v5_validation_failed
      echo "V5VAL exit $? $(date -u +%H:%M:%S) ($(tail -1 $O/v5_validation.txt 2>/dev/null))" >> $O/status.txt
      continue
    fi
    date -u "+$tag start %H:%M:%S on cuda:$g (v5)" >> $O/status.txt
    nice -n 10 $PY code/strand7_sea_page_v5.py --device cuda:$g $args --out $O --tag $tag > $O/$tag.log 2>&1
    echo "$tag exit $? $(date -u +%H:%M:%S)" >> $O/status.txt
    sleep 5
  done
}
slot 2 &
sleep 20
slot 1 &
wait
echo "DONE v6" >> $O/status.txt
