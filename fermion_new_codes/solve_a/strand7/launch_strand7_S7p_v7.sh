#!/bin/bash
# Item 7's sea runs, version 7 (SPEC_strand7_page.md sec. 14.6 (1), the lead's fallback): version 5's validation did not pass as written
# (S7p_run3/v5_validation.txt), so K2, H and R (j = 1/2) run on strand7_sea_page_v4.py and get the repair pass afterwards; J, the P j = 5/2
# rerun and C at j = 3/2 and 5/2 (version 5 only) wait for the lead's ruling on the validation numbers.  Two slots on GPUs 2 and 1 (one
# process per GPU, checked free at each start; K1 still runs on GPU 1), one CPU thread each, each start gated on the lane's busy processes
# (at most 15).  Builder analyst.
D=${QI2_ROOT}/s7_collapse
cd $D || exit 1
O=S7p_run3
PY=${QI2_ROOT}/python
L="--background locked --rs0 32"
cat > $O/jobs_v7.txt <<JOBS
K2|$L --channels 0.5 --T-end-over-rs 36 --tau-over-rs 0.25 --check-times 0,36
H|--background locked --rs0 32 --h 0.5 --Rmax-over-rs 40 --channels 0.5 --T-end-over-rs 40 --check-times 0,24,40
R|--background locked --rs0 64 --Rmax-over-rs 40 --channels 0.5 --T-end-over-rs 40 --check-times 0,24,40
JOBS
echo 0 > $O/next_v7.txt
date -u "+v7 queued %H:%M:%S (K2, H, R on strand7_sea_page_v4.py: the v5 validation did not pass as written)" >> $O/status.txt
gpu_free() { [ -z "$(nvidia-smi -i $1 --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; }
slot() {
  local g=$1
  while true; do
    local line
    line=$(flock $O/next_v7.txt bash -c 'k=$(cat '$O'/next_v7.txt); n=$(wc -l < '$O'/jobs_v7.txt); if [ "$k" -ge "$n" ]; then echo END; else sed -n "$((k+1))p" '$O'/jobs_v7.txt; echo $((k+1)) > '$O'/next_v7.txt; fi')
    [ "$line" = "END" ] && break
    local tag=${line%%|*}; local args=${line#*|}
    until [ "$(bash ${QI2_ROOT}/lane_threads_v1.sh)" -le 15 ] && gpu_free $g; do sleep 30; done
    date -u "+$tag start %H:%M:%S on cuda:$g (v4)" >> $O/status.txt
    nice -n 10 $PY code/strand7_sea_page_v4.py --device cuda:$g $args --out $O --tag $tag > $O/$tag.log 2>&1
    echo "$tag exit $? $(date -u +%H:%M:%S)" >> $O/status.txt
    sleep 5
  done
}
slot 2 &
sleep 20
slot 1 &
wait
echo "DONE v7" >> $O/status.txt
