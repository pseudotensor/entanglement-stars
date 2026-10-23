#!/bin/bash
# Item 7's sea runs, version 4 (SPEC_strand7_page.md secs. 12.4 and 12.6, the lead's re-ruling: the primary L, the prescribed locked collapse;
# the comparison C, the rows' own collapse (S7p_fields self E16); the controls): strand7_sea_page_v3.py (the stable sectors) on GPUs 1 and 2 through a two-slot queue
# (one process per GPU, checked free at each start; GPU 0 carries another lane's job), one CPU thread each, each start gated on the lane's
# busy processes (at most 15).  Order (sec. 12.6): S0 on L; P on L at j = 1/2 (with C-pg-11 at 24 r_s0), 3/2, 5/2; C at j = 1/2 with K1;
# K2, H, R, J; C at j = 3/2 and 5/2 last.  F0 ran under version 3 (23:30 UTC; background-free; kept).  The C runs wait for the self E16 page
# background (S7p_fields).  Then the data and figure scripts (v2; the draft figure).  Builder analyst.
D=${QI2_ROOT}/s7_collapse
cd $D || exit 1
O=S7p_run2
mkdir -p $O
date -u '+queued v4 %Y-%m-%d %H:%M:%S UTC' > $O/status.txt
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=${QI2_ROOT}/python
CF="--background fields --fields S7p_fields/s7a_self_E16_h1_fields.npz --s7a-json S7p_fields/s7a_self_E16_h1.json"
L="--background locked --rs0 32"
cat > $O/jobs.txt <<JOBS
S0_L|-|$L --static-from-over-rs 0 --channels 0.5 --T-end-over-rs 20 --tau-over-rs 0.5 --out-times 0,2,4,8,12,16,20 --check-times 0,20
P_j0.5|-|$L --channels 0.5 --connection-control 1 --connection-T0 24 --T-end-over-rs 40 --check-times 0,24,40
P_j1.5|-|$L --channels 1.5 --T-end-over-rs 40 --check-times 0,24,40
P_j2.5|-|$L --channels 2.5 --T-end-over-rs 40 --check-times 0,24,40
C_j0.5|self_E16_h1 exit|$CF --channels 0.5 --T-end-over-rs 60 --check-times 0,30,60
K1|-|$L --channels 0.5 --T-end-over-rs 36 --tau-over-rs 0.0625 --check-times 0,36
K2|-|$L --channels 0.5 --T-end-over-rs 36 --tau-over-rs 0.25 --check-times 0,36
H|-|--background locked --rs0 32 --h 0.5 --Rmax-over-rs 40 --channels 0.5 --T-end-over-rs 40 --check-times 0,24,40
R|-|--background locked --rs0 64 --Rmax-over-rs 40 --channels 0.5 --T-end-over-rs 40 --check-times 0,24,40
J|-|$L --channels 3.5 --T-end-over-rs 36 --tau-over-rs 0.25 --check-times 0,36
C_j1.5|self_E16_h1 exit|$CF --channels 1.5 --T-end-over-rs 60 --check-times 0,30,60
C_j2.5|self_E16_h1 exit|$CF --channels 2.5 --T-end-over-rs 60 --check-times 0,30,60
JOBS
echo 0 > $O/next.txt
gpu_free() {  # utilization below 10 percent and under 2 GB on GPU $1
  read u m <<< "$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -i $1 | tr -d ',')"
  [ "$u" -lt 10 ] && [ "$m" -lt 2000 ]
}
slot() {  # the worker of GPU $1: takes the next job from the list (flock on next.txt) until none is left
  local g=$1
  while true; do
    local line
    line=$(flock $O/next.txt bash -c 'k=$(cat '$O'/next.txt); n=$(wc -l < '$O'/jobs.txt); if [ "$k" -ge "$n" ]; then echo END; else sed -n "$((k+1))p" '$O'/jobs.txt; echo $((k+1)) > '$O'/next.txt; fi')
    [ "$line" = "END" ] && break
    local tag=${line%%|*}; local rest=${line#*|}; local wt=${rest%%|*}; local args=${rest#*|}
    if [ "$wt" != "-" ]; then until grep -q "$wt" S7p_fields/status.txt; do sleep 30; done; fi
    until [ "$(bash ${QI2_ROOT}/lane_threads_v1.sh)" -le 15 ] && gpu_free $g; do sleep 30; done
    date -u "+$tag start %H:%M:%S on cuda:$g" >> $O/status.txt
    nice -n 10 $PY code/strand7_sea_page_v3.py --device cuda:$g $args --out $O --tag $tag > $O/$tag.log 2>&1
    echo "$tag exit $? $(date -u +%H:%M:%S)" >> $O/status.txt
    sleep 5
  done
}
slot 1 &
sleep 20
slot 2 &
wait
P="$O/s7p_P_j0.5.json,$O/s7p_P_j1.5.json,$O/s7p_P_j2.5.json"
C=$(ls $O/s7p_C_j*.json 2>/dev/null | paste -sd, -)
nice -n 10 $PY code/strand7_page_data_v2.py --records $P ${C:+--records-c $C} --out $O/page_P.npz > $O/data.log 2>&1
nice -n 10 $PY code/strand7_page_fig_v2.py --data $O/page_P.npz --out $O/page_P_fig > $O/fig.log 2>&1
echo "data and figure exit $? $(date -u +%H:%M:%S)" >> $O/status.txt
echo DONE >> $O/status.txt
