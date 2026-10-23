#!/bin/bash
# Item 7's sea runs, version 9 (SPEC_strand7_page.md sec. 15, the lead's acceptance of version 5): J at j = 7/2 on the first slot that frees,
# the P j = 5/2 rerun (tag P5_j2.5, C-pg-17 against s7p_P_j2.5_r2.json) on the next, R, then C at j = 3/2 and 5/2, all on strand7_sea_page_v6.py
# --evolution eigh: version 5's propagator and analysis with the unresolved pairs classified by X's own singular values as sec. 14.3 defines them
# (version 5 evaluated ||X z_k|| over B_0's eigenvectors, which sits at the threshold at j = 5/2 beta from 30 r_s0; the builder's correction,
# labelled).  H runs on version 4 in version 7's slot (claimed before the drain) and gets the repair pass.  Two slots on GPUs 2 and 1 (one
# process per GPU, checked free at each start), one CPU thread each, each start gated on the lane's busy processes (at most 15); GPU 1's slot
# first waits for version 7's H to start there (that slot claimed H before the drain), so the two launchers never race for GPU 1.  Version 8
# was stopped before any job for that race.  Builder analyst.
D=${QI2_ROOT}/s7_collapse
cd $D || exit 1
O=S7p_run3
PY=${QI2_ROOT}/python
L="--background locked --rs0 32"
CF="--background fields --fields S7p_fields/s7a_self_E16_h1_fields.npz --s7a-json S7p_fields/s7a_self_E16_h1.json"
cat > $O/jobs_v9.txt <<JOBS
J|$L --channels 3.5 --T-end-over-rs 36 --tau-over-rs 0.25 --check-times 0,36
P5_j2.5|$L --channels 2.5 --T-end-over-rs 40 --check-times 0,24,40
R|--background locked --rs0 64 --Rmax-over-rs 40 --channels 0.5 --T-end-over-rs 40 --check-times 0,24,40
C_j1.5|$CF --channels 1.5 --T-end-over-rs 60 --check-times 0,30,60
C_j2.5|$CF --channels 2.5 --T-end-over-rs 60 --check-times 0,30,60
JOBS
echo 0 > $O/next_v9.txt
date -u "+v9 queued %H:%M:%S (J, P5_j2.5, R, C_j1.5, C_j2.5 on strand7_sea_page_v6.py --evolution eigh)" >> $O/status.txt
gpu_free() { [ -z "$(nvidia-smi -i $1 --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; }
slot() {
  local g=$1
  if [ "$g" = "1" ]; then until grep -q "^H start" $O/status.txt; do sleep 20; done; fi
  while true; do
    until [ "$(bash ${QI2_ROOT}/lane_threads_v1.sh)" -le 15 ] && gpu_free $g; do sleep 20; done
    local line
    line=$(flock $O/next_v9.txt bash -c 'k=$(cat '$O'/next_v9.txt); n=$(wc -l < '$O'/jobs_v9.txt); if [ "$k" -ge "$n" ]; then echo END; else sed -n "$((k+1))p" '$O'/jobs_v9.txt; echo $((k+1)) > '$O'/next_v9.txt; fi')
    [ "$line" = "END" ] && break
    local tag=${line%%|*}; local args=${line#*|}
    if ! gpu_free $g; then
      flock $O/next_v9.txt bash -c 'k=$(cat '$O'/next_v9.txt); echo $((k-1)) > '$O'/next_v9.txt'
      sleep 20; continue
    fi
    date -u "+$tag start %H:%M:%S on cuda:$g (v6 eigh)" >> $O/status.txt
    nice -n 10 $PY code/strand7_sea_page_v6.py --device cuda:$g --evolution eigh $args --out $O --tag $tag > $O/$tag.log 2>&1
    echo "$tag exit $? $(date -u +%H:%M:%S)" >> $O/status.txt
    sleep 5
  done
}
slot 2 &
sleep 10
slot 1 &
wait
echo "DONE v9" >> $O/status.txt
