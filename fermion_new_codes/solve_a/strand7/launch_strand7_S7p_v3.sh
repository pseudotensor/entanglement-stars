#!/bin/bash
# Item 7's sea runs, version 3 (SPEC_strand7_page.md secs. 2, 6, 10): strand7_sea_page_v1.py on GPUs 1 and 2 (one process per GPU, checked free at each
# launch; GPU 0 carries another lane's job), one CPU thread each.  Order: the code's controls F0 (flat) and S0 (the held-lock fields frozen at
# T = 0) when J5conv5's threads are free; then the primary P on the held lock at E_0 = 16, h = 1 (S7p_fields), the channels 1/2 (with the
# connection control C-pg-11 at 24 r_s0) and 3/2 in parallel, then 5/2; then the data and figure scripts (a draft figure).  Builder analyst.
# Version 2: the h = 1 held-lock background (S7p_fields, strand7_shell_v3.py --stop-rule page) ended at T = 30.05 r_s0 on a non-positive lapse at a
# marker (the throat crossed at 28.64 r_s0), so P runs to T_end = 30 r_s0, the background's end, instead of 60 (beyond its end the sea code would
# hold the last fields static); the lead re-rules the background.
# Version 3: each run starts when the lane runs at most 15 busy processes (lane_threads_v1.sh) and a GPU is free, instead of after J5conv5.
D=${QI2_ROOT}/s7_collapse
cd $D || exit 1
mkdir -p S7p_run1
date -u '+queued v3 %Y-%m-%d %H:%M:%S UTC' >> S7p_run1/status.txt
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=${QI2_ROOT}/python
free_gpu() {  # prints a free GPU among 1 and 2 (utilization below 10 percent, under 2 GB), waiting until one is
  while true; do
    for g in 1 2; do
      read u m <<< "$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -i $g | tr -d ',')"
      if [ "$u" -lt 10 ] && [ "$m" -lt 2000 ] && ! grep -q "busy $g" S7p_run1/gpus.txt 2>/dev/null; then echo $g; return; fi
    done
    sleep 30
  done
}
wait_thread() {  # until the lane runs at most 15 busy processes
  until [ "$(bash ${QI2_ROOT}/lane_threads_v1.sh)" -le 15 ]; do sleep 30; done
}
run_sea() {  # tag gpu args...
  local tag=$1 g=$2; shift 2
  echo "busy $g $tag" >> S7p_run1/gpus.txt
  date -u "+$tag start %H:%M:%S on cuda:$g" >> S7p_run1/status.txt
  nice -n 10 $PY code/strand7_sea_page_v1.py --device cuda:$g "$@" --out S7p_run1 --tag $tag > S7p_run1/$tag.log 2>&1
  echo "$tag exit $? $(date -u +%H:%M:%S)" >> S7p_run1/status.txt
  sed -i "/busy $g $tag/d" S7p_run1/gpus.txt
}
HL0=S7a_run2/s7a_heldlock_E16_R4_A_fields.npz
HL0J=S7a_run2/s7a_heldlock_E16_R4_A.json
wait_thread; g=$(free_gpu); run_sea F0 $g --flat 1 --rs0 32 --channels 0.5 --T-end-over-rs 20 --tau-over-rs 0.5 --out-times 0,2,4,8,12,16,20 --check-times 0,20 &
sleep 30
wait_thread; g=$(free_gpu); run_sea S0 $g --fields $HL0 --s7a-json $HL0J --static-from-over-rs 0 --channels 0.5 --T-end-over-rs 20 --tau-over-rs 0.5 --out-times 0,2,4,8,12,16,20 --check-times 0,20 &
wait
until grep -q "heldlock_E16_h1 exit" S7p_fields/status.txt; do sleep 30; done
HL=S7p_fields/s7a_heldlock_E16_h1_fields.npz
HLJ=S7p_fields/s7a_heldlock_E16_h1.json
wait_thread; g=$(free_gpu); run_sea P_j0.5 $g --fields $HL --s7a-json $HLJ --channels 0.5 --connection-control 1 --T-end-over-rs 30 &
sleep 30
wait_thread; g=$(free_gpu); run_sea P_j1.5 $g --fields $HL --s7a-json $HLJ --channels 1.5 --T-end-over-rs 30 &
sleep 60
wait_thread; g=$(free_gpu); run_sea P_j2.5 $g --fields $HL --s7a-json $HLJ --channels 2.5 --T-end-over-rs 30 &
wait
nice -n 10 $PY code/strand7_page_data_v1.py --records S7p_run1/s7p_P_j0.5.json,S7p_run1/s7p_P_j1.5.json,S7p_run1/s7p_P_j2.5.json --out S7p_run1/page_P.npz > S7p_run1/data.log 2>&1
nice -n 10 $PY code/strand7_page_fig_v1.py --data S7p_run1/page_P.npz --out S7p_run1/page_P_fig > S7p_run1/fig.log 2>&1
echo "data and figure exit $? $(date -u +%H:%M:%S)" >> S7p_run1/status.txt
echo DONE >> S7p_run1/status.txt
