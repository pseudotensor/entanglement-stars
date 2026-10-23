#!/bin/bash
# d_ext_chain_v1.sh (builder analyst; the D/S11 builder's handover, lead/api/turn47/d_s11_builder_report.md: "D, per new continuation slice:
# bash code/launch_massive_channel_D_ext_v2.sh b<n> <T>"): runs D on #36 B's continuation slices one at a time as they land, T/r_s 44 to 100
# every 2 (batches b3 to b31; b1 at 40 and b2r at 42 already ran): for each, waits until the slice file exists (polled every 60 s, at most 8 h),
# then calls launch_massive_channel_D_ext_v2.sh b<n> <T> (which itself waits for the run's json and fields npz to hold the record, at most 3 h,
# and runs D at 12 threads, nice 10). Sequential, so one D batch at a time on the box CPU. A batch that fails is logged and the chain goes on.
# Never stops anything. Log: Dmassive/d_ext_chain_v1.log. Usage (from ${QI2_ROOT}/engine): nohup bash code/d_ext_chain_v1.sh &
cd ${QI2_ROOT}/engine || exit 2
SD=states/c19_rs64_A_R2_tau8_ell4_avg_ext100
LOG=Dmassive/d_ext_chain_v1.log
echo "$(date -u +%H:%M:%S) UTC chain start (pid $$)" >> $LOG
n=3
for T in $(seq 44 2 100); do
  p=$(printf "%s/orbitals_c19_rs64_A_R2_tau8_ell4_avg_ext100_T%07.2f.npz" $SD $T)
  w=0
  until [ -f "$p" ] || [ $w -ge 480 ]; do sleep 60; w=$((w + 1)); done
  if [ ! -f "$p" ]; then echo "$(date -u +%H:%M:%S) UTC b$n T $T: slice not landed after 8 h; chain ends" >> $LOG; exit 5; fi
  sleep 30
  echo "$(date -u +%H:%M:%S) UTC b$n T $T start" >> $LOG
  bash code/launch_massive_channel_D_ext_v2.sh b$n $T >> $LOG 2>&1
  echo "$(date -u +%H:%M:%S) UTC b$n T $T exit $?" >> $LOG
  n=$((n + 1))
done
echo "$(date -u +%H:%M:%S) UTC chain done" >> $LOG
