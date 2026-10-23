#!/bin/bash
# d_v5_chain_v1.sh (the lane's analyst builder; engine spec 40.3 (2): "the D chain continues on every slice"): D version 5's readout on each new
# continuation slice from T/r_s 58 (every 2 r_s, as B stores them), one batch per slice by launch_massive_channel_D_v5_v1.sh (8 threads), then the
# signed report over every batch json (d_v5_signed_report_v1.py, no trigger) into Dmassive/v5/D_v5_signed_<T>.md. Waits at most 8 h per slice.
cd ${QI2_ROOT}/engine || exit 2
PY=${QI2_ROOT}/python
SD=states/c19_rs64_A_R2_tau8_ell4_avg_ext100
LOG=Dmassive/v5/chain_v1.log
for T in $(seq 58 2 100); do
  p=$(printf "%s/orbitals_c19_rs64_A_R2_tau8_ell4_avg_ext100_T%07.2f.npz" $SD $T)
  w=0; until [ -f "$p" ] || [ $w -ge 480 ]; do sleep 60; w=$((w + 1)); done
  [ -f "$p" ] || { echo "$(date -u +%H:%M:%S) UTC T $T: no slice after 8 h; the chain ends" >> $LOG; exit 5; }
  sleep 60
  bash code/launch_massive_channel_D_v5_v1.sh t$T "$p"
  ALL=$(ls Dmassive/v5/Dv5_*.json | grep -v partial | tr '\n' ',' | sed 's/,$//')
  $PY code/d_v5_signed_report_v1.py --jsons "$ALL" --md Dmassive/v5/D_v5_signed_$T.md --out-json Dmassive/v5/D_v5_signed_$T.json >> $LOG 2>&1
  echo "$(date -u +%H:%M:%S) UTC T $T done" >> $LOG
done
