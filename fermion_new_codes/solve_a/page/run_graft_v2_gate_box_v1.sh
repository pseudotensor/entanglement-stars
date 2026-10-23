#!/bin/bash
# run_graft_v2_gate_box_v1.sh (solve builder 10, analyst; engine spec 52.4 (2)): the graft version 2's bitwise part of its gate on box GPU 2,
# the device version 1's gate ran on. Waits for version 1's gate to end (its status line "exit"), then page_graft_pullback_v2 at --append 0
# on the stored c15 hold with the truncation off and the null readout on (0 by construction there), 60 r_s (30 rows per channel), compared
# number by number with version 1's rows (--compare-rows; BITWISE: yes is the pass) and with the stored summary (version 1's criterion).
# Under the 8192 MiB guard on GPU 2 (stopped and left for the builder). Outputs on ${QI2_ROOT}/runs.
set -u
G=${QI2_ROOT}/graft; cd $G || exit 1
PY=${QI2_ROOT}/python
REC=${QI2_ROOT}/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json
FLD=${QI2_ROOT}/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11_fields.npz
ST=${QI2_ROOT}/v6bA_s4_state_at_015.000.pt
until grep -q "^exit" status.txt; do sleep 60; done
echo "v2 gate (bitwise part) start $(date -u '+%F %T UTC')" >> status.txt
PYTHONPATH=$G/code CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 nice -n 10 $PY code_v2/page_graft_pullback_v2.py --record $REC --fields $FLD --init-state $ST \
  --freeze-at-rs 15 --append 0 --duration-rs 60 --null 1 --trunc 0 --device cuda:0 --out runs --tag graft_v2_gate_c15_off \
  --compare in/v6kA_c15_summary.json --compare-rows runs/graft_gate_c15_rows.json > logs/graft_v2_gate_c15_off.log 2>&1 &
P=$!
while [ -e /proc/$P ]; do sleep 30; f=$(nvidia-smi -i 2 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
  if [ "${f:-99999}" -lt 8192 ]; then kill $P; echo "v2 gate GUARD stop $(date -u '+%F %T UTC') free $f" >> status.txt; fi; done
wait $P; echo "v2 gate (bitwise part) exit $? $(date -u '+%F %T UTC')" >> status.txt
