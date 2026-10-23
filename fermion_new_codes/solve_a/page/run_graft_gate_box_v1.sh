#!/bin/bash
# run_graft_gate_box_v1.sh (solve builder 9, analyst; engine spec 51.2 (1)): the graft pull-back's gate on box GPU 2 beside the queue's
# v6lA_s4k3gensplit (in its CPU phase, about 1% of the card) under the 8192 MiB guard: this job (the placed one) is stopped by its PID below
# 8192 MiB free and left for the builder. page_graft_pullback_v1.py with --append 0 on the stored c15 hold (the same closed 8192-site
# system), compared with its summary. Code, inputs and outputs: ${QI2_ROOT}/.
set -u
G=${QI2_ROOT}/graft; cd $G || exit 1
PY=${QI2_ROOT}/python
REC=${QI2_ROOT}/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json
FLD=${QI2_ROOT}/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11_fields.npz
ST=${QI2_ROOT}/v6bA_s4_state_at_015.000.pt
echo "start $(date -u '+%F %T UTC')" >> status.txt
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 nice -n 10 $PY code/page_graft_pullback_v1.py --record $REC --fields $FLD --init-state $ST --freeze-at-rs 15 --append 0 \
  --device cuda:0 --out runs --tag graft_gate_c15 --compare in/v6kA_c15_summary.json > logs/graft_gate_c15.log 2>&1 &
P=$!
while [ -e /proc/$P ]; do sleep 30; f=$(nvidia-smi -i 2 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
  if [ "${f:-99999}" -lt 8192 ]; then kill $P; echo "GUARD stop $(date -u '+%F %T UTC') free $f" >> status.txt; fi; done
wait $P; echo "exit $? $(date -u '+%F %T UTC')" >> status.txt
