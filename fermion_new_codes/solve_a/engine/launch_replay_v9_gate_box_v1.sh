#!/bin/bash
# launch_replay_v9_gate_box_v1.sh (builder analyst; engine spec 37.5 (1)): the gates of engine_replay_v9 on the box CPU before its pod run
# (nice 10, 4 threads per job; three records each): (1) engine_replay_v8 and engine_replay_v9 on the v24 r_s 32 run (version 8's path,
# ops v2) and engine_replay_v9_gate_v1 comparing version 8's fields bitwise; (2) engine_replay_v9 on the v17 path (the set17 r_s 32 tau 16
# member, replay v2's handoff and spectra compared).
cd ${QI2_ROOT}/engine || exit 1
PY=${QI2_ROOT}/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
mkdir -p CERT3v9/gate
cd H100b || exit 1
( nice -n 10 $PY ../code/engine_replay_v8.py --run C3set24/c24_rs32_A_R2_tau2_ell4_avg --records 3 --md ../CERT3v9/gate/v8_c24.md --out-json ../CERT3v9/gate/v8_c24.json \
    --out-npz ../CERT3v9/gate/v8_c24 > ../CERT3v9/gate/v8_c24.log 2>&1; echo "v8 exit $?" >> ../CERT3v9/gate/status.txt ) &
( nice -n 10 $PY ../code/engine_replay_v9.py --run C3set24/c24_rs32_A_R2_tau2_ell4_avg --records 3 --md ../CERT3v9/gate/v9_c24.md --out-json ../CERT3v9/gate/v9_c24.json \
    --out-npz ../CERT3v9/gate/v9_c24 > ../CERT3v9/gate/v9_c24.log 2>&1; echo "v9 exit $?" >> ../CERT3v9/gate/status.txt ) &
( nice -n 10 $PY ../code/engine_replay_v9.py --run C3set/set17_rs32_A_R2_tau16_ell4_avg --records 3 --compare-prefix REPLAY2/replay2_set17_rs32_A_R2_tau16_ell4_avg \
    --md ../CERT3v9/gate/v9_set17_tau16.md --out-json ../CERT3v9/gate/v9_set17_tau16.json --out-npz ../CERT3v9/gate/v9_set17_tau16 > ../CERT3v9/gate/v9_set17_tau16.log 2>&1; \
    echo "v9 v17 exit $?" >> ../CERT3v9/gate/status.txt ) &
wait
nice -n 10 $PY ../code/engine_replay_v9_gate_v1.py --v8-json ../CERT3v9/gate/v8_c24.json --v8-handoff ../CERT3v9/gate/v8_c24_handoff.npz --v9-json ../CERT3v9/gate/v9_c24.json \
    --v9-handoff ../CERT3v9/gate/v9_c24_handoff.npz --md ../CERT3v9/gate/gate_v9_against_v8.md > ../CERT3v9/gate/gate.log 2>&1
echo "gate exit $?" >> ../CERT3v9/gate/status.txt
