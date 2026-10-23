#!/bin/bash
# engine_replay_v2 on box GPU 1 (engine spec 21.5: "GPU 1: the replay of the 102 records with stresses, rows and Gauss balance into the
# handoff"): first its smoke test on the r_s 16 v17 test run (V19T/t17_rs16, 7 records), then the stopped production's 102 records
# (H100/C3prod/c17_rs64_A_R2_tau8_ell4_avg) into REPLAY2/; one process at a time; the second only if the first exits 0. Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
PY=${QI2_ROOT}/python
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p REPLAY2
echo "replay2 smoke start $(date -u +%H:%M:%S)" >> REPLAY2/status.txt
nice -n 10 $PY code/engine_replay_v2.py --run V19T/t17_rs16 --gpu 0 --md REPLAY2/replay2_t17_rs16.md --out-json REPLAY2/replay2_t17_rs16.json --out-npz REPLAY2/replay2_t17_rs16 > REPLAY2/log_replay2_t17_rs16.log 2>&1
rc=$?; echo "replay2 smoke exit $rc $(date -u +%H:%M:%S)" >> REPLAY2/status.txt
[ $rc -ne 0 ] && { echo "DONE (stopped after the smoke test) $(date -u +%H:%M:%S)" >> REPLAY2/status.txt; exit 1; }
nice -n 10 $PY code/engine_replay_v2.py --run H100/C3prod/c17_rs64_A_R2_tau8_ell4_avg --gpu 0 --md REPLAY2/replay2_c17_rs64.md --out-json REPLAY2/replay2_c17_rs64.json --out-npz REPLAY2/replay2_c17_rs64 > REPLAY2/log_replay2_c17_rs64.log 2>&1
echo "replay2 production exit $? $(date -u +%H:%M:%S)" >> REPLAY2/status.txt
echo "DONE $(date -u +%H:%M:%S)" >> REPLAY2/status.txt
