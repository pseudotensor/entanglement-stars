#!/bin/bash
# launch_s7_copyslot_timing_v1.sh (builder analyst; engine spec 37.4 (1) and (3); the brief lead/BRIEF_s7_copyslot_replay9_builder.md: "step 1
# on the stored history on the box CPU (ssh compute-host; ${QI2_ROOT}/python, nice -n 10, at most 16 threads; one record
# timed first)"): the timing run of engine_s7_copyslot_v1.py before the full run. The box CPU (16 threads, nice 10); the tau 8 member at part
# 3's T0 18.000 (k0 72), 74 records: 72 replayed, the T0 record (the selection, the code's checks, the solves) and one record after it with
# every readout and the dense checks; the compose exercised on the timing json; the states npz written so that its save path is exercised.
cd ${QI2_ROOT}/engine || exit 1
PY=${QI2_ROOT}/python
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
mkdir -p S7
nice -n 10 $PY code/engine_s7_copyslot_v1.py --member H100b/C3set/set17_rs32_A_R2_tau8_ell4_avg --records 74 --threads 16 \
    --out-json S7/s7_copyslot_v1_timing.json --out-npz S7/s7_copyslot_v1_timing_states.npz > S7/s7_copyslot_v1_timing.log 2>&1
echo "member exit $?" > S7/s7_copyslot_v1_timing.done
nice -n 10 $PY code/engine_s7_copyslot_v1.py --compose S7/s7_copyslot_v1_timing.json --md S7/s7_copyslot_v1_timing.md \
    --out-json S7/s7_copyslot_v1_timing_compose.json --md-every-rs 0.25 >> S7/s7_copyslot_v1_timing.log 2>&1
echo "compose exit $?" >> S7/s7_copyslot_v1_timing.done
