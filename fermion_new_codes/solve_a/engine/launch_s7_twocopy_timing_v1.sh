#!/bin/bash
# launch_s7_twocopy_timing_v1.sh (builder analyst; engine spec 29.5 S7): the timing run of engine_s7_twocopy_v1.py before the full run
# (the brief: "Time one record before a full run"). The box CPU (16 threads, nice 10); the tau 8 member with T0 at its second record so
# that the C-init preparation, one replay record and three S7 records (one with the exact ladder and the solves) are timed; the
# windows are part 1's values (r_in/2 and (r_in + R_t)/2 of the record at T/rs 19.750); the compose is exercised on the timing json.
cd ${QI2_ROOT}/engine || exit 1
PY=${QI2_ROOT}/python
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
mkdir -p S7
nice -n 10 $PY code/engine_s7_twocopy_v1.py --member H100b/C3set/set17_rs32_A_R2_tau8_ell4_avg --T0-over-rs 0.25 \
    --windows 8.641383986367095,20.85537523489459 --records 4 --threads 16 \
    --out-json S7/s7_twocopy_v1_timing.json --out-npz S7/s7_twocopy_v1_timing_states.npz > S7/s7_twocopy_v1_timing.log 2>&1
nice -n 10 $PY code/engine_s7_twocopy_v1.py --compose S7/s7_twocopy_v1_timing.json,S7/s7_twocopy_v1_timing.json \
    --md S7/s7_twocopy_v1_timing.md --out-json S7/s7_twocopy_v1_timing_compose.json --md-every-rs 0.25 >> S7/s7_twocopy_v1_timing.log 2>&1
echo "exit $?" > S7/s7_twocopy_v1_timing.done
