#!/bin/bash
# launch_s7_twocopy_timing_v3.sh (builder analyst; engine spec 33.4 (2); the brief lead/BRIEF_s7_part3_builder.md: "Time one record first,
# then run both members on the box CPU"): the timing run of engine_s7_twocopy_v2.py before the full run. The box CPU (16 threads, nice 10);
# the tau 8 member with T0 by 33.4 (2)'s rule (the stored record in T/rs [17.5, 18.5] with the largest R_t - r_in: 18.000, k0 72), 75
# records (72 replayed; 3 S7 records: the law's labels and the run's at T0, the rule contrast at the first record after it, the end solves
# at the third); the compose exercised on the timing json given twice (version 1's timing did the same). No states npz. Versions 1 and 2
# of this launcher timed engine_s7_twocopy_v1.py.
cd ${QI2_ROOT}/engine || exit 1
PY=${QI2_ROOT}/python
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
mkdir -p S7
nice -n 10 $PY code/engine_s7_twocopy_v2.py --member H100b/C3set/set17_rs32_A_R2_tau8_ell4_avg --records 75 --threads 16 \
    --out-json S7/s7_twocopy_v2_timing.json > S7/s7_twocopy_v2_timing.log 2>&1
echo "member exit $?" > S7/s7_twocopy_v2_timing.done
nice -n 10 $PY code/engine_s7_twocopy_v2.py --compose S7/s7_twocopy_v2_timing.json,S7/s7_twocopy_v2_timing.json \
    --md S7/s7_twocopy_v2_timing.md --out-json S7/s7_twocopy_v2_timing_compose.json --md-every-rs 0.25 >> S7/s7_twocopy_v2_timing.log 2>&1
echo "compose exit $?" >> S7/s7_twocopy_v2_timing.done
