#!/bin/bash
# launch_s7_twocopy_timing_v4.sh (builder analyst; engine spec 36.3 (2); the brief lead/BRIEF_s7_v3_builder.md: "one record timed first"):
# the timing run of engine_s7_twocopy_v3.py before the members. The box CPU (16 threads, nice 10); the tau 8 member with T0 by 33.4 (2)'s
# rule (18.000, k0 72), 74 records: 72 replayed, then 2 S7 records (T0 with the law's 8 labels and the run's 2; 18.250 with the rule
# contrast, the end solves and the half-zone split beyond R_t(t)); the compose exercised on the timing json given twice, with the gate against
# version 2's tau 8 member json (also given twice). No states npz. The split's own cost was timed first by s7_halfzone_check_v1.py on version
# 2's stored end states: 0.41 to 0.51 s per configuration beyond R_t(T0) with the cross bound, 0.84 to 0.89 s over the whole chain.
# Versions 1 to 3 of this launcher timed engine_s7_twocopy_v1.py and v2.py.
cd ${QI2_ROOT}/engine || exit 1
PY=${QI2_ROOT}/python
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
mkdir -p S7
nice -n 10 $PY code/engine_s7_twocopy_v3.py --member H100b/C3set/set17_rs32_A_R2_tau8_ell4_avg --records 74 --threads 16 \
    --out-json S7/s7_twocopy_v3_timing.json > S7/s7_twocopy_v3_timing.log 2>&1
echo "member exit $?" > S7/s7_twocopy_v3_timing.done
nice -n 10 $PY code/engine_s7_twocopy_v3.py --compose S7/s7_twocopy_v3_timing.json,S7/s7_twocopy_v3_timing.json \
    --gate-v2 S7/s7_twocopy_v2_tau8.json,S7/s7_twocopy_v2_tau8.json \
    --md S7/s7_twocopy_v3_timing.md --out-json S7/s7_twocopy_v3_timing_compose.json --md-every-rs 0.25 >> S7/s7_twocopy_v3_timing.log 2>&1
echo "compose exit $?" >> S7/s7_twocopy_v3_timing.done
