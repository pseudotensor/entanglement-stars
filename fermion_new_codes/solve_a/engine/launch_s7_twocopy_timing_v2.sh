#!/bin/bash
# launch_s7_twocopy_timing_v2.sh (builder analyst; engine spec 30.5 S7): the timing run of engine_s7_twocopy_v1.py as amended by 30.5
# (the rotated references, the C^2 window at the dust minimum, the label and theta rule) before the full run. The box CPU (16 threads,
# nice 10); the tau 8 member with T0 at T/rs 16.5 (its first record holding both trapping zeros), 70 records (66 replayed, 4 S7 records:
# the label path, the two-copy continuation, the exact ladder and the end solves timed). Version 1 of this launcher timed the
# pre-30.5 draft (its outputs s7_twocopy_v1_timing.* superseded).
cd ${QI2_ROOT}/engine || exit 1
PY=${QI2_ROOT}/python
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
mkdir -p S7
nice -n 10 $PY code/engine_s7_twocopy_v1.py --member H100b/C3set/set17_rs32_A_R2_tau8_ell4_avg --T0-over-rs 16.5 --records 70 --threads 16 \
    --out-json S7/s7_twocopy_v1_timing2.json --out-npz S7/s7_twocopy_v1_timing2_states.npz > S7/s7_twocopy_v1_timing2.log 2>&1
echo "member exit $?" > S7/s7_twocopy_v1_timing2.done
nice -n 10 $PY code/engine_s7_twocopy_v1.py --compose S7/s7_twocopy_v1_timing2.json,S7/s7_twocopy_v1_timing2.json \
    --md S7/s7_twocopy_v1_timing2.md --out-json S7/s7_twocopy_v1_timing2_compose.json --md-every-rs 0.25 >> S7/s7_twocopy_v1_timing2.log 2>&1
echo "compose exit $?" >> S7/s7_twocopy_v1_timing2.done
