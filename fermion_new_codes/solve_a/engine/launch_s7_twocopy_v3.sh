#!/bin/bash
# launch_s7_twocopy_v3.sh (builder analyst; engine spec 36.3 (2), turn 44; the brief lead/BRIEF_s7_v3_builder.md: "both members as v2 did,
# outputs under ${QI2_ROOT}/S7 with v3 names"): engine_s7_twocopy_v3.py on the box CPU (16 threads, nice 10; the
# members one after the other, as version 2's launcher ran them).
# (1) The primary: the r_s 32 tau 8 member (set17_rs32_A_R2_tau8_ell4_avg), T0 by 33.4 (2)'s rule (18.000), the window and theta by
# version 2's rules (X 18.7956, w 4, theta the T0 law's root at 0.8 of the 1e-6 bar), to T/rs 20.0 (36.3 (2): "the tau 8 member to T/rs
# 20.0 only"; the closing clock 20.000): the half-zone split beyond R_t(t) per record from T0 to the closing, beside the unsplit bound.
# (2) The step control: the tau 4 member (set17_rs32_A_R2_tau4_ell4_avg) at the primary's T0, X, w and theta (--match-json, version 2's
# rule), to the same T/rs 20.0 (its closing clock 19.625): the brief's "both members".
# (3) The compose: the md and json with the split's tables and the gate against version 2's member jsons on the records both compute
# (the primary alone if the control fails, gated against version 2's primary).
# The timing (launch_s7_twocopy_timing_v4.sh, 74 records of the tau 8 member, the same script) is in S7/s7_twocopy_v3_timing.log.
cd ${QI2_ROOT}/engine || exit 1
PY=${QI2_ROOT}/python
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
mkdir -p S7
nice -n 10 $PY code/engine_s7_twocopy_v3.py --member H100b/C3set/set17_rs32_A_R2_tau8_ell4_avg --T-end-over-rs 20.0 --threads 16 \
    --out-json S7/s7_twocopy_v3_tau8.json --out-npz S7/s7_twocopy_v3_tau8_states.npz > S7/s7_twocopy_v3_tau8.log 2>&1
rc=$?
echo "tau8 exit $rc" > S7/s7_twocopy_v3_tau8.done
[ $rc -eq 0 ] || exit $rc
nice -n 10 $PY code/engine_s7_twocopy_v3.py --member H100b/C3set/set17_rs32_A_R2_tau4_ell4_avg --match-json S7/s7_twocopy_v3_tau8.json --T-end-over-rs 20.0 --threads 16 \
    --out-json S7/s7_twocopy_v3_tau4.json --out-npz S7/s7_twocopy_v3_tau4_states.npz > S7/s7_twocopy_v3_tau4.log 2>&1
rc4=$?
echo "tau4 exit $rc4" > S7/s7_twocopy_v3_tau4.done
if [ $rc4 -eq 0 ]; then
    nice -n 10 $PY code/engine_s7_twocopy_v3.py --compose S7/s7_twocopy_v3_tau8.json,S7/s7_twocopy_v3_tau4.json \
        --gate-v2 S7/s7_twocopy_v2_tau8.json,S7/s7_twocopy_v2_tau4.json \
        --md S7/s7_twocopy_v3.md --out-json S7/s7_twocopy_v3.json > S7/s7_twocopy_v3_compose.log 2>&1
else
    nice -n 10 $PY code/engine_s7_twocopy_v3.py --compose S7/s7_twocopy_v3_tau8.json --gate-v2 S7/s7_twocopy_v2_tau8.json \
        --md S7/s7_twocopy_v3.md --out-json S7/s7_twocopy_v3.json > S7/s7_twocopy_v3_compose.log 2>&1
fi
echo "compose exit $?" > S7/s7_twocopy_v3.done
