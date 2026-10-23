#!/bin/bash
# launch_s7_twocopy_v2.sh (builder analyst; engine spec 33.4 (2), S7 part 3 (binding, turn 41); the brief lead/BRIEF_s7_part3_builder.md):
# S7 part 3 on the box CPU (16 threads, nice 10; the members one after the other, so the lane's S7 work never holds more than 16 threads).
# (1) The primary: the r_s 32 tau 8 member (set17_rs32_A_R2_tau8_ell4_avg), T0 by 33.4 (2)'s rule (the stored record in T/rs [17.5, 18.5]
# with the largest R_t - r_in: 18.000, R_t - r_in 14.9838), the window at the dust shell density's interior minimum within [r_in, R_t] if one
# exists, else centred in the annulus, w 4; theta the T0 law's root at 0.8 of the 1e-6 bar; (2) the step control: the tau 4 member
# (set17_rs32_A_R2_tau4_ell4_avg) at the same T0 with the primary's window and theta (--match-json); (3) the compose: the results md and
# json (the primary alone if the control fails). The timing (launch_s7_twocopy_timing_v3.sh, 75 records of the tau 8 member, the same
# script): "C-init re-prepared in 231 s"; replay records 9.5 to 13.7 s (mean 13.3 s); the T0 record (the law's 8 labels, the run's 2) 56.4 s;
# S7 records 22.4 s (with the rule contrast) and 24.0 s (with the end solves); "done: 75 records in 1293 s". Projected (builder's
# arithmetic on those numbers, the S7 records without a solve taken 6 s cheaper): tau 8 3288 s (55 min), tau 4 5761 s (96 min), both 2.5 h.
cd ${QI2_ROOT}/engine || exit 1
PY=${QI2_ROOT}/python
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
mkdir -p S7
nice -n 10 $PY code/engine_s7_twocopy_v2.py --member H100b/C3set/set17_rs32_A_R2_tau8_ell4_avg --threads 16 \
    --out-json S7/s7_twocopy_v2_tau8.json --out-npz S7/s7_twocopy_v2_tau8_states.npz > S7/s7_twocopy_v2_tau8.log 2>&1
rc=$?
echo "tau8 exit $rc" > S7/s7_twocopy_v2_tau8.done
[ $rc -eq 0 ] || exit $rc
nice -n 10 $PY code/engine_s7_twocopy_v2.py --member H100b/C3set/set17_rs32_A_R2_tau4_ell4_avg --match-json S7/s7_twocopy_v2_tau8.json --threads 16 \
    --out-json S7/s7_twocopy_v2_tau4.json --out-npz S7/s7_twocopy_v2_tau4_states.npz > S7/s7_twocopy_v2_tau4.log 2>&1
rc4=$?
echo "tau4 exit $rc4" > S7/s7_twocopy_v2_tau4.done
if [ $rc4 -eq 0 ]; then
    nice -n 10 $PY code/engine_s7_twocopy_v2.py --compose S7/s7_twocopy_v2_tau8.json,S7/s7_twocopy_v2_tau4.json \
        --md S7/s7_twocopy_v2.md --out-json S7/s7_twocopy_v2.json > S7/s7_twocopy_v2_compose.log 2>&1
else
    nice -n 10 $PY code/engine_s7_twocopy_v2.py --compose S7/s7_twocopy_v2_tau8.json \
        --md S7/s7_twocopy_v2.md --out-json S7/s7_twocopy_v2.json > S7/s7_twocopy_v2_compose.log 2>&1
fi
echo "compose exit $?" > S7/s7_twocopy_v2.done
