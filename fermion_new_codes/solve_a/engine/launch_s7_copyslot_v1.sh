#!/bin/bash
# launch_s7_copyslot_v1.sh (builder analyst; engine spec 37.4 (1) to (3); the brief lead/BRIEF_s7_copyslot_replay9_builder.md): S7 part 4,
# step 1 on the box CPU (16 threads, nice 10; the members one after the other): (1) the primary, the r_s 32 tau 8 member
# (set17_rs32_A_R2_tau8_ell4_avg) at part 3's T0 18.000, both kappas 1, 2, eps 1e-8 and 1e-10; (2) the step control, the tau 4 member
# (set17_rs32_A_R2_tau4_ell4_avg) at the primary's T0 with the primary's orbitals projected onto its own retained subspaces (--match-json,
# --match-npz); (3) the compose (the primary alone if the control fails). Step 2 is not run: engine_collapse_v17 cannot start from a stored
# state (37.4 (2)'s condition). The timing (launch_s7_copyslot_timing_v1.sh, 74 records of the tau 8 member) is in S7/s7_copyslot_v1_timing.log.
cd ${QI2_ROOT}/engine || exit 1
PY=${QI2_ROOT}/python
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
mkdir -p S7
nice -n 10 $PY code/engine_s7_copyslot_v1.py --member H100b/C3set/set17_rs32_A_R2_tau8_ell4_avg --threads 16 \
    --out-json S7/s7_copyslot_v1_tau8.json --out-npz S7/s7_copyslot_v1_tau8_states.npz > S7/s7_copyslot_v1_tau8.log 2>&1
rc=$?
echo "tau8 exit $rc" > S7/s7_copyslot_v1_tau8.done
[ $rc -eq 0 ] || exit $rc
nice -n 10 $PY code/engine_s7_copyslot_v1.py --member H100b/C3set/set17_rs32_A_R2_tau4_ell4_avg --match-json S7/s7_copyslot_v1_tau8.json \
    --match-npz S7/s7_copyslot_v1_tau8_states.npz --threads 16 --out-json S7/s7_copyslot_v1_tau4.json --out-npz S7/s7_copyslot_v1_tau4_states.npz > S7/s7_copyslot_v1_tau4.log 2>&1
rc4=$?
echo "tau4 exit $rc4" > S7/s7_copyslot_v1_tau4.done
if [ $rc4 -eq 0 ]; then
    nice -n 10 $PY code/engine_s7_copyslot_v1.py --compose S7/s7_copyslot_v1_tau8.json,S7/s7_copyslot_v1_tau4.json \
        --md S7/s7_copyslot_v1.md --out-json S7/s7_copyslot_v1.json > S7/s7_copyslot_v1_compose.log 2>&1
else
    nice -n 10 $PY code/engine_s7_copyslot_v1.py --compose S7/s7_copyslot_v1_tau8.json \
        --md S7/s7_copyslot_v1.md --out-json S7/s7_copyslot_v1.json > S7/s7_copyslot_v1_compose.log 2>&1
fi
echo "compose exit $?" > S7/s7_copyslot_v1.done
