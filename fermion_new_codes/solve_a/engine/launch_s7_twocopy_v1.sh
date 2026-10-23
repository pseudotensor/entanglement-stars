#!/bin/bash
# launch_s7_twocopy_v1.sh (builder analyst; engine spec 30.5 S7 (binding), 29.5 and 28.5 S7): S7 part 2 on the box CPU (16 threads,
# nice 10; box GPU 2 held by other lanes' runs at launch, so the CPU per the brief). (1) The primary: the r_s 32 tau 8 member
# (set17_rs32_A_R2_tau8_ell4_avg), T0 by part 1's rule (the record nearest T/rs 20 holding both trapping zeros: 19.750), the C^2 window
# at the dust shell density's minimum inside [r_in, R_t] there, theta pi/4 and pi/2 with 30.5's reduction to pi/8; (2) the step control:
# the tau 4 member (set17_rs32_A_R2_tau4_ell4_avg) at the same T0 with the primary's window and thetas (--match-json; the tau 4 member
# holds no trapping zeros at 19.750); (3) the compose: the results md and json. The ell control: no r_s 32 ell 8 member on v17 exists
# (C3set25's v25 run stopped at T/rs 3.75), so none runs.
cd ${QI2_ROOT}/engine || exit 1
PY=${QI2_ROOT}/python
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
mkdir -p S7
nice -n 10 $PY code/engine_s7_twocopy_v1.py --member H100b/C3set/set17_rs32_A_R2_tau8_ell4_avg --threads 16 \
    --out-json S7/s7_twocopy_v1_tau8.json --out-npz S7/s7_twocopy_v1_tau8_states.npz > S7/s7_twocopy_v1_tau8.log 2>&1
rc=$?
echo "tau8 exit $rc" > S7/s7_twocopy_v1_tau8.done
[ $rc -eq 0 ] || exit $rc
nice -n 10 $PY code/engine_s7_twocopy_v1.py --member H100b/C3set/set17_rs32_A_R2_tau4_ell4_avg --T0-over-rs 19.75 \
    --match-json S7/s7_twocopy_v1_tau8.json --threads 16 \
    --out-json S7/s7_twocopy_v1_tau4.json --out-npz S7/s7_twocopy_v1_tau4_states.npz > S7/s7_twocopy_v1_tau4.log 2>&1
echo "tau4 exit $?" > S7/s7_twocopy_v1_tau4.done
nice -n 10 $PY code/engine_s7_twocopy_v1.py --compose S7/s7_twocopy_v1_tau8.json,S7/s7_twocopy_v1_tau4.json \
    --md S7/s7_twocopy_v1.md --out-json S7/s7_twocopy_v1.json > S7/s7_twocopy_v1_compose.log 2>&1
echo "compose exit $?" > S7/s7_twocopy_v1.done
