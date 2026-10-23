#!/bin/bash
# launch_B2_pod4_v1.sh (the lane's analyst builder; busi's RULING_box_runs_v1.md sec. 4.2 and the lead's instruction of 2026-10-17 about 08:30
# UTC: B2, the doubled chain, the paper's background for A2 and D past 56, on the dedicated pod 4 H100): engine_collapse_v19 with B's own
# arguments (the production's) at --nsites 4096 (the chain to 64 r_s at r_s/a 64), from T = 0 to T/r_s 100, a new tag, complex128 slices at
# 12, 16, 20, 24, 28, 36 and every 2 r_s from 40 to 100 (the ruling's list), the fields every record; 8 CPU threads; ${QI2_ROOT}/B2.
# The slices are mirrored off the pod (to the box, then the owner's dataset per COMMS rule 9) and deleted locally once read, the local buffer
# held under 80 GB (the lane's mirror loop, a separate script). Usage (pod 4): setsid nohup bash ${QI2_ROOT}/launch_B2_pod4_v1.sh &
cd ${QI2_ROOT}/engine || exit 1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
T=c19_rs64_A_R2_tau8_ell4_avg_B2n4096
mkdir -p B2 states/$T
echo "B2 start $(date -u '+%Y-%m-%d %H:%M:%S UTC') pid $$" >> B2/status.txt
python code/engine_collapse_v19.py --m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 4096 --ell 4 --kappa-sea 10 --tau 8 --T-over-rs 100 --reference R2 --half-offset 0 --collar-policy flag --support-rel 1e-14 --site-fields averaged --stop-E 1e-7 --stop-fields 1e-6 --maxit 20 --mix plain --delta-rtol-every 10 --commutator-every 1 --streams 0 --m1-simpson-steps auto --c-LZ 1.0 --window-max 4 --slices-c128 12,16,20,24,28,36,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96,98,100 --slice-dir states/$T --gpu 0 --out B2 --tag $T > B2/$T.log 2>&1
echo "B2 exit $? $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> B2/status.txt
