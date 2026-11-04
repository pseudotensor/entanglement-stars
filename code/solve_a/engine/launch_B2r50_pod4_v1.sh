#!/bin/bash
# launch_B2r50_pod4_v1.sh (solve builder 5): B2 stopped at 17:18 UTC (record 51.875 the json's last; pod 4's /workspace quota full, no exit
# line written). The restart from B2's own complex128 T/r_s 50 slice with the gated runner run_collapse_v19_from_slice_v1.py (the one B's box
# continuation uses: the sources from the slice against the stored shell totals, acceptance 1e-8, else no start), B2's own arguments
# unchanged, --resume-from the crash run's files copied to ${QI2_ROOT}/B2crash, the same --out B2 and tag so sync_B2_to_pod1_v1 (A2's feed)
# continues; records 50.125 to 51.875 recomputed are the restart's check against the crash run's. --min-free-gb 12: a slice is skipped
# (slice_saved null) rather than the run dying on a full quota. Usage (pod 4): setsid nohup bash ${QI2_ROOT}/launch_B2r50_pod4_v1.sh &
cd ${QI2_ROOT}/engine || exit 1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
T=c19_rs64_A_R2_tau8_ell4_avg_B2n4096
echo "B2r50 start $(date -u '+%Y-%m-%d %H:%M:%S UTC') pid $$" >> ${QI2_ROOT}/B2r50_status.txt
python code/run_collapse_v19_from_slice_v1.py --m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 4096 --ell 4 --kappa-sea 10 --tau 8 --T-over-rs 100 --reference R2 --half-offset 0 --collar-policy flag --support-rel 1e-14 --site-fields averaged --stop-E 1e-7 --stop-fields 1e-6 --maxit 20 --mix plain --delta-rtol-every 10 --commutator-every 1 --streams 0 --m1-simpson-steps auto --c-LZ 1.0 --window-max 4 --slices-c128 12,16,20,24,28,36,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96,98,100 --slice-dir states/$T --gpu 0 --resume-from ${QI2_ROOT}/$T --resume-T-over-rs 50 --slice states/$T/orbitals_${T}_T0050.00.npz --min-free-gb 12 --out B2 --tag $T > ${QI2_ROOT}/B2r50_$T.log 2>&1
echo "B2r50 exit $? $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> ${QI2_ROOT}/B2r50_status.txt
