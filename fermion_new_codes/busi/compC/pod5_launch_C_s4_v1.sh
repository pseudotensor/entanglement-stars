#!/bin/bash
# Computation C's pathfinder, the sigma/a 4 member, on pod 5 (inbox #71; turn 52 sec. 44 and turn 54 sec. 46.2 (2); RUN_ORDER_C_v1.md sec. 5
# re-pointed at engine_twofield_v3 on the engine_collapse_v36 base with --field-kernel-cache --v36-fixes, turn 51 43.1 (2)). Everything lives on
# the container disk under ${QI2_ROOT}/busi_compC (no /workspace use; pod 3's network volume stalled on 2026-10-17). The slices are mirrored to the
# box and the owner's dataset by c_mirror_v1.sh on the box (COMMS rule 9); the records are synced to the box. The guard (pod5_guard_C_v1.sh)
# stops this process, and only this process, when the card's free memory falls below 8 GB (turn 54 46.2 (1)).
B=${QI2_ROOT}/busi_compC; cd $B || exit 1; mkdir -p results/C32 states logs
export BUSI_ENGINE_DIR=$B/engine BUSI_PAGE_DIR=$B/engine PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
COMMON="--m 0.2 --g 0.25 --rs 32 --R0-over-rs 4 --reading A --nsites 2048 --prep-ell 4 --kappa-sea 10 --tau 2 --T-over-rs 40 --reference R2 --half-offset 0 --collar-policy flag --support-rel 1e-14 --site-fields averaged --stop-E 1e-7 --stop-fields 1e-6 --maxit 20 --mix newton --nk-maxiter 30 --nk-krylov 40 --nk-restarts 0 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3 --nk-jvp exact --delta-rtol-every 10 --commutator-every 1 --streams 0 --m1-simpson-steps auto --c-LZ 1.0 --window-max 4 --light-kappas 1,2 --light-guard-kappas 3 --light-coupling on --light-transport banded --light-d-min 1e-7 --s11-collars 8,16,24 --s11-stop 1 --slices-c128 0,4,8,12,13,14,15,16,17,18,19,20,22,24,28,32,36,40 --slices-light-c128 1,2,3,5,6,7,9,10,10.5,11,11.5,12.5,13.5,14.5,15.5,16.5,17.5,18.5,19.5,20.5,21,21.5,22.5,23,23.5,25,26,27,29,30,31,33,34,35,37,38,39 --field-kernel-cache --v36-fixes"
echo "C32_s4 launch $(date -u +%Y-%m-%dT%H:%M:%SZ) load $(cut -d' ' -f1 /proc/loadavg) gpu_used_MiB $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" >> $B/status.txt
nice -n 10 python $B/code/engine_twofield_v3.py $COMMON --ell 4 --slice-dir $B/states --gpu 0 --out $B/results/C32 --tag C32_s4_ell4_tau2_twofield_v3 > $B/logs/C32_s4.log 2>&1 &
PID=$!; echo $PID > $B/C32_s4.pid; echo "C32_s4 pid $PID" >> $B/status.txt
wait $PID; echo "C32_s4 exit $? $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> $B/status.txt
