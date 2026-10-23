#!/bin/bash
# Computation C's pathfinder, one member on any H100 pod: pod_launch_C_v2.sh s4|s2 [BASE]. pod5_launch_C_s4_v1.sh generalised by member and
# directory. The first launch on pod 5 (13:50:49 UTC) was stopped by the guard at 13:57:23 UTC, with C's device share at 33.6 GB after 11 records,
# above the run order's 27 GB. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is set to limit the caching allocator's fragmentation; it changes
# no arithmetic. Everything lives under BASE (default ${QI2_ROOT}/busi_compC) on the pod's container disk. The guard is pod_guard_C_v2.sh with the same
# member; the slices are mirrored by c_mirror (box) per COMMS rule 9.
M=${1:?member s4 or s2}; B=${2:-${QI2_ROOT}/busi_compC}
case $M in s4) ELL=4; TAG=C32_s4_ell4_tau2_twofield_v3;; s2) ELL=2; TAG=C32_s2_ell2_tau2_twofield_v3;; *) echo "member s4 or s2"; exit 2;; esac
cd $B || exit 1; mkdir -p results/C32 states logs
export BUSI_ENGINE_DIR=$B/engine BUSI_PAGE_DIR=$B/engine PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COMMON="--m 0.2 --g 0.25 --rs 32 --R0-over-rs 4 --reading A --nsites 2048 --prep-ell 4 --kappa-sea 10 --tau 2 --T-over-rs 40 --reference R2 --half-offset 0 --collar-policy flag --support-rel 1e-14 --site-fields averaged --stop-E 1e-7 --stop-fields 1e-6 --maxit 20 --mix newton --nk-maxiter 30 --nk-krylov 40 --nk-restarts 0 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3 --nk-jvp exact --delta-rtol-every 10 --commutator-every 1 --streams 0 --m1-simpson-steps auto --c-LZ 1.0 --window-max 4 --light-kappas 1,2 --light-guard-kappas 3 --light-coupling on --light-transport banded --light-d-min 1e-7 --s11-collars 8,16,24 --s11-stop 1 --slices-c128 0,4,8,12,13,14,15,16,17,18,19,20,22,24,28,32,36,40 --slices-light-c128 1,2,3,5,6,7,9,10,10.5,11,11.5,12.5,13.5,14.5,15.5,16.5,17.5,18.5,19.5,20.5,21,21.5,22.5,23,23.5,25,26,27,29,30,31,33,34,35,37,38,39 --field-kernel-cache --v36-fixes"
echo "C32_$M launch $(date -u +%Y-%m-%dT%H:%M:%SZ) load $(cut -d' ' -f1 /proc/loadavg) gpu_used_MiB $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) free_MiB $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" >> $B/status.txt
nice -n 10 python $B/code/engine_twofield_v3.py $COMMON --ell $ELL --slice-dir $B/states --gpu 0 --out $B/results/C32 --tag $TAG > $B/logs/C32_$M.log 2>&1 &
PID=$!; echo $PID > $B/C32_$M.pid; echo "C32_$M pid $PID" >> $B/status.txt
wait $PID; echo "C32_$M exit $? $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> $B/status.txt
