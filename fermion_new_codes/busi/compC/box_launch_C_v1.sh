#!/bin/bash
# Computation C's pathfinder member on a box A6000 (inbox #84; REVIEWA_gpu_economics_v1.md sec. 3 and line 6): box_launch_C_v1.sh s4|s2 GPU.
# A 3-record pace test that continues as the member's run if its record stays within about 2 times the H100's 23 s. Engine engine_twofield_v3
# on the v36 base (the box's verified copies: code/compC, code/compC_engine_v36), RUN_ORDER_C_v1 sec. 5's arguments plus --field-kernel-cache
# --v36-fixes; CUDA_VISIBLE_DEVICES pins the card. Everything under ${QI2_ROOT}/busi_compC_box (the root filesystem is kept free). The
# guard is box_guard_C_v1.sh on the same card. The remote-box rule: waits while the 1-minute load is at or above 90; nice -n 10; 16 threads.
M=${1:?s4 or s2}; G=${2:?gpu index}
case $M in s4) ELL=4; TAG=C32_s4_ell4_tau2_twofield_v3_box;; s2) ELL=2; TAG=C32_s2_ell2_tau2_twofield_v3_box;; *) exit 2;; esac
B=${QI2_ROOT}/$M; mkdir -p $B/results/C32 $B/states $B/logs; cd $B
C=${QI2_ROOT}/compC; E=${QI2_ROOT}/compC_engine_v36
export BUSI_ENGINE_DIR=$E BUSI_PAGE_DIR=$E PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$G
while [ $(cut -d' ' -f1 /proc/loadavg | cut -d. -f1) -ge 90 ]; do sleep 300; done
COMMON="--m 0.2 --g 0.25 --rs 32 --R0-over-rs 4 --reading A --nsites 2048 --prep-ell 4 --kappa-sea 10 --tau 2 --T-over-rs 40 --reference R2 --half-offset 0 --collar-policy flag --support-rel 1e-14 --site-fields averaged --stop-E 1e-7 --stop-fields 1e-6 --maxit 20 --mix newton --nk-maxiter 30 --nk-krylov 40 --nk-restarts 0 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3 --nk-jvp exact --delta-rtol-every 10 --commutator-every 1 --streams 0 --m1-simpson-steps auto --c-LZ 1.0 --window-max 4 --light-kappas 1,2 --light-guard-kappas 3 --light-coupling on --light-transport banded --light-d-min 1e-7 --s11-collars 8,16,24 --s11-stop 1 --slices-c128 0,4,8,12,13,14,15,16,17,18,19,20,22,24,28,32,36,40 --slices-light-c128 1,2,3,5,6,7,9,10,10.5,11,11.5,12.5,13.5,14.5,15.5,16.5,17.5,18.5,19.5,20.5,21,21.5,22.5,23,23.5,25,26,27,29,30,31,33,34,35,37,38,39 --field-kernel-cache --v36-fixes"
echo "C32_$M box launch $(date -u +%Y-%m-%dT%H:%M:%SZ) gpu $G load $(cut -d' ' -f1 /proc/loadavg) gpu_used_MiB $(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits)" >> $B/status.txt
nice -n 10 ${QI2_ROOT}/python $C/engine_twofield_v3.py $COMMON --ell $ELL --slice-dir $B/states --gpu 0 --out $B/results/C32 --tag $TAG > $B/logs/C32_$M.log 2>&1 &
PID=$!; echo $PID > $B/C32_$M.pid; echo "C32_$M pid $PID" >> $B/status.txt
wait $PID; echo "C32_$M exit $? $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> $B/status.txt
