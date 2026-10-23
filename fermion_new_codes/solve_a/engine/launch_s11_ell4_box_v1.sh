#!/bin/bash
# launch_s11_ell4_box_v1.sh (the lane's analyst builder; engine spec 41.1 (3), the pathfinder's factor-2 control): the S11 ell 4a member, S11's
# preparation and arguments at --ell 4 (ell/2GE 0.25) on engine_collapse_v35 --nk-jvp exact (41.3 (3)), the LZ convention, to T/r_s 40 or its
# stop, complex128 slices at S11's clocks, on box GPU 0 (the A6000, the ell 2a member's own device type) with 4 CPU threads, nice 10, as soon as
# the certified-jvp segment has freed it (the builder's placement: 41.6 names pod 1's H100, whose queue holds the r_s/a 32 cost measurement and
# the jvp member first; the lead may move it). Output S11ell4/ (tag s11_ball_rs16_ell4_tau2_v35jvp).
cd ${QI2_ROOT}/engine || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PY=${QI2_ROOT}/python
TAG=s11_ball_rs16_ell4_tau2_v35jvp
mkdir -p S11ell4 states/$TAG
echo "start $(date -u '+%Y-%m-%d %H:%M:%S UTC') gpu 0 pid $$" >> S11ell4/status_$TAG.txt
nice -n 10 $PY code/engine_collapse_v35.py --m 0 --g 0.25 --rs 16 --nsites 1024 --ell 4 --kappa-sea 10 --tau 2 --T-over-rs 40 --reference R2 --half-offset 0 --collar-policy flag --support-rel 1e-14 --site-fields averaged --stop-E 1e-7 --stop-fields 1e-6 --maxit 20 --mix newton --nk-maxiter 30 --nk-krylov 40 --nk-restarts 2 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3 --c-LZ 1.0 --window-max 4 --m1-simpson-steps auto --massless-ball --ball-R0-over-rs 4 --ball-x0-over-R0 0.5 --ball-sigma-over-R0 0.25 --ball-orbitals 1:6 --fields-v17 --s11-collars 8,16 --s11-stop 1 --slices-c128 0,1,2,3,4,5,6,8,10,12,16,20,24,28,32,36,40 --slice-dir states/$TAG --gpu 0 --nk-jvp exact --out S11ell4 --tag $TAG > S11ell4/$TAG.log 2>&1
echo "exit $? $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> S11ell4/status_$TAG.txt
