#!/bin/bash
# v36gate_pod5_v1.sh (the lane's analyst builder; engine spec 42.1 (3)): engine_collapse_v36's gates on pod 5's H100 (the GPU near idle beside
# S11's CPU-bound restart; 4 CPU threads, nice 10), in order: G1v36 bitwise against v35 with every v36 option off: (a) v35 and (b) v36 on S11's
# 3 records 1.125 to 1.375 from the T/r_s 1.0 slice (--nk-jvp exact, both here), compare_v33_gate_v1; (c) v36 options off on the cost run's
# preparation plus 2 records against pod 1's v35 cost run (compare_v33_gate_v1; a cross-device difference is reported as such); the whole-run
# gate: (d) v36 --field-kernel-cache --v36-fixes on S11's 1.125 to 1.5 against pod 1's v35 jvp segment and (e) on the cost run's preparation
# plus 2 records against pod 1's v35 cost run, compare_member_floor_v1 at twice the floor; walls, evaluations and products per record in the jsons.
D=${QI2_ROOT}/v36gate; cd $D || exit 1
export PYTHONPATH=${QI2_ROOT}/code OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
C=${QI2_ROOT}/code; S=$D/status.txt; TAG=s11_ball_rs16_ell2_tau2_v1
st() { echo "$* $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> $S; }
S11="--m 0 --g 0.25 --rs 16 --nsites 1024 --ell 2 --kappa-sea 10 --tau 2 --reference R2 --half-offset 0 --collar-policy flag --support-rel 1e-14 --site-fields averaged --stop-E 1e-7 --stop-fields 1e-6 --maxit 20 --mix newton --nk-maxiter 30 --nk-krylov 40 --nk-restarts 2 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3 --c-LZ 1.0 --window-max 4 --m1-simpson-steps auto --massless-ball --ball-R0-over-rs 4 --ball-x0-over-R0 0.5 --ball-sigma-over-R0 0.25 --ball-orbitals 1:6 --fields-v17 --s11-collars 8,16 --s11-stop 1 --gpu 0 --nk-jvp exact --T-over-rs 40 --resume-slice states/$TAG/orbitals_${TAG}_T0001.00.npz --resume-run S11cert/snap/$TAG"
COST="--m 0 --g 0.25 --rs 32 --nsites 2048 --ell 2 --kappa-sea 10 --tau 2 --reference R2 --half-offset 0 --collar-policy flag --support-rel 1e-14 --site-fields averaged --stop-E 1e-7 --stop-fields 1e-6 --maxit 20 --mix newton --nk-maxiter 30 --nk-krylov 40 --nk-restarts 2 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3 --c-LZ 1.0 --window-max 4 --m1-simpson-steps auto --massless-ball --ball-R0-over-rs 4 --ball-x0-over-R0 0.5 --ball-sigma-over-R0 0.25 --ball-orbitals 1:6 --fields-v17 --s11-collars 8,16 --s11-stop 1 --gpu 0 --nk-jvp exact --T-over-rs 0.125"
mkdir -p a b c d e
st "a start"; nice -n 10 python $C/engine_collapse_v35.py $S11 --segment-end-T-over-rs 1.375 --out a --tag g1_s11 > a/log 2>&1; st "a exit $?"
st "b start"; nice -n 10 python $C/engine_collapse_v36.py $S11 --segment-end-T-over-rs 1.375 --out b --tag g1_s11 > b/log 2>&1; st "b exit $?"
python $C/compare_v33_gate_v1.py a/g1_s11 b/g1_s11 > G1_s11.txt 2>&1; st "G1 s11 $(tail -1 G1_s11.txt)"
st "d start"; nice -n 10 python $C/engine_collapse_v36.py $S11 --segment-end-T-over-rs 1.5 --field-kernel-cache --v36-fixes --out d --tag w_s11 > d/log 2>&1; st "d exit $?"
python $C/compare_member_floor_v1.py --run d/w_s11 --ref ref_s11/gpu_jvp --label-run v36_cache_fixes --label-ref v35_jvp_pod1 --md W_s11.md --out-json W_s11.json > /dev/null 2>&1; st "W s11 $(grep -m1 'every one within' W_s11.md)"
st "e start"; nice -n 10 python $C/engine_collapse_v36.py $COST --field-kernel-cache --v36-fixes --out e --tag w_cost > e/log 2>&1; st "e exit $?"
python $C/compare_member_floor_v1.py --run e/w_cost --ref ref_cost/rs32_cost_ell2 --label-run v36_cache_fixes --label-ref v35_cost_pod1 --md W_cost.md --out-json W_cost.json > /dev/null 2>&1; st "W cost $(grep -m1 'every one within' W_cost.md)"
st "c start"; nice -n 10 python $C/engine_collapse_v36.py $COST --out c --tag g1_cost > c/log 2>&1; st "c exit $?"
python $C/compare_v33_gate_v1.py ref_cost/rs32_cost_ell2 c/g1_cost > G1_cost.txt 2>&1; st "G1 cost $(tail -1 G1_cost.txt)"
st "all done"
