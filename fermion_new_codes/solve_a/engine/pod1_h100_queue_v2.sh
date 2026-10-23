#!/bin/bash
# pod1_h100_queue_v2.sh: pod1_h100_queue_v1.sh without its step (3): the S11 ell 4a member runs on box GPU 0 from 07:48 UTC
# (launch_s11_ell4_box_v1.sh). v1's text follows.
# pod1_h100_queue_v1.sh (the lane's analyst builder; engine spec 41.6's order for pod 1's H100, with the owner's r_s/a 32 funding of 07:10 UTC):
# after the interim massive benchmark (bench53_pod1_gpu_v2.sh) and the r_s/a 32 cost measurement (launch_rs32_cost_pod1_v1.sh) have ended:
# (1) the interim benchmark read by compare_member_floor_v1 (v35 --nk-jvp exact against v34 from the member's initial data, twice the floor,
#     masks equal; 41.3 (2)) into ${QI2_ROOT}/interim_v35_vs_v34.md;
# (2) if every common record is within, the jvp member (41.3 (4)): the sigma/a 2 ladder member's exact arguments (engine_collapse_v32 on pod 2,
#     untouched) on engine_collapse_v35 --nk-jvp exact from T = 0 to T/r_s 40 or the member's end rule, its complex64 slices at the member's
#     clocks, ${QI2_ROOT}/jvpmember (tag c32_rs64_A_R2_tau8_ell2_prep4_avg_v35jvp); else the S11 ell 4a member (41.1 (3)) takes the GPU instead and the
#     jvp member is not launched (recorded);
# (3) the S11 ell 4a member (41.1 (3)): S11's arguments at --ell 4 on v35 --nk-jvp exact, to T/r_s 40 or its stop, complex128 slices at S11's
#     clocks, ${QI2_ROOT}/s11ell4 (tag s11_ball_rs16_ell4_tau2_v35jvp) — launched after the jvp member (41.6: beside it only once the capture
#     minimum has freed the CPU and the memory allows; the lane launches it beside by hand when that holds).
# One GPU job at a time from this queue; 4 CPU threads each; never touches pod 2.
export PYTHONPATH=${QI2_ROOT}/code OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
C=${QI2_ROOT}/code; S=${QI2_ROOT}/pod1_h100_queue_v2.status
now() { date -u '+%Y-%m-%d %H:%M:%S UTC'; }
until grep -q "massive base exit" ${QI2_ROOT}/status.txt 2>/dev/null && grep -q "rs32 cost exit" ${QI2_ROOT}/status.txt 2>/dev/null; do sleep 30; done
cd ${QI2_ROOT}/bench53_gpu2 && python $C/compare_member_floor_v1.py --run jvp/m_jvp --ref base/m_base --label-run "v35 --nk-jvp exact" --label-ref "v34" --md interim_v35_vs_v34.md --out-json interim_v35_vs_v34.json > interim_read.txt 2>&1
echo "$(now): interim read: $(tail -1 interim_read.txt)" >> $S
S11ARGS="--m 0 --g 0.25 --rs 16 --nsites 1024 --kappa-sea 10 --tau 2 --reference R2 --half-offset 0 --collar-policy flag --support-rel 1e-14 --site-fields averaged --stop-E 1e-7 --stop-fields 1e-6 --maxit 20 --mix newton --nk-maxiter 30 --nk-krylov 40 --nk-restarts 2 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3 --c-LZ 1.0 --window-max 4 --m1-simpson-steps auto --massless-ball --ball-R0-over-rs 4 --ball-x0-over-R0 0.5 --ball-sigma-over-R0 0.25 --ball-orbitals 1:6 --fields-v17 --s11-collars 8,16 --s11-stop 1 --gpu 0 --nk-jvp exact"
if python -c "import json,sys; sys.exit(0 if json.load(open('${QI2_ROOT}/interim_v35_vs_v34.json'))['all_within'] else 1)"; then
  mkdir -p ${QI2_ROOT}/jvpmember && cd ${QI2_ROOT}/jvpmember
  T=c32_rs64_A_R2_tau8_ell2_prep4_avg_v35jvp
  echo "$(now): jvp member start" >> $S
  python $C/engine_collapse_v35.py --m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 2048 --kappa-sea 10 --reference R2 --site-fields averaged --T-over-rs 40 --prep-ell 4 --tau 8 --gpu 0 --slices-c64 12,13,14,16,20,25,30,40 --ell 2 --mix newton --nk-maxiter 30 --nk-krylov 40 --nk-restarts 2 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3 --slice-dir states/$T --nk-jvp exact --out . --tag $T > $T.log 2>&1
  echo "$(now): jvp member exit $?" >> $S
else
  echo "$(now): the interim read is not within twice the floor: the jvp member is not launched (41.3 (4)); the S11 ell 4a member next" >> $S
fi
echo "$(now): the S11 ell 4a member runs on box GPU 0 (launch_s11_ell4_box_v1.sh); nothing further queued here" >> $S
