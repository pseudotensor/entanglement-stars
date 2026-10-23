#!/bin/bash
# The S4 / S9 unit tests of engine_collapse_v26 (the brief BRIEF_v26_builder_S4_S9): box CPU (GPU 2 for the vf test only when free), nice 10,
# at most 16 threads in total. Builder analyst. Run from the box's engine directory: bash code/launch_v26_S4_tests_v1.sh <step>
#   smoke   : the r_s 16 massless member (512 sites, R_0 96, (k_0, w) = (0.45, 18)) through the preparation and 8 re-solves (T/rs 1, tau 2),
#             four variants in parallel (3 threads each): the default v26, the angle isolation, the empty-interior tolerance 1e-14, and
#             Anderson mixing with the not-contracting stop off
#   smokerep: engine_collapse_v26_smoke_report_v1 over the four runs and the rs32 timing run
#   gate    : the r_s 32 dust ball (the stored C3set25 member's arguments, 2 re-solves) on engine_collapse_v25 and on engine_collapse_v26 with
#             the items off and with the items on at version 25's stop rule (4 threads each)
#   gaterep : engine_collapse_v26_gate_v1 over the three, the stored H100 record beside
#   rs32    : the relaunch member (r_s 32, 1024 sites, tau 2, R_0 = 6 r_s, (k_0, w) = (0.4, 20)) through the preparation and 2 re-solves (timing)
#   vf      : engine_ops_v4_vf_unit_v1 on vf9's channels 1 to 4 (GPU 2)
#   l12     : engine_ops_v4_l12_unit_v1
#   fgate   : engine_fields_v17_gate_v2
cd ${QI2_ROOT}/engine || exit 1
PY=${QI2_ROOT}/python
MS16="--massless-shell --m 0 --g 0.25 --rs 16 --nsites 512 --ell 4 --kappa-sea 10 --tau 2 --T-over-rs 1.0 --reference R2 --site-fields averaged --shell-R0-over-rs 6 --shell-width-over-rs 1.125 --shell-k0 0.45 --c-LZ 1"
case "$1" in
  smoke)
    mkdir -p S4/smoke_final
    export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
    nohup nice -n 10 $PY code/engine_collapse_v26.py $MS16 --out S4/smoke_final --tag v26ms_rs16_default > S4/smoke_final/log_default.log 2>&1 &
    nohup nice -n 10 $PY code/engine_collapse_v26.py $MS16 --oriented-isolation angle --out S4/smoke_final --tag v26ms_rs16_angle > S4/smoke_final/log_angle.log 2>&1 &
    nohup nice -n 10 $PY code/engine_collapse_v26.py $MS16 --empty-rel 1e-14 --out S4/smoke_final --tag v26ms_rs16_empty1e-14 > S4/smoke_final/log_empty1e-14.log 2>&1 &
    nohup nice -n 10 $PY code/engine_collapse_v26.py $MS16 --mix anderson --not-contracting-stop 0 --out S4/smoke_final --tag v26ms_rs16_anderson_nostop > S4/smoke_final/log_anderson_nostop.log 2>&1 &
    ;;
  smokerep)
    export OMP_NUM_THREADS=1
    nice -n 10 $PY code/engine_collapse_v26_smoke_report_v1.py --run default=S4/smoke_final/v26ms_rs16_default --run angle=S4/smoke_final/v26ms_rs16_angle \
      --run empty1e-14=S4/smoke_final/v26ms_rs16_empty1e-14 --run anderson_nostop=S4/smoke_final/v26ms_rs16_anderson_nostop \
      --run rs32_relaunch_member_timing=S4/rs32/v26ms_rs32_k0.4_w20_tau2_timing \
      --md S4/smoke_report_v1.md --json S4/smoke_report_v1.json > S4/log_smoke_report_v1.log 2>&1
    ;;
  gate)
    mkdir -p S4/gate
    export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
    G="--m 0.2 --g 0.25 --rs 32 --R0-over-rs 4 --reading A --nsites 1024 --ell 4 --kappa-sea 10 --tau 1 --T-over-rs 0.0625 --reference R2 --site-fields averaged --out S4/gate"
    nohup nice -n 10 $PY code/engine_collapse_v25.py $G --tag g25cpu_rs32_tau1 > S4/gate/log_g25cpu.log 2>&1 &
    nohup nice -n 10 $PY code/engine_collapse_v26.py $G --v26-items-off --tag g26off_rs32_tau1 > S4/gate/log_g26off.log 2>&1 &
    nohup nice -n 10 $PY code/engine_collapse_v26.py $G --stop-rule v25 --tag g26on_v25stop_rs32_tau1 > S4/gate/log_g26on_v25stop.log 2>&1 &
    ;;
  gaterep)
    export OMP_NUM_THREADS=1
    nice -n 10 $PY code/engine_collapse_v26_gate_v1.py --ref S4/gate/g25cpu_rs32_tau1 --run v26_items_off=S4/gate/g26off_rs32_tau1 \
      --run v26_items_on_v25_stop=S4/gate/g26on_v25stop_rs32_tau1 --stored stored_H100_v25=H100b/C3set25/c25_rs32_A_R2_tau1_ell4_avg \
      --md S4/collapse_v26_gate_v1.md --json S4/collapse_v26_gate_v1.json > S4/log_collapse_v26_gate_v1.log 2>&1
    ;;
  rs32)
    mkdir -p S4/rs32
    export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
    nohup nice -n 10 $PY code/engine_collapse_v26.py --massless-shell --m 0 --g 0.25 --rs 32 --nsites 1024 --ell 4 --kappa-sea 10 --tau 2 --T-over-rs 0.125 \
      --reference R2 --site-fields averaged --shell-R0-over-rs 6 --shell-width-over-rs 0.625 --shell-k0 0.4 --c-LZ 1 --out S4/rs32 --tag v26ms_rs32_k0.4_w20_tau2_timing \
      > S4/rs32/log_v26ms_rs32_timing.log 2>&1 &
    ;;
  vf)
    export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
    nohup nice -n 10 $PY code/engine_ops_v4_vf_unit_v1.py --kappas 1,2,3,4 --gpu 2 --md S4/ops_v4_vf_unit_v1.md --json S4/ops_v4_vf_unit_v1.json > S4/log_ops_v4_vf_unit_v1.log 2>&1 &
    ;;
  l12)
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    nohup nice -n 10 $PY code/engine_ops_v4_l12_unit_v1.py --md S4/ops_v4_l12_unit_v1.md --json S4/ops_v4_l12_unit_v1.json > S4/log_ops_v4_l12_unit_v1.log 2>&1 &
    ;;
  fgate)
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    nohup nice -n 10 $PY code/engine_fields_v17_gate_v2.py --gate-empty-rel 1e-6 --gate H100b/C3set25/c25_rs32_A_R2_tau1_ell4_avg:0,1,400,-1 --gate H100b/C3gate25/g25_rs64_tau8:0,1,2 \
      --massless V22S/ms15_rs16_k0.45_w1.125:0,1,8 --md S4/fields_v17_gate_v2.md --json S4/fields_v17_gate_v2.json > S4/log_fields_v17_gate_v2.log 2>&1 &
    ;;
  *) echo "usage: bash code/launch_v26_S4_tests_v1.sh smoke|smokerep|gate|gaterep|rs32|vf|l12|fgate"; exit 2;;
esac
